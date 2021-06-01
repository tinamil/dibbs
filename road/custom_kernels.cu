#include "Constants.h"
#include "custom_kernels.cuh"
//#include "node.h"
//#include "hash_array.h"
#include "cuda_helper.h"
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

__device__
uint32_t calculateDistanceInMeter(const Coordinate& coord1, const Coordinate& coord2)
{
  constexpr float DIAMETER_PI_F = AVERAGE_DIAMETER_OF_EARTH_M * PI * RAD_TO_DEG_I360;
  constexpr float EARTH_PERIMETER_F = EARTH_PERIMETER * RAD_TO_DEG_I360;

  const float earthCyclePerimeter = EARTH_PERIMETER_F * __cosf(__double2float_rd(coord1.lat + coord2.lat) / 2.0f);
  const float dx = __double2float_rd(coord1.lng - coord2.lng) * earthCyclePerimeter;
  const float dy = DIAMETER_PI_F * __double2float_rd(coord1.lat - coord2.lat);

  return __float2uint_rd(.99f * __fsqrt_rz(dx * dx + dy * dy));

  //const auto latDistance = __sinf(__double2float_rd((coord1.lat - coord2.lat) / 2));
  //const auto lngDistance = __sinf(__double2float_rd((coord1.lng - coord2.lng) / 2));
  //const auto a = latDistance * latDistance + __cosf(__double2float_rd(coord1.lat)) * __cosf(__double2float_rd(coord2.lat)) * lngDistance * lngDistance;
  //const auto c = AVERAGE_DIAMETER_OF_EARTH_M * asinf(fminf(1.0f, __fsqrt_rd(a)));
  //return __double2uint_rd(c);
}

constexpr uint32_t npow2(uint32_t v)
{
  //return v == 1 ? 1 : 1 << (64 - __lzcnt(v - 1));
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
static_assert(npow2(3) == 4);

#define REDUCE_THREADS 128
template <typename T>
__global__ void myReduceMinSharedMem(const uint32_t xDim, const uint32_t xStride, const T* __restrict__ mult_results, T* __restrict__ batch_answers)
{
  cg::thread_block block = cg::this_thread_block();
  __shared__ T sharedMin[REDUCE_THREADS];
  constexpr T max_val = std::numeric_limits<T>::max();
  //for(int batch_idx = blockIdx.lng; batch_idx < num_batch; batch_idx += gridDim.lng) {
  const T* start_results = mult_results + blockIdx.x * xStride;
  T localMin = max_val;
  for(unsigned int i = threadIdx.x; i < xDim; i += blockDim.x)
  {
    const T tmpMinVal = start_results[i];
    localMin = MIN(localMin, tmpMinVal);
  }
  sharedMin[threadIdx.x] = localMin;
  block.sync();
  for(unsigned int s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
    if(threadIdx.x < s) {
      const T tmpMinVal = sharedMin[threadIdx.x + s];
      if(localMin > tmpMinVal) {
        localMin = tmpMinVal;
        sharedMin[threadIdx.x] = localMin;
      }
    }
    block.sync();
  }
  if(0 == threadIdx.x)
  {
    batch_answers[blockIdx.x] = localMin;
  }
//}
}


template<typename T>
__global__
void cuda_min_kernel(int cols_a, int rows_b, const T* __restrict__ mult_results, T* __restrict__ batch_answers)
{
  for(int row = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x; row < rows_b; row += stride)
  {
    const T* __restrict__ start_results = mult_results + row * cols_a;
    T min = start_results[0];
    for(int col = 1; col < cols_a; ++col)
    {
      if(start_results[col] < min) min = start_results[col];
    }
    batch_answers[row] = min;
  }
}

//__global__ void transpose_device(const int rowDim, const int colDim, const uint32_t* __restrict__ input, uint32_t* output)
//{
//  for(int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < rowDim * colDim; tid += blockDim.x * gridDim.x) {
//    const int orow = tid % colDim;
//    const int ocol = tid / colDim;
//    output[orow * rowDim + ocol] = input[tid];
//  }
//}

#define TILE_X 16
#define TILE_Y 16
__global__
void tiled_cuda_bitwise_set_intersection(const uint32_t cols_a,
                                         const uint32_t rows_b,
                                         const uint32_t max_a,
                                         const uint32_t* __restrict__ hash_a,
                                         const uint32_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b,
                                         const Coordinate* __restrict__ coordinates,
                                         uint32_t* __restrict__ results)
{
  assert(blockIdx.x * blockDim.x < cols_a);
  assert(blockIdx.y * blockDim.y < rows_b);

  __shared__ uint32_t sMin[TILE_Y][TILE_X];
  uint32_t localMin = UINT32_MAX;

  __shared__ Coordinate sA[TILE_X];
  __shared__ Coordinate sB[TILE_Y];
  Coordinate localB;

  cg::thread_block block = cg::this_thread_block();

  //for(uint32_t by = blockIdx.y; by < int_div_ceil(rows_b, NUM_INTS_PER_PANCAKE); by += gridDim.y) {

  const uint32_t output_row = blockIdx.y * blockDim.y + threadIdx.y;

  if(output_row < rows_b) {
    if(threadIdx.x == 0) {
      memcpy(sB + threadIdx.y, coordinates + hash_b[output_row], sizeof(Coordinate));
    }
  }

  block.sync();

  if(output_row < rows_b) {
    localB = sB[threadIdx.y];
  }

  //bx goes 0 to rows_a
  for(uint32_t bx = blockIdx.x; bx < max_a; bx += gridDim.x) {
    uint32_t output_col = bx * blockDim.x + threadIdx.x;
    if(output_col < cols_a && threadIdx.y == 0) {
      memcpy(sA + threadIdx.x, coordinates + hash_a[output_col], sizeof(Coordinate));
    }
    block.sync();
    if(output_row < rows_b && output_col < cols_a) {
      localMin = umin(localMin, g_vals[output_col] + calculateDistanceInMeter(sA[threadIdx.x], localB));
    }
    block.sync();
  }

  sMin[threadIdx.y][threadIdx.x] = localMin;
  block.sync();

  for(unsigned int s = TILE_X / 2; s > 0; s >>= 1) {
    if(threadIdx.x < s) {
      const uint32_t tmpMinVal = sMin[threadIdx.y][threadIdx.x + s];
      if(tmpMinVal < localMin) {
        localMin = tmpMinVal;
        sMin[threadIdx.y][threadIdx.x] = tmpMinVal;
      }
    }
    block.sync();
  }

  if(0 == threadIdx.x && output_row < rows_b)
  {
    results[output_row * gridDim.x + blockIdx.x] = localMin;
  }
//}
}

__global__
void naive_cuda_heuristic(int cols_a, int rows_b, const uint32_t* __restrict__ a_nodes, const uint32_t* __restrict__ g_vals,
                          const uint32_t* __restrict__ b_nodes, const Coordinate* __restrict__ coordinates, uint32_t* __restrict__ results)
{
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x, max = cols_a * rows_b, stride = blockDim.x * gridDim.x; batch_idx < max; batch_idx += stride)
  {
    int col = batch_idx % cols_a;
    int row = batch_idx / cols_a;
    results[batch_idx] = g_vals[col] + calculateDistanceInMeter(coordinates[a_nodes[col]], coordinates[b_nodes[row]]);
  }
}

void bitwise_set_intersection(cudaStream_t stream,
                              int rows_a,
                              int rows_b,
                              const uint32_t* __restrict__ hash_a,
                              const uint32_t* __restrict__ g_vals,
                              const uint32_t* __restrict__ hash_b,
                              uint32_t* __restrict__ mult_results,
                              uint32_t* __restrict__ d_answers,
                              const Coordinate* __restrict__ d_coordinates)
{
  //constexpr int threadsPerBlockInt = 256;
  //int blocksPerGridInt = (rows_a * rows_b + threadsPerBlockInt - 1) / threadsPerBlockInt;
  //naive_cuda_heuristic << <blocksPerGridInt, threadsPerBlockInt, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, d_coordinates, mult_results);
  constexpr uint32_t MAX_BLOCKS_X = 1024;
  dim3 threadsPerBlock(TILE_X, TILE_Y, 1);
  int max_a = int_div_ceil(rows_a, threadsPerBlock.x);
  uint32_t gridDimX = MIN(MAX_BLOCKS_X, max_a);
  uint32_t gridDimY = int_div_ceil(rows_b, threadsPerBlock.y);
  assert(gridDimY <= 65535);
  dim3 blocksPerGrid(gridDimX, gridDimY, 1);

  tiled_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, max_a, hash_a, g_vals, hash_b, d_coordinates, mult_results);
  CUDA_CHECK_RESULT(cudaGetLastError());
  //threadsPerBlock = dim3(TILE, TILE, 1);
  //reduceMin2 << <blocksPerGrid, threadsPerBlock, TILE* gridDimX * sizeof(uint8_t), stream >> > (rows_a, int_div_ceil(rows_a, gridDimX) rows_b, mult_results, d_answers);
  //myReduceMinAtomic << <1024, 96, 0, stream >> > (rows_b, rows_a, mult_results, d_answers);
  myReduceMinSharedMem << <rows_b, REDUCE_THREADS, 0, stream >> > (gridDimX, gridDimX, mult_results, d_answers);
  //cuda_min_kernel << <blocksPerGridInt, threadsPerBlockInt, 0, stream >> > (rows_a, rows_b, mult_results, d_answers);
  CUDA_CHECK_RESULT(cudaGetLastError());
}
__global__ void EmptyKernel() {}

void empty(cudaStream_t stream, int rows_a, int rows_b)
{
  constexpr uint32_t MAX_BLOCKS_X = 1024;
  dim3 threadsPerBlock(TILE_X, TILE_Y, 1);
  int max_a = int_div_ceil(rows_a, threadsPerBlock.x);
  uint32_t gridDimX = MIN(MAX_BLOCKS_X, max_a);
  uint32_t gridDimY = int_div_ceil(rows_b, threadsPerBlock.y);
  assert(gridDimY <= 65535);
  dim3 blocksPerGrid(gridDimX, gridDimY, 1);

  EmptyKernel <<<blocksPerGrid, threadsPerBlock, 0, stream >>> ();
  CUDA_CHECK_RESULT(cudaGetLastError());
  EmptyKernel <<<rows_b, REDUCE_THREADS, 0, stream >>> ();
  CUDA_CHECK_RESULT(cudaGetLastError());
}

//void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ mult_results, uint32_t* __restrict__ d_batch_answers)
//{
//  constexpr int threadsPerBlock = 96;
//  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 16384);
//  myReduceMinAtomic << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, mult_results, d_batch_answers);
//}

//void transpose_cuda(cudaStream_t stream,
//                    const int rows,
//                    const int cols,
//                    const uint32_t* __restrict__ input,
//                    uint32_t* __restrict output)
//{
//  constexpr int threadsPerBlock = 96;
//  int blocksPerGrid = MIN(int_div_ceil(rows * cols, threadsPerBlock), 16384);
//  transpose_device << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows, cols, input, output);
//  CUDA_CHECK_RESULT(cudaGetLastError());
//}

