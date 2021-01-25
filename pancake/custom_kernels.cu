#include <stdio.h>
#include "custom_kernels.cuh"
#include "Pancake.h"
#include "hash_array.h"

#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;


#define int_div_ceil(x,y) ((x + y - 1) / y)

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



//https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu
template <typename T>
__device__ void atomicMinFloat(T* const address, const T value)
{
  if(*address <= value)
  {
    return;
  }

  int* const addressAsI = (int*)address;
  int old = *addressAsI, assumed;

  do
  {
    assumed = old;
    if(__int_as_float(assumed) <= value)
    {
      break;
    }

    old = atomicCAS(addressAsI, assumed, __float_as_int(value));
  } while(assumed != old);
}

__device__ char atomicMinChar(char* address, char val)
{
  unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int sel = selectors[(size_t)address & 3];
  unsigned int old, assumed, min_, new_;

  old = *base_address;
  do {
    assumed = old;
    min_ = min(val, (char)__byte_perm(old, 0, ((size_t)address & 3)));
    new_ = __byte_perm(old, min_, sel);
    old = atomicCAS(base_address, assumed, new_);
  } while(assumed != old);

  return old;
}

template <typename T>
__global__ void reduceMin(int num_batch, int num_frontier, const T* __restrict__ mult_results, T* __restrict__ batch_answers)
{
  __shared__ char sharedMin;

  for(int batch_idx = blockIdx.x; batch_idx < num_batch; batch_idx += gridDim.x) {
    const T* start_results = mult_results + batch_idx * num_frontier;

    __syncthreads();

    if(0 == threadIdx.x)
    {
      sharedMin = INT8_MAX;
    }

    __syncthreads();

    T localMin = INT8_MAX;

    for(int i = threadIdx.x; i < num_frontier; i += blockDim.x)
    {
      localMin = MIN(localMin, start_results[i]);
    }
    atomicMinChar(&sharedMin, localMin);

    __syncthreads();

    if(0 == threadIdx.x)
    {
      batch_answers[batch_idx] = sharedMin;
    }
  }
}

template<typename T>
__global__
void cuda_min_kernel(int num_batch, int num_frontier, const T* __restrict__ mult_results, T* __restrict__ batch_answers)
{
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x; batch_idx < num_batch; batch_idx += stride)
  {
    const T* __restrict__ start_results = mult_results + batch_idx * num_frontier;
    T min = start_results[0];
    for(int frontier_idx = 1; frontier_idx < num_frontier; ++frontier_idx)
    {
      if(start_results[frontier_idx] < min) min = start_results[frontier_idx];
    }
    batch_answers[batch_idx] = min;
  }
}

#define TILE 16
__global__
void tiled_cuda_bitwise_set_intersection(const uint32_t rows_a,//x-axis
                                         const uint32_t rows_b,//y-axis
                                         const unsigned max_a,
                                         const uint32_t* __restrict__ hash_a,
                                         const uint8_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b,
                                         uint8_t* __restrict__ results)
{
  assert(threadIdx.x < TILE);
  assert(threadIdx.y < TILE);
  assert(blockIdx.x * blockDim.x < rows_a);
  assert(blockIdx.y * blockDim.y < rows_b);

  __shared__ uint32_t sA[NUM_INTS_PER_PANCAKE][TILE];
  __shared__ uint32_t sB[TILE][NUM_INTS_PER_PANCAKE];
  uint32_t localB[NUM_INTS_PER_PANCAKE];
  uint32_t localA[NUM_INTS_PER_PANCAKE];
  volatile __shared__ uint8_t sMin[TILE][TILE];
  volatile uint8_t minVal = UINT8_MAX;

  cg::thread_block block = cg::this_thread_block();

  //for(uint32_t by = blockIdx.y; by < int_div_ceil(rows_b, NUM_INTS_PER_PANCAKE); by += gridDim.y) {
  const uint32_t output_row = blockIdx.y * blockDim.y + threadIdx.y;

  if(output_row < rows_b) {
    for(uint32_t tidx = threadIdx.x; tidx < NUM_INTS_PER_PANCAKE; tidx += blockDim.x) {
      sB[threadIdx.y][tidx] = hash_b[output_row * NUM_INTS_PER_PANCAKE + tidx];
    }
  }

  block.sync();

  if(output_row < rows_b) {
    #pragma unroll
    for(uint32_t i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      localB[i] = sB[threadIdx.y][i];
    }
  }

  //x goes 0 to rows_a
  for(uint32_t bx = blockIdx.x; bx < max_a; bx += gridDim.x) {
    uint32_t output_col = bx * blockDim.x + threadIdx.x;
    if(output_col < rows_a) {
      for(int tidy = threadIdx.y; tidy < NUM_INTS_PER_PANCAKE; tidy += blockDim.y) {
        sA[tidy][threadIdx.x] = hash_a[output_col * NUM_INTS_PER_PANCAKE + tidy];
      }
    }
    block.sync();
    if(output_row < rows_b && output_col < rows_a) {
      #pragma unroll
      for(uint32_t i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
        localA[i] = sA[i][threadIdx.x];
      }
      constexpr Mask gap_mask;
      uint32_t tmpF = 0;
      uint32_t tmpB = 0;
      uint32_t tmpMin;
      #pragma unroll
      for(uint32_t i = 0; i < NUM_GAP_INTS; ++i) {
        uint32_t A = localA[i];
        uint32_t B = localB[i];
        tmpF += __popc(B & (A | gap_mask[i]));
        tmpB += __popc(A & (B | gap_mask[i]));
      }
      tmpMin = MIN(tmpF, tmpB);
      #pragma unroll
      for(uint32_t i = NUM_GAP_INTS; i < NUM_INTS_PER_PANCAKE; ++i) {
        uint32_t A = localA[i];
        uint32_t B = localB[i];
        tmpMin += __popc(A & B);
      }
      results[output_row * rows_a + output_col] = static_cast<uint8_t>(NUM_PANCAKES + g_vals[output_col] - tmpMin);
      //minVal = MIN(minVal, NUM_PANCAKES + g_vals[col] - tmpMin);
      //assert(minVal > GAPX);
    }
    block.sync();
  }
  
  //assert(minVal > GAPX);
  //sMin[threadIdx.y][threadIdx.x] = minVal;
  //block.sync();
  //if(threadIdx.x == 0 && output_row < rows_b) {
  //  for(uint32_t a = 1; a < TILE && blockIdx.x * blockDim.x + a < rows_a; a++) {
  //    minVal = MIN(minVal, sMin[threadIdx.y][a]);
  //    assert(minVal > GAPX);
  //  }
  //  results[output_row * gridDim.x + blockIdx.x] = minVal;
  //}
//}
}

__global__
void naive_cuda_bitwise_set_intersection(int rows_a, int rows_b, const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b, uint32_t* __restrict__ results)
{
  constexpr Mask gap_mask;
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x, max = rows_a * rows_b, stride = blockDim.x * gridDim.x; batch_idx < max; batch_idx += stride)
  {
    int col = batch_idx / rows_a;
    int row = batch_idx % rows_a;
    int tmpF = 0;
    int tmpB = 0;
    for(int i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      uint32_t A = hash_a[row * NUM_INTS_PER_PANCAKE + i];
      uint32_t B = hash_b[col * NUM_INTS_PER_PANCAKE + i];
      tmpF += __popc(B & (A | gap_mask[i]));
      tmpB += __popc(A & (B | gap_mask[i]));
    }
    results[batch_idx] = NUM_PANCAKES + g_vals[row] - MIN(tmpF, tmpB);
  }
}

__global__ void sharedReduceMin(const uint32_t xDim, 
                                const uint32_t yDim, 
                                const uint8_t* __restrict__ mult_results, 
                                uint8_t* __restrict__ batch_answers)
{
  cg::thread_block block = cg::this_thread_block();
  volatile uint8_t minVal = UINT8_MAX;
  volatile __shared__ uint8_t sharedMin[TILE][TILE];
  assert(threadIdx.y < TILE);
  const uint32_t output_row = blockIdx.y * blockDim.y + threadIdx.y;
  if(output_row < yDim) {
    const uint8_t* __restrict__ start_row = mult_results + output_row * xDim;
    for(unsigned input_column = blockIdx.x * blockDim.x + threadIdx.x; input_column < xDim; input_column += gridDim.x) {
      minVal = MIN(start_row[input_column], minVal);
      assert(minVal > GAPX);
    }
  }
  sharedMin[threadIdx.y][threadIdx.x] = minVal;
  block.sync();
  if(threadIdx.x == 0 && output_row < yDim) {
    #pragma unroll
    for(int i = 1; i < TILE && blockIdx.x * blockDim.x + i < xDim; ++i) {
      minVal = MIN(minVal, sharedMin[threadIdx.y][i]);
      assert(minVal > GAPX);
    }
    batch_answers[output_row] = minVal;
  }
}

void bitwise_set_intersection(cudaStream_t stream,
                              int rows_a,
                              int rows_b,
                              const uint32_t* __restrict__ hash_a,
                              const uint8_t* __restrict__ g_vals,
                              const uint32_t* __restrict__ hash_b,
                              uint8_t* __restrict__ mult_results,
                              uint8_t* __restrict__ d_answers)
{
  
  //constexpr int threadsPerBlock = 256;
  //int blocksPerGrid = (rows_a * rows_b + threadsPerBlock - 1) / threadsPerBlock;
  //naive_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  constexpr uint32_t MAX_BLOCKS_X = 1024;
  constexpr uint32_t MAX_BLOCKS_Y = 65535;
  constexpr uint32_t THREADS_X = TILE;
  constexpr uint32_t THREADS_Y = TILE;
  dim3 threadsPerBlock(MIN(rows_a, THREADS_X), MIN(rows_b, THREADS_Y), 1);
  uint32_t gridDimX = MIN(MAX_BLOCKS_X, int_div_ceil(rows_a, threadsPerBlock.x));
  uint32_t gridDimY = MIN(MAX_BLOCKS_Y, int_div_ceil(rows_b, threadsPerBlock.y));
  int max_a = int_div_ceil(rows_a, threadsPerBlock.x);
  assert(gridDimY < 65535);
  dim3 blocksPerGrid(gridDimX, gridDimY, 1);
  
  //TODO: DO THIS WHERE IT SHOULD BE DONE
  //cudaMalloc(mult_results, rows_b * gridDimY * sizeof(uint8_t));

  tiled_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, max_a, hash_a, g_vals, hash_b, mult_results);

  threadsPerBlock = dim3(TILE, TILE, 1);
  blocksPerGrid = dim3(int_div_ceil(gridDimX, threadsPerBlock.x), gridDimY, 1);
  //sharedReduceMin <<<blocksPerGrid, threadsPerBlock, 0, stream >>> (gridDimX, rows_b, mult_results, d_answers);
  reduceMin << <96, 16384, 0, stream >> > (rows_b, rows_a, mult_results, d_answers);
}

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint8_t* __restrict__ mult_results, uint8_t* __restrict__ d_batch_answers)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 16384);
  reduceMin << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, mult_results, d_batch_answers);
}

