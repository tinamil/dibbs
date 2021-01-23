#include <stdio.h>
#include "custom_kernels.cuh"
#include "Pancake.h"
#include "hash_array.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename T>
inline T int_div_ceil(T x, T y)
{
  return (x + y - 1) / y;
}


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

template <typename T>
__global__ void reduceMin(int num_batch, int num_frontier, const T* __restrict__ mult_results, T* __restrict__ batch_answers)
{
  __shared__ T sharedMin;

  for(int batch_idx = blockIdx.x, end = num_batch, stride = gridDim.x; batch_idx < end; batch_idx += stride) {
    const T* start_results = mult_results + batch_idx * num_frontier;

    __syncthreads();

    if(0 == threadIdx.x)
    {
      sharedMin = 999999;
    }

    __syncthreads();

    T localMin = 999999;

    for(int i = threadIdx.x; i < num_frontier; i += blockDim.x)
    {
      localMin = MIN(localMin, start_results[i]);
    }

    atomicMin(&sharedMin, localMin);

    __syncthreads();

    if(0 == threadIdx.x)
    {
      batch_answers[batch_idx] = sharedMin;
    }
  }
}

template<typename T>
__global__
void cuda_vector_add_matrix_kernel(int num_batch, int num_frontier, const T* __restrict__ g_vals, T* __restrict__ mult_results)
{
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x, end = num_batch * num_frontier, stride = blockDim.x * gridDim.x; idx < end; idx += stride)
  {
    mult_results[idx] = NUM_PANCAKES + g_vals[idx % num_frontier] - mult_results[idx];
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

#define TILE_WIDTH 16
template <size_t count_x>
__global__
void tiled_cuda_bitwise_set_intersection(int rows_a,
                                         int rows_b,
                                         const uint32_t* __restrict__ hash_a,
                                         const uint32_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b,
                                         uint32_t* __restrict__ results)
{
  __shared__ uint32_t sA[TILE_WIDTH][NUM_INTS_PER_PANCAKE];
  __shared__ uint32_t sB[TILE_WIDTH][NUM_INTS_PER_PANCAKE];
  //__shared__ uint32_t sG[TILE_WIDTH];
  
  cg::thread_block block = cg::this_thread_block();

  int row = blockIdx.y * blockDim.y + threadIdx.y; //0 to rows_b
  int b_row = row;

  #pragma unroll
  for(int tidx = threadIdx.x; tidx < NUM_INTS_PER_PANCAKE; tidx += blockDim.x) {
    if(b_row < rows_b) {
      sB[threadIdx.y][tidx] = hash_b[b_row * NUM_INTS_PER_PANCAKE + tidx];
    }
  }

  #pragma unroll
  for(uint32_t gridx = 0; gridx < count_x; ++gridx) {
    int col = gridx * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x; //0 to rows_a
    int a_row = col;

    #pragma unroll
    for(int tidy = threadIdx.y; tidy < NUM_INTS_PER_PANCAKE; tidy += blockDim.y) {
      if(a_row < rows_a) {
        sA[threadIdx.x][tidy] = hash_a[a_row * NUM_INTS_PER_PANCAKE + tidy];
      }
    }

    //if(threadIdx.y == 0 && a_row < rows_a) {
    //  sG[threadIdx.x] = g_vals[a_row];
    //}
    block.sync();

    if(b_row < rows_b && a_row < rows_a) {
      constexpr Mask gap_mask;
      uint32_t tmpF = 0;
      uint32_t tmpB = 0;
      uint32_t tmpMin;
      #pragma unroll
      for(int i = 0; i < NUM_GAP_INTS; ++i) {
        uint32_t A = sA[threadIdx.x][i];
        uint32_t B = sB[threadIdx.y][i];
        tmpF += __popc(B & (A | gap_mask[i]));
        tmpB += __popc(A & (B | gap_mask[i]));
      }
      assert(tmpF >= GAPX);
      assert(tmpB >= GAPX);
      tmpMin = MIN(tmpF, tmpB);
      #pragma unroll
      for(int i = NUM_GAP_INTS; i < NUM_INTS_PER_PANCAKE; ++i) {
        uint32_t A = sA[threadIdx.x][i];
        uint32_t B = sB[threadIdx.y][i];
        tmpMin += __popc(A & B);
      }
      assert(tmpMin <= NUM_PANCAKES);
      assert(tmpMin >= GAPX);
      results[row * rows_a + col] = NUM_PANCAKES + g_vals[a_row] - tmpMin;
    }
    block.sync();
  }
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

void bitwise_set_intersection(cudaStream_t stream,
                              int rows_a,
                              int rows_b,
                              const uint32_t* __restrict__ hash_a,
                              const uint32_t* __restrict__ g_vals,
                              const uint32_t* __restrict__ hash_b,
                              uint32_t* __restrict__ mult_results)
{
  //constexpr int threadsPerBlock = 256;
  //int blocksPerGrid = (rows_a * rows_b + threadsPerBlock - 1) / threadsPerBlock;
  //naive_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);

  cudaFuncSetAttribute(naive_cuda_bitwise_set_intersection, cudaFuncAttributeMaxDynamicSharedMemorySize, cudaSharedmemCarveoutMaxShared);
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
  uint32_t gridDimX = int_div_ceil(rows_a, TILE_WIDTH);
  uint32_t gridDimY = int_div_ceil(rows_b, TILE_WIDTH);
  uint32_t count_x = int_div_ceil(gridDimX, 65535u);
  dim3 blocksPerGrid(MIN(65535u, gridDimX), gridDimY, 1u);
  if(count_x == 1)
    tiled_cuda_bitwise_set_intersection<1> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 2)
    tiled_cuda_bitwise_set_intersection<2> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 3)
    tiled_cuda_bitwise_set_intersection<3> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 4)
    tiled_cuda_bitwise_set_intersection<4> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 5)
    tiled_cuda_bitwise_set_intersection<5> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 6)
    tiled_cuda_bitwise_set_intersection<6> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 7)
    tiled_cuda_bitwise_set_intersection<7> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 8)
    tiled_cuda_bitwise_set_intersection<8> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 9)
    tiled_cuda_bitwise_set_intersection<9> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else if(count_x == 10)
    tiled_cuda_bitwise_set_intersection<10> << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  else
    std::cout << "ERROR: " << count_x << " count-x values\n";
}

void vector_add(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ g_vals, uint32_t* __restrict__ mult_results)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 65535);
  cuda_vector_add_matrix_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, g_vals, mult_results);
}

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ mult_results, uint32_t* __restrict__ d_batch_answers)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 65535);
  reduceMin <<<blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, mult_results, d_batch_answers);
}

