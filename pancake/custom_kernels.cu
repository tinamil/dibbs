#include <stdio.h>
#include "custom_kernels.cuh"
#include "Pancake.h"

#define MIN(x,y) ((x < y) ? x : y)
#define MAX(x,y) ((x < y) ? y : x)

//https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu
template <typename T>
__device__ void atomicMin(T* const address, const T value)
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
__global__
void cuda_bitwise_set_intersection(int rows_a, int rows_b, const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ g_vals,
                                   const uint32_t* __restrict__ hash_b, uint32_t* __restrict__ results)
{
  __shared__ uint32_t sA[TILE_WIDTH][NUM_INTS_PER_PANCAKE];
  __shared__ uint32_t sB[TILE_WIDTH][NUM_INTS_PER_PANCAKE];
  __shared__ uint32_t sG[TILE_WIDTH];

  int col = blockIdx.x * blockDim.x + threadIdx.x; //0 to rows_a
  int row = blockIdx.y * blockDim.y + threadIdx.y; //0 to rows_b

  int b_row = row;
  int a_row = col;

  #pragma unroll
  for(int tidy = threadIdx.y; tidy < NUM_INTS_PER_PANCAKE; tidy += blockDim.y) {
    if(a_row < rows_a) {
      sA[threadIdx.x][tidy] = hash_a[a_row * NUM_INTS_PER_PANCAKE + tidy];
    }
  }
  #pragma unroll
  for(int tidx = threadIdx.x; tidx < NUM_INTS_PER_PANCAKE; tidx += blockDim.x) {
    if(b_row < rows_b) {
      sB[threadIdx.y][tidx] = hash_b[b_row * NUM_INTS_PER_PANCAKE + tidx];
    }
  }
  if(a_row < rows_a && threadIdx.y == 0) sG[threadIdx.x] = g_vals[a_row];
  __syncthreads();
  if(b_row < rows_b && a_row < rows_a) {
    uint32_t tmp = 0, tmp_a = 0, tmp_b = 0;
    #pragma unroll
    for(int i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      /*if constexpr(GAPX > 0) {
        tmp_a += __popc(sA[threadIdx.x][i]);
        tmp_b += __popc(sB[threadIdx.y][i]);
      }*/
      tmp += __popc((sA[threadIdx.x][i] & sB[threadIdx.y][i]));
    }
    results[row * rows_a + col] = NUM_PANCAKES - GAPX + sG[threadIdx.x] - tmp;
  }
}

__global__
void naive_cuda_bitwise_set_intersection(int rows_a, int rows_b, const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b, uint32_t* __restrict__ results)
{
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x, max = rows_a * rows_b, stride = blockDim.x * gridDim.x; batch_idx < max; batch_idx += stride)
  {
    int col = batch_idx / rows_a;
    int row = batch_idx % rows_a;
    uint32_t tmp = 0;
    for(int i = 0; i < NUM_INTS_PER_PANCAKE; ++i) {
      tmp += __popc(hash_a[row * NUM_INTS_PER_PANCAKE + i] & hash_b[col * NUM_INTS_PER_PANCAKE + i]);
    }
    results[batch_idx] = NUM_PANCAKES - 2 * GAPX + g_vals[row] - tmp;
  }
}

void bitwise_set_intersection(cudaStream_t stream, int rows_a, int rows_b, const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ g_vals, const uint32_t* __restrict__ hash_b, uint32_t* __restrict__ mult_results)
{
  //constexpr int threadsPerBlock = 96;
  //int blocksPerGrid = (rows_a * rows_b + threadsPerBlock - 1) / threadsPerBlock;
  //naive_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 blocksPerGrid((rows_a + TILE_WIDTH - 1) / TILE_WIDTH, (rows_b + TILE_WIDTH - 1) / TILE_WIDTH, 1);
  cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
}

void vector_add(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ g_vals, uint32_t* __restrict__ mult_results)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = (num_batch * num_frontier + threadsPerBlock - 1) / threadsPerBlock;
  cuda_vector_add_matrix_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, g_vals, mult_results);
}

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ mult_results, uint32_t* __restrict__ d_batch_answers)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = (num_batch * num_frontier + threadsPerBlock - 1) / threadsPerBlock;
  reduceMin << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, mult_results, d_batch_answers);
}

