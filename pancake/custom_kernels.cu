#include <stdio.h>
#include "custom_kernels.cuh"
#include "Pancake.h"

#define MIN(x,y) ((x < y) ? x : y)

//https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu
__device__ void atomicMin(float* const address, const float value)
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

  for(int batch_idx = blockIdx.x, end = num_batch, stride = gridDim.x; batch_idx < end; batch_idx += stride){
    const T* __restrict__ start_results = mult_results + batch_idx * num_frontier;
    
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
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x; batch_idx < num_batch; batch_idx += blockDim.x * gridDim.x)
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


void vector_add(cudaStream_t stream, int num_batch, int num_frontier, const float* __restrict__ g_vals, float* __restrict__ mult_results)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = (num_batch * num_frontier + threadsPerBlock - 1) / threadsPerBlock;
  cuda_vector_add_matrix_kernel <<<blocksPerGrid, threadsPerBlock, 0, stream >>> (num_batch, num_frontier, g_vals, mult_results);
}

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const float* __restrict__ mult_results, float* __restrict__ d_batch_answers)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = (num_batch * num_frontier + threadsPerBlock - 1) / threadsPerBlock;
  reduceMin <<<blocksPerGrid, threadsPerBlock, 0, stream>>> (num_batch, num_frontier, mult_results, d_batch_answers);
}

