#include <stdio.h>
#include "custom_kernels.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Pancake.h"

__global__
void extract_min_kernel(int n, float* x, float* y)
{
  float min = x[0];
  for(int i = 1; i < n; ++i)
  {
    if(x[i] < min) min = x[i];
  }
  *y = min;
}

void extract_min(int n, float* x, float* y)
{
  extract_min_kernel << <1, 1 >> > (n, x, y);
}

#define min(x,y) ((x < y) ? x : y)


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
  __device__ inline operator T* ()
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T* () const
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
  __device__ inline operator double* ()
  {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }

  __device__ inline operator const double* () const
  {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }
};

/*
 This version adds multiple elements per thread sequentially.  This reduces the overall
 cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 (Brent's Theorem optimization)

 Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 If blockSize > 32, allocate blockSize*sizeof(T) bytes.
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(const T* __restrict__ g_idata, T* __restrict__ g_odata, unsigned int n)
{
  T* sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;


  T myMin = 99999;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if(nIsPow2)
  {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while(i < n)
    {
      myMin = min(g_idata[i], myMin);
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if((i + blockSize) < n)
      {
        myMin = min(g_idata[i + blockSize], myMin);
      }
      i += gridSize;
    }
  }
  else
  {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while(i < n)
    {
      myMin = min(g_idata[i], myMin);
      i += gridSize;
    }
  }

  i += gridSize;

// each thread puts its local sum into shared memory
  sdata[tid] = myMin;
  __syncthreads();

  // do reduction in shared mem
  if((blockSize >= 512) && (tid < 256))
  {
    sdata[tid] = myMin = min(sdata[tid + 256], myMin);
  }

  __syncthreads();

  if((blockSize >= 256) && (tid < 128))
  {
    sdata[tid] = myMin = min(sdata[tid + 128], myMin);
  }

  __syncthreads();

  if((blockSize >= 128) && (tid < 64))
  {
    sdata[tid] = myMin = min(sdata[tid + 64], myMin);
  }

  __syncthreads();

      // fully unroll reduction within a single warp
  if((blockSize >= 64) && (tid < 32))
  {
    sdata[tid] = myMin = min(sdata[tid + 32], myMin);
  }

  __syncthreads();

  if((blockSize >= 32) && (tid < 16))
  {
    sdata[tid] = myMin = min(sdata[tid + 16], myMin);
  }

  __syncthreads();

  if((blockSize >= 16) && (tid < 8))
  {
    sdata[tid] = myMin = min(sdata[tid + 8], myMin);
  }

  __syncthreads();

  if((blockSize >= 8) && (tid < 4))
  {
    sdata[tid] = myMin = min(sdata[tid + 4], myMin);
  }

  __syncthreads();

  if((blockSize >= 4) && (tid < 2))
  {
    sdata[tid] = myMin = min(sdata[tid + 2], myMin);
  }

  __syncthreads();

  if((blockSize >= 2) && (tid < 1))
  {
    sdata[tid] = myMin = min(sdata[tid + 1], myMin);
  }

  __syncthreads();
  // write result for this block to global mem
  if(tid == 0)
  {
    g_odata[blockIdx.x] = myMin;
  }
}

extern "C"
bool isPow2(unsigned int x);

template <class T>
void
min_reduce(int size, int threads, int blocks, T* d_idata, T* d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  if(isPow2(size))
  {
    switch(threads)
    {
      case 512:
        reduce6<T, 512, true> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 256:
        reduce6<T, 256, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 128:
        reduce6<T, 128, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 64:
        reduce6<T, 64, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 32:
        reduce6<T, 32, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 16:
        reduce6<T, 16, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  8:
        reduce6<T, 8, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  4:
        reduce6<T, 4, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  2:
        reduce6<T, 2, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  1:
        reduce6<T, 1, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;
    }
  }
  else
  {
    switch(threads)
    {
      case 512:
        reduce6<T, 512, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 256:
        reduce6<T, 256, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 128:
        reduce6<T, 128, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 64:
        reduce6<T, 64, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 32:
        reduce6<T, 32, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case 16:
        reduce6<T, 16, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  8:
        reduce6<T, 8, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  4:
        reduce6<T, 4, false> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  2:
        reduce6<T, 2, false> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;

      case  1:
        reduce6<T, 1, false> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
        break;
    }
  }
}

// Instantiate the reduction function for 3 types
template void
min_reduce<int>(int size, int threads, int blocks, int* d_idata, int* d_odata);

template void
min_reduce<float>(int size, int threads, int blocks, float* d_idata, float* d_odata);

template void
min_reduce<double>(int size, int threads, int blocks, double* d_idata, double* d_odata);

//hash_matrix is a 2d array, num_batch rows and num_hash cols
//frontier is a 2d array, num_hash rows and num_frontier cols
//g_vals is a vector of size num_frontier
//mult_results is a 2d array of num_batch rows and num_hash cols
__global__
void cuda_heuristic_kernel(int num_batch, int num_hash, int num_frontier, float* hash_matrix, float* frontier, float* g_vals, float* mult_results, float* batch_answers)
{
  for(int batch_idx = blockIdx.x * blockDim.x + threadIdx.x; batch_idx < num_batch; batch_idx += blockDim.x * gridDim.x)
  {
    for(int frontier_idx = 0; frontier_idx < num_frontier; frontier_idx += 1)
    {
      float sum = 0;
      for(int hash_idx = 0; hash_idx < num_hash; hash_idx += 1)
      {
        sum += frontier[frontier_idx * num_hash + hash_idx] * hash_matrix[batch_idx * num_hash + hash_idx];
      }
      mult_results[batch_idx * num_frontier + frontier_idx] = NUM_PANCAKES + g_vals[frontier_idx] - sum;
    }
    float* start_results = mult_results + batch_idx * num_frontier;
    float min = start_results[0];
    for(int i = 1; i < num_frontier; ++i)
    {
      if(start_results[i] < min) min = start_results[i];
    }
    batch_answers[batch_idx] = min;
  }
}

void cuda_heuristic(int num_batch, int num_hash, int num_frontier, float* hash_matrix, float* frontier, float* g_vals, float* mult_results, float* d_batch_answers)
{
  cuda_heuristic_kernel <<<1024, 64>>>(num_batch, num_hash, num_frontier, hash_matrix, frontier, g_vals, mult_results, d_batch_answers);
}

