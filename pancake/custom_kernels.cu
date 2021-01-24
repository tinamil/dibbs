#include <stdio.h>
#include "custom_kernels.cuh"
#include "Pancake.h"
#include "hash_array.h"

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

#define int_div_ceil(x,y) ((x + y - 1) / y)


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

  for(int batch_idx = blockIdx.x, end = num_batch, stride = gridDim.x; batch_idx < end; batch_idx += stride) {
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

#define TILE_WIDTH_X 16
#define TILE_WIDTH_Y 16
__global__
void tiled_cuda_bitwise_set_intersection(int rows_a,
                                         int rows_b,
                                         const uint32_t* __restrict__ hash_a,
                                         const uint8_t* __restrict__ g_vals,
                                         const uint32_t* __restrict__ hash_b,
                                         uint8_t* __restrict__ results)
{
  int max_a = int_div_ceil(rows_a, TILE_WIDTH_X);
  int max_b = int_div_ceil(rows_b, TILE_WIDTH_Y);
  __shared__ uint32_t sA[NUM_INTS_PER_PANCAKE][TILE_WIDTH_X];
  __shared__ uint32_t sB[TILE_WIDTH_Y][NUM_INTS_PER_PANCAKE];
  cg::thread_block block = cg::this_thread_block();

  //y goes 0 to rows_b
  for(uint32_t by = blockIdx.y; by < max_b; by += gridDim.y) {
    uint32_t row = by * blockDim.y + threadIdx.y;

    for(int tidx = threadIdx.x; tidx < NUM_INTS_PER_PANCAKE; tidx += blockDim.x) {
      if(row < rows_b) {
        //cg::memcpy_async(block, sB[threadIdx.y], hash_b + row * NUM_INTS_PER_PANCAKE, sizeof(uint32_t) * NUM_INTS_PER_PANCAKE);
        sB[threadIdx.y][tidx] = hash_b[row * NUM_INTS_PER_PANCAKE + tidx];
      }
    }
    //x goes 0 to rows_a
    for(uint32_t bx = blockIdx.x; bx < max_a; bx += gridDim.x) {
      uint32_t col = bx * blockDim.x + threadIdx.x;
      for(int tidy = threadIdx.y; tidy < NUM_INTS_PER_PANCAKE; tidy += blockDim.y) {
        if(col < rows_a) {
          //cg::memcpy_async(block, sA[tidy] + threadIdx.x, hash_a + col * NUM_INTS_PER_PANCAKE + tidy, sizeof(uint32_t));
          sA[tidy][threadIdx.x] = hash_a[col * NUM_INTS_PER_PANCAKE + tidy];
        }
      }

      block.sync();
      if(row < rows_b && col < rows_a) {
        constexpr Mask gap_mask;
        uint32_t tmpF = 0;
        uint32_t tmpB = 0;
        uint32_t tmpMin;
        #pragma unroll
        for(int i = 0; i < NUM_GAP_INTS; ++i) {
          uint32_t A = sA[i][threadIdx.x];
          uint32_t B = sB[threadIdx.y][i];
          tmpF += __popc(B & (A | gap_mask[i]));
          tmpB += __popc(A & (B | gap_mask[i]));
        }
        tmpMin = MIN(tmpF, tmpB);
        #pragma unroll
        for(int i = NUM_GAP_INTS; i < NUM_INTS_PER_PANCAKE; ++i) {
          uint32_t A = sA[i][threadIdx.x];
          uint32_t B = sB[threadIdx.y][i];
          tmpMin += __popc(A & B);
        }
        results[row * rows_a + col] = static_cast<uint8_t>(NUM_PANCAKES + g_vals[col] - tmpMin);
      }
      block.sync();
    }
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

//bool __CUDA_INIT = false;

void bitwise_set_intersection(cudaStream_t stream,
                              int rows_a,
                              int rows_b,
                              const uint32_t* __restrict__ hash_a,
                              const uint8_t* __restrict__ g_vals,
                              const uint32_t* __restrict__ hash_b,
                              uint8_t* __restrict__ mult_results)
{
  //constexpr int threadsPerBlock = 256;
  //int blocksPerGrid = (rows_a * rows_b + threadsPerBlock - 1) / threadsPerBlock;
  //naive_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
  //if(!__CUDA_INIT) {
  //  __CUDA_INIT = true;
    //cudaFuncSetSharedMemConfig(tiled_cuda_bitwise_set_intersection, cudaSharedMemBankSizeFourByte);
    //cudaFuncSetAttribute(naive_cuda_bitwise_set_intersection, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
  //}
  constexpr uint32_t MAX_BLOCKS_X = 256u;
  constexpr uint32_t MAX_BLOCKS_Y = 256u;
  dim3 threadsPerBlock(TILE_WIDTH_X, TILE_WIDTH_Y, 1);
  uint32_t gridDimX = MIN(MAX_BLOCKS_X, int_div_ceil(rows_a, TILE_WIDTH_X));
  uint32_t gridDimY = MIN(MAX_BLOCKS_Y, int_div_ceil(rows_b, TILE_WIDTH_Y));
  dim3 blocksPerGrid(gridDimX, gridDimY, 1);
  tiled_cuda_bitwise_set_intersection << <blocksPerGrid, threadsPerBlock, 0, stream >> > (rows_a, rows_b, hash_a, g_vals, hash_b, mult_results);
}

void vector_add(cudaStream_t stream, int num_batch, int num_frontier, const uint32_t* __restrict__ g_vals, uint32_t* __restrict__ mult_results)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 65535);
  cuda_vector_add_matrix_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, g_vals, mult_results);
}

void reduce_min(cudaStream_t stream, int num_batch, int num_frontier, const uint8_t* __restrict__ mult_results, uint8_t* __restrict__ d_batch_answers)
{
  constexpr int threadsPerBlock = 96;
  int blocksPerGrid = MIN(int_div_ceil(num_batch * num_frontier, threadsPerBlock), 65535);
  reduceMin << <blocksPerGrid, threadsPerBlock, 0, stream >> > (num_batch, num_frontier, mult_results, d_batch_answers);
}

