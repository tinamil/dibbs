#include "mycuda.h"
#include <vector>
#include <thrust/fill.h>
#include "Pancake.h"

mycuda::mycuda() : d_a(nullptr), d_hash_vals(nullptr), d_mult_results(nullptr), d_g_vals(nullptr), num_pancakes_constant(nullptr)
{
  CUBLAS_CHECK_RESULT(cublasCreate(&handle));
  CUBLAS_CHECK_RESULT(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK_RESULT(cublasSetStream(handle, cudaStreamPerThread));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_idx, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&one, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&neg_one, sizeof(float)));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&zero, sizeof(float)));
  static constexpr float a = 1, b = -1, c = 0;
  CUDA_CHECK_RESULT(cudaMemcpy(one, &a, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMemcpy(neg_one, &b, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaMemcpy(zero, &c, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_RESULT(cudaStreamCreate(&stream1));

}

mycuda::~mycuda()
{
  if(d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
  if(d_hash_vals) CUDA_CHECK_RESULT(cudaFree(d_hash_vals));
  if(d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
  if(d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
  CUDA_CHECK_RESULT(cudaFree(d_idx));
  cublasDestroy(handle);
  CUDA_CHECK_RESULT(cudaStreamDestroy(stream1));
}

// A must be an m by n matrix
// x must be an n-vector
// y must be an m-vector

void mycuda::set_matrix(size_t num_pancakes, size_t num_hash, const float* A, const float* g_vals)
{
  if(!d_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_hash_vals, num_hash * sizeof(float)));

  if(d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_a, num_hash * num_pancakes * sizeof(*A)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_a, A, num_hash * num_pancakes * sizeof(*A), cudaMemcpyHostToDevice));

  if(d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_g_vals, num_pancakes * sizeof(float)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_g_vals, g_vals, num_pancakes * sizeof(*g_vals), cudaMemcpyHostToDevice));

  if(d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, num_pancakes * sizeof(float)));

  filler.resize(num_pancakes, NUM_PANCAKES);
  if(num_pancakes_constant) CUDA_CHECK_RESULT(cudaFree(num_pancakes_constant));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&num_pancakes_constant, num_pancakes * sizeof(float)));
  CUDA_CHECK_RESULT(cudaMemcpy(num_pancakes_constant, filler.data(), num_pancakes * sizeof(*g_vals), cudaMemcpyHostToDevice));
}

//TODO: Collect batches of up to 1000? nodes before performing copy 
__declspec(noinline) float mycuda::min_vector_matrix(size_t num_pancakes, size_t num_hash, const float* hash_vals)
{
  int min_idx;
  float compare_answer;

  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_hash_vals, hash_vals, sizeof(float) * num_hash, cudaMemcpyHostToDevice));

  CUBLAS_CHECK_RESULT(cublasSgemv(handle, CUBLAS_OP_T, num_hash, num_pancakes, neg_one, d_a, num_hash, d_hash_vals, 1, zero, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, d_g_vals, 1, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasSaxpy(handle, num_pancakes, one, num_pancakes_constant, 1, d_mult_results, 1));
  CUBLAS_CHECK_RESULT(cublasIsamin(handle, num_pancakes, d_mult_results, 1, d_idx));
  CUDA_CHECK_RESULT(cudaMemcpyAsync(&min_idx, d_idx, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RESULT(cudaMemcpyAsync(&compare_answer, d_mult_results + (min_idx - 1), sizeof(float), cudaMemcpyDeviceToHost));
  return compare_answer;
}