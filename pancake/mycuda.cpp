#pragma once
#include "mycuda.h"

void mycuda::set_ptrs(size_t num_pancakes, float* A, float* g_vals)
{
  d_a = A;
  d_g_vals = g_vals;
  
  if (num_pancakes > d_mult_results_size) {
    d_mult_results_size = num_pancakes;
    if (d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
    CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, MAX_BATCH * num_pancakes * sizeof(float)));
  }
}

void mycuda::set_matrix(size_t num_pancakes, const float* A, const float* g_vals)
{
  if(d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_g_vals, num_pancakes * sizeof(float)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_g_vals, g_vals, num_pancakes * sizeof(float), cudaMemcpyHostToDevice));

  if(d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_a, MAX_PANCAKES * num_pancakes * sizeof(float)));
  CUDA_CHECK_RESULT(cudaMemcpy(d_a, A, MAX_PANCAKES * num_pancakes * sizeof(float), cudaMemcpyHostToDevice));

  if(d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, MAX_BATCH * num_pancakes * sizeof(float)));
}

float* mycuda::batch_vector_matrix(size_t num_pancakes, size_t num_vals, float* hash_vals)
{
  assert(num_vals <= MAX_BATCH);

  CUDA_CHECK_RESULT(cudaMemcpy(d_batch_hash_vals, hash_vals, sizeof(float) * num_vals * MAX_PANCAKES, cudaMemcpyHostToDevice));
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_pancakes, num_vals, MAX_PANCAKES, one, d_a, MAX_PANCAKES, d_batch_hash_vals, MAX_PANCAKES, zero, d_mult_results, num_pancakes);
  vector_add(num_vals, num_pancakes, d_g_vals, d_mult_results);
  reduce_min(num_vals, num_pancakes, d_mult_results, d_batch_answers);
  CUDA_CHECK_RESULT(cudaMemcpy(batch_answers, d_batch_answers, sizeof(float) * num_vals, cudaMemcpyDeviceToHost));

  return batch_answers;
}