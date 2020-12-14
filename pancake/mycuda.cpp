#pragma once
#include "mycuda.h"

void mycuda::set_ptrs(size_t m_rows, float* A, float* g_vals)
{
  d_a = A;
  d_g_vals = g_vals;
  num_pancakes_opposite_frontier = m_rows;
  if (num_pancakes_opposite_frontier > d_mult_results_size) {
    d_mult_results_size = num_pancakes_opposite_frontier;
    if (d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
    CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, MAX_BATCH * num_pancakes_opposite_frontier * sizeof(float)));
  }
}

//void mycuda::set_matrix(size_t num_pancakes, const float* A, const float* g_vals)
//{
//  if (d_g_vals) CUDA_CHECK_RESULT(cudaFree(d_g_vals));
//  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_g_vals, num_pancakes * sizeof(float)));
//  CUDA_CHECK_RESULT(cudaMemcpy(d_g_vals, g_vals, num_pancakes * sizeof(float), cudaMemcpyHostToDevice));
//
//  if (d_a) CUDA_CHECK_RESULT(cudaFree(d_a));
//  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_a, MAX_PANCAKES * num_pancakes * sizeof(float)));
//  CUDA_CHECK_RESULT(cudaMemcpy(d_a, A, MAX_PANCAKES * num_pancakes * sizeof(float), cudaMemcpyHostToDevice));
//
//  if (d_mult_results) CUDA_CHECK_RESULT(cudaFree(d_mult_results));
//  CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, MAX_BATCH * num_pancakes * sizeof(float)));
//}

void mycuda::batch_vector_matrix()
{
  assert(num_vals <= MAX_BATCH);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_hash_vals, h_hash_vals, sizeof(float) * num_vals * MAX_PANCAKES, cudaMemcpyHostToDevice, stream));
  cublasSetStream(handle, stream);
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_pancakes_opposite_frontier, num_vals, MAX_PANCAKES, one, d_a, MAX_PANCAKES, d_hash_vals, MAX_PANCAKES, zero, d_mult_results, num_pancakes_opposite_frontier);
  vector_add(stream, num_vals, num_pancakes_opposite_frontier, d_g_vals, d_mult_results);
  reduce_min(stream, num_vals, num_pancakes_opposite_frontier, d_mult_results, d_answers);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(h_answers, d_answers, sizeof(float) * num_vals, cudaMemcpyDeviceToHost, stream));
}