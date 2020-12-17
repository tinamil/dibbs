#pragma once
#include "mycuda.h"

void mycuda::set_ptrs(size_t m_rows, size_t n_cols, float* A, float* g_vals)
{
  d_a = A;
  d_g_vals = g_vals;
  other_num_pancakes = n_cols;
  my_num_pancakes = m_rows;

  if(n_cols > max_other_pancakes) {
    max_other_pancakes = n_cols;
    if(h_answers) { cudaFreeHost(h_answers);       h_answers = nullptr; }
    if(d_answers) { cudaFree(d_answers);           d_answers = nullptr; }
    if(h_hash_vals) { cudaFreeHost(h_hash_vals);   h_hash_vals = nullptr; }
    if(d_hash_vals) { cudaFree(d_hash_vals);       d_hash_vals = nullptr; }
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_answers, max_other_pancakes * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK_RESULT(cudaMalloc(&d_answers, max_other_pancakes * sizeof(float)));
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_hash_vals, max_other_pancakes * MAX_PANCAKES * sizeof(float), cudaHostAllocWriteCombined));
    CUDA_CHECK_RESULT(cudaMalloc(&d_hash_vals, max_other_pancakes * MAX_PANCAKES * sizeof(float)));
  }

  if(my_num_pancakes > max_my_pancakes || n_cols > max_other_pancakes) {
    if(d_mult_results) {
      CUDA_CHECK_RESULT(cudaFree(d_mult_results));
      d_mult_results = nullptr;
    }
    max_my_pancakes = my_num_pancakes;
    CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, max_my_pancakes * max_other_pancakes * sizeof(float)));
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
  //assert(other_num_pancakes <= MAX_BATCH);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_hash_vals, h_hash_vals, sizeof(float) * other_num_pancakes * MAX_PANCAKES, cudaMemcpyHostToDevice, stream));
  cublasSetStream(handle, stream);
  CUBLAS_CHECK_RESULT(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, my_num_pancakes, other_num_pancakes, MAX_PANCAKES, one, d_a, MAX_PANCAKES, d_hash_vals, MAX_PANCAKES, zero, d_mult_results, max_my_pancakes));
  vector_add(stream, other_num_pancakes, my_num_pancakes, d_g_vals, d_mult_results);
  reduce_min(stream, other_num_pancakes, my_num_pancakes, d_mult_results, d_answers);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(h_answers, d_answers, sizeof(float) * other_num_pancakes, cudaMemcpyDeviceToHost, stream));
}