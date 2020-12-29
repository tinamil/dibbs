#pragma once
#include "mycuda.h"

void mycuda::set_ptrs(size_t m_rows, size_t n_cols, uint32_t* A, uint32_t* g_vals)
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
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_answers, max_other_pancakes * sizeof(uint32_t), cudaHostAllocDefault));
    CUDA_CHECK_RESULT(cudaMalloc(&d_answers, max_other_pancakes * sizeof(uint32_t)));
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_hash_vals, max_other_pancakes * sizeof(hash_array), cudaHostAllocWriteCombined));
    CUDA_CHECK_RESULT(cudaMalloc(&d_hash_vals, max_other_pancakes * sizeof(hash_array)));
  }

  if(other_num_pancakes * my_num_pancakes > d_mult_results_size) {
    d_mult_results_size = other_num_pancakes * my_num_pancakes;
    if(d_mult_results) {
      CUDA_CHECK_RESULT(cudaFree(d_mult_results));
      d_mult_results = nullptr;
    }
    CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, d_mult_results_size * sizeof(uint32_t)));
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


void mycuda::load_then_batch_vector_matrix()
{
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_hash_vals, h_hash_vals, sizeof(uint32_t) * other_num_pancakes * NUM_INTS_PER_PANCAKE, cudaMemcpyHostToDevice, stream));
  batch_vector_matrix();
}

void mycuda::batch_vector_matrix()
{
  //cublasSetStream(handle, stream);
  //CUBLAS_CHECK_RESULT(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, my_num_pancakes, other_num_pancakes, MAX_PANCAKES, one, d_a, MAX_PANCAKES, d_hash_vals, MAX_PANCAKES, zero, d_mult_results, my_num_pancakes));
  bitwise_set_intersection(stream, my_num_pancakes, other_num_pancakes, d_a, d_g_vals, d_hash_vals, d_mult_results);
  //vector_add(stream, other_num_pancakes, my_num_pancakes, d_g_vals, d_mult_results);
  reduce_min(stream, other_num_pancakes, my_num_pancakes, d_mult_results, d_answers);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(h_answers, d_answers, sizeof(uint32_t) * other_num_pancakes, cudaMemcpyDeviceToHost, stream));
}