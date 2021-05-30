#pragma once
#include "mycuda.h"

void mycuda::set_ptrs(size_t m_rows, size_t n_cols, uint32_t* nodes, uint32_t* g_vals)
{
  d_a = nodes;
  d_g_vals = g_vals;
  other_num_nodes = n_cols;
  my_num_nodes = m_rows;

  if(n_cols > max_other_nodes) {
    max_other_nodes = n_cols;
    if(h_answers) { cudaFreeHost(h_answers);       h_answers = nullptr; }
    if(d_answers) { cudaFree(d_answers);           d_answers = nullptr; }
    if(h_nodes) { cudaFreeHost(h_nodes);           h_nodes = nullptr; }
    if(d_nodes) { cudaFree(d_nodes);               d_nodes = nullptr; }
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_answers, max_other_nodes * sizeof(uint32_t), cudaHostAllocDefault));
    CUDA_CHECK_RESULT(cudaMalloc(&d_answers, max_other_nodes * sizeof(uint32_t)));
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_nodes, max_other_nodes * sizeof(uint32_t), cudaHostAllocWriteCombined));
    CUDA_CHECK_RESULT(cudaMalloc(&d_nodes, max_other_nodes * sizeof(uint32_t)));
  }

  size_t results_size = MAX(int_div_ceil(other_num_nodes, 16), 1024) * my_num_nodes;
  if(results_size > d_mult_results_size) {
    d_mult_results_size = MAX(results_size, d_mult_results_size * 1.5);
    if(d_mult_results) {
      CUDA_CHECK_RESULT(cudaFree(d_mult_results));
      d_mult_results = nullptr;
    }
    size_t d_mem = d_mult_results_size * sizeof(uint32_t);
    size_t free_mem, total_mem;
    CUDA_CHECK_RESULT(cudaMemGetInfo(&free_mem, &total_mem));
    if(d_mem > free_mem) {
      if(other_num_nodes * my_num_nodes * sizeof(uint32_t) > free_mem) {
        std::cout << "Out of memory: " << free_mem / 1024. / 1024. / 1024. << " GB\n";
        throw new std::exception("Out of memory");
      }
      else {
        d_mult_results_size = other_num_nodes * my_num_nodes;
        d_mem = d_mult_results_size * sizeof(uint32_t);
      }
    }
    CUDA_CHECK_RESULT(cudaMalloc((void**)&d_mult_results, d_mem));
  }

}

void mycuda::load_then_batch_vector_matrix()
{
  CUDA_CHECK_RESULT(cudaMemcpyAsync(d_nodes, h_nodes, sizeof(uint32_t) * other_num_nodes, cudaMemcpyHostToDevice, stream));
  batch_vector_matrix();
}

void mycuda::batch_vector_matrix()
{
  assert(d_coordinates);
  bitwise_set_intersection(stream, my_num_nodes, other_num_nodes, d_a, d_g_vals, d_nodes, d_mult_results, d_answers, d_coordinates);
  CUDA_CHECK_RESULT(cudaMemcpyAsync(h_answers, d_answers, sizeof(uint32_t) * other_num_nodes, cudaMemcpyDeviceToHost, stream));
}