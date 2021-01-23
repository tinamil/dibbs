#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cassert>
#include <vector>
#include "custom_kernels.cuh"
#include "Pancake.h"
#include "cuda_helper.h"
#include "ftf-pancake.h"
#include "hash_array.h"

class mycuda
{
  static inline cublasHandle_t  handle = nullptr;
  static inline float* one = nullptr;
  static inline float* neg_one = nullptr;
  static inline float* zero = nullptr;

  size_t my_num_pancakes = 0;
  size_t max_other_pancakes = 0;
  size_t d_mult_results_size = 0;
  uint32_t* d_hash_vals = nullptr;
  uint32_t* d_mult_results = nullptr;
  uint32_t* d_answers = nullptr;
  uint32_t* h_answers = nullptr;
  uint32_t* d_a = nullptr;
  uint32_t* d_g_vals = nullptr;

public:
  cudaStream_t stream = nullptr;
  size_t other_num_pancakes = 0;
  hash_array* h_hash_vals = nullptr;
  //static constexpr size_t MAX_BATCH = 16384;

  mycuda()
  {
    std::cout << "Allocating CUDA variables and stream\n";
    initialize();
    CUDA_CHECK_RESULT(cudaStreamCreate(&stream));
  }

  ~mycuda()
  {
    cudaDeviceSynchronize();
    if(stream) { cudaStreamDestroy(stream); stream = nullptr; }
    if(h_answers) { cudaFreeHost(h_answers); h_answers = nullptr; }
    if(d_answers) { cudaFree(d_answers); d_answers = nullptr; }
    if(h_hash_vals) { cudaFreeHost(h_hash_vals); h_hash_vals = nullptr; }
    if(d_hash_vals) { cudaFree(d_hash_vals); d_hash_vals = nullptr; }
    if(d_mult_results) { cudaFree(d_mult_results); d_mult_results = nullptr; }
  }

  void set_d_hash_vals(uint32_t* d_vals)
  {
    if(d_hash_vals) {
      cudaFree(d_hash_vals);
    }
    d_hash_vals = d_vals;
  }

  void clear_d_hash_vals()
  {
    d_hash_vals = nullptr;
  }

  uint32_t* get_answers()
  {
    cudaStreamSynchronize(stream);
    return h_answers;
  }

  static void initialize()
  {
    if(!handle)
    {
      CUBLAS_CHECK_RESULT(cublasCreate(&handle));
      CUBLAS_CHECK_RESULT(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      CUBLAS_CHECK_RESULT(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
      CUBLAS_CHECK_RESULT(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
    static constexpr int a = 1, b = -1, c = 0;
    if(!one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(one, &a, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!neg_one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&neg_one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(neg_one, &b, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!zero)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&zero, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(zero, &c, sizeof(float), cudaMemcpyHostToDevice));
    }
    //if (!compare_answer) CUDA_CHECK_RESULT(cudaHostAlloc(&compare_answer, sizeof(float), cudaHostAllocDefault));
    //if (!batch_answers) CUDA_CHECK_RESULT(cudaHostAlloc(&batch_answers, sizeof(float) * MAX_BATCH, cudaHostAllocDefault));

    //if (!d_batch_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_hash_vals, MAX_BATCH * MAX_PANCAKES * sizeof(float)));
    //if (!d_batch_answers) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_answers, MAX_BATCH * sizeof(float)));
  }
  void set_ptrs(size_t m_rows, size_t n_cols, uint32_t* A, uint32_t* g_vals);
  //void set_matrix(size_t m_rows, const float* A, const float* g_vals);

  void batch_vector_matrix();
  void load_then_batch_vector_matrix();
};

