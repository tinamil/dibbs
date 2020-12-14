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

class mycuda
{
  static inline cublasHandle_t  handle = nullptr;
  static inline float *d_a = nullptr;
  //static inline float* d_batch_hash_vals = nullptr;
  //static inline float* d_mult_results = nullptr;
  //static inline size_t d_mult_results_size = 0;
  static inline float *d_g_vals = nullptr;
  //static inline float* d_batch_answers = nullptr;
  static inline float *one = nullptr;
  static inline float *neg_one = nullptr;
  static inline float *zero = nullptr;
  //static inline float* compare_answer = nullptr;
  //static inline float* batch_answers = nullptr;

  //static inline cudaStream_t cublas_stream1 = nullptr;

  size_t d_mult_results_size = 0;
  size_t num_pancakes_opposite_frontier = 0;
  float *d_hash_vals = nullptr;
  float *d_mult_results = nullptr;
  float *d_answers = nullptr;
  cudaStream_t stream = nullptr;
  float *h_answers = nullptr;

public:

  size_t num_vals = 0;
  float *h_hash_vals = nullptr;
  static constexpr size_t MAX_BATCH = 256;

  mycuda()
  {
    initialize();
    CUDA_CHECK_RESULT(cudaStreamCreate(&stream));
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_answers, sizeof(float) * MAX_BATCH, cudaHostAllocDefault));
    CUDA_CHECK_RESULT(cudaMalloc((void **)&d_answers, MAX_BATCH * sizeof(float)));
    CUDA_CHECK_RESULT(cudaHostAlloc(&h_hash_vals, MAX_BATCH * MAX_PANCAKES * sizeof(float), cudaHostAllocWriteCombined));
    CUDA_CHECK_RESULT(cudaMalloc((void **)&d_hash_vals, MAX_BATCH * MAX_PANCAKES * sizeof(float)));
  }

  ~mycuda()
  {
    cudaDeviceSynchronize();
    if(stream) cudaStreamDestroy(stream);
    if(h_answers) cudaFreeHost(h_answers);
    if(d_answers) cudaFree(d_answers);
    if(h_hash_vals) cudaFreeHost(h_hash_vals);
    if(d_hash_vals) cudaFree(d_hash_vals);
    if(d_mult_results) cudaFree(d_mult_results);
  }

  float* get_answers()
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
      CUBLAS_CHECK_RESULT(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    static constexpr float a = 1, b = -1, c = 0;
    if(!one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void **)&one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(one, &a, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!neg_one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void **)&neg_one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(neg_one, &b, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!zero)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void **)&zero, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpy(zero, &c, sizeof(float), cudaMemcpyHostToDevice));
    }
    //if (!compare_answer) CUDA_CHECK_RESULT(cudaHostAlloc(&compare_answer, sizeof(float), cudaHostAllocDefault));
    //if (!batch_answers) CUDA_CHECK_RESULT(cudaHostAlloc(&batch_answers, sizeof(float) * MAX_BATCH, cudaHostAllocDefault));

    //if (!d_batch_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_hash_vals, MAX_BATCH * MAX_PANCAKES * sizeof(float)));
    //if (!d_batch_answers) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_answers, MAX_BATCH * sizeof(float)));
  }
  void set_ptrs(size_t m_rows, float *A, float *g_vals);
  //void set_matrix(size_t m_rows, const float* A, const float* g_vals);
  void batch_vector_matrix();
};

