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

class mycuda
{
  static inline cublasHandle_t  handle = nullptr;
  static inline float* d_a = nullptr;
  static inline float* d_batch_hash_vals = nullptr;
  static inline float* d_mult_results = nullptr;
  static inline float* d_g_vals = nullptr;
  static inline float* d_batch_answers = nullptr;
  static inline float* one = nullptr;
  static inline float* neg_one = nullptr;
  static inline float* zero = nullptr;
  static inline float* compare_answer = nullptr;
  static inline float* batch_answers = nullptr;
  static inline cudaStream_t stream = nullptr;

public:
  static constexpr size_t MAX_BATCH = 256;
  static void initialize()
  {
    if(!handle)
    {
      CUBLAS_CHECK_RESULT(cublasCreate(&handle));
      CUBLAS_CHECK_RESULT(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      CUDA_CHECK_RESULT(cudaStreamCreate(&stream));
      CUBLAS_CHECK_RESULT(cublasSetStream(handle, stream));
      CUBLAS_CHECK_RESULT(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
      CUBLAS_CHECK_RESULT(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    static constexpr float a = 1, b = -1, c = 0;
    if(!one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpyAsync(one, &a, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!neg_one)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&neg_one, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpyAsync(neg_one, &b, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!zero)
    {
      CUDA_CHECK_RESULT(cudaMalloc((void**)&zero, sizeof(float)));
      CUDA_CHECK_RESULT(cudaMemcpyAsync(zero, &c, sizeof(float), cudaMemcpyHostToDevice));
    }
    if(!compare_answer) CUDA_CHECK_RESULT(cudaHostAlloc(&compare_answer, sizeof(float), cudaHostAllocDefault));
    if(!batch_answers) CUDA_CHECK_RESULT(cudaHostAlloc(&batch_answers, sizeof(float) * MAX_BATCH, cudaHostAllocDefault));

    if(!d_batch_hash_vals) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_hash_vals, MAX_BATCH * MAX_PANCAKES * sizeof(float)));
    if(!d_batch_answers) CUDA_CHECK_RESULT(cudaMalloc((void**)&d_batch_answers, MAX_BATCH * sizeof(float)));
  }
  static void set_ptrs(size_t m_rows, float* A, float* g_vals);
  static void set_matrix(size_t m_rows, const float* A, const float* g_vals);
  static float* batch_vector_matrix(size_t num_pancakes, size_t num_vals, float* hash_vals);
};

