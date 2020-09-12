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

static inline std::string errorString(cublasStatus_t errorCode)
{
  switch(errorCode)
  {
    #define STR(r) case CUBLAS_STATUS_ ##r: return #r
    STR(SUCCESS);
    STR(NOT_INITIALIZED);
    STR(ALLOC_FAILED);
    STR(INVALID_VALUE);
    STR(ARCH_MISMATCH);
    STR(MAPPING_ERROR);
    STR(EXECUTION_FAILED);
    STR(INTERNAL_ERROR);
    STR(NOT_SUPPORTED);
    STR(LICENSE_ERROR);
    #undef STR
    default:
      return "UNKNOWN_ERROR";
  }
}

static inline std::string errorString(cudaError_t errorCode)
{
  switch(errorCode)
  {
    #define STR(r) case cudaError ##r: return #r
    STR(MissingConfiguration);
    STR(MemoryAllocation);
    STR(InitializationError);
    STR(LaunchFailure);
    STR(PriorLaunchFailure);
    STR(LaunchTimeout);
    STR(LaunchOutOfResources);
    STR(InvalidDeviceFunction);
    STR(InvalidConfiguration);
    STR(InvalidDevice);
    STR(InvalidValue);
    STR(InvalidPitchValue);
    STR(InvalidSymbol);
    STR(MapBufferObjectFailed);
    STR(UnmapBufferObjectFailed);
    STR(InvalidHostPointer);
    STR(InvalidDevicePointer);
    STR(InvalidTexture);
    STR(InvalidTextureBinding);
    STR(InvalidChannelDescriptor);
    STR(InvalidMemcpyDirection);
    STR(AddressOfConstant);
    STR(TextureFetchFailed);
    STR(TextureNotBound);
    STR(SynchronizationError);
    STR(InvalidFilterSetting);
    STR(InvalidNormSetting);
    STR(MixedDeviceExecution);
    STR(CudartUnloading);
    STR(Unknown);
    STR(NotYetImplemented);
    STR(MemoryValueTooLarge);
    STR(InvalidResourceHandle);
    STR(NotReady);
    STR(InsufficientDriver);
    STR(SetOnActiveProcess);
    STR(NoDevice);
    STR(StartupFailure);
    STR(ApiFailureBase);
    #undef STR
    default:
      return "UNKNOWN_ERROR";
  }
}
//#ifdef NDEBUG
//#define CUDA_CHECK_RESULT(f) (f)
//#define CUBLAS_CHECK_RESULT(f) (f)
//#else
#define CUDA_CHECK_RESULT(f)																				\
{																										\
	cudaError_t res = (f);																					\
	if (res != cudaSuccess)																				\
	{																									\
		std::cout << "Fatal : cudaError is \"" << errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << std::endl; \
		assert(res == cudaSuccess);																		\
	}																									\
}
#define CUBLAS_CHECK_RESULT(f)																				\
{																										\
	cublasStatus_t res = (f);																					\
	if (res != CUBLAS_STATUS_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : cublasError is \"" << errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << std::endl; \
		assert(res == CUBLAS_STATUS_SUCCESS);																		\
	}																									\
}
//#endif

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
public:
  static constexpr size_t MAX_BATCH = 512;
  static void initialize()
  {
    if(!handle)
    {
      CUBLAS_CHECK_RESULT(cublasCreate(&handle));
      CUBLAS_CHECK_RESULT(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      //CUBLAS_CHECK_RESULT(cublasSetStream(handle, cudaStreamPerThread));
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
  static void set_matrix(size_t m_rows, const float* A, const float* g_vals);
  static float* batch_vector_matrix(size_t num_pancakes, size_t num_vals, float* hash_vals);
};

