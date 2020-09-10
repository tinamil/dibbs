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
#ifdef NDEBUG
#define CUDA_CHECK_RESULT(f) (f)
#define CUBLAS_CHECK_RESULT(f) (f)
#else
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
#endif

class mycuda
{
  cublasHandle_t  handle;
  float* d_a, * d_hash_vals, * d_batch_hash_vals, * d_mult_results, * d_g_vals, * num_pancakes_constant, * d_batch_answers;
  int* d_idx, * min_idx;
  float* one, * neg_one, * zero;
  float* compare_answer;
  float* batch_answers;
public:
  static constexpr size_t MAX_BATCH = (NUM_PANCAKES - 1) * 64;
  mycuda();
  ~mycuda();
  void set_matrix(size_t m_rows, size_t n_cols, const float* A, const float* g_vals);
  float min_vector_matrix(size_t a_rows, size_t a_cols, const float* hash_vals);
  float* batch_vector_matrix(size_t num_pancakes, size_t num_hash, size_t num_vals, const float* hash_vals);
  float matrix_matrix(size_t num_pancakes_A, size_t num_pancakes_B, size_t num_hash, const float* A, const float* B);
};

