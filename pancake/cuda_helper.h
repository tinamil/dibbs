#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
