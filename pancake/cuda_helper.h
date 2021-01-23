#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>

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
    STR(InvalidValue);
    STR(MemoryAllocation);
    STR(InitializationError);
    STR(CudartUnloading);
    STR(ProfilerDisabled);
    STR(ProfilerNotInitialized);
    STR(ProfilerAlreadyStarted);
    STR(ProfilerAlreadyStopped);
    STR(InvalidConfiguration);
    STR(InvalidPitchValue);
    STR(InvalidSymbol);
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
    STR(NotYetImplemented);
    STR(MemoryValueTooLarge);
    STR(StubLibrary);
    STR(InsufficientDriver);
    STR(CallRequiresNewerDriver);
    STR(InvalidSurface);
    STR(DuplicateVariableName);
    STR(DuplicateTextureName);
    STR(DuplicateSurfaceName);
    STR(DevicesUnavailable);
    STR(IncompatibleDriverContext);
    STR(MissingConfiguration);
    STR(PriorLaunchFailure);
    STR(LaunchMaxDepthExceeded);
    STR(LaunchFileScopedTex);
    STR(LaunchFileScopedSurf);
    STR(SyncDepthExceeded);
    STR(LaunchPendingCountExceeded);
    STR(InvalidDeviceFunction);
    STR(NoDevice);
    STR(InvalidDevice);
    STR(DeviceNotLicensed);
    STR(StartupFailure);
    STR(InvalidKernelImage);
    STR(DeviceUninitialized);
    STR(MapBufferObjectFailed);
    STR(UnmapBufferObjectFailed);
    STR(ArrayIsMapped);
    STR(AlreadyMapped);
    STR(NoKernelImageForDevice);
    STR(AlreadyAcquired);
    STR(NotMapped);
    STR(NotMappedAsArray);
    STR(NotMappedAsPointer);
    STR(ECCUncorrectable);
    STR(UnsupportedLimit);
    STR(DeviceAlreadyInUse);
    STR(PeerAccessUnsupported);
    STR(InvalidPtx);
    STR(InvalidGraphicsContext);
    STR(NvlinkUncorrectable);
    STR(JitCompilerNotFound);
    STR(UnsupportedPtxVersion);
    STR(InvalidSource);
    STR(FileNotFound);
    STR(SharedObjectSymbolNotFound);
    STR(SharedObjectInitFailed);
    STR(OperatingSystem);
    STR(InvalidResourceHandle);
    STR(IllegalState);
    STR(SymbolNotFound);
    STR(NotReady);
    STR(IllegalAddress);
    STR(LaunchOutOfResources);
    STR(LaunchTimeout);
    STR(LaunchIncompatibleTexturing);
    STR(PeerAccessAlreadyEnabled);
    STR(PeerAccessNotEnabled);
    STR(SetOnActiveProcess);
    STR(ContextIsDestroyed);
    STR(Assert);
    STR(TooManyPeers);
    STR(HostMemoryAlreadyRegistered);
    STR(HostMemoryNotRegistered);
    STR(HardwareStackError);
    STR(IllegalInstruction);
    STR(MisalignedAddress);
    STR(InvalidAddressSpace);
    STR(InvalidPc);
    STR(LaunchFailure);
    STR(CooperativeLaunchTooLarge);
    STR(NotPermitted);
    STR(NotSupported);
    STR(SystemNotReady);
    STR(SystemDriverMismatch);
    STR(CompatNotSupportedOnDevice);
    STR(StreamCaptureUnsupported);
    STR(StreamCaptureInvalidated);
    STR(StreamCaptureMerge);
    STR(StreamCaptureUnmatched);
    STR(StreamCaptureUnjoined);
    STR(StreamCaptureIsolation);
    STR(StreamCaptureImplicit);
    STR(CapturedEvent);
    STR(StreamCaptureWrongThread);
    STR(Timeout);
    STR(GraphExecUpdateFailure);
    STR(Unknown);
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
		if(res != cudaSuccess) throw std::exception("FATAL CUDA"); \
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
