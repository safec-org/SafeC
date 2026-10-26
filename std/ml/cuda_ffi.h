#pragma once
// Internal: included by the CUDA backend sources (gpu_cuda.sc,
// gpu_cuda_attention.sc, gpu_cudnn.sc, device_cuda.sc, dsp/fft_cuda.sc), not
// a public header. The opaque handle types of the CUDA driver API, NVRTC,
// cuBLAS / cuBLASLt and cuDNN, for their 'extern' declarations: each C
// 'void*' handle (CUcontext, CUevent, cublasHandle_t, ...) is a pointer to
// its own marker struct here, so one kind of handle can't be passed where
// another is expected. None is ever constructed or read; only the libraries
// they come from look behind them.
#include <std/ml/gpu_cuda.h>

namespace std {

// CUstream and CUmodule are gpu_cuda.h's CudaNativeStream / CudaNativeModule.
struct CuContext_ { char opaque_; };        // CUcontext
struct CuEvent_ { char opaque_; };          // CUevent
struct CuFunction_ { char opaque_; };       // CUfunction
struct CuGraph_ { char opaque_; };          // CUgraph
struct CuGraphExec_ { char opaque_; };      // CUgraphExec
// Device memory as the C APIs that take it as a pointer see it (cuBLAS,
// cuDNN, cuFFT, a kernel's pointer arguments): a CUdeviceptr's value. Never
// dereferenced on the host.
struct CuDevPtr_ { char opaque_; };
// One kernel argument as cuLaunchKernel's 'kernelParams' sees it: the
// address of the argument's value (whatever its type), which the driver
// copies as the kernel's parameter layout says.
struct CuKernelArg_ { char opaque_; };
struct NvrtcProgram_ { char opaque_; };     // nvrtcProgram
struct CublasHandle_ { char opaque_; };     // cublasHandle_t
struct CublasLtHandle_ { char opaque_; };   // cublasLtHandle_t
struct CublasLtDesc_ { char opaque_; };     // cublasLtMatmulDesc_t
struct CublasLtLayout_ { char opaque_; };   // cublasLtMatrixLayout_t
struct CublasLtPref_ { char opaque_; };     // cublasLtMatmulPreference_t
struct CudnnHandle_ { char opaque_; };      // cudnnHandle_t
struct CudnnTensorDesc_ { char opaque_; };  // cudnnTensorDescriptor_t
struct CudnnFilterDesc_ { char opaque_; };  // cudnnFilterDescriptor_t
struct CudnnConvDesc_ { char opaque_; };    // cudnnConvolutionDescriptor_t
struct CudnnActDesc_ { char opaque_; };     // cudnnActivationDescriptor_t
struct CudnnPoolDesc_ { char opaque_; };    // cudnnPoolingDescriptor_t
struct CudnnBackendDesc_ { char opaque_; }; // cudnnBackendDescriptor_t
// A dynamically loaded library (dlopen) and a symbol looked up in it,
// untyped until the caller casts it to the function's real signature.
struct CuLib_ { char opaque_; };
typedef ?fn void() CuSym_;

} // namespace std
