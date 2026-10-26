#pragma once
// Internal: included by the CUDA backend sources (gpu_cuda.sc,
// gpu_cuda_attention.sc, gpu_cudnn.sc, device_cuda.sc, dsp/fft_cuda.sc), not
// a public header. The CUDA driver API, NVRTC and cuBLAS / cuBLASLt
// declarations they share, and the opaque handle types of those libraries
// and cuDNN for their 'extern' declarations: each C 'void*' handle (CUcontext, CUevent, cublasHandle_t, ...) is a pointer to
// its own marker struct here, so one kind of handle can't be passed where
// another is expected. None is ever constructed or read; only the libraries
// they come from look behind them.
#include <std/stddef.h>
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

// ── Driver API / NVRTC / cuBLAS / cuBLASLt C ABI (cuda.h, nvrtc.h, cublas_v2.h, cublasLt.h) ──
extern int cuInit(unsigned int flags);
extern int cuDeviceGetCount(int* count);
extern int cuDeviceGet(int* device, int ordinal);
extern int cuDeviceGetAttribute(int* value, int attr, int device);
extern int cuDeviceGetName(char* name, int len, int device);
extern int cuDevicePrimaryCtxRetain(CuContext_** pctx, int device);
extern int cuCtxSetCurrent(CuContext_* ctx);
extern int cuCtxGetCurrent(CuContext_** pctx);
extern int cuStreamCreate(CudaNativeStream** stream, unsigned int flags);
extern int cuStreamSynchronize(CudaNativeStream* stream);
extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemAllocHost_v2(unsigned char** pp, size_t bytes);
extern int cuMemcpyHtoDAsync_v2(unsigned long long dst, const unsigned char* src, size_t bytes, CudaNativeStream* stream);
extern int cuMemcpyDtoHAsync_v2(unsigned char* dst, unsigned long long src, size_t bytes, CudaNativeStream* stream);
extern int cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src, size_t bytes, CudaNativeStream* stream);
extern int cuEventCreate(CuEvent_** event, unsigned int flags);
extern int cuEventRecord(CuEvent_* event, CudaNativeStream* stream);
extern int cuEventSynchronize(CuEvent_* event);
extern int cuStreamWaitEvent(CudaNativeStream* stream, CuEvent_* event, unsigned int flags);
extern int cuCtxGetDevice(int* device);
extern int cuEventElapsedTime(float* ms, CuEvent_* start, CuEvent_* end);
extern int cuEventDestroy_v2(CuEvent_* event);
extern int cuFuncSetAttribute(CuFunction_* func, int attr, int value);
extern int cuMemsetD32Async(unsigned long long dst, unsigned int value, size_t n, CudaNativeStream* stream);
extern int cuModuleLoadData(CudaNativeModule** module, const char* image);
extern int cuModuleGetFunction(CuFunction_** func, CudaNativeModule* module, const char* name);
extern int cuLaunchKernel(CuFunction_* f, unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz,
                          unsigned int shared, CudaNativeStream* stream, CuKernelArg_** params, CuKernelArg_** extra);
extern int cuStreamBeginCapture_v2(CudaNativeStream* stream, int mode);
extern int cuStreamEndCapture(CudaNativeStream* stream, CuGraph_** graph);
extern int cuGraphInstantiateWithFlags(CuGraphExec_** exec, CuGraph_* graph, unsigned long long flags);
extern int cuGraphLaunch(CuGraphExec_* exec, CudaNativeStream* stream);
extern int cuGraphDestroy(CuGraph_* graph);
extern int cuGraphExecDestroy(CuGraphExec_* exec);

extern int nvrtcCreateProgram(NvrtcProgram_** prog, const char* src, const char* name, int numHeaders,
                              const char** headers, const char** includeNames);
extern int nvrtcCompileProgram(NvrtcProgram_* prog, int numOptions, const char** options);
extern int nvrtcGetCUBINSize(NvrtcProgram_* prog, size_t* size);
extern int nvrtcGetCUBIN(NvrtcProgram_* prog, char* cubin);
extern int nvrtcGetProgramLogSize(NvrtcProgram_* prog, size_t* size);
extern int nvrtcGetProgramLog(NvrtcProgram_* prog, char* log);
extern int nvrtcDestroyProgram(NvrtcProgram_** prog);

extern int cublasCreate_v2(CublasHandle_** handle);
extern int cublasSetStream_v2(CublasHandle_* handle, CudaNativeStream* stream);
extern int cublasSetWorkspace_v2(CublasHandle_* handle, CuDevPtr_* workspace, size_t bytes);
extern int cublasSetMathMode(CublasHandle_* handle, int mode);
extern int cublasSgemm_v2(CublasHandle_* handle, int transa, int transb, int m, int n, int k,
                          const float* alpha, const float* A, int lda,
                          const float* B, int ldb,
                          const float* beta, float* C, int ldc);

// alpha / beta: float (computeType CUBLAS_COMPUTE_32F*)
extern int cublasGemmEx(CublasHandle_* handle, int transa, int transb, int m, int n, int k, const float* alpha,
                        const CuDevPtr_* a, int aType, int lda, const CuDevPtr_* b, int bType, int ldb,
                        const float* beta, CuDevPtr_* c, int cType, int ldc, int computeType, int algo);
extern int cublasGemmStridedBatchedEx(CublasHandle_* handle, int transa, int transb, int m, int n, int k, const float* alpha,
                                      const CuDevPtr_* a, int aType, int lda, long long strideA,
                                      const CuDevPtr_* b, int bType, int ldb, long long strideB, const float* beta,
                                      CuDevPtr_* c, int cType, int ldc, long long strideC, int batch,
                                      int computeType, int algo);
extern int cublasSgemmStridedBatched(CublasHandle_* handle, int transa, int transb, int m, int n, int k,
                                     const float* alpha, const float* a, int lda, long long sa,
                                     const float* b, int ldb, long long sb, const float* beta,
                                     float* c, int ldc, long long sc, int batch);
extern int cublasLtCreate(CublasLtHandle_** handle);
extern int cublasLtMatmulDescCreate(CublasLtDesc_** desc, int computeType, int scaleType);
// buf: the attribute's value, 'size' bytes
extern int cublasLtMatmulDescSetAttribute(CublasLtDesc_* desc, int attr, const unsigned char* buf, size_t size);
extern int cublasLtMatrixLayoutCreate(CublasLtLayout_** layout, int type, unsigned long long rows,
                                      unsigned long long cols, long long ld);
extern int cublasLtMatmulPreferenceCreate(CublasLtPref_** pref);
extern int cublasLtMatmulPreferenceSetAttribute(CublasLtPref_* pref, int attr, const unsigned char* buf, size_t size);
extern int cublasLtMatmulPreferenceDestroy(CublasLtPref_* pref);
// results: 'requested' cublasLtMatmulHeuristicResult_t (LT_RESULT_WORDS words each)
extern int cublasLtMatmulAlgoGetHeuristic(CublasLtHandle_* handle, CublasLtDesc_* desc, CublasLtLayout_* a,
                                          CublasLtLayout_* b, CublasLtLayout_* c, CublasLtLayout_* d,
                                          CublasLtPref_* pref, int requested, unsigned long* results, int* returned);
// algo: a heuristic result's leading cublasLtMatmulAlgo_t
extern int cublasLtMatmul(CublasLtHandle_* handle, CublasLtDesc_* desc, const float* alpha, const CuDevPtr_* A,
                          CublasLtLayout_* Adesc, const CuDevPtr_* B, CublasLtLayout_* Bdesc, const float* beta,
                          const CuDevPtr_* C, CublasLtLayout_* Cdesc, CuDevPtr_* D, CublasLtLayout_* Ddesc,
                          const unsigned long* algo, CuDevPtr_* workspace, size_t wsBytes,
                          CudaNativeStream* stream);

} // namespace std
