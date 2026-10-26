#pragma once
// Internal: included by the ROCm backend sources (gpu_rocm.sc,
// device_rocm.sc, gpu_miopen.sc, dsp/fft_rocm.sc), not a public header. The
// HIP runtime, hiprtc and rocBLAS declarations they share, and the opaque
// handle types of those libraries and MIOpen / hipFFT for their 'extern'
// declarations: each C 'void*' handle (hipModule_t, rocblas_handle, ...) is a
// pointer to its own marker struct here, so one kind of handle can't be
// passed where another is expected. None is ever constructed or read; only
// the libraries they come from look behind them.
#include <std/stddef.h>
#include <std/ml/gpu_rocm.h>

namespace std {

// hipStream_t is gpu_rocm.h's RocmNativeStream.
struct HipModule_ { char opaque_; };        // hipModule_t
struct HipFunction_ { char opaque_; };      // hipFunction_t
struct HipGraph_ { char opaque_; };         // hipGraph_t
struct HipGraphExec_ { char opaque_; };     // hipGraphExec_t
struct HipGraphNode_ { char opaque_; };     // hipGraphNode_t
// Device memory (hipMalloc's result, a RocmBuf's dev): never dereferenced on
// the host.
struct HipDevPtr_ { char opaque_; };
// One kernel argument as hipModuleLaunchKernel's 'kernelParams' sees it: the
// address of the argument's value (whatever its type).
struct HipKernelArg_ { char opaque_; };
struct HiprtcProgram_ { char opaque_; };    // hiprtcProgram
struct RocblasHandle_ { char opaque_; };    // rocblas_handle
struct MiopenHandle_ { char opaque_; };     // miopenHandle_t
struct MiopenTensorDesc_ { char opaque_; }; // miopenTensorDescriptor_t
struct MiopenConvDesc_ { char opaque_; };   // miopenConvolutionDescriptor_t
struct MiopenPoolDesc_ { char opaque_; };   // miopenPoolingDescriptor_t
struct HipfftPlan_ { char opaque_; };       // hipfftHandle
// miopenConvAlgoPerf_t: the algorithm (fwd / bwdData / bwdWeights, a union),
// its time and its workspace bytes.
struct MiopenConvPerf_ {
    int           algo;
    float         time;
    unsigned long memory;
};

// ── HIP runtime / hiprtc / rocBLAS C ABI (hip_runtime_api.h, hiprtc.h, rocblas.h) ──
extern int hipInit(unsigned int flags);
extern int hipGetDeviceCount(int* count);
extern int hipDeviceGetAttribute(int* value, int attr, int device);
extern int hipDeviceGetName(char* name, int len, int device);
extern int hipSetDevice(int device);
extern int hipStreamCreate(RocmNativeStream** stream);
extern int hipStreamSynchronize(RocmNativeStream* stream);
extern int hipMalloc(HipDevPtr_** ptr, unsigned long size);
extern int hipFree(HipDevPtr_* ptr);
// dst / src: host or device addresses, as 'kind' (hipMemcpyKind) says
extern int hipMemcpyAsync(unsigned char* dst, const unsigned char* src, unsigned long size, int kind,
                          RocmNativeStream* stream);
extern int hipMemsetAsync(HipDevPtr_* dst, int value, unsigned long size, RocmNativeStream* stream);
extern int hipModuleLoadData(HipModule_** module, const char* image);
extern int hipModuleGetFunction(HipFunction_** func, HipModule_* module, const char* name);
extern int hipModuleLaunchKernel(HipFunction_* func, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int shared, RocmNativeStream* stream, HipKernelArg_** params,
                                 HipKernelArg_** extra);
extern int hipStreamBeginCapture(RocmNativeStream* stream, int mode);
extern int hipStreamEndCapture(RocmNativeStream* stream, HipGraph_** graph);
extern int hipGraphInstantiate(HipGraphExec_** exec, HipGraph_* graph, HipGraphNode_** errNode, char* log,
                               unsigned long logSize);
extern int hipGraphLaunch(HipGraphExec_* exec, RocmNativeStream* stream);
extern int hipGraphDestroy(HipGraph_* graph);
extern int hipGraphExecDestroy(HipGraphExec_* exec);

extern int hiprtcCreateProgram(HiprtcProgram_** prog, const char* src, const char* name, int numHeaders,
                               const char** headers, const char** includeNames);
extern int hiprtcCompileProgram(HiprtcProgram_* prog, int numOptions, const char** options);
extern int hiprtcGetCodeSize(HiprtcProgram_* prog, unsigned long* size);
extern int hiprtcGetCode(HiprtcProgram_* prog, char* code);
extern int hiprtcGetProgramLogSize(HiprtcProgram_* prog, unsigned long* size);
extern int hiprtcGetProgramLog(HiprtcProgram_* prog, char* log);
extern int hiprtcDestroyProgram(HiprtcProgram_** prog);

extern int rocblas_create_handle(RocblasHandle_** handle);
extern int rocblas_set_stream(RocblasHandle_* handle, RocmNativeStream* stream);
// alpha / beta: float (computeType rocblas_datatype_f32_r)
extern int rocblas_gemm_ex(RocblasHandle_* handle, int transA, int transB, int m, int n, int k,
                           const float* alpha, const HipDevPtr_* A, int aType, int lda,
                           const HipDevPtr_* B, int bType, int ldb, const float* beta,
                           const HipDevPtr_* C, int cType, int ldc, HipDevPtr_* D, int dType, int ldd,
                           int computeType, int algo, int solutionIndex, unsigned int flags);
extern int rocblas_gemm_ex_get_solutions(RocblasHandle_* handle, int transA, int transB, int m, int n, int k,
                                         const float* alpha, const HipDevPtr_* A, int aType, int lda,
                                         const HipDevPtr_* B, int bType, int ldb, const float* beta,
                                         const HipDevPtr_* C, int cType, int ldc, HipDevPtr_* D, int dType, int ldd,
                                         int computeType, int algo, unsigned int flags,
                                         int* list, int* size);
extern int rocblas_sgemm(RocblasHandle_* handle, int transA, int transB, int m, int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* B, int ldb,
                         const float* beta, float* C, int ldc);
extern int rocblas_sgemm_strided_batched(RocblasHandle_* handle, int transA, int transB, int m, int n, int k,
                                         const float* alpha, const float* a, int lda, long long sa,
                                         const float* b, int ldb, long long sb, const float* beta,
                                         float* c, int ldc, long long sc, int batch);
extern int rocblas_gemm_strided_batched_ex(RocblasHandle_* handle, int transA, int transB, int m, int n, int k,
                                           const float* alpha, const HipDevPtr_* a, int aType, int lda, long long sa,
                                           const HipDevPtr_* b, int bType, int ldb, long long sb, const float* beta,
                                           const HipDevPtr_* c, int cType, int ldc, long long sc,
                                           HipDevPtr_* d, int dType, int ldd, long long sd, int batch,
                                           int computeType, int algo, int solution, unsigned int flags);

} // namespace std
