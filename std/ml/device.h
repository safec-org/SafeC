#pragma once
// SafeC Standard Library — devices: where a tensor's values live.
//
// Every tensor has a device: DEVICE_CPU, or a GPU backend (DEVICE_ROCM for
// AMD, DEVICE_CUDA for NVIDIA, DEVICE_MPS for Apple Silicon). An op runs on
// the device of its operands -- if any operand lives on a GPU, the op runs
// there, uploading the others -- and its result lives there too, so a model
// whose parameters are on a GPU trains on that GPU through the same code
// that trains it on the CPU. The backward pass runs each node where its
// forward pass ran. An op without a kernel on the device computes on the
// CPU (the operands are copied back first), so every op works on every
// device; the GPU only changes the speed.
//
// Choosing a device:
//   explicitly  t.to(DEVICE_ROCM) (a copy, a graph node), p.place(DEVICE_ROCM)
//               (moves a parameter in place), model_to(&net, DEVICE_ROCM),
//               TrainConfig.device;
//   by default  device_set_default(d) for everything that asks for
//               device_default();
//   automatically device_auto(): the first usable GPU among the backends the
//               program was built with (ROCm, CUDA, Metal), else the CPU.
//               SAFEC_DEVICE=cpu|rocm|cuda|mps in the environment overrides it.
//
// Backends are enabled at build time by the program's features (safeguard
// build --features rocm / cuda / mps, reaching the sources as
// SAFEC_FEATURE_ROCM, SAFEC_FEATURE_CUDA, SAFEC_FEATURE_MPS); a program built
// without any runs everything on the CPU and links no GPU library. Link:
//   rocm  -lamdhip64 -lhiprtc -lrocblas (-lMIOpen for MIOpen convolutions)
//   cuda  -lcuda -lcublas -lnvrtc (-lcudnn for cuDNN convolutions)
//   mps   -framework Metal -framework Foundation -lobjc
//
// Each backend is a struct of static functions (the interface below),
// resolved at compile time; the runtime dispatch in device.sc calls them once
// the program's tensors say which device an op runs on.
#include <std/stddef.h>

namespace std {

#define DEVICE_CPU   0
#define DEVICE_ROCM  1
#define DEVICE_CUDA  2
#define DEVICE_MPS   3
#define DEVICE_COUNT 4

// 1 if the program was built with the backend (always for DEVICE_CPU).
int device_compiled(int device);
// 1 if the backend is compiled in and a GPU is present (initializes it).
int device_available(int device);
// The first available GPU (ROCm, then CUDA, then Metal), else DEVICE_CPU;
// SAFEC_DEVICE in the environment picks one explicitly.
int device_auto();
// What device_default() returns: set with device_set_default (a device that
// isn't available falls back to device_auto()), DEVICE_AUTO (-1) to go back
// to automatic selection.
#define DEVICE_AUTO (-1)
void device_set_default(int device);
int  device_default();
// "cpu", "rocm", "cuda", "mps".
const &char device_name(int device);
// Waits for the device's queued work (no-op on the CPU). 1 on success.
int device_sync(int device);
// Device memory the tensor cache holds (in use + cached free), in bytes.
unsigned long device_bytes_reserved(int device);
// Returns the device's cached free blocks to the driver.
void device_cache_clear(int device);
// Counters since the program started (or the last device_stats_reset): what
// moved between host and device, and how many kernels and GEMMs ran. A
// training step that keeps everything on the device shows no downloads
// besides the loss it reads, and no uploads besides its batch.
// cpuLaunches counts the kernels and GEMMs a backend ran on the CPU
// (device_ref.sc) for want of a GPU version: 0 when the GPU ran them all.
struct DeviceStats {
    unsigned long launches;
    unsigned long gemms;
    unsigned long uploads;
    unsigned long downloads;
    unsigned long uploadBytes;
    unsigned long downloadBytes;
    unsigned long cpuLaunches;
};
struct DeviceStats device_stats(int device);
void device_stats_reset(int device);

// The device traits (DeviceTensor, RocmTensor, CudaTensor, MpsTensor) are in
// tensor.h, next to trait Tensor they inherit.

// ── Backends ─────────────────────────────────────────────────────────────────
// A device kernel launch: the kernel (DK_* in device_kernels.h), how many
// threads (one per element unless the kernel says otherwise), and its
// arguments: device buffers, integers and floats, passed to the kernel as
// one struct (the same layout on every backend).
#define DEVK_PTRS   8
#define DEVK_INTS   8
#define DEVK_FLOATS 8
struct DevArgs {
    unsigned long p[DEVK_PTRS];     // device addresses (byte offsets already applied)
    unsigned int  u[DEVK_INTS];
    float         f[DEVK_FLOATS];
};

// Precisions (device_set_precision): what convolutions and matrix products
// read on a device. 16-bit modes keep values, weights, gradients and the
// optimizer state in fp32, feed these ops 16-bit copies and accumulate in
// fp32; fp16 also scales the loss (device_loss_scale).
#define DEVICE_FP32 0
#define DEVICE_FP16 1
#define DEVICE_BF16 2
#define DEVICE_PRECISION_KEEP (-1)   // TrainConfig.precision: the device's own setting

// 16-bit storage: on a backend that takes it (storage16), activations of a
// 16-bit precision live on the device in fp16 / bf16. A buffer's element
// type rides in the top two bits of its handle (DEV_TAG_*); every kernel,
// copy, GEMM and convolution reads it from there.
#define DEV_TAG_SHIFT (sizeof(unsigned long) * 8UL - 2UL)
#define DEV_TAG_MASK  (3UL << DEV_TAG_SHIFT)
static unsigned long dev_tag_(unsigned long h) { return (h & DEV_TAG_MASK) >> DEV_TAG_SHIFT; }   // DEVICE_FP32 / FP16 / BF16
static unsigned long dev_addr_(unsigned long h) { return h & ~DEV_TAG_MASK; }
static unsigned long dev_tagged_(unsigned long h, unsigned long tag) { return h == 0UL ? 0UL : dev_addr_(h) | (tag << DEV_TAG_SHIFT); }
static unsigned long dev_esize_(unsigned long tag) { return tag == 0UL ? 4UL : 2UL; }
// Bytes for n elements of 'tag' (whole floats: copies move 4-byte words).
static unsigned long dev_bytes_(unsigned long tag, unsigned long n) { return tag == 0UL ? 4UL * n : 4UL * ((n + 1UL) / 2UL); }

// A strided batch of GEMMs: for i < batch, c_i = alpha op(a_i) op(b_i) + beta c_i
// with x_i = x + i * sx (strides in elements), as gemm(). dtype (DEVICE_FP32 /
// FP16 / BF16) is a and b's element type; c is fp32 and accumulates in fp32.
struct DevGemm {
    int dtype;              // a and b's element type
    int ctype;              // c's (DEVICE_FP32 / FP16 / BF16)
    int ta; int tb;
    unsigned long m; unsigned long n; unsigned long k;
    float alpha;
    unsigned long a; unsigned long lda; unsigned long sa;
    unsigned long b; unsigned long ldb; unsigned long sb;
    float beta;
    unsigned long c; unsigned long ldc; unsigned long sc;
    unsigned long batch;
};

// A 2D convolution (transpose 0) or transposed convolution (1) in NCHW,
// in the layer's terms: x the input [batch, inC, inH, inW], y the output
// [batch, outC, outH, outW], w [outC, inC, kH, kW] (a transposed one's
// [inC, outC, kH, kW]). op 0: y = conv(x, w); 1: x = the input gradient from
// y's; 2: w = the weight gradient from y's and x. The result is overwritten.
#define DEV_CONV_FWD  0
#define DEV_CONV_BWD_DATA 1
#define DEV_CONV_BWD_WEIGHTS 2
struct DevConv {
    int op;
    int dtype;              // DEVICE_FP32 / FP16 / BF16: x, w and y's element type
    int transpose;
    unsigned long batch;
    unsigned long inC; unsigned long inH; unsigned long inW;
    unsigned long outC; unsigned long outH; unsigned long outW;
    unsigned long kH; unsigned long kW; unsigned long sH; unsigned long sW;
    unsigned long pH; unsigned long pW; unsigned long dH; unsigned long dW;
    unsigned long x; unsigned long w; unsigned long y;
};

// What a GPU backend implements: a struct (device_rocm.sc's RocmBackend,
// device_cuda.sc's CudaBackend, device_mps.sc's MpsBackend) with these static
// functions, reached through the type parameter of
// device_register_backend_<B> (traits name instance methods only, so this
// static interface is spelled out here). Device memory is named by unsigned
// long handles: device addresses on ROCm / CUDA, an MTLBuffer on Metal. Each
// call is queued on the backend's stream (ordered with gpu_rocm.h /
// gpu_cuda.h / gpu_mps.h work) unless it says it waits.
//   static int  available();
//   static unsigned long alloc(unsigned long bytes);          0 on failure
//   static void release(unsigned long handle);
//   static int  upload(unsigned long dst, const &float src, unsigned long n);
//   static int  download(&float dst, unsigned long src, unsigned long n);   waits
//   static int  copy(unsigned long dst, unsigned long src, unsigned long n);
//   static int  sync();
//   // c[m,n] = alpha op(a) op(b) + beta c, row-major with leading dimensions
//   static int  gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
//                    unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
//                    float beta, unsigned long c, unsigned long ldc);
//   static int  gemm_batched(const &DevGemm g);    0 for a dtype it lacks
//   // 1 when its kernels take 16-bit buffers (tagged handles)
//   static int  storage16();
//   // a convolution through the platform's library (MIOpen, cuDNN); 0 when
//   // it has none or can't run this one (the op then runs im2col + GEMM)
//   static int  conv(const &DevConv c);
//   // one DK_* kernel over n threads; 0 when the backend has no such kernel
//   // (the op then computes on the CPU)
//   static int  launch(int kernel, unsigned long n, const &DevArgs args);
//   // handle + byte offset (0 where sub-buffers aren't addressable)
//   static unsigned long offset(unsigned long handle, unsigned long bytes);

// The dispatch table device.sc calls through (internal): one entry per GPU
// device, filled in by the backend's register function when the program
// links it.
typedef fn int() DevAvailFn;
typedef fn unsigned long(unsigned long) DevAllocFn;
typedef fn void(unsigned long) DevReleaseFn;
typedef fn int(unsigned long, const &float, unsigned long) DevUploadFn;
typedef fn int(&float, unsigned long, unsigned long) DevDownloadFn;
typedef fn int(unsigned long, unsigned long, unsigned long) DevCopyFn;
typedef fn int() DevSyncFn;
typedef fn int(int, int, unsigned long, unsigned long, unsigned long, float, unsigned long, unsigned long,
               unsigned long, unsigned long, float, unsigned long, unsigned long) DevGemmFn;
typedef fn int(const &DevGemm) DevGemmBatchedFn;
typedef fn int(const &DevConv) DevConvFn;
typedef fn int() DevStorageFn;
typedef fn int(int, unsigned long, const &DevArgs) DevLaunchFn;
typedef fn unsigned long(unsigned long, unsigned long) DevOffsetFn;

struct DeviceOps {
    DevAvailFn    available;
    DevAllocFn    alloc;
    DevReleaseFn  release;
    DevUploadFn   upload;
    DevDownloadFn download;
    DevCopyFn     copy;
    DevSyncFn     sync;
    DevGemmFn     gemm;
    DevGemmBatchedFn gemm_batched;
    DevConvFn     conv;
    DevStorageFn  storage16;
    DevLaunchFn   launch;
    DevOffsetFn   offset;
    int           registered;
};

// Fills the table entry for 'device' from a backend B (the static interface above).
void device_register_ops_(int device, struct DeviceOps ops);

// 'tag' only names the backend type (B is inferred from it).
generic<B>
void device_register_backend_(int device, const &B tag) {
    struct DeviceOps ops;
    ops.available = B::available;
    ops.alloc = B::alloc;
    ops.release = B::release;
    ops.upload = B::upload;
    ops.download = B::download;
    ops.copy = B::copy;
    ops.sync = B::sync;
    ops.gemm = B::gemm;
    ops.gemm_batched = B::gemm_batched;
    ops.conv = B::conv;
    ops.storage16 = B::storage16;
    ops.launch = B::launch;
    ops.offset = B::offset;
    ops.registered = 1;
    device_register_ops_(device, ops);
}

// Each backend's registration (device_rocm.sc / device_cuda.sc /
// device_mps.sc).
void device_rocm_register_();
void device_cuda_register_();
void device_mps_register_();

// Called by device.sc on first use; the program's own translation units
// define it (weakly, once per unit -- they are identical) for the backends
// their features enable, which is what links a backend into a program.
void device_link_backends_() __attribute__((weak));
#if defined(SAFEC_FEATURE_ROCM) || defined(SAFEC_FEATURE_CUDA) || defined(SAFEC_FEATURE_MPS)
void device_link_backends_() __attribute__((weak)) {
#if defined(SAFEC_FEATURE_ROCM)
    device_rocm_register_();
#endif
#if defined(SAFEC_FEATURE_CUDA)
    device_cuda_register_();
#endif
#if defined(SAFEC_FEATURE_MPS)
    device_mps_register_();
#endif
}
#endif

// Sets what convolutions and matrix products on 'device' compute in (see
// DEVICE_FP32 above); 1 on success. The previous setting comes back from
// device_precision.
int device_set_precision(int device, int precision);
int device_precision(int device);
// How a 16-bit precision picks per shape: DEVICE_PICK_TIMED (the default)
// times each new convolution / matmul shape in 16 bits against fp32 and keeps
// the faster; DEVICE_PICK_ALWAYS runs every one it can in 16 bits (tests,
// or to measure). Returns the previous mode.
#define DEVICE_PICK_TIMED  0
#define DEVICE_PICK_ALWAYS 1
int device_set_precision_pick(int mode);
int device_precision_pick();
// The fp16 loss scale on 'device' (waits): gradients are scaled by it during
// backward and unscaled when backward ends, a step that overflowed is skipped
// by the optimizer, and the scale halves on overflow and doubles after 2000
// clean steps. 1 when the device isn't in fp16.
float device_loss_scale(int device);

// ── Internal: memory and launches for the tensor ops ─────────────────────────
// Device memory from the per-device cache (size classes, reused across
// steps); 0 on failure.
unsigned long device_mem_alloc_(int device, unsigned long bytes);
void device_mem_free_(int device, unsigned long handle, unsigned long bytes);
int device_upload_(int device, unsigned long dst, const &float src, unsigned long n);
int device_download_(int device, &float dst, unsigned long src, unsigned long n);
int device_copy_(int device, unsigned long dst, unsigned long src, unsigned long n);
int device_gemm_(int device, int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                 unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                 float beta, unsigned long c, unsigned long ldc);
int device_gemm_batched_(int device, const &DevGemm g);
// c (+)= op(a) op(b) with a, b 'dtype' (16-bit) values, c fp32; 0 when the backend lacks it.
int device_gemm16_(int device, int dtype, int ta, int tb, unsigned long m, unsigned long n, unsigned long k,
                   unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb, float beta,
                   unsigned long c, unsigned long ldc);
// The loss scale state [scale, 1/scale, overflow, good steps] on an fp16
// device (allocated on first use), else 0.
unsigned long device_amp_state_(int device);
// The element type activations are stored in on 'device': its precision when
// the backend takes 16-bit buffers, else DEVICE_FP32.
int device_store_tag_(int device);
// handle + n elements of its type
unsigned long device_offset_el_(int device, unsigned long handle, unsigned long n);
// 16-bit copies of parameters: a buffer marked cacheable keeps its 16-bit
// copy across GEMMs and convolutions until it is dropped (written, uploaded
// to or freed: the tensor layer drops it).
void device_shadow_mark_(int device, unsigned long handle);
void device_shadow_drop_(int device, unsigned long handle);
int device_conv_(int device, const &DevConv c);
int device_launch_(int device, int kernel, unsigned long n, const &DevArgs args);
unsigned long device_offset_(int device, unsigned long handle, unsigned long bytes);
// A zeroed argument struct.
struct DevArgs dev_args_();
// A backend ran a kernel or GEMM on the CPU (DeviceStats.cpuLaunches).
void device_note_cpu_launch_(int device);
// The kernels on the CPU over host addresses (device_ref.sc): what a backend
// runs for a kernel it has no GPU version of, and the reference the GPU
// kernels are tested against.
int device_ref_launch_(int kernel, unsigned long n, const &DevArgs args);
int device_ref_gemm_(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                     unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                     float beta, unsigned long c, unsigned long ldc);
// A batch as a loop of gemm(): for backends without a batched GEMM.
int device_loop_gemm_batched_(int device, const &DevGemm g);
int device_ref_gemm_batched_(const &DevGemm g);

} // namespace std
