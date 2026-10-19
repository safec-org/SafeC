#pragma once
// SafeC Standard Library — AMD GPU backend (ROCm / HIP).
//
// Two layers:
//
//  1. Device-resident buffers (RocmBuf) and queued device ops — the fast
//     path. Upload once, run any number of ops on the GPU, read back only
//     what you need. Matmuls go to rocBLAS; everything else runs custom
//     HIP kernels (elementwise math, ReLU and its gradient, in-place SGD,
//     sum reduction, and a fused MSE-loss gradient) that this library
//     compiles at first use with hiprtc for whatever GPU is present — no
//     offline build step and no per-architecture binaries.
//  2. Host-array convenience ops (rocm_*_f32) — upload, compute, read back
//     in one call. Simple, but each call pays two PCIe transfers; use (1)
//     inside loops.
//
// Ops on layer (1) are queued in order on one HIP stream and return as
// soon as they're enqueued; rocm_read() and rocm_sync() wait. A sequence
// of queued ops can be recorded once into a RocmGraph and replayed with a
// single launch, which removes per-kernel launch overhead from small,
// repeated workloads such as a training step.
//
// The backend picks the GPU with the most compute units (discrete over
// integrated), so on a machine with both an Instinct/Radeon card and an
// APU it uses the card. Link with: -lamdhip64 -lhiprtc -lrocblas
// (from /opt/rocm/lib).
namespace std {

// ── Device and buffers ───────────────────────────────────────────────────────

// 1 if a HIP GPU is usable (initializes the backend on first call).
int rocm_available();

// Name of the selected GPU ("AMD Instinct MI100"), or "" if none.
const char* rocm_device_name();

// A float32 buffer in GPU memory. 'dev' is the device address — never
// dereferenced on the host — and 0 for an invalid (failed or freed)
// buffer; 'len' is the element count every op checks against.
struct RocmBuf {
    unsigned long dev;
    unsigned long len;
};

struct RocmBuf rocm_alloc(unsigned long n);                       // uninitialized; dev == 0 on failure
struct RocmBuf rocm_upload(const &float host, unsigned long n);  // dev == 0 on failure
int  rocm_write(struct RocmBuf dst, const &float host, unsigned long n);  // 1 on success
int  rocm_read(struct RocmBuf src, &float host, unsigned long n);         // waits for queued work
void rocm_free(&RocmBuf b);                                       // b.dev := 0
int  rocm_sync();                                                 // wait for all queued work

// ── Queued device ops (1 = enqueued, 0 = invalid buffer/size or no GPU) ─────
// Matrices are row-major. 'ta'/'tb' = 1 means that operand is stored
// transposed (A as k x m, B as n x k).

// c[m,n] = op(a)[m,k] * op(b)[k,n]  (rocBLAS sgemm)
int rocm_gemm(struct RocmBuf a, int ta, struct RocmBuf b, int tb, struct RocmBuf c,
              unsigned long m, unsigned long n, unsigned long k);

int rocm_add(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);   // out = a + b
int rocm_sub(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_mul(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_div(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_pow(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_scale(struct RocmBuf a, float k, struct RocmBuf out);         // out = a * k
int rocm_relu(struct RocmBuf a, struct RocmBuf out);
int rocm_exp(struct RocmBuf a, struct RocmBuf out);
int rocm_log(struct RocmBuf a, struct RocmBuf out);
int rocm_sqrt(struct RocmBuf a, struct RocmBuf out);
int rocm_relu_grad(struct RocmBuf z, struct RocmBuf g, struct RocmBuf out);   // out = z > 0 ? g : 0
int rocm_sgd(struct RocmBuf w, struct RocmBuf g, float lr);            // w -= lr * g, in place
int rocm_sum(struct RocmBuf a, struct RocmBuf out1);                   // out1[0] = sum(a)
// Fused MSE gradient: dy = 2 (y - t) and loss1[0] = sum((y - t)^2), in one pass.
int rocm_mse_grad(struct RocmBuf y, struct RocmBuf t, struct RocmBuf dy, struct RocmBuf loss1);

// ── Graphs: record queued ops once, replay with one launch ───────────────────

struct RocmGraph {
    unsigned long exec;   // 0 = invalid
};

// Ops issued between begin and end are recorded instead of run. Don't call
// rocm_read/rocm_write/rocm_sync while recording.
int  rocm_capture_begin();
struct RocmGraph rocm_capture_end();          // exec == 0 on failure
int  rocm_graph_launch(struct RocmGraph g);   // queued like any other op
void rocm_graph_free(&RocmGraph g);

// ── Host-array convenience ops (upload → compute → read back) ────────────────
// 1 on success, 0 on failure (no GPU, allocation failure).

int rocm_add_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_sub_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_mul_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_div_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_pow_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_relu_f32(const &float a, &float out, unsigned long n);
int rocm_log_f32(const &float a, &float out, unsigned long n);
int rocm_exp_f32(const &float a, &float out, unsigned long n);
int rocm_sqrt_f32(const &float a, &float out, unsigned long n);
int rocm_scale_f32(const &float a, float k, &float out, unsigned long n);
int rocm_sum_f32(const &float a, &float out1, unsigned long n);

// out[M,N] = a[M,K] * b[K,N] with this library's own tiled HIP kernel
// (64x64 tiles in LDS, 4x4 per thread). Kept as a dependency-free
// reference; rocm_matmul_f32_blas is several times faster at every size.
int rocm_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N);
// Same product through rocBLAS.
int rocm_matmul_f32_blas(const &float a, const &float b, &float out,
                         unsigned long M, unsigned long K, unsigned long N);

} // namespace std
