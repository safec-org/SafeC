// SafeC Standard Library — flash and linear attention on NVIDIA GPUs (see
// gpu_cuda_attention.h).
//
// The kernels are CUDA C++ below, compiled per variant through gpu_cuda's
// cuda_load_module (NVRTC, -arch=sm_<cc>) the first time a (kernel, dtype,
// head dim) combination is used: VK selects the kernel, VT the element type
// (0 fp32, 1 fp16, 2 bf16) and VD the head dim, and every variant exports
// one entry point, attn_entry. All launches go on cuda_stream_handle().
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_cuda_attention.h>
#include <std/ml/gpu_cuda.h>
#include <std/mem.h>

namespace std {

// ── CUDA driver C ABI (cuda.h) ───────────────────────────────────────────────
extern int cuModuleGetFunction(void** func, void* module, const char* name);
extern int cuFuncSetAttribute(void* func, int attr, int value);
extern int cuLaunchKernel(void* f, unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz,
                          unsigned int shared, void* stream, void** params, void** extra);
extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemsetD32Async(unsigned long long dst, unsigned int value, size_t n, void* stream);
extern int cuStreamSynchronize(void* stream);
extern size_t strlen(const char* s);
extern void* memcpy(void* dst, const void* src, size_t n);

#define ATT_FN_MAX_DYN_SMEM  8    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define ATT_SIMT             1    // flash attention, CUDA cores, any dtype
#define ATT_MMA              2    // flash attention, tensor cores, fp16/bf16, sm_80+
#define ATT_LIN_KV           3    // linear attention pass 1: S = phi(K)^T V, z
#define ATT_LIN_OUT          4    // linear attention pass 2: out = phi(Q) S / phi(Q) z
#define ATT_LIN_CAUSAL       5    // linear attention, causal chunked scan
#define ATT_KINDS            5
#define ATT_DIMS             5
#define ATT_VARIANTS         75   // ATT_KINDS * 3 dtypes * ATT_DIMS

// ── Kernel source (gpu_cuda_attention.cu) ────────────────────────────────────
static const char* attn_kernel_source_() {
    return
    "// SafeC std/ml/gpu_cuda_attention kernels. Compiled per variant by NVRTC\n"
    "// with VK (kernel kind), VT (0 fp32, 1 fp16, 2 bf16) and VD (head dim)\n"
    "// defined in front of this source; each variant exports attn_entry.\n"
    "//\n"
    "// Layout: Q/O [batch, heads, seqLen, D], K/V [batch, kvHeads, kvLen, D],\n"
    "// row-major. Query head h reads kv head h / (heads / kvHeads). Causal mask\n"
    "// is aligned bottom-right: query i sees keys j <= i + (kvLen - seqLen).\n"
    "\n"
    "#define FULL 0xffffffffu\n"
    "#define NEG_INF (-__int_as_float(0x7f800000))\n"
    "\n"
    "__device__ __forceinline__ float h2f(unsigned short h) { float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f; }\n"
    "__device__ __forceinline__ unsigned short f2h(float f) { unsigned short h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f)); return h; }\n"
    "__device__ __forceinline__ float b2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }\n"
    "__device__ __forceinline__ unsigned short f2b(float f) {   // round to nearest even; NaN stays NaN\n"
    "    unsigned u = __float_as_uint(f);\n"
    "    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);\n"
    "    u += 0x7fffu + ((u >> 16) & 1u);\n"
    "    return (unsigned short)(u >> 16);\n"
    "}\n"
    "template <int T> __device__ __forceinline__ float ldv(const void* p, size_t i) {\n"
    "    if (T == 0) return ((const float*)p)[i];\n"
    "    unsigned short u = ((const unsigned short*)p)[i];\n"
    "    return T == 1 ? h2f(u) : b2f(u);\n"
    "}\n"
    "template <int T> __device__ __forceinline__ void stv(void* p, size_t i, float x) {\n"
    "    if (T == 0) ((float*)p)[i] = x;\n"
    "    else ((unsigned short*)p)[i] = T == 1 ? f2h(x) : f2b(x);\n"
    "}\n"
    "// elu(x) + 1, the linear-attention feature map\n"
    "__device__ __forceinline__ float phi(float x) { return x > 0.0f ? x + 1.0f : __expf(x); }\n"
    "\n"
    "// ── Flash attention, SIMT (any dtype, D up to 256) ──────────────────────────\n"
    "// 32 query rows per 128-thread block, 8 per warp. Per 32-key tile each lane\n"
    "// scores one key against the warp's 8 rows, the warp reduces max/sum with\n"
    "// shuffles (online softmax, base 2), and then each lane accumulates D/32\n"
    "// output columns of P V. Q is pre-scaled by scale * log2(e).\n"
    "template <int T, int D>\n"
    "__device__ void flash_simt(const void* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,\n"
    "                           void* __restrict__ O, int Sq, int Sk, int H, int Hkv, float sl2, int causal) {\n"
    "    constexpr int BM = 32, BN = 32, RW = 8, DP = D + 1, DL = (D + 31) / 32;\n"
    "    extern __shared__ float sm[];\n"
    "    float* Qs = sm;               // BM x D\n"
    "    float* Ks = Qs + BM * D;      // BN x DP\n"
    "    float* Vs = Ks + BN * DP;     // BN x D\n"
    "    float* Ps = Vs + BN * D;      // 4 warps x RW x BN\n"
    "    const int bh = blockIdx.y, b = bh / H, h = bh % H, hk = h / (H / Hkv);\n"
    "    const size_t qoff = (size_t)bh * Sq * D, koff = (size_t)(b * Hkv + hk) * Sk * D;\n"
    "    const int m0 = blockIdx.x * BM, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n"
    "    const int off = Sk - Sq;\n"
    "    for (int i = tid; i < BM * D; i += 128) {\n"
    "        int r = i / D, d = i - r * D, gr = m0 + r;\n"
    "        Qs[i] = gr < Sq ? ldv<T>(Q, qoff + (size_t)gr * D + d) * sl2 : 0.0f;\n"
    "    }\n"
    "    float o[RW][DL], mr[RW], lr[RW];\n"
    "#pragma unroll\n"
    "    for (int r = 0; r < RW; ++r) {\n"
    "        mr[r] = NEG_INF; lr[r] = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < DL; ++t) o[r][t] = 0.0f;\n"
    "    }\n"
    "    int kend = Sk;\n"
    "    if (causal) {\n"
    "        int last = min(m0 + BM, Sq) - 1;\n"
    "        kend = max(0, min(Sk, last + off + 1));\n"
    "    }\n"
    "    for (int n0 = 0; n0 < kend; n0 += BN) {\n"
    "        __syncthreads();\n"
    "        for (int i = tid; i < BN * D; i += 128) {\n"
    "            int r = i / D, d = i - r * D, gk = n0 + r;\n"
    "            bool ok = gk < Sk;\n"
    "            Ks[r * DP + d] = ok ? ldv<T>(K, koff + (size_t)gk * D + d) : 0.0f;\n"
    "            Vs[r * D + d] = ok ? ldv<T>(V, koff + (size_t)gk * D + d) : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        const int j = n0 + lane;\n"
    "        const float* kr = Ks + lane * DP;\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < RW; ++r) {\n"
    "            const int lrow = warp * RW + r, gq = m0 + lrow;\n"
    "            const float* qr = Qs + lrow * D;\n"
    "            float s = 0.0f;\n"
    "#pragma unroll 8\n"
    "            for (int d = 0; d < D; ++d) s += qr[d] * kr[d];\n"
    "            bool valid = j < Sk && (!causal || j <= gq + off);\n"
    "            s = valid ? s : NEG_INF;\n"
    "            float mx = s;\n"
    "#pragma unroll\n"
    "            for (int x = 16; x > 0; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, x));\n"
    "            float mn = fmaxf(mr[r], mx);\n"
    "            float mu = mn == NEG_INF ? 0.0f : mn;\n"
    "            float p = exp2f(s - mu);\n"
    "            float alpha = exp2f(mr[r] - mu);\n"
    "            float ps = p;\n"
    "#pragma unroll\n"
    "            for (int x = 16; x > 0; x >>= 1) ps += __shfl_xor_sync(FULL, ps, x);\n"
    "            lr[r] = lr[r] * alpha + ps;\n"
    "            mr[r] = mn;\n"
    "            Ps[lrow * BN + lane] = p;\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < DL; ++t) o[r][t] *= alpha;\n"
    "        }\n"
    "        __syncwarp();\n"
    "        for (int jj = 0; jj < BN; ++jj) {\n"
    "            float vv[DL];\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < DL; ++t) { int d = lane + 32 * t; vv[t] = d < D ? Vs[jj * D + d] : 0.0f; }\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < RW; ++r) {\n"
    "                float p = Ps[(warp * RW + r) * BN + jj];\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < DL; ++t) o[r][t] += p * vv[t];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int r = 0; r < RW; ++r) {\n"
    "        int gq = m0 + warp * RW + r;\n"
    "        if (gq >= Sq) continue;\n"
    "        float inv = lr[r] > 0.0f ? 1.0f / lr[r] : 0.0f;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < DL; ++t) {\n"
    "            int d = lane + 32 * t;\n"
    "            if (d < D) stv<T>(O, qoff + (size_t)gq * D + d, o[r][t] * inv);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// ── Flash attention, tensor cores (fp16 / bf16, sm_80+) ─────────────────────\n"
    "// FlashAttention-2 forward. 64 query rows per 128-thread block, 16 per warp;\n"
    "// 64-key tiles of K (row-major) and V (transposed) in shared memory. Each\n"
    "// warp keeps its Q rows as mma A-fragments, computes S = Q K^T with\n"
    "// mma.sync.m16n8k16 (fp32 accumulate), runs the online softmax on the\n"
    "// accumulators, and reuses them as the A-fragments of P for O += P V.\n"
    "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800\n"
    "template <int BF>\n"
    "__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {\n"
    "    if (BF)\n"
    "        asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\"\n"
    "                     : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n"
    "                     : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1));\n"
    "    else\n"
    "        asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\"\n"
    "                     : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n"
    "                     : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1));\n"
    "}\n"
    "// Packs two floats into a 16-bit pair, 'lo' in the low half (lower address).\n"
    "template <int BF> __device__ __forceinline__ unsigned pack2(float lo, float hi) {\n"
    "    unsigned r;\n"
    "    if (BF) asm(\"cvt.rn.bf16x2.f32 %0, %1, %2;\" : \"=r\"(r) : \"f\"(hi), \"f\"(lo));\n"
    "    else asm(\"cvt.rn.f16x2.f32 %0, %1, %2;\" : \"=r\"(r) : \"f\"(hi), \"f\"(lo));\n"
    "    return r;\n"
    "}\n"
    "\n"
    "template <int BF, int D>\n"
    "__device__ void flash_mma(const unsigned short* __restrict__ Q, const unsigned short* __restrict__ K,\n"
    "                          const unsigned short* __restrict__ V, unsigned short* __restrict__ O,\n"
    "                          int Sq, int Sk, int H, int Hkv, float sl2, int causal) {\n"
    "    constexpr int BM = 64, BN = 64, KP = D + 8, VP = BN + 8, KS = D / 16, NB = BN / 8, ND = D / 8;\n"
    "    __shared__ __align__(16) unsigned short Ks[BN * KP];\n"
    "    __shared__ __align__(16) unsigned short Vt[D * VP];\n"
    "    const int bh = blockIdx.y, b = bh / H, h = bh % H, hk = h / (H / Hkv);\n"
    "    const size_t qoff = (size_t)bh * Sq * D, koff = (size_t)(b * Hkv + hk) * Sk * D;\n"
    "    const int m0 = blockIdx.x * BM, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n"
    "    const int g = lane >> 2, c = lane & 3, off = Sk - Sq;\n"
    "    const int r0 = m0 + warp * 16 + g, r1 = r0 + 8;   // this thread's two query rows\n"
    "\n"
    "    unsigned qa[KS][4];\n"
    "#pragma unroll\n"
    "    for (int kk = 0; kk < KS; ++kk) {\n"
    "        int col = kk * 16 + c * 2;\n"
    "        qa[kk][0] = r0 < Sq ? *(const unsigned*)(Q + qoff + (size_t)r0 * D + col) : 0u;\n"
    "        qa[kk][1] = r1 < Sq ? *(const unsigned*)(Q + qoff + (size_t)r1 * D + col) : 0u;\n"
    "        qa[kk][2] = r0 < Sq ? *(const unsigned*)(Q + qoff + (size_t)r0 * D + col + 8) : 0u;\n"
    "        qa[kk][3] = r1 < Sq ? *(const unsigned*)(Q + qoff + (size_t)r1 * D + col + 8) : 0u;\n"
    "    }\n"
    "    float o[ND][4];\n"
    "#pragma unroll\n"
    "    for (int n = 0; n < ND; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;\n"
    "    float m_0 = NEG_INF, m_1 = NEG_INF, l_0 = 0.0f, l_1 = 0.0f;\n"
    "\n"
    "    int kend = Sk;\n"
    "    if (causal) {\n"
    "        int last = min(m0 + BM, Sq) - 1;\n"
    "        kend = max(0, min(Sk, last + off + 1));\n"
    "    }\n"
    "    for (int n0 = 0; n0 < kend; n0 += BN) {\n"
    "        __syncthreads();\n"
    "        // K tile: 16-byte vectors; V tile transposed into Vt[d][key]\n"
    "        for (int i = tid; i < BN * (D / 8); i += 128) {\n"
    "            int r = i / (D / 8), d8 = (i - r * (D / 8)) * 8, gk = n0 + r;\n"
    "            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);\n"
    "            if (gk < Sk) {\n"
    "                kv = *(const uint4*)(K + koff + (size_t)gk * D + d8);\n"
    "                vv = *(const uint4*)(V + koff + (size_t)gk * D + d8);\n"
    "            }\n"
    "            *(uint4*)(Ks + r * KP + d8) = kv;\n"
    "            const unsigned short* vs = (const unsigned short*)&vv;\n"
    "#pragma unroll\n"
    "            for (int e = 0; e < 8; ++e) Vt[(d8 + e) * VP + r] = vs[e];\n"
    "        }\n"
    "        __syncthreads();\n"
    "\n"
    "        float s[NB][4];\n"
    "#pragma unroll\n"
    "        for (int nb = 0; nb < NB; ++nb) {\n"
    "            s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f;\n"
    "            const unsigned short* kr = Ks + (nb * 8 + g) * KP + c * 2;\n"
    "#pragma unroll\n"
    "            for (int kk = 0; kk < KS; ++kk)\n"
    "                mma16816<BF>(s[nb], qa[kk], *(const unsigned*)(kr + kk * 16), *(const unsigned*)(kr + kk * 16 + 8));\n"
    "        }\n"
    "        // scale, mask, row max over this thread's 2 x 16 values, then across the 4 lanes of a row\n"
    "        float mx0 = NEG_INF, mx1 = NEG_INF;\n"
    "#pragma unroll\n"
    "        for (int nb = 0; nb < NB; ++nb) {\n"
    "#pragma unroll\n"
    "            for (int e = 0; e < 4; ++e) {\n"
    "                int j = n0 + nb * 8 + c * 2 + (e & 1);\n"
    "                int qi = e < 2 ? r0 : r1;\n"
    "                bool ok = j < Sk && (!causal || j <= qi + off);\n"
    "                s[nb][e] = ok ? s[nb][e] * sl2 : NEG_INF;\n"
    "            }\n"
    "            mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));\n"
    "            mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));\n"
    "        }\n"
    "        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n"
    "        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n"
    "        float mn0 = fmaxf(m_0, mx0), mn1 = fmaxf(m_1, mx1);\n"
    "        float mu0 = mn0 == NEG_INF ? 0.0f : mn0, mu1 = mn1 == NEG_INF ? 0.0f : mn1;\n"
    "        float al0 = exp2f(m_0 - mu0), al1 = exp2f(m_1 - mu1);\n"
    "        m_0 = mn0; m_1 = mn1;\n"
    "        float ps0 = 0.0f, ps1 = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int nb = 0; nb < NB; ++nb) {\n"
    "            s[nb][0] = exp2f(s[nb][0] - mu0); s[nb][1] = exp2f(s[nb][1] - mu0);\n"
    "            s[nb][2] = exp2f(s[nb][2] - mu1); s[nb][3] = exp2f(s[nb][3] - mu1);\n"
    "            ps0 += s[nb][0] + s[nb][1];\n"
    "            ps1 += s[nb][2] + s[nb][3];\n"
    "        }\n"
    "        l_0 = l_0 * al0 + ps0;\n"
    "        l_1 = l_1 * al1 + ps1;\n"
    "#pragma unroll\n"
    "        for (int n = 0; n < ND; ++n) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }\n"
    "        // O += P V: accumulators of key blocks (2ks, 2ks+1) form P's A-fragment for k-step ks\n"
    "#pragma unroll\n"
    "        for (int ks = 0; ks < BN / 16; ++ks) {\n"
    "            unsigned pa[4];\n"
    "            pa[0] = pack2<BF>(s[2 * ks][0], s[2 * ks][1]);\n"
    "            pa[1] = pack2<BF>(s[2 * ks][2], s[2 * ks][3]);\n"
    "            pa[2] = pack2<BF>(s[2 * ks + 1][0], s[2 * ks + 1][1]);\n"
    "            pa[3] = pack2<BF>(s[2 * ks + 1][2], s[2 * ks + 1][3]);\n"
    "#pragma unroll\n"
    "            for (int n = 0; n < ND; ++n) {\n"
    "                const unsigned short* vr = Vt + (n * 8 + g) * VP + ks * 16 + c * 2;\n"
    "                mma16816<BF>(o[n], pa, *(const unsigned*)vr, *(const unsigned*)(vr + 8));\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    l_0 += __shfl_xor_sync(FULL, l_0, 1); l_0 += __shfl_xor_sync(FULL, l_0, 2);\n"
    "    l_1 += __shfl_xor_sync(FULL, l_1, 1); l_1 += __shfl_xor_sync(FULL, l_1, 2);\n"
    "    float i0 = l_0 > 0.0f ? 1.0f / l_0 : 0.0f, i1 = l_1 > 0.0f ? 1.0f / l_1 : 0.0f;\n"
    "#pragma unroll\n"
    "    for (int n = 0; n < ND; ++n) {\n"
    "        int col = n * 8 + c * 2;\n"
    "        if (r0 < Sq) *(unsigned*)(O + qoff + (size_t)r0 * D + col) = pack2<BF>(o[n][0] * i0, o[n][1] * i0);\n"
    "        if (r1 < Sq) *(unsigned*)(O + qoff + (size_t)r1 * D + col) = pack2<BF>(o[n][2] * i1, o[n][3] * i1);\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "// ── Linear attention, non-causal ────────────────────────────────────────────\n"
    "// Pass 1 (lin_kv): per kv head, S = phi(K)^T V ([D, D]) and z = sum phi(K)\n"
    "// ([D]) into fp32 workspace, each block adding one 256-key slice with\n"
    "// atomics (S/z zeroed first). Pass 2 (lin_out): out_i = phi(q_i) S / phi(q_i) . z.\n"
    "template <int T, int D>\n"
    "__device__ void lin_kv(const void* __restrict__ K, const void* __restrict__ V, float* __restrict__ S,\n"
    "                       float* __restrict__ Z, int Sk) {\n"
    "    // grid: (kv head, 256-key slice, 32-row slice of S); block 256\n"
    "    constexpr int CH = 32, RT = 32, PER = RT * D / 256, SLICE = 256;\n"
    "    extern __shared__ float sm[];\n"
    "    float* Kf = sm;               // CH x RT   phi(K) columns di0 .. di0+31\n"
    "    float* Vs = Kf + CH * RT;     // CH x D\n"
    "    const int kvh = blockIdx.x, di0 = blockIdx.z * RT, tid = threadIdx.x;\n"
    "    const size_t koff = (size_t)kvh * Sk * D;\n"
    "    const int j0 = blockIdx.y * SLICE, j1 = min(Sk, j0 + SLICE);\n"
    "    float acc[PER], zacc = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int e = 0; e < PER; ++e) acc[e] = 0.0f;\n"
    "    for (int c0 = j0; c0 < j1; c0 += CH) {\n"
    "        __syncthreads();\n"
    "        for (int i = tid; i < CH * RT; i += 256) {\n"
    "            int r = i / RT, cc = i - r * RT, gk = c0 + r;\n"
    "            Kf[i] = gk < j1 ? phi(ldv<T>(K, koff + (size_t)gk * D + di0 + cc)) : 0.0f;\n"
    "        }\n"
    "        for (int i = tid; i < CH * D; i += 256) {\n"
    "            int r = i / D, d = i - r * D, gk = c0 + r;\n"
    "            Vs[i] = gk < j1 ? ldv<T>(V, koff + (size_t)gk * D + d) : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll\n"
    "        for (int e = 0; e < PER; ++e) {\n"
    "            int idx = tid + 256 * e, di = idx / D, dj = idx - di * D;\n"
    "            float a = acc[e];\n"
    "#pragma unroll 8\n"
    "            for (int j = 0; j < CH; ++j) a += Kf[j * RT + di] * Vs[j * D + dj];\n"
    "            acc[e] = a;\n"
    "        }\n"
    "        if (tid < RT)\n"
    "            for (int j = 0; j < CH; ++j) zacc += Kf[j * RT + tid];\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int e = 0; e < PER; ++e) {\n"
    "        int idx = tid + 256 * e, di = idx / D, dj = idx - di * D;\n"
    "        atomicAdd(S + (size_t)kvh * D * D + (size_t)(di0 + di) * D + dj, acc[e]);\n"
    "    }\n"
    "    if (tid < RT) atomicAdd(Z + (size_t)kvh * D + di0 + tid, zacc);\n"
    "}\n"
    "\n"
    "template <int T, int D>\n"
    "__device__ void lin_out(const void* __restrict__ Q, const float* __restrict__ S, const float* __restrict__ Z,\n"
    "                        void* __restrict__ O, int Sq, int H, int Hkv) {\n"
    "    // grid: (32-row block, batch*heads, 64-column slice of the output); block 256\n"
    "    constexpr int RB = 32, CS = D % 64 == 0 ? 64 : 32, DL = CS / 32;\n"
    "    extern __shared__ float sm[];\n"
    "    float* Ss = sm;               // D x CS\n"
    "    float* Zs = Ss + D * CS;      // D\n"
    "    float* Qf = Zs + D;           // RB x D\n"
    "    const int bh = blockIdx.y, b = bh / H, h = bh % H, kvh = b * Hkv + h / (H / Hkv);\n"
    "    const int m0 = blockIdx.x * RB, c0 = blockIdx.z * CS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n"
    "    const size_t qoff = (size_t)bh * Sq * D;\n"
    "    for (int i = tid; i < D * CS; i += 256) {\n"
    "        int di = i / CS, cc = i - di * CS;\n"
    "        Ss[i] = S[(size_t)kvh * D * D + (size_t)di * D + c0 + cc];\n"
    "    }\n"
    "    for (int i = tid; i < D; i += 256) Zs[i] = Z[(size_t)kvh * D + i];\n"
    "    for (int i = tid; i < RB * D; i += 256) {\n"
    "        int r = i / D, d = i - r * D, gr = m0 + r;\n"
    "        Qf[i] = gr < Sq ? phi(ldv<T>(Q, qoff + (size_t)gr * D + d)) : 0.0f;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    for (int r = warp; r < RB; r += 8) {       // 8 warps, 4 rows each\n"
    "        int gr = m0 + r;\n"
    "        if (gr >= Sq) break;\n"
    "        const float* q = Qf + r * D;\n"
    "        float num[DL], den = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < DL; ++t) num[t] = 0.0f;\n"
    "        for (int di = 0; di < D; ++di) {\n"
    "            float qv = q[di];\n"
    "            den += qv * Zs[di];\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < DL; ++t) num[t] += qv * Ss[di * CS + lane + 32 * t];\n"
    "        }\n"
    "        float inv = den != 0.0f ? 1.0f / den : 0.0f;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < DL; ++t) stv<T>(O, qoff + (size_t)gr * D + c0 + lane + 32 * t, num[t] * inv);\n"
    "    }\n"
    "}\n"
    "\n"
    "// ── Linear attention, causal (chunked scan) ─────────────────────────────────\n"
    "// One block per (query head, 32-column slice of V/O); it walks the sequence\n"
    "// in 32-row chunks with a running state S[:, slice] = sum phi(k_j) v_j^T and\n"
    "// z = sum phi(k_j) over all earlier keys. Keys before the first query\n"
    "// (kvLen > seqLen, e.g. a cached prefix) are folded into the state first.\n"
    "// For a chunk: out = (A V + phi(Q) S) / (rowsum(A) + phi(Q) . z), where\n"
    "// A = phi(Q) phi(K)^T masked to j <= i inside the chunk.\n"
    "template <int T, int D>\n"
    "__device__ void lin_causal(const void* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,\n"
    "                           void* __restrict__ O, int Sq, int Sk, int H, int Hkv) {\n"
    "    constexpr int C = 32, DP = D + 1, SPER = D * 32 / 256;\n"
    "    extern __shared__ float sm[];\n"
    "    float* Qf = sm;               // C x D\n"
    "    float* Kf = Qf + C * D;       // C x DP\n"
    "    float* Vs = Kf + C * DP;      // C x 32\n"
    "    float* St = Vs + C * 32;      // D x 32 state slice\n"
    "    float* Zs = St + D * 32;      // D\n"
    "    float* A = Zs + D;            // C x (C+1)\n"
    "    const int bh = blockIdx.x, b = bh / H, h = bh % H, kvh = b * Hkv + h / (H / Hkv);\n"
    "    const int col0 = blockIdx.y * 32, tid = threadIdx.x;\n"
    "    const size_t qoff = (size_t)bh * Sq * D, koff = (size_t)kvh * Sk * D;\n"
    "    const int off = Sk - Sq;\n"
    "    for (int i = tid; i < D * 32; i += 256) St[i] = 0.0f;\n"
    "    for (int i = tid; i < D; i += 256) Zs[i] = 0.0f;\n"
    "    // thread -> (row i = tid / 8, 4 columns), for the output phase\n"
    "    const int oi = tid >> 3, oc = (tid & 7) * 4;\n"
    "    for (int base = 0; base < Sk; base += C) {\n"
    "        const int qstart = base - off;          // query index aligned with key 'base'\n"
    "        __syncthreads();\n"
    "        for (int i = tid; i < C * D; i += 256) {\n"
    "            int r = i / D, d = i - r * D, gk = base + r, gq = qstart + r;\n"
    "            Kf[r * DP + d] = gk < Sk ? phi(ldv<T>(K, koff + (size_t)gk * D + d)) : 0.0f;\n"
    "            Qf[i] = (gq >= 0 && gq < Sq) ? phi(ldv<T>(Q, qoff + (size_t)gq * D + d)) : 0.0f;\n"
    "        }\n"
    "        for (int i = tid; i < C * 32; i += 256) {\n"
    "            int r = i >> 5, cc = i & 31, gk = base + r;\n"
    "            Vs[i] = gk < Sk ? ldv<T>(V, koff + (size_t)gk * D + col0 + cc) : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        const bool hasQ = qstart + C > 0 && qstart < Sq;\n"
    "        if (hasQ) {\n"
    "            // A = phi(Q) phi(K)^T, masked j <= i\n"
    "            for (int e = tid; e < C * C; e += 256) {\n"
    "                int i = e / C, j = e - i * C;\n"
    "                float a = 0.0f;\n"
    "                if (j <= i) {\n"
    "#pragma unroll 8\n"
    "                    for (int d = 0; d < D; ++d) a += Qf[i * D + d] * Kf[j * DP + d];\n"
    "                }\n"
    "                A[i * (C + 1) + j] = a;\n"
    "            }\n"
    "            __syncthreads();\n"
    "            float num[4] = {0.0f, 0.0f, 0.0f, 0.0f}, den = 0.0f;\n"
    "            for (int j = 0; j <= oi; ++j) {\n"
    "                float a = A[oi * (C + 1) + j];\n"
    "                den += a;\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < 4; ++t) num[t] += a * Vs[j * 32 + oc + t];\n"
    "            }\n"
    "            for (int d = 0; d < D; ++d) {\n"
    "                float q = Qf[oi * D + d];\n"
    "                den += q * Zs[d];\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < 4; ++t) num[t] += q * St[d * 32 + oc + t];\n"
    "            }\n"
    "            int gq = qstart + oi;\n"
    "            if (gq >= 0 && gq < Sq) {\n"
    "                float inv = den != 0.0f ? 1.0f / den : 0.0f;\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < 4; ++t) stv<T>(O, qoff + (size_t)gq * D + col0 + oc + t, num[t] * inv);\n"
    "            }\n"
    "            __syncthreads();\n"
    "        }\n"
    "        // state += phi(K_c)^T V_c, z += sum phi(K_c)\n"
    "#pragma unroll\n"
    "        for (int e = 0; e < SPER; ++e) {\n"
    "            int idx = tid + 256 * e, d = idx >> 5, cc = idx & 31;\n"
    "            float a = St[idx];\n"
    "            for (int j = 0; j < C; ++j) a += Kf[j * DP + d] * Vs[j * 32 + cc];\n"
    "            St[idx] = a;\n"
    "        }\n"
    "        if (tid < D) {\n"
    "            float a = Zs[tid];\n"
    "            for (int j = 0; j < C; ++j) a += Kf[j * DP + tid];\n"
    "            Zs[tid] = a;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// ── Entry point for the variant this source was compiled as ─────────────────\n"
    "#if VK == 1\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const void* q, const void* k, const void* v, void* o,\n"
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal) {\n"
    "    flash_simt<VT, VD>(q, k, v, o, Sq, Sk, H, Hkv, sl2, causal);\n"
    "}\n"
    "#elif VK == 2\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const void* q, const void* k, const void* v, void* o,\n"
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal) {\n"
    "#if __CUDA_ARCH__ >= 800\n"
    "    flash_mma<VT == 2, VD>((const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,\n"
    "                           (unsigned short*)o, Sq, Sk, H, Hkv, sl2, causal);\n"
    "#endif\n"
    "}\n"
    "#elif VK == 3\n"
    "extern \"C\" __global__ __launch_bounds__(256) void attn_entry(const void* k, const void* v, float* S, float* Z, int Sk) {\n"
    "    lin_kv<VT, VD>(k, v, S, Z, Sk);\n"
    "}\n"
    "#elif VK == 4\n"
    "extern \"C\" __global__ __launch_bounds__(256) void attn_entry(const void* q, const float* S, const float* Z, void* o,\n"
    "                                                            int Sq, int H, int Hkv) {\n"
    "    lin_out<VT, VD>(q, S, Z, o, Sq, H, Hkv);\n"
    "}\n"
    "#elif VK == 5\n"
    "extern \"C\" __global__ __launch_bounds__(256) void attn_entry(const void* q, const void* k, const void* v, void* o,\n"
    "                                                            int Sq, int Sk, int H, int Hkv) {\n"
    "    lin_causal<VT, VD>(q, k, v, o, Sq, Sk, H, Hkv);\n"
    "}\n"
    "#endif\n";
}

// ── Variant cache ────────────────────────────────────────────────────────────
static void* g_att_fn[ATT_VARIANTS];
static int   g_att_state[ATT_VARIANTS];   // 0 = not tried, 1 = ready, -1 = failed
static unsigned long long g_att_ws = 0ULL;   // linear attention S/z workspace
static unsigned long g_att_ws_bytes = 0UL;

static int attn_dim_index_(unsigned long d) {
    if (d == 32UL) return 0;
    if (d == 64UL) return 1;
    if (d == 96UL) return 2;
    if (d == 128UL) return 3;
    if (d == 256UL) return 4;
    return -1;
}

int cuda_attention_head_dim_ok(unsigned long headDim) { return attn_dim_index_(headDim) >= 0; }

// Dynamic shared memory each kernel needs (must match the .cu layouts).
static unsigned long attn_smem_(int kind, unsigned long d) {
    if (kind == ATT_SIMT) return (32UL * d + 32UL * (d + 1UL) + 32UL * d + 4UL * 8UL * 32UL) * 4UL;
    if (kind == ATT_LIN_KV) return (32UL * 32UL + 32UL * d) * 4UL;
    if (kind == ATT_LIN_OUT) {
        unsigned long cs = d % 64UL == 0UL ? 64UL : 32UL;
        return (d * cs + d + 32UL * d) * 4UL;
    }
    if (kind == ATT_LIN_CAUSAL) return (32UL * d + 32UL * (d + 1UL) + 32UL * 32UL + d * 32UL + d + 32UL * 33UL) * 4UL;
    return 0UL;   // ATT_MMA: static shared memory only
}

// Writes "#define <name> <value>\n" at buf[p], returns the new end.
static unsigned long attn_put_define_(char* buf, unsigned long p, const char* name, unsigned long value) {
    unsafe {
        const char* pre = "#define ";
        unsigned long i = 0UL;
        while (pre[i] != (char)0) { buf[p] = pre[i]; p = p + 1UL; i = i + 1UL; }
        i = 0UL;
        while (name[i] != (char)0) { buf[p] = name[i]; p = p + 1UL; i = i + 1UL; }
        buf[p] = ' '; p = p + 1UL;
        char digits[24];
        unsigned long n = 0UL;
        if (value == 0UL) { digits[0] = '0'; n = 1UL; }
        while (value > 0UL) { digits[n] = (char)(48UL + value % 10UL); value = value / 10UL; n = n + 1UL; }
        while (n > 0UL) { n = n - 1UL; buf[p] = digits[n]; p = p + 1UL; }
        buf[p] = '\n'; p = p + 1UL;
    }
    return p;
}

// The kernel for (kind, dtype, head dim), compiled and loaded on first use;
// null if it can't be built.
static ?&void attn_kernel_(int kind, int dtype, unsigned long d) {
    int di = attn_dim_index_(d);
    if (di < 0 || kind < 1 || kind > ATT_KINDS || dtype < 0 || dtype > 2) return (void*)0;
    int idx = ((kind - 1) * 3 + dtype) * ATT_DIMS + di;
    unsafe {
        if (g_att_state[idx] == 1) return g_att_fn[idx];
        if (g_att_state[idx] == -1) return (void*)0;
        g_att_state[idx] = -1;
        const char* src = attn_kernel_source_();
        unsigned long srcLen = (unsigned long)strlen(src);
        char* buf = (char*)alloc_raw_(srcLen + 128UL);
        if (buf == (char*)0) return (void*)0;
        unsigned long p = 0UL;
        p = attn_put_define_(buf, p, "VK", (unsigned long)kind);
        p = attn_put_define_(buf, p, "VT", (unsigned long)dtype);
        p = attn_put_define_(buf, p, "VD", d);
        memcpy((void*)(buf + p), (const void*)src, srcLen);
        buf[p + srcLen] = (char)0;
        void* module = cuda_load_module((const &char)buf, "safec_attention.cu");
        dealloc_raw_((void*)buf);
        if (module == (void*)0) return (void*)0;
        void* func = (void*)0;
        if (cuModuleGetFunction(&func, module, "attn_entry") != 0) return (void*)0;
        unsigned long smem = attn_smem_(kind, d);
        if (smem > 49152UL && cuFuncSetAttribute(func, ATT_FN_MAX_DYN_SMEM, (int)smem) != 0) return (void*)0;
        g_att_fn[idx] = func;
        g_att_state[idx] = 1;
        return func;
    }
}

static int attn_launch_(?&void func, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int block,
                        unsigned long smem, &void params) {
    if (func == (void*)0) return 0;
    unsafe {
        return cuLaunchKernel((void*)func, gx, gy, gz, block, 1U, 1U, (unsigned int)smem,
                              (void*)cuda_stream_handle(), (void**)params, (void**)0) == 0;
    }
}

// ── Shape checks ─────────────────────────────────────────────────────────────

static int attn_shape_ok_(struct CudaAttnShape s) {
    if (s.batch == 0UL || s.heads == 0UL || s.kvHeads == 0UL || s.seqLen == 0UL || s.kvLen == 0UL) return 0;
    if (s.heads % s.kvHeads != 0UL || attn_dim_index_(s.headDim) < 0) return 0;
    if (s.seqLen > 2147483647UL || s.kvLen > 2147483647UL || s.heads > 65535UL) return 0;
    if (s.batch * s.heads > 65535UL) return 0;   // grid y
    return 1;
}

static unsigned long attn_q_elems_(struct CudaAttnShape s) {
    return checked_mul_size(checked_mul_size(checked_mul_size(s.batch, s.heads), s.seqLen), s.headDim);
}

static unsigned long attn_kv_elems_(struct CudaAttnShape s) {
    return checked_mul_size(checked_mul_size(checked_mul_size(s.batch, s.kvHeads), s.kvLen), s.headDim);
}

static int attn_bufs_ok_(unsigned long q, unsigned long k, unsigned long v, unsigned long o,
                         unsigned long qn, unsigned long kn, unsigned long vn, unsigned long on,
                         struct CudaAttnShape s) {
    return q != 0UL && k != 0UL && v != 0UL && o != 0UL &&
           qn >= attn_q_elems_(s) && on >= attn_q_elems_(s) &&
           kn >= attn_kv_elems_(s) && vn >= attn_kv_elems_(s);
}

// ── Flash attention ──────────────────────────────────────────────────────────

static int attn_flash_(unsigned long q, unsigned long k, unsigned long v, unsigned long o, int dtype,
                       struct CudaAttnShape s, float scale, int causal) {
    if (!cuda_available()) return 0;
    float sc = scale;
    if (!(sc > 0.0f)) {
        // 1 / sqrt(headDim) for the supported dims
        if (s.headDim == 32UL) sc = (float)0.17677669529663688;
        else if (s.headDim == 64UL) sc = 0.125f;
        else if (s.headDim == 96UL) sc = (float)0.10206207261596575;
        else if (s.headDim == 128UL) sc = (float)0.08838834764831845;
        else sc = 0.0625f;
    }
    float sl2 = sc * (float)1.4426950408889634;   // softmax in base 2
    // Tensor cores for fp16/bf16 on sm_80+ up to head dim 128.
    int useMma = dtype != 0 && cuda_compute_capability() >= 80 && s.headDim <= 128UL;
    int kind = useMma ? ATT_MMA : ATT_SIMT;
    unsigned long rows = useMma ? 64UL : 32UL;
    ?&void func = attn_kernel_(kind, dtype, s.headDim);
    unsafe {
        void* pq = (void*)q;
        void* pk = (void*)k;
        void* pv = (void*)v;
        void* po = (void*)o;
        int sq = (int)s.seqLen;
        int sk = (int)s.kvLen;
        int h = (int)s.heads;
        int hkv = (int)s.kvHeads;
        int c = causal ? 1 : 0;
        void* params[10];
        params[0] = (void*)&pq; params[1] = (void*)&pk; params[2] = (void*)&pv; params[3] = (void*)&po;
        params[4] = (void*)&sq; params[5] = (void*)&sk; params[6] = (void*)&h; params[7] = (void*)&hkv;
        params[8] = (void*)&sl2; params[9] = (void*)&c;
        return attn_launch_(func, (unsigned int)((s.seqLen + rows - 1UL) / rows), (unsigned int)(s.batch * s.heads), 1U,
                            128U, attn_smem_(kind, s.headDim), (&void)&params[0]);
    }
}

int cuda_flash_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                         struct CudaAttnShape s, float scale, int causal) {
    if (!attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    return attn_flash_(q.dev, k.dev, v.dev, out.dev, 0, s, scale, causal);
}

int cuda_half_flash_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                              struct CudaHalfBuf out, struct CudaAttnShape s, float scale, int causal) {
    if (!attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    if (q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != out.dtype) return 0;
    if (q.dtype != CUDA_FP16 && q.dtype != CUDA_BF16) return 0;
    return attn_flash_(q.dev, k.dev, v.dev, out.dev, q.dtype, s, scale, causal);
}

// ── Linear attention ─────────────────────────────────────────────────────────

// Grows the S/z workspace to 'bytes' (waits for queued work before freeing
// the old one, so call this outside graph capture).
static int attn_workspace_(unsigned long bytes) {
    if (bytes <= g_att_ws_bytes) return 1;
    unsafe {
        if (g_att_ws != 0ULL) {
            cuStreamSynchronize((void*)cuda_stream_handle());
            cuMemFree_v2(g_att_ws);
            g_att_ws = 0ULL;
            g_att_ws_bytes = 0UL;
        }
        if (cuMemAlloc_v2(&g_att_ws, bytes) != 0) { g_att_ws = 0ULL; return 0; }
    }
    g_att_ws_bytes = bytes;
    return 1;
}

static int attn_linear_(unsigned long q, unsigned long k, unsigned long v, unsigned long o, int dtype,
                        struct CudaAttnShape s, int causal) {
    if (!cuda_available()) return 0;
    unsigned long d = s.headDim;
    unsafe {
        void* pq = (void*)q;
        void* pk = (void*)k;
        void* pv = (void*)v;
        void* po = (void*)o;
        int sq = (int)s.seqLen;
        int sk = (int)s.kvLen;
        int h = (int)s.heads;
        int hkv = (int)s.kvHeads;
        if (causal) {
            if (s.kvLen < s.seqLen) return 0;
            ?&void func = attn_kernel_(ATT_LIN_CAUSAL, dtype, d);
            void* params[8];
            params[0] = (void*)&pq; params[1] = (void*)&pk; params[2] = (void*)&pv; params[3] = (void*)&po;
            params[4] = (void*)&sq; params[5] = (void*)&sk; params[6] = (void*)&h; params[7] = (void*)&hkv;
            return attn_launch_(func, (unsigned int)(s.batch * s.heads), (unsigned int)(d / 32UL), 1U, 256U,
                                attn_smem_(ATT_LIN_CAUSAL, d), (&void)&params[0]);
        }
        // pass 1: S [kv heads, d, d] and z [kv heads, d] in the workspace
        unsigned long kvHeadsTotal = s.batch * s.kvHeads;
        unsigned long sElems = checked_mul_size(kvHeadsTotal, d * d);
        unsigned long zElems = kvHeadsTotal * d;
        if (!attn_workspace_(checked_mul_size(sElems + zElems, 4UL))) return 0;
        ?&void fkv = attn_kernel_(ATT_LIN_KV, dtype, d);
        ?&void fout = attn_kernel_(ATT_LIN_OUT, dtype, d);
        if (fkv == (void*)0 || fout == (void*)0) return 0;
        void* stream = (void*)cuda_stream_handle();
        if (cuMemsetD32Async(g_att_ws, 0U, sElems + zElems, stream) != 0) return 0;
        void* pS = (void*)(unsigned long)g_att_ws;
        void* pZ = (void*)((unsigned long)g_att_ws + sElems * 4UL);
        void* p1[5];
        p1[0] = (void*)&pk; p1[1] = (void*)&pv; p1[2] = (void*)&pS; p1[3] = (void*)&pZ; p1[4] = (void*)&sk;
        if (!attn_launch_(fkv, (unsigned int)kvHeadsTotal, (unsigned int)((s.kvLen + 255UL) / 256UL),
                          (unsigned int)(d / 32UL), 256U, attn_smem_(ATT_LIN_KV, d), (&void)&p1[0])) return 0;
        unsigned long cs = d % 64UL == 0UL ? 64UL : 32UL;
        void* p2[7];
        p2[0] = (void*)&pq; p2[1] = (void*)&pS; p2[2] = (void*)&pZ; p2[3] = (void*)&po;
        p2[4] = (void*)&sq; p2[5] = (void*)&h; p2[6] = (void*)&hkv;
        return attn_launch_(fout, (unsigned int)((s.seqLen + 31UL) / 32UL), (unsigned int)(s.batch * s.heads),
                            (unsigned int)(d / cs), 256U, attn_smem_(ATT_LIN_OUT, d), (&void)&p2[0]);
    }
}

int cuda_linear_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                          struct CudaAttnShape s, int causal) {
    if (!attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    return attn_linear_(q.dev, k.dev, v.dev, out.dev, 0, s, causal);
}

int cuda_half_linear_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                               struct CudaHalfBuf out, struct CudaAttnShape s, int causal) {
    if (!attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    if (q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != out.dtype) return 0;
    if (q.dtype != CUDA_FP16 && q.dtype != CUDA_BF16) return 0;
    return attn_linear_(q.dev, k.dev, v.dev, out.dev, q.dtype, s, causal);
}

} // namespace std
