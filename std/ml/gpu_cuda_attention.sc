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
    "// row-major. Query head h reads kv head h / (heads / kvHeads). Query i sits\n"
    "// at position pos = i + (kvLen - seqLen): causal hides keys j > pos, and a\n"
    "// sliding window 'win' > 0 hides keys j <= pos - win.\n"
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
    "                           void* __restrict__ O, int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
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
    "    const int kbeg = win > 0 ? max(0, m0 + off - win + 1) : 0;\n"
    "    for (int n0 = kbeg; n0 < kend; n0 += BN) {\n"
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
    "            bool valid = j < Sk && (!causal || j <= gq + off) && (win <= 0 || j > gq + off - win);\n"
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
    "                          int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
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
    "    const int kbeg = win > 0 ? max(0, m0 + off - win + 1) : 0;\n"
    "    for (int n0 = kbeg; n0 < kend; n0 += BN) {\n"
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
    "                bool ok = j < Sk && (!causal || j <= qi + off) && (win <= 0 || j > qi + off - win);\n"
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
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
    "    flash_simt<VT, VD>(q, k, v, o, Sq, Sk, H, Hkv, sl2, causal, win);\n"
    "}\n"
    "#elif VK == 2\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const void* q, const void* k, const void* v, void* o,\n"
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
    "#if __CUDA_ARCH__ >= 800\n"
    "    flash_mma<VT == 2, VD>((const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,\n"
    "                           (unsigned short*)o, Sq, Sk, H, Hkv, sl2, causal, win);\n"
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

// ── cuDNN fused attention (fp16 / bf16) ─────────────────────────────────────
// For fp16/bf16 on sm_80+, flash attention goes to cuDNN's fused SDPA
// engine (on Hopper the wgmma "flash fprop" kernels PyTorch also uses)
// when libcudnn.so.9 can be loaded; cuDNN is opened with dlopen, so this
// library doesn't need it at link time and falls back to its own kernels
// without it. The operation graph is the one cudnn-frontend's sdpa() builds:
// Q K^T -> * scale -> [bottom-right causal mask] -> [sliding-window mask] ->
// softmax (max, sub, exp, sum, div) -> P V, fp32 inside. One cached plan per
// shape; the scalars (scale, lengths, window) are passed by value at execute.
extern ?&void dlopen(const char* file, int mode);
extern ?&void dlsym(void* handle, const char* name);

#define SD_RTLD_NOW          2
#define SD_TENSOR            17   // CUDNN_BACKEND_TENSOR_DESCRIPTOR
#define SD_POINTWISE         0
#define SD_ENGINECFG         3
#define SD_ENGINEHEUR        4
#define SD_PLAN              5
#define SD_OP_POINTWISE      13
#define SD_OPGRAPH           15
#define SD_VARIANT_PACK      16
#define SD_MATMUL            18
#define SD_OP_MATMUL         19
#define SD_REDUCTION         21
#define SD_OP_REDUCTION      22
#define SD_T_HANDLE          0    // cudnnBackendAttributeType_t
#define SD_T_DATA_TYPE       1
#define SD_T_BOOLEAN         2
#define SD_T_INT64           3
#define SD_T_VOID_PTR        6
#define SD_T_HEUR_MODE       8
#define SD_T_POINTWISE_MODE  14
#define SD_T_DESCRIPTOR      15
#define SD_T_REDUCTION_OP    18
#define SD_D_FLOAT           0    // cudnnDataType_t
#define SD_D_HALF            2
#define SD_D_INT32           4
#define SD_D_BF16            9
#define SD_D_BOOL            11
#define SD_PW_ADD            0    // cudnnPointwiseMode_t
#define SD_PW_MUL            1
#define SD_PW_DIV            6
#define SD_PW_SUB            9
#define SD_PW_EXP            13
#define SD_PW_LOG            15
#define SD_PW_CMP_GT         302
#define SD_PW_CMP_GE         303
#define SD_PW_GEN_INDEX      501
#define SD_PW_SELECT         601
#define SD_RED_ADD           0
#define SD_RED_MAX           3
#define SD_HEUR_A            3
#define SD_A_PW_MODE         0    // cudnnBackendAttributeName_t
#define SD_A_PW_PREC         1
#define SD_A_PW_AXIS         9
#define SD_A_HEUR_MODE       200
#define SD_A_HEUR_GRAPH      201
#define SD_A_HEUR_RESULTS    202
#define SD_A_PLAN_HANDLE     400
#define SD_A_PLAN_CFG        401
#define SD_A_PLAN_WS         402
#define SD_A_OPPW_DESC       750
#define SD_A_OPPW_X          751
#define SD_A_OPPW_B          752
#define SD_A_OPPW_Y          753
#define SD_A_OPPW_T          758
#define SD_A_GRAPH_HANDLE    800
#define SD_A_GRAPH_OPS       801
#define SD_A_T_ALIGN         900
#define SD_A_T_TYPE          901
#define SD_A_T_DIMS          902
#define SD_A_T_STRIDES       903
#define SD_A_T_UID           906
#define SD_A_T_VIRTUAL       907
#define SD_A_T_BYVAL         908
#define SD_A_VP_UIDS         1000
#define SD_A_VP_PTRS         1001
#define SD_A_VP_WS           1003
#define SD_A_MM_COMP         1500
#define SD_A_OPMM_A          1520
#define SD_A_OPMM_B          1521
#define SD_A_OPMM_C          1522
#define SD_A_OPMM_DESC       1523
#define SD_A_RED_OP          1600
#define SD_A_RED_COMP        1601
#define SD_A_OPRED_X         1610
#define SD_A_OPRED_Y         1611
#define SD_A_OPRED_DESC      1612
#define SD_SLOTS             16
#define SD_DESCS             128  // descriptors kept alive per plan (a masked graph uses ~80)
#define SD_CFGS              8
#define SD_MAX_WS            268435456UL   // 256 MiB

static int   g_sd_state = 0;      // 0 = not tried, 1 = ready, -1 = unavailable
static int   g_sd_enabled = 1;    // cuda_attention_use_cudnn
static void* g_sd_handle = (void*)0;
static fn int(void**) p_cudnnCreate;
static fn int(void*, void*) p_cudnnSetStream;
static fn int(int, void**) p_create;
static fn int(void*) p_destroy;
static fn int(void*, int, int, long, const void*) p_set;
static fn int(void*, int, int, long, long*, void*) p_get;
static fn int(void*) p_finalize;
static fn int(void*, void*, void*) p_execute;

// Per-shape plan cache.
static long  g_sd_key[SD_SLOTS * 9];       // dtype, b, hq, hk, sq, sk, d, causal, window>0; key[0] == 0: free
static int   g_sd_ok[SD_SLOTS];            // 1 = plan, -1 = cuDNN can't run this shape
static void* g_sd_plan[SD_SLOTS];
static void* g_sd_vp[SD_SLOTS];
static unsigned long long g_sd_ws[SD_SLOTS];
static unsigned long g_sd_ptr[SD_SLOTS * 4];   // q, k, v, o the variant pack points at
static void* g_sd_desc[SD_SLOTS * SD_DESCS];
static int   g_sd_ndesc[SD_SLOTS];
static float g_sd_f[SD_SLOTS * 4];          // by-value: scale, -inf, -inf, window offset
static int   g_sd_i[SD_SLOTS * 2];          // by-value: kvLen, seqLen

static int sd_load_() {
    if (g_sd_state != 0) return g_sd_state == 1;
    g_sd_state = -1;
    unsafe {
        void* lib = (void*)dlopen("libcudnn.so.9", SD_RTLD_NOW);
        if (lib == (void*)0) lib = (void*)dlopen("libcudnn.so", SD_RTLD_NOW);
        if (lib == (void*)0) return 0;
        void* f0 = (void*)dlsym(lib, "cudnnCreate");
        void* f1 = (void*)dlsym(lib, "cudnnSetStream");
        void* f2 = (void*)dlsym(lib, "cudnnBackendCreateDescriptor");
        void* f3 = (void*)dlsym(lib, "cudnnBackendDestroyDescriptor");
        void* f4 = (void*)dlsym(lib, "cudnnBackendSetAttribute");
        void* f5 = (void*)dlsym(lib, "cudnnBackendGetAttribute");
        void* f6 = (void*)dlsym(lib, "cudnnBackendFinalize");
        void* f7 = (void*)dlsym(lib, "cudnnBackendExecute");
        if (f0 == (void*)0 || f1 == (void*)0 || f2 == (void*)0 || f3 == (void*)0 || f4 == (void*)0 ||
            f5 == (void*)0 || f6 == (void*)0 || f7 == (void*)0) return 0;
        p_cudnnCreate = (fn int(void**))f0;
        p_cudnnSetStream = (fn int(void*, void*))f1;
        p_create = (fn int(int, void**))f2;
        p_destroy = (fn int(void*))f3;
        p_set = (fn int(void*, int, int, long, const void*))f4;
        p_get = (fn int(void*, int, int, long, long*, void*))f5;
        p_finalize = (fn int(void*))f6;
        p_execute = (fn int(void*, void*, void*))f7;
        if (p_cudnnCreate(&g_sd_handle) != 0) return 0;
        if (p_cudnnSetStream(g_sd_handle, (void*)cuda_stream_handle()) != 0) return 0;
    }
    g_sd_state = 1;
    return 1;
}

int cuda_attention_use_cudnn(int on) {
    g_sd_enabled = on ? 1 : 0;
    if (!on || !cuda_available()) return 0;
    return sd_load_();
}

// Keeps a descriptor alive with slot 's''s plan.
static ?&void sd_keep_(int s, ?&void d) {
    if (d == (void*)0) return (void*)0;
    unsafe {
        if (g_sd_ndesc[s] >= SD_DESCS) return (void*)0;
        g_sd_desc[s * SD_DESCS + g_sd_ndesc[s]] = (void*)d;
        g_sd_ndesc[s] = g_sd_ndesc[s] + 1;
    }
    return d;
}

static ?&void sd_tensor_(int s, long uid, int dtype, long d0, long d1, long d2, long d3,
                         long s0, long s1, long s2, long s3, int isVirtual, int byValue) {
    unsafe {
        void* d = (void*)0;
        if (p_create(SD_TENSOR, &d) != 0) return (void*)0;
        long dims[4];
        long strides[4];
        dims[0] = d0; dims[1] = d1; dims[2] = d2; dims[3] = d3;
        strides[0] = s0; strides[1] = s1; strides[2] = s2; strides[3] = s3;
        long align = 16L;
        unsigned char yes = (unsigned char)1;
        int ok = p_set(d, SD_A_T_TYPE, SD_T_DATA_TYPE, 1L, (const void*)&dtype) == 0 &&
                 p_set(d, SD_A_T_DIMS, SD_T_INT64, 4L, (const void*)&dims[0]) == 0 &&
                 p_set(d, SD_A_T_STRIDES, SD_T_INT64, 4L, (const void*)&strides[0]) == 0 &&
                 p_set(d, SD_A_T_UID, SD_T_INT64, 1L, (const void*)&uid) == 0 &&
                 p_set(d, SD_A_T_ALIGN, SD_T_INT64, 1L, (const void*)&align) == 0;
        if (ok && isVirtual) ok = p_set(d, SD_A_T_VIRTUAL, SD_T_BOOLEAN, 1L, (const void*)&yes) == 0;
        if (ok && byValue) ok = p_set(d, SD_A_T_BYVAL, SD_T_BOOLEAN, 1L, (const void*)&yes) == 0;
        if (!ok || p_finalize(d) != 0) { p_destroy(d); return (void*)0; }
        return sd_keep_(s, d);
    }
}

// A by-value scalar [1,1,1,1].
static ?&void sd_scalar_(int s, long uid, int dtype) {
    return sd_tensor_(s, uid, dtype, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 0, 1);
}

// Pointwise op y = mode(x [, b] [, t]) in 'prec'; axis >= 0 for GEN_INDEX.
static ?&void sd_pointwise_(int s, int mode, int prec, long axis, ?&void x, ?&void b, ?&void t, ?&void y) {
    if (x == (void*)0 || y == (void*)0) return (void*)0;
    unsafe {
        void* pw = (void*)0;
        if (p_create(SD_POINTWISE, &pw) != 0) return (void*)0;
        int ok = p_set(pw, SD_A_PW_MODE, SD_T_POINTWISE_MODE, 1L, (const void*)&mode) == 0 &&
                 p_set(pw, SD_A_PW_PREC, SD_T_DATA_TYPE, 1L, (const void*)&prec) == 0;
        if (ok && axis >= 0L) ok = p_set(pw, SD_A_PW_AXIS, SD_T_INT64, 1L, (const void*)&axis) == 0;
        if (!ok || p_finalize(pw) != 0) { p_destroy(pw); return (void*)0; }
        sd_keep_(s, pw);
        void* op = (void*)0;
        if (p_create(SD_OP_POINTWISE, &op) != 0) return (void*)0;
        void* xd = (void*)x;
        void* yd = (void*)y;
        ok = p_set(op, SD_A_OPPW_DESC, SD_T_DESCRIPTOR, 1L, (const void*)&pw) == 0 &&
             p_set(op, SD_A_OPPW_X, SD_T_DESCRIPTOR, 1L, (const void*)&xd) == 0 &&
             p_set(op, SD_A_OPPW_Y, SD_T_DESCRIPTOR, 1L, (const void*)&yd) == 0;
        if (ok && b != (void*)0) { void* bd = (void*)b; ok = p_set(op, SD_A_OPPW_B, SD_T_DESCRIPTOR, 1L, (const void*)&bd) == 0; }
        if (ok && t != (void*)0) { void* td = (void*)t; ok = p_set(op, SD_A_OPPW_T, SD_T_DESCRIPTOR, 1L, (const void*)&td) == 0; }
        if (!ok || p_finalize(op) != 0) { p_destroy(op); return (void*)0; }
        return sd_keep_(s, op);
    }
}

static ?&void sd_reduction_(int s, int redOp, ?&void x, ?&void y) {
    if (x == (void*)0 || y == (void*)0) return (void*)0;
    unsafe {
        void* rd = (void*)0;
        if (p_create(SD_REDUCTION, &rd) != 0) return (void*)0;
        int comp = SD_D_FLOAT;
        int ok = p_set(rd, SD_A_RED_OP, SD_T_REDUCTION_OP, 1L, (const void*)&redOp) == 0 &&
                 p_set(rd, SD_A_RED_COMP, SD_T_DATA_TYPE, 1L, (const void*)&comp) == 0;
        if (!ok || p_finalize(rd) != 0) { p_destroy(rd); return (void*)0; }
        sd_keep_(s, rd);
        void* op = (void*)0;
        if (p_create(SD_OP_REDUCTION, &op) != 0) return (void*)0;
        void* xd = (void*)x;
        void* yd = (void*)y;
        ok = p_set(op, SD_A_OPRED_DESC, SD_T_DESCRIPTOR, 1L, (const void*)&rd) == 0 &&
             p_set(op, SD_A_OPRED_X, SD_T_DESCRIPTOR, 1L, (const void*)&xd) == 0 &&
             p_set(op, SD_A_OPRED_Y, SD_T_DESCRIPTOR, 1L, (const void*)&yd) == 0;
        if (!ok || p_finalize(op) != 0) { p_destroy(op); return (void*)0; }
        return sd_keep_(s, op);
    }
}

static ?&void sd_matmul_(int s, ?&void a, ?&void b, ?&void c) {
    if (a == (void*)0 || b == (void*)0 || c == (void*)0) return (void*)0;
    unsafe {
        void* md = (void*)0;
        if (p_create(SD_MATMUL, &md) != 0) return (void*)0;
        int comp = SD_D_FLOAT;
        if (p_set(md, SD_A_MM_COMP, SD_T_DATA_TYPE, 1L, (const void*)&comp) != 0 || p_finalize(md) != 0) {
            p_destroy(md);
            return (void*)0;
        }
        sd_keep_(s, md);
        void* op = (void*)0;
        if (p_create(SD_OP_MATMUL, &op) != 0) return (void*)0;
        void* ad = (void*)a;
        void* bd = (void*)b;
        void* cd = (void*)c;
        int ok = p_set(op, SD_A_OPMM_A, SD_T_DESCRIPTOR, 1L, (const void*)&ad) == 0 &&
                 p_set(op, SD_A_OPMM_B, SD_T_DESCRIPTOR, 1L, (const void*)&bd) == 0 &&
                 p_set(op, SD_A_OPMM_C, SD_T_DESCRIPTOR, 1L, (const void*)&cd) == 0 &&
                 p_set(op, SD_A_OPMM_DESC, SD_T_DESCRIPTOR, 1L, (const void*)&md) == 0;
        if (!ok || p_finalize(op) != 0) { p_destroy(op); return (void*)0; }
        return sd_keep_(s, op);
    }
}

// Builds slot s's operation graph and plan; 1 on success.
static int sd_build_(int s, int dtype, struct CudaAttnShape a, int causal) {
    long b = (long)a.batch;
    long hq = (long)a.heads;
    long hk = (long)a.kvHeads;
    long sq = (long)a.seqLen;
    long sk = (long)a.kvLen;
    long d = (long)a.headDim;
    int io = dtype == CUDA_BF16 ? SD_D_BF16 : SD_D_HALF;
    int F = SD_D_FLOAT;
    long sS0 = hq * sq * sk;
    long sS1 = sq * sk;
    unsafe {
        void* ops[24];
        int nops = 0;
        ?&void tq = sd_tensor_(s, 1L, io, b, hq, sq, d, hq * sq * d, sq * d, d, 1L, 0, 0);
        ?&void tkt = sd_tensor_(s, 2L, io, b, hk, d, sk, hk * sk * d, sk * d, 1L, d, 0, 0);   // K^T view of K
        ?&void tv = sd_tensor_(s, 3L, io, b, hk, sk, d, hk * sk * d, sk * d, d, 1L, 0, 0);
        ?&void to = sd_tensor_(s, 4L, io, b, hq, sq, d, hq * sq * d, sq * d, d, 1L, 0, 0);
        ?&void tscale = sd_scalar_(s, 5L, F);
        ?&void s0 = sd_tensor_(s, 6L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
        ops[nops] = (void*)sd_matmul_(s, tq, tkt, s0); nops = nops + 1;
        ?&void last = sd_tensor_(s, 7L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_MUL, F, -1L, s0, tscale, (void*)0, last); nops = nops + 1;
        if (causal) {
            // keep where row + (kvLen - seqLen) >= col
            ?&void row = sd_tensor_(s, 8L, SD_D_INT32, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ?&void col = sd_tensor_(s, 9L, SD_D_INT32, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_GEN_INDEX, F, 2L, last, (void*)0, (void*)0, row); nops = nops + 1;
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_GEN_INDEX, F, 3L, last, (void*)0, (void*)0, col); nops = nops + 1;
            ?&void tskv = sd_scalar_(s, 10L, SD_D_INT32);
            ?&void r1 = sd_tensor_(s, 11L, SD_D_INT32, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_ADD, SD_D_INT32, -1L, row, tskv, (void*)0, r1); nops = nops + 1;
            ?&void tsq = sd_scalar_(s, 12L, SD_D_INT32);
            ?&void r2 = sd_tensor_(s, 13L, SD_D_INT32, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_SUB, SD_D_INT32, -1L, r1, tsq, (void*)0, r2); nops = nops + 1;
            ?&void keep = sd_tensor_(s, 14L, SD_D_BOOL, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_CMP_GE, SD_D_BOOL, -1L, r2, col, (void*)0, keep); nops = nops + 1;
            ?&void ninf = sd_scalar_(s, 15L, F);
            ?&void masked = sd_tensor_(s, 16L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_SELECT, F, -1L, last, ninf, keep, masked); nops = nops + 1;
            last = masked;
        }
        if (a.window > 0UL) {
            // keep where col + (window - kvLen + seqLen) > row
            ?&void row = sd_tensor_(s, 17L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ?&void col = sd_tensor_(s, 18L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_GEN_INDEX, F, 2L, last, (void*)0, (void*)0, row); nops = nops + 1;
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_GEN_INDEX, F, 3L, last, (void*)0, (void*)0, col); nops = nops + 1;
            ?&void toff = sd_scalar_(s, 19L, F);
            ?&void c1 = sd_tensor_(s, 20L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_ADD, F, -1L, col, toff, (void*)0, c1); nops = nops + 1;
            ?&void keep = sd_tensor_(s, 21L, SD_D_BOOL, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_CMP_GT, SD_D_BOOL, -1L, c1, row, (void*)0, keep); nops = nops + 1;
            ?&void ninf = sd_scalar_(s, 22L, F);
            ?&void masked = sd_tensor_(s, 23L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
            ops[nops] = (void*)sd_pointwise_(s, SD_PW_SELECT, F, -1L, last, ninf, keep, masked); nops = nops + 1;
            last = masked;
        }
        // softmax with the row statistics cudnn-frontend also forms
        ?&void mx = sd_tensor_(s, 26L, F, b, hq, sq, 1L, hq * sq, sq, 1L, 1L, 1, 0);
        ops[nops] = (void*)sd_reduction_(s, SD_RED_MAX, last, mx); nops = nops + 1;
        ?&void sub = sd_tensor_(s, 27L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_SUB, F, -1L, last, mx, (void*)0, sub); nops = nops + 1;
        ?&void ex = sd_tensor_(s, 28L, F, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_EXP, F, -1L, sub, (void*)0, (void*)0, ex); nops = nops + 1;
        ?&void sum = sd_tensor_(s, 29L, F, b, hq, sq, 1L, hq * sq, sq, 1L, 1L, 1, 0);
        ops[nops] = (void*)sd_reduction_(s, SD_RED_ADD, ex, sum); nops = nops + 1;
        ?&void lg = sd_tensor_(s, 30L, F, b, hq, sq, 1L, hq * sq, sq, 1L, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_LOG, F, -1L, sum, (void*)0, (void*)0, lg); nops = nops + 1;
        ?&void stats = sd_tensor_(s, 24L, F, b, hq, sq, 1L, hq * sq, sq, 1L, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_ADD, F, -1L, mx, lg, (void*)0, stats); nops = nops + 1;
        ?&void pr = sd_tensor_(s, 25L, io, b, hq, sq, sk, sS0, sS1, sk, 1L, 1, 0);
        ops[nops] = (void*)sd_pointwise_(s, SD_PW_DIV, F, -1L, ex, sum, (void*)0, pr); nops = nops + 1;
        ops[nops] = (void*)sd_matmul_(s, pr, tv, to); nops = nops + 1;
        int i = 0;
        while (i < nops) { if (ops[i] == (void*)0) return 0; i = i + 1; }

        void* graph = (void*)0;
        if (p_create(SD_OPGRAPH, &graph) != 0) return 0;
        sd_keep_(s, graph);
        long nl = (long)nops;
        if (p_set(graph, SD_A_GRAPH_OPS, SD_T_DESCRIPTOR, nl, (const void*)&ops[0]) != 0 ||
            p_set(graph, SD_A_GRAPH_HANDLE, SD_T_HANDLE, 1L, (const void*)&g_sd_handle) != 0 ||
            p_finalize(graph) != 0) return 0;

        // First heuristic engine config that finalizes within the workspace limit.
        void* heur = (void*)0;
        int hm = SD_HEUR_A;
        if (p_create(SD_ENGINEHEUR, &heur) != 0) return 0;
        sd_keep_(s, heur);
        if (p_set(heur, SD_A_HEUR_GRAPH, SD_T_DESCRIPTOR, 1L, (const void*)&graph) != 0 ||
            p_set(heur, SD_A_HEUR_MODE, SD_T_HEUR_MODE, 1L, (const void*)&hm) != 0 ||
            p_finalize(heur) != 0) return 0;
        void* cfgs[SD_CFGS];
        i = 0;
        while (i < SD_CFGS) {
            cfgs[i] = (void*)0;
            if (p_create(SD_ENGINECFG, &cfgs[i]) != 0) return 0;
            sd_keep_(s, cfgs[i]);
            i = i + 1;
        }
        long ncfg = 0L;
        if (p_get(heur, SD_A_HEUR_RESULTS, SD_T_DESCRIPTOR, (long)SD_CFGS, &ncfg, (void*)&cfgs[0]) != 0) return 0;
        long c = 0L;
        while (c < ncfg && c < (long)SD_CFGS) {
            void* plan = (void*)0;
            long ws = 0L;
            long got = 0L;
            if (p_create(SD_PLAN, &plan) == 0) {
                if (p_set(plan, SD_A_PLAN_HANDLE, SD_T_HANDLE, 1L, (const void*)&g_sd_handle) == 0 &&
                    p_set(plan, SD_A_PLAN_CFG, SD_T_DESCRIPTOR, 1L, (const void*)&cfgs[c]) == 0 &&
                    p_finalize(plan) == 0 &&
                    p_get(plan, SD_A_PLAN_WS, SD_T_INT64, 1L, &got, (void*)&ws) == 0 &&
                    ws >= 0L && (unsigned long)ws <= SD_MAX_WS) {
                    unsigned long long wsp = 0ULL;
                    if (ws == 0L || cuMemAlloc_v2(&wsp, (size_t)ws) == 0) {
                        g_sd_plan[s] = plan;
                        g_sd_ws[s] = wsp;
                        return 1;
                    }
                }
                p_destroy(plan);
            }
            c = c + 1L;
        }
    }
    return 0;
}

// (Re)binds slot s's variant pack to these buffers.
static int sd_bind_(int s, unsigned long q, unsigned long k, unsigned long v, unsigned long o, int causal, int windowed) {
    if (g_sd_vp[s] != (void*)0 && g_sd_ptr[s * 4] == q && g_sd_ptr[s * 4 + 1] == k &&
        g_sd_ptr[s * 4 + 2] == v && g_sd_ptr[s * 4 + 3] == o) return 1;
    unsafe {
        if (g_sd_vp[s] != (void*)0) {
            cuStreamSynchronize((void*)cuda_stream_handle());   // queued work may still use the old pack
            p_destroy(g_sd_vp[s]);
            g_sd_vp[s] = (void*)0;
        }
        long uids[10];
        void* ptrs[10];
        int n = 0;
        uids[n] = 1L; ptrs[n] = (void*)q; n = n + 1;
        uids[n] = 2L; ptrs[n] = (void*)k; n = n + 1;
        uids[n] = 3L; ptrs[n] = (void*)v; n = n + 1;
        uids[n] = 4L; ptrs[n] = (void*)o; n = n + 1;
        uids[n] = 5L; ptrs[n] = (void*)&g_sd_f[s * 4]; n = n + 1;
        if (causal) {
            uids[n] = 10L; ptrs[n] = (void*)&g_sd_i[s * 2]; n = n + 1;
            uids[n] = 12L; ptrs[n] = (void*)&g_sd_i[s * 2 + 1]; n = n + 1;
            uids[n] = 15L; ptrs[n] = (void*)&g_sd_f[s * 4 + 1]; n = n + 1;
        }
        if (windowed) {
            uids[n] = 19L; ptrs[n] = (void*)&g_sd_f[s * 4 + 3]; n = n + 1;
            uids[n] = 22L; ptrs[n] = (void*)&g_sd_f[s * 4 + 2]; n = n + 1;
        }
        void* vp = (void*)0;
        if (p_create(SD_VARIANT_PACK, &vp) != 0) return 0;
        void* ws = (void*)(unsigned long)g_sd_ws[s];
        long nl = (long)n;
        if (p_set(vp, SD_A_VP_UIDS, SD_T_INT64, nl, (const void*)&uids[0]) != 0 ||
            p_set(vp, SD_A_VP_PTRS, SD_T_VOID_PTR, nl, (const void*)&ptrs[0]) != 0 ||
            p_set(vp, SD_A_VP_WS, SD_T_VOID_PTR, 1L, (const void*)&ws) != 0 ||
            p_finalize(vp) != 0) { p_destroy(vp); return 0; }
        g_sd_vp[s] = vp;
    }
    g_sd_ptr[s * 4] = q; g_sd_ptr[s * 4 + 1] = k; g_sd_ptr[s * 4 + 2] = v; g_sd_ptr[s * 4 + 3] = o;
    return 1;
}

// Runs flash attention through cuDNN; 1 = enqueued, 0 = use the fallback.
static int sd_run_(unsigned long q, unsigned long k, unsigned long v, unsigned long o, int dtype,
                   struct CudaAttnShape a, float scale, int causal) {
    if (!g_sd_enabled || !sd_load_()) return 0;
    long key[9];
    key[0] = (long)dtype; key[1] = (long)a.batch; key[2] = (long)a.heads; key[3] = (long)a.kvHeads;
    key[4] = (long)a.seqLen; key[5] = (long)a.kvLen; key[6] = (long)a.headDim; key[7] = causal ? 1L : 0L;
    key[8] = a.window > 0UL ? 1L : 0L;
    int s = -1;
    int free = -1;
    int i = 0;
    while (i < SD_SLOTS && s < 0) {
        int same = g_sd_key[i * 9] != 0L;
        int j = 0;
        while (same && j < 9) { if (g_sd_key[i * 9 + j] != key[j]) same = 0; j = j + 1; }
        if (same) s = i;
        if (free < 0 && g_sd_key[i * 9] == 0L) free = i;
        i = i + 1;
    }
    if (s < 0) {
        if (free < 0) return 0;   // cache full: own kernels
        s = free;
        i = 0;
        while (i < 9) { g_sd_key[s * 9 + i] = key[i]; i = i + 1; }
        g_sd_ok[s] = sd_build_(s, dtype, a, causal) ? 1 : -1;
    }
    if (g_sd_ok[s] != 1) return 0;
    unsafe {
        unsigned int ninf = 0xff800000U;
        float nf = 0.0f;
        memcpy((void*)&nf, (const void*)&ninf, 4UL);
        g_sd_f[s * 4] = scale;
        g_sd_f[s * 4 + 1] = nf;
        g_sd_f[s * 4 + 2] = nf;
        g_sd_f[s * 4 + 3] = (float)((long)a.window - (long)a.kvLen + (long)a.seqLen);
        g_sd_i[s * 2] = (int)a.kvLen;
        g_sd_i[s * 2 + 1] = (int)a.seqLen;
        if (!sd_bind_(s, q, k, v, o, causal, a.window > 0UL)) return 0;
        return p_execute(g_sd_handle, g_sd_plan[s], g_sd_vp[s]) == 0;
    }
}

// ── Flash attention ──────────────────────────────────────────────────────────

static int attn_flash_(unsigned long q, unsigned long k, unsigned long v, unsigned long o, int dtype,
                       struct CudaAttnShape s, float scale, int causal) {
    if (s.window > 2147483647UL) return 0;
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
    if (dtype != 0 && cuda_compute_capability() >= 80 && sd_run_(q, k, v, o, dtype, s, sc, causal)) return 1;
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
        int win = (int)s.window;
        void* params[11];
        params[0] = (void*)&pq; params[1] = (void*)&pk; params[2] = (void*)&pv; params[3] = (void*)&po;
        params[4] = (void*)&sq; params[5] = (void*)&sk; params[6] = (void*)&h; params[7] = (void*)&hkv;
        params[8] = (void*)&sl2; params[9] = (void*)&c; params[10] = (void*)&win;
        return attn_launch_(func, (unsigned int)((s.seqLen + rows - 1UL) / rows), (unsigned int)(s.batch * s.heads), 1U,
                            128U, attn_smem_(kind, s.headDim), (&void)&params[0]);
    }
}

// The rocm_flash_attention / rocm_linear_attention form: one batch, kvHeads == heads.
static struct CudaAttnShape attn_heads_shape_(unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                                              unsigned long headDim) {
    struct CudaAttnShape s;
    s.batch = 1UL; s.heads = heads; s.kvHeads = heads; s.seqLen = seqLen; s.kvLen = kvLen;
    s.headDim = headDim; s.window = 0UL;
    return s;
}

int cuda_flash_attention_ex(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                            struct CudaAttnShape s, float scale, int causal) {
    if (!attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    return attn_flash_(q.dev, k.dev, v.dev, out.dev, 0, s, scale, causal);
}

int cuda_flash_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                         unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                         float scale, int causal) {
    return cuda_flash_attention_ex(q, k, v, out, attn_heads_shape_(heads, seqLen, kvLen, headDim), scale, causal);
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

int cuda_linear_attention_ex(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                             struct CudaAttnShape s, int causal) {
    if (s.window != 0UL || !attn_shape_ok_(s) ||
        !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    return attn_linear_(q.dev, k.dev, v.dev, out.dev, 0, s, causal);
}

int cuda_linear_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                          unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                          int causal) {
    return cuda_linear_attention_ex(q, k, v, out, attn_heads_shape_(heads, seqLen, kvLen, headDim), causal);
}

int cuda_half_linear_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                               struct CudaHalfBuf out, struct CudaAttnShape s, int causal) {
    if (s.window != 0UL || !attn_shape_ok_(s) || !attn_bufs_ok_(q.dev, k.dev, v.dev, out.dev, q.len, k.len, v.len, out.len, s)) return 0;
    if (q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != out.dtype) return 0;
    if (q.dtype != CUDA_FP16 && q.dtype != CUDA_BF16) return 0;
    return attn_linear_(q.dev, k.dev, v.dev, out.dev, q.dtype, s, causal);
}

// ── Host-array fp32 versions (upload → compute → read back) ──────────────────

static int attn_host_(const &float q, const &float k, const &float v, &float out,
                      unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                      float scale, int causal, int linear) {
    struct CudaAttnShape s = attn_heads_shape_(heads, seqLen, kvLen, headDim);
    if (!attn_shape_ok_(s)) return 0;
    unsigned long qn = attn_q_elems_(s);
    unsigned long kn = attn_kv_elems_(s);
    struct CudaBuf dq = cuda_upload(q, qn);
    struct CudaBuf dk = cuda_upload(k, kn);
    struct CudaBuf dv = cuda_upload(v, kn);
    struct CudaBuf dout = cuda_alloc(qn);
    int ok = dq.dev != 0UL && dk.dev != 0UL && dv.dev != 0UL && dout.dev != 0UL;
    if (ok) {
        if (linear) ok = cuda_linear_attention_ex(dq, dk, dv, dout, s, causal);
        else ok = cuda_flash_attention_ex(dq, dk, dv, dout, s, scale, causal);
    }
    ok = ok && cuda_read(dout, out, qn);
    cuda_free(&dq); cuda_free(&dk); cuda_free(&dv); cuda_free(&dout);
    return ok;
}

int cuda_flash_attention_f32(const &float q, const &float k, const &float v, &float out,
                             unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                             unsigned long headDim, float scale, int causal) {
    return attn_host_(q, k, v, out, heads, seqLen, kvLen, headDim, scale, causal, 0);
}

int cuda_linear_attention_f32(const &float q, const &float k, const &float v, &float out,
                              unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                              unsigned long headDim, int causal) {
    return attn_host_(q, k, v, out, heads, seqLen, kvLen, headDim, 0.0f, causal, 1);
}

} // namespace std
