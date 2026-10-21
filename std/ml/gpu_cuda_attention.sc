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
#define ATT_LIN_CAUSAL       5    // linear attention, causal sequential scan (head dim 256)
#define ATT_LIN_HSTATE       6    // linear attention, per-head chunk states (head dim <= 128)
#define ATT_LIN_CSCAN        7    // linear attention, prefix / total over the state parts
#define ATT_LIN_HOUT         8    // linear attention, chunk outputs
#define ATT_TF32X3           9    // flash attention, fp32 on tensor cores (3xTF32), sm_80+
#define ATT_DEC_PART         10   // decode attention split partials; 10..14 = up to 1, 2, 4, 8, 16 query rows
#define ATT_DEC_COMBINE      15   // decode attention, merge of the split partials
#define ATT_KINDS            15
#define ATT_DIMS             5
#define ATT_VARIANTS         225  // ATT_KINDS * 3 dtypes * ATT_DIMS
#define ATT_DEC_ROWS         16UL // decode path: at most this many query rows per kv head
#define ATT_DEC_SPLITS       512UL // most key splits (dec_combine holds 1024 weights)
#define ATT_LC               64UL // linear attention chunk rows (LinChunk in the .cu)
#define ATT_LIN_BLOCKS       512UL // lin_hstate blocks to aim for (parts x kv heads)

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
    "// ── Flash attention, fp32 on tensor cores (3xTF32, sm_80+) ──────────────────\n"
    "// Same FlashAttention-2 structure as flash_mma, for fp32 data: every product\n"
    "// is split as a = a_hi + a_lo (both TF32) and computed as\n"
    "// a_hi*b_hi + a_hi*b_lo + a_lo*b_hi with mma.sync.m16n8k8 (fp32 accumulate),\n"
    "// which keeps close to full fp32 accuracy (the \"fast fp32\" CUTLASS's\n"
    "// memory-efficient attention uses). Q, K, V and the warp's P tile go through\n"
    "// shared memory; operands are split as they are loaded into registers.\n"
    "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800\n"
    "// Rounds to TF32 (10 mantissa bits), to nearest with ties away from zero: the same result as\n"
    "// cvt.rna.tf32.f32 for finite values, which sm_90 emulates with a much longer sequence.\n"
    "__device__ __forceinline__ unsigned rna_tf32(float x) { return (__float_as_uint(x) + 0x1000u) & 0xffffe000u; }\n"
    "__device__ __forceinline__ void split_tf32(float x, unsigned& hi, unsigned& lo) {\n"
    "    unsigned h = rna_tf32(x);\n"
    "    hi = h;\n"
    "    lo = rna_tf32(x - __uint_as_float(h));\n"
    "}\n"
    "__device__ __forceinline__ void mma1688(float* c, const unsigned* a, unsigned b0, unsigned b1) {\n"
    "    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n\"\n"
    "                 : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n"
    "                 : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1));\n"
    "}\n"
    "// c += a * b in 3xTF32; a as fp32 A-fragment values, b as an fp32 pair.\n"
    "__device__ __forceinline__ void mma3xtf32(float* c, const float* af, float b0f, float b1f) {\n"
    "    unsigned ah[4], al[4], bh0, bl0, bh1, bl1;\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < 4; ++i) split_tf32(af[i], ah[i], al[i]);\n"
    "    split_tf32(b0f, bh0, bl0);\n"
    "    split_tf32(b1f, bh1, bl1);\n"
    "    mma1688(c, al, bh0, bh1);\n"
    "    mma1688(c, ah, bl0, bl1);\n"
    "    mma1688(c, ah, bh0, bh1);\n"
    "}\n"
    "\n"
    "// 64 query rows, 64-key tiles (32 above head dim 64, to fit two blocks per SM). Operands are\n"
    "// split as they are loaded into registers: splitting K / V (or Q) once into shared memory\n"
    "// instead doubles that operand's shared-memory traffic, which is the tighter limit.\n"
    "template <int D>\n"
    "__device__ void flash_tf32x3(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,\n"
    "                             float* __restrict__ O, int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
    "    constexpr int BM = 64, BN = D > 64 ? 32 : 64, DP = D + 4, PP = BN + 4, NB = BN / 8, ND = D / 8;\n"
    "    extern __shared__ float sm[];\n"
    "    float* Qs = sm; float* Ks = Qs + BM * DP; float* Vs = Ks + BN * DP; float* Ps = Vs + BN * DP;\n"
    "    const int bh = blockIdx.y, b = bh / H, h = bh % H, hk = h / (H / Hkv);\n"
    "    const size_t qoff = (size_t)bh * Sq * D, koff = (size_t)(b * Hkv + hk) * Sk * D;\n"
    "    const int m0 = blockIdx.x * BM, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n"
    "    const int g = lane >> 2, c = lane & 3, off = Sk - Sq;\n"
    "    const int r0 = m0 + warp * 16 + g, r1 = r0 + 8;\n"
    "    for (int i = tid; i < BM * (D / 4); i += 128) {\n"
    "        int r = i / (D / 4), d4 = (i - r * (D / 4)) * 4, gr = m0 + r;\n"
    "        float4 x = gr < Sq ? *(const float4*)(Q + qoff + (size_t)gr * D + d4) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "        *(float4*)(Qs + r * DP + d4) = make_float4(x.x * sl2, x.y * sl2, x.z * sl2, x.w * sl2);\n"
    "    }\n"
    "    float o[ND][4];\n"
    "    for (int n = 0; n < ND; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;\n"
    "    float m_0 = NEG_INF, m_1 = NEG_INF, l_0 = 0.0f, l_1 = 0.0f;\n"
    "    int kend = Sk;\n"
    "    if (causal) { int last = min(m0 + BM, Sq) - 1; kend = max(0, min(Sk, last + off + 1)); }\n"
    "    const int kbeg = win > 0 ? max(0, m0 + off - win + 1) : 0;\n"
    "    const float* qw = Qs + (warp * 16) * DP; float* pw = Ps + warp * 16 * PP;\n"
    "    for (int n0 = kbeg; n0 < kend; n0 += BN) {\n"
    "        __syncthreads();\n"
    "        for (int i = tid; i < BN * (D / 4); i += 128) {\n"
    "            int r = i / (D / 4), d4 = (i - r * (D / 4)) * 4, gk = n0 + r;\n"
    "            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;\n"
    "            if (gk < Sk) { kv = *(const float4*)(K + koff + (size_t)gk * D + d4); vv = *(const float4*)(V + koff + (size_t)gk * D + d4); }\n"
    "            *(float4*)(Ks + r * DP + d4) = kv; *(float4*)(Vs + r * DP + d4) = vv;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        float s[NB][4];\n"
    "        for (int nb = 0; nb < NB; ++nb) s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int kk = 0; kk < D / 8; ++kk) {\n"
    "            unsigned ah[4], al[4];\n"
    "            const int q0 = g * DP + kk * 8 + c, q1 = q0 + 8 * DP;\n"
    "            split_tf32(qw[q0], ah[0], al[0]); split_tf32(qw[q1], ah[1], al[1]);\n"
    "            split_tf32(qw[q0 + 4], ah[2], al[2]); split_tf32(qw[q1 + 4], ah[3], al[3]);\n"
    "#pragma unroll\n"
    "            for (int nb = 0; nb < NB; ++nb) {\n"
    "                const float* kr = Ks + (nb * 8 + g) * DP + kk * 8 + c;\n"
    "                unsigned bh0, bl0, bh1, bl1;\n"
    "                split_tf32(kr[0], bh0, bl0); split_tf32(kr[4], bh1, bl1);\n"
    "                mma1688(s[nb], al, bh0, bh1);\n"
    "                mma1688(s[nb], ah, bl0, bl1);\n"
    "                mma1688(s[nb], ah, bh0, bh1);\n"
    "            }\n"
    "        }\n"
    "        // masks only on tiles that cross the sequence end, the causal diagonal or the window edge\n"
    "        const bool edge = n0 + BN > Sk || (causal && n0 + BN - 1 > m0 + off) || (win > 0 && n0 <= m0 + BM - 1 + off - win);\n"
    "        float mx0 = NEG_INF, mx1 = NEG_INF;\n"
    "        for (int nb = 0; nb < NB; ++nb) {\n"
    "            if (edge)\n"
    "                for (int e = 0; e < 4; ++e) { int j = n0 + nb * 8 + c * 2 + (e & 1); int qi = e < 2 ? r0 : r1; if (!(j < Sk && (!causal || j <= qi + off) && (win <= 0 || j > qi + off - win))) s[nb][e] = NEG_INF; }\n"
    "            mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1])); mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));\n"
    "        }\n"
    "        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n"
    "        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n"
    "        float mn0 = fmaxf(m_0, mx0), mn1 = fmaxf(m_1, mx1), mu0 = mn0 == NEG_INF ? 0.0f : mn0, mu1 = mn1 == NEG_INF ? 0.0f : mn1;\n"
    "        float al0 = exp2f(m_0 - mu0), al1 = exp2f(m_1 - mu1); m_0 = mn0; m_1 = mn1;\n"
    "        float ps0 = 0.0f, ps1 = 0.0f;\n"
    "        for (int nb = 0; nb < NB; ++nb) {\n"
    "            float p0 = exp2f(s[nb][0] - mu0), p1 = exp2f(s[nb][1] - mu0), p2 = exp2f(s[nb][2] - mu1), p3 = exp2f(s[nb][3] - mu1);\n"
    "            ps0 += p0 + p1; ps1 += p2 + p3;\n"
    "            pw[g * PP + nb * 8 + c * 2] = p0; pw[g * PP + nb * 8 + c * 2 + 1] = p1; pw[(g + 8) * PP + nb * 8 + c * 2] = p2; pw[(g + 8) * PP + nb * 8 + c * 2 + 1] = p3;\n"
    "        }\n"
    "        l_0 = l_0 * al0 + ps0; l_1 = l_1 * al1 + ps1;\n"
    "        for (int n = 0; n < ND; ++n) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }\n"
    "        __syncwarp();\n"
    "#pragma unroll\n"
    "        for (int ks = 0; ks < BN / 8; ++ks) {\n"
    "            unsigned ah[4], al[4];\n"
    "            split_tf32(pw[g * PP + ks * 8 + c], ah[0], al[0]); split_tf32(pw[(g + 8) * PP + ks * 8 + c], ah[1], al[1]);\n"
    "            split_tf32(pw[g * PP + ks * 8 + c + 4], ah[2], al[2]); split_tf32(pw[(g + 8) * PP + ks * 8 + c + 4], ah[3], al[3]);\n"
    "#pragma unroll\n"
    "            for (int n = 0; n < ND; ++n) {\n"
    "                const float* vr = Vs + (ks * 8 + c) * DP + n * 8 + g;\n"
    "                unsigned bh0, bl0, bh1, bl1;\n"
    "                split_tf32(vr[0], bh0, bl0); split_tf32(vr[4 * DP], bh1, bl1);\n"
    "                mma1688(o[n], al, bh0, bh1);\n"
    "                mma1688(o[n], ah, bl0, bl1);\n"
    "                mma1688(o[n], ah, bh0, bh1);\n"
    "            }\n"
    "        }\n"
    "        __syncwarp();\n"
    "    }\n"
    "    l_0 += __shfl_xor_sync(FULL, l_0, 1); l_0 += __shfl_xor_sync(FULL, l_0, 2); l_1 += __shfl_xor_sync(FULL, l_1, 1); l_1 += __shfl_xor_sync(FULL, l_1, 2);\n"
    "    float i0 = l_0 > 0.0f ? 1.0f / l_0 : 0.0f, i1 = l_1 > 0.0f ? 1.0f / l_1 : 0.0f;\n"
    "    for (int n = 0; n < ND; ++n) { int col = n * 8 + c * 2;\n"
    "        if (r0 < Sq) *(float2*)(O + qoff + (size_t)r0 * D + col) = make_float2(o[n][0] * i0, o[n][1] * i0);\n"
    "        if (r1 < Sq) *(float2*)(O + qoff + (size_t)r1 * D + col) = make_float2(o[n][2] * i1, o[n][3] * i1); }\n"
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
    "// ── Linear attention, chunked (head dim <= 128) ─────────────────────────────\n"
    "// Keys are split into 64-row chunks c, and the chunks into P parts of 'cpp'\n"
    "// consecutive chunks. Three passes:\n"
    "//   lin_hstate:  one block per (part, kv head) walks its part's chunks in\n"
    "//                order with the D x D state in registers (8x8 per thread at\n"
    "//                head dim 128). Causal: before adding chunk c it stores the\n"
    "//                state so far (the part's chunks before c) in W[c], Zc[c].\n"
    "//                At the end it stores the part's total in T[p], ZT[p].\n"
    "//   lin_cscan    over the P part totals: causal, T[p] := sum of T over the\n"
    "//                parts before p; non-causal, T[0] := sum of all T.\n"
    "//   lin_hout:    one block per 64-row chunk of the output and query head:\n"
    "//                causal: S = W[c] + T[c / cpp] and\n"
    "//                out = (A V_c + phi(Q_c) S) / (rowsum A + phi(Q_c) . z) with\n"
    "//                A = phi(Q_c) phi(K_c)^T masked to j <= i;\n"
    "//                non-causal: S = T[0], out = phi(Q_c) S / phi(Q_c) . z.\n"
    "// Register tiles fed by 2- or 4-wide shared-memory loads (4x8 outputs per\n"
    "// thread at head dim 128); phi(Q) and phi(K) are stored transposed for the\n"
    "// products. The per-chunk states stay in fp32 global memory between passes.\n"
    "template <int D> struct LinChunk { static constexpr int value = 64; };\n"
    "\n"
    "// grid (ceil((D*D + D) / 256), batch*kvHeads): one state element per thread, walked over\n"
    "// chunks 8 at a time (the 8 loads are issued together, so they overlap).\n"
    "template <int D>\n"
    "__device__ void lin_cscan(float* __restrict__ W, float* __restrict__ Zc, int nc, int causal) {\n"
    "    const int kvh = blockIdx.y, e = blockIdx.x * 256 + threadIdx.x;\n"
    "    if (e >= D * D + D) return;\n"
    "    float* p;\n"
    "    size_t stride;\n"
    "    if (e < D * D) { p = W + (size_t)kvh * nc * D * D + e; stride = (size_t)D * D; }\n"
    "    else { p = Zc + (size_t)kvh * nc * D + (e - D * D); stride = D; }\n"
    "    float run = 0.0f;\n"
    "    for (int c0 = 0; c0 < nc; c0 += 8) {\n"
    "        float v[8];\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < 8; ++u) v[u] = c0 + u < nc ? p[(size_t)(c0 + u) * stride] : 0.0f;\n"
    "        if (causal) {\n"
    "#pragma unroll\n"
    "            for (int u = 0; u < 8; ++u)\n"
    "                if (c0 + u < nc) { p[(size_t)(c0 + u) * stride] = run; run += v[u]; }\n"
    "        } else {\n"
    "#pragma unroll\n"
    "            for (int u = 0; u < 8; ++u) run += v[u];\n"
    "        }\n"
    "    }\n"
    "    if (!causal) p[0] = run;               // total in chunk 0's slot\n"
    "}\n"
    "\n"
    "template <int N> __device__ __forceinline__ void lds(float* r, const float* p) {\n"
    "    if (N == 4) { float4 x = *(const float4*)p; r[0] = x.x; r[1] = x.y; r[2] = x.z; r[3] = x.w; }\n"
    "    else if (N == 8) { lds<4>(r, p); lds<4>(r + 4, p + 4); }\n"
    "    else if (N == 2) { float2 x = *(const float2*)p; r[0] = x.x; r[1] = x.y; }\n"
    "    else r[0] = p[0];\n"
    "}\n"
    "template <int N> __device__ __forceinline__ void sts(float* p, const float* r) {\n"
    "    if (N == 4) *(float4*)p = make_float4(r[0], r[1], r[2], r[3]);\n"
    "    else if (N == 2) *(float2*)p = make_float2(r[0], r[1]);\n"
    "    else p[0] = r[0];\n"
    "}\n"
    "// 4 consecutive elements at element offset i (a multiple of 4) as floats\n"
    "template <int T> __device__ __forceinline__ float4 ld4(const void* p, size_t i) {\n"
    "    if (T == 0) return *(const float4*)((const float*)p + i);\n"
    "    uint2 u = *(const uint2*)((const unsigned short*)p + i);\n"
    "    unsigned short h[4] = {(unsigned short)(u.x & 0xffffu), (unsigned short)(u.x >> 16), (unsigned short)(u.y & 0xffffu), (unsigned short)(u.y >> 16)};\n"
    "    return T == 1 ? make_float4(h2f(h[0]), h2f(h[1]), h2f(h[2]), h2f(h[3])) : make_float4(b2f(h[0]), b2f(h[1]), b2f(h[2]), b2f(h[3]));\n"
    "}\n"
    "// Columns of an N-wide slice owned by thread tx of 16: NV groups of VW at g * 16 * VW + tx * VW.\n"
    "template <int N> struct Cols { static constexpr int per = N / 16, VW = per % 4 == 0 ? 4 : 2, NV = per / VW; };\n"
    "\n"
    "template <int T, int D>\n"
    "__device__ void lin_hstate(const void* __restrict__ K, const void* __restrict__ V, float* __restrict__ W,\n"
    "                           float* __restrict__ Zc, float* __restrict__ Tt, float* __restrict__ ZT,\n"
    "                           int Sk, int nc, int cpp, int causal) {\n"
    "    constexpr int LC = LinChunk<D>::value, TL = LC * D / 4 / 256;\n"
    "    typedef Cols<D> CC;   // thread (ty, tx) owns state rows CC-mapped by ty and columns by tx\n"
    "    extern __shared__ __align__(16) float sm[];\n"
    "    float* Kf = sm;                 // LC x D    phi(K_c)\n"
    "    float* Vs = Kf + LC * D;        // LC x D    V_c\n"
    "    const int p = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;\n"
    "    const size_t koff = (size_t)kvh * Sk * D;\n"
    "    const int cb = p * cpp, ce = min(nc, cb + cpp);\n"
    "    float acc[CC::per][CC::per], z = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int a = 0; a < CC::per; ++a)\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < CC::per; ++b) acc[a][b] = 0.0f;\n"
    "    auto store = [&](float* dst) {\n"
    "#pragma unroll\n"
    "        for (int ga = 0; ga < CC::NV; ++ga)\n"
    "#pragma unroll\n"
    "            for (int a = 0; a < CC::VW; ++a) {\n"
    "                int row = ga * 16 * CC::VW + ty * CC::VW + a;\n"
    "#pragma unroll\n"
    "                for (int gb = 0; gb < CC::NV; ++gb)\n"
    "                    sts<CC::VW>(dst + (size_t)row * D + gb * 16 * CC::VW + tx * CC::VW, &acc[ga * CC::VW + a][gb * CC::VW]);\n"
    "            }\n"
    "    };\n"
    "    for (int c = cb; c < ce; ++c) {\n"
    "        float4 rk[TL], rv[TL];\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < TL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4, gk = c * LC + r;\n"
    "            bool ok = gk < Sk;\n"
    "            rk[u] = ok ? ld4<T>(K, koff + (size_t)gk * D + c4) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "            rv[u] = ok ? ld4<T>(V, koff + (size_t)gk * D + c4) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "        }\n"
    "        if (causal) {\n"
    "            store(W + ((size_t)kvh * nc + c) * D * D);\n"
    "            if (tid < D) Zc[((size_t)kvh * nc + c) * D + tid] = z;\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < TL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4;\n"
    "            bool ok = c * LC + r < Sk;\n"
    "            float4 x = rk[u];\n"
    "            *(float4*)(Kf + r * D + c4) = ok ? make_float4(phi(x.x), phi(x.y), phi(x.z), phi(x.w)) : x;\n"
    "            *(float4*)(Vs + r * D + c4) = rv[u];\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll 2\n"
    "        for (int j = 0; j < LC; ++j) {\n"
    "            float kr[CC::per], vr[CC::per];\n"
    "#pragma unroll\n"
    "            for (int g = 0; g < CC::NV; ++g) lds<CC::VW>(kr + g * CC::VW, Kf + j * D + g * 16 * CC::VW + ty * CC::VW);\n"
    "#pragma unroll\n"
    "            for (int g = 0; g < CC::NV; ++g) lds<CC::VW>(vr + g * CC::VW, Vs + j * D + g * 16 * CC::VW + tx * CC::VW);\n"
    "#pragma unroll\n"
    "            for (int a = 0; a < CC::per; ++a)\n"
    "#pragma unroll\n"
    "                for (int b = 0; b < CC::per; ++b) acc[a][b] += kr[a] * vr[b];\n"
    "        }\n"
    "        if (tid < D) {\n"
    "            float s = 0.0f;\n"
    "            for (int j = 0; j < LC; ++j) s += Kf[j * D + tid];\n"
    "            z += s;\n"
    "        }\n"
    "    }\n"
    "    store(Tt + ((size_t)kvh * gridDim.x + p) * D * D);\n"
    "    if (tid < D) ZT[((size_t)kvh * gridDim.x + p) * D + tid] = z;\n"
    "}\n"
    "\n"
    "template <int T, int D>\n"
    "__device__ void lin_hout(const void* __restrict__ Q, const void* __restrict__ K, const void* __restrict__ V,\n"
    "                         const float* __restrict__ W, const float* __restrict__ Zc, const float* __restrict__ Tt,\n"
    "                         const float* __restrict__ ZT, void* __restrict__ O, int Sq, int Sk, int H, int Hkv,\n"
    "                         int nc, int cpp, int P, int causal) {\n"
    "    constexpr int LC = LinChunk<D>::value, LDT = LC + 4, RW = LC / 16, SR = 32, NS = D / SR;\n"
    "    constexpr int TL = LC * D / 4 / 256;      // float4s of a chunk tile per thread\n"
    "    constexpr int SL = SR * D / 4 / 256;      // float4s of a state slab per thread\n"
    "    constexpr int NT = LC / RW;               // A is NT x NT tiles of RW x RW; the lower triangle is computed\n"
    "    typedef Cols<D> CC;\n"
    "    extern __shared__ __align__(16) float sm[];\n"
    "    float* Qt = sm;                 // D x LDT   phi(Q_c)^T\n"
    "    float* Kt = Qt + D * LDT;       // D x LDT   phi(K_c)^T, then V_c (LC x D), then two state slabs (SR x D)\n"
    "    float* At = Kt + D * LDT;       // LC x LDT  A^T          (causal)\n"
    "    float* Zs = causal ? At + LC * LDT : Kt + 2 * SR * D;   // D\n"
    "    const int c = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H, kvh = b * Hkv + h / (H / Hkv);\n"
    "    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;\n"
    "    const int off = Sk - Sq;\n"
    "    const int qstart = causal ? c * LC - off : c * LC;   // query row of local row 0\n"
    "    if (qstart + LC <= 0 || qstart >= Sq) return;\n"
    "    const size_t qoff = (size_t)bh * Sq * D, koff = (size_t)kvh * Sk * D;\n"
    "    // the state: causal W[c] plus the parts before c's (prefix over parts); non-causal the total\n"
    "    const int pc = causal ? c / cpp : 0;\n"
    "    const float* ws = causal ? W + ((size_t)kvh * nc + c) * D * D : (const float*)0;\n"
    "    const float* ts = Tt + ((size_t)kvh * P + pc) * D * D;\n"
    "    // issue the global loads first: phi(Q_c) (and phi(K_c)) tiles and the first state slab\n"
    "    float4 rq[TL], rk[TL], rs[SL];\n"
    "    int tr[TL], td[TL];\n"
    "#pragma unroll\n"
    "    for (int u = 0; u < TL; ++u) {\n"
    "        // each warp reads 8 rows x 16 consecutive elements, stored transposed with 2-way bank conflicts\n"
    "        int i = tid + 256 * u, w = i >> 5, l = i & 31;\n"
    "        tr[u] = (w % (LC / 8)) * 8 + (l & 7); td[u] = ((w / (LC / 8)) * 4 + (l >> 3)) * 4;\n"
    "        int gq = qstart + tr[u], gk = c * LC + tr[u];\n"
    "        rq[u] = gq >= 0 && gq < Sq ? ld4<T>(Q, qoff + (size_t)gq * D + td[u]) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "        rk[u] = causal && gk < Sk ? ld4<T>(K, koff + (size_t)gk * D + td[u]) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "    }\n"
    "    auto load_slab = [&](int d0) {\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < SL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4;\n"
    "            size_t e = (size_t)(d0 + r) * D + c4;\n"
    "            float4 x = *(const float4*)(ts + e);\n"
    "            if (causal) { float4 y = *(const float4*)(ws + e); x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w; }\n"
    "            rs[u] = x;\n"
    "        }\n"
    "    };\n"
    "    auto store_slab = [&](float* dst) {\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < SL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4;\n"
    "            *(float4*)(dst + r * D + c4) = rs[u];\n"
    "        }\n"
    "    };\n"
    "    load_slab(0);\n"
    "    for (int i = tid; i < D; i += 256) {\n"
    "        float zz = ZT[((size_t)kvh * P + pc) * D + i];\n"
    "        Zs[i] = causal ? zz + Zc[((size_t)kvh * nc + c) * D + i] : zz;\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int u = 0; u < TL; ++u) {\n"
    "        float4 x = rq[u]; int r = tr[u], d = td[u];\n"
    "        bool okq = qstart + r >= 0 && qstart + r < Sq;\n"
    "        Qt[(d + 0) * LDT + r] = okq ? phi(x.x) : 0.0f; Qt[(d + 1) * LDT + r] = okq ? phi(x.y) : 0.0f;\n"
    "        Qt[(d + 2) * LDT + r] = okq ? phi(x.z) : 0.0f; Qt[(d + 3) * LDT + r] = okq ? phi(x.w) : 0.0f;\n"
    "        if (causal) {\n"
    "            float4 y = rk[u]; bool okk = c * LC + r < Sk;\n"
    "            Kt[(d + 0) * LDT + r] = okk ? phi(y.x) : 0.0f; Kt[(d + 1) * LDT + r] = okk ? phi(y.y) : 0.0f;\n"
    "            Kt[(d + 2) * LDT + r] = okk ? phi(y.z) : 0.0f; Kt[(d + 3) * LDT + r] = okk ? phi(y.w) : 0.0f;\n"
    "        }\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float acc[RW][CC::per], den[RW];\n"
    "#pragma unroll\n"
    "    for (int a = 0; a < RW; ++a) {\n"
    "        den[a] = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int e = 0; e < CC::per; ++e) acc[a][e] = 0.0f;\n"
    "    }\n"
    "    if (causal) {\n"
    "        // V_c into registers while A = phi(Q_c) phi(K_c)^T is computed: one RW x RW tile of the\n"
    "        // lower triangle (j <= i) per thread for the first NT (NT + 1) / 2 threads\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < TL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4, gk = c * LC + r;\n"
    "            rk[u] = gk < Sk ? ld4<T>(V, koff + (size_t)gk * D + c4) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "        }\n"
    "        if (tid < NT * (NT + 1) / 2) {\n"
    "            int ti = (int)((sqrtf(8.0f * tid + 1.0f) - 1.0f) * 0.5f);\n"
    "            if ((ti + 1) * (ti + 2) / 2 <= tid) ++ti;\n"
    "            if (ti * (ti + 1) / 2 > tid) --ti;\n"
    "            const int tj = tid - ti * (ti + 1) / 2;\n"
    "            float s[RW][RW];\n"
    "#pragma unroll\n"
    "            for (int a = 0; a < RW; ++a)\n"
    "#pragma unroll\n"
    "                for (int e = 0; e < RW; ++e) s[a][e] = 0.0f;\n"
    "#pragma unroll 4\n"
    "            for (int d = 0; d < D; ++d) {\n"
    "                float qv[RW], kv[RW];\n"
    "                lds<RW>(qv, Qt + d * LDT + ti * RW);\n"
    "                lds<RW>(kv, Kt + d * LDT + tj * RW);\n"
    "#pragma unroll\n"
    "                for (int a = 0; a < RW; ++a)\n"
    "#pragma unroll\n"
    "                    for (int e = 0; e < RW; ++e) s[a][e] += qv[a] * kv[e];\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int e = 0; e < RW; ++e) {\n"
    "                float col[RW];\n"
    "#pragma unroll\n"
    "                for (int a = 0; a < RW; ++a) col[a] = (tj * RW + e <= ti * RW + a) ? s[a][e] : 0.0f;\n"
    "                sts<RW>(At + (tj * RW + e) * LDT + ti * RW, col);\n"
    "            }\n"
    "        }\n"
    "        __syncthreads();\n"
    "        float* Vs = Kt;\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < TL; ++u) {\n"
    "            int i = tid + 256 * u, r = i / (D / 4), c4 = (i - r * (D / 4)) * 4;\n"
    "            *(float4*)(Vs + r * D + c4) = rk[u];\n"
    "        }\n"
    "        __syncthreads();\n"
    "        // A^T rows above the diagonal tile (j > i for all of this thread's rows) are zero and skipped\n"
    "        const int jend = ty * RW + RW;\n"
    "        for (int j = 0; j < jend; ++j) {\n"
    "            float av[RW], vv[CC::per];\n"
    "            lds<RW>(av, At + j * LDT + ty * RW);\n"
    "#pragma unroll\n"
    "            for (int g = 0; g < CC::NV; ++g) lds<CC::VW>(vv + g * CC::VW, Vs + j * D + g * 16 * CC::VW + tx * CC::VW);\n"
    "#pragma unroll\n"
    "            for (int a = 0; a < RW; ++a) {\n"
    "                den[a] += av[a];\n"
    "#pragma unroll\n"
    "                for (int e = 0; e < CC::per; ++e) acc[a][e] += av[a] * vv[e];\n"
    "            }\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    // acc += phi(Q_c) S, den += phi(Q_c) . z, S through two SR-row slab buffers (prefetched in registers)\n"
    "    float* Sb = Kt;\n"
    "    store_slab(Sb);\n"
    "    __syncthreads();\n"
    "#pragma unroll 1\n"
    "    for (int k = 0; k < NS; ++k) {\n"
    "        const float* sb = Sb + (k & 1) * SR * D;\n"
    "        if (k + 1 < NS) load_slab((k + 1) * SR);\n"
    "#pragma unroll 4\n"
    "        for (int dd = 0; dd < SR; ++dd) {\n"
    "            const int d = k * SR + dd;\n"
    "            float qv[RW], sv[CC::per];\n"
    "            const float zd = Zs[d];\n"
    "            lds<RW>(qv, Qt + d * LDT + ty * RW);\n"
    "#pragma unroll\n"
    "            for (int g = 0; g < CC::NV; ++g) lds<CC::VW>(sv + g * CC::VW, sb + dd * D + g * 16 * CC::VW + tx * CC::VW);\n"
    "#pragma unroll\n"
    "            for (int a = 0; a < RW; ++a) {\n"
    "                den[a] += qv[a] * zd;\n"
    "#pragma unroll\n"
    "                for (int e = 0; e < CC::per; ++e) acc[a][e] += qv[a] * sv[e];\n"
    "            }\n"
    "        }\n"
    "        if (k + 1 < NS) { store_slab(Sb + ((k + 1) & 1) * SR * D); __syncthreads(); }\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int a = 0; a < RW; ++a) {\n"
    "        int gq = qstart + ty * RW + a;\n"
    "        if (gq < 0 || gq >= Sq) continue;\n"
    "        float inv = den[a] != 0.0f ? 1.0f / den[a] : 0.0f;\n"
    "#pragma unroll\n"
    "        for (int g = 0; g < CC::NV; ++g)\n"
    "#pragma unroll\n"
    "            for (int e = 0; e < CC::VW; ++e)\n"
    "                stv<T>(O, qoff + (size_t)gq * D + g * 16 * CC::VW + tx * CC::VW + e, acc[a][g * CC::VW + e] * inv);\n"
    "    }\n"
    "}\n"
    "\n"
    "// ── Decode attention (flash-decoding): a few query rows against a long KV cache ──\n"
    "// One block per (batch, kv head, key split): the R = (heads / kvHeads) * seqLen\n"
    "// query rows that share a kv head read each key and value once. Scores are\n"
    "// four threads per key (32-key tiles, q pre-scaled in shared memory), then an online\n"
    "// softmax per row, then P V with each thread on two dims of one key group.\n"
    "// Each split writes its unnormalized output, max and sum; dec_combine merges.\n"
    "template <int D> struct DecCfg {\n"
    "    static constexpr int W = D / 2;                     // threads per key group (two dims each)\n"
    "    static constexpr int KG = D >= 256 ? 1 : 128 / W;   // key groups\n"
    "};\n"
    "template <int T> __device__ __forceinline__ float2 ld2(const void* p, size_t i) {\n"
    "    if (T == 0) return __ldg((const float2*)((const float*)p + i));\n"
    "    const unsigned w = __ldg((const unsigned*)((const unsigned short*)p + i));\n"
    "    if (T == 1) return make_float2(h2f((unsigned short)(w & 0xffffu)), h2f((unsigned short)(w >> 16)));\n"
    "    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));\n"
    "}\n"
    "// Raw 8-element loads (16 bytes of fp16/bf16, 32 of fp32) and their conversion,\n"
    "// so a tile's keys can be fetched while the previous tile is still in use.\n"
    "template <int T> struct Raw8 { uint4 a, b; };\n"
    "template <int T> __device__ __forceinline__ Raw8<T> ldraw8(const void* p, size_t i) {\n"
    "    Raw8<T> r;\n"
    "    if (T == 0) { r.a = __ldg((const uint4*)((const float*)p + i)); r.b = __ldg((const uint4*)((const float*)p + i + 4)); }\n"
    "    else r.a = __ldg((const uint4*)((const unsigned short*)p + i));\n"
    "    return r;\n"
    "}\n"
    "template <int T> __device__ __forceinline__ void cvt8(const Raw8<T>& r, float* f) {\n"
    "    if (T == 0) {\n"
    "        f[0] = __uint_as_float(r.a.x); f[1] = __uint_as_float(r.a.y); f[2] = __uint_as_float(r.a.z); f[3] = __uint_as_float(r.a.w);\n"
    "        f[4] = __uint_as_float(r.b.x); f[5] = __uint_as_float(r.b.y); f[6] = __uint_as_float(r.b.z); f[7] = __uint_as_float(r.b.w);\n"
    "    } else {\n"
    "        const unsigned w[4] = {r.a.x, r.a.y, r.a.z, r.a.w};\n"
    "#pragma unroll\n"
    "        for (int e = 0; e < 4; ++e) {\n"
    "            f[2 * e] = T == 1 ? h2f((unsigned short)(w[e] & 0xffffu)) : __uint_as_float(w[e] << 16);\n"
    "            f[2 * e + 1] = T == 1 ? h2f((unsigned short)(w[e] >> 16)) : __uint_as_float(w[e] & 0xffff0000u);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "template <int T, int D, int RM> __device__ void dec_part(const void* q, const void* k, const void* v, float* Op, float* Ml,\n"
    "                                                       int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win,\n"
    "                                                       int chunk) {\n"
    "    typedef DecCfg<D> C;\n"
    "    __shared__ __align__(16) float qs[RM][D];\n"
    "    __shared__ float ps[RM][32];\n"
    "    __shared__ float red[C::KG > 1 ? RM * C::KG * D : 1];\n"
    "    __shared__ float sm_m[RM], sm_l[RM], sm_a[RM];\n"
    "    const int tid = threadIdx.x, ns = gridDim.x, split = blockIdx.x, bk = blockIdx.y;\n"
    "    const int b = bk / Hkv, kh = bk - b * Hkv, G = H / Hkv, R = G * Sq;\n"
    "    for (int e = tid; e < R * D; e += 128) {\n"
    "        const int r = e / D, d = e - r * D, g = r / Sq, i = r - g * Sq;\n"
    "        qs[r][d] = ldv<T>(q, (((size_t)b * H + kh * G + g) * Sq + i) * D + d) * sl2;\n"
    "    }\n"
    "    if (tid < RM) { sm_m[tid] = NEG_INF; sm_l[tid] = 0.0f; }\n"
    "    const size_t kvoff = (size_t)bk * Sk * D;\n"
    "    const int j0 = split * chunk, j1 = min(Sk, j0 + chunk);\n"
    "    const int jk = tid >> 2, pp = tid & 3;                 // scores: four threads per key\n"
    "    const int kg = tid / C::W, d0 = (tid - kg * C::W) * 2;   // P V: two dims of one key group\n"
    "    const bool pv = kg < C::KG;\n"
    "    const int warp = tid >> 5, lane = tid & 31;\n"
    "    float acc[RM][2];\n"
    "#pragma unroll\n"
    "    for (int r = 0; r < RM; ++r) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }\n"
    "    __syncthreads();\n"
    "    // 8-dim slices round-robin over the four threads: the q reads hit four\n"
    "    // consecutive 32-byte blocks (no bank conflicts), the k reads 64 bytes.\n"
    "    // The next tile's keys are loaded right after this tile's scores.\n"
    "    constexpr int NC = D / 32;\n"
    "    Raw8<T> kr[NC];\n"
    "    if (j0 + jk < j1)\n"
    "#pragma unroll\n"
    "        for (int c = 0; c < NC; ++c) kr[c] = ldraw8<T>(k, kvoff + (size_t)(j0 + jk) * D + pp * 8 + c * 32);\n"
    "    for (int t0 = j0; t0 < j1; t0 += 32) {\n"
    "        const int j = t0 + jk;\n"
    "        float s[RM];\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < RM; ++r) s[r] = 0.0f;\n"
    "        if (j < j1) {\n"
    "#pragma unroll\n"
    "            for (int cc = 0; cc < NC; ++cc) {\n"
    "                const int c = pp * 8 + cc * 32;\n"
    "                float kf[8];\n"
    "                cvt8<T>(kr[cc], kf);\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < RM; ++r)\n"
    "                    if (r < R) {\n"
    "                        const float4 a = *(const float4*)&qs[r][c], e = *(const float4*)&qs[r][c + 4];\n"
    "                        s[r] += a.x * kf[0] + a.y * kf[1] + a.z * kf[2] + a.w * kf[3] +\n"
    "                                e.x * kf[4] + e.y * kf[5] + e.z * kf[6] + e.w * kf[7];\n"
    "                    }\n"
    "            }\n"
    "        }\n"
    "        if (j + 32 < j1)\n"
    "#pragma unroll\n"
    "            for (int c = 0; c < NC; ++c) kr[c] = ldraw8<T>(k, kvoff + (size_t)(j + 32) * D + pp * 8 + c * 32);\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < RM; ++r) {\n"
    "            s[r] += __shfl_xor_sync(FULL, s[r], 1);\n"
    "            s[r] += __shfl_xor_sync(FULL, s[r], 2);\n"
    "        }\n"
    "        if (pp == 0)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < RM; ++r)\n"
    "                if (r < R) {\n"
    "                    const int pos = r % Sq + Sk - Sq;\n"
    "                    const bool ok = j < j1 && (!causal || j <= pos) && (win <= 0 || j > pos - win);\n"
    "                    ps[r][jk] = ok ? s[r] : NEG_INF;\n"
    "                }\n"
    "        __syncthreads();\n"
    "        for (int r = warp; r < R; r += 4) {\n"
    "            float x = ps[r][lane];\n"
    "            float mt = x;\n"
    "            for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(FULL, mt, off));\n"
    "            const float mo = sm_m[r], mn = fmaxf(mo, mt);\n"
    "            float a = 1.0f;\n"
    "            if (mn != NEG_INF) { a = exp2f(mo - mn); x = exp2f(x - mn); } else x = 0.0f;\n"
    "            ps[r][lane] = x;\n"
    "            float sum = x;\n"
    "            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(FULL, sum, off);\n"
    "            if (lane == 0) { sm_m[r] = mn; sm_l[r] = sm_l[r] * a + sum; sm_a[r] = a; }\n"
    "        }\n"
    "        __syncthreads();\n"
    "        if (pv) {\n"
    "            const int tl = min(32, j1 - t0);\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < RM; ++r) if (r < R) { acc[r][0] *= sm_a[r]; acc[r][1] *= sm_a[r]; }\n"
    "#pragma unroll 4\n"
    "            for (int jj = kg; jj < tl; jj += C::KG) {\n"
    "                const float2 vv = ld2<T>(v, kvoff + (size_t)(t0 + jj) * D + d0);\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < RM; ++r)\n"
    "                    if (r < R) { const float p = ps[r][jj]; acc[r][0] += p * vv.x; acc[r][1] += p * vv.y; }\n"
    "            }\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    const size_t base = (size_t)bk * R;\n"
    "    if (C::KG > 1) {\n"
    "        if (pv)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < RM; ++r)\n"
    "                if (r < R) { red[(kg * RM + r) * D + d0] = acc[r][0]; red[(kg * RM + r) * D + d0 + 1] = acc[r][1]; }\n"
    "        __syncthreads();\n"
    "        for (int e = tid; e < R * D; e += 128) {\n"
    "            const int r = e / D, d = e - r * D;\n"
    "            float t = 0.0f;\n"
    "#pragma unroll\n"
    "            for (int g = 0; g < C::KG; ++g) t += red[(g * RM + r) * D + d];\n"
    "            Op[((base + r) * ns + split) * D + d] = t;\n"
    "        }\n"
    "    } else if (pv) {\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < RM; ++r)\n"
    "            if (r < R) {\n"
    "                Op[((base + r) * ns + split) * D + d0] = acc[r][0];\n"
    "                Op[((base + r) * ns + split) * D + d0 + 1] = acc[r][1];\n"
    "            }\n"
    "    }\n"
    "    if (tid < R) { Ml[((base + tid) * ns + split) * 2] = sm_m[tid]; Ml[((base + tid) * ns + split) * 2 + 1] = sm_l[tid]; }\n"
    "}\n"
    "// One block per (query row, 32 dims): out = sum_s 2^(m_s - M) O_s / sum_s 2^(m_s - M) l_s\n"
    "// (0 if no key is visible); lanes on dims, the four warps on every fourth split.\n"
    "template <int T, int D> __device__ void dec_combine(const float* Op, const float* Ml, void* o, int Sq, int H, int Hkv, int ns) {\n"
    "    __shared__ float wts[1024];\n"
    "    __shared__ float rs[4];\n"
    "    __shared__ float part[4][32];\n"
    "    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, row = blockIdx.x, d = blockIdx.y * 32 + lane;\n"
    "    const int G = H / Hkv, R = G * Sq, bk = row / R, r = row - bk * R, b = bk / Hkv, kh = bk - b * Hkv;\n"
    "    const int g = r / Sq, i = r - g * Sq;\n"
    "    const float* ml = Ml + (size_t)row * ns * 2;\n"
    "    float m = NEG_INF;\n"
    "    for (int s = tid; s < ns; s += 128) m = fmaxf(m, ml[2 * s]);\n"
    "    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, off));\n"
    "    if (lane == 0) rs[warp] = m;\n"
    "    __syncthreads();\n"
    "    m = fmaxf(fmaxf(rs[0], rs[1]), fmaxf(rs[2], rs[3]));\n"
    "    __syncthreads();\n"
    "    float l = 0.0f;\n"
    "    for (int s = tid; s < ns; s += 128) {\n"
    "        const float ms = ml[2 * s];\n"
    "        const float w = ms == NEG_INF ? 0.0f : exp2f(ms - m);\n"
    "        wts[s] = w;\n"
    "        l += w * ml[2 * s + 1];\n"
    "    }\n"
    "    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(FULL, l, off);\n"
    "    if (lane == 0) rs[warp] = l;\n"
    "    __syncthreads();\n"
    "    l = (rs[0] + rs[1]) + (rs[2] + rs[3]);\n"
    "    float t = 0.0f;\n"
    "    if (d < D) {\n"
    "        const float* op = Op + (size_t)row * ns * D + d;\n"
    "#pragma unroll 8\n"
    "        for (int s = warp; s < ns; s += 4) t += wts[s] * op[(size_t)s * D];\n"
    "    }\n"
    "    part[warp][lane] = t;\n"
    "    __syncthreads();\n"
    "    if (warp == 0 && d < D) {\n"
    "        t = (part[0][lane] + part[1][lane]) + (part[2][lane] + part[3][lane]);\n"
    "        stv<T>(o, (((size_t)b * H + kh * G + g) * Sq + i) * D + d, l > 0.0f ? t / l : 0.0f);\n"
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
    "#elif VK == 6\n"
    "extern \"C\" __global__ __launch_bounds__(256, 2) void attn_entry(const void* k, const void* v, float* W, float* Zc, float* Tt,\n"
    "                                                               float* ZT, int Sk, int nc, int cpp, int causal) {\n"
    "    lin_hstate<VT, VD>(k, v, W, Zc, Tt, ZT, Sk, nc, cpp, causal);\n"
    "}\n"
    "#elif VK == 7\n"
    "extern \"C\" __global__ __launch_bounds__(256) void attn_entry(float* W, float* Zc, int nc, int causal) {\n"
    "    lin_cscan<VD>(W, Zc, nc, causal);\n"
    "}\n"
    "#elif VK == 8\n"
    "extern \"C\" __global__ __launch_bounds__(256, 2) void attn_entry(const void* q, const void* k, const void* v, const float* W,\n"
    "                                                               const float* Zc, const float* Tt, const float* ZT, void* o,\n"
    "                                                               int Sq, int Sk, int H, int Hkv, int nc, int cpp, int P,\n"
    "                                                               int causal) {\n"
    "    lin_hout<VT, VD>(q, k, v, W, Zc, Tt, ZT, o, Sq, Sk, H, Hkv, nc, cpp, P, causal);\n"
    "}\n"
    "#elif VK == 9\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const void* q, const void* k, const void* v, void* o,\n"
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win) {\n"
    "#if __CUDA_ARCH__ >= 800\n"
    "    flash_tf32x3<VD>((const float*)q, (const float*)k, (const float*)v, (float*)o, Sq, Sk, H, Hkv, sl2, causal, win);\n"
    "#endif\n"
    "}\n"
    "#elif VK >= 10 && VK <= 14\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const void* q, const void* k, const void* v, float* Op, float* Ml,\n"
    "                                                            int Sq, int Sk, int H, int Hkv, float sl2, int causal, int win,\n"
    "                                                            int chunk) {\n"
    "    dec_part<VT, VD, 1 << (VK - 10)>(q, k, v, Op, Ml, Sq, Sk, H, Hkv, sl2, causal, win, chunk);\n"
    "}\n"
    "#elif VK == 15\n"
    "extern \"C\" __global__ __launch_bounds__(128) void attn_entry(const float* Op, const float* Ml, void* o, int Sq, int H, int Hkv,\n"
    "                                                            int ns) {\n"
    "    dec_combine<VT, VD>(Op, Ml, o, Sq, H, Hkv, ns);\n"
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

// lin_hout's dynamic shared memory: causal phi(Q)^T, phi(K)^T (reused for V and the state
// slabs), A^T and z; non-causal phi(Q)^T, two 32-row state slabs and z.
static unsigned long attn_hout_smem_(unsigned long d, int causal) {
    unsigned long ldt = ATT_LC + 4UL;
    if (causal) return (2UL * d * ldt + ATT_LC * ldt + d) * 4UL;
    return (d * ldt + 64UL * d + d) * 4UL;
}

// Dynamic shared memory each kernel needs (must match the .cu layouts).
static unsigned long attn_smem_(int kind, unsigned long d) {
    if (kind == ATT_SIMT) return (32UL * d + 32UL * (d + 1UL) + 32UL * d + 4UL * 8UL * 32UL) * 4UL;
    if (kind == ATT_LIN_KV) return (32UL * 32UL + 32UL * d) * 4UL;
    if (kind == ATT_LIN_OUT) {
        unsigned long cs = d % 64UL == 0UL ? 64UL : 32UL;
        return (d * cs + d + 32UL * d) * 4UL;
    }
    if (kind == ATT_LIN_CAUSAL) return (32UL * d + 32UL * (d + 1UL) + 32UL * 32UL + d * 32UL + d + 32UL * 33UL) * 4UL;
    if (kind == ATT_LIN_HSTATE) return 2UL * ATT_LC * d * 4UL;
    if (kind == ATT_LIN_HOUT) return attn_hout_smem_(d, 1);
    if (kind == ATT_TF32X3) {
        unsigned long bn = d > 64UL ? 32UL : 64UL;
        return ((64UL + 2UL * bn) * (d + 4UL) + 4UL * 16UL * (bn + 4UL)) * 4UL;
    }
    return 0UL;   // ATT_MMA, ATT_LIN_CSCAN: static shared memory only
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
#define SD_SLOTS             64
#define SD_DESCS             128  // descriptors kept alive per plan (a masked graph uses ~80)
#define SD_CFGS              8
#define SD_MAX_WS            268435456UL   // 256 MiB

static int   g_sd_state = 0;      // 0 = not tried, 1 = ready, -1 = unavailable
static int   g_sd_enabled = 1;    // cuda_attention_use_cudnn
static int   g_att_decode = 1;    // cuda_attention_use_decode
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
static int   g_sd_next = 0;                 // next slot to evict when the cache is full

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

int cuda_attention_use_decode(int on) {
    g_att_decode = on ? 1 : 0;
    return g_att_decode;
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

// Frees slot s's plan, variant pack, descriptors and workspace.
static void sd_evict_(int s) {
    unsafe {
        cuStreamSynchronize((void*)cuda_stream_handle());
        if (g_sd_vp[s] != (void*)0) p_destroy(g_sd_vp[s]);
        if (g_sd_plan[s] != (void*)0) p_destroy(g_sd_plan[s]);
        int i = g_sd_ndesc[s] - 1;
        while (i >= 0) { p_destroy(g_sd_desc[s * SD_DESCS + i]); i = i - 1; }
        if (g_sd_ws[s] != 0ULL) cuMemFree_v2(g_sd_ws[s]);
    }
    g_sd_vp[s] = (void*)0;
    g_sd_plan[s] = (void*)0;
    g_sd_ndesc[s] = 0;
    g_sd_ws[s] = 0ULL;
    g_sd_ok[s] = 0;
    g_sd_ptr[s * 4] = 0UL; g_sd_ptr[s * 4 + 1] = 0UL; g_sd_ptr[s * 4 + 2] = 0UL; g_sd_ptr[s * 4 + 3] = 0UL;
    int j = 0;
    while (j < 9) { g_sd_key[s * 9 + j] = 0L; j = j + 1; }
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
        if (free < 0) {
            // Cache full: evict slots round-robin (after queued work that may use them).
            free = g_sd_next;
            g_sd_next = (g_sd_next + 1) % SD_SLOTS;
            sd_evict_(free);
        }
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

// ── Decode attention ─────────────────────────────────────────────────────────
// Flash-decoding for few query rows per kv head ((heads / kvHeads) * seqLen
// <= 16, e.g. one token per step against a KV cache): the keys are split
// over enough blocks to fill the GPU (about eight per SM, 32-key multiples),
// each split's partial softmax goes to a workspace and a second kernel merges
// them. cuDNN's and the tiled kernels' grids are one block per 32 / 64 query
// rows per head, which leaves most of the GPU idle on such shapes.
static unsigned long long g_att_dec_ws = 0ULL;
static unsigned long g_att_dec_ws_bytes = 0UL;

static int attn_decode_(unsigned long q, unsigned long k, unsigned long v, unsigned long o, int dtype,
                        struct CudaAttnShape s, float sl2, int causal) {
    unsigned long rows = (s.heads / s.kvHeads) * s.seqLen;
    if (rows > ATT_DEC_ROWS) return 0;
    if (q % 16UL != 0UL || k % 16UL != 0UL || v % 16UL != 0UL) return 0;   // 16-byte K loads
    unsigned long bk = s.batch * s.kvHeads;
    if (bk > 65535UL || bk * rows > 2147483647UL) return 0;
    int kind = ATT_DEC_PART;
    unsigned long rm = 1UL;
    while (rm < rows) { rm = rm * 2UL; kind = kind + 1; }
    unsigned long sms = (unsigned long)cuda_sm_count();
    if (sms == 0UL) sms = 1UL;
    // Splits at least 8 * rows / elementBytes keys long, so the fp32 partials
    // stay at most a quarter of the K and V bytes they summarize.
    unsigned long elem = dtype == 0 ? 4UL : 2UL;
    unsigned long minChunk = (8UL * rows / elem + 31UL) / 32UL * 32UL;
    if (minChunk < 32UL) minChunk = 32UL;
    unsigned long ns = (sms * 8UL + bk - 1UL) / bk;
    unsigned long most = (s.kvLen + minChunk - 1UL) / minChunk;
    if (ns > most) ns = most;
    if (ns > ATT_DEC_SPLITS) ns = ATT_DEC_SPLITS;
    unsigned long chunk = (s.kvLen + ns - 1UL) / ns;
    chunk = (chunk + 31UL) / 32UL * 32UL;
    ns = (s.kvLen + chunk - 1UL) / chunk;
    ?&void part = attn_kernel_(kind, dtype, s.headDim);
    ?&void comb = attn_kernel_(ATT_DEC_COMBINE, dtype, s.headDim);
    if (part == (void*)0 || comb == (void*)0) return 0;
    unsigned long prow = checked_mul_size(bk * rows, ns);
    unsigned long opBytes = checked_mul_size(prow, s.headDim * 4UL);
    unsigned long bytes = opBytes + prow * 8UL;
    unsafe {
        if (bytes > g_att_dec_ws_bytes) {
            if (g_att_dec_ws != 0ULL) {
                cuStreamSynchronize((void*)cuda_stream_handle());
                cuMemFree_v2(g_att_dec_ws);
                g_att_dec_ws = 0ULL;
                g_att_dec_ws_bytes = 0UL;
            }
            if (cuMemAlloc_v2(&g_att_dec_ws, bytes) != 0) { g_att_dec_ws = 0ULL; return 0; }
            g_att_dec_ws_bytes = bytes;
        }
        void* pq = (void*)q;
        void* pk = (void*)k;
        void* pv = (void*)v;
        void* po = (void*)o;
        void* pop = (void*)(unsigned long)g_att_dec_ws;
        void* pml = (void*)((unsigned long)g_att_dec_ws + opBytes);
        int sq = (int)s.seqLen;
        int sk = (int)s.kvLen;
        int h = (int)s.heads;
        int hkv = (int)s.kvHeads;
        float l2 = sl2;
        int c = causal ? 1 : 0;
        int win = (int)s.window;
        int ch = (int)chunk;
        int nsi = (int)ns;
        void* params[13];
        params[0] = (void*)&pq; params[1] = (void*)&pk; params[2] = (void*)&pv; params[3] = (void*)&pop;
        params[4] = (void*)&pml; params[5] = (void*)&sq; params[6] = (void*)&sk; params[7] = (void*)&h;
        params[8] = (void*)&hkv; params[9] = (void*)&l2; params[10] = (void*)&c; params[11] = (void*)&win;
        params[12] = (void*)&ch;
        if (!attn_launch_(part, (unsigned int)ns, (unsigned int)bk, 1U, 128U, 0UL, (&void)&params[0])) return 0;
        void* cparams[7];
        cparams[0] = (void*)&pop; cparams[1] = (void*)&pml; cparams[2] = (void*)&po; cparams[3] = (void*)&sq;
        cparams[4] = (void*)&h; cparams[5] = (void*)&hkv; cparams[6] = (void*)&nsi;
        return attn_launch_(comb, (unsigned int)(bk * rows), (unsigned int)((s.headDim + 31UL) / 32UL), 1U, 128U, 0UL,
                            (&void)&cparams[0]);
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
    float sl2 = sc * (float)1.4426950408889634;   // softmax in base 2
    if (g_att_decode && attn_decode_(q, k, v, o, dtype, s, sl2, causal)) return 1;
    if (dtype != 0 && cuda_compute_capability() >= 80 && sd_run_(q, k, v, o, dtype, s, sc, causal)) return 1;
    // Tensor cores on sm_80+ up to head dim 128: mma.sync in fp16/bf16, 3xTF32 for fp32.
    int tc = cuda_compute_capability() >= 80 && s.headDim <= 128UL;
    int kind = ATT_SIMT;
    if (tc) kind = dtype != 0 ? ATT_MMA : ATT_TF32X3;
    unsigned long rows = kind == ATT_SIMT ? 32UL : 64UL;
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
        if (causal && s.kvLen < s.seqLen) return 0;
        if (d <= 128UL) {
            // per-chunk states by part (lin_hstate), prefix / total over the parts, then outputs
            unsigned long nc = (s.kvLen + ATT_LC - 1UL) / ATT_LC;
            unsigned long kvTotal = s.batch * s.kvHeads;
            unsigned long parts = (ATT_LIN_BLOCKS + kvTotal - 1UL) / kvTotal;
            if (parts > nc) parts = nc;
            if (parts < 1UL) parts = 1UL;
            unsigned long cpp = (nc + parts - 1UL) / parts;
            parts = (nc + cpp - 1UL) / cpp;
            unsigned long wElems = causal ? checked_mul_size(checked_mul_size(kvTotal, nc), d * d) : 0UL;
            unsigned long zElems = causal ? kvTotal * nc * d : 0UL;
            unsigned long tElems = checked_mul_size(checked_mul_size(kvTotal, parts), d * d);
            unsigned long tzElems = kvTotal * parts * d;
            if (!attn_workspace_(checked_mul_size(wElems + zElems + tElems + tzElems, 4UL))) return 0;
            ?&void fst = attn_kernel_(ATT_LIN_HSTATE, dtype, d);
            ?&void fsc = attn_kernel_(ATT_LIN_CSCAN, dtype, d);
            ?&void fou = attn_kernel_(ATT_LIN_HOUT, dtype, d);
            if (fst == (void*)0 || fsc == (void*)0 || fou == (void*)0) return 0;
            unsigned long base = (unsigned long)g_att_ws;
            void* pW = (void*)base;
            void* pZ = (void*)(base + wElems * 4UL);
            void* pT = (void*)(base + (wElems + zElems) * 4UL);
            void* pTZ = (void*)(base + (wElems + zElems + tElems) * 4UL);
            int ncI = (int)nc;
            int cppI = (int)cpp;
            int partsI = (int)parts;
            int c = causal ? 1 : 0;
            void* p1[10];
            p1[0] = (void*)&pk; p1[1] = (void*)&pv; p1[2] = (void*)&pW; p1[3] = (void*)&pZ;
            p1[4] = (void*)&pT; p1[5] = (void*)&pTZ; p1[6] = (void*)&sk; p1[7] = (void*)&ncI;
            p1[8] = (void*)&cppI; p1[9] = (void*)&c;
            if (!attn_launch_(fst, (unsigned int)parts, (unsigned int)kvTotal, 1U, 256U,
                              attn_smem_(ATT_LIN_HSTATE, d), (&void)&p1[0])) return 0;
            void* p2[4];
            p2[0] = (void*)&pT; p2[1] = (void*)&pTZ; p2[2] = (void*)&partsI; p2[3] = (void*)&c;
            if (!attn_launch_(fsc, (unsigned int)((d * d + d + 255UL) / 256UL), (unsigned int)kvTotal, 1U, 256U,
                              0UL, (&void)&p2[0])) return 0;
            void* p3[16];
            p3[0] = (void*)&pq; p3[1] = (void*)&pk; p3[2] = (void*)&pv; p3[3] = (void*)&pW;
            p3[4] = (void*)&pZ; p3[5] = (void*)&pT; p3[6] = (void*)&pTZ; p3[7] = (void*)&po;
            p3[8] = (void*)&sq; p3[9] = (void*)&sk; p3[10] = (void*)&h; p3[11] = (void*)&hkv;
            p3[12] = (void*)&ncI; p3[13] = (void*)&cppI; p3[14] = (void*)&partsI; p3[15] = (void*)&c;
            unsigned long gx = causal ? nc : (s.seqLen + ATT_LC - 1UL) / ATT_LC;
            return attn_launch_(fou, (unsigned int)gx, (unsigned int)(s.batch * s.heads), 1U, 256U,
                                attn_hout_smem_(d, causal), (&void)&p3[0]);
        }
        if (causal) {
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
