// SafeC Standard Library — device-resident training on ROCm. See nn_rocm.h.
//
// The training kernels are HIP source compiled with hiprtc on first use
// (like gpu_rocm.sc's, as a separate module so programs that only run
// inference don't pay for compiling them) and launched on gpu_rocm's
// stream, so they order with its ops and are recorded by its graph capture.
#pragma once
#include <std/ml/nn_rocm.h>
#include <std/math.h>
#include <std/time.h>

namespace std {

extern int hipModuleLoadData(void** module, const void* image);
extern int hipModuleGetFunction(void** func, void* module, const char* name);
extern int hipModuleLaunchKernel(void* func, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int sharedMem, void* stream, void** params, void** extra);
extern int hipStreamIsCapturing(void* stream, int* status);
extern int hiprtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders,
                               const char** headers, const char** includeNames);
extern int hiprtcCompileProgram(void* prog, int numOptions, const char** options);
extern int hiprtcGetCodeSize(void* prog, unsigned long* size);
extern int hiprtcGetCode(void* prog, char* code);
extern int hiprtcDestroyProgram(void** prog);
extern void* malloc(unsigned long size);
extern void  free(void* ptr);

#define NR_BLOCK      256U
#define NR_MAX_BLOCKS 4096UL
#define NR_TAB        32UL     // tensors per optimizer launch (the kernel's OptTab)
#define NR_BN_SLICES  64UL     // most blocks per (channel, segment) in a batch norm's sums
#define NR_SPLITK_MIN_K 2048UL   // weight gradients over at least this long a k may split it across blocks

// ── Kernel source ────────────────────────────────────────────────────────────
// Activation tags are nn.h's ACT_*. Gradients of ReLU, leaky ReLU, tanh and
// sigmoid are computed from the activation's output, GELU's and SiLU's from
// its input. Reductions go per wavefront with shuffles, per block through
// LDS, then one atomicAdd per block.
static const char* nr_kernel_source_() {
    return
    // Activation storage: NR_ACT 0 float, 1 fp16, 2 bf16 (each precision is its
    // own compiled module). Arithmetic is fp32; a 16-bit store rounds to nearest
    // even.
    "#ifndef NR_ACT\n"
    "#define NR_ACT 0\n"
    "#endif\n"
    "#if NR_ACT\n"
    "typedef unsigned short act_t;\n"
    "__device__ inline float LD(const act_t* p, unsigned long long i) {\n"
    "    unsigned short v = p[i];\n"
    "    return NR_ACT == 2 ? __builtin_bit_cast(float, (unsigned)v << 16) : (float)__builtin_bit_cast(_Float16, v);\n"
    "}\n"
    "__device__ inline void ST(act_t* p, unsigned long long i, float x) {\n"
    "    if (NR_ACT == 2) {\n"
    "        unsigned u = __builtin_bit_cast(unsigned, x);\n"
    "        p[i] = (u & 0x7fffffffu) > 0x7f800000u ? (unsigned short)((u >> 16) | 0x40u)\n"
    "                                                : (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);\n"
    "    } else {\n"
    "        p[i] = __builtin_bit_cast(unsigned short, (_Float16)x);\n"
    "    }\n"
    "}\n"
    "#else\n"
    "typedef float act_t;\n"
    "__device__ inline float LD(const act_t* p, unsigned long long i) { return p[i]; }\n"
    "__device__ inline void ST(act_t* p, unsigned long long i, float x) { p[i] = x; }\n"
    "#endif\n"
    "#define GS(i, n) for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += gridDim.x * blockDim.x)\n"
    // Every thread of the block must call it; the total is valid in thread 0.
    "__device__ inline float block_sum(float s) {\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "    __shared__ float part[1024 / 32];\n"
    "    __syncthreads();   // a previous call's thread 0 may still be reading part\n"
    "    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;\n"
    "    if (lane == 0) part[wid] = s;\n"
    "    __syncthreads();\n"
    "    s = 0.0f;\n"
    "    if (threadIdx.x == 0) {\n"
    "        int nw = (blockDim.x + warpSize - 1) / warpSize;\n"
    "        for (int w = 0; w < nw; ++w) s += part[w];\n"
    "    }\n"
    "    return s;\n"
    "}\n"
    "__device__ inline bool last_block(unsigned* count, unsigned total) {\n"
    "    __syncthreads();\n"
    "    __shared__ bool last;\n"
    "    if (threadIdx.x == 0) {\n"
    "        __threadfence();\n"
    "        last = atomicAdd(count, 1u) == total - 1u;\n"
    "        if (last) *count = 0u;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    return last;\n"
    "}\n"
    "#define GELU_C 0.7978845608028654f\n"
    "__device__ inline float act_f(float x, int a) {\n"
    "    switch (a) {\n"
    "    case 1: return x > 0.0f ? x : 0.0f;\n"
    "    case 2: return x > 0.0f ? x : 0.2f * x;\n"
    "    case 3: return tanhf(x);\n"
    "    case 4: return 1.0f / (1.0f + expf(-x));\n"
    "    case 5: return 0.5f * x * (1.0f + tanhf(GELU_C * (x + 0.044715f * x * x * x)));\n"
    "    case 6: return x / (1.0f + expf(-x));\n"
    "    default: return x;\n"
    "    }\n"
    "}\n"
    // v: the output for tags 1-4, the input for 5-6
    "__device__ inline float act_d(float v, int a) {\n"
    "    switch (a) {\n"
    "    case 1: return v > 0.0f ? 1.0f : 0.0f;\n"
    "    case 2: return v > 0.0f ? 1.0f : 0.2f;\n"
    "    case 3: return 1.0f - v * v;\n"
    "    case 4: return v * (1.0f - v);\n"
    "    case 5: { float t = tanhf(GELU_C * (v + 0.044715f * v * v * v));\n"
    "              return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * GELU_C * (1.0f + 3.0f * 0.044715f * v * v); }\n"
    "    case 6: { float s = 1.0f / (1.0f + expf(-v)); return s * (1.0f + v * (1.0f - s)); }\n"
    "    default: return 1.0f;\n"
    "    }\n"
    "}\n"
    // h = act(z + b[channel]); z keeps the sum when it is a separate buffer.
    // b may be null (an activation alone).
    // channel = (i / plane) % C: plane 1 for rows, batch*H*W for channel-major.
    "extern \"C\" __global__ void bias_act(act_t* z, act_t* h, const float* b, unsigned n, unsigned plane,\n"
    "                                    unsigned C, int act) {\n"
    "    GS(i, n) {\n"
    "        float v = b ? LD(z, i) + b[(i / plane) % C] : LD(z, i);\n"
    "        if (z != h) ST(z, i, v);\n"
    "        ST(h, i, act_f(v, act));\n"
    "    }\n"
    "}\n"
    // d[i] *= act'(src[i]) over channel c's 'count' elements (blockIdx.x = c,
    // blockIdx.y one of gridDim.y slices), gb[c] = their sum: each block stores
    // its partial in part[c * gridDim.y + slice], and the channel's last block
    // to finish (counter in part's tail, left at 0) adds them up in order, so
    // the sum does not depend on scheduling.
    "extern \"C\" __global__ void act_grad_bias(act_t* d, const act_t* src, float* gb, float* part, unsigned count,\n"
    "                                         unsigned plane, unsigned C, int act) {\n"
    "    unsigned c = blockIdx.x, S = gridDim.y;\n"
    "    float s = 0.0f;\n"
    "    for (unsigned e = blockIdx.y * blockDim.x + threadIdx.x; e < count; e += S * blockDim.x) {\n"
    "        unsigned long long i = (unsigned long long)(e / plane) * plane * C + c * plane + e % plane;\n"
    "        float g = LD(d, i);\n"
    "        if (act) { g *= act_d(LD(src, i), act); ST(d, i, g); }\n"
    "        s += g;\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) part[c * S + blockIdx.y] = s;\n"
    "    if (last_block((unsigned*)part + C * S + c, S) && threadIdx.x == 0) {\n"
    "        float t = 0.0f;\n"
    "        for (unsigned k = 0; k < S; ++k) t += part[c * S + k];\n"
    "        gb[c] = t;\n"
    "    }\n"
    "}\n"
    // The same for row-major [rows][C] (a fully connected layer): 16 columns per
    // block, 16 threads down each column, then a fixed-order sum of the 16.
    "extern \"C\" __global__ void act_grad_rows(act_t* d, const act_t* src, float* gb, unsigned rows, unsigned C, int act) {\n"
    "    __shared__ float part[16][17];\n"
    "    const unsigned tx = threadIdx.x % 16, ty = threadIdx.x / 16, c = blockIdx.x * 16 + tx;\n"
    "    float s = 0.0f;\n"
    "    if (c < C) {\n"
    "        for (unsigned r = ty; r < rows; r += 16) {\n"
    "            unsigned long long i = (unsigned long long)r * C + c;\n"
    "            float g = LD(d, i);\n"
    "            if (act) { g *= act_d(LD(src, i), act); ST(d, i, g); }\n"
    "            s += g;\n"
    "        }\n"
    "    }\n"
    "    part[ty][tx] = s;\n"
    "    __syncthreads();\n"
    "    if (ty == 0 && c < C) {\n"
    "        float t = 0.0f;\n"
    "        for (unsigned k = 0; k < 16; ++k) t += part[k][tx];\n"
    "        gb[c] = t;\n"
    "    }\n"
    "}\n"
    // NCHW images x [B][C][H][W] -> col [C*kh*kw][B*oh*ow] (column b*P + p). g
    // packs the geometry: kh kw sh sw ph pw dh dw (kernel, stride, padding, dilation).
    "struct Geom { unsigned kh, kw, sh, sw, ph, pw, dh, dw; };\n"
    "extern \"C\" __global__ void im2col(const float* x, float* col, unsigned C, unsigned H, unsigned W,\n"
    "                                  unsigned oh, unsigned ow, Geom g, unsigned B) {\n"
    "    unsigned P = oh * ow, N = B * P;\n"
    "    GS(t, C * g.kh * g.kw * N) {\n"
    "        unsigned n = t % N, kk = t / N;\n"
    "        unsigned j = kk % g.kw, i = (kk / g.kw) % g.kh, c = kk / (g.kw * g.kh);\n"
    "        unsigned b = n / P, p = n % P;\n"
    "        int iy = (int)((p / ow) * g.sh + i * g.dh) - (int)g.ph, ix = (int)((p % ow) * g.sw + j * g.dw) - (int)g.pw;\n"
    "        col[t] = (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) ? x[((b * C + c) * H + iy) * W + ix] : 0.0f;\n"
    "    }\n"
    "}\n"
    // The adjoint, as a gather: x[b][c][iy][ix] (NCHW) = sum of the col entries that
    // im2col would read it into. With a bias it is a transposed convolution's
    // output: h = act(sum + bias[c]), z = the sum plus bias when z is a
    // separate buffer.
    "extern \"C\" __global__ void col2im(const float* col, float* z, float* h, const float* bias, int act, unsigned C,\n"
    "                                  unsigned H, unsigned W, unsigned oh, unsigned ow, Geom g, unsigned B) {\n"
    "    unsigned N = B * oh * ow;\n"
    "    GS(t, C * B * H * W) {\n"
    "        unsigned ix = t % W, iy = (t / W) % H, c = (t / (W * H)) % C, b = t / (W * H * C);\n"
    "        float sum = 0.0f;\n"
    "        // without dilation only kernel rows (columns) congruent to iy + ph mod sh can hit\n"
    "        const unsigned i0 = g.dh == 1 ? (iy + g.ph) % g.sh : 0, is = g.dh == 1 ? g.sh : 1;\n"
    "        const unsigned j0 = g.dw == 1 ? (ix + g.pw) % g.sw : 0, js = g.dw == 1 ? g.sw : 1;\n"
    "        for (unsigned i = i0; i < g.kh; i += is) {\n"
    "            int ty = (int)(iy + g.ph) - (int)(i * g.dh);\n"
    "            if (ty < 0) break;\n"
    "            if (ty % (int)g.sh) continue;\n"
    "            unsigned oy = (unsigned)ty / g.sh;\n"
    "            if (oy >= oh) continue;\n"
    "            for (unsigned j = j0; j < g.kw; j += js) {\n"
    "                int tx = (int)(ix + g.pw) - (int)(j * g.dw);\n"
    "                if (tx < 0) break;\n"
    "                if (tx % (int)g.sw) continue;\n"
    "                unsigned ox = (unsigned)tx / g.sw;\n"
    "                if (ox >= ow) continue;\n"
    "                sum += col[((c * g.kh + i) * g.kw + j) * N + (b * oh + oy) * ow + ox];\n"
    "            }\n"
    "        }\n"
    "        if (bias) {\n"
    "            sum += bias[c];\n"
    "            if (z != h) z[t] = sum;\n"
    "            sum = act_f(sum, act);\n"
    "        }\n"
    "        h[t] = sum;\n"
    "    }\n"
    "}\n"
    // c[m][n] = sum_k a[m][k] b[n][k] for few outputs and a long k (a
    // convolution's weight gradient): one wavefront per output, lanes striding k.
    "extern \"C\" __global__ void gemm_nt_long(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned K) {\n"
    "    unsigned lane = threadIdx.x % warpSize;\n"
    "    unsigned o = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;\n"
    "    if (o >= m * n) return;\n"
    "    const float* ar = a + (unsigned long long)(o / n) * K;\n"
    "    const float* br = b + (unsigned long long)(o % n) * K;\n"
    "    float s0 = 0.0f, s1 = 0.0f;\n"
    "    unsigned kk = lane;\n"
    "    for (; kk + warpSize < K; kk += 2 * warpSize) { s0 += ar[kk] * br[kk]; s1 += ar[kk + warpSize] * br[kk + warpSize]; }\n"
    "    if (kk < K) s0 += ar[kk] * br[kk];\n"
    "    float s = s0 + s1;\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "    if (lane == 0) c[o] = s;\n"
    "}\n"
    // [A][Bd][P] -> [Bd][A][P]
    "extern \"C\" __global__ void swap01(const float* in, float* out, unsigned A, unsigned Bd, unsigned P) {\n"
    "    GS(t, A * Bd * P) {\n"
    "        unsigned p = t % P, bi = (t / P) % Bd, a = t / (P * Bd);\n"
    "        out[(bi * A + a) * P + p] = in[t];\n"
    "    }\n"
    "}\n"
    // One wavefront per row: lanes stride the classes, max and sum by shuffles.
    "__device__ inline float wave_max(float v) {\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));\n"
    "    return v;\n"
    "}\n"
    "__device__ inline float wave_sum(float v) {\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);\n"
    "    return v;\n"
    "}\n"
    "extern \"C\" __global__ void ce_loss(const float* lg, const float* lab, float* d, float* loss,\n"
    "                                   unsigned rows, unsigned C) {\n"
    "    const unsigned lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;\n"
    "    float s = 0.0f, inv = 1.0f / (float)rows;\n"
    "    for (unsigned r = blockIdx.x * wpb + threadIdx.x / warpSize; r < rows; r += gridDim.x * wpb) {\n"
    "        const float* x = lg + (unsigned long long)r * C;\n"
    "        float mx = -3.402823e38f;\n"
    "        for (unsigned c = lane; c < C; c += warpSize) mx = fmaxf(mx, x[c]);\n"
    "        mx = wave_max(mx);\n"
    "        float se = 0.0f;\n"
    "        for (unsigned c = lane; c < C; c += warpSize) se += expf(x[c] - mx);\n"
    "        float lse = mx + logf(wave_sum(se));\n"
    "        unsigned y = (unsigned)lab[r];\n"
    "        for (unsigned c = lane; c < C; c += warpSize)\n"
    "            d[(unsigned long long)r * C + c] = (expf(x[c] - lse) - (c == y ? 1.0f : 0.0f)) * inv;\n"
    "        if (lane == 0) s += lse - x[y < C ? y : 0];\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(loss, s * inv);\n"
    "}\n"
    "extern \"C\" __global__ void bce_loss(const float* x, const float* t, float* d, float* loss, unsigned n, float scale) {\n"
    "    float s = 0.0f, k = scale / (float)n;\n"
    "    GS(i, n) {\n"
    "        float v = x[i], y = t[i];\n"
    "        s += fmaxf(v, 0.0f) - v * y + log1pf(expf(-fabsf(v)));\n"
    "        d[i] = (1.0f / (1.0f + expf(-v)) - y) * k;\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(loss, s * k);\n"
    "}\n"
    "extern \"C\" __global__ void mse_loss(const float* x, const float* t, float* d, float* loss, unsigned n, float scale) {\n"
    "    float s = 0.0f, k = scale / (float)n;\n"
    "    GS(i, n) {\n"
    "        float e = x[i] - t[i];\n"
    "        s += e * e;\n"
    "        d[i] = 2.0f * e * k;\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(loss, s * k);\n"
    "}\n"
    // VAE: head rows are (mu | logvar); eps ~ N(0, 1) from a counter-based
    // hash of (seed, draw counter, index); z = mu + exp(logvar/2) eps; the
    // loss gets beta * KL averaged over rows.
    "__device__ inline unsigned long long mix64(unsigned long long z) {\n"
    "    z += 0x9e3779b97f4a7c15ULL;\n"
    "    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;\n"
    "    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;\n"
    "    return z ^ (z >> 31);\n"
    "}\n"
    "extern \"C\" __global__ void vae_fwd(const float* hd, float* eps, float* z, float* loss, const float* ctr,\n"
    "                                   unsigned rows, unsigned L, float beta, unsigned long long seed) {\n"
    "    float s = 0.0f;\n"
    "    unsigned long long c = (unsigned long long)ctr[0];\n"
    "    GS(t, rows * L) {\n"
    "        unsigned r = t / L, j = t % L;\n"
    "        float mu = hd[r * 2 * L + j], lv = hd[r * 2 * L + L + j];\n"
    "        unsigned long long h = mix64(seed ^ mix64((c << 32) | t));\n"
    "        float u1 = (float)((h >> 40) + 1ULL) * 5.9604645e-8f;\n"
    "        float u2 = (float)((h >> 16) & 0xFFFFFFULL) * 5.9604645e-8f;\n"
    "        float e = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);\n"
    "        eps[t] = e;\n"
    "        z[t] = mu + expf(0.5f * lv) * e;\n"
    "        s += 1.0f + lv - mu * mu - expf(lv);\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(loss, beta * -0.5f * s / (float)rows);\n"
    "}\n"
    "extern \"C\" __global__ void vae_bwd(const float* hd, const float* eps, const float* dz, float* dhd, float* ctr,\n"
    "                                   unsigned rows, unsigned L, float beta) {\n"
    "    float k = beta / (float)rows;\n"
    "    GS(t, rows * L) {\n"
    "        unsigned r = t / L, j = t % L;\n"
    "        float mu = hd[r * 2 * L + j], lv = hd[r * 2 * L + L + j], g = dz[t];\n"
    "        dhd[r * 2 * L + j] = g + k * mu;\n"
    "        dhd[r * 2 * L + L + j] = g * eps[t] * 0.5f * expf(0.5f * lv) - 0.5f * k * (1.0f - expf(lv));\n"
    "    }\n"
    "    if (blockIdx.x == 0 && threadIdx.x == 0) ctr[0] += 1.0f;\n"
    "}\n"
    // Optimizer, as nn.h's Optimizer::step. st: 0 lr, 1 momentum, 2 beta1,
    // 3 beta2, 4 eps, 5 weight decay, 6 clip norm, 7 steps taken, 11 sum of
    // squared gradients (clipping), 12 finished-block count. Every block reads
    // st[7] and st[11] first; the last block of the step's last launch to
    // finish (all others have read them by then) advances st[7] and clears
    // st[11], so a step is one launch.
    "struct OptTab { float* w[32]; const float* g[32]; float* m[32]; float* v[32]; unsigned n[32]; };\n"
    // ls: the fp16 loss scale [scale, good steps] (null: none). The gradients
    // hold it; the sums are of the unscaled values, inf / NaN on an overflow.
    "extern \"C\" __global__ void opt_sumsq(OptTab t, float* st, const float* ls) {\n"
    "    __shared__ const float* sg;\n"
    "    __shared__ unsigned sn;\n"
    "    if (threadIdx.x == 0) { sg = t.g[blockIdx.y]; sn = t.n[blockIdx.y]; }\n"
    "    __syncthreads();\n"
    "    unsigned n = sn;\n"
    "    const float* g = sg;\n"
    "    const float inv = ls ? 1.0f / ls[0] : 1.0f;\n"
    "    float s = 0.0f;\n"
    "    GS(i, n) { float u = g[i] * inv; s += u * u; }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(&st[11], s);\n"
    "}\n"
    // With a loss scale, a step whose gradients overflowed changes no weight or
    // step count and halves the scale; 2000 good steps in a row double it.
    "extern \"C\" __global__ void opt_step(OptTab tab, float* st, int kind, int clip, int finalLaunch, float clampW,\n"
    "                                    float* ls) {\n"
    "#pragma clang fp contract(off)\n"
    "    __shared__ float* sp[4];\n"
    "    __shared__ unsigned sn;\n"
    "    if (threadIdx.x == 0) {\n"
    "        unsigned b = blockIdx.y;\n"
    "        sp[0] = tab.w[b]; sp[1] = (float*)tab.g[b]; sp[2] = tab.m[b]; sp[3] = tab.v[b]; sn = tab.n[b];\n"
    "    }\n"
    "    __syncthreads();\n"
    "    unsigned n = sn;\n"
    "    float* w = sp[0]; const float* gr = sp[1]; float* m = sp[2]; float* v = sp[3];\n"
    "    float lr = st[0], mu = st[1], b1 = st[2], b2 = st[3], eps = st[4], wd = st[5];\n"
    "    float t = st[7] + 1.0f;\n"
    "    __shared__ float corr[4];   // 1/(1-b1^t), 1/sqrt(1-b2^t), gradient scale, skip: once per block\n"
    "    if (threadIdx.x == 0) {\n"
    "        corr[0] = 1.0f / (1.0f - powf(b1, t));\n"
    "        corr[1] = 1.0f / sqrtf(1.0f - powf(b2, t));\n"
    "        float gsc = ls ? 1.0f / ls[0] : 1.0f;\n"
    "        if (clip) {\n"
    "            float nrm = sqrtf(st[11]);\n"
    "            if (nrm > st[6] && nrm > 0.0f) gsc *= st[6] / nrm;\n"
    "        }\n"
    "        corr[2] = gsc;\n"
    "        corr[3] = ls && !isfinite(st[11]) ? 1.0f : 0.0f;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float ibc1 = corr[0], isbc2 = corr[1], gs = corr[2];\n"
    "    const bool skip = corr[3] != 0.0f;\n"
    "    if (!skip) GS(i, n) {\n"
    "        float g = gr[i] * gs, wi = w[i];\n"
    "        if (kind == 0) {\n"
    "            w[i] = wi - lr * (g + wd * wi);\n"
    "        } else if (kind == 1 || kind == 2) {\n"
    "            g = g + wd * wi;\n"
    "            float mi = mu * m[i] + g;\n"
    "            m[i] = mi;\n"
    "            w[i] = wi - lr * (kind == 1 ? mi : g + mu * mi);\n"
    "        } else if (kind == 6) {\n"
    "            g = g + wd * wi;\n"
    "            float vi = v[i] + g * g;\n"
    "            v[i] = vi;\n"
    "            w[i] = wi - lr * g / (sqrtf(vi) + eps);\n"
    "        } else if (kind == 7) {\n"
    "            float c = b1 * m[i] + (1.0f - b1) * g;\n"
    "            float sg = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);\n"
    "            w[i] = wi - lr * (sg + wd * wi);\n"
    "            m[i] = b2 * m[i] + (1.0f - b2) * g;\n"
    "        } else if (kind == 3) {\n"
    "            g = g + wd * wi;\n"
    "            float vi = b2 * v[i] + (1.0f - b2) * g * g;\n"
    "            v[i] = vi;\n"
    "            w[i] = wi - lr * g / (sqrtf(vi) + eps);\n"
    "        } else {\n"
    "            float l2 = kind == 5 ? 0.0f : wd, dec = kind == 5 ? wd : 0.0f;\n"
    "            g = g + l2 * wi;\n"
    "            float mi = b1 * m[i] + (1.0f - b1) * g;\n"
    "            float vi = b2 * v[i] + (1.0f - b2) * g * g;\n"
    "            m[i] = mi; v[i] = vi;\n"
    "            float upd = (mi * ibc1) / (sqrtf(vi) * isbc2 + eps) + dec * wi;\n"
    "            w[i] = wi - lr * upd;\n"
    "        }\n"
    "        if (clampW > 0.0f) { float v = w[i]; w[i] = v > clampW ? clampW : (v < -clampW ? -clampW : v); }\n"
    "    }\n"
    "    if (finalLaunch && last_block((unsigned*)&st[12], gridDim.x * gridDim.y) && threadIdx.x == 0) {\n"
    "        if (!skip) st[7] = t;\n"
    "        if (clip || ls) st[11] = 0.0f;\n"
    "        if (ls) {\n"
    "            if (skip) { ls[0] *= 0.5f; ls[1] = 0.0f; }\n"
    "            else if ((ls[1] += 1.0f) >= 2000.0f) { ls[0] *= 2.0f; ls[1] = 0.0f; }\n"
    "        }\n"
    "    }\n"
    "}\n"
    // Batches: rows order[cur .. cur+rows) of x and y, in one launch; the
    // last block moves the cursor (cur[0]) past them (cur[1]: block count).
    "extern \"C\" __global__ void gather2(const float* sx, float* dx, unsigned fx, const float* sy, float* dy,\n"
    "                                   unsigned fy, const float* order, float* cur, unsigned rows) {\n"
    "    unsigned base = (unsigned)cur[0], f = fx + fy;\n"
    "    GS(t, rows * f) {\n"
    "        unsigned r = t / f, c = t % f;\n"
    "        unsigned long long src = (unsigned long long)order[base + r];\n"
    "        if (c < fx) dx[r * fx + c] = sx[src * fx + c];\n"
    "        else dy[r * fy + c - fx] = sy[src * fy + c - fx];\n"
    "    }\n"
    "    if (last_block((unsigned*)&cur[1], gridDim.x) && threadIdx.x == 0) cur[0] = (float)(base + rows);\n"
    "}\n"
    // Small fully connected nets: forward, loss and backward of a whole batch
    // in one workgroup with every activation and gradient in LDS (see
    // nr_small_: at most SN_LDS floats) -- one launch instead of ~5 per layer,
    // and no global-memory round trips between the phases. Writes the
    // gradients gw / gb and the output h of the last layer; the other layers'
    // h / dh buffers are not touched. loss kind 0 cross-entropy (y: class
    // indices), 1 BCE on logits, 2 MSE; loss1[0] += the mean loss.
    "struct NetTab { const float* w[8]; const float* b[8]; float* gw[8]; float* gb[8]; float* h[8]; float* z[8];\n"
    "                float* dh[8]; unsigned in[8]; unsigned out[8]; int act[8]; int n; };\n"
    "#define SN_LDS 16000\n"
    "__device__ inline void lds_load(float* dst, const float* src, unsigned n) {\n"
    "    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];\n"
    "}\n"
    // weight [I][O] with rows padded to O + 1 floats: reading a column across
    // threads (dx) then hits distinct banks
    "__device__ inline void lds_load_w(float* dst, const float* src, unsigned I, unsigned O) {\n"
    "    for (unsigned i = threadIdx.x; i < I * O; i += blockDim.x) dst[(i / O) * (O + 1) + i % O] = src[i];\n"
    "}\n"
    "__device__ inline bool needs_z(int a) { return a == 5 || a == 6; }\n"
    // A workgroup barrier that only waits for LDS traffic. __syncthreads()
    // also waits for every outstanding global store (gfx908 has no back-off
    // barrier, so the compiler drains all counters first), which costs ~1-2 us
    // per phase here; nothing in this kernel reads its global stores back.
    "#if defined(__GFX9__) || defined(__GFX10__) || defined(__GFX11__)\n"
    "#define LDS_SYNC() asm volatile(\"s_waitcnt lgkmcnt(0)\\n\\ts_barrier\" ::: \"memory\")\n"
    "#else\n"
    "#define LDS_SYNC() __syncthreads()\n"
    "#endif\n"
    "extern \"C\" __global__ __launch_bounds__(1024) void small_net_step(NetTab targ, const float* x, const float* y,\n"
    "                                                                  float* loss1, unsigned B, int kind) {\n"
    "    __shared__ float sm[SN_LDS];\n"
    "    // indexing the by-value table with a loop variable would copy it to\n"
    "    // every thread's scratch; LDS holds one copy\n"
    "    __shared__ NetTab t;\n"
    "    for (unsigned i = threadIdx.x; i < sizeof(NetTab) / 4; i += blockDim.x)\n"
    "        ((unsigned*)&t)[i] = ((const unsigned*)&targ)[i];\n"
    "    __syncthreads();\n"
    "    const unsigned T = blockDim.x;\n"
    "    const int L = t.n - 1;\n"
    "    // layout: x | per layer h (and z) | dzA | dzB | weight\n"
    "    // per-layer offsets in LDS as well: a local array indexed by l would\n"
    "    // live in scratch\n"
    "    __shared__ unsigned hoff[8], zoff[8], dzo[3];\n"
    "    if (threadIdx.x == 0) {\n"
    "        unsigned off = B * t.in[0], maxO = 0;\n"
    "        for (int l = 0; l <= L; ++l) {\n"
    "            hoff[l] = off; off += B * t.out[l];\n"
    "            zoff[l] = hoff[l];\n"
    "            if (needs_z(t.act[l])) { zoff[l] = off; off += B * t.out[l]; }\n"
    "            maxO = max(maxO, max(t.out[l], t.in[l]));\n"
    "        }\n"
    "        dzo[0] = off; dzo[1] = off + B * maxO; dzo[2] = off + 2 * B * maxO;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float* sx = sm;\n"
    "    float* dzA = sm + dzo[0];\n"
    "    float* dzB = sm + dzo[1];\n"
    "    float* sw = sm + dzo[2];\n"
    "    lds_load(sx, x, B * t.in[0]);\n"
    "    for (int l = 0; l <= L; ++l) {\n"
    "        const unsigned I = t.in[l], O = t.out[l];\n"
    "        const float* in = l > 0 ? sm + hoff[l - 1] : sx;\n"
    "        lds_load_w(sw, t.w[l], I, O);\n"
    "        LDS_SYNC();\n"
    "        const float* bb = t.b[l];\n"
    "        const int a = t.act[l];\n"
    "        for (unsigned e = threadIdx.x; e < B * O; e += T) {\n"
    "            unsigned r = e / O, c = e % O;\n"
    "            const float* xr = in + r * I;\n"
    "            float s = 0.0f;\n"
    "            float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;\n"
    "            unsigned k = 0;\n"
    "            for (; k + 4 <= I; k += 4) {\n"
    "                s += xr[k] * sw[k * (O + 1) + c];\n"
    "                s1 += xr[k + 1] * sw[(k + 1) * (O + 1) + c];\n"
    "                s2 += xr[k + 2] * sw[(k + 2) * (O + 1) + c];\n"
    "                s3 += xr[k + 3] * sw[(k + 3) * (O + 1) + c];\n"
    "            }\n"
    "            for (; k < I; ++k) s += xr[k] * sw[k * (O + 1) + c];\n"
    "            s = (s + s1) + (s2 + s3);\n"
    "            s += bb[c];\n"
    "            float hv = act_f(s, a);\n"
    "            sm[zoff[l] + e] = s;\n"
    "            sm[hoff[l] + e] = hv;\n"
    "            if (l == L) t.h[L][e] = hv;\n"
    "        }\n"
    "        LDS_SYNC();\n"
    "    }\n"
    "    const unsigned O = t.out[L];\n"
    "    const float* out = sm + hoff[L];\n"
    "    float* d = dzA;\n"
    "    float s = 0.0f;\n"
    "    if (kind == 0) {\n"
    "        const float inv = 1.0f / (float)B;\n"
    "        for (unsigned r = threadIdx.x; r < B; r += T) {\n"
    "            const float* xr = out + r * O;\n"
    "            float mx = xr[0];\n"
    "            for (unsigned c = 1; c < O; ++c) mx = fmaxf(mx, xr[c]);\n"
    "            float se = 0.0f;\n"
    "            for (unsigned c = 0; c < O; ++c) se += expf(xr[c] - mx);\n"
    "            float lse = mx + logf(se);\n"
    "            unsigned yy = (unsigned)y[r];\n"
    "            for (unsigned c = 0; c < O; ++c) d[r * O + c] = (expf(xr[c] - lse) - (c == yy ? 1.0f : 0.0f)) * inv;\n"
    "            s += (lse - xr[yy < O ? yy : 0]) * inv;\n"
    "        }\n"
    "    } else {\n"
    "        const float k = 1.0f / (float)(B * O);\n"
    "        for (unsigned e = threadIdx.x; e < B * O; e += T) {\n"
    "            float v = out[e], tt = y[e];\n"
    "            if (kind == 1) {\n"
    "                s += (fmaxf(v, 0.0f) - v * tt + log1pf(expf(-fabsf(v)))) * k;\n"
    "                d[e] = (1.0f / (1.0f + expf(-v)) - tt) * k;\n"
    "            } else {\n"
    "                float er = v - tt;\n"
    "                s += er * er * k;\n"
    "                d[e] = 2.0f * er * k;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) atomicAdd(loss1, s);\n"
    "    LDS_SYNC();\n"
    "    float* dz = dzA;\n"
    "    float* dn = dzB;\n"
    "    for (int l = L; l >= 0; --l) {\n"
    "        const unsigned I = t.in[l], O = t.out[l];\n"
    "        const int a = t.act[l];\n"
    "        if (a) {\n"
    "            const float* src = sm + (needs_z(a) ? zoff[l] : hoff[l]);\n"
    "            for (unsigned e = threadIdx.x; e < B * O; e += T) dz[e] *= act_d(src[e], a);\n"
    "        }\n"
    "        if (l > 0) lds_load_w(sw, t.w[l], I, O);\n"
    "        LDS_SYNC();\n"
    "        const float* xin = l > 0 ? sm + hoff[l - 1] : sx;\n"
    "        for (unsigned c = threadIdx.x; c < O; c += T) {\n"
    "            float g = 0.0f;\n"
    "            for (unsigned r = 0; r < B; ++r) g += dz[r * O + c];\n"
    "            t.gb[l][c] = g;\n"
    "        }\n"
    "        for (unsigned e = threadIdx.x; e < I * O; e += T) {\n"
    "            unsigned k = e / O, c = e % O;\n"
    "            float g = 0.0f;\n"
    "            float g1 = 0.0f;\n"
    "            unsigned r = 0;\n"
    "            for (; r + 2 <= B; r += 2) {\n"
    "                g += xin[r * I + k] * dz[r * O + c];\n"
    "                g1 += xin[(r + 1) * I + k] * dz[(r + 1) * O + c];\n"
    "            }\n"
    "            for (; r < B; ++r) g += xin[r * I + k] * dz[r * O + c];\n"
    "            g += g1;\n"
    "            t.gw[l][e] = g;\n"
    "        }\n"
    "        if (l > 0) {\n"
    "            for (unsigned e = threadIdx.x; e < B * I; e += T) {\n"
    "                unsigned r = e / I, k = e % I;\n"
    "                const float* wr = sw + k * (O + 1);\n"
    "                const float* dr = dz + r * O;\n"
    "                float g = 0.0f;\n"
    "                float g1 = 0.0f;\n"
    "                unsigned c = 0;\n"
    "                for (; c + 2 <= O; c += 2) { g += dr[c] * wr[c]; g1 += dr[c + 1] * wr[c + 1]; }\n"
    "                for (; c < O; ++c) g += dr[c] * wr[c];\n"
    "                g += g1;\n"
    "                dn[e] = g;\n"
    "            }\n"
    "        }\n"
    "        LDS_SYNC();\n"
    "        float* tmp = dz; dz = dn; dn = tmp;\n"
    "    }\n"
    "}\n"
    // d[i] *= act'(src[i]) (a layer without a bias of its own)
    "extern \"C\" __global__ void act_grad_ew(act_t* d, const act_t* src, unsigned n, int act) {\n"
    "    GS(i, n) ST(d, i, LD(d, i) * act_d(LD(src, i), act));\n"
    "}\n"
    // Pooling over 'planes' channel-major planes (C*B of them) of H x W into
    // oH x oW, the window geometry as convnet.h's Pool2dSpec (padding clipped to
    // the plane). mode 0 max (idx = the window's first largest input, as an
    // offset in its plane), 1 average over the window's extent in the padded
    // plane (count_include_pad), 2 average over its positions in the plane.
    "__device__ inline void pool_window(unsigned oy, unsigned ox, unsigned H, unsigned W, Geom g,\n"
    "                                   int& hs, int& he, int& ws, int& we, float& padded) {\n"
    "    int h0 = (int)(oy * g.sh) - (int)g.ph, w0 = (int)(ox * g.sw) - (int)g.pw;\n"
    "    int h1 = h0 + (int)g.kh, w1 = w0 + (int)g.kw;\n"
    "    int hp = min(h1, (int)(H + g.ph)), wp = min(w1, (int)(W + g.pw));\n"
    "    padded = (float)((hp - h0) * (wp - w0));\n"
    "    hs = max(h0, 0); ws = max(w0, 0); he = min(h1, (int)H); we = min(w1, (int)W);\n"
    "}\n"
    "extern \"C\" __global__ void pool_fwd(const act_t* x, act_t* z, act_t* h, float* idx, unsigned planes, unsigned H,\n"
    "                                    unsigned W, unsigned oH, unsigned oW, Geom g, int mode, int act) {\n"
    "    GS(t, planes * oH * oW) {\n"
    "        unsigned ox = t % oW, oy = (t / oW) % oH, pl = t / (oW * oH);\n"
    "        int hs, he, ws, we; float padded;\n"
    "        pool_window(oy, ox, H, W, g, hs, he, ws, we, padded);\n"
    "        const act_t* img = x + (unsigned long long)pl * H * W;\n"
    "        float v;\n"
    "        if (mode == 0) {\n"
    "            unsigned best = hs * W + ws;\n"
    "            float bv = LD(img, best);\n"
    "            for (int i = hs; i < he; ++i)\n"
    "                for (int j = ws; j < we; ++j) {\n"
    "                    unsigned at = i * W + j;\n"
    "                    float xv = LD(img, at);\n"
    "                    if (xv > bv) { bv = xv; best = at; }\n"
    "                }\n"
    "            idx[t] = (float)best;\n"
    "            v = bv;\n"
    "        } else {\n"
    "            float acc = 0.0f;\n"
    "            for (int i = hs; i < he; ++i)\n"
    "                for (int j = ws; j < we; ++j) acc += LD(img, i * W + j);\n"
    "            v = acc / (mode == 1 ? padded : (float)((he - hs) * (we - ws)));\n"
    "        }\n"
    "        if (z != h) ST(z, t, v);\n"
    "        ST(h, t, act_f(v, act));\n"
    "    }\n"
    "}\n"
    // dx by gathering, for each input, the windows that cover it.
    "extern \"C\" __global__ void pool_bwd(const act_t* d, const float* idx, act_t* dx, unsigned planes, unsigned H,\n"
    "                                    unsigned W, unsigned oH, unsigned oW, Geom g, int mode) {\n"
    "    GS(t, planes * H * W) {\n"
    "        unsigned x0 = t % W, y0 = (t / W) % H, pl = t / (W * H);\n"
    "        unsigned yy = y0 + g.ph, xx = x0 + g.pw;\n"
    "        unsigned oyLo = yy >= g.kh ? (yy - g.kh) / g.sh + 1 : 0, oyHi = min(yy / g.sh, oH - 1);\n"
    "        unsigned oxLo = xx >= g.kw ? (xx - g.kw) / g.sw + 1 : 0, oxHi = min(xx / g.sw, oW - 1);\n"
    "        float s = 0.0f;\n"
    "        for (unsigned oy = oyLo; oy <= oyHi && oy < oH; ++oy)\n"
    "            for (unsigned ox = oxLo; ox <= oxHi && ox < oW; ++ox) {\n"
    "                unsigned o = (pl * oH + oy) * oW + ox;\n"
    "                if (mode == 0) {\n"
    "                    if ((unsigned)idx[o] == y0 * W + x0) s += LD(d, o);\n"
    "                } else {\n"
    "                    int hs, he, ws, we; float padded;\n"
    "                    pool_window(oy, ox, H, W, g, hs, he, ws, we, padded);\n"
    "                    s += LD(d, o) / (mode == 1 ? padded : (float)((he - hs) * (we - ws)));\n"
    "                }\n"
    "            }\n"
    "        ST(dx, t, s);\n"
    "    }\n"
    "}\n"
    // Global pooling: each of 'planes' planes of P values to one (idx: the
    // first largest for max).
    "extern \"C\" __global__ void gpool_fwd(const act_t* x, act_t* z, act_t* h, float* idx, unsigned planes, unsigned P,\n"
    "                                     int mode, int act) {\n"
    "    GS(t, planes) {\n"
    "        const act_t* src = x + (unsigned long long)t * P;\n"
    "        float v;\n"
    "        if (mode == 0) {\n"
    "            unsigned best = 0;\n"
    "            float bv = LD(src, 0);\n"
    "            for (unsigned q = 1; q < P; ++q) { float xv = LD(src, q); if (xv > bv) { bv = xv; best = q; } }\n"
    "            idx[t] = (float)best;\n"
    "            v = bv;\n"
    "        } else {\n"
    "            float acc = 0.0f;\n"
    "            for (unsigned q = 0; q < P; ++q) acc += LD(src, q);\n"
    "            v = acc / (float)P;\n"
    "        }\n"
    "        if (z != h) ST(z, t, v);\n"
    "        ST(h, t, act_f(v, act));\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void gpool_bwd(const act_t* d, const float* idx, act_t* dx, unsigned planes, unsigned P, int mode) {\n"
    "    GS(t, planes * P) {\n"
    "        unsigned pl = t / P, q = t % P;\n"
    "        ST(dx, t, mode == 0 ? ((unsigned)idx[pl] == q ? LD(d, pl) : 0.0f) : LD(d, pl) / (float)P);\n"
    "    }\n"
    "}\n"
    // Nearest upsampling by f of planes H x W, and its adjoint (sums of f x f blocks).
    "extern \"C\" __global__ void upsample_fwd(const act_t* x, act_t* z, act_t* h, unsigned planes, unsigned H, unsigned W,\n"
    "                                        unsigned f, int act) {\n"
    "    unsigned OW = W * f, OH = H * f;\n"
    "    GS(t, planes * OH * OW) {\n"
    "        unsigned ox = t % OW, oy = (t / OW) % OH, pl = t / (OW * OH);\n"
    "        float v = LD(x, ((unsigned long long)pl * H + oy / f) * W + ox / f);\n"
    "        if (z != h) ST(z, t, v);\n"
    "        ST(h, t, act_f(v, act));\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void upsample_bwd(const act_t* d, act_t* dx, unsigned planes, unsigned H, unsigned W, unsigned f) {\n"
    "    unsigned OW = W * f, OH = H * f;\n"
    "    GS(t, planes * H * W) {\n"
    "        unsigned x0 = t % W, y0 = (t / W) % H, pl = t / (W * H);\n"
    "        float s = 0.0f;\n"
    "        for (unsigned i = 0; i < f; ++i)\n"
    "            for (unsigned j = 0; j < f; ++j) s += LD(d, ((unsigned long long)pl * OH + y0 * f + i) * OW + x0 * f + j);\n"
    "        ST(dx, t, s);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void gn_fwd(const act_t* x, act_t* z, act_t* h, const float* gamma, const float* beta,\n"
    "                                  float* stats, unsigned C, unsigned B, unsigned P, unsigned G, float eps, int act) {\n"
    "    const unsigned u = blockIdx.x, b = u / G, g = u % G, Cg = C / G, n = Cg * P;\n"
    "    const unsigned long long base = ((unsigned long long)b * C + g * Cg) * P;\n"
    "    __shared__ float mean_s, inv_s;\n"
    "    float a = 0.0f;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) a += LD(x, base + e);\n"
    "    a = block_sum(a);\n"
    "    if (threadIdx.x == 0) mean_s = a / (float)n;\n"
    "    __syncthreads();\n"
    "    const float mean = mean_s;\n"
    "    float q = 0.0f;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) { float dd = LD(x, base + e) - mean; q += dd * dd; }\n"
    "    q = block_sum(q);\n"
    "    if (threadIdx.x == 0) { inv_s = 1.0f / sqrtf(q / (float)n + eps); stats[u * 2] = mean; stats[u * 2 + 1] = inv_s; }\n"
    "    __syncthreads();\n"
    "    const float inv = inv_s;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) {\n"
    "        unsigned c = g * Cg + e / P;\n"
    "        float v = gamma[c] * (LD(x, base + e) - mean) * inv + beta[c];\n"
    "        if (z != h) ST(z, base + e, v);\n"
    "        ST(h, base + e, act_f(v, act));\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void gn_bwd(const act_t* x, const act_t* dy, act_t* dx, const float* gamma, const float* stats,\n"
    "                                  unsigned C, unsigned B, unsigned P, unsigned G) {\n"
    "    const unsigned u = blockIdx.x, b = u / G, g = u % G, Cg = C / G, n = Cg * P;\n"
    "    const unsigned long long base = ((unsigned long long)b * C + g * Cg) * P;\n"
    "    const float mean = stats[u * 2], inv = stats[u * 2 + 1];\n"
    "    __shared__ float m1_s, m2_s;\n"
    "    float a = 0.0f, q = 0.0f;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) {\n"
    "        float dxh = LD(dy, base + e) * gamma[g * Cg + e / P];\n"
    "        a += dxh;\n"
    "        q += dxh * (LD(x, base + e) - mean) * inv;\n"
    "    }\n"
    "    a = block_sum(a);\n"
    "    q = block_sum(q);\n"
    "    if (threadIdx.x == 0) { m1_s = a / (float)n; m2_s = q / (float)n; }\n"
    "    __syncthreads();\n"
    "    const float m1 = m1_s, m2 = m2_s;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) {\n"
    "        float xh = (LD(x, base + e) - mean) * inv;\n"
    "        ST(dx, base + e, inv * (LD(dy, base + e) * gamma[g * Cg + e / P] - m1 - xh * m2));\n"
    "    }\n"
    "}\n"
    // dgamma[c] = sum dy * xhat, dbeta[c] = sum dy over images and plane; one block per channel.
    "extern \"C\" __global__ void gn_params(const act_t* x, const act_t* dy, const float* stats, float* dgamma, float* dbeta,\n"
    "                                     unsigned C, unsigned B, unsigned P, unsigned G) {\n"
    "    const unsigned c = blockIdx.x, g = c / (C / G), n = B * P;\n"
    "    float a = 0.0f, q = 0.0f;\n"
    "    for (unsigned e = threadIdx.x; e < n; e += blockDim.x) {\n"
    "        unsigned b = e / P;\n"
    "        unsigned long long i = ((unsigned long long)b * C + c) * P + e % P;\n"
    "        float mean = stats[(b * G + g) * 2], inv = stats[(b * G + g) * 2 + 1];\n"
    "        float gy = LD(dy, i);\n"
    "        a += gy;\n"
    "        q += gy * (LD(x, i) - mean) * inv;\n"
    "    }\n"
    "    a = block_sum(a);\n"
    "    q = block_sum(q);\n"
    "    if (threadIdx.x == 0) { dgamma[c] = q; dbeta[c] = a; }\n"
    "}\n"
    // out[r] = (x[r] (cx values), one plane of 'plane' values per class: 1 in label[r]'s)
    // (row r's label is lab[r % labRows]: stacked batches share one label list)
    "extern \"C\" __global__ void append_labels(const float* x, const float* lab, float* out, unsigned rows, unsigned cx,\n"
    "                                         unsigned plane, unsigned classes, unsigned labRows) {\n"
    "    const unsigned w = cx + classes * plane;\n"
    "    GS(t, rows * w) {\n"
    "        unsigned r = t / w, c = t % w;\n"
    "        out[t] = c < cx ? x[(unsigned long long)r * cx + c]\n"
    "                        : ((c - cx) / plane == (unsigned)lab[r % labRows] ? 1.0f : 0.0f);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void slice_cols(const act_t* src, act_t* dst, unsigned rows, unsigned sc, unsigned dc, unsigned off) {\n"
    "    GS(t, rows * dc) ST(dst, t, LD(src, (unsigned long long)(t / dc) * sc + off + t % dc));\n"
    "}\n"
    // out[b] = (a[b] (na values), s[b] (ns values)): a per-image channel concat.
    "extern \"C\" __global__ void concat_rows(const act_t* a, unsigned na, const act_t* s, unsigned ns, act_t* out, unsigned B) {\n"
    "    const unsigned w = na + ns;\n"
    "    GS(t, B * w) {\n"
    "        unsigned r = t / w, c = t % w;\n"
    "        ST(out, t, c < na ? LD(a, (unsigned long long)r * na + c) : LD(s, (unsigned long long)r * ns + c - na));\n"
    "    }\n"
    "}\n"
    // Precision boundaries: float -> act (k * x) and act -> float (k * x); k
    // from *kp when kp is set (the loss scale, or its inverse with inv).
    "extern \"C\" __global__ void to_act(const float* x, act_t* y, unsigned n, const float* kp, int inv) {\n"
    "    float k = kp ? (inv ? 1.0f / kp[0] : kp[0]) : 1.0f;\n"
    "    GS(i, n) ST(y, i, x[i] * k);\n"
    "}\n"
    "extern \"C\" __global__ void from_act(const act_t* x, float* y, unsigned n, const float* kp, int inv) {\n"
    "    float k = kp ? (inv ? 1.0f / kp[0] : kp[0]) : 1.0f;\n"
    "    GS(i, n) y[i] = LD(x, i) * k;\n"
    "}\n"
    // ── GAN ──
    // Counter-based noise: value i of draw 'salt' in step ctr[0].
    "__device__ inline float hash_uniform(unsigned long long seed, unsigned long long c, unsigned salt, unsigned i, int second) {\n"
    "    unsigned long long h = mix64(seed ^ mix64((c << 32) ^ ((unsigned long long)salt << 24) ^ i));\n"
    "    return second ? (float)((h >> 16) & 0xFFFFFFULL) * 5.9604645e-8f : (float)((h >> 40) + 1ULL) * 5.9604645e-8f;\n"
    "}\n"
    "extern \"C\" __global__ void noise_fill(float* out, unsigned n, unsigned long long seed, const float* ctr, unsigned salt,\n"
    "                                      int normal) {\n"
    "    unsigned long long c = (unsigned long long)ctr[0];\n"
    "    GS(i, n) {\n"
    "        float u1 = hash_uniform(seed, c, salt, i, 0);\n"
    "        out[i] = normal ? sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * hash_uniform(seed, c, salt, i, 1)) : u1 - 5.9604645e-8f;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void fill_value(float* out, unsigned n, float v) { GS(i, n) out[i] = v; }\n"
    // xh[r] = t[r] real[r] + (1 - t[r]) fake[r]; also clears the penalty slot st[4].
    "extern \"C\" __global__ void lerp_rows(const float* real, const float* fake, const float* t, float* xh, unsigned m,\n"
    "                                     unsigned d, float* st) {\n"
    "    if (blockIdx.x == 0 && threadIdx.x == 0) st[4] = 0.0f;\n"
    "    GS(i, m * d) {\n"
    "        float tt = t[i / d];\n"
    "        xh[i] = tt * real[i] + (1.0f - tt) * fake[i];\n"
    "    }\n"
    "}\n"
    // WGAN-GP: one block per row of g = grad_x D(xh): xp / xm = xh +- eps g / |g|
    // into stack rows 2m + r / 3m + r, coef[r] = w / m * 2 (|g| - 1) / (2 eps),
    // st[4] += w (|g| - 1)^2 / m (as gan.sc's gan_gp_core_).
    "extern \"C\" __global__ void gp_prep(const float* g, const float* xh, float* stack, float* coef, float* st, unsigned m,\n"
    "                                   unsigned d, float w, float eps) {\n"
    "    unsigned r = blockIdx.x;\n"
    "    const float* gr = g + (unsigned long long)r * d;\n"
    "    float q = 0.0f;\n"
    "    for (unsigned c = threadIdx.x; c < d; c += blockDim.x) q += gr[c] * gr[c];\n"
    "    q = block_sum(q);\n"
    "    __shared__ float nrm_s;\n"
    "    if (threadIdx.x == 0) {\n"
    "        float nrm = sqrtf(q);\n"
    "        nrm_s = nrm;\n"
    "        coef[r] = w / (float)m * 2.0f * (nrm - 1.0f) / (2.0f * eps);\n"
    "        atomicAdd(&st[4], w * (nrm - 1.0f) * (nrm - 1.0f) / (float)m);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float inv = nrm_s > 1e-12f ? 1.0f / nrm_s : 0.0f;\n"
    "    for (unsigned c = threadIdx.x; c < d; c += blockDim.x) {\n"
    "        float step = eps * gr[c] * inv;\n"
    "        float x = xh[(unsigned long long)r * d + c];\n"
    "        stack[(unsigned long long)(2 * m + r) * d + c] = x + step;\n"
    "        stack[(unsigned long long)(3 * m + r) * d + c] = x - step;\n"
    "    }\n"
    "}\n"
    "__device__ inline float softplus(float x) { return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))); }\n"
    "__device__ inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }\n"
    // GAN losses on D's scores (gan.sc's gan_d_loss_ / gan_g_loss_): one block.
    // gen 0: s = (real | fake) [m each] (and, for the gradient penalty (kind 4),
    // (xp | xm) with coef), loss into st[2] (+ st[4]); gen 1: s = D(G(z)) [m],
    // loss into st[3]. d = d(loss)/ds.
    "extern \"C\" __global__ void gan_loss(const float* s, float* d, float* st, const float* coef, unsigned m, int kind,\n"
    "                                    int gen) {\n"
    "    float acc = 0.0f;\n"
    "    const float inv = 1.0f / (float)m;\n"
    "    if (gen) {\n"
    "        for (unsigned i = threadIdx.x; i < m; i += blockDim.x) {\n"
    "            float v = s[i];\n"
    "            if (kind == 0) { acc += softplus(-v); d[i] = (sigm(v) - 1.0f) * inv; }\n"
    "            else if (kind == 1) { acc += 0.5f * (v - 1.0f) * (v - 1.0f); d[i] = (v - 1.0f) * inv; }\n"
    "            else { acc -= v; d[i] = -inv; }\n"
    "        }\n"
    "    } else {\n"
    "        for (unsigned i = threadIdx.x; i < 2 * m; i += blockDim.x) {\n"
    "            float v = s[i];\n"
    "            int real = i < m;\n"
    "            if (kind == 0) {\n"
    "                acc += real ? softplus(-v) : softplus(v);\n"
    "                d[i] = (sigm(v) - (real ? 1.0f : 0.0f)) * inv;\n"
    "            } else if (kind == 1) {\n"
    "                float t = real ? 1.0f : 0.0f;\n"
    "                acc += 0.5f * (v - t) * (v - t);\n"
    "                d[i] = (v - t) * inv;\n"
    "            } else if (kind == 3) {\n"
    "                float a = real ? 1.0f - v : 1.0f + v;\n"
    "                if (a > 0.0f) acc += a;\n"
    "                d[i] = a > 0.0f ? (real ? -inv : inv) : 0.0f;\n"
    "            } else {\n"
    "                acc += real ? -v : v;\n"
    "                d[i] = real ? -inv : inv;\n"
    "            }\n"
    "        }\n"
    "        if (kind == 4)\n"
    "            for (unsigned i = threadIdx.x; i < 2 * m; i += blockDim.x)\n"
    "                d[2 * m + i] = i < m ? coef[i] : -coef[i - m];\n"
    "    }\n"
    "    acc = block_sum(acc);\n"
    "    if (threadIdx.x == 0) {\n"
    "        if (gen) st[3] = acc * inv;\n"
    "        else st[2] = acc * inv + (kind == 4 ? st[4] : 0.0f);\n"
    "    }\n"
    "}\n"
    // dst = tau src + (1 - tau) dst over up to 32 buffers (blockIdx.y), tau = 1 -
    // decay; with 'schedule' decay = min(decay, (1 + t) / (10 + t)), t = st[1]
    // (gan.h's averaged generator). The last block of the final launch
    // advances st[0] (the noise counter) and st[1] (steps); st[5] counts blocks.
    "struct EmaTab { float* d[32]; const float* s[32]; unsigned n[32]; };\n"
    "extern \"C\" __global__ void soft_update(EmaTab tab, float* st, float decay, int schedule, int finalLaunch) {\n"
    "#pragma clang fp contract(off)\n"
    "    __shared__ float* dp;\n"
    "    __shared__ const float* sp;\n"
    "    __shared__ unsigned sn;\n"
    "    if (threadIdx.x == 0) { dp = tab.d[blockIdx.y]; sp = tab.s[blockIdx.y]; sn = tab.n[blockIdx.y]; }\n"
    "    __syncthreads();\n"
    "    float dec = decay;\n"
    "    if (schedule) {\n"
    "        float t = st[1];\n"
    "        float warm = (1.0f + t) / (10.0f + t);\n"
    "        dec = decay < warm ? decay : warm;\n"
    "    }\n"
    "    const float tau = schedule == 2 ? decay : 1.0f - dec;   // schedule 2: decay is tau itself\n"
    "    float* dd = dp;\n"
    "    const float* ss = sp;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < sn; i += gridDim.x * blockDim.x)\n"
    "        dd[i] = tau * ss[i] + (1.0f - tau) * dd[i];\n"
    "    if (finalLaunch && last_block((unsigned*)&st[5], gridDim.x * gridDim.y) && threadIdx.x == 0) {\n"
    "        st[0] += 1.0f;\n"
    "        st[1] += 1.0f;\n"
    "    }\n"
    "}\n"
    // ── RL ──
    // Replay rows [slot][obs D | next D | action | reward | terminal]; rows idx[r].
    "extern \"C\" __global__ void replay_gather(const float* rp, const float* idx, float* ob, float* nx, float* act,\n"
    "                                         float* rew, float* term, unsigned B, unsigned D) {\n"
    "    const unsigned W = 2 * D + 3;\n"
    "    GS(t, B * W) {\n"
    "        unsigned r = t / W, c = t % W;\n"
    "        float v = rp[(unsigned long long)idx[r] * W + c];\n"
    "        if (c < D) ob[r * D + c] = v;\n"
    "        else if (c < 2 * D) nx[r * D + c - D] = v;\n"
    "        else if (c == 2 * D) act[r] = v;\n"
    "        else if (c == 2 * D + 1) rew[r] = v;\n"
    "        else term[r] = v;\n"
    "    }\n"
    "}\n"
    // y = r + gamma (1 - terminal) qt[a*], a* = the first largest of qo's row.
    "extern \"C\" __global__ void dqn_target(const float* qo, const float* qt, const float* rew, const float* term, float* y,\n"
    "                                      unsigned B, unsigned A, float gamma) {\n"
    "    GS(r, B) {\n"
    "        unsigned best = 0;\n"
    "        for (unsigned c = 1; c < A; ++c) if (qo[r * A + c] > qo[r * A + best]) best = c;\n"
    "        y[r] = term[r] > 0.5f ? rew[r] : rew[r] + gamma * qt[r * A + best];\n"
    "    }\n"
    "}\n"
    // Huber loss of q[r][a_r] against y[r], mean over rows (one block); d is
    // q's gradient (zero but at the taken action). st[2] = the loss.
    "extern \"C\" __global__ void dqn_loss(const float* q, const float* act, const float* y, float* d, float* st,\n"
    "                                    unsigned B, unsigned A, float delta) {\n"
    "    float acc = 0.0f;\n"
    "    const float inv = 1.0f / (float)B;\n"
    "    for (unsigned t = threadIdx.x; t < B * A; t += blockDim.x) {\n"
    "        unsigned r = t / A, c = t % A;\n"
    "        if (c != (unsigned)act[r]) { d[t] = 0.0f; continue; }\n"
    "        float e = q[t] - y[r];\n"
    "        float ae = fabsf(e);\n"
    "        acc += ae <= delta ? 0.5f * e * e : delta * (ae - 0.5f * delta);\n"
    "        d[t] = fminf(fmaxf(e, -delta), delta) * inv;\n"
    "    }\n"
    "    acc = block_sum(acc);\n"
    "    if (threadIdx.x == 0) st[2] = acc * inv;\n"
    "}\n"
    // out[r] = log softmax(logits[r])[act[r]]
    "extern \"C\" __global__ void pg_logp(const float* lg, const float* act, float* out, unsigned N, unsigned A) {\n"
    "    GS(r, N) {\n"
    "        const float* l = lg + (unsigned long long)r * A;\n"
    "        float mx = l[0];\n"
    "        for (unsigned c = 1; c < A; ++c) mx = fmaxf(mx, l[c]);\n"
    "        float se = 0.0f;\n"
    "        for (unsigned c = 0; c < A; ++c) se += expf(l[c] - mx);\n"
    "        out[r] = l[(unsigned)act[r]] - mx - logf(se);\n"
    "    }\n"
    "}\n"
    // Returns and advantages, backwards over the rollout (one thread; rl.sc's
    // PolicyAgent::update). algo 0: Monte Carlo returns (an episode cut short bootstraps
    // from V(next)); else GAE(lambda).
    "extern \"C\" __global__ void pg_gae(const float* v, const float* vn, const float* rew, const float* term,\n"
    "                                  const float* done, float* adv, float* ret, unsigned N, float gamma, float lam,\n"
    "                                  int algo) {\n"
    "    if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
    "    float G = 0.0f, gae = 0.0f;\n"
    "    for (int t = (int)N - 1; t >= 0; --t) {\n"
    "        if (algo == 0) {\n"
    "            float boot = term[t] > 0.5f ? 0.0f : vn[t];\n"
    "            if (done[t] > 0.5f || t == (int)N - 1) G = rew[t] + gamma * boot;\n"
    "            else G = rew[t] + gamma * G;\n"
    "            ret[t] = G;\n"
    "            adv[t] = G - v[t];\n"
    "        } else {\n"
    "            float notTerm = term[t] > 0.5f ? 0.0f : 1.0f;\n"
    "            float notDone = done[t] > 0.5f ? 0.0f : 1.0f;\n"
    "            float delta = rew[t] + gamma * notTerm * vn[t] - v[t];\n"
    "            gae = delta + gamma * lam * notDone * gae;\n"
    "            adv[t] = gae;\n"
    "            ret[t] = gae + v[t];\n"
    "        }\n"
    "    }\n"
    "}\n"
    // adv = (adv - mean) / (std + 1e-8), population std (one block).
    "extern \"C\" __global__ void pg_norm(float* adv, unsigned N) {\n"
    "    __shared__ float mean_s, sd_s;\n"
    "    float a = 0.0f, q = 0.0f;\n"
    "    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) { a += adv[i]; q += adv[i] * adv[i]; }\n"
    "    a = block_sum(a);\n"
    "    q = block_sum(q);\n"
    "    if (threadIdx.x == 0) {\n"
    "        double m = (double)a / (double)N;\n"
    "        mean_s = (float)m;\n"
    "        sd_s = (float)(sqrt(fmax((double)q / (double)N - m * m, 0.0)) + 1e-8);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) adv[i] = (adv[i] - mean_s) / sd_s;\n"
    "}\n"
    // Minibatch rows idx[cur .. cur+m) of the rollout; the last block moves
    // the cursor (cur[0]) past them (cur[1]: block count).
    "extern \"C\" __global__ void pg_gather(const float* idx, float* cur, const float* obs, const float* act, const float* adv,\n"
    "                                     const float* ret, const float* lpo, float* mo, float* ma, float* madv, float* mret,\n"
    "                                     float* mlpo, unsigned m, unsigned D) {\n"
    "    unsigned base = (unsigned)cur[0];\n"
    "    const unsigned W = D + 4;\n"
    "    GS(t, m * W) {\n"
    "        unsigned r = t / W, c = t % W;\n"
    "        unsigned long long s = (unsigned long long)idx[base + r];\n"
    "        if (c < D) mo[r * D + c] = obs[s * D + c];\n"
    "        else if (c == D) ma[r] = act[s];\n"
    "        else if (c == D + 1) madv[r] = adv[s];\n"
    "        else if (c == D + 2) mret[r] = ret[s];\n"
    "        else mlpo[r] = lpo[s];\n"
    "    }\n"
    "    if (last_block((unsigned*)&cur[1], gridDim.x) && threadIdx.x == 0) cur[0] = (float)(base + m);\n"
    "}\n"
    // The policy-gradient loss of rl.sc's pg_step_ (one block): policy term
    // (PPO's clipped surrogate for algo 2, else -mean(logp adv)) + vc * MSE of
    // the value - ec * entropy; dl / dv its gradients for the logits / values.
    // st[2] = loss, st[3] = entropy.
    "extern \"C\" __global__ void pg_loss(const float* lg, const float* v, const float* act, const float* adv,\n"
    "                                   const float* ret, const float* lpo, float* dl, float* dv, float* st, unsigned m,\n"
    "                                   unsigned A, int algo, float clipEps, float vc, float ec) {\n"
    "    float lp = 0.0f, lv = 0.0f, le = 0.0f;\n"
    "    const float inv = 1.0f / (float)m;\n"
    "    for (unsigned r = threadIdx.x; r < m; r += blockDim.x) {\n"
    "        const float* l = lg + (unsigned long long)r * A;\n"
    "        float mx = l[0];\n"
    "        for (unsigned c = 1; c < A; ++c) mx = fmaxf(mx, l[c]);\n"
    "        float se = 0.0f;\n"
    "        for (unsigned c = 0; c < A; ++c) se += expf(l[c] - mx);\n"
    "        float lse = mx + logf(se);\n"
    "        float H = 0.0f;\n"
    "        for (unsigned c = 0; c < A; ++c) { float lpc = l[c] - lse; H -= expf(lpc) * lpc; }\n"
    "        unsigned a = (unsigned)act[r];\n"
    "        float logp = l[a] - lse;\n"
    "        float dlogp;\n"
    "        if (algo == 2) {\n"
    "            float ratio = expf(logp - lpo[r]);\n"
    "            float cl = fminf(fmaxf(ratio, 1.0f - clipEps), 1.0f + clipEps);\n"
    "            float s1 = ratio * adv[r], s2 = cl * adv[r];\n"
    "            lp -= fminf(s1, s2);\n"
    "            dlogp = (s1 <= s2 || cl == ratio) ? -adv[r] * ratio * inv : 0.0f;\n"
    "        } else {\n"
    "            lp -= logp * adv[r];\n"
    "            dlogp = -adv[r] * inv;\n"
    "        }\n"
    "        le += H;\n"
    "        for (unsigned c = 0; c < A; ++c) {\n"
    "            float lpc = l[c] - lse, p = expf(lpc);\n"
    "            dl[(unsigned long long)r * A + c] = dlogp * ((c == a ? 1.0f : 0.0f) - p) + ec * p * (lpc + H) * inv;\n"
    "        }\n"
    "        float e = v[r] - ret[r];\n"
    "        lv += e * e;\n"
    "        dv[r] = vc * 2.0f * e * inv;\n"
    "    }\n"
    "    lp = block_sum(lp);\n"
    "    lv = block_sum(lv);\n"
    "    le = block_sum(le);\n"
    "    if (threadIdx.x == 0) {\n"
    "        st[2] = lp * inv + vc * (lv * inv) - ec * (le * inv);\n"
    "        st[3] = le * inv;\n"
    "    }\n"
    "}\n"
    // ── Skips, resize, dropout, spectral norm ──
    // z = a + b + bias[channel] (b, bias may be null), h = act(z); a may be z.
    "extern \"C\" __global__ void add_act(const act_t* a, const act_t* b, const float* bias, act_t* z, act_t* h, unsigned n,\n"
    "                                   unsigned plane, unsigned C, int act) {\n"
    "    GS(i, n) {\n"
    "        float v = LD(a, i);\n"
    "        if (b) v += LD(b, i);\n"
    "        if (bias) v += bias[(i / plane) % C];\n"
    "        if (z != h) ST(z, i, v);\n"
    "        ST(h, i, act_f(v, act));\n"
    "    }\n"
    "}\n"
    // out = a + b (out may be a or b), or out = a * b
    "extern \"C\" __global__ void ew2(const act_t* a, const act_t* b, act_t* out, unsigned n, int mul) {\n"
    "    GS(i, n) ST(out, i, mul ? LD(a, i) * LD(b, i) : LD(a, i) + LD(b, i));\n"
    "}\n"
    // Sparse resampling along one axis with CSR tap lists (start[o] .. start[o+1]
    // into tidx / tw): x viewed as [outer][inLen][inner] -> out [outer][outLen][inner],
    // out[a][o][c] = sum_k tw[k] x[a][tidx[k]][c].
    "extern \"C\" __global__ void taps_apply(const act_t* x, act_t* out, const float* start, const float* tidx, const float* tw,\n"
    "                                      unsigned outer, unsigned inLen, unsigned outLen, unsigned inner) {\n"
    "    GS(t, outer * outLen * inner) {\n"
    "        unsigned c = t % inner, o = (t / inner) % outLen, a = t / (inner * outLen);\n"
    "        const act_t* src = x + (unsigned long long)a * inLen * inner + c;\n"
    "        float s = 0.0f;\n"
    "        for (unsigned k = (unsigned)start[o]; k < (unsigned)start[o + 1]; ++k)\n"
    "            s += tw[k] * LD(src, (unsigned long long)tidx[k] * inner);\n"
    "        ST(out, t, s);\n"
    "    }\n"
    "}\n"
    // Dropout: per unit (a value, or a plane of 'plane' values for dropout2d)
    // mask = 0 with probability p else 1 / (1 - p) from a counter-based hash of
    // (seed, draw counter ctr[0], unit); out = x * mask. The last block
    // advances ctr[0] (ctr[1]: block count).
    "extern \"C\" __global__ void dropout_fwd(const act_t* x, act_t* out, act_t* mask, unsigned n, unsigned plane, float p,\n"
    "                                       unsigned long long seed, float* ctr) {\n"
    "    unsigned long long c = (unsigned long long)ctr[0];\n"
    "    float keep = p >= 1.0f ? 0.0f : 1.0f / (1.0f - p);\n"
    "    GS(i, n) {\n"
    "        unsigned u = i / plane;\n"
    "        float r = (float)(mix64(seed ^ mix64((c << 32) | u)) >> 40) * 5.9604645e-8f;\n"
    "        float m = r < p ? 0.0f : keep;\n"
    "        ST(mask, i, m);\n"
    "        ST(out, i, LD(x, i) * m);\n"
    "    }\n"
    "    if (last_block((unsigned*)&ctr[1], gridDim.x) && threadIdx.x == 0) ctr[0] += 1.0f;\n"
    "}\n"
    // v = v / max(|v|, 1e-12) (one block); with 'dot', st[0] = max(u . v, 1e-12) instead.
    "extern \"C\" __global__ void sn_vec(float* v, const float* u, float* st, unsigned n, int dot) {\n"
    "    __shared__ float s_s;\n"
    "    float q = 0.0f;\n"
    "    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) q += dot ? u[i] * v[i] : v[i] * v[i];\n"
    "    q = block_sum(q);\n"
    "    if (threadIdx.x == 0) {\n"
    "        if (dot) st[0] = q < 1e-12f ? 1e-12f : q;\n"
    "        else s_s = 1.0f / fmaxf(sqrtf(q), 1e-12f);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    if (!dot) for (unsigned i = threadIdx.x; i < n; i += blockDim.x) v[i] *= s_s;\n"
    "}\n"
    "extern \"C\" __global__ void sn_scale(const float* w, float* wsn, unsigned n, const float* st) {\n"
    "    float inv = 1.0f / st[0];\n"
    "    GS(i, n) wsn[i] = w[i] * inv;\n"
    "}\n"
    // The gradient through W / sigma: g = g / sigma - (sum g w) / sigma^2 u v^T.
    // Pass 1 (one block per slice): partial sums of g w into part, the last
    // block their total into st[1]; pass 2 the update.
    "extern \"C\" __global__ void sn_dot(const float* g, const float* w, float* part, float* st, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    GS(i, n) s += g[i] * w[i];\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) part[blockIdx.x] = s;\n"
    "    if (last_block((unsigned*)&st[3], gridDim.x) && threadIdx.x == 0) {\n"
    "        float t = 0.0f;\n"
    "        for (unsigned k = 0; k < gridDim.x; ++k) t += part[k];\n"
    "        st[1] = t;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void sn_fix(float* g, const float* u, const float* v, const float* st, unsigned rows, unsigned cols) {\n"
    "    float sigma = st[0], inv = 1.0f / sigma, k = st[1] / (sigma * sigma);\n"
    "    GS(i, rows * cols) g[i] = g[i] * inv - k * u[i / cols] * v[i % cols];\n"
    "}\n"
    // ── Split-K weight gradients ──
    // part[z][m][n] = sum over z's k-chunk of a[m][k] b[n][k]: 32 x 32 tiles, 2 x 2 per thread.
    "extern \"C\" __global__ __launch_bounds__(256) void gemm_nt_splitk(const float* a, const float* b, float* part,\n"
    "                                                                 unsigned m, unsigned n, unsigned k, unsigned kc) {\n"
    "    __shared__ float As[32][33], Bs[32][33];\n"
    "    const unsigned tx = threadIdx.x % 16, ty = threadIdx.x / 16;\n"
    "    const unsigned row0 = blockIdx.y * 32, col0 = blockIdx.x * 32;\n"
    "    const unsigned kBeg = blockIdx.z * kc, kEnd = min(k, kBeg + kc);\n"
    "    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;\n"
    "    for (unsigned kt = kBeg; kt < kEnd; kt += 32) {\n"
    "        for (unsigned e = threadIdx.x; e < 1024; e += 256) {\n"
    "            unsigned r = e / 32, kk = e % 32, kg = kt + kk;\n"
    "            As[r][kk] = (row0 + r < m && kg < kEnd) ? a[(unsigned long long)(row0 + r) * k + kg] : 0.0f;\n"
    "            Bs[r][kk] = (col0 + r < n && kg < kEnd) ? b[(unsigned long long)(col0 + r) * k + kg] : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll 8\n"
    "        for (unsigned kk = 0; kk < 32; ++kk) {\n"
    "            float a0 = As[ty * 2][kk], a1 = As[ty * 2 + 1][kk], b0 = Bs[tx * 2][kk], b1 = Bs[tx * 2 + 1][kk];\n"
    "            acc00 += a0 * b0; acc01 += a0 * b1; acc10 += a1 * b0; acc11 += a1 * b1;\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    unsigned r = row0 + ty * 2, c = col0 + tx * 2;\n"
    "    float* p = part + (unsigned long long)blockIdx.z * m * n;\n"
    "    if (r < m && c < n) p[r * n + c] = acc00;\n"
    "    if (r < m && c + 1 < n) p[r * n + c + 1] = acc01;\n"
    "    if (r + 1 < m && c < n) p[(r + 1) * n + c] = acc10;\n"
    "    if (r + 1 < m && c + 1 < n) p[(r + 1) * n + c + 1] = acc11;\n"
    "}\n"
    // c = sum over z of part[z] (fixed order)
    "extern \"C\" __global__ void splitk_reduce(const float* part, float* c, unsigned mn, unsigned S) {\n"
    "    GS(i, mn) {\n"
    "        float s = 0.0f;\n"
    "        for (unsigned z = 0; z < S; ++z) s += part[(unsigned long long)z * mn + i];\n"
    "        c[i] = s;\n"
    "    }\n"
    "}\n"
    // ── Batch norm, spread over blocks ──
    // Unit u = sg * C + c: channel c's planes in segment sg's rows (B/S images),
    // n = (B/S)*P values at (b*C + c)*P + p (NCHW). cb (optional) is the bias of
    // a convolution folded into this norm: its input is x + cb[c]. stats[u*8 +
    // ...]: 0 mean, 1 1/sqrt(var+eps), 2 unbiased var, 3 sum dy, 4 sum dy*xhat.
    // part: 4 values per (unit, slice), then (at a fixed offset) a counter per unit.
    "struct BnArgs {\n"
    "    const act_t* x; const act_t* dy; const act_t* src; float* stats; float* part; float* rm; float* rv;\n"
    "    float* dgamma; float* dbeta; const float* gamma; const float* beta; const float* cb; float* cgb;\n"
    "    act_t* z; act_t* h;\n"
    "    unsigned C, B, P, S;\n"
    "    float eps, momentum;\n"
    "    int act, training;\n"
    "};\n"
    "__device__ inline unsigned long long bn_index(const BnArgs& a, unsigned sg, unsigned c, unsigned e) {\n"
    "    return ((unsigned long long)(sg * (a.B / a.S) + e / a.P) * a.C + c) * a.P + e % a.P;\n"
    "}\n"
    // Forward statistics in one launch: each block its slice's count, mean and
    // sum of squared deviations (two passes over its own values), the unit's last
    // block merges them (Chan et al.) -- the CPU's two-pass accuracy without a
    // second launch. With S == 1 it also moves the running statistics.
    "extern \"C\" __global__ void bn_stats(BnArgs a) {\n"
    "    const unsigned u = blockIdx.x, c = u % a.C, sg = u / a.C, n = (a.B / a.S) * a.P, slices = gridDim.y;\n"
    "    const float cb = a.cb ? a.cb[c] : 0.0f;\n"
    "    __shared__ float m_s;\n"
    "    float s = 0.0f, cnt = 0.0f;\n"
    "    for (unsigned e = blockIdx.y * blockDim.x + threadIdx.x; e < n; e += slices * blockDim.x) {\n"
    "        s += LD(a.x, bn_index(a, sg, c, e)) + cb;\n"
    "        cnt += 1.0f;\n"
    "    }\n"
    "    s = block_sum(s);\n"
    "    cnt = block_sum(cnt);\n"
    "    if (threadIdx.x == 0) m_s = cnt > 0.0f ? s / cnt : 0.0f;\n"
    "    __syncthreads();\n"
    "    const float mb = m_s;\n"
    "    float q = 0.0f;\n"
    "    for (unsigned e = blockIdx.y * blockDim.x + threadIdx.x; e < n; e += slices * blockDim.x) {\n"
    "        float d = LD(a.x, bn_index(a, sg, c, e)) + cb - mb;\n"
    "        q += d * d;\n"
    "    }\n"
    "    q = block_sum(q);\n"
    "    float* pp = a.part + (u * slices + blockIdx.y) * 4;\n"
    "    if (threadIdx.x == 0) { pp[0] = cnt; pp[1] = mb; pp[2] = q; }\n"
    "    unsigned* cntr = (unsigned*)(a.part + 4ull * a.C * 8 * 64);\n"
    "    if (last_block(&cntr[u], slices) && threadIdx.x == 0) {\n"
    "        float N = 0.0f, sm = 0.0f;\n"
    "        for (unsigned k = 0; k < slices; ++k) { const float* r = a.part + (u * slices + k) * 4; N += r[0]; sm += r[0] * r[1]; }\n"
    "        float mean = sm / N, m2 = 0.0f;\n"
    "        for (unsigned k = 0; k < slices; ++k) {\n"
    "            const float* r = a.part + (u * slices + k) * 4;\n"
    "            float d = r[1] - mean;\n"
    "            m2 += r[2] + r[0] * d * d;\n"
    "        }\n"
    "        float var = m2 / N;\n"
    "        float unb = n > 1 ? m2 / (float)(n - 1) : var;\n"
    "        float* st = a.stats + u * 8;\n"
    "        st[0] = mean; st[1] = 1.0f / sqrtf(var + a.eps); st[2] = unb;\n"
    "        if (a.S == 1) {\n"
    "            a.rm[c] = (1.0f - a.momentum) * a.rm[c] + a.momentum * mean;\n"
    "            a.rv[c] = (1.0f - a.momentum) * a.rv[c] + a.momentum * unb;\n"
    "        }\n"
    "    }\n"
    "}\n"
    // The output gradient through the norm's own activation: dy act'(src).
    "__device__ inline float bn_dy(const BnArgs& a, unsigned long long i) {\n"
    "    float g = LD(a.dy, i);\n"
    "    return a.act ? g * act_d(LD(a.src, i), a.act) : g;\n"
    "}\n"
    // Backward sums (sum dy, sum dy*xhat) per unit; with S == 1 the last block
    // also writes dgamma / dbeta and a folded convolution's bias gradient (the
    // sum of dx: 0 with batch statistics, gamma inv sum dy with running ones).
    "extern \"C\" __global__ void bn_grads(BnArgs a) {\n"
    "    const unsigned u = blockIdx.x, c = u % a.C, sg = u / a.C, n = (a.B / a.S) * a.P, slices = gridDim.y;\n"
    "    const float cb = a.cb ? a.cb[c] : 0.0f;\n"
    "    const float mean = a.stats[u * 8], inv = a.stats[u * 8 + 1];\n"
    "    float s0 = 0.0f, s1 = 0.0f;\n"
    "    for (unsigned e = blockIdx.y * blockDim.x + threadIdx.x; e < n; e += slices * blockDim.x) {\n"
    "        unsigned long long i = bn_index(a, sg, c, e);\n"
    "        float g = bn_dy(a, i);\n"
    "        s0 += g;\n"
    "        s1 += g * (LD(a.x, i) + cb - mean) * inv;\n"
    "    }\n"
    "    s0 = block_sum(s0);\n"
    "    s1 = block_sum(s1);\n"
    "    float* pp = a.part + (u * slices + blockIdx.y) * 4;\n"
    "    if (threadIdx.x == 0) { pp[0] = s0; pp[1] = s1; }\n"
    "    unsigned* cntr = (unsigned*)(a.part + 4ull * a.C * 8 * 64);\n"
    "    if (last_block(&cntr[u], slices) && threadIdx.x == 0) {\n"
    "        float t0 = 0.0f, t1 = 0.0f;\n"
    "        for (unsigned k = 0; k < slices; ++k) { t0 += a.part[(u * slices + k) * 4]; t1 += a.part[(u * slices + k) * 4 + 1]; }\n"
    "        a.stats[u * 8 + 3] = t0;\n"
    "        a.stats[u * 8 + 4] = t1;\n"
    "        if (a.S == 1) {\n"
    "            a.dgamma[c] = t1;\n"
    "            a.dbeta[c] = t0;\n"
    "            if (a.cgb) a.cgb[c] = a.training ? 0.0f : a.gamma[c] * inv * t0;\n"
    "        }\n"
    "    }\n"
    "}\n"
    // One thread per channel. phase 0: running statistics from each segment in
    // order (training, S > 1); 1: dgamma / dbeta (and a folded bias's gradient)
    // summed over segments (S > 1); 2: inference statistics (every segment's
    // mean / inv from the running ones).
    "extern \"C\" __global__ void bn_finish(BnArgs a, int phase) {\n"
    "    GS(c, a.C) {\n"
    "        if (phase == 0) {\n"
    "            for (unsigned sg = 0; sg < a.S; ++sg) {\n"
    "                const float* st = a.stats + (sg * a.C + c) * 8;\n"
    "                a.rm[c] = (1.0f - a.momentum) * a.rm[c] + a.momentum * st[0];\n"
    "                a.rv[c] = (1.0f - a.momentum) * a.rv[c] + a.momentum * st[2];\n"
    "            }\n"
    "        } else if (phase == 1) {\n"
    "            float g = 0.0f, b = 0.0f, cg = 0.0f;\n"
    "            for (unsigned sg = 0; sg < a.S; ++sg) {\n"
    "                const float* st = a.stats + (sg * a.C + c) * 8;\n"
    "                g += st[4]; b += st[3];\n"
    "                cg += a.gamma[c] * st[1] * st[3];\n"
    "            }\n"
    "            a.dgamma[c] = g;\n"
    "            a.dbeta[c] = b;\n"
    "            if (a.cgb) a.cgb[c] = a.training ? 0.0f : cg;\n"
    "        } else {\n"
    "            float inv = 1.0f / sqrtf(a.rv[c] + a.eps);\n"
    "            for (unsigned sg = 0; sg < a.S; ++sg) { a.stats[(sg * a.C + c) * 8] = a.rm[c]; a.stats[(sg * a.C + c) * 8 + 1] = inv; }\n"
    "        }\n"
    "    }\n"
    "}\n"
    // Elementwise: y = act(gamma (x + cb - mean) inv + beta) (mode 0), or the
    // input gradient into h (mode 1; the activation's gradient applied first).
    "extern \"C\" __global__ void bn_apply(BnArgs a, int mode) {\n"
    "    const unsigned Bs = a.B / a.S;\n"
    "    GS(t, a.C * a.B * a.P) {\n"
    "        unsigned c = (t / a.P) % a.C, sg = (t / (a.C * a.P)) / Bs;\n"
    "        const float* st = a.stats + (sg * a.C + c) * 8;\n"
    "        const float mean = st[0], inv = st[1];\n"
    "        const float xv = LD(a.x, t) + (a.cb ? a.cb[c] : 0.0f);\n"
    "        if (mode == 0) {\n"
    "            float v = a.gamma[c] * (xv - mean) * inv + a.beta[c];\n"
    "            if (a.z != a.h) ST(a.z, t, v);\n"
    "            ST(a.h, t, act_f(v, a.act));\n"
    "        } else {\n"
    "            const float n = (float)(Bs * a.P), k = a.gamma[c] * inv, g = bn_dy(a, t);\n"
    "            ST(a.h, t, a.training ? k / n * (n * g - st[3] - (xv - mean) * inv * st[4]) : k * g);\n"
    "        }\n"
    "    }\n"
    "}\n"
    // The im2col engine's GEMM result cm [C][B*P] (channel-major) to NCHW:
    // h = act(cm + bias[c] (+ add)), z the pre-activation copy when separate.
    "extern \"C\" __global__ void cm_out(const float* cm, const float* add, const float* bias, float* z, float* h, unsigned C,\n"
    "                                  unsigned B, unsigned P, int act) {\n"
    "    GS(t, C * B * P) {\n"
    "        unsigned p = t % P, c = (t / P) % C, b = t / (P * C);\n"
    "        float v = cm[((unsigned long long)c * B + b) * P + p];\n"
    "        if (bias) v += bias[c];\n"
    "        if (add) v += add[t];\n"
    "        if (z != h) z[t] = v;\n"
    "        h[t] = act_f(v, act);\n"
    "    }\n"
    "}\n";
}

#define NR_BIAS_ACT 0
#define NR_ACT_GRAD 1
#define NR_IM2COL 2
#define NR_COL2IM 3
#define NR_SWAP01 4
#define NR_CE 5
#define NR_BCE 6
#define NR_MSE 7
#define NR_VAE_FWD 8
#define NR_VAE_BWD 9
#define NR_SUMSQ 10
#define NR_OPT 11
#define NR_GATHER 12
#define NR_SMALL 13
#define NR_GEMM_NT 14
#define NR_ACT_ROWS 15
#define NR_ACT_EW 16
#define NR_POOL_FWD 17
#define NR_POOL_BWD 18
#define NR_GPOOL_FWD 19
#define NR_GPOOL_BWD 20
#define NR_UP_FWD 21
#define NR_UP_BWD 22
#define NR_CM_OUT 23
#define NR_GN_FWD 24
#define NR_GN_BWD 25
#define NR_GN_PARAMS 26
#define NR_APPEND 27
#define NR_SLICE 28
#define NR_NOISE 29
#define NR_FILL 30
#define NR_LERP 31
#define NR_GP_PREP 32
#define NR_GAN_LOSS 33
#define NR_SOFT_UPDATE 34
#define NR_REPLAY_GATHER 35
#define NR_DQN_TARGET 36
#define NR_DQN_LOSS 37
#define NR_PG_LOGP 38
#define NR_PG_GAE 39
#define NR_PG_NORM 40
#define NR_PG_GATHER 41
#define NR_PG_LOSS 42
#define NR_ADD_ACT 43
#define NR_EW2 44
#define NR_TAPS 45
#define NR_DROPOUT 46
#define NR_SN_VEC 47
#define NR_SN_SCALE 48
#define NR_SN_DOT 49
#define NR_SN_FIX 50
#define NR_SPLITK 51
#define NR_SPLITK_RED 52
#define NR_BN_STATS 53
#define NR_BN_FINISH 54
#define NR_BN_APPLY 55
#define NR_BN_GRADS 56
#define NR_CONCAT 57
#define NR_TO_ACT 58
#define NR_FROM_ACT 59
#define NR_COUNT 60

static const char* nr_kernel_name_(int k) {
    if (k == NR_BIAS_ACT) return "bias_act";
    if (k == NR_ACT_GRAD) return "act_grad_bias";
    if (k == NR_IM2COL) return "im2col";
    if (k == NR_COL2IM) return "col2im";
    if (k == NR_SWAP01) return "swap01";
    if (k == NR_CE) return "ce_loss";
    if (k == NR_BCE) return "bce_loss";
    if (k == NR_MSE) return "mse_loss";
    if (k == NR_VAE_FWD) return "vae_fwd";
    if (k == NR_VAE_BWD) return "vae_bwd";
    if (k == NR_SUMSQ) return "opt_sumsq";
    if (k == NR_OPT) return "opt_step";
    if (k == NR_GATHER) return "gather2";
    if (k == NR_GEMM_NT) return "gemm_nt_long";
    if (k == NR_ACT_ROWS) return "act_grad_rows";
    if (k == NR_ACT_EW) return "act_grad_ew";
    if (k == NR_POOL_FWD) return "pool_fwd";
    if (k == NR_POOL_BWD) return "pool_bwd";
    if (k == NR_GPOOL_FWD) return "gpool_fwd";
    if (k == NR_GPOOL_BWD) return "gpool_bwd";
    if (k == NR_UP_FWD) return "upsample_fwd";
    if (k == NR_UP_BWD) return "upsample_bwd";
    if (k == NR_CM_OUT) return "cm_out";
    if (k == NR_GN_FWD) return "gn_fwd";
    if (k == NR_GN_BWD) return "gn_bwd";
    if (k == NR_GN_PARAMS) return "gn_params";
    if (k == NR_APPEND) return "append_labels";
    if (k == NR_SLICE) return "slice_cols";
    if (k == NR_NOISE) return "noise_fill";
    if (k == NR_FILL) return "fill_value";
    if (k == NR_LERP) return "lerp_rows";
    if (k == NR_GP_PREP) return "gp_prep";
    if (k == NR_GAN_LOSS) return "gan_loss";
    if (k == NR_SOFT_UPDATE) return "soft_update";
    if (k == NR_REPLAY_GATHER) return "replay_gather";
    if (k == NR_DQN_TARGET) return "dqn_target";
    if (k == NR_DQN_LOSS) return "dqn_loss";
    if (k == NR_PG_LOGP) return "pg_logp";
    if (k == NR_PG_GAE) return "pg_gae";
    if (k == NR_PG_NORM) return "pg_norm";
    if (k == NR_PG_GATHER) return "pg_gather";
    if (k == NR_PG_LOSS) return "pg_loss";
    if (k == NR_ADD_ACT) return "add_act";
    if (k == NR_EW2) return "ew2";
    if (k == NR_TAPS) return "taps_apply";
    if (k == NR_DROPOUT) return "dropout_fwd";
    if (k == NR_SN_VEC) return "sn_vec";
    if (k == NR_SN_SCALE) return "sn_scale";
    if (k == NR_SN_DOT) return "sn_dot";
    if (k == NR_SN_FIX) return "sn_fix";
    if (k == NR_SPLITK) return "gemm_nt_splitk";
    if (k == NR_SPLITK_RED) return "splitk_reduce";
    if (k == NR_BN_STATS) return "bn_stats";
    if (k == NR_BN_GRADS) return "bn_grads";
    if (k == NR_CONCAT) return "concat_rows";
    if (k == NR_TO_ACT) return "to_act";
    if (k == NR_FROM_ACT) return "from_act";
    if (k == NR_BN_FINISH) return "bn_finish";
    if (k == NR_BN_APPLY) return "bn_apply";
    return "small_net_step";
}

// ── Module state ─────────────────────────────────────────────────────────────
// The source compiles once per activation type (NR_ACT 0 fp32, 1 fp16, 2
// bf16); g_nr_prec picks the module a launch uses (a 16-bit net's passes
// set it, everything else runs fp32).
static int   g_nr_state = 0;      // 0 = not tried, 1 = ready, -1 = unavailable
static int   g_nr_half[3];        // the same for the 16-bit modules
static int   g_nr_prec = 0;
static void* g_nr_stream = (void*)0;
static void* g_nr_fn[3 * NR_COUNT];

static int nr_compile_(int prec) {
    unsafe {
        void* prog = (void*)0;
        if (hiprtcCreateProgram(&prog, nr_kernel_source_(), "safec_nn_rocm.hip", 0,
                                (const char**)0, (const char**)0) != 0) return 0;
        const char* opts[2];
        opts[0] = "-O3";
        opts[1] = prec == 0 ? "-DNR_ACT=0" : (prec == 1 ? "-DNR_ACT=1" : "-DNR_ACT=2");
        if (hiprtcCompileProgram(prog, 2, &opts[0]) != 0) { hiprtcDestroyProgram(&prog); return 0; }
        unsigned long codeSize = 0UL;
        hiprtcGetCodeSize(prog, &codeSize);
        char* code = (char*)malloc(codeSize);
        if (code == (char*)0) { hiprtcDestroyProgram(&prog); return 0; }
        hiprtcGetCode(prog, code);
        hiprtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = hipModuleLoadData(&module, (const void*)code);
        free((void*)code);
        if (loaded != 0) return 0;
        int k = 0;
        while (k < NR_COUNT) {
            if (hipModuleGetFunction(&g_nr_fn[prec * NR_COUNT + k], module, nr_kernel_name_(k)) != 0) return 0;
            k = k + 1;
        }
    }
    return 1;
}

static int nr_ensure_() {
    if (g_nr_state != 0) return g_nr_state == 1;
    g_nr_state = -1;
    if (!rocm_available()) return 0;
    ?&void sh = rocm_stream_handle();
    match (sh) {
        case null: { return 0; }
        case some(s): { unsafe { g_nr_stream = (void*)s; } }
    }
    if (!nr_compile_(0)) return 0;
    g_nr_state = 1;
    return 1;
}

// The module for activation type prec (compiled on first use).
static int nr_ensure_prec_(int prec) {
    if (!nr_ensure_() || prec < 0 || prec > 2) return 0;
    if (prec == 0) return 1;
    if (g_nr_half[prec] == 0) g_nr_half[prec] = nr_compile_(prec) ? 1 : -1;
    return g_nr_half[prec] == 1;
}

static int nr_capturing_() {
    int st = 0;
    unsafe { if (hipStreamIsCapturing(g_nr_stream, &st) != 0) return 0; }
    return st == 1;
}

static int nr_launch_grid_(int k, unsigned long gx, unsigned long gy, &void params) {
    unsafe {
        return hipModuleLaunchKernel(g_nr_fn[g_nr_prec * NR_COUNT + k], (unsigned int)gx, (unsigned int)gy, 1U, NR_BLOCK, 1U, 1U, 0U,
                                     g_nr_stream, (void**)params, (void**)0) == 0;
    }
}

// Grid-stride kernels over n elements.
static int nr_launch_(int k, unsigned long n, &void params) {
    unsigned long blocks = (n + (unsigned long)NR_BLOCK - 1UL) / (unsigned long)NR_BLOCK;
    if (blocks > NR_MAX_BLOCKS) blocks = NR_MAX_BLOCKS;
    if (blocks == 0UL) blocks = 1UL;
    return nr_launch_grid_(k, blocks, 1UL, params);
}

static int nr_fits_(struct RocmBuf b, unsigned long n) { return b.dev != 0UL && b.len >= n && n <= 4294967295UL; }

static struct RocmBuf nr_none_() {
    struct RocmBuf b;
    b.dev = 0UL;
    b.len = 0UL;
    return b;
}

static struct RocmBuf nr_view_(struct RocmBuf b, unsigned long n) {
    struct RocmBuf v = b;
    if (n < v.len) v.len = n;
    return v;
}

// rocm_write from host memory that may go out of scope: wait for the copy.
static int nr_write_(struct RocmBuf dst, const &float host, unsigned long n) {
    return rocm_write(dst, host, n) && rocm_sync();
}

// n activations (of the current module's type) from src to dst.
static int nr_copy_(struct RocmBuf src, struct RocmBuf dst, unsigned long n) {
    return rocm_copy(src, 0UL, dst, 0UL, g_nr_prec != 0 ? (n + 1UL) / 2UL : n);
}

// The first n 16-bit values of b (a RocmBuf holding them).
static struct RocmHalfBuf nr_hv_(struct RocmBuf b, unsigned long n) {
    struct RocmHalfBuf v;
    v.dev = b.dev;
    v.len = n;
    v.dtype = g_nr_prec;
    return v;
}

// float -> activation (toAct) or back, times *kp (1 / *kp with inv) when kp is set.
static int nr_cvt_(int toAct, struct RocmBuf src, struct RocmBuf dst, unsigned long n, struct RocmBuf kp, int inv) {
    if (!nr_fits_(src, toAct ? n : (n + 1UL) / 2UL) || !nr_fits_(dst, toAct ? (n + 1UL) / 2UL : n)) return 0;
    unsafe {
        void* ps = (void*)src.dev; void* pd = (void*)dst.dev; void* pk = (void*)kp.dev;
        unsigned int nn = (unsigned int)n; int iv = inv;
        void* p[5];
        p[0] = (void*)&ps; p[1] = (void*)&pd; p[2] = (void*)&nn; p[3] = (void*)&pk; p[4] = (void*)&iv;
        return nr_launch_(toAct ? NR_TO_ACT : NR_FROM_ACT, n, (&void)&p[0]);
    }
}

// ── Kernel wrappers ──────────────────────────────────────────────────────────

static int nr_bias_act_(struct RocmBuf z, struct RocmBuf h, struct RocmBuf b, unsigned long n, unsigned long plane,
                        unsigned long C, int act) {
    if (!nr_fits_(z, n) || !nr_fits_(h, n) || (b.dev != 0UL && !nr_fits_(b, C))) return 0;   // b may be absent
    unsafe {
        void* pz = (void*)z.dev; void* ph = (void*)h.dev; void* pb = (void*)b.dev;
        unsigned int nn = (unsigned int)n; unsigned int pl = (unsigned int)plane; unsigned int cc = (unsigned int)C;
        int a = act;
        void* p[7];
        p[0] = (void*)&pz; p[1] = (void*)&ph; p[2] = (void*)&pb; p[3] = (void*)&nn; p[4] = (void*)&pl;
        p[5] = (void*)&cc; p[6] = (void*)&a;
        return nr_launch_(NR_BIAS_ACT, n, (&void)&p[0]);
    }
}

#define NR_GB_SLICES 64UL   // most blocks per channel in act_grad_bias

// part: the layer's C * (NR_GB_SLICES + 1) scratch floats (partials and counters).
static int nr_act_grad_(struct RocmBuf d, struct RocmBuf src, struct RocmBuf gb, struct RocmBuf part,
                        unsigned long count, unsigned long plane, unsigned long C, int act) {
    if (!nr_fits_(d, count * C) || !nr_fits_(src, count * C) || !nr_fits_(gb, C) ||
        !nr_fits_(part, C * (NR_GB_SLICES + 1UL)) || C > 65535UL) return 0;
    unsigned long slices = (count + 1023UL) / 1024UL;
    if (slices > NR_GB_SLICES) slices = NR_GB_SLICES;
    if (slices == 0UL) slices = 1UL;
    unsafe {
        void* pd = (void*)d.dev; void* ps = (void*)src.dev; void* pg = (void*)gb.dev; void* pp = (void*)part.dev;
        unsigned int cn = (unsigned int)count; unsigned int pl = (unsigned int)plane; unsigned int cc = (unsigned int)C;
        int a = act;
        void* p[8];
        p[0] = (void*)&pd; p[1] = (void*)&ps; p[2] = (void*)&pg; p[3] = (void*)&pp; p[4] = (void*)&cn;
        p[5] = (void*)&pl; p[6] = (void*)&cc; p[7] = (void*)&a;
        return nr_launch_grid_(NR_ACT_GRAD, C, slices, (&void)&p[0]);
    }
}

static int nr_act_grad_rows_(struct RocmBuf d, struct RocmBuf src, struct RocmBuf gb, unsigned long rows,
                             unsigned long C, int act) {
    if (!nr_fits_(d, rows * C) || !nr_fits_(src, rows * C) || !nr_fits_(gb, C)) return 0;
    unsafe {
        void* pd = (void*)d.dev; void* ps = (void*)src.dev; void* pg = (void*)gb.dev;
        unsigned int r = (unsigned int)rows; unsigned int cc = (unsigned int)C;
        int a = act;
        void* p[6];
        p[0] = (void*)&pd; p[1] = (void*)&ps; p[2] = (void*)&pg; p[3] = (void*)&r; p[4] = (void*)&cc; p[5] = (void*)&a;
        return nr_launch_grid_(NR_ACT_ROWS, (C + 15UL) / 16UL, 1UL, (&void)&p[0]);
    }
}

// The kernels' Geom: kernel, stride, padding, dilation per axis.
struct NrGeom_ {
    unsigned int kh; unsigned int kw; unsigned int sh; unsigned int sw;
    unsigned int ph; unsigned int pw; unsigned int dh; unsigned int dw;
};

static struct NrGeom_ nr_conv_geom_(const &Conv2dSpec c) {
    struct NrGeom_ g;
    g.kh = (unsigned int)c.kH; g.kw = (unsigned int)c.kW; g.sh = (unsigned int)c.strideH; g.sw = (unsigned int)c.strideW;
    g.ph = (unsigned int)c.padH; g.pw = (unsigned int)c.padW; g.dh = (unsigned int)c.dilationH;
    g.dw = (unsigned int)c.dilationW;
    return g;
}

static struct NrGeom_ nr_pool_geom_(const &Pool2dSpec p) {
    struct NrGeom_ g;
    g.kh = (unsigned int)p.kH; g.kw = (unsigned int)p.kW; g.sh = (unsigned int)p.strideH; g.sw = (unsigned int)p.strideW;
    g.ph = (unsigned int)p.padH; g.pw = (unsigned int)p.padW; g.dh = 1U; g.dw = 1U;
    return g;
}

// col2im into h (no bias), or a transposed convolution's output
// h = act(col2im + bias) with z the pre-activation copy when separate.
static int nr_col2im_(struct RocmBuf col, struct RocmBuf z, struct RocmBuf h, struct RocmBuf bias, int act,
                      unsigned long C, unsigned long H, unsigned long W, unsigned long oh, unsigned long ow,
                      struct NrGeom_ g, unsigned long B) {
    unsigned long ni = C * B * H * W;
    if (!nr_fits_(h, ni) || !nr_fits_(col, C * (unsigned long)g.kh * (unsigned long)g.kw * B * oh * ow)) return 0;
    if (bias.dev != 0UL && (!nr_fits_(bias, C) || !nr_fits_(z, ni))) return 0;
    unsafe {
        void* pc = (void*)col.dev; void* pz = (void*)z.dev; void* ph = (void*)h.dev; void* pb = (void*)bias.dev;
        int a = act;
        unsigned int a0 = (unsigned int)C; unsigned int a1 = (unsigned int)H; unsigned int a2 = (unsigned int)W;
        unsigned int a3 = (unsigned int)oh; unsigned int a4 = (unsigned int)ow; unsigned int a5 = (unsigned int)B;
        struct NrGeom_ gg = g;
        void* p[12];
        p[0] = (void*)&pc; p[1] = (void*)&pz; p[2] = (void*)&ph; p[3] = (void*)&pb; p[4] = (void*)&a;
        p[5] = (void*)&a0; p[6] = (void*)&a1; p[7] = (void*)&a2; p[8] = (void*)&a3; p[9] = (void*)&a4;
        p[10] = (void*)&gg; p[11] = (void*)&a5;
        return nr_launch_(NR_COL2IM, ni, (&void)&p[0]);
    }
}

// rocm_gemm, except for c = a b^T with few outputs over a long k (a
// convolution's weight gradient), where rocBLAS's tiles leave most of the
// GPU idle (16 x 9 outputs over k = 2048: 20 us; gemm_nt_long: ~3 us).
static int nr_splitk_gemm_(struct RocmBuf a, struct RocmBuf b, struct RocmBuf c, unsigned long m, unsigned long n,
                           unsigned long k);

// c = a b^T over a long k (a convolution's weight gradient over batch x
// plane) has three candidates: rocBLAS's tiles, k split across blocks
// (split-K), and one wavefront per output (few outputs). Their speeds cross
// over unpredictably with the shape, so the first call for each (m, n, k)
// outside graph capture times them on its own buffers and the fastest is
// kept, as gpu_rocm.sc does for rocBLAS's solutions; during capture an
// untimed shape takes rocBLAS.
#define NR_NT_PLANS 512
static unsigned long g_nt_m[NR_NT_PLANS];
static unsigned long g_nt_n[NR_NT_PLANS];
static unsigned long g_nt_k[NR_NT_PLANS];
static int g_nt_pick[NR_NT_PLANS];
static int g_nt_count = 0;

static int nr_nt_long_(struct RocmBuf a, struct RocmBuf b, struct RocmBuf c, unsigned long m, unsigned long n,
                       unsigned long k) {
    if (!nr_fits_(a, m * k) || !nr_fits_(b, n * k) || !nr_fits_(c, m * n)) return 0;
    unsafe {
        void* pa = (void*)a.dev; void* pb = (void*)b.dev; void* pc = (void*)c.dev;
        unsigned int mm = (unsigned int)m; unsigned int nn = (unsigned int)n; unsigned int kk = (unsigned int)k;
        void* p[6];
        p[0] = (void*)&pa; p[1] = (void*)&pb; p[2] = (void*)&pc; p[3] = (void*)&mm; p[4] = (void*)&nn; p[5] = (void*)&kk;
        // 4 outputs per 256-thread block with 64-lane wavefronts, 8 with 32
        return nr_launch_grid_(NR_GEMM_NT, (m * n + 3UL) / 4UL, 1UL, (&void)&p[0]);
    }
}

// pick 0 rocBLAS, 1 split-K, 2 a wavefront per output
static int nr_nt_run_(int pick, struct RocmBuf a, struct RocmBuf b, struct RocmBuf c, unsigned long m, unsigned long n,
                      unsigned long k) {
    if (pick == 1) return nr_splitk_gemm_(a, b, c, m, n, k);
    if (pick == 2) return nr_nt_long_(a, b, c, m, n, k);
    return rocm_gemm(a, 0, b, 1, c, m, n, k);
}

static double nr_nt_time_(int pick, struct RocmBuf a, struct RocmBuf b, struct RocmBuf c, unsigned long m,
                          unsigned long n, unsigned long k) {
    if (!nr_nt_run_(pick, a, b, c, m, n, k) || !rocm_sync()) return -1.0;
    long long t0 = time_mono_ns();
    int r = 0;
    while (r < 3) { nr_nt_run_(pick, a, b, c, m, n, k); r = r + 1; }
    rocm_sync();
    return (double)(time_mono_ns() - t0);
}

static int nr_gemm_(struct RocmBuf a, int ta, struct RocmBuf b, int tb, struct RocmBuf c,
                    unsigned long m, unsigned long n, unsigned long k) {
    if (ta || !tb || k < 512UL) return rocm_gemm(a, ta, b, tb, c, m, n, k);
    int i = 0;
    while (i < g_nt_count) {
        if (g_nt_m[i] == m && g_nt_n[i] == n && g_nt_k[i] == k) return nr_nt_run_(g_nt_pick[i], a, b, c, m, n, k);
        i = i + 1;
    }
    if (nr_capturing_()) return rocm_gemm(a, ta, b, tb, c, m, n, k);
    int best = 0;
    double bestT = nr_nt_time_(0, a, b, c, m, n, k);
    if (bestT < 0.0) return 0;
    int cand = 1;
    while (cand <= 2) {
        int usable = cand == 1 ? k >= 2048UL : m * n <= 16384UL;
        if (usable) {
            double t = nr_nt_time_(cand, a, b, c, m, n, k);
            if (t >= 0.0 && t < bestT) { bestT = t; best = cand; }
        }
        cand = cand + 1;
    }
    if (g_nt_count < NR_NT_PLANS) {
        g_nt_m[g_nt_count] = m; g_nt_n[g_nt_count] = n; g_nt_k[g_nt_count] = k; g_nt_pick[g_nt_count] = best;
        g_nt_count = g_nt_count + 1;
    }
    return nr_nt_run_(best, a, b, c, m, n, k);
}

// im2col: the channel-major image img [C][B][H][W] -> col [C*kh*kw][B*oh*ow].
static int nr_im2col_(struct RocmBuf img, struct RocmBuf col, unsigned long C, unsigned long H, unsigned long W,
                      unsigned long oh, unsigned long ow, struct NrGeom_ g, unsigned long B) {
    unsigned long nc = C * (unsigned long)g.kh * (unsigned long)g.kw * B * oh * ow;
    if (!nr_fits_(img, C * B * H * W) || !nr_fits_(col, nc)) return 0;
    unsafe {
        void* pa = (void*)img.dev;
        void* pb = (void*)col.dev;
        unsigned int a0 = (unsigned int)C; unsigned int a1 = (unsigned int)H; unsigned int a2 = (unsigned int)W;
        unsigned int a3 = (unsigned int)oh; unsigned int a4 = (unsigned int)ow; unsigned int a5 = (unsigned int)B;
        struct NrGeom_ gg = g;
        void* p[9];
        p[0] = (void*)&pa; p[1] = (void*)&pb; p[2] = (void*)&a0; p[3] = (void*)&a1; p[4] = (void*)&a2;
        p[5] = (void*)&a3; p[6] = (void*)&a4; p[7] = (void*)&gg; p[8] = (void*)&a5;
        return nr_launch_(NR_IM2COL, nc, (&void)&p[0]);
    }
}

// [A][Bd][P] -> [Bd][A][P]
static int nr_swap01_(struct RocmBuf in, struct RocmBuf out, unsigned long A, unsigned long Bd, unsigned long P) {
    unsigned long n = A * Bd * P;
    if (!nr_fits_(in, n) || !nr_fits_(out, n)) return 0;
    unsafe {
        void* pi = (void*)in.dev; void* po = (void*)out.dev;
        unsigned int a = (unsigned int)A; unsigned int b = (unsigned int)Bd; unsigned int pp = (unsigned int)P;
        void* p[5];
        p[0] = (void*)&pi; p[1] = (void*)&po; p[2] = (void*)&a; p[3] = (void*)&b; p[4] = (void*)&pp;
        return nr_launch_(NR_SWAP01, n, (&void)&p[0]);
    }
}

// bce_loss / mse_loss
static int nr_pointwise_loss_(int kernel, struct RocmBuf x, struct RocmBuf t, struct RocmBuf d, struct RocmBuf loss1,
                              unsigned long n, float scale) {
    if (!nr_ensure_() || n == 0UL || !nr_fits_(x, n) || !nr_fits_(t, n) || !nr_fits_(d, n) || !nr_fits_(loss1, 1UL))
        return 0;
    unsafe {
        void* px = (void*)x.dev; void* pt = (void*)t.dev; void* pd = (void*)d.dev; void* pl = (void*)loss1.dev;
        unsigned int nn = (unsigned int)n;
        float sc = scale;
        void* p[6];
        p[0] = (void*)&px; p[1] = (void*)&pt; p[2] = (void*)&pd; p[3] = (void*)&pl; p[4] = (void*)&nn; p[5] = (void*)&sc;
        return nr_launch_(kernel, n, (&void)&p[0]);
    }
}

// ── Losses ───────────────────────────────────────────────────────────────────

int rocm_cross_entropy(struct RocmBuf logits, struct RocmBuf labels, struct RocmBuf d, struct RocmBuf loss1,
                       unsigned long rows, unsigned long classes) {
    if (!nr_ensure_() || rows == 0UL || classes == 0UL || !nr_fits_(logits, rows * classes) ||
        !nr_fits_(labels, rows) || !nr_fits_(d, rows * classes) || !nr_fits_(loss1, 1UL)) return 0;
    unsafe {
        void* px = (void*)logits.dev; void* py = (void*)labels.dev; void* pd = (void*)d.dev;
        void* pl = (void*)loss1.dev;
        unsigned int r = (unsigned int)rows; unsigned int c = (unsigned int)classes;
        void* p[6];
        p[0] = (void*)&px; p[1] = (void*)&py; p[2] = (void*)&pd; p[3] = (void*)&pl; p[4] = (void*)&r; p[5] = (void*)&c;
        return nr_launch_(NR_CE, rows * 64UL, (&void)&p[0]);   // a wavefront (up to 64 lanes) per row
    }
}

int rocm_bce_logits(struct RocmBuf logits, struct RocmBuf target, struct RocmBuf d, struct RocmBuf loss1,
                    unsigned long n, float scale) {
    return nr_pointwise_loss_(NR_BCE, logits, target, d, loss1, n, scale);
}

int rocm_mse(struct RocmBuf pred, struct RocmBuf target, struct RocmBuf d, struct RocmBuf loss1,
             unsigned long n, float scale) {
    return nr_pointwise_loss_(NR_MSE, pred, target, d, loss1, n, scale);
}

// ── Layer kernels ────────────────────────────────────────────────────────────

static int nr_act_ew_(struct RocmBuf d, struct RocmBuf src, unsigned long n, int act) {
    if (act == ACT_NONE) return 1;
    if (!nr_fits_(d, n) || !nr_fits_(src, n)) return 0;
    unsafe {
        void* pd = (void*)d.dev; void* ps = (void*)src.dev;
        unsigned int nn = (unsigned int)n;
        int a = act;
        void* p[4];
        p[0] = (void*)&pd; p[1] = (void*)&ps; p[2] = (void*)&nn; p[3] = (void*)&a;
        return nr_launch_(NR_ACT_EW, n, (&void)&p[0]);
    }
}

// pool_fwd: mode 0 max, 1 average counting the padding, 2 average without it.
static int nr_pool_fwd_(struct RocmBuf x, struct RocmBuf z, struct RocmBuf h, struct RocmBuf idx, unsigned long planes,
                        unsigned long H, unsigned long W, unsigned long oH, unsigned long oW, struct NrGeom_ g,
                        int mode, int act) {
    unsigned long n = planes * oH * oW;
    if (!nr_fits_(x, planes * H * W) || !nr_fits_(z, n) || !nr_fits_(h, n) || (mode == 0 && !nr_fits_(idx, n)))
        return 0;
    unsafe {
        void* px = (void*)x.dev; void* pz = (void*)z.dev; void* ph = (void*)h.dev; void* pi = (void*)idx.dev;
        unsigned int a0 = (unsigned int)planes; unsigned int a1 = (unsigned int)H; unsigned int a2 = (unsigned int)W;
        unsigned int a3 = (unsigned int)oH; unsigned int a4 = (unsigned int)oW;
        struct NrGeom_ gg = g;
        int m = mode; int a = act;
        void* p[12];
        p[0] = (void*)&px; p[1] = (void*)&pz; p[2] = (void*)&ph; p[3] = (void*)&pi; p[4] = (void*)&a0;
        p[5] = (void*)&a1; p[6] = (void*)&a2; p[7] = (void*)&a3; p[8] = (void*)&a4; p[9] = (void*)&gg;
        p[10] = (void*)&m; p[11] = (void*)&a;
        return nr_launch_(NR_POOL_FWD, n, (&void)&p[0]);
    }
}

static int nr_pool_bwd_(struct RocmBuf d, struct RocmBuf idx, struct RocmBuf dx, unsigned long planes, unsigned long H,
                        unsigned long W, unsigned long oH, unsigned long oW, struct NrGeom_ g, int mode) {
    unsigned long n = planes * H * W;
    if (!nr_fits_(d, planes * oH * oW) || !nr_fits_(dx, n)) return 0;
    unsafe {
        void* pd = (void*)d.dev; void* pi = (void*)idx.dev; void* px = (void*)dx.dev;
        unsigned int a0 = (unsigned int)planes; unsigned int a1 = (unsigned int)H; unsigned int a2 = (unsigned int)W;
        unsigned int a3 = (unsigned int)oH; unsigned int a4 = (unsigned int)oW;
        struct NrGeom_ gg = g;
        int m = mode;
        void* p[10];
        p[0] = (void*)&pd; p[1] = (void*)&pi; p[2] = (void*)&px; p[3] = (void*)&a0; p[4] = (void*)&a1;
        p[5] = (void*)&a2; p[6] = (void*)&a3; p[7] = (void*)&a4; p[8] = (void*)&gg; p[9] = (void*)&m;
        return nr_launch_(NR_POOL_BWD, n, (&void)&p[0]);
    }
}

static int nr_gpool_fwd_(struct RocmBuf x, struct RocmBuf z, struct RocmBuf h, struct RocmBuf idx,
                         unsigned long planes, unsigned long P, int mode, int act) {
    if (!nr_fits_(x, planes * P) || !nr_fits_(z, planes) || !nr_fits_(h, planes) || (mode == 0 && !nr_fits_(idx, planes)))
        return 0;
    unsafe {
        void* px = (void*)x.dev; void* pz = (void*)z.dev; void* ph = (void*)h.dev; void* pi = (void*)idx.dev;
        unsigned int a0 = (unsigned int)planes; unsigned int a1 = (unsigned int)P;
        int m = mode; int a = act;
        void* p[8];
        p[0] = (void*)&px; p[1] = (void*)&pz; p[2] = (void*)&ph; p[3] = (void*)&pi; p[4] = (void*)&a0;
        p[5] = (void*)&a1; p[6] = (void*)&m; p[7] = (void*)&a;
        return nr_launch_(NR_GPOOL_FWD, planes, (&void)&p[0]);
    }
}

static int nr_gpool_bwd_(struct RocmBuf d, struct RocmBuf idx, struct RocmBuf dx, unsigned long planes,
                         unsigned long P, int mode) {
    if (!nr_fits_(d, planes) || !nr_fits_(dx, planes * P)) return 0;
    unsafe {
        void* pd = (void*)d.dev; void* pi = (void*)idx.dev; void* px = (void*)dx.dev;
        unsigned int a0 = (unsigned int)planes; unsigned int a1 = (unsigned int)P;
        int m = mode;
        void* p[6];
        p[0] = (void*)&pd; p[1] = (void*)&pi; p[2] = (void*)&px; p[3] = (void*)&a0; p[4] = (void*)&a1; p[5] = (void*)&m;
        return nr_launch_(NR_GPOOL_BWD, planes * P, (&void)&p[0]);
    }
}

static int nr_upsample_(int backward, struct RocmBuf src, struct RocmBuf z, struct RocmBuf dst, unsigned long planes,
                        unsigned long H, unsigned long W, unsigned long f, int act) {
    unsigned long small = planes * H * W;
    unsigned long big = small * f * f;
    if (!nr_fits_(src, backward ? big : small) || !nr_fits_(dst, backward ? small : big)) return 0;
    unsafe {
        void* ps = (void*)src.dev; void* pz = (void*)z.dev; void* pd = (void*)dst.dev;
        unsigned int a0 = (unsigned int)planes; unsigned int a1 = (unsigned int)H; unsigned int a2 = (unsigned int)W;
        unsigned int a3 = (unsigned int)f;
        int a = act;
        void* p[8];
        if (backward) {
            p[0] = (void*)&ps; p[1] = (void*)&pd; p[2] = (void*)&a0; p[3] = (void*)&a1; p[4] = (void*)&a2; p[5] = (void*)&a3;
            return nr_launch_(NR_UP_BWD, small, (&void)&p[0]);
        }
        p[0] = (void*)&ps; p[1] = (void*)&pz; p[2] = (void*)&pd; p[3] = (void*)&a0; p[4] = (void*)&a1;
        p[5] = (void*)&a2; p[6] = (void*)&a3; p[7] = (void*)&a;
        return nr_launch_(NR_UP_FWD, big, (&void)&p[0]);
    }
}

// NCHW h = act(cm + bias (+ add)) from the im2col engine's channel-major cm [C][B*P].
static int nr_cm_out_(struct RocmBuf cm, struct RocmBuf add, struct RocmBuf bias, struct RocmBuf z, struct RocmBuf h,
                      unsigned long C, unsigned long B, unsigned long P, int act) {
    unsigned long n = C * B * P;
    if (!nr_fits_(cm, n) || !nr_fits_(z, n) || !nr_fits_(h, n)) return 0;
    unsafe {
        void* pc = (void*)cm.dev; void* pa = (void*)add.dev; void* pb = (void*)bias.dev; void* pz = (void*)z.dev;
        void* ph = (void*)h.dev;
        unsigned int a0 = (unsigned int)C; unsigned int a1 = (unsigned int)B; unsigned int a2 = (unsigned int)P;
        int ac = act;
        void* p[9];
        p[0] = (void*)&pc; p[1] = (void*)&pa; p[2] = (void*)&pb; p[3] = (void*)&pz; p[4] = (void*)&ph;
        p[5] = (void*)&a0; p[6] = (void*)&a1; p[7] = (void*)&a2; p[8] = (void*)&ac;
        return nr_launch_(NR_CM_OUT, n, (&void)&p[0]);
    }
}

static struct RocmBuf nr_at_(struct RocmBuf b, unsigned long off, unsigned long n) {
    struct RocmBuf v;
    v.dev = b.dev == 0UL || off + n > b.len ? 0UL : b.dev + off * sizeof(float);
    v.len = n;
    return v;
}

// ── Layers ───────────────────────────────────────────────────────────────────

static int nr_needs_z_(int act) { return act == ACT_GELU || act == ACT_SILU; }

static unsigned long nr_in_size_(const &RocmLayer L) { return L.inC * L.inH * L.inW; }
static unsigned long nr_out_size_(const &RocmLayer L) { return L.outC * L.outH * L.outW; }
// Every layer but the fully connected one keeps its activations channel-major.
static int nr_is_cm_(const &RocmLayer L) { return L.kind != ROCM_LAYER_LINEAR; }

struct RocmNet rocm_net_new(unsigned long maxBatch) {
    struct RocmNet n;
    n.layers = vec_new();
    n.maxBatch = maxBatch == 0UL ? 1UL : maxBatch;
    n.inC = 0UL; n.inH = 0UL; n.inW = 0UL;
    n.training = 1;
    n.segments = 1UL;
    n.ok = nr_ensure_();
    n.precision = 0;
    n.scale = nr_none_(); n.xh = nr_none_(); n.dxh = nr_none_(); n.out32 = nr_none_(); n.dout32 = nr_none_();
    return n;
}

int RocmNet::set_input(unsigned long channels, unsigned long height, unsigned long width) {
    if (self.layers.length() != 0UL || channels * height * width == 0UL) { self.ok = 0; return 0; }
    self.inC = channels; self.inH = height; self.inW = width;
    return 1;
}

int RocmNet::set_training(int training) {
    int was = self.training;
    self.training = training ? 1 : 0;
    return was;
}

static void nr_layer_free_(&RocmLayer L) {
    L.w.free(); L.b.free(); L.gw.free(); L.gb.free();
    L.runMean.free(); L.runVar.free(); L.stats.free();
    L.z.free(); L.col.free(); L.cm.free(); L.part.free();
    if ((L.alias & 1) == 0) L.h.free();
    if ((L.alias & 2) == 0) L.dh.free();
    L.wsn.free(); L.aux.free(); L.wh.free();
}

static int nr_alloc_(&RocmBuf b, unsigned long n) {
    *b = rocm_alloc(n);
    return b.dev != 0UL;
}

static int nr_tensor_ok_(const &Tensor t, unsigned long size) {
    unsafe { return t->size == size; }
}

static int nr_upload_(&RocmBuf b, const &Tensor t, unsigned long size) {
    if (!nr_tensor_ok_(t, size) || !nr_alloc_(b, size)) return 0;
    unsafe { return rocm_write(*b, (const &float)(const float*)t->data, size); }
}

// The current output geometry (the last layer's, or the input's); 0 when unknown.
static int nr_cur_geom_(const &RocmNet n, &unsigned long c, &unsigned long h, &unsigned long w) {
    unsigned long count = n.layers.length();
    if (count > 0UL) {
        &RocmLayer prev = n.layers.at(count - 1UL);
        *c = prev.outC; *h = prev.outH; *w = prev.outW;
        return 1;
    }
    *c = n.inC; *h = n.inH; *w = n.inW;
    return n.inC * n.inH * n.inW != 0UL;
}

// A layer template: kind, act and geometry; every buffer empty.
static struct RocmLayer nr_layer_(int kind, int act, unsigned long inC, unsigned long inH, unsigned long inW) {
    struct RocmLayer L;
    L.kind = kind; L.act = act;
    L.inC = inC; L.inH = inH; L.inW = inW;
    L.outC = inC; L.outH = inH; L.outW = inW;
    L.conv = conv2d_spec(1UL, 1UL, 0UL);
    L.pool = pool2d_spec(1UL, 1UL, 0UL);
    L.factor = 1UL; L.momentum = (float)0.1; L.eps = (float)0.00001;
    L.w = nr_none_(); L.b = nr_none_(); L.gw = nr_none_(); L.gb = nr_none_();
    L.runMean = nr_none_(); L.runVar = nr_none_(); L.stats = nr_none_();
    L.z = nr_none_(); L.h = nr_none_(); L.dh = nr_none_(); L.col = nr_none_(); L.cm = nr_none_(); L.part = nr_none_();
    L.slot = 0UL; L.skipC = 0UL; L.skipH = 0UL; L.skipW = 0UL; L.skipAccum = 0;
    L.mode = 0; L.flag = 0; L.prob = (float)0.0; L.seed = 0UL; L.spectral = 0; L.foldBias = 0; L.alias = 0;
    L.wsn = nr_none_(); L.aux = nr_none_(); L.wh = nr_none_();
    return L;
}

// Appends L (kind, act, geometry and any parameters uploaded): checks it
// against the current output, picks the input conversion and allocates its
// buffers. Frees L's buffers and clears n.ok on failure.
static int nr_push_(&RocmNet n, struct RocmLayer L) {
    if (!n.ok) { nr_layer_free_(&L); return 0; }
    // every activation is NCHW (a row per sample), so a layer only needs the
    // previous output's size to match: no layout conversions
    unsigned long pc = 0UL; unsigned long ph = 0UL; unsigned long pw = 0UL;
    if (n.layers.length() == 0UL && n.inC * n.inH * n.inW == 0UL) { n.inC = L.inC; n.inH = L.inH; n.inW = L.inW; }
    nr_cur_geom_(n, &pc, &ph, &pw);
    int ok = pc * ph * pw == nr_in_size_(&L) && nr_out_size_(&L) != 0UL;
    unsigned long B = n.maxBatch;
    unsigned long outN = B * nr_out_size_(&L);
    if ((L.alias & 1) == 0) ok = ok && nr_alloc_(&L.h, outN);
    if ((L.alias & 2) == 0) ok = ok && nr_alloc_(&L.dh, outN);
    if (ok && L.w.dev != 0UL) ok = nr_alloc_(&L.gw, L.w.len) && nr_alloc_(&L.gb, L.b.len);
    if (ok && nr_needs_z_(L.act)) ok = nr_alloc_(&L.z, outN);
    unsigned long kk = L.conv.kH * L.conv.kW;
    if (ok && (L.kind == ROCM_LAYER_CONV || L.kind == ROCM_LAYER_CONVT))
        ok = nr_alloc_(&L.part, L.outC * (NR_GB_SLICES + 1UL)) && rocm_zero(L.part);
    if (ok && (L.kind == ROCM_LAYER_MAXPOOL || L.kind == ROCM_LAYER_GLOBAL_MAXPOOL)) ok = nr_alloc_(&L.stats, outN);
    if (ok && L.kind == ROCM_LAYER_BATCHNORM) ok = nr_alloc_(&L.stats, 8UL * ROCM_MAX_SEGMENTS * L.outC);
    if (ok && L.kind == ROCM_LAYER_GROUPNORM) ok = nr_alloc_(&L.stats, 2UL * B * L.factor);
    // the im2col engine's channel-major scratch: a convolution's GEMM output and
    // output gradient, a transposed convolution's input (then its input gradient)
    if (ok && L.kind == ROCM_LAYER_ADD && L.flag) ok = nr_alloc_(&L.cm, outN);
    unsigned long skipN = B * L.skipC * L.skipH * L.skipW;
    if (ok && L.kind == ROCM_LAYER_SAVE) ok = nr_alloc_(&L.aux, outN);
    if (ok && L.kind == ROCM_LAYER_CONCAT) ok = nr_alloc_(&L.aux, skipN);
    if (ok && L.kind == ROCM_LAYER_ADD && L.flag)
        ok = nr_alloc_(&L.col, L.skipC * B * L.outH * L.outW) && nr_alloc_(&L.aux, skipN) &&
             nr_alloc_(&L.part, L.outC * (NR_GB_SLICES + 1UL)) && rocm_zero(L.part);
    if (ok && L.kind == ROCM_LAYER_RESIZE) {
        unsigned long a = B * L.inC * L.inH * L.outW;
        unsigned long b = B * L.inC * L.outH * L.inW;
        ok = nr_alloc_(&L.col, a > b ? a : b);
    }
    if (ok && L.kind == ROCM_LAYER_DROPOUT) ok = nr_alloc_(&L.stats, outN) && nr_alloc_(&L.aux, 2UL) && rocm_zero(L.aux);
    if (ok && L.kind == ROCM_LAYER_BATCHNORM)
        ok = nr_alloc_(&L.part, L.outC * ROCM_MAX_SEGMENTS * (4UL * NR_BN_SLICES + 1UL)) && rocm_zero(L.part);
    unsigned long count = n.layers.length();
    if (ok && count > 0UL) {
        &RocmLayer prev = n.layers.at(count - 1UL);
        // a batch norm right after a convolution without an activation applies the
        // convolution's bias itself (it cancels in batch statistics): one pass less
        // each way
        if (L.kind == ROCM_LAYER_BATCHNORM && (prev.kind == ROCM_LAYER_CONV || prev.kind == ROCM_LAYER_CONVT) &&
            prev.act == ACT_NONE && prev.outC == L.inC) {
            prev.foldBias = 1;
            L.flag = 1;
            L.slot = count - 1UL;
        }
        // an ADD's input gradient is its own output gradient: the previous layer
        // reads it from the ADD's buffer instead of a copy
        if (L.kind == ROCM_LAYER_ADD && prev.alias == 0 && prev.kind != ROCM_LAYER_SAVE) {
            prev.dh.free();
            prev.dh = L.dh;
            prev.alias = 2;
        }
    }
    // the im2col engine's scratch (with MIOpen it is allocated only if a shape falls back to it)
    if (ok && (L.kind == ROCM_LAYER_CONV || L.kind == ROCM_LAYER_CONVT) && !nr_mio_on_()) {
        n.layers.push(L);   // nr_conv_scratch_ reads the net's maxBatch only
        &RocmLayer last = n.layers.at(n.layers.length() - 1UL);
        ok = nr_conv_scratch_(n, last) && rocm_sync();
        if (!ok) { nr_layer_free_(last); n.layers.pop(); n.ok = 0; return 0; }
        return 1;
    }
    if (ok && L.kind == ROCM_LAYER_ADD && L.flag) ok = nr_splitk_reserve_(L.outC, L.skipC, B * L.outH * L.outW);
    ok = ok && rocm_sync();
    if (!ok) { nr_layer_free_(&L); n.ok = 0; return 0; }
    n.layers.push(L);
    return 1;
}

int RocmNet::add_linear(const &Tensor w, const &Tensor b, int act) {
    unsigned long in;
    unsigned long out;
    unsafe {
        if (w->ndim != 2UL) { self.ok = 0; return 0; }
        in = w->shape[0];
        out = w->shape[1];
    }
    // the input keeps the producer's geometry (for the layout conversion)
    unsigned long c = in; unsigned long h = 1UL; unsigned long wd = 1UL;
    unsigned long pc = 0UL; unsigned long ph = 0UL; unsigned long pw = 0UL;
    if (nr_cur_geom_(self, &pc, &ph, &pw) && pc * ph * pw == in) { c = pc; h = ph; wd = pw; }
    struct RocmLayer L = nr_layer_(ROCM_LAYER_LINEAR, act, c, h, wd);
    L.outC = out; L.outH = 1UL; L.outW = 1UL;
    if (!nr_upload_(&L.w, w, in * out) || !nr_upload_(&L.b, b, out)) { nr_layer_free_(&L); self.ok = 0; return 0; }
    return nr_push_(self, L);
}

// A fully connected layer from host arrays w [in, out] and b [out].
static int nr_add_linear_raw_(&RocmNet n, const &float w, const &float b, unsigned long in, unsigned long out, int act) {
    struct RocmLayer L = nr_layer_(ROCM_LAYER_LINEAR, act, in, 1UL, 1UL);
    L.outC = out; L.outH = 1UL; L.outW = 1UL;
    if (!nr_alloc_(&L.w, in * out) || !nr_alloc_(&L.b, out) || !rocm_write(L.w, w, in * out) ||
        !rocm_write(L.b, b, out)) { nr_layer_free_(&L); n.ok = 0; return 0; }
    return nr_push_(n, L);
}

static int nr_add_conv_(&RocmNet n, int kind, const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                        unsigned long inW, const &Conv2dSpec spec, int act) {
    struct RocmLayer L = nr_layer_(kind, act, inC, inH, inW);
    L.conv = *spec;
    if (L.conv.groups == 0UL) L.conv.groups = 1UL;
    if (L.conv.strideH == 0UL) L.conv.strideH = 1UL;
    if (L.conv.strideW == 0UL) L.conv.strideW = 1UL;
    if (L.conv.dilationH == 0UL) L.conv.dilationH = 1UL;
    if (L.conv.dilationW == 0UL) L.conv.dilationW = 1UL;
    unsigned long outC;
    unsafe { outC = b->size; }
    unsigned long G = L.conv.groups;
    L.outC = outC;
    if (kind == ROCM_LAYER_CONV) {
        L.outH = conv_out_side_ex(inH, L.conv.kH, L.conv.strideH, L.conv.padH, L.conv.dilationH);
        L.outW = conv_out_side_ex(inW, L.conv.kW, L.conv.strideW, L.conv.padW, L.conv.dilationW);
    } else {
        L.outH = conv_transpose_out_side_ex(inH, L.conv.kH, L.conv.strideH, L.conv.padH, L.conv.dilationH,
                                            L.conv.outPadH);
        L.outW = conv_transpose_out_side_ex(inW, L.conv.kW, L.conv.strideW, L.conv.padW, L.conv.dilationW,
                                            L.conv.outPadW);
    }
    unsigned long wn = (inC / G) * outC * L.conv.kH * L.conv.kW;
    if (outC == 0UL || inC % G != 0UL || outC % G != 0UL || L.outH == 0UL || L.outW == 0UL ||
        !nr_upload_(&L.w, w, wn) || !nr_upload_(&L.b, b, outC)) { nr_layer_free_(&L); n.ok = 0; return 0; }
    return nr_push_(n, L);
}

int RocmNet::add_conv2d_ex(const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH, unsigned long inW,
                           const &Conv2dSpec spec, int act) {
    return nr_add_conv_(self, ROCM_LAYER_CONV, w, b, inC, inH, inW, spec, act);
}

int RocmNet::add_conv_transpose2d_ex(const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                                     unsigned long inW, const &Conv2dSpec spec, int act) {
    return nr_add_conv_(self, ROCM_LAYER_CONVT, w, b, inC, inH, inW, spec, act);
}

int RocmNet::add_conv2d(const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH, unsigned long inW,
                        unsigned long kernel, unsigned long stride, unsigned long padding, int act) {
    struct Conv2dSpec s = conv2d_spec(kernel, stride, padding);
    return nr_add_conv_(self, ROCM_LAYER_CONV, w, b, inC, inH, inW, &s, act);
}

int RocmNet::add_conv_transpose2d(const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                                  unsigned long inW, unsigned long kernel, unsigned long stride,
                                  unsigned long padding, int act) {
    struct Conv2dSpec s = conv2d_spec(kernel, stride, padding);
    return nr_add_conv_(self, ROCM_LAYER_CONVT, w, b, inC, inH, inW, &s, act);
}

// A layer that takes the current output's geometry.
static int nr_geom_layer_(&RocmNet n, int kind, int act, &RocmLayer out) {
    unsigned long c = 0UL; unsigned long h = 0UL; unsigned long w = 0UL;
    if (!n.ok || !nr_cur_geom_(n, &c, &h, &w)) { n.ok = 0; return 0; }
    *out = nr_layer_(kind, act, c, h, w);
    return 1;
}

static int nr_add_pool_(&RocmNet n, int kind, const &Pool2dSpec spec, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(n, kind, act, &L)) return 0;
    L.pool = *spec;
    if (L.pool.strideH == 0UL) L.pool.strideH = 1UL;
    if (L.pool.strideW == 0UL) L.pool.strideW = 1UL;
    L.outH = pool_out_side(L.inH, L.pool.kH, L.pool.strideH, L.pool.padH, L.pool.ceilMode);
    L.outW = pool_out_side(L.inW, L.pool.kW, L.pool.strideW, L.pool.padW, L.pool.ceilMode);
    if (L.outH == 0UL || L.outW == 0UL) { n.ok = 0; return 0; }
    return nr_push_(n, L);
}

int RocmNet::add_maxpool(const &Pool2dSpec spec, int act) {
    return nr_add_pool_(self, ROCM_LAYER_MAXPOOL, spec, act);
}

int RocmNet::add_avgpool(const &Pool2dSpec spec, int act) {
    return nr_add_pool_(self, ROCM_LAYER_AVGPOOL, spec, act);
}

int RocmNet::add_global_pool(int max, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, max ? ROCM_LAYER_GLOBAL_MAXPOOL : ROCM_LAYER_GLOBAL_AVGPOOL, act, &L)) return 0;
    L.outH = 1UL; L.outW = 1UL;
    return nr_push_(self, L);
}

int RocmNet::add_upsample(unsigned long factor, int act) {
    struct RocmLayer L;
    if (factor == 0UL || !nr_geom_layer_(self, ROCM_LAYER_UPSAMPLE, act, &L)) { self.ok = 0; return 0; }
    L.factor = factor;
    L.outH = L.inH * factor; L.outW = L.inW * factor;
    return nr_push_(self, L);
}

static int nr_add_reshape_(&RocmNet n, unsigned long channels, unsigned long height, unsigned long width, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(n, ROCM_LAYER_RESHAPE, act, &L)) return 0;
    L.outC = channels; L.outH = height; L.outW = width;
    if (nr_out_size_(&L) != nr_in_size_(&L)) { n.ok = 0; return 0; }
    return nr_push_(n, L);
}

int RocmNet::add_reshape(unsigned long channels, unsigned long height, unsigned long width) {
    return nr_add_reshape_(self, channels, height, width, ACT_NONE);
}

int RocmNet::add_batchnorm(const &Tensor gamma, const &Tensor beta, const &Tensor runMean, const &Tensor runVar,
                           float momentum, float eps, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_BATCHNORM, act, &L)) return 0;
    L.momentum = momentum;
    L.eps = eps;
    unsigned long C = L.inC;
    if (!nr_upload_(&L.w, gamma, C) || !nr_upload_(&L.b, beta, C) || !nr_upload_(&L.runMean, runMean, C) ||
        !nr_upload_(&L.runVar, runVar, C)) { nr_layer_free_(&L); self.ok = 0; return 0; }
    return nr_push_(self, L);
}

int RocmNet::add_groupnorm(const &Tensor gamma, const &Tensor beta, unsigned long groups, float eps, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_GROUPNORM, act, &L)) return 0;
    if (groups == 0UL || L.inC % groups != 0UL) { self.ok = 0; return 0; }
    L.factor = groups;
    L.eps = eps;
    if (!nr_upload_(&L.w, gamma, L.inC) || !nr_upload_(&L.b, beta, L.inC)) { nr_layer_free_(&L); self.ok = 0; return 0; }
    return nr_push_(self, L);
}


// ── Skips, resize, dropout, spectral norm ──

unsigned long RocmNet::add_save() {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_SAVE, ACT_NONE, &L)) return 0UL;
    unsigned long count = self.layers.length();
    if (count > 0UL) {
        // the saved activation is the previous layer's output itself, and its
        // gradient goes straight into the previous layer's (no copies)
        &RocmLayer prev = self.layers.at(count - 1UL);
        L.h = prev.h;
        L.dh = prev.dh;
        L.alias = 3;
    }
    if (!nr_push_(self, L)) return 0UL;
    return self.layers.length() - 1UL;
}

int RocmNet::add_skip(unsigned long slot, ?&Tensor projW, ?&Tensor projB, int act) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_ADD, act, &L)) return 0;
    if (slot >= self.layers.length() || self.layers.at(slot).kind != ROCM_LAYER_SAVE) { self.ok = 0; return 0; }
    &RocmLayer sv = self.layers.at(slot);
    L.slot = slot;
    L.skipC = sv.outC; L.skipH = sv.outH; L.skipW = sv.outW;
    int same = sv.outC == L.inC && sv.outH == L.inH && sv.outW == L.inW;
    if (!same) {
        // projection shortcut: the 1x1 stride that maps the saved size onto the current one
        unsigned long st = 1UL;
        while (st <= sv.outH && (conv_out_side(sv.outH, 1UL, st, 0UL) != L.inH || conv_out_side(sv.outW, 1UL, st, 0UL) != L.inW))
            st = st + 1UL;
        if (st > sv.outH) { self.ok = 0; return 0; }
        L.flag = 1;
        L.conv = conv2d_spec(1UL, st, 0UL);
        int up = 0;
        match (projW) {
            case some(w): {
                match (projB) {
                    case some(b): { up = nr_upload_(&L.w, w, L.inC * L.skipC) && nr_upload_(&L.b, b, L.inC); }
                    case null: { }
                }
            }
            case null: { }
        }
        if (!up) { nr_layer_free_(&L); self.ok = 0; return 0; }
    }
    return nr_push_(self, L);
}

int RocmNet::add_concat(unsigned long slot) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_CONCAT, ACT_NONE, &L)) return 0;
    if (slot >= self.layers.length() || self.layers.at(slot).kind != ROCM_LAYER_SAVE) { self.ok = 0; return 0; }
    &RocmLayer sv = self.layers.at(slot);
    if (sv.outH != L.inH || sv.outW != L.inW) { self.ok = 0; return 0; }
    L.slot = slot;
    L.skipC = sv.outC; L.skipH = sv.outH; L.skipW = sv.outW;
    L.outC = L.inC + sv.outC;
    return nr_push_(self, L);
}

// Resampling taps along one axis, exactly as convnet.sc's cn_taps_.
static unsigned long nr_taps_(unsigned long o, unsigned long in, unsigned long out, int mode, int align, &unsigned long idx,
                              &float w) {
    unsafe {
        unsigned long* ix = (unsigned long*)idx;
        float* wt = (float*)w;
        if (mode == RESIZE_NEAREST) {
            unsigned long i = (unsigned long)floor_f((float)o * ((float)in / (float)out));
            ix[0] = i < in ? i : in - 1UL;
            wt[0] = (float)1.0;
            return 1UL;
        }
        float s;
        if (align) s = out <= 1UL ? (float)0.0 : (float)o * (float)(in - 1UL) / (float)(out - 1UL);
        else s = ((float)o + (float)0.5) * (float)in / (float)out - (float)0.5;
        if (mode == RESIZE_BILINEAR) {
            if (!align && s < (float)0.0) s = (float)0.0;
            unsigned long i0 = (unsigned long)s;
            if (i0 > in - 1UL) i0 = in - 1UL;
            float l = s - (float)i0;
            ix[0] = i0;
            ix[1] = i0 + 1UL < in ? i0 + 1UL : in - 1UL;
            wt[0] = (float)1.0 - l;
            wt[1] = l;
            return 2UL;
        }
        float f = floor_f(s);
        long i = (long)f;
        float t = s - f;
        float A = (float)-0.75;
        float x1 = t + (float)1.0;
        float x2 = (float)1.0 - t;
        float x3 = (float)2.0 - t;
        wt[0] = ((A * x1 - (float)5.0 * A) * x1 + (float)8.0 * A) * x1 - (float)4.0 * A;
        wt[1] = ((A + (float)2.0) * t - (A + (float)3.0)) * t * t + (float)1.0;
        wt[2] = ((A + (float)2.0) * x2 - (A + (float)3.0)) * x2 * x2 + (float)1.0;
        wt[3] = ((A * x3 - (float)5.0 * A) * x3 + (float)8.0 * A) * x3 - (float)4.0 * A;
        unsigned long k = 0UL;
        while (k < 4UL) {
            long j = i - 1L + (long)k;
            ix[k] = j < 0L ? 0UL : (j >= (long)in ? in - 1UL : (unsigned long)j);
            k = k + 1UL;
        }
        return 4UL;
    }
}

// Size (floats) of one axis's tables: forward CSR (start, idx, w) then its
// transpose's.
static unsigned long nr_axis_len_(unsigned long in, unsigned long out, unsigned long nt) {
    return (out + 1UL) + 2UL * out * nt + (in + 1UL) + 2UL * out * nt;
}

// Writes one axis's tables at dst (nt taps per output).
static void nr_axis_tables_(&float dst, unsigned long in, unsigned long out, int mode, int align, unsigned long nt) {
    unsafe {
        float* d = (float*)dst;
        float* fs = d;
        float* fi = fs + out + 1UL;
        float* fw = fi + out * nt;
        float* as = fw + out * nt;
        float* ai = as + in + 1UL;
        float* aw = ai + out * nt;
        unsigned long idx[4];
        float w[4];
        unsigned long* cnt = (unsigned long*)malloc(sizeof(unsigned long) * (in + 1UL));
        unsigned long i = 0UL;
        while (i <= in) { cnt[i] = 0UL; i = i + 1UL; }
        unsigned long o = 0UL;
        while (o < out) {
            unsigned long n = nr_taps_(o, in, out, mode, align, (&unsigned long)&idx[0], (&float)&w[0]);
            fs[o] = (float)(o * nt);
            unsigned long k = 0UL;
            while (k < nt) {
                fi[o * nt + k] = k < n ? (float)idx[k] : (float)0.0;
                fw[o * nt + k] = k < n ? w[k] : (float)0.0;
                if (k < n) cnt[idx[k] + 1UL] = cnt[idx[k] + 1UL] + 1UL;
                k = k + 1UL;
            }
            o = o + 1UL;
        }
        fs[out] = (float)(out * nt);
        // transpose: for each input, the (output, weight) pairs that read it
        i = 0UL;
        while (i < in) { cnt[i + 1UL] = cnt[i + 1UL] + cnt[i]; i = i + 1UL; }
        i = 0UL;
        while (i <= in) { as[i] = (float)cnt[i]; i = i + 1UL; }
        o = 0UL;
        while (o < out) {
            unsigned long n = nr_taps_(o, in, out, mode, align, (&unsigned long)&idx[0], (&float)&w[0]);
            unsigned long k = 0UL;
            while (k < n) {
                unsigned long at = cnt[idx[k]];
                ai[at] = (float)o;
                aw[at] = w[k];
                cnt[idx[k]] = at + 1UL;
                k = k + 1UL;
            }
            o = o + 1UL;
        }
        // the transposed lists are shorter when an output had fewer taps than nt: pad the tail
        unsigned long used = (unsigned long)as[in];
        while (used < out * nt) { ai[used] = (float)0.0; aw[used] = (float)0.0; used = used + 1UL; }
        free((void*)cnt);
    }
}

static unsigned long nr_resize_taps_(int mode) { return mode == RESIZE_NEAREST ? 1UL : (mode == RESIZE_BILINEAR ? 2UL : 4UL); }

int RocmNet::add_resize(unsigned long outH, unsigned long outW, int mode, int alignCorners) {
    struct RocmLayer L;
    if (outH == 0UL || outW == 0UL || mode < RESIZE_NEAREST || mode > RESIZE_BICUBIC ||
        !nr_geom_layer_(self, ROCM_LAYER_RESIZE, ACT_NONE, &L)) { self.ok = 0; return 0; }
    L.outH = outH; L.outW = outW;
    L.mode = mode;
    L.flag = mode == RESIZE_NEAREST ? 0 : alignCorners;
    unsigned long nt = nr_resize_taps_(mode);
    unsigned long ly = nr_axis_len_(L.inH, outH, nt);
    unsigned long lx = nr_axis_len_(L.inW, outW, nt);
    int ok = 0;
    unsafe {
        float* t = (float*)malloc(sizeof(float) * (ly + lx));
        if (t != (float*)0) {
            nr_axis_tables_((&float)t, L.inH, outH, mode, L.flag, nt);
            nr_axis_tables_((&float)&t[ly], L.inW, outW, mode, L.flag, nt);
            ok = nr_alloc_(&L.aux, ly + lx) && rocm_write(L.aux, (const &float)(const float*)t, ly + lx) && rocm_sync();
            free((void*)t);
        }
    }
    if (!ok) { nr_layer_free_(&L); self.ok = 0; return 0; }
    return nr_push_(self, L);
}

int RocmNet::add_dropout(float p, int byChannel, unsigned long seed) {
    struct RocmLayer L;
    if (!nr_geom_layer_(self, ROCM_LAYER_DROPOUT, ACT_NONE, &L)) return 0;
    L.prob = p;
    L.flag = byChannel ? 1 : 0;
    L.seed = seed;
    return nr_push_(self, L);
}

// Rows / columns of a layer's weight as a matrix (shape[0] of its tensor).
static unsigned long nr_w_rows_(const &RocmLayer L) {
    if (L.kind == ROCM_LAYER_LINEAR) return nr_in_size_(L);
    return L.kind == ROCM_LAYER_CONV ? L.outC : L.inC;
}

int RocmNet::spectral_norm(const &Tensor u) {
    unsigned long c = self.layers.length();
    if (!self.ok || c == 0UL) { self.ok = 0; return 0; }
    &RocmLayer L = self.layers.at(c - 1UL);
    if (L.kind != ROCM_LAYER_LINEAR && L.kind != ROCM_LAYER_CONV && L.kind != ROCM_LAYER_CONVT) { self.ok = 0; return 0; }
    unsigned long rows = nr_w_rows_(L);
    unsigned long cols = L.w.len / rows;
    // aux: u [rows] | v [cols] | W v [rows] | sigma, dot, -, counter | partial sums [64]
    if (!nr_tensor_ok_(u, rows) || !nr_alloc_(&L.wsn, L.w.len) || !nr_alloc_(&L.aux, 2UL * rows + cols + 4UL + 64UL) ||
        !rocm_zero(L.aux)) { self.ok = 0; return 0; }
    unsafe { if (!rocm_write(L.aux, (const &float)(const float*)u->data, rows) || !rocm_sync()) { self.ok = 0; return 0; } }
    L.spectral = 1;
    return 1;
}

struct RocmNet rocm_net_from_mlp(const &Mlp m, unsigned long maxBatch) {
    struct RocmNet n = rocm_net_new(maxBatch);
    unsigned long l = 0UL;
    while (n.ok && l < m.nLayers) {
        int act = l + 1UL == m.nLayers ? m.outAct : m.act;
        n.add_linear(*m.params.at(2UL * l), *m.params.at(2UL * l + 1UL), act);
        l = l + 1UL;
    }
    return n;
}

struct RocmNet rocm_net_from_convnet(const &ConvNet c, unsigned long maxBatch) {
    struct RocmNet n = rocm_net_new(maxBatch);
    n.set_input(c.inC, c.inH, c.inW);
    n.training = c.training ? 1 : 0;
    struct Vec<unsigned long> slots = vec_new();   // ConvNet slot -> SAVE layer index
    unsigned long i = 0UL;
    while (n.ok && i < c.layers.length()) {
        &ConvLayer l = c.layers.at(i);
        int k = l.kind;
        if (k == CONV_LAYER_CONV) {
            n.add_conv2d_ex(*c.params.at(l.param), *c.params.at(l.param + 1UL), l.inC, l.inH, l.inW, &l.conv, l.act);
        } else if (k == CONV_LAYER_DECONV) {
            n.add_conv_transpose2d_ex(*c.params.at(l.param), *c.params.at(l.param + 1UL), l.inC, l.inH, l.inW, &l.conv, l.act);
        } else if (k == CONV_LAYER_DENSE) {
            n.add_linear(*c.params.at(l.param), *c.params.at(l.param + 1UL), l.act);
        } else if (k == CONV_LAYER_MAXPOOL) {
            n.add_maxpool(&l.pool, l.act);
        } else if (k == CONV_LAYER_AVGPOOL) {
            n.add_avgpool(&l.pool, l.act);
        } else if (k == CONV_LAYER_GLOBAL_AVGPOOL || k == CONV_LAYER_GLOBAL_MAXPOOL) {
            n.add_global_pool(k == CONV_LAYER_GLOBAL_MAXPOOL, l.act);
        } else if (k == CONV_LAYER_UPSAMPLE) {
            n.add_upsample(l.factor, l.act);
        } else if (k == CONV_LAYER_BATCHNORM) {
            n.add_batchnorm(*c.params.at(l.param), *c.params.at(l.param + 1UL), *c.buffers.at(l.buffer), *c.buffers.at(l.buffer + 1UL), (float)0.1, (float)0.00001, l.act);
        } else if (k == CONV_LAYER_GROUPNORM) {
            n.add_groupnorm(*c.params.at(l.param), *c.params.at(l.param + 1UL), l.factor, (float)0.00001, l.act);
        } else if (k == CONV_LAYER_RESHAPE || k == CONV_LAYER_IDENTITY) {
            if (k == CONV_LAYER_RESHAPE || l.act != ACT_NONE) nr_add_reshape_(&n, l.outC, l.outH, l.outW, l.act);
        } else if (k == CONV_LAYER_SAVE) {
            while (slots.length() <= l.slot) slots.push(0UL);
            *slots.at(l.slot) = n.add_save();
        } else if (k == CONV_LAYER_ADD || k == CONV_LAYER_CONCAT) {
            if (l.slot >= slots.length()) { n.ok = 0; }
            else if (k == CONV_LAYER_CONCAT) { n.add_concat(*slots.at(l.slot)); }
            else if (l.flag) {
                n.add_skip(*slots.at(l.slot), (?&Tensor)*c.params.at(l.param), (?&Tensor)*c.params.at(l.param + 1UL), l.act);
            } else {
                n.add_skip(*slots.at(l.slot), null, null, l.act);
            }
        } else if (k == CONV_LAYER_RESIZE) {
            n.add_resize(l.outH, l.outW, l.mode, l.flag);
        } else if (k == CONV_LAYER_DROPOUT) {
            struct Rng r = *c.rngs.at(l.buffer);
            n.add_dropout(l.prob, l.flag, r.u64());
        } else {
            n.ok = 0;
        }
        if (n.ok && l.spectral && (k == CONV_LAYER_CONV || k == CONV_LAYER_DECONV || k == CONV_LAYER_DENSE))
            n.spectral_norm(*c.buffers.at(l.buffer));
        i = i + 1UL;
    }
    slots.free();
    return n;
}

unsigned long RocmNet::in_dim() const { return self.inC * self.inH * self.inW; }

unsigned long RocmNet::out_dim() const {
    unsigned long c = self.layers.length();
    if (c == 0UL) return 0UL;
    return nr_out_size_(self.layers.at(c - 1UL));
}

// Copies every (w, b) pair (stats 0) or batch norm's (runMean, runVar)
// (stats 1) between the net and host tensors.
static int nr_transfer_(&RocmNet n, &Vec<&Tensor> ts, int stats, int download) {
    if (!n.ok) return 0;
    unsigned long k = 0UL;
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        &RocmLayer L = n.layers.at(i);
        if (stats && L.spectral) {
            // a spectral norm's u
            unsigned long rows = nr_w_rows_(L);
            if (k + 1UL > ts.length()) return 0;
            &Tensor tu = *ts.at(k);
            if (!nr_tensor_ok_(tu, rows)) return 0;
            unsafe {
                if (download) { if (!rocm_read(nr_at_(L.aux, 0UL, rows), (&float)(float*)tu->data, rows)) return 0; }
                else if (!rocm_write(nr_at_(L.aux, 0UL, rows), (const &float)(const float*)tu->data, rows)) return 0;
            }
            k = k + 1UL;
        }
        int has = stats ? L.kind == ROCM_LAYER_BATCHNORM : L.w.dev != 0UL;
        if (has) {
            if (k + 2UL > ts.length()) return 0;
            struct RocmBuf a = stats ? L.runMean : L.w;
            struct RocmBuf b = stats ? L.runVar : L.b;
            &Tensor ta = *ts.at(k);
            &Tensor tb = *ts.at(k + 1UL);
            if (!nr_tensor_ok_(ta, a.len) || !nr_tensor_ok_(tb, b.len)) return 0;
            unsafe {
                if (download) {
                    if (!rocm_read(a, (&float)(float*)ta->data, a.len) || !rocm_read(b, (&float)(float*)tb->data, b.len))
                        return 0;
                } else {
                    if (!rocm_write(a, (const &float)(const float*)ta->data, a.len) ||
                        !rocm_write(b, (const &float)(const float*)tb->data, b.len)) return 0;
                }
            }
            k = k + 2UL;
        }
        i = i + 1UL;
    }
    return k == ts.length() && (download || rocm_sync());
}

int RocmNet::download(&Vec<&Tensor> params) const {
    unsafe { return nr_transfer_((&RocmNet)self, params, 0, 1); }
}

int RocmNet::upload(const &Vec<&Tensor> params) {
    unsafe { return nr_transfer_(self, (&Vec<&Tensor>)params, 0, 0); }
}

int RocmNet::download_stats(&Vec<&Tensor> buffers) const {
    unsafe { return nr_transfer_((&RocmNet)self, buffers, 1, 1); }
}

int RocmNet::upload_stats(const &Vec<&Tensor> buffers) {
    unsafe { return nr_transfer_(self, (&Vec<&Tensor>)buffers, 1, 0); }
}

int RocmNet::download_convnet(&ConvNet c) const {
    return self.download(&c.params) && self.download_stats(&c.buffers);
}

void RocmNet::free() {
    unsigned long i = 0UL;
    while (i < self.layers.length()) { nr_layer_free_(self.layers.at(i)); i = i + 1UL; }
    self.layers.free();
    self.scale.free(); self.xh.free(); self.dxh.free(); self.out32.free(); self.dout32.free();
    self.precision = 0;
    self.ok = 0;
}

// ── Batch norm passes (bn_stats / bn_grads / bn_finish / bn_apply) ──

// The kernels' BnArgs.
struct NrBnArgs_ {
    unsigned long x; unsigned long dy; unsigned long src; unsigned long stats; unsigned long part; unsigned long rm;
    unsigned long rv; unsigned long dgamma; unsigned long dbeta; unsigned long gamma; unsigned long beta;
    unsigned long cb; unsigned long cgb; unsigned long z; unsigned long h;
    unsigned int C; unsigned int B; unsigned int P; unsigned int S;
    float eps; float momentum;
    int act; int training;
};

// A batch norm's arguments; 'folded' is the convolution whose bias it applies (or null).
static struct NrBnArgs_ nr_bn_args_(const &RocmNet n, const &RocmLayer L, struct RocmBuf x, struct RocmBuf dy,
                                    struct RocmBuf z, struct RocmBuf h, unsigned long B, unsigned long S) {
    struct NrBnArgs_ a;
    a.x = x.dev; a.dy = dy.dev; a.src = L.z.dev != 0UL ? L.z.dev : L.h.dev; a.stats = L.stats.dev; a.part = L.part.dev;
    a.rm = L.runMean.dev; a.rv = L.runVar.dev; a.dgamma = L.gw.dev; a.dbeta = L.gb.dev; a.gamma = L.w.dev;
    a.beta = L.b.dev; a.cb = 0UL; a.cgb = 0UL; a.z = z.dev; a.h = h.dev;
    if (L.flag) {
        &RocmLayer conv = n.layers.at(L.slot);
        a.cb = conv.b.dev;
        a.cgb = conv.gb.dev;
    }
    a.C = (unsigned int)L.inC; a.B = (unsigned int)B; a.P = (unsigned int)(L.outH * L.outW); a.S = (unsigned int)S;
    a.eps = L.eps; a.momentum = L.momentum; a.act = L.act; a.training = n.training;
    return a;
}

static unsigned long nr_bn_slices_(const &RocmLayer L, unsigned long B, unsigned long S) {
    unsigned long sl = (B / S) * L.outH * L.outW / 4096UL;
    if (sl > NR_BN_SLICES) sl = NR_BN_SLICES;
    return sl == 0UL ? 1UL : sl;
}

// kernel NR_BN_STATS / NR_BN_GRADS over (C*S, slices); NR_BN_FINISH (phase) over C;
// NR_BN_APPLY (mode = phase) over every value
static int nr_bn_launch_(int kernel, struct NrBnArgs_ a, int phase, unsigned long slices) {
    unsafe {
        struct NrBnArgs_ aa = a;
        int ph = phase;
        void* p[2];
        p[0] = (void*)&aa; p[1] = (void*)&ph;
        if (kernel == NR_BN_STATS || kernel == NR_BN_GRADS)
            return nr_launch_grid_(kernel, (unsigned long)a.C * (unsigned long)a.S, slices, (&void)&p[0]);
        if (kernel == NR_BN_FINISH) return nr_launch_(kernel, (unsigned long)a.C, (&void)&p[0]);
        return nr_launch_(kernel, (unsigned long)a.C * (unsigned long)a.B * (unsigned long)a.P, (&void)&p[0]);
    }
}

// ── Split-K weight gradients ──

// The scratch for split-K partial products, shared by every net (one stream:
// one product at a time); grown when a layer is added, never during capture.
static struct RocmBuf g_nr_splitk;
static int g_nr_splitk_init = 0;

// Slices for c[m,n] = a[m,k] b[n,k]^T: ~480 blocks of 32 x 32 tiles, at least 256 of k each.
static unsigned long nr_splitk_slices_(unsigned long m, unsigned long n, unsigned long k, &unsigned long kc) {
    unsigned long tiles = ((m + 31UL) / 32UL) * ((n + 31UL) / 32UL);
    unsigned long s = (480UL + tiles - 1UL) / tiles;
    unsigned long most = (k + 255UL) / 256UL;
    if (s > most) s = most;
    if (s == 0UL) s = 1UL;
    unsigned long c = ((k + s - 1UL) / s + 31UL) / 32UL * 32UL;
    *kc = c;
    return (k + c - 1UL) / c;
}

static int nr_splitk_reserve_(unsigned long m, unsigned long n, unsigned long k) {
    if (k < NR_SPLITK_MIN_K) return 1;
    unsigned long kc = 0UL;
    unsigned long need = nr_splitk_slices_(m, n, k, &kc) * m * n;
    if (!g_nr_splitk_init) { g_nr_splitk = nr_none_(); g_nr_splitk_init = 1; }
    if (g_nr_splitk.len >= need) return 1;
    rocm_sync();
    g_nr_splitk.free();
    g_nr_splitk = rocm_alloc(need);
    return g_nr_splitk.dev != 0UL;
}

static int nr_splitk_gemm_(struct RocmBuf a, struct RocmBuf b, struct RocmBuf c, unsigned long m, unsigned long n,
                           unsigned long k) {
    unsigned long kc = 0UL;
    unsigned long S = nr_splitk_slices_(m, n, k, &kc);
    if (!g_nr_splitk_init || g_nr_splitk.len < S * m * n || !nr_fits_(a, m * k) || !nr_fits_(b, n * k) ||
        !nr_fits_(c, m * n)) return 0;
    unsafe {
        void* pa = (void*)a.dev; void* pb = (void*)b.dev; void* pp = (void*)g_nr_splitk.dev; void* pc = (void*)c.dev;
        unsigned int mm = (unsigned int)m; unsigned int nn = (unsigned int)n; unsigned int kk = (unsigned int)k;
        unsigned int cc = (unsigned int)kc; unsigned int mn = (unsigned int)(m * n); unsigned int ss = (unsigned int)S;
        void* p[7];
        p[0] = (void*)&pa; p[1] = (void*)&pb; p[2] = (void*)&pp; p[3] = (void*)&mm; p[4] = (void*)&nn; p[5] = (void*)&kk;
        p[6] = (void*)&cc;
        if (hipModuleLaunchKernel(g_nr_fn[NR_SPLITK], (unsigned int)((n + 31UL) / 32UL), (unsigned int)((m + 31UL) / 32UL),
                                  ss, NR_BLOCK, 1U, 1U, 0U, g_nr_stream, &p[0], (void**)0) != 0) return 0;
        void* q[4];
        q[0] = (void*)&pp; q[1] = (void*)&pc; q[2] = (void*)&mn; q[3] = (void*)&ss;
        return nr_launch_(NR_SPLITK_RED, m * n, (&void)&q[0]);
    }
}

// ── MIOpen convolution engine (optional) ─────────────────────────────────────
// Weak references: a program linked with -lMIOpen gets MIOpen's convolution
// kernels (implicit GEMM, Winograd, matrix cores) for CONV / CONVT layers;
// without it these resolve to null and every convolution runs on the
// im2col engine. MIOpen sees the NCHW activations as they are and queues
// its kernels on gpu_rocm's stream, so they order with the rest and record
// into graphs.
extern int miopenCreate(void** handle) __attribute__((weak));
extern int miopenSetStream(void* handle, void* stream) __attribute__((weak));
extern int miopenCreateTensorDescriptor(void** desc) __attribute__((weak));
extern int miopenSet4dTensorDescriptor(void* desc, int dataType, int n, int c, int h, int w) __attribute__((weak));
extern int miopenCreateConvolutionDescriptor(void** desc) __attribute__((weak));
extern int miopenInitConvolutionDescriptor(void* desc, int mode, int padH, int padW, int strideH, int strideW,
                                           int dilationH, int dilationW) __attribute__((weak));
extern int miopenSetConvolutionGroupCount(void* desc, int groups) __attribute__((weak));
extern int miopenSetTransposeConvOutputPadding(void* desc, int adjH, int adjW) __attribute__((weak));
extern int miopenConvolutionForwardGetWorkSpaceSize(void* handle, void* wDesc, void* xDesc, void* convDesc,
                                                    void* yDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenConvolutionBackwardDataGetWorkSpaceSize(void* handle, void* dyDesc, void* wDesc, void* convDesc,
                                                         void* dxDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenConvolutionBackwardWeightsGetWorkSpaceSize(void* handle, void* dyDesc, void* xDesc, void* convDesc,
                                                            void* dwDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenFindConvolutionForwardAlgorithm(void* handle, void* xDesc, const void* x, void* wDesc, const void* w,
                                                 void* convDesc, void* yDesc, void* y, int request, int* returned,
                                                 void* perf, void* ws, unsigned long wsBytes, int exhaustive)
    __attribute__((weak));
extern int miopenFindConvolutionBackwardDataAlgorithm(void* handle, void* dyDesc, const void* dy, void* wDesc,
                                                      const void* w, void* convDesc, void* dxDesc, void* dx,
                                                      int request, int* returned, void* perf, void* ws,
                                                      unsigned long wsBytes, int exhaustive) __attribute__((weak));
extern int miopenFindConvolutionBackwardWeightsAlgorithm(void* handle, void* dyDesc, const void* dy, void* xDesc,
                                                         const void* x, void* convDesc, void* dwDesc, void* dw,
                                                         int request, int* returned, void* perf, void* ws,
                                                         unsigned long wsBytes, int exhaustive) __attribute__((weak));
extern int miopenConvolutionForward(void* handle, const void* alpha, void* xDesc, const void* x, void* wDesc,
                                    const void* w, void* convDesc, int algo, const void* beta, void* yDesc, void* y,
                                    void* ws, unsigned long wsBytes) __attribute__((weak));
extern int miopenConvolutionBackwardData(void* handle, const void* alpha, void* dyDesc, const void* dy, void* wDesc,
                                         const void* w, void* convDesc, int algo, const void* beta, void* dxDesc,
                                         void* dx, void* ws, unsigned long wsBytes) __attribute__((weak));
extern int miopenConvolutionBackwardWeights(void* handle, const void* alpha, void* dyDesc, const void* dy,
                                            void* xDesc, const void* x, void* convDesc, int algo, const void* beta,
                                            void* dwDesc, void* dw, void* ws, unsigned long wsBytes)
    __attribute__((weak));

// miopenConvAlgoPerf_t
struct NrMioPerf_ {
    int           algo;
    float         time;
    unsigned long memory;
};

#define NR_MIO_PLANS 512
#define NR_MIO_KEY   20

static int   g_mio_engine = ROCM_CONV_AUTO;
static int   g_mio_state = 0;           // 0 untried, 1 ready, -1 MIOpen not linked or failed
static void* g_mio_handle = (void*)0;
static struct RocmBuf g_mio_ws;         // workspace (floats), grown outside capture
static struct RocmBuf g_mio_tmp;        // a destination for the backward Finds
static unsigned long g_mio_key[NR_MIO_PLANS * NR_MIO_KEY];
static void* g_mio_desc[NR_MIO_PLANS * 4];   // x, y, w, conv
static int   g_mio_algo[NR_MIO_PLANS * 3];   // forward, backward data, backward weights
static int   g_mio_good[NR_MIO_PLANS];       // 0: MIOpen failed for this shape (the im2col engine runs it)
static int   g_mio_count = 0;
static int   g_mio_fresh = -1;               // the plan the last nr_mio_plan_ created (-1: none)

static int nr_mio_on_() {
    if (g_mio_engine != ROCM_CONV_AUTO) return 0;
    if (g_mio_state == 0) {
        g_mio_state = -1;
        g_mio_ws = nr_none_();
        g_mio_tmp = nr_none_();
        unsafe {
            if (nr_ensure_() && (void*)miopenCreate != (void*)0 && miopenCreate(&g_mio_handle) == 0 &&
                miopenSetStream(g_mio_handle, g_nr_stream) == 0) g_mio_state = 1;
        }
    }
    return g_mio_state == 1;
}

int rocm_conv_engine(int engine) {
    g_mio_engine = engine == ROCM_CONV_IM2COL ? ROCM_CONV_IM2COL : ROCM_CONV_AUTO;
    return nr_mio_on_();
}

static void nr_mio_fill_key_(const &RocmLayer L, unsigned long B, &unsigned long key) {
    unsafe {
        unsigned long* k = (unsigned long*)key;
        k[0] = (unsigned long)L.kind; k[1] = B; k[2] = L.inC; k[3] = L.inH; k[4] = L.inW; k[5] = L.outC;
        k[6] = L.outH; k[7] = L.outW; k[8] = L.conv.kH; k[9] = L.conv.kW; k[10] = L.conv.strideH;
        k[11] = L.conv.strideW; k[12] = L.conv.padH; k[13] = L.conv.padW; k[14] = L.conv.dilationH;
        k[15] = L.conv.dilationW; k[16] = L.conv.groups; k[17] = L.conv.outPadH; k[18] = L.conv.outPadW;
        k[19] = (unsigned long)g_nr_prec;
    }
}

static int nr_mio_find_plan_(const &RocmLayer L, unsigned long B) {
    unsigned long key[NR_MIO_KEY];
    nr_mio_fill_key_(L, B, &key[0]);
    int i = 0;
    while (i < g_mio_count) {
        int same = 1;
        int j = 0;
        while (same && j < NR_MIO_KEY) { if (g_mio_key[i * NR_MIO_KEY + j] != key[j]) same = 0; j = j + 1; }
        if (same) return i;
        i = i + 1;
    }
    return -1;
}

static int nr_mio_grow_(&RocmBuf b, unsigned long bytes) {
    unsigned long n = (bytes + 3UL) / 4UL;
    if (n == 0UL) n = 1UL;
    if (b.len >= n) return 1;
    rocm_sync();
    b.free();
    *b = rocm_alloc(n);
    return b.dev != 0UL;
}

// The plan for L at batch B: found earlier, or (outside capture) created now
// with all three Finds on these buffers. -1 when MIOpen can't run it.
static int nr_mio_plan_(const &RocmLayer L, unsigned long B, struct RocmBuf x, struct RocmBuf y, struct RocmBuf w,
                        struct RocmBuf dy) {
    int found = nr_mio_find_plan_(L, B);
    if (found >= 0) return g_mio_good[found] ? found : -1;
    if (nr_capturing_() || g_mio_count >= NR_MIO_PLANS) return -1;
    int i = g_mio_count;
    g_mio_count = g_mio_count + 1;
    unsafe { nr_mio_fill_key_(L, B, (&unsigned long)&g_mio_key[i * NR_MIO_KEY]); }
    g_mio_good[i] = 0;
    int transposed = L.kind == ROCM_LAYER_CONVT;
    unsigned long G = L.conv.groups;
    int ok = 1;
    unsafe {
        void* xd = (void*)0; void* yd = (void*)0; void* wd = (void*)0; void* cd = (void*)0;
        ok = miopenCreateTensorDescriptor(&xd) == 0 && miopenCreateTensorDescriptor(&yd) == 0 &&
             miopenCreateTensorDescriptor(&wd) == 0 && miopenCreateConvolutionDescriptor(&cd) == 0;
        int dt = g_nr_prec == 0 ? 1 : (g_nr_prec == 1 ? 0 : 5);   // miopenFloat, miopenHalf, miopenBFloat16
        ok = ok && miopenSet4dTensorDescriptor(xd, dt, (int)B, (int)L.inC, (int)L.inH, (int)L.inW) == 0 &&
             miopenSet4dTensorDescriptor(yd, dt, (int)B, (int)L.outC, (int)L.outH, (int)L.outW) == 0;
        if (transposed) ok = ok && miopenSet4dTensorDescriptor(wd, dt, (int)L.inC, (int)(L.outC / G), (int)L.conv.kH, (int)L.conv.kW) == 0;
        else ok = ok && miopenSet4dTensorDescriptor(wd, dt, (int)L.outC, (int)(L.inC / G), (int)L.conv.kH, (int)L.conv.kW) == 0;
        ok = ok && miopenInitConvolutionDescriptor(cd, transposed ? 1 : 0, (int)L.conv.padH, (int)L.conv.padW,
                                                   (int)L.conv.strideH, (int)L.conv.strideW, (int)L.conv.dilationH,
                                                   (int)L.conv.dilationW) == 0;
        if (ok && G > 1UL) ok = miopenSetConvolutionGroupCount(cd, (int)G) == 0;
        if (ok && transposed && (L.conv.outPadH > 0UL || L.conv.outPadW > 0UL))
            ok = miopenSetTransposeConvOutputPadding(cd, (int)L.conv.outPadH, (int)L.conv.outPadW) == 0;
        g_mio_desc[i * 4] = xd; g_mio_desc[i * 4 + 1] = yd; g_mio_desc[i * 4 + 2] = wd; g_mio_desc[i * 4 + 3] = cd;
        unsigned long wf = 0UL; unsigned long wb = 0UL; unsigned long ww = 0UL;
        ok = ok && miopenConvolutionForwardGetWorkSpaceSize(g_mio_handle, wd, xd, cd, yd, &wf) == 0 &&
             miopenConvolutionBackwardDataGetWorkSpaceSize(g_mio_handle, yd, wd, cd, xd, &wb) == 0 &&
             miopenConvolutionBackwardWeightsGetWorkSpaceSize(g_mio_handle, yd, xd, cd, wd, &ww) == 0;
        unsigned long most = wf > wb ? wf : wb;
        if (ww > most) most = ww;
        unsigned long tmpN = B * nr_in_size_(L);
        if (w.len > tmpN) tmpN = w.len;
        ok = ok && nr_mio_grow_(&g_mio_ws, most) && nr_mio_grow_(&g_mio_tmp, tmpN * 4UL);
        struct NrMioPerf_ pf;
        int got = 0;
        void* ws = (void*)g_mio_ws.dev;
        unsigned long wsb = g_mio_ws.len * 4UL;
        ok = ok && miopenFindConvolutionForwardAlgorithm(g_mio_handle, xd, (const void*)x.dev, wd, (const void*)w.dev,
                                                         cd, yd, (void*)y.dev, 1, &got, (void*)&pf, ws, wsb, 0) == 0 &&
             got > 0;
        g_mio_algo[i * 3] = pf.algo;
        got = 0;
        ok = ok && miopenFindConvolutionBackwardDataAlgorithm(g_mio_handle, yd, (const void*)dy.dev, wd, (const void*)w.dev,
                                                              cd, xd, (void*)g_mio_tmp.dev, 1, &got, (void*)&pf, ws,
                                                              wsb, 0) == 0 && got > 0;
        g_mio_algo[i * 3 + 1] = pf.algo;
        got = 0;
        ok = ok && miopenFindConvolutionBackwardWeightsAlgorithm(g_mio_handle, yd, (const void*)dy.dev, xd,
                                                                 (const void*)x.dev, cd, wd, (void*)g_mio_tmp.dev, 1,
                                                                 &got, (void*)&pf, ws, wsb, 0) == 0 && got > 0;
        g_mio_algo[i * 3 + 2] = pf.algo;
    }
    g_mio_good[i] = ok;
    g_mio_fresh = ok ? i : -1;
    return ok ? i : -1;
}

// op 0: y = conv(x); 1: dx from dy; 2: dw from dy and x (overwritten: beta 0)
static int nr_mio_run_(int plan, int op, struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) {
    float one = (float)1.0;
    float zero = (float)0.0;
    unsafe {
        void* xd = g_mio_desc[plan * 4]; void* yd = g_mio_desc[plan * 4 + 1];
        void* wd = g_mio_desc[plan * 4 + 2]; void* cd = g_mio_desc[plan * 4 + 3];
        void* ws = (void*)g_mio_ws.dev;
        unsigned long wsb = g_mio_ws.len * 4UL;
        if (op == 0)   // a = x, b = w
            return miopenConvolutionForward(g_mio_handle, (const void*)&one, xd, (const void*)a.dev, wd, (const void*)b.dev,
                                            cd, g_mio_algo[plan * 3], (const void*)&zero, yd, (void*)out.dev, ws, wsb) == 0;
        if (op == 1)   // a = dy, b = w
            return miopenConvolutionBackwardData(g_mio_handle, (const void*)&one, yd, (const void*)a.dev, wd,
                                                 (const void*)b.dev, cd, g_mio_algo[plan * 3 + 1], (const void*)&zero,
                                                 xd, (void*)out.dev, ws, wsb) == 0;
        // a = dy, b = x
        return miopenConvolutionBackwardWeights(g_mio_handle, (const void*)&one, yd, (const void*)a.dev, xd,
                                                (const void*)b.dev, cd, g_mio_algo[plan * 3 + 2], (const void*)&zero,
                                                wd, (void*)out.dev, ws, wsb) == 0;
    }
}

// The weight gradient (op 2) into L.gw: through a 16-bit scratch for a 16-bit net.
static int nr_mio_dw_(int plan, &RocmLayer L, struct RocmBuf dy, struct RocmBuf x) {
    if (g_nr_prec == 0) return nr_mio_run_(plan, 2, dy, x, L.gw);
    return nr_mio_run_(plan, 2, dy, x, g_mio_tmp) && nr_cvt_(0, g_mio_tmp, L.gw, L.gw.len, nr_none_(), 0);
}

// The im2col engine's scratch for a convolution layer, allocated the first
// time it runs on that engine (never during capture).
static int nr_conv_scratch_(const &RocmNet n, &RocmLayer L) {
    if (L.col.dev != 0UL) return 1;
    if (nr_capturing_()) return 0;
    unsigned long B = n.maxBatch;
    unsigned long kk = L.conv.kH * L.conv.kW;
    unsigned long G = L.conv.groups;
    if (L.kind == ROCM_LAYER_CONV) {
        return nr_alloc_(&L.col, L.inC * kk * B * L.outH * L.outW) && nr_alloc_(&L.cm, B * nr_out_size_(L)) &&
               nr_splitk_reserve_(L.outC / G, (L.inC / G) * kk, B * L.outH * L.outW);
    }
    return nr_alloc_(&L.col, L.outC * kk * B * L.inH * L.inW) && nr_alloc_(&L.cm, B * nr_in_size_(L)) &&
           nr_splitk_reserve_(L.inC / G, (L.outC / G) * kk, B * L.inH * L.inW);
}

// ── Skip / resize / dropout / spectral-norm passes ──

static struct RocmBuf nr_wt_(const &RocmLayer L) { return L.spectral ? L.wsn : L.w; }
// The weight a pass computes with: its 16-bit copy in a 16-bit net.
static struct RocmBuf nr_wk_(const &RocmLayer L) { return g_nr_prec != 0 ? L.wh : nr_wt_(L); }
static int nr_has_wh_(const &RocmLayer L) {
    int k = L.kind;
    return L.w.dev != 0UL && (k == ROCM_LAYER_LINEAR || k == ROCM_LAYER_CONV || k == ROCM_LAYER_CONVT ||
                              (k == ROCM_LAYER_ADD && L.flag));
}

static int nr_ew2_(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out, unsigned long n, int mul) {
    if (!nr_fits_(a, n) || !nr_fits_(b, n) || !nr_fits_(out, n)) return 0;
    unsafe {
        void* pa = (void*)a.dev; void* pb = (void*)b.dev; void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n; int m = mul;
        void* p[5];
        p[0] = (void*)&pa; p[1] = (void*)&pb; p[2] = (void*)&po; p[3] = (void*)&nn; p[4] = (void*)&m;
        return nr_launch_(NR_EW2, n, (&void)&p[0]);
    }
}

// z = a + b (+ bias[channel]), h = act(z); b and bias may be dev 0.
static int nr_add_act_(struct RocmBuf a, struct RocmBuf b, struct RocmBuf bias, struct RocmBuf z, struct RocmBuf h,
                       unsigned long n, unsigned long plane, unsigned long C, int act) {
    if (!nr_fits_(a, n) || !nr_fits_(z, n) || !nr_fits_(h, n)) return 0;
    unsafe {
        void* pa = (void*)a.dev; void* pb = (void*)b.dev; void* pc = (void*)bias.dev; void* pz = (void*)z.dev;
        void* ph = (void*)h.dev;
        unsigned int nn = (unsigned int)n; unsigned int pl = (unsigned int)plane; unsigned int cc = (unsigned int)C;
        int ac = act;
        void* p[9];
        p[0] = (void*)&pa; p[1] = (void*)&pb; p[2] = (void*)&pc; p[3] = (void*)&pz; p[4] = (void*)&ph;
        p[5] = (void*)&nn; p[6] = (void*)&pl; p[7] = (void*)&cc; p[8] = (void*)&ac;
        return nr_launch_(NR_ADD_ACT, n, (&void)&p[0]);
    }
}

// x [outer][inLen][inner] -> out [outer][outLen][inner] through CSR tables at 'tab'.
static int nr_taps_apply_(struct RocmBuf x, struct RocmBuf out, struct RocmBuf start, struct RocmBuf tidx,
                          struct RocmBuf tw, unsigned long outer, unsigned long inLen, unsigned long outLen,
                          unsigned long inner) {
    unsigned long n = outer * outLen * inner;
    if (!nr_fits_(x, outer * inLen * inner) || !nr_fits_(out, n) || start.dev == 0UL) return 0;
    unsafe {
        void* px = (void*)x.dev; void* po = (void*)out.dev; void* ps = (void*)start.dev; void* pi = (void*)tidx.dev;
        void* pw = (void*)tw.dev;
        unsigned int a0 = (unsigned int)outer; unsigned int a1 = (unsigned int)inLen; unsigned int a2 = (unsigned int)outLen;
        unsigned int a3 = (unsigned int)inner;
        void* p[9];
        p[0] = (void*)&px; p[1] = (void*)&po; p[2] = (void*)&ps; p[3] = (void*)&pi; p[4] = (void*)&pw;
        p[5] = (void*)&a0; p[6] = (void*)&a1; p[7] = (void*)&a2; p[8] = (void*)&a3;
        return nr_launch_(NR_TAPS, n, (&void)&p[0]);
    }
}

// One axis's tables in a resize layer's aux: forward (adj 0) or transposed (adj 1).
static void nr_axis_views_(struct RocmBuf aux, unsigned long off, unsigned long in, unsigned long out, unsigned long nt,
                           int adj, &RocmBuf start, &RocmBuf tidx, &RocmBuf tw) {
    unsigned long fl = out * nt;
    if (!adj) {
        *start = nr_at_(aux, off, out + 1UL);
        *tidx = nr_at_(aux, off + out + 1UL, fl);
        *tw = nr_at_(aux, off + out + 1UL + fl, fl);
    } else {
        unsigned long base = off + out + 1UL + 2UL * fl;
        *start = nr_at_(aux, base, in + 1UL);
        *tidx = nr_at_(aux, base + in + 1UL, fl);
        *tw = nr_at_(aux, base + in + 1UL + fl, fl);
    }
}

// Resize forward (adj 0: x [planes][H][W] -> out [planes][oH][oW]) or its
// adjoint (adj 1: dout -> dx), separably through L.col.
static int nr_resize_(&RocmLayer L, struct RocmBuf src, struct RocmBuf dst, unsigned long B, int adj) {
    unsigned long nt = nr_resize_taps_(L.mode);
    unsigned long ly = nr_axis_len_(L.inH, L.outH, nt);
    unsigned long planes = L.inC * B;
    struct RocmBuf ys = nr_none_(); struct RocmBuf yi = nr_none_(); struct RocmBuf yw = nr_none_();
    struct RocmBuf xs = nr_none_(); struct RocmBuf xi = nr_none_(); struct RocmBuf xw = nr_none_();
    nr_axis_views_(L.aux, 0UL, L.inH, L.outH, nt, adj, &ys, &yi, &yw);
    nr_axis_views_(L.aux, ly, L.inW, L.outW, nt, adj, &xs, &xi, &xw);
    if (!adj) {
        // along x: [planes*H][W] -> [planes*H][oW]; along y: [planes][H][oW] -> [planes][oH][oW]
        return nr_taps_apply_(src, L.col, xs, xi, xw, planes * L.inH, L.inW, L.outW, 1UL) &&
               nr_taps_apply_(L.col, dst, ys, yi, yw, planes, L.inH, L.outH, L.outW);
    }
    return nr_taps_apply_(src, L.col, xs, xi, xw, planes * L.outH, L.outW, L.inW, 1UL) &&
           nr_taps_apply_(L.col, dst, ys, yi, yw, planes, L.outH, L.inH, L.inW);
}

// dst = src (n values), or dst += src when accumulate.
static int nr_put_(struct RocmBuf src, struct RocmBuf dst, unsigned long n, int accumulate) {
    if (accumulate) return nr_ew2_(dst, src, dst, n, 0);
    return nr_copy_(src, dst, n);
}

// An ADD's projection shortcut as a 1x1 convolution layer (for its MIOpen plan).
static struct RocmLayer nr_proj_layer_(const &RocmLayer L) {
    struct RocmLayer p = nr_layer_(ROCM_LAYER_CONV, ACT_NONE, L.skipC, L.skipH, L.skipW);
    p.outC = L.outC; p.outH = L.outH; p.outW = L.outW;
    p.conv = L.conv;
    return p;
}

static int nr_skip_fwd_(&RocmNet n, &RocmLayer L, struct RocmBuf x, struct RocmBuf zt, unsigned long B) {
    int k = L.kind;
    unsigned long N = L.outC * B * L.outH * L.outW;
    if (k == ROCM_LAYER_SAVE) return L.alias != 0 || nr_copy_(x, L.h, N);
    if (k == ROCM_LAYER_RESIZE) {
        if (!nr_resize_(L, x, zt, B, 0)) return 0;
        return L.act == ACT_NONE || nr_bias_act_(zt, L.h, nr_none_(), N, 1UL, 1UL, L.act);
    }
    if (k == ROCM_LAYER_DROPOUT) {
        if (!n.training || L.prob <= (float)0.0) return nr_copy_(x, L.h, N);
        unsafe {
            void* px = (void*)x.dev; void* po = (void*)L.h.dev; void* pm = (void*)L.stats.dev; void* pc = (void*)L.aux.dev;
            unsigned int nn = (unsigned int)N;
            unsigned int pl = L.flag ? (unsigned int)(L.outH * L.outW) : 1U;
            float pp = L.prob; unsigned long sd = L.seed;
            void* p[8];
            p[0] = (void*)&px; p[1] = (void*)&po; p[2] = (void*)&pm; p[3] = (void*)&nn; p[4] = (void*)&pl;
            p[5] = (void*)&pp; p[6] = (void*)&sd; p[7] = (void*)&pc;
            unsigned long blocks = (N + (unsigned long)NR_BLOCK - 1UL) / (unsigned long)NR_BLOCK;
            if (blocks > 256UL) blocks = 256UL;   // each block takes a turn on the draw counter
            return nr_launch_grid_(NR_DROPOUT, blocks, 1UL, (&void)&p[0]);
        }
    }
    &RocmLayer sv = n.layers.at(L.slot);
    if (k == ROCM_LAYER_CONCAT) {
        // per image: x's channels, then the saved ones
        unsigned long nx = L.inC * L.inH * L.inW;
        unsigned long ns = L.skipC * L.skipH * L.skipW;
        if (!nr_fits_(x, B * nx) || !nr_fits_(sv.h, B * ns) || !nr_fits_(L.h, B * (nx + ns))) return 0;
        unsafe {
            void* pa = (void*)x.dev; void* ps = (void*)sv.h.dev; void* po = (void*)L.h.dev;
            unsigned int a0 = (unsigned int)nx; unsigned int a1 = (unsigned int)ns; unsigned int bb = (unsigned int)B;
            void* p[6];
            p[0] = (void*)&pa; p[1] = (void*)&a0; p[2] = (void*)&ps; p[3] = (void*)&a1; p[4] = (void*)&po; p[5] = (void*)&bb;
            return nr_launch_(NR_CONCAT, B * (nx + ns), (&void)&p[0]);
        }
    }
    // ADD
    unsigned long P = L.outH * L.outW;
    if (!L.flag) return nr_add_act_(x, sv.h, nr_none_(), zt, L.h, N, 1UL, 1UL, L.act);
    if (nr_mio_on_()) {
        struct RocmLayer pl = nr_proj_layer_(L);
        int plan = nr_mio_plan_(&pl, B, sv.h, zt, nr_wk_(L), L.dh);
        if (plan >= 0)
            return nr_mio_run_(plan, 0, sv.h, nr_wk_(L), zt) && nr_add_act_(zt, x, L.b, zt, L.h, N, P, L.outC, L.act);
    }
    if (g_nr_prec != 0) return 0;   // 16-bit convolutions run on MIOpen only
    struct NrGeom_ g = nr_conv_geom_(&L.conv);
    if (!nr_im2col_(sv.h, L.col, L.skipC, L.skipH, L.skipW, L.outH, L.outW, g, B)) return 0;
    if (!rocm_gemm(L.w, 0, L.col, 0, L.cm, L.outC, B * P, L.skipC)) return 0;
    return nr_cm_out_(L.cm, x, L.b, zt, L.h, L.outC, B, P, L.act);
}

// L.dh holds the pre-activation gradient. Sends d(saved) to the SAVE layer
// (its aux: written by the first consumer met going backwards, added to by the rest).
static int nr_skip_bwd_(&RocmNet n, &RocmLayer L, struct RocmBuf x, unsigned long B, int needDx, struct RocmBuf dx) {
    int k = L.kind;
    unsigned long N = L.outC * B * L.outH * L.outW;
    if (k == ROCM_LAYER_SAVE) {
        if (L.alias != 0) return !L.flag || nr_ew2_(L.dh, L.aux, L.dh, N, 0);   // the gradient is already in place
        if (!needDx) return 1;
        if (L.flag) return nr_ew2_(L.dh, L.aux, dx, N, 0);
        return nr_copy_(L.dh, dx, N);
    }
    if (k == ROCM_LAYER_RESIZE) return !needDx || nr_resize_(L, L.dh, dx, B, 1);
    if (k == ROCM_LAYER_DROPOUT) {
        if (!needDx) return 1;
        if (!n.training || L.prob <= (float)0.0) return nr_copy_(L.dh, dx, N);
        return nr_ew2_(L.dh, L.stats, dx, N, 1);
    }
    &RocmLayer sv = n.layers.at(L.slot);
    int acc = sv.flag;
    sv.flag = 1;
    unsigned long skipN = B * L.skipC * L.skipH * L.skipW;
    if (k == ROCM_LAYER_CONCAT) {
        // per image: the first inC*P values of each row back to x, the rest to the saved activation
        unsigned long nx = L.inC * L.inH * L.inW;
        unsigned long ns = L.skipC * L.skipH * L.skipW;
        if (needDx && !rocm_slice_cols(L.dh, dx, B, nx + ns, nx, 0UL)) return 0;
        if (!acc) return rocm_slice_cols(L.dh, sv.aux, B, nx + ns, ns, nx);
        return rocm_slice_cols(L.dh, L.aux, B, nx + ns, ns, nx) && nr_ew2_(sv.aux, L.aux, sv.aux, B * ns, 0);
    }
    // ADD (the previous layer usually reads L.dh itself)
    if (needDx && dx.dev != L.dh.dev && !nr_copy_(L.dh, dx, N)) return 0;
    if (!L.flag) return nr_put_(L.dh, sv.aux, N, acc);
    unsigned long P = L.outH * L.outW;
    unsigned long Nc = B * P;
    struct NrGeom_ g = nr_conv_geom_(&L.conv);
    // bias gradient (the activation's gradient is already applied), weight gradient, d(saved)
    if (!nr_act_grad_(L.dh, L.h, L.gb, L.part, Nc, P, L.outC, ACT_NONE)) return 0;
    if (nr_mio_on_()) {
        struct RocmLayer pl = nr_proj_layer_(L);
        int plan = nr_mio_find_plan_(&pl, B);
        if (plan >= 0 && g_mio_good[plan]) {
            struct RocmBuf dst = acc ? L.aux : sv.aux;
            return nr_mio_dw_(plan, L, L.dh, sv.h) && nr_mio_run_(plan, 1, L.dh, nr_wk_(L), dst) &&
                   (!acc || nr_ew2_(sv.aux, L.aux, sv.aux, skipN, 0));
        }
    }
    if (g_nr_prec != 0) return 0;
    if (!nr_swap01_(L.dh, L.cm, B, L.outC, P)) return 0;
    if (!nr_gemm_(L.cm, 0, L.col, 1, L.gw, L.outC, L.skipC, Nc)) return 0;
    if (!rocm_gemm(L.w, 1, L.cm, 0, L.col, L.skipC, Nc, L.outC)) return 0;
    struct RocmBuf dst = acc ? L.aux : sv.aux;
    if (!nr_col2im_(L.col, nr_none_(), dst, nr_none_(), ACT_NONE, L.skipC, L.skipH, L.skipW, L.outH, L.outW, g, B))
        return 0;
    return !acc || nr_ew2_(sv.aux, L.aux, sv.aux, skipN, 0);
}

static int nr_sn_gemv_(struct RocmBuf w, int transposed, struct RocmBuf in, struct RocmBuf out, unsigned long rows,
                       unsigned long cols) {
    // W^T u: [1, rows] x [rows, cols]; W v: [rows, cols] x [cols, 1]
    if (transposed) return rocm_gemm(in, 0, w, 0, out, 1UL, cols, rows);
    return rocm_gemm(w, 0, in, 0, out, rows, 1UL, cols);
}

static int nr_sn_vec_(struct RocmBuf v, struct RocmBuf u, struct RocmBuf st, unsigned long n, int dot) {
    unsafe {
        void* pv = (void*)v.dev; void* pu = (void*)u.dev; void* ps = (void*)st.dev;
        unsigned int nn = (unsigned int)n; int d = dot;
        void* p[5];
        p[0] = (void*)&pv; p[1] = (void*)&pu; p[2] = (void*)&ps; p[3] = (void*)&nn; p[4] = (void*)&d;
        return nr_launch_grid_(NR_SN_VEC, 1UL, 1UL, (&void)&p[0]);
    }
}

// Spectral norm: (training: one power iteration on u), then v, sigma and W / sigma (tensor_spectral_norm).
static int nr_sn_forward_(&RocmNet n, &RocmLayer L) {
    unsigned long rows = nr_w_rows_(L);
    unsigned long cols = L.w.len / rows;
    struct RocmBuf u = nr_at_(L.aux, 0UL, rows);
    struct RocmBuf v = nr_at_(L.aux, rows, cols);
    struct RocmBuf t = nr_at_(L.aux, rows + cols, rows);
    struct RocmBuf st = nr_at_(L.aux, 2UL * rows + cols, 4UL);
    if (n.training) {
        if (!nr_sn_gemv_(L.w, 1, u, v, rows, cols) || !nr_sn_vec_(v, v, st, cols, 0)) return 0;
        if (!nr_sn_gemv_(L.w, 0, v, u, rows, cols) || !nr_sn_vec_(u, u, st, rows, 0)) return 0;
    }
    if (!nr_sn_gemv_(L.w, 1, u, v, rows, cols) || !nr_sn_vec_(v, v, st, cols, 0)) return 0;
    if (!nr_sn_gemv_(L.w, 0, v, t, rows, cols) || !nr_sn_vec_(t, u, st, rows, 1)) return 0;
    unsafe {
        void* pw = (void*)L.w.dev; void* po = (void*)L.wsn.dev; void* ps = (void*)st.dev;
        unsigned int nn = (unsigned int)L.w.len;
        void* p[4];
        p[0] = (void*)&pw; p[1] = (void*)&po; p[2] = (void*)&nn; p[3] = (void*)&ps;
        return nr_launch_(NR_SN_SCALE, L.w.len, (&void)&p[0]);
    }
}

// gw (the gradient for W / sigma) -> the gradient for W.
static int nr_sn_backward_(&RocmLayer L) {
    unsigned long rows = nr_w_rows_(L);
    unsigned long cols = L.w.len / rows;
    struct RocmBuf u = nr_at_(L.aux, 0UL, rows);
    struct RocmBuf v = nr_at_(L.aux, rows, cols);
    struct RocmBuf st = nr_at_(L.aux, 2UL * rows + cols, 4UL);
    struct RocmBuf part = nr_at_(L.aux, 2UL * rows + cols + 4UL, 64UL);
    unsigned long n = L.w.len;
    unsigned long blocks = (n + (unsigned long)NR_BLOCK - 1UL) / (unsigned long)NR_BLOCK;
    if (blocks > 64UL) blocks = 64UL;
    unsafe {
        void* pg = (void*)L.gw.dev; void* pw = (void*)L.w.dev; void* pp = (void*)part.dev; void* ps = (void*)st.dev;
        unsigned int nn = (unsigned int)n;
        void* p[5];
        p[0] = (void*)&pg; p[1] = (void*)&pw; p[2] = (void*)&pp; p[3] = (void*)&ps; p[4] = (void*)&nn;
        if (!nr_launch_grid_(NR_SN_DOT, blocks, 1UL, (&void)&p[0])) return 0;
        void* pu = (void*)u.dev; void* pv = (void*)v.dev;
        unsigned int r = (unsigned int)rows; unsigned int c = (unsigned int)cols;
        void* q[6];
        q[0] = (void*)&pg; q[1] = (void*)&pu; q[2] = (void*)&pv; q[3] = (void*)&ps; q[4] = (void*)&r; q[5] = (void*)&c;
        return nr_launch_(NR_SN_FIX, n, (&void)&q[0]);
    }
}

// ── Forward / backward ───────────────────────────────────────────────────────

static int nr_layer_fwd_(&RocmNet n, &RocmLayer L, struct RocmBuf x, unsigned long B) {
    struct RocmBuf zt = L.z.dev != 0UL ? L.z : L.h;
    int k = L.kind;
    if (L.spectral && !nr_sn_forward_(n, L)) return 0;
    if (L.spectral && g_nr_prec != 0 && !nr_cvt_(1, L.wsn, L.wh, L.w.len, nr_none_(), 0)) return 0;
    struct RocmBuf wt = nr_wk_(L);
    if (k == ROCM_LAYER_LINEAR) {
        unsigned long in = nr_in_size_(L);
        if (g_nr_prec != 0) {
            if (!rocm_half_gemm(nr_hv_(x, B * in), 0, nr_hv_(wt, in * L.outC), 0, nr_hv_(zt, B * L.outC), B, L.outC, in))
                return 0;
        } else if (!rocm_gemm(x, 0, wt, 0, zt, B, L.outC, in)) {
            return 0;
        }
        return nr_bias_act_(zt, L.h, L.b, B * L.outC, 1UL, L.outC, L.act);
    }
    unsigned long Pin = L.inH * L.inW;
    unsigned long P = L.outH * L.outW;
    unsigned long planes = L.inC * B;
    if (k == ROCM_LAYER_CONV || k == ROCM_LAYER_CONVT) {
        unsigned long G = L.conv.groups;
        unsigned long kk = L.conv.kH * L.conv.kW;
        struct NrGeom_ g = nr_conv_geom_(&L.conv);
        unsigned long ci = L.inC / G;
        unsigned long co = L.outC / G;
        unsigned long gi = 0UL;
        if (nr_mio_on_()) {
            int plan = nr_mio_plan_(L, B, x, zt, wt, L.dh);
            if (plan >= 0)
                return nr_mio_run_(plan, 0, x, wt, zt) &&
                       (L.foldBias || nr_bias_act_(zt, L.h, L.b, L.outC * B * P, P, L.outC, L.act));
        }
        if (g_nr_prec != 0 || !nr_conv_scratch_(n, L)) return 0;
        if (k == ROCM_LAYER_CONV) {
            // im2col -> per group W x col into cm [outC][B*P] -> NCHW with bias and activation
            unsigned long K = ci * kk;
            unsigned long N = B * P;
            if (!nr_im2col_(x, L.col, L.inC, L.inH, L.inW, L.outH, L.outW, g, B)) return 0;
            while (gi < G) {
                if (!rocm_gemm(nr_at_(wt, gi * co * K, co * K), 0, nr_at_(L.col, gi * K * N, K * N), 0,
                               nr_at_(L.cm, gi * co * N, co * N), co, N, K)) return 0;
                gi = gi + 1UL;
            }
            return nr_cm_out_(L.cm, nr_none_(), L.foldBias ? nr_none_() : L.b, zt, L.h, L.outC, B, P, L.act);
        }
        // input to channel-major cm [inC][B*Pin], per group W^T x cm into col, col2im to NCHW
        unsigned long KT = co * kk;
        unsigned long Nin = B * Pin;
        if (!nr_swap01_(x, L.cm, B, L.inC, Pin)) return 0;
        while (gi < G) {
            if (!rocm_gemm(nr_at_(wt, gi * ci * KT, ci * KT), 1, nr_at_(L.cm, gi * ci * Nin, ci * Nin), 0,
                           nr_at_(L.col, gi * KT * Nin, KT * Nin), KT, Nin, ci)) return 0;
            gi = gi + 1UL;
        }
        // col2im adds the bias and applies the activation
        if (L.foldBias) return nr_col2im_(L.col, nr_none_(), L.h, nr_none_(), ACT_NONE, L.outC, L.outH, L.outW, L.inH, L.inW, g, B);
        return nr_col2im_(L.col, zt, L.h, L.b, L.act, L.outC, L.outH, L.outW, L.inH, L.inW, g, B);
    }
    if (k == ROCM_LAYER_MAXPOOL || k == ROCM_LAYER_AVGPOOL) {
        int mode = k == ROCM_LAYER_MAXPOOL ? 0 : (L.pool.countIncludePad ? 1 : 2);
        return nr_pool_fwd_(x, zt, L.h, L.stats, planes, L.inH, L.inW, L.outH, L.outW, nr_pool_geom_(&L.pool), mode,
                            L.act);
    }
    if (k == ROCM_LAYER_GLOBAL_AVGPOOL || k == ROCM_LAYER_GLOBAL_MAXPOOL)
        return nr_gpool_fwd_(x, zt, L.h, L.stats, planes, Pin, k == ROCM_LAYER_GLOBAL_MAXPOOL ? 0 : 1, L.act);
    if (k == ROCM_LAYER_UPSAMPLE) return nr_upsample_(0, x, zt, L.h, planes, L.inH, L.inW, L.factor, L.act);
    if (k == ROCM_LAYER_RESHAPE) {
        // the same NCHW rows
        return nr_bias_act_(x, L.h, nr_none_(), L.outC * B * P, 1UL, 1UL, L.act) &&
               (L.z.dev == 0UL || nr_copy_(x, L.z, L.outC * B * P));
    }
    if (k >= ROCM_LAYER_SAVE) return nr_skip_fwd_(n, L, x, zt, B);
    if (k == ROCM_LAYER_BATCHNORM) {
        unsigned long S = n.segments == 0UL ? 1UL : n.segments;
        if (S > ROCM_MAX_SEGMENTS || B % S != 0UL || L.inC * S > 65535UL) return 0;
        struct NrBnArgs_ a = nr_bn_args_(n, L, x, nr_none_(), zt, L.h, B, S);
        if (n.training) {
            if (!nr_bn_launch_(NR_BN_STATS, a, 0, nr_bn_slices_(L, B, S))) return 0;
            if (S > 1UL && !nr_bn_launch_(NR_BN_FINISH, a, 0, 1UL)) return 0;
        } else if (!nr_bn_launch_(NR_BN_FINISH, a, 2, 1UL)) {
            return 0;
        }
        return nr_bn_launch_(NR_BN_APPLY, a, 0, 1UL);
    }
    // group norm
    unsafe {
        void* px = (void*)x.dev; void* pz = (void*)zt.dev; void* ph = (void*)L.h.dev; void* pg = (void*)L.w.dev;
        void* pb = (void*)L.b.dev; void* ps = (void*)L.stats.dev;
        unsigned int a0 = (unsigned int)L.inC; unsigned int a1 = (unsigned int)B; unsigned int a2 = (unsigned int)P;
        unsigned int a3 = (unsigned int)L.factor;
        float ep = L.eps; int ac = L.act;
        void* p[12];
        p[0] = (void*)&px; p[1] = (void*)&pz; p[2] = (void*)&ph; p[3] = (void*)&pg; p[4] = (void*)&pb;
        p[5] = (void*)&ps; p[6] = (void*)&a0; p[7] = (void*)&a1; p[8] = (void*)&a2; p[9] = (void*)&a3;
        p[10] = (void*)&ep; p[11] = (void*)&ac;
        return nr_launch_grid_(NR_GN_FWD, B * L.factor, 1UL, (&void)&p[0]);
    }
}

// Gradients of L's parameters from L.dh (made the pre-activation gradient
// in place) and its input x; d(loss)/dx into dx when needDx.
static int nr_layer_bwd_core_(&RocmNet n, &RocmLayer L, struct RocmBuf x, unsigned long B, int needDx,
                              struct RocmBuf dx) {
    struct RocmBuf src = L.z.dev != 0UL ? L.z : L.h;
    int k = L.kind;
    struct RocmBuf wt = nr_wk_(L);
    if (k == ROCM_LAYER_LINEAR) {
        unsigned long in = nr_in_size_(L);
        if (!nr_act_grad_rows_(L.dh, src, L.gb, B, L.outC, L.act)) return 0;
        if (g_nr_prec != 0) {
            struct RocmHalfBuf d = nr_hv_(L.dh, B * L.outC);
            return rocm_half_gemm_f32(nr_hv_(x, B * in), 1, d, 0, L.gw, in, L.outC, B) &&
                   (!needDx || rocm_half_gemm(d, 0, nr_hv_(wt, in * L.outC), 1, nr_hv_(dx, B * in), B, in, L.outC));
        }
        if (!rocm_gemm(x, 1, L.dh, 0, L.gw, in, L.outC, B)) return 0;
        return !needDx || rocm_gemm(L.dh, 0, wt, 1, dx, B, in, L.outC);
    }
    unsigned long Pin = L.inH * L.inW;
    unsigned long P = L.outH * L.outW;
    unsigned long planes = L.inC * B;
    if (k == ROCM_LAYER_CONV || k == ROCM_LAYER_CONVT) {
        unsigned long N = B * P;
        // a folded bias's gradient comes from the batch norm (already run)
        if (!L.foldBias && !nr_act_grad_(L.dh, src, L.gb, L.part, N, P, L.outC, L.act)) return 0;
        if (nr_mio_on_()) {
            int plan = nr_mio_find_plan_(L, B);
            if (plan >= 0 && g_mio_good[plan])
                return nr_mio_dw_(plan, L, L.dh, x) && (!needDx || nr_mio_run_(plan, 1, L.dh, wt, dx));
        }
        if (g_nr_prec != 0) return 0;
        unsigned long G = L.conv.groups;
        unsigned long kk = L.conv.kH * L.conv.kW;
        struct NrGeom_ g = nr_conv_geom_(&L.conv);
        unsigned long ci = L.inC / G;
        unsigned long co = L.outC / G;
        unsigned long gi = 0UL;
        if (k == ROCM_LAYER_CONV) {
            // dz to channel-major cm; dW = dz col^T; dcol = W^T dz; col2im to NCHW dx
            unsigned long K = ci * kk;
            if (!nr_swap01_(L.dh, L.cm, B, L.outC, P)) return 0;
            while (gi < G) {
                if (!nr_gemm_(nr_at_(L.cm, gi * co * N, co * N), 0, nr_at_(L.col, gi * K * N, K * N), 1,
                              nr_at_(L.gw, gi * co * K, co * K), co, K, N)) return 0;
                gi = gi + 1UL;
            }
            if (!needDx) return 1;
            gi = 0UL;
            while (gi < G) {
                if (!rocm_gemm(nr_at_(wt, gi * co * K, co * K), 1, nr_at_(L.cm, gi * co * N, co * N), 0,
                               nr_at_(L.col, gi * K * N, K * N), K, N, co)) return 0;
                gi = gi + 1UL;
            }
            return nr_col2im_(L.col, nr_none_(), dx, nr_none_(), ACT_NONE, L.inC, L.inH, L.inW, L.outH, L.outW, g, B);
        }
        // dcol = im2col(dz); dW = x_cm dcol^T (cm still holds the input); dx_cm = W dcol -> NCHW
        unsigned long KT = co * kk;
        unsigned long Nin = B * Pin;
        if (!nr_im2col_(L.dh, L.col, L.outC, L.outH, L.outW, L.inH, L.inW, g, B)) return 0;
        while (gi < G) {
            if (!nr_gemm_(nr_at_(L.cm, gi * ci * Nin, ci * Nin), 0, nr_at_(L.col, gi * KT * Nin, KT * Nin), 1,
                          nr_at_(L.gw, gi * ci * KT, ci * KT), ci, KT, Nin)) return 0;
            gi = gi + 1UL;
        }
        if (!needDx) return 1;
        gi = 0UL;
        while (gi < G) {
            if (!rocm_gemm(nr_at_(wt, gi * ci * KT, ci * KT), 0, nr_at_(L.col, gi * KT * Nin, KT * Nin), 0,
                           nr_at_(L.cm, gi * ci * Nin, ci * Nin), ci, Nin, KT)) return 0;
            gi = gi + 1UL;
        }
        return nr_swap01_(L.cm, dx, L.inC, B, Pin);
    }
    if (k == ROCM_LAYER_BATCHNORM) {
        // the activation's gradient is applied inside bn_grads / bn_apply
        unsigned long S = n.segments == 0UL ? 1UL : n.segments;
        struct RocmBuf dst = needDx ? dx : L.dh;   // in place is safe: each value is read before it is written
        struct NrBnArgs_ a = nr_bn_args_(n, L, x, L.dh, nr_none_(), dst, B, S);
        if (!nr_bn_launch_(NR_BN_GRADS, a, 0, nr_bn_slices_(L, B, S))) return 0;
        if (S > 1UL && !nr_bn_launch_(NR_BN_FINISH, a, 1, 1UL)) return 0;
        return nr_bn_launch_(NR_BN_APPLY, a, 1, 1UL);
    }
    if (!nr_act_ew_(L.dh, src, L.outC * B * P, L.act)) return 0;
    if (k >= ROCM_LAYER_SAVE) return nr_skip_bwd_(n, L, x, B, needDx, dx);
    if (k == ROCM_LAYER_GROUPNORM) {
        unsafe {
            void* px = (void*)x.dev; void* pd = (void*)L.dh.dev; void* po = (void*)dx.dev; void* pg = (void*)L.w.dev;
            void* ps = (void*)L.stats.dev; void* pgw = (void*)L.gw.dev; void* pgb = (void*)L.gb.dev;
            unsigned int a0 = (unsigned int)L.inC; unsigned int a1 = (unsigned int)B; unsigned int a2 = (unsigned int)P;
            unsigned int a3 = (unsigned int)L.factor;
            void* p[9];
            p[0] = (void*)&px; p[1] = (void*)&pd; p[2] = (void*)&ps; p[3] = (void*)&pgw; p[4] = (void*)&pgb;
            p[5] = (void*)&a0; p[6] = (void*)&a1; p[7] = (void*)&a2; p[8] = (void*)&a3;
            if (!nr_launch_grid_(NR_GN_PARAMS, L.inC, 1UL, (&void)&p[0])) return 0;
            if (!needDx) return 1;
            void* q[9];
            q[0] = (void*)&px; q[1] = (void*)&pd; q[2] = (void*)&po; q[3] = (void*)&pg; q[4] = (void*)&ps;
            q[5] = (void*)&a0; q[6] = (void*)&a1; q[7] = (void*)&a2; q[8] = (void*)&a3;
            return nr_launch_grid_(NR_GN_BWD, B * L.factor, 1UL, (&void)&q[0]);
        }
    }
    if (!needDx) return 1;
    if (k == ROCM_LAYER_MAXPOOL || k == ROCM_LAYER_AVGPOOL) {
        int mode = k == ROCM_LAYER_MAXPOOL ? 0 : (L.pool.countIncludePad ? 1 : 2);
        return nr_pool_bwd_(L.dh, L.stats, dx, planes, L.inH, L.inW, L.outH, L.outW, nr_pool_geom_(&L.pool), mode);
    }
    if (k == ROCM_LAYER_GLOBAL_AVGPOOL || k == ROCM_LAYER_GLOBAL_MAXPOOL)
        return nr_gpool_bwd_(L.dh, L.stats, dx, planes, Pin, k == ROCM_LAYER_GLOBAL_MAXPOOL ? 0 : 1);
    if (k == ROCM_LAYER_UPSAMPLE) return nr_upsample_(1, L.dh, nr_none_(), dx, planes, L.inH, L.inW, L.factor, ACT_NONE);
    return nr_copy_(L.dh, dx, L.outC * B * P);   // reshape
}

static int nr_layer_bwd_(&RocmNet n, &RocmLayer L, struct RocmBuf x, unsigned long B, int needDx, struct RocmBuf dx) {
    if (!nr_layer_bwd_core_(n, L, x, B, needDx, dx)) return 0;
    return !L.spectral || nr_sn_backward_(L);
}

// A 16-bit net's weights rounded for this pass (spectral ones in their layer).
static int nr_round_weights_(&RocmNet n) {
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        &RocmLayer L = n.layers.at(i);
        if (nr_has_wh_(L) && !L.spectral && !nr_cvt_(1, L.w, L.wh, L.w.len, nr_none_(), 0)) return 0;
        i = i + 1UL;
    }
    return 1;
}

// A new fp32 convolution shape (MIOpen plan 'plan'): its forward and
// backward timed on both engines (best of 3 after one run each), and
// MIOpen dropped for it when the im2col engine is faster (small
// convolutions, where MIOpen's cost per call dominates). The scratch the
// im2col engine needs is freed again when MIOpen wins.
static void nr_engine_pick_(&RocmNet n, &RocmLayer L, struct RocmBuf x, unsigned long B, int plan) {
    if (!nr_conv_scratch_(n, L)) return;
    double best[2];
    int e = 0;
    while (e < 2) {
        g_mio_engine = e == 0 ? ROCM_CONV_AUTO : ROCM_CONV_IM2COL;
        best[e] = 1.0e30;
        int r = 0;
        while (r < 4) {
            rocm_sync();
            long long t0 = time_mono_ns();
            int ok = nr_layer_fwd_(n, L, x, B) && nr_layer_bwd_core_(n, L, x, B, 1, g_mio_tmp);
            ok = rocm_sync() && ok;
            double t = (double)(time_mono_ns() - t0);
            if (ok && r > 0 && t < best[e]) best[e] = t;
            r = r + 1;
        }
        e = e + 1;
    }
    g_mio_engine = ROCM_CONV_AUTO;
    // (a margin keeps timing noise from moving a shape off MIOpen)
    if (best[1] < (double)0.9 * best[0]) {
        g_mio_good[plan] = 0;
        nr_layer_fwd_(n, L, x, B);   // the scratch the backward pass reads (the timed one overwrote it)
    } else {
        L.col.free();
        L.cm.free();
    }
}

static int nr_forward_(&RocmNet n, struct RocmBuf x, unsigned long batch) {
    struct RocmBuf cur = x;
    if (n.precision != 0) {
        if (!nr_cvt_(1, x, n.xh, batch * n.in_dim(), nr_none_(), 0) || !nr_round_weights_(n)) return 0;
        cur = n.xh;
    }
    unsigned long c = n.layers.length();
    unsigned long i = 0UL;
    while (i < c) {
        &RocmLayer L = n.layers.at(i);
        g_mio_fresh = -1;
        if (!nr_layer_fwd_(n, L, cur, batch)) return 0;
        if (g_mio_fresh >= 0 && n.precision == 0 && !L.spectral &&
            (L.kind == ROCM_LAYER_CONV || L.kind == ROCM_LAYER_CONVT)) nr_engine_pick_(n, L, cur, batch, g_mio_fresh);
        cur = L.h;
        i = i + 1UL;
    }
    return n.precision == 0 || nr_cvt_(0, cur, n.out32, batch * n.out_dim(), nr_none_(), 0);
}

struct RocmBuf RocmNet::forward(struct RocmBuf x, unsigned long batch) {
    unsigned long c = self.layers.length();
    if (!self.ok || c == 0UL || batch == 0UL || batch > self.maxBatch || !nr_fits_(x, batch * self.in_dim()))
        return nr_none_();
    g_nr_prec = self.precision;
    int ok = nr_forward_(self, x, batch);
    g_nr_prec = 0;
    if (!ok) return nr_none_();
    &RocmLayer last = self.layers.at(c - 1UL);
    if (self.precision != 0) return nr_view_(self.out32, batch * nr_out_size_(last));
    return nr_view_(last.h, batch * nr_out_size_(last));
}

struct RocmBuf RocmNet::grad_output() const {
    unsigned long c = self.layers.length();
    if (c == 0UL) return nr_none_();
    if (self.precision != 0) return self.dout32;
    return self.layers.at(c - 1UL).dh;
}

static int nr_backward_(&RocmNet n, struct RocmBuf x, unsigned long batch, struct RocmBuf dx) {
    unsigned long c = n.layers.length();
    // fp16: the loss scale in the gradients from here back
    struct RocmBuf ls = n.precision == ROCM_FP16 ? n.scale : nr_none_();
    struct RocmBuf in0 = x;
    struct RocmBuf dx0 = dx;
    if (n.precision != 0) {
        &RocmLayer last = n.layers.at(c - 1UL);
        if (!nr_cvt_(1, n.dout32, last.dh, batch * nr_out_size_(last), ls, 0)) return 0;
        in0 = n.xh;
        if (dx.dev != 0UL) dx0 = n.dxh;
    }
    // SAVE layers: no skip gradient written yet (flag)
    unsigned long i = 0UL;
    while (i < c) {
        &RocmLayer L = n.layers.at(i);
        if (L.kind == ROCM_LAYER_SAVE) L.flag = 0;
        i = i + 1UL;
    }
    i = c;
    while (i > 0UL) {
        i = i - 1UL;
        &RocmLayer L = n.layers.at(i);
        struct RocmBuf in = i > 0UL ? n.layers.at(i - 1UL).h : in0;
        struct RocmBuf target = i > 0UL ? n.layers.at(i - 1UL).dh : dx0;
        int needDx = i > 0UL || dx.dev != 0UL;
        if (needDx && i == 0UL && !nr_fits_(dx, batch * nr_in_size_(L))) return 0;
        if (!nr_layer_bwd_(n, L, in, batch, needDx, target)) return 0;
    }
    return n.precision == 0 || dx.dev == 0UL || nr_cvt_(0, n.dxh, dx, batch * n.in_dim(), ls, 1);
}

int RocmNet::backward(struct RocmBuf x, unsigned long batch, struct RocmBuf dx) {
    unsigned long c = self.layers.length();
    if (!self.ok || c == 0UL || batch == 0UL || batch > self.maxBatch) return 0;
    g_nr_prec = self.precision;
    int ok = nr_backward_(self, x, batch, dx);
    g_nr_prec = 0;
    return ok;
}

int RocmNet::set_loss_scale(float scale) {
    if (self.scale.dev == 0UL || !(scale > (float)0.0)) return 0;
    float v[2];
    v[0] = scale; v[1] = (float)0.0;
    return nr_write_(self.scale, &v[0], 2UL);
}

float RocmNet::loss_scale() const {
    if (self.precision != ROCM_FP16 || self.scale.dev == 0UL) return (float)1.0;
    float v[2];
    v[0] = (float)1.0;
    if (!rocm_read(self.scale, &v[0], 2UL)) return (float)0.0;
    return v[0];
}

int RocmNet::set_precision(int dtype) {
    if (!self.ok || self.layers.length() == 0UL || (dtype != 0 && dtype != ROCM_FP16 && dtype != ROCM_BF16)) return 0;
    if (dtype == 0) { self.precision = 0; return 1; }
    if (!nr_ensure_prec_(dtype)) return 0;
    unsigned long i = 0UL;
    while (i < self.layers.length()) {
        &RocmLayer L = self.layers.at(i);
        int k = L.kind;
        int conv = k == ROCM_LAYER_CONV || k == ROCM_LAYER_CONVT || (k == ROCM_LAYER_ADD && L.flag);
        if (conv && !nr_mio_on_()) return 0;
        if (nr_has_wh_(L) && L.wh.dev == 0UL && !nr_alloc_(&L.wh, (L.w.len + 1UL) / 2UL)) return 0;
        i = i + 1UL;
    }
    unsigned long B = self.maxBatch;
    unsigned long in = self.in_dim();
    unsigned long out = self.out_dim();
    if (self.scale.dev == 0UL &&
        (!nr_alloc_(&self.scale, 2UL) || !nr_alloc_(&self.xh, (B * in + 1UL) / 2UL) || !nr_alloc_(&self.dxh, (B * in + 1UL) / 2UL) ||
         !nr_alloc_(&self.out32, B * out) || !nr_alloc_(&self.dout32, B * out))) {
        self.ok = 0;
        return 0;
    }
    self.precision = dtype;
    return self.set_loss_scale(dtype == ROCM_FP16 ? (float)65536.0 : (float)1.0);
}

// ── Training steps ───────────────────────────────────────────────────────────

// The kernel's NetTab.
struct NrNetTab_ {
    unsigned long w[8];
    unsigned long b[8];
    unsigned long gw[8];
    unsigned long gb[8];
    unsigned long h[8];
    unsigned long z[8];
    unsigned long dh[8];
    unsigned int  in[8];
    unsigned int  out[8];
    int           act[8];
    int           n;
};

#define NR_SMALL_LDS  16000UL    // the kernel's SN_LDS
#define NR_SMALL_MACS 262144UL   // multiply-adds per forward pass; above this the layer-by-layer path wins

// Small fully connected nets run a whole step in one workgroup
// (small_net_step); the per-launch cost (~1.3 us on an MI100, also inside
// a graph) dominates the layer-by-layer path for them. It needs the batch's
// input, every activation (twice for GELU / SiLU), two gradient buffers and
// the largest padded weight in LDS.
static int nr_small_(const &RocmNet n, unsigned long rows) {
    unsigned long c = n.layers.length();
    if (c == 0UL || c > 8UL || n.precision != 0) return 0;
    unsigned long macs = 0UL;
    unsigned long floats = rows * n.layers.at(0UL).inC;
    unsigned long maxO = 0UL;
    unsigned long maxW = 0UL;
    unsigned long i = 0UL;
    while (i < c) {
        &RocmLayer L = n.layers.at(i);
        if (L.kind != ROCM_LAYER_LINEAR) return 0;
        macs = macs + rows * L.inC * L.outC;
        floats = floats + rows * L.outC * (nr_needs_z_(L.act) ? 2UL : 1UL);
        if (L.outC > maxO) maxO = L.outC;
        if (L.inC > maxO) maxO = L.inC;
        if (L.inC * (L.outC + 1UL) > maxW) maxW = L.inC * (L.outC + 1UL);
        i = i + 1UL;
    }
    floats = floats + 2UL * rows * maxO + maxW;
    return macs <= NR_SMALL_MACS && floats <= NR_SMALL_LDS;
}

// kind: 0 cross-entropy, 1 BCE on logits, 2 MSE
static int nr_small_step_(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1,
                          int kind) {
    unsigned long c = n.layers.length();
    unsigned long outDim = n.out_dim();
    if (!n.ok || rows == 0UL || rows > n.maxBatch || !nr_fits_(x, rows * n.in_dim()) ||
        !nr_fits_(y, kind == 0 ? rows : rows * outDim) || !nr_fits_(loss1, 1UL)) return 0;
    struct NrNetTab_ t;
    unsigned long i = 0UL;
    while (i < 8UL) {
        if (i < c) {
            &RocmLayer L = n.layers.at(i);
            t.w[i] = L.w.dev; t.b[i] = L.b.dev; t.gw[i] = L.gw.dev; t.gb[i] = L.gb.dev;
            t.h[i] = L.h.dev; t.z[i] = L.z.dev; t.dh[i] = L.dh.dev;
            t.in[i] = (unsigned int)L.inC; t.out[i] = (unsigned int)L.outC; t.act[i] = L.act;
        } else {
            t.w[i] = 0UL; t.b[i] = 0UL; t.gw[i] = 0UL; t.gb[i] = 0UL; t.h[i] = 0UL; t.z[i] = 0UL; t.dh[i] = 0UL;
            t.in[i] = 0U; t.out[i] = 0U; t.act[i] = 0;
        }
        i = i + 1UL;
    }
    t.n = (int)c;
    unsafe {
        void* px = (void*)x.dev; void* py = (void*)y.dev; void* pl = (void*)loss1.dev;
        unsigned int r = (unsigned int)rows;
        int k = kind;
        void* p[6];
        p[0] = (void*)&t; p[1] = (void*)&px; p[2] = (void*)&py; p[3] = (void*)&pl; p[4] = (void*)&r; p[5] = (void*)&k;
        return hipModuleLaunchKernel(g_nr_fn[NR_SMALL], 1U, 1U, 1U, 1024U, 1U, 1U, 0U, g_nr_stream, &p[0],
                                     (void**)0) == 0;
    }
}

int rocm_net_ce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    if (nr_small_(n, rows)) return nr_small_step_(n, x, y, rows, loss1, 0);
    struct RocmBuf out = n.forward(x, rows);
    if (out.dev == 0UL) return 0;
    return rocm_cross_entropy(out, y, n.grad_output(), loss1, rows, n.out_dim()) &&
           n.backward(x, rows, nr_none_());
}

static int nr_pointwise_step_(&RocmNet n, int kernel, struct RocmBuf x, struct RocmBuf t, unsigned long rows,
                              struct RocmBuf loss1) {
    if (nr_small_(n, rows)) return nr_small_step_(n, x, t, rows, loss1, kernel == NR_BCE ? 1 : 2);
    struct RocmBuf out = n.forward(x, rows);
    if (out.dev == 0UL) return 0;
    return nr_pointwise_loss_(kernel, out, t, n.grad_output(), loss1, rows * n.out_dim(), (float)1.0) &&
           n.backward(x, rows, nr_none_());
}

int rocm_net_mse_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    return nr_pointwise_step_(n, NR_MSE, x, y, rows, loss1);
}

int rocm_net_bce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    return nr_pointwise_step_(n, NR_BCE, x, y, rows, loss1);
}

int rocm_net_recon_bce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    return nr_pointwise_step_(n, NR_BCE, x, x, rows, loss1);
}

int rocm_net_recon_mse_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    return nr_pointwise_step_(n, NR_MSE, x, x, rows, loss1);
}

// ── Optimizer ────────────────────────────────────────────────────────────────

// st[0..6] from the host fields
static void nr_opt_fields_(const &RocmOptimizer o, &float out) {
    unsafe {
        float* f = (float*)out;
        f[0] = o.lr; f[1] = o.momentum; f[2] = o.beta1; f[3] = o.beta2;
        f[4] = o.eps; f[5] = o.weightDecay; f[6] = o.clipNorm;
    }
}

struct RocmOptimizer rocm_optimizer_new(int kind, float lr) {
    struct RocmOptimizer o;
    o.w = vec_new(); o.g = vec_new(); o.m = vec_new(); o.v = vec_new();
    o.kind = kind;
    o.lr = lr;
    o.momentum = (float)0.9;
    o.beta1 = (float)0.9;
    o.beta2 = kind == OPT_RMSPROP || kind == OPT_LION ? (float)0.99 : (float)0.999;
    o.eps = kind == OPT_ADAGRAD ? (float)0.0000000001 : (float)0.00000001;
    o.weightDecay = kind == OPT_ADAMW ? (float)0.01 : (float)0.0;
    o.clipNorm = (float)0.0;
    o.clampWeights = (float)0.0;
    o.t = 0UL;
    o.lossScale = nr_none_();
    o.ok = nr_ensure_();
    o.state = o.ok ? rocm_alloc(13UL) : nr_none_();
    if (o.state.dev == 0UL) { o.ok = 0; return o; }
    float init[13];
    unsigned long z = 0UL;
    while (z < 13UL) { init[z] = (float)0.0; z = z + 1UL; }
    nr_opt_fields_(&o, &init[0]);
    o.ok = nr_write_(o.state, &init[0], 13UL);
    unsigned long i = 0UL;
    while (i < 7UL) { o.sent[i] = init[i]; i = i + 1UL; }
    return o;
}

int RocmOptimizer::sync() {
    if (!self.ok) return 0;
    float f[7];
    nr_opt_fields_(self, &f[0]);
    int same = 1;
    unsigned long i = 0UL;
    while (i < 7UL) { if (f[i] != self.sent[i]) same = 0; i = i + 1UL; }
    if (same) return 1;
    if (!nr_write_(self.state, &f[0], 7UL)) return 0;
    i = 0UL;
    while (i < 7UL) { self.sent[i] = f[i]; i = i + 1UL; }
    return 1;
}

int RocmOptimizer::add(struct RocmBuf w, struct RocmBuf g) {
    if (!self.ok || w.dev == 0UL || !nr_fits_(g, w.len)) return 0;
    struct RocmBuf m = rocm_alloc(w.len);
    struct RocmBuf v = rocm_alloc(w.len);
    if (m.dev == 0UL || v.dev == 0UL || !rocm_zero(m) || !rocm_zero(v)) {
        m.free(); v.free();
        self.ok = 0;
        return 0;
    }
    self.w.push(w); self.g.push(g); self.m.push(m); self.v.push(v);
    return 1;
}

int RocmNet::add_params(&RocmOptimizer o) const {
    if (!self.ok) { o.ok = 0; return 0; }
    if (self.precision == ROCM_FP16) o.lossScale = self.scale;
    unsigned long i = 0UL;
    while (i < self.layers.length()) {
        &RocmLayer L = self.layers.at(i);
        // layers without parameters (pooling, resize, skips without a projection, ...) have none to add
        if (L.w.dev != 0UL && (!o.add(L.w, L.gw) || !o.add(L.b, L.gb))) return 0;
        i = i + 1UL;
    }
    return 1;
}

// The kernel's OptTab.
struct NrOptTab_ {
    unsigned long w[32];
    unsigned long g[32];
    unsigned long m[32];
    unsigned long v[32];
    unsigned int  n[32];
};

// opt_sumsq or opt_step over every tensor, NR_TAB per launch.
static int nr_opt_launch_(const &RocmOptimizer o, int kernel, int clip) {
    unsigned long count = o.w.length();
    unsigned long base = 0UL;
    while (base < count) {
        struct NrOptTab_ t;
        unsigned long c = count - base < NR_TAB ? count - base : NR_TAB;
        unsigned long most = 1UL;
        unsigned long i = 0UL;
        while (i < NR_TAB) {
            if (i < c) {
                struct RocmBuf w = *o.w.at(base + i);
                t.w[i] = w.dev; t.g[i] = o.g.at(base + i).dev;
                t.m[i] = o.m.at(base + i).dev; t.v[i] = o.v.at(base + i).dev;
                t.n[i] = (unsigned int)w.len;
                if (w.len > most) most = w.len;
            } else {
                t.w[i] = 0UL; t.g[i] = 0UL; t.m[i] = 0UL; t.v[i] = 0UL; t.n[i] = 0U;
            }
            i = i + 1UL;
        }
        // Few blocks, several elements per thread: every block of the last
        // launch takes a turn on one atomic counter (the step count).
        unsigned long blocks = (most + 4UL * (unsigned long)NR_BLOCK - 1UL) / (4UL * (unsigned long)NR_BLOCK);
        if (blocks > 64UL) blocks = 64UL;
        unsafe {
            void* ps = (void*)o.state.dev;
            int kind = o.kind;
            int cl = clip;
            int fin = base + c == count ? 1 : 0;
            float cw = o.clampWeights;
            void* pl = (void*)o.lossScale.dev;
            void* p[7];
            p[0] = (void*)&t; p[1] = (void*)&ps; p[2] = (void*)&kind; p[3] = (void*)&cl; p[4] = (void*)&fin;
            p[5] = (void*)&cw; p[6] = (void*)&pl;
            if (kernel == NR_SUMSQ) p[2] = (void*)&pl;
            if (!nr_launch_grid_(kernel, blocks, c, (&void)&p[0])) return 0;
        }
        base = base + c;
    }
    return 1;
}

int RocmOptimizer::step() {
    if (!self.ok) return 0;
    if (!nr_capturing_() && !self.sync()) return 0;
    if (self.w.length() == 0UL) return 1;
    int clip = self.clipNorm > (float)0.0 ? 1 : 0;
    if ((clip || self.lossScale.dev != 0UL) && !nr_opt_launch_(self, NR_SUMSQ, clip)) return 0;
    if (!nr_opt_launch_(self, NR_OPT, clip)) return 0;
    self.t = self.t + 1UL;
    return 1;
}

void RocmOptimizer::free() {
    unsigned long i = 0UL;
    while (i < self.m.length()) { self.m.at(i).free(); self.v.at(i).free(); i = i + 1UL; }
    self.w.free(); self.g.free(); self.m.free(); self.v.free();
    self.state.free();
    self.ok = 0;
}

// ── Row ops ──────────────────────────────────────────────────────────────────

int rocm_append_labels(struct RocmBuf x, struct RocmBuf labels, struct RocmBuf out, unsigned long rows,
                       unsigned long channels, unsigned long plane, unsigned long classes) {
    unsigned long n = rows * (channels + classes) * plane;
    if (!nr_ensure_() || n == 0UL || !nr_fits_(x, rows * channels * plane) || !nr_fits_(labels, rows) ||
        !nr_fits_(out, n)) return 0;
    unsafe {
        void* px = (void*)x.dev; void* pl = (void*)labels.dev; void* po = (void*)out.dev;
        unsigned int a0 = (unsigned int)rows; unsigned int a1 = (unsigned int)(channels * plane);
        unsigned int a2 = (unsigned int)plane; unsigned int a3 = (unsigned int)classes;
        void* p[8];
        p[0] = (void*)&px; p[1] = (void*)&pl; p[2] = (void*)&po; p[3] = (void*)&a0; p[4] = (void*)&a1;
        p[5] = (void*)&a2; p[6] = (void*)&a3; p[7] = (void*)&a0;   // one label per row
        return nr_launch_(NR_APPEND, n, (&void)&p[0]);
    }
}

int rocm_slice_cols(struct RocmBuf src, struct RocmBuf dst, unsigned long rows, unsigned long srcCols,
                    unsigned long dstCols, unsigned long offset) {
    unsigned long n = rows * dstCols;
    if (!nr_ensure_() || n == 0UL || offset + dstCols > srcCols || !nr_fits_(src, rows * srcCols) || !nr_fits_(dst, n))
        return 0;
    unsafe {
        void* ps = (void*)src.dev; void* pd = (void*)dst.dev;
        unsigned int a0 = (unsigned int)rows; unsigned int a1 = (unsigned int)srcCols;
        unsigned int a2 = (unsigned int)dstCols; unsigned int a3 = (unsigned int)offset;
        void* p[6];
        p[0] = (void*)&ps; p[1] = (void*)&pd; p[2] = (void*)&a0; p[3] = (void*)&a1; p[4] = (void*)&a2; p[5] = (void*)&a3;
        return nr_launch_(NR_SLICE, n, (&void)&p[0]);
    }
}

// ── VAE ──────────────────────────────────────────────────────────────────────

// The rest of a RocmVae once d.enc / d.dec and the geometry are set: the
// fused head from the mu / log-variance heads, and the scratch buffers.
static void nr_vae_finish_(&RocmVae d, const &Mlp muHead, const &Mlp logvarHead) {
    d.eps = nr_none_(); d.zbuf = nr_none_(); d.dz = nr_none_(); d.counter = nr_none_();
    d.encIn = nr_none_(); d.decIn = nr_none_(); d.decGrad = nr_none_();
    d.head = rocm_net_new(d.maxBatch);
    d.ok = d.enc.ok && d.head.ok && d.dec.ok;
    if (!d.ok) return;
    // fused head: W [hidden, 2L] rows (Wmu[r] | Wlv[r]), b (bmu | blv)
    unsigned long H = muHead.inDim;
    unsigned long L = d.latent;
    unsafe {
        &Tensor wm = *muHead.params.at(0UL);
        &Tensor bm = *muHead.params.at(1UL);
        &Tensor wl = *logvarHead.params.at(0UL);
        &Tensor bl = *logvarHead.params.at(1UL);
        float* w = (float*)malloc(sizeof(float) * H * 2UL * L);
        float* b = (float*)malloc(sizeof(float) * 2UL * L);
        if (w == (float*)0 || b == (float*)0) { free((void*)w); free((void*)b); d.ok = 0; return; }
        unsigned long r = 0UL;
        while (r < H) {
            unsigned long j = 0UL;
            while (j < L) {
                w[r * 2UL * L + j] = wm->data[r * L + j];
                w[r * 2UL * L + L + j] = wl->data[r * L + j];
                j = j + 1UL;
            }
            r = r + 1UL;
        }
        unsigned long j = 0UL;
        while (j < L) { b[j] = bm->data[j]; b[L + j] = bl->data[j]; j = j + 1UL; }
        d.ok = nr_add_linear_raw_(&d.head, (const &float)w, (const &float)b, H, 2UL * L, ACT_NONE) && rocm_sync();
        free((void*)w);
        free((void*)b);
    }
    unsigned long n = d.maxBatch * L;
    unsigned long c = d.classes;
    d.ok = d.ok && nr_alloc_(&d.eps, n) && nr_alloc_(&d.zbuf, n) && nr_alloc_(&d.dz, n) &&
           nr_alloc_(&d.counter, 1UL) && rocm_zero(d.counter);
    if (d.ok && c > 0UL)
        d.ok = nr_alloc_(&d.encIn, d.maxBatch * (d.encChannels + c) * d.encPlane) &&
               nr_alloc_(&d.decIn, d.maxBatch * (L + c)) && nr_alloc_(&d.decGrad, d.maxBatch * (L + c));
}

static struct RocmVae nr_vae_blank_(const &VaeConfig cfg, unsigned long maxBatch) {
    struct RocmVae d;
    d.maxBatch = maxBatch == 0UL ? 1UL : maxBatch;
    d.latent = cfg.latent;
    d.classes = cfg.classes;
    d.recon = cfg.recon;
    d.beta = cfg.beta;
    d.seed = cfg.seed;
    return d;
}

struct RocmVae rocm_vae_new(const &Vae v, unsigned long maxBatch) {
    struct RocmVae d = nr_vae_blank_(&v.cfg, maxBatch);
    d.inDim = v.dec.outDim;
    d.encChannels = d.inDim;
    d.encPlane = 1UL;
    d.enc = rocm_net_from_mlp(&v.enc, d.maxBatch);
    d.dec = rocm_net_from_mlp(&v.dec, d.maxBatch);
    nr_vae_finish_(&d, &v.muHead, &v.logvarHead);
    return d;
}

struct RocmVae rocm_conv_vae_new(const &ConvVae v, unsigned long maxBatch) {
    struct RocmVae d = nr_vae_blank_(&v.cfg, maxBatch);
    d.inDim = v.dec.out_dim();
    d.encChannels = v.enc.inC - v.cfg.classes;
    d.encPlane = v.enc.inH * v.enc.inW;
    d.enc = rocm_net_from_convnet(&v.enc, d.maxBatch);
    d.dec = rocm_net_from_convnet(&v.dec, d.maxBatch);
    d.enc.training = 1;   // conv_vae_loss runs both nets in training mode
    d.dec.training = 1;
    nr_vae_finish_(&d, &v.muHead, &v.logvarHead);
    return d;
}

// The fused head back into the mu / log-variance heads.
static int nr_vae_head_download_(const &RocmVae d, &Mlp muHead, &Mlp logvarHead) {
    &RocmLayer hl = d.head.layers.at(0UL);
    unsigned long H = hl.inC;
    unsigned long L = d.latent;
    int ok = 0;
    unsafe {
        float* w = (float*)malloc(sizeof(float) * H * 2UL * L);
        float* b = (float*)malloc(sizeof(float) * 2UL * L);
        if (w != (float*)0 && b != (float*)0 && rocm_read(hl.w, (&float)w, H * 2UL * L) &&
            rocm_read(hl.b, (&float)b, 2UL * L)) {
            &Tensor wm = *muHead.params.at(0UL);
            &Tensor bm = *muHead.params.at(1UL);
            &Tensor wl = *logvarHead.params.at(0UL);
            &Tensor bl = *logvarHead.params.at(1UL);
            unsigned long r = 0UL;
            while (r < H) {
                unsigned long j = 0UL;
                while (j < L) {
                    wm->data[r * L + j] = w[r * 2UL * L + j];
                    wl->data[r * L + j] = w[r * 2UL * L + L + j];
                    j = j + 1UL;
                }
                r = r + 1UL;
            }
            unsigned long j = 0UL;
            while (j < L) { bm->data[j] = b[j]; bl->data[j] = b[L + j]; j = j + 1UL; }
            ok = 1;
        }
        free((void*)w);
        free((void*)b);
    }
    return ok;
}

int RocmVae::download(&Vae v) const {
    return self.ok && self.enc.download(&v.enc.params) && self.dec.download(&v.dec.params) &&
           nr_vae_head_download_(self, &v.muHead, &v.logvarHead);
}

int RocmVae::download_conv_vae(&ConvVae v) const {
    return self.ok && self.enc.download_convnet(&v.enc) && self.dec.download_convnet(&v.dec) &&
           nr_vae_head_download_(self, &v.muHead, &v.logvarHead);
}

int rocm_vae_step(&RocmVae v, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1) {
    if (!v.ok || rows == 0UL || rows > v.maxBatch || !nr_fits_(loss1, 1UL)) return 0;
    unsigned long c = v.classes;
    unsigned long L = v.latent;
    struct RocmBuf encX = x;
    if (c > 0UL) {
        if (!rocm_append_labels(x, y, v.encIn, rows, v.encChannels, v.encPlane, c)) return 0;
        encX = v.encIn;
    }
    struct RocmBuf h = v.enc.forward(encX, rows);
    if (h.dev == 0UL) return 0;
    struct RocmBuf hd = v.head.forward(h, rows);
    if (hd.dev == 0UL) return 0;
    unsigned long n = rows * L;
    unsafe {
        void* ph = (void*)hd.dev; void* pe = (void*)v.eps.dev; void* pz = (void*)v.zbuf.dev;
        void* pl = (void*)loss1.dev; void* pc = (void*)v.counter.dev;
        unsigned int r = (unsigned int)rows; unsigned int LL = (unsigned int)L;
        float beta = v.beta;
        unsigned long seed = v.seed;
        void* p[9];
        p[0] = (void*)&ph; p[1] = (void*)&pe; p[2] = (void*)&pz; p[3] = (void*)&pl; p[4] = (void*)&pc;
        p[5] = (void*)&r; p[6] = (void*)&LL; p[7] = (void*)&beta; p[8] = (void*)&seed;
        if (!nr_launch_(NR_VAE_FWD, n, (&void)&p[0])) return 0;
    }
    struct RocmBuf decX = v.zbuf;
    if (c > 0UL) {
        if (!rocm_append_labels(v.zbuf, y, v.decIn, rows, L, 1UL, c)) return 0;
        decX = v.decIn;
    }
    struct RocmBuf out = v.dec.forward(decX, rows);
    if (out.dev == 0UL) return 0;
    // reconstruction summed over features, averaged over rows
    if (!nr_pointwise_loss_(v.recon == RECON_BCE ? NR_BCE : NR_MSE, out, x, v.dec.grad_output(), loss1,
                            rows * v.inDim, (float)v.inDim)) return 0;
    if (c > 0UL) {
        if (!v.dec.backward(decX, rows, v.decGrad) || !rocm_slice_cols(v.decGrad, v.dz, rows, L + c, L, 0UL))
            return 0;
    } else if (!v.dec.backward(decX, rows, v.dz)) {
        return 0;
    }
    struct RocmBuf dhd = v.head.grad_output();
    unsafe {
        void* ph = (void*)hd.dev; void* pe = (void*)v.eps.dev; void* pd = (void*)v.dz.dev;
        void* po = (void*)dhd.dev; void* pc = (void*)v.counter.dev;
        unsigned int r = (unsigned int)rows; unsigned int LL = (unsigned int)L;
        float beta = v.beta;
        void* p[8];
        p[0] = (void*)&ph; p[1] = (void*)&pe; p[2] = (void*)&pd; p[3] = (void*)&po; p[4] = (void*)&pc;
        p[5] = (void*)&r; p[6] = (void*)&LL; p[7] = (void*)&beta;
        if (!nr_launch_(NR_VAE_BWD, n, (&void)&p[0])) return 0;
    }
    return v.head.backward(h, rows, v.enc.grad_output()) &&
           v.enc.backward(encX, rows, nr_none_());
}

int RocmVae::add_params(&RocmOptimizer o) const {
    if (!self.ok) { o.ok = 0; return 0; }
    return self.enc.add_params(o) && self.head.add_params(o) &&
           self.dec.add_params(o);
}

void RocmVae::free() {
    self.enc.free(); self.head.free(); self.dec.free();
    self.eps.free(); self.zbuf.free(); self.dz.free(); self.counter.free();
    self.encIn.free(); self.decIn.free(); self.decGrad.free();
    self.ok = 0;
}

// ── Convolutional autoencoder ────────────────────────────────────────────────

struct RocmConvAe rocm_conv_autoencoder_new(const &ConvAutoencoder ae, unsigned long maxBatch) {
    struct RocmConvAe a;
    a.enc = rocm_net_from_convnet(&ae.enc, maxBatch);
    a.dec = rocm_net_from_convnet(&ae.dec, maxBatch);
    a.enc.training = 1;   // conv_autoencoder_loss runs both nets in training mode
    a.dec.training = 1;
    a.recon = ae.recon;
    a.ok = a.enc.ok && a.dec.ok && a.enc.out_dim() == a.dec.in_dim();
    return a;
}

int rocm_conv_autoencoder_step(&RocmConvAe a, struct RocmBuf x, struct RocmBuf y, unsigned long rows,
                               struct RocmBuf loss1) {
    if (!a.ok) return 0;
    struct RocmBuf z = a.enc.forward(x, rows);
    if (z.dev == 0UL) return 0;
    struct RocmBuf out = a.dec.forward(z, rows);
    if (out.dev == 0UL) return 0;
    return nr_pointwise_loss_(a.recon == RECON_BCE ? NR_BCE : NR_MSE, out, x, a.dec.grad_output(), loss1,
                              rows * a.dec.out_dim(), (float)1.0) &&
           a.dec.backward(z, rows, a.enc.grad_output()) &&
           a.enc.backward(x, rows, nr_none_());
}

int RocmConvAe::add_params(&RocmOptimizer o) const {
    return self.enc.add_params(o) && self.dec.add_params(o);
}

int RocmConvAe::download(&ConvAutoencoder ae) const {
    return self.ok && self.enc.download_convnet(&ae.enc) && self.dec.download_convnet(&ae.dec);
}

void RocmConvAe::free() {
    self.enc.free();
    self.dec.free();
    self.ok = 0;
}

// ── GANs ─────────────────────────────────────────────────────────────────────

// The kernel's EmaTab.
struct NrEmaTab_ {
    unsigned long d[32];
    unsigned long s[32];
    unsigned int  n[32];
};

// dst[i] = tau src[i] + (1 - tau) dst[i] over the lists (soft_update);
// the final launch advances state[0] and state[1].
static int nr_soft_update_(const &Vec<struct RocmBuf> dst, const &Vec<struct RocmBuf> src, struct RocmBuf state,
                           float decay, int schedule) {
    unsigned long count = dst.length();
    if (count != src.length() || count == 0UL) return 0;
    unsigned long base = 0UL;
    while (base < count) {
        struct NrEmaTab_ t;
        unsigned long c = count - base < NR_TAB ? count - base : NR_TAB;
        unsigned long most = 1UL;
        unsigned long i = 0UL;
        while (i < NR_TAB) {
            if (i < c) {
                struct RocmBuf d = *dst.at(base + i);
                struct RocmBuf s = *src.at(base + i);
                if (s.len < d.len) return 0;
                t.d[i] = d.dev; t.s[i] = s.dev; t.n[i] = (unsigned int)d.len;
                if (d.len > most) most = d.len;
            } else {
                t.d[i] = 0UL; t.s[i] = 0UL; t.n[i] = 0U;
            }
            i = i + 1UL;
        }
        unsigned long blocks = (most + 4UL * (unsigned long)NR_BLOCK - 1UL) / (4UL * (unsigned long)NR_BLOCK);
        if (blocks > 64UL) blocks = 64UL;
        unsafe {
            void* ps = (void*)state.dev;
            float dc = decay;
            int sc = schedule;
            int fin = base + c == count ? 1 : 0;
            void* p[5];
            p[0] = (void*)&t; p[1] = (void*)&ps; p[2] = (void*)&dc; p[3] = (void*)&sc; p[4] = (void*)&fin;
            if (!nr_launch_grid_(NR_SOFT_UPDATE, blocks, c, (&void)&p[0])) return 0;
        }
        base = base + c;
    }
    return 1;
}

static int nr_noise_(struct RocmBuf out, unsigned long n, unsigned long seed, struct RocmBuf ctr, unsigned int salt,
                     int normal) {
    if (!nr_fits_(out, n) || !nr_fits_(ctr, 1UL)) return 0;
    unsafe {
        void* po = (void*)out.dev; void* pc = (void*)ctr.dev;
        unsigned int nn = (unsigned int)n; unsigned long sd = seed; unsigned int sa = salt; int nm = normal;
        void* p[6];
        p[0] = (void*)&po; p[1] = (void*)&nn; p[2] = (void*)&sd; p[3] = (void*)&pc; p[4] = (void*)&sa; p[5] = (void*)&nm;
        return nr_launch_(NR_NOISE, n, (&void)&p[0]);
    }
}

static int nr_fill_(struct RocmBuf out, unsigned long n, float v) {
    if (!nr_fits_(out, n)) return 0;
    unsafe {
        void* po = (void*)out.dev; unsigned int nn = (unsigned int)n; float vv = v;
        void* p[3];
        p[0] = (void*)&po; p[1] = (void*)&nn; p[2] = (void*)&vv;
        return nr_launch_(NR_FILL, n, (&void)&p[0]);
    }
}

// append_labels with row r's label lab[r % labRows]
static int nr_append_(struct RocmBuf x, struct RocmBuf labels, struct RocmBuf out, unsigned long rows,
                      unsigned long channels, unsigned long plane, unsigned long classes, unsigned long labRows) {
    unsigned long n = rows * (channels + classes) * plane;
    if (n == 0UL || !nr_fits_(x, rows * channels * plane) || !nr_fits_(labels, labRows) || !nr_fits_(out, n)) return 0;
    unsafe {
        void* px = (void*)x.dev; void* pl = (void*)labels.dev; void* po = (void*)out.dev;
        unsigned int a0 = (unsigned int)rows; unsigned int a1 = (unsigned int)(channels * plane);
        unsigned int a2 = (unsigned int)plane; unsigned int a3 = (unsigned int)classes; unsigned int a4 = (unsigned int)labRows;
        void* p[8];
        p[0] = (void*)&px; p[1] = (void*)&pl; p[2] = (void*)&po; p[3] = (void*)&a0; p[4] = (void*)&a1;
        p[5] = (void*)&a2; p[6] = (void*)&a3; p[7] = (void*)&a4;
        return nr_launch_(NR_APPEND, n, (&void)&p[0]);
    }
}

struct RocmOptimizer rocm_optimizer_from(const &Optimizer o) {
    struct RocmOptimizer d = rocm_optimizer_new(o.kind, o.lr);
    d.momentum = o.momentum;
    d.beta1 = o.beta1;
    d.beta2 = o.beta2;
    d.eps = o.eps;
    d.weightDecay = o.weightDecay;
    d.sync();
    return d;
}

static unsigned long nr_gan_critics_(const &RocmGan g) { return g.cfg.nCritic == 0UL ? 1UL : g.cfg.nCritic; }

// Offsets in g.noise: critic step c's latent batch, the generator's, critic step c's weights.
static unsigned long nr_gan_z_(const &RocmGan g, unsigned long c) { return c * g.maxBatch * g.cfg.latent; }
static unsigned long nr_gan_t_(const &RocmGan g, unsigned long c) {
    return (nr_gan_critics_(g) + 1UL) * g.maxBatch * g.cfg.latent + c * g.maxBatch;
}

// Shared by both constructors once gen / disc are built: optimizers,
// averaged-generator buffers (uploaded from emaParams / emaStats), scratch.
static void nr_gan_finish_(&RocmGan d, const &Optimizer optG, const &Optimizer optD, const &Vec<&Tensor> emaParams,
                           const &Vec<&Tensor> emaStats) {
    d.ema = vec_new();
    d.emaSrc = vec_new();
    d.ok = d.ok && d.gen.ok && d.disc.ok;
    d.optG = rocm_optimizer_from(optG);
    d.optD = rocm_optimizer_from(optD);
    if (d.cfg.loss == GAN_WGAN) d.optD.clampWeights = d.cfg.clip;
    d.ok = d.ok && d.gen.add_params(&d.optG) && d.disc.add_params(&d.optD);
    // averaged generator: (w, b) of every parameterized layer, then (mean, var) of every batch norm
    unsigned long k = 0UL;
    unsigned long s = 0UL;
    int pass = 0;
    while (d.ok && pass < 2) {
        unsigned long i = 0UL;
        while (d.ok && i < d.gen.layers.length()) {
            &RocmLayer L = d.gen.layers.at(i);
            int has = pass == 0 ? L.w.dev != 0UL : L.kind == ROCM_LAYER_BATCHNORM;
            if (has) {
                struct RocmBuf a = nr_none_();
                struct RocmBuf b = nr_none_();
                if (pass == 0) {
                    d.ok = k + 2UL <= emaParams.length() && nr_upload_(&a, *emaParams.at(k), L.w.len) &&
                           nr_upload_(&b, *emaParams.at(k + 1UL), L.b.len);
                    k = k + 2UL;
                    d.emaSrc.push(L.w); d.emaSrc.push(L.b);
                } else {
                    d.ok = s + 2UL <= emaStats.length() && nr_upload_(&a, *emaStats.at(s), L.runMean.len) &&
                           nr_upload_(&b, *emaStats.at(s + 1UL), L.runVar.len);
                    s = s + 2UL;
                    d.emaSrc.push(L.runMean); d.emaSrc.push(L.runVar);
                }
                d.ema.push(a); d.ema.push(b);
            }
            i = i + 1UL;
        }
        pass = pass + 1;
    }
    unsigned long m = d.maxBatch;
    unsigned long nc = nr_gan_critics_(d);
    unsigned long c = d.cfg.classes;
    unsigned long rowsD = d.cfg.loss == GAN_WGAN_GP ? 4UL * m : 2UL * m;
    unsigned long dCols = (d.channels + c) * d.plane;
    d.ok = d.ok && nr_alloc_(&d.noise, (nc + 1UL) * m * d.cfg.latent + nc * m) &&
           nr_alloc_(&d.dData, rowsD * d.dataDim) && nr_alloc_(&d.dGrad, rowsD * dCols) &&
           nr_alloc_(&d.state, 8UL) && rocm_zero(d.state);
    if (d.ok && c > 0UL) d.ok = nr_alloc_(&d.gIn, m * (d.cfg.latent + c)) && nr_alloc_(&d.dIn, rowsD * dCols);
    if (d.ok && d.cfg.loss == GAN_WGAN_GP)
        d.ok = nr_alloc_(&d.xh, m * d.dataDim) && nr_alloc_(&d.gData, m * d.dataDim) && nr_alloc_(&d.coef, m) &&
               (c == 0UL || nr_alloc_(&d.xhIn, m * dCols));
    d.ok = d.ok && rocm_sync();
}

static struct RocmGan nr_gan_blank_(const &GanConfig cfg, unsigned long maxBatch) {
    struct RocmGan d;
    d.cfg = *cfg;
    d.maxBatch = maxBatch == 0UL ? 1UL : maxBatch;
    d.noise = nr_none_(); d.gIn = nr_none_(); d.dData = nr_none_(); d.dIn = nr_none_(); d.dGrad = nr_none_();
    d.xh = nr_none_(); d.xhIn = nr_none_(); d.gData = nr_none_(); d.coef = nr_none_(); d.state = nr_none_();
    d.ok = 1;
    return d;
}

struct RocmGan rocm_gan_new(const &Gan g, unsigned long maxBatch) {
    struct RocmGan d = nr_gan_blank_(&g.cfg, maxBatch);
    unsigned long rowsD = g.cfg.loss == GAN_WGAN_GP ? 4UL * d.maxBatch : 2UL * d.maxBatch;
    d.gen = rocm_net_from_mlp(&g.gen, d.maxBatch);
    d.disc = rocm_net_from_mlp(&g.disc, rowsD);
    d.dataDim = g.gen.outDim;
    d.channels = d.dataDim;
    d.plane = 1UL;
    d.steps = g.steps;
    nr_gan_finish_(&d, &g.optG, &g.optD, &g.genEma.params, &g.genEma.params);   // an MLP has no batch norm
    return d;
}

struct RocmGan rocm_dcgan_new(const &Dcgan g, unsigned long maxBatch) {
    struct RocmGan d = nr_gan_blank_(&g.cfg, maxBatch);
    unsigned long rowsD = g.cfg.loss == GAN_WGAN_GP ? 4UL * d.maxBatch : 2UL * d.maxBatch;
    d.gen = rocm_net_from_convnet(&g.gen, d.maxBatch);
    d.disc = rocm_net_from_convnet(&g.disc, rowsD);
    // training steps run both nets in training mode, as Dcgan::train_step
    d.gen.training = 1;
    d.disc.training = 1;
    d.dataDim = g.gen.out_dim();
    d.channels = g.disc.inC - g.cfg.classes;
    d.plane = g.disc.inH * g.disc.inW;
    d.steps = g.steps;
    nr_gan_finish_(&d, &g.optG, &g.optD, &g.genEma.params, &g.genEma.buffers);
    return d;
}

// G's or D's input: x itself, or x with the label planes appended into 'buf'.
static struct RocmBuf nr_gan_cond_(&RocmGan g, struct RocmBuf x, struct RocmBuf labels, unsigned long rows,
                                   unsigned long channels, unsigned long plane, unsigned long labRows,
                                   struct RocmBuf buf) {
    if (g.cfg.classes == 0UL) return x;
    if (!nr_append_(x, labels, buf, rows, channels, plane, g.cfg.classes, labRows)) return nr_none_();
    return buf;
}

static int nr_gan_loss_(&RocmGan g, struct RocmBuf s, struct RocmBuf d, unsigned long m, int gen) {
    unsafe {
        void* ps = (void*)s.dev; void* pd = (void*)d.dev; void* pst = (void*)g.state.dev; void* pc = (void*)g.coef.dev;
        unsigned int mm = (unsigned int)m; int k = g.cfg.loss; int ge = gen;
        void* p[7];
        p[0] = (void*)&ps; p[1] = (void*)&pd; p[2] = (void*)&pst; p[3] = (void*)&pc; p[4] = (void*)&mm;
        p[5] = (void*)&k; p[6] = (void*)&ge;
        return nr_launch_grid_(NR_GAN_LOSS, 1UL, 1UL, (&void)&p[0]);
    }
}

#define NR_GAN_GP_EPS 0.01

int rocm_gan_step_(&RocmGan g, struct RocmBuf real, struct RocmBuf labels, unsigned long rows, int drawNoise) {
    unsigned long m = rows;
    unsigned long D = g.dataDim;
    unsigned long Z = g.cfg.latent;
    unsigned long c = g.cfg.classes;
    int gp = g.cfg.loss == GAN_WGAN_GP;
    if (!g.ok || m == 0UL || m > g.maxBatch || !nr_fits_(real, m * D) || (c > 0UL && !nr_fits_(labels, m))) return 0;
    unsigned long dCols = (g.channels + c) * g.plane;
    unsigned long nc = nr_gan_critics_(g);
    unsigned long ci = 0UL;
    while (ci < nc) {
        struct RocmBuf z = nr_at_(g.noise, nr_gan_z_(g, ci), m * Z);
        struct RocmBuf t = nr_at_(g.noise, nr_gan_t_(g, ci), m);
        if (drawNoise && !nr_noise_(z, m * Z, g.cfg.seed, g.state, (unsigned int)ci, 1)) return 0;
        if (drawNoise && gp && !nr_noise_(t, m, g.cfg.seed, g.state, 100U + (unsigned int)ci, 0)) return 0;
        struct RocmBuf fake = g.gen.forward(nr_gan_cond_(g, z, labels, m, Z, 1UL, m, g.gIn), m);
        if (fake.dev == 0UL) return 0;
        if (!rocm_copy(real, 0UL, g.dData, 0UL, m * D) || !rocm_copy(fake, 0UL, g.dData, m * D, m * D)) return 0;
        unsigned long rowsD = 2UL * m;
        g.disc.segments = 2UL;
        if (gp) {
            // input gradient of sum D at the interpolates, then xp / xm and the coefficients
            unsafe {
                void* pr = (void*)real.dev; void* pf = (void*)fake.dev; void* pt = (void*)t.dev; void* px = (void*)g.xh.dev;
                unsigned int mm = (unsigned int)m; unsigned int dd = (unsigned int)D; void* pst = (void*)g.state.dev;
                void* p[7];
                p[0] = (void*)&pr; p[1] = (void*)&pf; p[2] = (void*)&pt; p[3] = (void*)&px; p[4] = (void*)&mm;
                p[5] = (void*)&dd; p[6] = (void*)&pst;
                if (!nr_launch_(NR_LERP, m * D, (&void)&p[0])) return 0;
            }
            g.disc.segments = 1UL;
            struct RocmBuf xin = nr_gan_cond_(g, g.xh, labels, m, g.channels, g.plane, m, g.xhIn);
            if (g.disc.forward(xin, m).dev == 0UL) return 0;
            if (!nr_fill_(g.disc.grad_output(), m, (float)1.0)) return 0;
            if (c > 0UL) {
                if (!g.disc.backward(xin, m, g.dGrad) || !rocm_slice_cols(g.dGrad, g.gData, m, dCols, D, 0UL))
                    return 0;
            } else if (!g.disc.backward(xin, m, g.gData)) {
                return 0;
            }
            unsafe {
                void* pg = (void*)g.gData.dev; void* px = (void*)g.xh.dev; void* ps = (void*)g.dData.dev;
                void* pc = (void*)g.coef.dev; void* pst = (void*)g.state.dev;
                unsigned int mm = (unsigned int)m; unsigned int dd = (unsigned int)D;
                float w = g.cfg.gpWeight; float ep = (float)NR_GAN_GP_EPS;
                void* p[9];
                p[0] = (void*)&pg; p[1] = (void*)&px; p[2] = (void*)&ps; p[3] = (void*)&pc; p[4] = (void*)&pst;
                p[5] = (void*)&mm; p[6] = (void*)&dd; p[7] = (void*)&w; p[8] = (void*)&ep;
                if (!nr_launch_grid_(NR_GP_PREP, m, 1UL, (&void)&p[0])) return 0;
            }
            rowsD = 4UL * m;
        }
        struct RocmBuf din = nr_gan_cond_(g, g.dData, labels, rowsD, g.channels, g.plane, m, g.dIn);
        struct RocmBuf s = g.disc.forward(din, rowsD);
        if (s.dev == 0UL) return 0;
        if (!nr_gan_loss_(g, s, g.disc.grad_output(), m, 0)) return 0;
        if (!g.disc.backward(din, rowsD, nr_none_()) || !g.optD.step()) return 0;
        ci = ci + 1UL;
    }
    // generator step through D
    struct RocmBuf zg = nr_at_(g.noise, nr_gan_z_(g, nc), m * Z);
    if (drawNoise && !nr_noise_(zg, m * Z, g.cfg.seed, g.state, (unsigned int)nc, 1)) return 0;
    struct RocmBuf gx = nr_gan_cond_(g, zg, labels, m, Z, 1UL, m, g.gIn);
    struct RocmBuf fake = g.gen.forward(gx, m);
    if (fake.dev == 0UL) return 0;
    g.disc.segments = 1UL;
    struct RocmBuf din = nr_gan_cond_(g, fake, labels, m, g.channels, g.plane, m, g.dIn);
    struct RocmBuf s = g.disc.forward(din, m);
    if (s.dev == 0UL || !nr_gan_loss_(g, s, g.disc.grad_output(), m, 1)) return 0;
    if (c > 0UL) {
        if (!g.disc.backward(din, m, g.dGrad) ||
            !rocm_slice_cols(g.dGrad, g.gen.grad_output(), m, dCols, D, 0UL)) return 0;
    } else if (!g.disc.backward(din, m, g.gen.grad_output())) {
        return 0;
    }
    if (!g.gen.backward(gx, m, nr_none_()) || !g.optG.step()) return 0;
    // averaged generator (a copy when emaDecay is 0); advances the noise counter and step count
    if (!nr_soft_update_(&g.ema, &g.emaSrc, g.state, g.cfg.emaDecay > (float)0.0 ? g.cfg.emaDecay : (float)0.0,
                         g.cfg.emaDecay > (float)0.0 ? 1 : 0)) return 0;
    g.steps = g.steps + 1UL;
    return 1;
}

int RocmGan::step(struct RocmBuf real, struct RocmBuf labels, unsigned long rows) {
    return rocm_gan_step_(self, real, labels, rows, 1);
}

struct GanLosses RocmGan::losses() const {
    struct GanLosses l;
    l.d = (float)0.0;
    l.g = (float)0.0;
    float st[8];
    if (self.ok && rocm_read(self.state, &st[0], 8UL)) { l.d = st[2]; l.g = st[3]; }
    return l;
}

struct GanLosses RocmGan::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses last;
    last.d = (float)0.0;
    last.g = (float)0.0;
    struct RocmBatches bt = rocm_batches_new(data, batchSize, 1, self.cfg.seed + 101UL);
    struct RocmGraph graphs[2];
    graphs[0].exec = 0UL;
    graphs[1].exec = 0UL;
    int seen[2];
    seen[0] = 0; seen[1] = 0;
    int ok = bt.ok && self.ok && bt.batchSize <= self.maxBatch;
    unsigned long e = 0UL;
    while (ok && e < epochs) {
        ok = bt.epoch() && self.optG.sync() && self.optD.sync();
        unsigned long pos = 0UL;
        while (ok && pos < bt.n) {
            unsigned long rows = bt.n - pos < bt.batchSize ? bt.n - pos : bt.batchSize;
            int k = rows == bt.batchSize ? 0 : 1;
            unsigned long s0 = self.steps;
            unsigned long tG = self.optG.t;
            unsigned long tD = self.optD.t;
            if (graphs[k].exec != 0UL) {
                ok = rocm_graph_launch(graphs[k]);
            } else if (seen[k] && rocm_capture_begin()) {
                int rec = bt.gather(rows) && self.step(bt.xb, bt.yb, rows);
                graphs[k] = rocm_capture_end();
                ok = rec && graphs[k].exec != 0UL && rocm_graph_launch(graphs[k]);
            } else {
                ok = bt.gather(rows) && self.step(bt.xb, bt.yb, rows);
                seen[k] = 1;
            }
            // a replayed step advances these only on the GPU
            self.steps = s0 + 1UL;
            self.optG.t = tG + 1UL;
            self.optD.t = tD + nr_gan_critics_(self);
            pos = pos + rows;
        }
        e = e + 1UL;
    }
    if (ok) last = self.losses();
    graphs[0].free();
    graphs[1].free();
    bt.free();
    return last;
}

static int nr_download_list_(const &Vec<struct RocmBuf> bufs, unsigned long from, &Vec<&Tensor> ts) {
    unsigned long i = 0UL;
    while (i < ts.length()) {
        if (from + i >= bufs.length()) return 0;
        struct RocmBuf b = *bufs.at(from + i);
        &Tensor t = *ts.at(i);
        if (!nr_tensor_ok_(t, b.len)) return 0;
        unsafe { if (!rocm_read(b, (&float)(float*)t->data, b.len)) return 0; }
        i = i + 1UL;
    }
    return 1;
}

int RocmGan::download(&Gan g) const {
    if (!self.ok || !self.gen.download(&g.gen.params) || !self.disc.download(&g.disc.params) ||
        !nr_download_list_(&self.ema, 0UL, &g.genEma.params)) return 0;
    g.steps = self.steps;
    return 1;
}

int RocmGan::download_dcgan(&Dcgan g) const {
    if (!self.ok || !self.gen.download_convnet(&g.gen) || !self.disc.download_convnet(&g.disc) ||
        !nr_download_list_(&self.ema, 0UL, &g.genEma.params) ||
        !nr_download_list_(&self.ema, g.genEma.params.length(), &g.genEma.buffers)) return 0;
    g.steps = self.steps;
    return 1;
}

void RocmGan::free() {
    self.optG.free();
    self.optD.free();
    unsigned long i = 0UL;
    while (i < self.ema.length()) { self.ema.at(i).free(); i = i + 1UL; }
    self.ema.free();
    self.emaSrc.free();
    self.gen.free();
    self.disc.free();
    self.noise.free(); self.gIn.free(); self.dData.free(); self.dIn.free(); self.dGrad.free();
    self.xh.free(); self.xhIn.free(); self.gData.free(); self.coef.free(); self.state.free();
    self.ok = 0;
}

// ── Reinforcement learning ───────────────────────────────────────────────────

// rl.sc's rl_argmax_: the largest, ties broken uniformly with rng.
static unsigned long nr_argmax_(const &float v, unsigned long n, &Rng rng) {
    unsigned long best = 0UL;
    unsigned long ties = 1UL;
    unsafe {
        const float* p = (const float*)v;
        unsigned long i = 1UL;
        while (i < n) {
            if (p[i] > p[best]) { best = i; ties = 1UL; }
            else if (p[i] == p[best]) {
                ties = ties + 1UL;
                if (rng.below(ties) == 0UL) best = i;
            }
            i = i + 1UL;
        }
    }
    return best;
}

static void nr_param_list_(const &RocmNet n, &Vec<struct RocmBuf> out) {
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        &RocmLayer L = n.layers.at(i);
        if (L.w.dev != 0UL) { out.push(L.w); out.push(L.b); }
        i = i + 1UL;
    }
}

// One row through a net: obs (dim values) uploaded through 'stage' into
// obs1, the n outputs read back into stage.
static int nr_forward1_(&RocmNet net, struct RocmBuf obs1, const &Tensor obs, unsigned long dim, &heap float stage,
                        unsigned long n) {
    unsafe {
        if (!rocm_write(obs1, (const &float)(const float*)obs->data, dim)) return 0;
        struct RocmBuf out = net.forward(obs1, 1UL);
        return out.dev != 0UL && rocm_read(out, (&float)(float*)stage, n);
    }
}

static int nr_launch1_(int kernel, &void params) { return nr_launch_grid_(kernel, 1UL, 1UL, params); }

struct RocmDqn rocm_dqn_new(const &Dqn d) {
    struct RocmDqn r;
    r.cfg = d.cfg;
    r.rng = d.rng;
    r.obsDim = d.obsDim;
    r.actions = d.actions;
    r.steps = d.steps;
    r.updates = d.updates;
    r.size = d.buffer.size;
    r.pos = d.buffer.pos;
    r.graph.exec = 0UL;
    r.replay = nr_none_(); r.idx = nr_none_(); r.ob = nr_none_(); r.nx = nr_none_(); r.acts = nr_none_();
    r.rew = nr_none_(); r.term = nr_none_(); r.y = nr_none_(); r.state = nr_none_(); r.obs1 = nr_none_();
    r.qParams = vec_new();
    r.tParams = vec_new();
    unsigned long B = d.cfg.batchSize == 0UL ? 1UL : d.cfg.batchSize;
    unsigned long D = d.obsDim;
    unsigned long W = 2UL * D + 3UL;
    unsigned long slots = d.buffer.slots;
    unsigned long st = W > B ? W : B;
    if (d.actions > st) st = d.actions;
    unsafe { r.stage = (&heap float)(float*)malloc(sizeof(float) * (st > slots * W ? st : slots * W)); }
    r.q = rocm_net_from_mlp(&d.q, B);
    r.target = rocm_net_from_mlp(&d.target, B);
    r.opt = rocm_optimizer_from(&d.opt);
    r.opt.clipNorm = d.cfg.clipNorm;
    r.ok = r.q.ok && r.target.ok && r.q.add_params(&r.opt);
    nr_param_list_(&r.q, &r.qParams);
    nr_param_list_(&r.target, &r.tParams);
    r.ok = r.ok && nr_alloc_(&r.replay, slots * W) && nr_alloc_(&r.idx, B) && nr_alloc_(&r.ob, B * D) &&
           nr_alloc_(&r.nx, B * D) && nr_alloc_(&r.acts, B) && nr_alloc_(&r.rew, B) && nr_alloc_(&r.term, B) &&
           nr_alloc_(&r.y, B) && nr_alloc_(&r.state, 8UL) && rocm_zero(r.state) && nr_alloc_(&r.obs1, D);
    // the replay buffer's contents, as rows
    if (r.ok) {
        unsafe {
            float* s = (float*)r.stage;
            unsigned long i = 0UL;
            while (i < slots) {
                unsigned long c = 0UL;
                while (c < D) {
                    s[i * W + c] = d.buffer.obs->data[i * D + c];
                    s[i * W + D + c] = d.buffer.next->data[i * D + c];
                    c = c + 1UL;
                }
                s[i * W + 2UL * D] = d.buffer.action->data[i];
                s[i * W + 2UL * D + 1UL] = d.buffer.reward->data[i];
                s[i * W + 2UL * D + 2UL] = d.buffer.terminal->data[i];
                i = i + 1UL;
            }
            r.ok = rocm_write(r.replay, (const &float)(const float*)r.stage, slots * W) && rocm_sync();
        }
    }
    return r;
}

float RocmDqn::epsilon() const {
    if (self.steps >= self.cfg.epsDecaySteps) return self.cfg.epsEnd;
    float f = (float)self.steps / (float)self.cfg.epsDecaySteps;
    return self.cfg.epsStart + f * (self.cfg.epsEnd - self.cfg.epsStart);
}

unsigned long RocmDqn::act(const &Tensor obs, int explore) {
    if (explore && self.rng.uniform() < self.epsilon()) return self.rng.below(self.actions);
    if (!self.ok || !nr_forward1_(&self.q, self.obs1, obs, self.obsDim, self.stage, self.actions)) return 0UL;
    unsafe { return nr_argmax_((const &float)(const float*)self.stage, self.actions, &self.rng); }
}

// One update on the sampled slots in d.idx (dqn_train_).
static int nr_dqn_update_(&RocmDqn d) {
    unsigned long B = d.cfg.batchSize;
    unsigned long D = d.obsDim;
    unsigned long A = d.actions;
    unsafe {
        void* pr = (void*)d.replay.dev; void* pi = (void*)d.idx.dev; void* po = (void*)d.ob.dev; void* pn = (void*)d.nx.dev;
        void* pa = (void*)d.acts.dev; void* pw = (void*)d.rew.dev; void* pt = (void*)d.term.dev;
        unsigned int bb = (unsigned int)B; unsigned int dd = (unsigned int)D;
        void* p[9];
        p[0] = (void*)&pr; p[1] = (void*)&pi; p[2] = (void*)&po; p[3] = (void*)&pn; p[4] = (void*)&pa;
        p[5] = (void*)&pw; p[6] = (void*)&pt; p[7] = (void*)&bb; p[8] = (void*)&dd;
        if (!nr_launch_(NR_REPLAY_GATHER, B * (2UL * D + 3UL), (&void)&p[0])) return 0;
    }
    struct RocmBuf qt = d.target.forward(d.nx, B);
    if (qt.dev == 0UL) return 0;
    struct RocmBuf qo = qt;
    if (d.cfg.doubleDqn) {
        qo = d.q.forward(d.nx, B);
        if (qo.dev == 0UL) return 0;
    }
    unsafe {
        void* pq = (void*)qo.dev; void* pt = (void*)qt.dev; void* pw = (void*)d.rew.dev; void* pm = (void*)d.term.dev;
        void* py = (void*)d.y.dev;
        unsigned int bb = (unsigned int)B; unsigned int aa = (unsigned int)A; float g = d.cfg.gamma;
        void* p[8];
        p[0] = (void*)&pq; p[1] = (void*)&pt; p[2] = (void*)&pw; p[3] = (void*)&pm; p[4] = (void*)&py;
        p[5] = (void*)&bb; p[6] = (void*)&aa; p[7] = (void*)&g;
        if (!nr_launch_(NR_DQN_TARGET, B, (&void)&p[0])) return 0;
    }
    struct RocmBuf q = d.q.forward(d.ob, B);
    if (q.dev == 0UL) return 0;
    struct RocmBuf dq = d.q.grad_output();
    unsafe {
        void* pq = (void*)q.dev; void* pa = (void*)d.acts.dev; void* py = (void*)d.y.dev; void* pd = (void*)dq.dev;
        void* ps = (void*)d.state.dev;
        unsigned int bb = (unsigned int)B; unsigned int aa = (unsigned int)A; float dl = d.cfg.huberDelta;
        void* p[8];
        p[0] = (void*)&pq; p[1] = (void*)&pa; p[2] = (void*)&py; p[3] = (void*)&pd; p[4] = (void*)&ps;
        p[5] = (void*)&bb; p[6] = (void*)&aa; p[7] = (void*)&dl;
        if (!nr_launch1_(NR_DQN_LOSS, (&void)&p[0])) return 0;
    }
    if (!d.q.backward(d.ob, B, nr_none_()) || !d.opt.step()) return 0;
    // soft target update (tau as given: schedule 2)
    return d.cfg.targetEvery > 0UL || nr_soft_update_(&d.tParams, &d.qParams, d.state, d.cfg.tau, 2);
}

int RocmDqn::observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    if (!self.ok) return -1;
    unsigned long D = self.obsDim;
    unsigned long W = 2UL * D + 3UL;
    unsigned long slots = self.cfg.bufferSize;
    unsafe {
        float* s = (float*)self.stage;
        unsigned long c = 0UL;
        while (c < D) { s[c] = obs->data[c]; s[D + c] = next->data[c]; c = c + 1UL; }
        s[2UL * D] = (float)action;
        s[2UL * D + 1UL] = reward;
        s[2UL * D + 2UL] = terminal ? (float)1.0 : (float)0.0;
        if (!rocm_write(nr_at_(self.replay, self.pos * W, W), (const &float)(const float*)s, W)) return -1;
    }
    self.pos = (self.pos + 1UL) % slots;
    if (self.size < slots) self.size = self.size + 1UL;
    self.steps = self.steps + 1UL;
    if (self.size < self.cfg.learningStarts || self.size < self.cfg.batchSize) return 0;
    if (self.steps % self.cfg.trainEvery != 0UL) return 0;
    // sample on the host, from the agent's own stream (as dqn_train_)
    unsigned long B = self.cfg.batchSize;
    unsafe {
        float* s = (float*)self.stage;
        unsigned long r = 0UL;
        while (r < B) { s[r] = (float)self.rng.below(self.size); r = r + 1UL; }
        if (!rocm_write(self.idx, (const &float)(const float*)s, B)) return -1;
    }
    unsigned long t0 = self.opt.t;
    int ok = 1;
    if (self.graph.exec != 0UL) {
        ok = rocm_graph_launch(self.graph);
    } else if (self.updates > 0UL && self.opt.sync() && rocm_capture_begin()) {
        int rec = nr_dqn_update_(self);
        self.graph = rocm_capture_end();
        ok = rec && self.graph.exec != 0UL && rocm_graph_launch(self.graph);
    } else {
        ok = nr_dqn_update_(self);
    }
    self.opt.t = t0 + 1UL;
    self.updates = self.updates + 1UL;
    if (ok && self.cfg.targetEvery > 0UL && self.updates % self.cfg.targetEvery == 0UL)
        ok = nr_soft_update_(&self.tParams, &self.qParams, self.state, (float)1.0, 2);   // hard copy
    return ok ? 1 : -1;
}

float RocmDqn::loss() const {
    float st[8];
    if (!self.ok || !rocm_read(self.state, &st[0], 8UL)) return (float)-1.0;
    return st[2];
}

int RocmDqn::download(&Dqn h) const {
    if (!self.ok || !self.q.download(&h.q.params) || !self.target.download(&h.target.params)) return 0;
    h.rng = self.rng;
    h.steps = self.steps;
    h.updates = self.updates;
    return 1;
}

void RocmDqn::free() {
    self.graph.free();
    self.opt.free();
    self.q.free();
    self.target.free();
    self.qParams.free();
    self.tParams.free();
    self.replay.free(); self.idx.free(); self.ob.free(); self.nx.free(); self.acts.free();
    self.rew.free(); self.term.free(); self.y.free(); self.state.free(); self.obs1.free();
    unsafe { free((void*)self.stage); }
    self.ok = 0;
}

// ── Policy gradient ──

struct RocmPolicy rocm_pg_new(const &PolicyAgent a, unsigned long maxRollout) {
    struct RocmPolicy r;
    r.cfg = a.cfg;
    r.rng = a.rng;
    r.obsDim = a.obsDim;
    r.actions = a.actions;
    r.maxRollout = maxRollout == 0UL ? 1UL : maxRollout;
    r.roll = vec_new();
    r.rollout = 0UL;
    r.lastEntropy = a.lastEntropy;
    r.graphs[0].exec = 0UL; r.graphs[1].exec = 0UL;
    r.graphRows[0] = 0UL; r.graphRows[1] = 0UL;
    r.data = nr_none_(); r.v = nr_none_(); r.vn = nr_none_(); r.logpOld = nr_none_(); r.adv = nr_none_();
    r.ret = nr_none_(); r.order = nr_none_(); r.cursor = nr_none_(); r.mObs = nr_none_(); r.mAct = nr_none_();
    r.mAdv = nr_none_(); r.mRet = nr_none_(); r.mLpo = nr_none_(); r.state = nr_none_(); r.obs1 = nr_none_();
    unsigned long N = r.maxRollout;
    unsigned long D = a.obsDim;
    unsigned long W = 2UL * D + 4UL;
    unsafe { r.stage = (&heap float)(float*)malloc(sizeof(float) * (N * W + a.actions + 1UL)); }
    r.policy = rocm_net_from_mlp(&a.policy, N);
    r.value = rocm_net_from_mlp(&a.value, N);
    r.opt = rocm_optimizer_from(&a.opt);
    r.opt.clipNorm = a.cfg.clipNorm;
    r.ok = r.policy.ok && r.value.ok && r.policy.add_params(&r.opt) &&
           r.value.add_params(&r.opt) &&
           nr_alloc_(&r.data, N * W) && nr_alloc_(&r.v, N) && nr_alloc_(&r.vn, N) && nr_alloc_(&r.logpOld, N) &&
           nr_alloc_(&r.adv, N) && nr_alloc_(&r.ret, N) && nr_alloc_(&r.order, N) && nr_alloc_(&r.cursor, 2UL) &&
           nr_alloc_(&r.mObs, N * D) && nr_alloc_(&r.mAct, N) && nr_alloc_(&r.mAdv, N) && nr_alloc_(&r.mRet, N) &&
           nr_alloc_(&r.mLpo, N) && nr_alloc_(&r.state, 4UL) && nr_alloc_(&r.obs1, D) && rocm_sync();
    return r;
}

unsigned long RocmPolicy::act(const &Tensor obs, int sample) {
    if (!self.ok || !nr_forward1_(&self.policy, self.obs1, obs, self.obsDim, self.stage, self.actions)) return 0UL;
    unsigned long A = self.actions;
    unsigned long choice = 0UL;
    unsafe {
        const float* l = (const float*)self.stage;
        if (!sample) return nr_argmax_((const &float)l, A, &self.rng);
        float mx = l[0];
        unsigned long i = 1UL;
        while (i < A) { if (l[i] > mx) mx = l[i]; i = i + 1UL; }
        float s = (float)0.0;
        i = 0UL;
        while (i < A) { s = s + exp_f(l[i] - mx); i = i + 1UL; }
        float u = self.rng.uniform() * s;
        float acc = (float)0.0;
        choice = A - 1UL;
        i = 0UL;
        while (i < A) { acc = acc + exp_f(l[i] - mx); if (u < acc) { choice = i; break; } i = i + 1UL; }
    }
    return choice;
}

void RocmPolicy::record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                        int truncated) {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.obsDim) { self.roll.push(obs->data[i]); i = i + 1UL; }
        i = 0UL;
        while (i < self.obsDim) { self.roll.push(next->data[i]); i = i + 1UL; }
    }
    self.roll.push((float)action);
    self.roll.push(reward);
    self.roll.push(terminated ? (float)1.0 : (float)0.0);
    self.roll.push(terminated || truncated ? (float)1.0 : (float)0.0);
    self.rollout = self.rollout + 1UL;
}

// One gradient step on the next m rows of the minibatch order (pg_step_).
static int nr_pg_step_(&RocmPolicy a, unsigned long m) {
    unsigned long N = a.rollout;
    unsigned long D = a.obsDim;
    unsafe {
        void* pi = (void*)a.order.dev; void* pc = (void*)a.cursor.dev; void* po = (void*)a.data.dev;
        void* pa = (void*)(a.data.dev + 2UL * N * D * sizeof(float));
        void* pv = (void*)a.adv.dev; void* pr = (void*)a.ret.dev; void* pl = (void*)a.logpOld.dev;
        void* qo = (void*)a.mObs.dev; void* qa = (void*)a.mAct.dev; void* qv = (void*)a.mAdv.dev;
        void* qr = (void*)a.mRet.dev; void* ql = (void*)a.mLpo.dev;
        unsigned int mm = (unsigned int)m; unsigned int dd = (unsigned int)D;
        void* p[14];
        p[0] = (void*)&pi; p[1] = (void*)&pc; p[2] = (void*)&po; p[3] = (void*)&pa; p[4] = (void*)&pv;
        p[5] = (void*)&pr; p[6] = (void*)&pl; p[7] = (void*)&qo; p[8] = (void*)&qa; p[9] = (void*)&qv;
        p[10] = (void*)&qr; p[11] = (void*)&ql; p[12] = (void*)&mm; p[13] = (void*)&dd;
        unsigned long blocks = (m * (D + 4UL) + (unsigned long)NR_BLOCK - 1UL) / (unsigned long)NR_BLOCK;
        if (blocks > 64UL) blocks = 64UL;
        if (!nr_launch_grid_(NR_PG_GATHER, blocks, 1UL, (&void)&p[0])) return 0;
    }
    struct RocmBuf lg = a.policy.forward(a.mObs, m);
    struct RocmBuf vv = a.value.forward(a.mObs, m);
    if (lg.dev == 0UL || vv.dev == 0UL) return 0;
    struct RocmBuf dl = a.policy.grad_output();
    struct RocmBuf dv = a.value.grad_output();
    unsafe {
        void* pl = (void*)lg.dev; void* pv = (void*)vv.dev; void* pa = (void*)a.mAct.dev; void* pd = (void*)a.mAdv.dev;
        void* pr = (void*)a.mRet.dev; void* po = (void*)a.mLpo.dev; void* gl = (void*)dl.dev; void* gv = (void*)dv.dev;
        void* ps = (void*)a.state.dev;
        unsigned int mm = (unsigned int)m; unsigned int aa = (unsigned int)a.actions;
        int al = a.cfg.algo; float ce = a.cfg.clipEps; float vc = a.cfg.valueCoef; float ec = a.cfg.entropyCoef;
        void* p[15];
        p[0] = (void*)&pl; p[1] = (void*)&pv; p[2] = (void*)&pa; p[3] = (void*)&pd; p[4] = (void*)&pr;
        p[5] = (void*)&po; p[6] = (void*)&gl; p[7] = (void*)&gv; p[8] = (void*)&ps; p[9] = (void*)&mm;
        p[10] = (void*)&aa; p[11] = (void*)&al; p[12] = (void*)&ce; p[13] = (void*)&vc; p[14] = (void*)&ec;
        if (!nr_launch1_(NR_PG_LOSS, (&void)&p[0])) return 0;
    }
    return a.policy.backward(a.mObs, m, nr_none_()) &&
           a.value.backward(a.mObs, m, nr_none_()) && a.opt.step();
}

// Runs nr_pg_step_ for m rows: eagerly the first time a size is seen, then
// recorded once per size and replayed.
static int nr_pg_run_(&RocmPolicy a, unsigned long m) {
    int k = m == a.graphRows[0] || a.graphRows[0] == 0UL ? 0 : 1;
    unsigned long t0 = a.opt.t;
    int ok = 1;
    if (a.graphRows[k] == m && a.graphs[k].exec != 0UL) {
        ok = rocm_graph_launch(a.graphs[k]);
    } else if (a.graphRows[k] == m && rocm_capture_begin()) {
        int rec = nr_pg_step_(a, m);
        a.graphs[k] = rocm_capture_end();
        ok = rec && a.graphs[k].exec != 0UL && rocm_graph_launch(a.graphs[k]);
    } else {
        // a new size: run it, and record it next time (a second short size replaces the first)
        if (a.graphs[k].exec != 0UL) a.graphs[k].free();
        a.graphRows[k] = m;
        ok = nr_pg_step_(a, m);
    }
    a.opt.t = t0 + 1UL;
    return ok;
}

// The epoch's row order to the GPU, the cursor rewound.
static int nr_pg_order_(&RocmPolicy a, const &unsigned long idx, unsigned long N) {
    unsafe {
        float* s = (float*)a.stage;
        const unsigned long* ix = (const unsigned long*)idx;
        unsigned long i = 0UL;
        while (i < N) { s[i] = (float)ix[i]; i = i + 1UL; }
        return rocm_write(a.order, (const &float)(const float*)s, N) && rocm_sync() && rocm_zero(a.cursor);
    }
}

float RocmPolicy::update() {
    unsigned long N = self.rollout;
    if (N == 0UL) return (float)0.0;
    if (!self.ok || N > self.maxRollout) return (float)-1.0;
    unsigned long D = self.obsDim;
    unsigned long W = 2UL * D + 4UL;
    int ok = 1;
    // the rollout as SoA, one upload
    unsafe {
        float* s = (float*)self.stage;
        const float* rr = (const float*)self.roll.at(0UL);
        unsigned long t = 0UL;
        while (t < N) {
            const float* row = &rr[t * W];
            unsigned long c = 0UL;
            while (c < D) { s[t * D + c] = row[c]; s[N * D + t * D + c] = row[D + c]; c = c + 1UL; }
            s[2UL * N * D + t] = row[2UL * D];
            s[2UL * N * D + N + t] = row[2UL * D + 1UL];
            s[2UL * N * D + 2UL * N + t] = row[2UL * D + 2UL];
            s[2UL * N * D + 3UL * N + t] = row[2UL * D + 3UL];
            t = t + 1UL;
        }
        ok = rocm_write(self.data, (const &float)(const float*)s, N * W) && rocm_sync();
    }
    struct RocmBuf obs = nr_at_(self.data, 0UL, N * D);
    struct RocmBuf nxt = nr_at_(self.data, N * D, N * D);
    struct RocmBuf act = nr_at_(self.data, 2UL * N * D, N);
    struct RocmBuf rew = nr_at_(self.data, 2UL * N * D + N, N);
    struct RocmBuf term = nr_at_(self.data, 2UL * N * D + 2UL * N, N);
    struct RocmBuf done = nr_at_(self.data, 2UL * N * D + 3UL * N, N);
    // values, next values, old log-probabilities
    struct RocmBuf vt = ok ? self.value.forward(obs, N) : nr_none_();
    ok = vt.dev != 0UL && rocm_copy(vt, 0UL, self.v, 0UL, N);
    struct RocmBuf vn = ok ? self.value.forward(nxt, N) : nr_none_();
    ok = ok && vn.dev != 0UL && rocm_copy(vn, 0UL, self.vn, 0UL, N);
    struct RocmBuf lg = ok ? self.policy.forward(obs, N) : nr_none_();
    ok = ok && lg.dev != 0UL;
    unsafe {
        if (ok) {
            void* pl = (void*)lg.dev; void* pa = (void*)act.dev; void* po = (void*)self.logpOld.dev;
            unsigned int nn = (unsigned int)N; unsigned int aa = (unsigned int)self.actions;
            void* p[5];
            p[0] = (void*)&pl; p[1] = (void*)&pa; p[2] = (void*)&po; p[3] = (void*)&nn; p[4] = (void*)&aa;
            ok = nr_launch_(NR_PG_LOGP, N, (&void)&p[0]);
        }
        if (ok) {
            void* pv = (void*)self.v.dev; void* pn = (void*)self.vn.dev; void* pr = (void*)rew.dev; void* pt = (void*)term.dev;
            void* pd = (void*)done.dev; void* pa = (void*)self.adv.dev; void* pe = (void*)self.ret.dev;
            unsigned int nn = (unsigned int)N; float g = self.cfg.gamma; float lam = self.cfg.gaeLambda;
            int al = self.cfg.algo == PG_REINFORCE ? 0 : 1;
            void* p[11];
            p[0] = (void*)&pv; p[1] = (void*)&pn; p[2] = (void*)&pr; p[3] = (void*)&pt; p[4] = (void*)&pd;
            p[5] = (void*)&pa; p[6] = (void*)&pe; p[7] = (void*)&nn; p[8] = (void*)&g; p[9] = (void*)&lam;
            p[10] = (void*)&al;
            ok = nr_launch1_(NR_PG_GAE, (&void)&p[0]);
        }
        if (ok && self.cfg.normalizeAdv && N > 1UL) {
            void* pa = (void*)self.adv.dev; unsigned int nn = (unsigned int)N;
            void* p[2];
            p[0] = (void*)&pa; p[1] = (void*)&nn;
            ok = nr_launch1_(NR_PG_NORM, (&void)&p[0]);
        }
    }
    unsigned long* idx;
    unsafe {
        idx = (unsigned long*)malloc(sizeof(unsigned long) * N);
        unsigned long i = 0UL;
        while (i < N) { idx[i] = i; i = i + 1UL; }
    }
    if (self.cfg.algo == PG_PPO) {
        unsigned long mb = self.cfg.minibatch == 0UL ? N : self.cfg.minibatch;
        unsigned long e = 0UL;
        while (ok && e < self.cfg.epochs) {
            unsafe {
                unsigned long i = N;
                while (i > 1UL) {
                    unsigned long j = self.rng.below(i);
                    unsigned long tmp = idx[i - 1UL]; idx[i - 1UL] = idx[j]; idx[j] = tmp;
                    i = i - 1UL;
                }
                ok = nr_pg_order_(self, (const &unsigned long)idx, N);
            }
            unsigned long start = 0UL;
            while (ok && start < N) {
                unsigned long m = N - start < mb ? N - start : mb;
                ok = nr_pg_run_(self, m);
                start = start + m;
            }
            e = e + 1UL;
        }
    } else {
        unsafe { ok = ok && nr_pg_order_(self, (const &unsigned long)idx, N) && nr_pg_step_(self, N); }
    }
    unsafe { free((void*)idx); }
    float st[4];
    float last = (float)-1.0;
    if (ok && rocm_read(self.state, &st[0], 4UL)) { last = st[2]; self.lastEntropy = st[3]; }
    self.roll.clear();
    self.rollout = 0UL;
    return last;
}

int RocmPolicy::download(&PolicyAgent a) const {
    if (!self.ok || !self.policy.download(&a.policy.params) || !self.value.download(&a.value.params))
        return 0;
    a.rng = self.rng;
    a.lastEntropy = self.lastEntropy;
    return 1;
}

void RocmPolicy::free() {
    self.graphs[0].free();
    self.graphs[1].free();
    self.opt.free();
    self.policy.free();
    self.value.free();
    self.roll.free();
    self.data.free(); self.v.free(); self.vn.free(); self.logpOld.free(); self.adv.free();
    self.ret.free(); self.order.free(); self.cursor.free(); self.mObs.free(); self.mAct.free();
    self.mAdv.free(); self.mRet.free(); self.mLpo.free(); self.state.free(); self.obs1.free();
    unsafe { free((void*)self.stage); }
    self.ok = 0;
}

// ── Batches ──────────────────────────────────────────────────────────────────

// DataLoader's shuffle (datasets.sc's dl_shuffle_), for the same row order.
static void nr_shuffle_(&RocmBatches b) {
    unsafe {
        unsigned long i = b.n;
        while (i > 1UL) {
            unsigned long j = b.rng.below(i);
            unsigned long t = b.perm[i - 1UL]; b.perm[i - 1UL] = b.perm[j]; b.perm[j] = t;
            i = i - 1UL;
        }
    }
}

struct RocmBatches rocm_batches_new(const &Dataset d, unsigned long batchSize, int shuffle, unsigned long seed) {
    struct RocmBatches b;
    b.n = d.n;
    b.batchSize = batchSize == 0UL ? 1UL : batchSize;
    if (b.batchSize > b.n && b.n > 0UL) b.batchSize = b.n;
    b.shuffle = shuffle;
    b.rng = rng_new(seed);
    unsafe {
        b.features = d.x->shape[1];
        b.targets = d.y->shape[1];
        unsigned long* perm = (unsigned long*)malloc(sizeof(unsigned long) * (d.n + 1UL));
        float* permf = (float*)malloc(sizeof(float) * (d.n + 1UL));
        unsigned long i = 0UL;
        while (i < d.n) { perm[i] = i; i = i + 1UL; }
        b.perm = (&heap unsigned long)perm;
        b.permf = (&heap float)permf;
    }
    if (shuffle) nr_shuffle_(&b);
    b.x = nr_none_(); b.y = nr_none_(); b.order = nr_none_(); b.cursor = nr_none_();
    b.xb = nr_none_(); b.yb = nr_none_(); b.loss = nr_none_();
    b.ok = nr_ensure_() && b.n > 0UL && b.n < 16777216UL &&
           nr_alloc_(&b.x, b.n * b.features) && nr_alloc_(&b.y, b.n * b.targets) &&
           nr_alloc_(&b.order, b.n) && nr_alloc_(&b.cursor, 2UL) && nr_alloc_(&b.loss, 1UL) &&
           nr_alloc_(&b.xb, b.batchSize * b.features) && nr_alloc_(&b.yb, b.batchSize * b.targets);
    unsafe {
        b.ok = b.ok && rocm_write(b.x, (const &float)(const float*)d.x->data, b.n * b.features) &&
               rocm_write(b.y, (const &float)(const float*)d.y->data, b.n * b.targets) && rocm_sync();
    }
    return b;
}

int RocmBatches::epoch() {
    if (!self.ok) return 0;
    // the previous epoch's reads of permf are done once its loss was read
    if (!rocm_sync()) return 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < self.n) { self.permf[i] = (float)self.perm[i]; i = i + 1UL; }
        if (!nr_write_(self.order, (const &float)(const float*)self.permf, self.n)) return 0;
    }
    // DataLoader reshuffles at the end of each epoch for the next
    if (self.shuffle) nr_shuffle_(self);
    return rocm_zero(self.cursor) && rocm_zero(self.loss);
}

int RocmBatches::gather(unsigned long rows) {
    if (!self.ok || rows == 0UL || rows > self.batchSize) return 0;
    unsafe {
        void* sx = (void*)self.x.dev; void* dx = (void*)self.xb.dev; void* sy = (void*)self.y.dev; void* dy = (void*)self.yb.dev;
        void* po = (void*)self.order.dev; void* pc = (void*)self.cursor.dev;
        unsigned int fx = (unsigned int)self.features; unsigned int fy = (unsigned int)self.targets;
        unsigned int r = (unsigned int)rows;
        void* p[9];
        p[0] = (void*)&sx; p[1] = (void*)&dx; p[2] = (void*)&fx; p[3] = (void*)&sy; p[4] = (void*)&dy;
        p[5] = (void*)&fy; p[6] = (void*)&po; p[7] = (void*)&pc; p[8] = (void*)&r;
        unsigned long n = rows * (self.features + self.targets);
        unsigned long blocks = (n + 4UL * (unsigned long)NR_BLOCK - 1UL) / (4UL * (unsigned long)NR_BLOCK);
        if (blocks > 128UL) blocks = 128UL;   // each block takes a turn on the cursor's counter
        return nr_launch_grid_(NR_GATHER, blocks, 1UL, (&void)&p[0]);
    }
}

float RocmBatches::loss_sum() {
    float v = (float)0.0;
    if (!self.ok || !rocm_read(self.loss, &v, 1UL)) return (float)0.0;
    return v;
}

void RocmBatches::free() {
    self.x.free(); self.y.free(); self.order.free(); self.cursor.free();
    self.xb.free(); self.yb.free(); self.loss.free();
    unsafe { free((void*)self.perm); free((void*)self.permf); }
    self.ok = 0;
}

// ── The training traits ──────────────────────────────────────────────────────

struct RocmFitGraphs rocm_fit_graphs_new_() {
    struct RocmFitGraphs g;
    g.full.exec = 0UL;
    g.tail.exec = 0UL;
    g.chunk.exec = 0UL;
    g.seenFull = 0;
    g.seenTail = 0;
    return g;
}

void rocm_fit_graphs_free_(&RocmFitGraphs g) {
    g.full.free();
    g.tail.free();
    g.chunk.free();
    g.seenFull = 0;
    g.seenTail = 0;
}

int rocm_trainer_reserve_(&RocmBuf b, unsigned long n) {
    if (b.dev != 0UL && b.len >= n) return 1;
    b.free();
    return nr_alloc_(b, n);
}

int RocmNet::download_mlp(&Mlp m) const { return self.download(&m.params); }

static fn int(&RocmNet, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) nr_loss_step_(int loss) {
    if (loss == TRAIN_LOSS_MSE) return rocm_net_mse_step;
    if (loss == TRAIN_LOSS_BCE) return rocm_net_bce_step;
    return rocm_net_ce_step;
}

// The fields every RocmTrainer constructor sets the same way, after 'dev'.
generic<H, D>
static void nr_trainer_finish_(&RocmTrainer<H, D> t, const &Optimizer opt, const &TrainerOptions o) {
    t.opt = rocm_optimizer_from(opt);
    t.maxBatch = o.maxBatch;
    t.xs = nr_none_();
    t.ys = nr_none_();
    t.loss1 = nr_none_();
    t.active = 0;
}

// RocmTrainer / RocmGanTrainer call these through their fn fields.
static int  nr_download_mlp_(const &RocmNet n, &Mlp m) { return n.download_mlp(m); }
static int  nr_download_convnet_(const &RocmNet n, &ConvNet c) { return n.download_convnet(c); }
static void nr_free_net_(&RocmNet n) { n.free(); }
static int  nr_download_vae_(const &RocmVae d, &Vae v) { return d.download(v); }
static int  nr_download_conv_vae_(const &RocmVae d, &ConvVae v) { return d.download_conv_vae(v); }
static void nr_free_vae_(&RocmVae v) { v.free(); }
static int  nr_download_cae_(const &RocmConvAe a, &ConvAutoencoder ae) { return a.download(ae); }
static void nr_free_cae_(&RocmConvAe a) { a.free(); }
static int  nr_download_gan_(const &RocmGan d, &Gan g) { return d.download(g); }
static int  nr_download_dcgan_(const &RocmGan d, &Dcgan g) { return d.download_dcgan(g); }

struct RocmTrainer<struct Mlp, struct RocmNet> rocm_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt,
                                                                    const &TrainerOptions o) {
    struct RocmTrainer<struct Mlp, struct RocmNet> t;
    t.host = m;
    t.dev = rocm_net_from_mlp(&t.host, o.maxBatch);
    nr_trainer_finish_(&t, opt, o);
    t.ok = t.dev.ok && (o.precision == 0 || t.dev.set_precision(o.precision)) &&
           t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = nr_loss_step_(loss);
    t.downloadFn = nr_download_mlp_;
    t.freeFn = nr_free_net_;
    return t;
}

struct RocmTrainer<struct ConvNet, struct RocmNet> rocm_convnet_trainer_new(struct ConvNet n, int loss,
                                                                            const &Optimizer opt,
                                                                            const &TrainerOptions o) {
    struct RocmTrainer<struct ConvNet, struct RocmNet> t;
    t.host = n;
    t.dev = rocm_net_from_convnet(&t.host, o.maxBatch);
    nr_trainer_finish_(&t, opt, o);
    t.ok = t.dev.ok && (o.precision == 0 || t.dev.set_precision(o.precision)) &&
           t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = nr_loss_step_(loss);
    t.downloadFn = nr_download_convnet_;
    t.freeFn = nr_free_net_;
    return t;
}

struct RocmTrainer<struct Vae, struct RocmVae> rocm_vae_trainer_new(struct Vae v, const &Optimizer opt,
                                                                    const &TrainerOptions o) {
    struct RocmTrainer<struct Vae, struct RocmVae> t;
    t.host = v;
    t.dev = rocm_vae_new(&t.host, o.maxBatch);
    nr_trainer_finish_(&t, opt, o);
    t.ok = t.dev.ok && o.precision == 0 && t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = rocm_vae_step;
    t.downloadFn = nr_download_vae_;
    t.freeFn = nr_free_vae_;
    return t;
}

struct RocmTrainer<struct ConvVae, struct RocmVae> rocm_conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt,
                                                                             const &TrainerOptions o) {
    struct RocmTrainer<struct ConvVae, struct RocmVae> t;
    t.host = v;
    t.dev = rocm_conv_vae_new(&t.host, o.maxBatch);
    nr_trainer_finish_(&t, opt, o);
    t.ok = t.dev.ok && o.precision == 0 && t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = rocm_vae_step;
    t.downloadFn = nr_download_conv_vae_;
    t.freeFn = nr_free_vae_;
    return t;
}

struct RocmTrainer<struct ConvAutoencoder, struct RocmConvAe> rocm_conv_ae_trainer_new(struct ConvAutoencoder ae,
                                                                                       const &Optimizer opt,
                                                                                       const &TrainerOptions o) {
    struct RocmTrainer<struct ConvAutoencoder, struct RocmConvAe> t;
    t.host = ae;
    t.dev = rocm_conv_autoencoder_new(&t.host, o.maxBatch);
    nr_trainer_finish_(&t, opt, o);
    t.ok = t.dev.ok && o.precision == 0 && t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = rocm_conv_autoencoder_step;
    t.downloadFn = nr_download_cae_;
    t.freeFn = nr_free_cae_;
    return t;
}

struct RocmGanTrainer<struct Gan> rocm_gan_trainer_new(struct Gan g, const &TrainerOptions o) {
    struct RocmGanTrainer<struct Gan> t;
    t.host = g;
    t.dev = rocm_gan_new(&t.host, o.maxBatch);
    t.downloadFn = nr_download_gan_;
    t.xs = nr_none_();
    t.ls = nr_none_();
    return t;
}

struct RocmGanTrainer<struct Dcgan> rocm_dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o) {
    struct RocmGanTrainer<struct Dcgan> t;
    t.host = g;
    t.dev = rocm_dcgan_new(&t.host, o.maxBatch);
    t.downloadFn = nr_download_dcgan_;
    t.xs = nr_none_();
    t.ls = nr_none_();
    return t;
}

struct RocmDqnAgent rocm_dqn_agent_new(struct Dqn d) {
    struct RocmDqnAgent a;
    a.host = d;
    a.dev = rocm_dqn_new(&a.host);
    return a;
}

float RocmDqnAgent::epsilon() { return self.dev.epsilon(); }

unsigned long RocmDqnAgent::act(const &Tensor obs, int explore) { return self.dev.act(obs, explore); }

int RocmDqnAgent::observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    return self.dev.observe(obs, action, reward, next, terminal);
}

float RocmDqnAgent::last_loss() { return self.dev.loss(); }

unsigned long RocmDqnAgent::updates() { return self.dev.updates; }

int RocmDqnAgent::sync() { return self.dev.download(&self.host); }

void RocmDqnAgent::free() { self.dev.free(); }

struct RocmPgAgent rocm_pg_agent_new(struct PolicyAgent a, const &TrainerOptions o) {
    struct RocmPgAgent p;
    p.host = a;
    p.dev = rocm_pg_new(&p.host, o.maxBatch);
    return p;
}

unsigned long RocmPgAgent::act(const &Tensor obs, int sample) { return self.dev.act(obs, sample); }

void RocmPgAgent::record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                         int truncated) {
    self.dev.record(obs, action, reward, next, terminated, truncated);
}

float RocmPgAgent::update() { return self.dev.update(); }

int RocmPgAgent::sync() { return self.dev.download(&self.host); }

void RocmPgAgent::free() { self.dev.free(); }

} // namespace std
