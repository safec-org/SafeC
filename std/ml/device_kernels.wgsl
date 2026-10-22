// SafeC Standard Library — the device kernels behind std::ml's tensor ops on
// WebGPU (device_webgpu.sc): device_kernels.cu's DK_* kernels in WGSL, one
// entry point per kernel under the same name (dk_ewop_<op> are dk_ew with
// the pipeline constant OPC = op), with the same arguments and semantics.
// device_kernels.cu is the specification and device_ref.sc its CPU
// reference; gen_device_kernels_wgsl.sh embeds this file in
// device_kernels_wgsl.h and fails unless every DK_KERNEL of
// device_kernels.cu has its entry point here. Edit both together.
//
// What differs from the CUDA source, and why:
//   - WGSL has no pointers and no 64-bit integers. All device memory is one
//     storage buffer (device_webgpu.sc's heap), bound as H (32-bit words)
//     and HA (the same words as atomics); a device address is a byte offset
//     into it. The backend passes DevArgs as 32 words: p0..p7 as byte
//     offsets (0 = none), their element types (DEV_TAG_*), u0..u7, f0..f7.
//     The tables of handles DK_OPT_MULTI / DK_AMP_UNSCALE_MULTI read hold
//     the same offsets in their 64-bit entries' low words.
//   - No float atomics: atomicAdd on a float is a compare-exchange loop on
//     its bits (atomic_addf).
//   - No doubles: dk_double accumulators are f32, as on Metal (the
//     statistics stay two-pass, which holds device_check's tolerance), and
//     dk_bn_partial's partials are floats in the buffer sized for doubles.
//   - No 16-bit stores: a 16-bit element is written by clearing and setting
//     its half of the word atomically (two threads may own the two halves).
//     The elementwise kernels with a 16-bit output instead take two
//     elements per thread and write whole words, as device_kernels.cu's
//     pair paths do. fp16 converts with pack2x16float / unpack2x16float
//     (core WGSL, no shader-f16 feature needed), bf16 with integer
//     arithmetic (the same rounding as f2b_bits).
//   - The kernels compile twice through the pipeline constant TAGGED, as
//     device_kernels.cu compiles with and without DK_TAGGED: the plain
//     pipelines read every buffer as fp32 and pay nothing for 16 bits.
//   - The fused attention kernels' tile (ATT_T x ATT_T threads) and largest
//     head dim (ATT_MAXD) are pipeline constants: 16 x 16 tiles for head
//     dims up to 64 and 8 x 8 up to 128, as Metal's two builds (32 KB of
//     workgroup memory doesn't hold four 16-row tiles of 128 floats).
//   - No subgroup operations (no adapter here reports them): reductions go
//     through workgroup memory.
// Besides the DK_* kernels, the wk_* entry points do what rocBLAS and MIOpen
// do on ROCm and device_kernels.cu's mk_* on Metal: GEMMs, convolutions as
// implicit GEMMs, copies, and gpu_webgpu.h's few ops of its own.

struct Args { w: array<vec4u, 8> }
@group(0) @binding(0) var<uniform> A: Args;
@group(0) @binding(1) var<storage, read_write> H: array<u32>;
@group(0) @binding(2) var<storage, read_write> HA: array<atomic<u32>>;

override TAGGED: bool = false;   // the 16-bit build (some argument is fp16 / bf16)
override OPC: i32 = -1;          // dk_ew's op: >= 0 for dk_ewop_<op>

// ── Arguments and element access ─────────────────────────────────────────────
fn aw(i: u32) -> u32 { return A.w[i >> 2u][i & 3u]; }
fn P(k: u32) -> u32 { return aw(k); }               // p<k>: byte offset, 0 = none
fn TG(k: u32) -> u32 { return aw(8u + k); }         // p<k>'s element type: 0 fp32, 1 fp16, 2 bf16
fn U(k: u32) -> u32 { return aw(16u + k); }
fn F(k: u32) -> f32 { return bitcast<f32>(aw(24u + k)); }
fn N() -> u32 { return aw(23u); }                   // u7: the thread count

fn h2f(h: u32, t: u32) -> f32 {
    if (t == 1u) { return unpack2x16float(h).x; }
    return bitcast<f32>(h << 16u);
}
fn f2h(v: f32, t: u32) -> u32 {
    if (t == 1u) { return pack2x16float(vec2f(v, 0.0)) & 0xffffu; }
    let x = bitcast<u32>(v);
    if ((x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffffu) != 0u) { return ((x >> 16u) | 0x40u) & 0xffffu; }
    return ((x + 0x7fffu + ((x >> 16u) & 1u)) >> 16u) & 0xffffu;
}
fn pack16(lo: f32, hi: f32, t: u32) -> u32 { return f2h(lo, t) | (f2h(hi, t) << 16u); }

// Element i of the buffer at byte offset b holding type t.
fn ldp(b: u32, t: u32, i: u32) -> f32 {
    if (!TAGGED || t == 0u) { return bitcast<f32>(H[(b >> 2u) + i]); }
    let a = b + 2u * i;
    return h2f((H[a >> 2u] >> ((a & 2u) * 8u)) & 0xffffu, t);
}
fn stp(b: u32, t: u32, i: u32, v: f32) {
    if (!TAGGED || t == 0u) { H[(b >> 2u) + i] = bitcast<u32>(v); return; }
    let a = b + 2u * i;
    let s = (a & 2u) * 8u;
    atomicAnd(&HA[a >> 2u], ~(0xffffu << s));
    atomicOr(&HA[a >> 2u], f2h(v, t) << s);
}
// F(k)[i] read / written in its type; ACCUM(F(k)[i], v)
fn ld(k: u32, i: u32) -> f32 { return ldp(P(k), TG(k), i); }
fn st(k: u32, i: u32, v: f32) { stp(P(k), TG(k), i, v); }
fn accum(k: u32, i: u32, v: f32) {
    if (U(6u) != 0u) { st(k, i, v); } else { st(k, i, ld(k, i) + v); }
}
// FA(k): fp32 buffers (gradients, sums, statistics)
fn ldf(k: u32, i: u32) -> f32 { return bitcast<f32>(H[(P(k) >> 2u) + i]); }
fn stf(k: u32, i: u32, v: f32) { H[(P(k) >> 2u) + i] = bitcast<u32>(v); }
fn rdw(w: u32) -> f32 { return bitcast<f32>(H[w]); }
fn wrw(w: u32, v: f32) { H[w] = bitcast<u32>(v); }
fn atomic_addf(w: u32, v: f32) {
    var old = atomicLoad(&HA[w]);
    loop {
        let r = atomicCompareExchangeWeak(&HA[w], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}
fn atadd(k: u32, i: u32, v: f32) { atomic_addf((P(k) >> 2u) + i, v); }
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
// Output k is 16-bit and word-aligned: elementwise kernels write it in pairs.
fn pairs_out(k: u32) -> bool { return TAGGED && TG(k) != 0u && (P(k) & 3u) == 0u; }

// ── Elementwise: op codes match tensor.h's EW_* ─────────────────────────────
// p0 = in a, p1 = in b, p2 = out, p3 = aux; f0 = k, f1 = k2; u1 = 1: the
// accumulating forms write (see device_kernels.cu).
const EW_ADD = 0;
const EW_SUB = 1;
const EW_MUL = 2;
const EW_SCALE = 3;
const EW_RELU = 4;
const EW_FILL = 5;
const EW_COPY = 6;
const EW_ACC = 7;
const EW_DEC = 8;
const EW_ACC_MUL = 9;
const EW_ACC_SCALE = 10;
const EW_ACC_RELU = 11;
const EW_ACC_K = 12;
const EW_ACC_SIGB = 13;
const EW_ACC_TANHB = 14;
const EW_EXP = 20;
const EW_LOG = 21;
const EW_ABS = 22;
const EW_LEAKY = 23;
const EW_CLAMP = 24;
const EW_SIGMOID = 25;
const EW_TANH = 26;
const EW_GELU = 27;
const EW_SILU = 28;
const EW_ADDK = 29;
const EW_MIN = 30;
const EW_ELU = 31;
const EW_LERP = 32;
const EW_MULS = 33;
const EW_ACC_PK = 40;
const EW_ACC_EXPB = 41;
const EW_ACC_LOGB = 42;
const EW_ACC_ABSB = 43;
const EW_ACC_LEAKYB = 44;
const EW_ACC_CLAMPB = 45;
const EW_ACC_GELUB = 46;
const EW_ACC_SILUB = 47;
const EW_ACC_MINA = 48;
const EW_ACC_MINB = 49;
const EW_ACC_ELUB = 50;
const EW_ACC_PSCALE = 51;

fn gelu_f(x: f32) -> f32 {
    let c = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}
fn gelu_d(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = tanh(c * (x + 0.044715 * x * x * x));
    let du = c * (1.0 + 0.134145 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
}
fn sig_f(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

fn ew_val(op: i32, x: f32, y: f32, z: f32, o: f32, k: f32, k2: f32, ow: bool) -> f32 {
    let ob = select(o, 0.0, ow);
    switch op {
        case EW_ADD: { return x + y; }
        case EW_SUB: { return x - y; }
        case EW_MUL: { return x * y; }
        case EW_SCALE: { return x * k; }
        case EW_RELU: { return select(0.0, x, x > 0.0); }
        case EW_FILL: { return k; }
        case EW_COPY: { return x; }
        case EW_ACC: { return ob + x; }
        case EW_DEC: { return ob - x; }
        case EW_ACC_MUL: { return fma(x, y, ob); }
        case EW_ACC_SCALE: { return fma(x, k, ob); }
        case EW_ACC_RELU: { return ob + select(0.0, x, y > 0.0); }
        case EW_ACC_K: { return ob + k; }
        case EW_ACC_SIGB: { return fma(x * y, 1.0 - y, ob); }
        case EW_ACC_TANHB: { return fma(x, 1.0 - y * y, ob); }
        case EW_EXP: { return exp(x); }
        case EW_LOG: { return log(x); }
        case EW_ABS: { return abs(x); }
        case EW_LEAKY: { return select(k * x, x, x > 0.0); }
        case EW_CLAMP: { return min(max(x, k), k2); }
        case EW_SIGMOID: { return sig_f(x); }
        case EW_TANH: { return tanh(x); }
        case EW_GELU: { return gelu_f(x); }
        case EW_SILU: { return x * sig_f(x); }
        case EW_ADDK: { return x + k; }
        case EW_MIN: { return select(y, x, x <= y); }
        case EW_ELU: { return select(k * (exp(x) - 1.0), x, x > 0.0); }
        case EW_LERP: { return fma(k, x - o, o); }
        case EW_MULS: { return x * y; }
        case EW_ACC_PK: { return ob + y; }
        case EW_ACC_EXPB: { return fma(x, y, ob); }
        case EW_ACC_LOGB: { return ob + x / y; }
        case EW_ACC_ABSB: { return ob + select(select(0.0, -x, y < 0.0), x, y > 0.0); }
        case EW_ACC_LEAKYB: { return ob + x * select(k, 1.0, y > 0.0); }
        case EW_ACC_CLAMPB: { return ob + select(0.0, x, y == z); }
        case EW_ACC_GELUB: { return fma(x, gelu_d(y), ob); }
        case EW_ACC_SILUB: { let s = sig_f(y); return fma(x, s + y * s * (1.0 - s), ob); }
        case EW_ACC_MINA: { return ob + select(0.0, x, y <= z); }
        case EW_ACC_MINB: { return ob + select(0.0, x, !(z <= y)); }
        case EW_ACC_ELUB: { return ob + x * select(z + k, 1.0, y > 0.0); }
        case EW_ACC_PSCALE: { return ob + k * y; }
        default: { }
    }
    return 0.0;
}
fn ew_uses_x(op: i32) -> bool { return op != EW_FILL && op != EW_ACC_K && op != EW_ACC_PK && op != EW_ACC_PSCALE; }
fn ew_y_scalar(op: i32) -> bool { return op == EW_MULS || op == EW_ACC_PK || op == EW_ACC_PSCALE; }
fn ew_uses_y(op: i32) -> bool {
    return op == EW_ADD || op == EW_SUB || op == EW_MUL || op == EW_ACC_MUL || op == EW_ACC_RELU || op == EW_ACC_SIGB ||
           op == EW_ACC_TANHB || op == EW_MIN || (op >= EW_ACC_EXPB && op <= EW_ACC_ELUB);
}
fn ew_uses_z(op: i32) -> bool { return op == EW_ACC_CLAMPB || op == EW_ACC_MINA || op == EW_ACC_MINB || op == EW_ACC_ELUB; }
fn ew_reads_o(op: i32, ow: bool) -> bool {
    return op == EW_LERP || (!ow && ((op >= EW_ACC && op <= EW_ACC_TANHB) || (op >= EW_ACC_PK && op <= EW_ACC_PSCALE)));
}
fn ew_op() -> i32 { return select(i32(U(0u)), OPC, OPC >= 0); }
// Element i's result.
fn ew_at(op: i32, i: u32, y0: f32) -> f32 {
    let ow = U(1u) != 0u;
    var x = 0.0;
    if (ew_uses_x(op)) { x = ld(0u, i); }
    var y = y0;
    if (ew_uses_y(op)) { y = ld(1u, i); }
    var z = 0.0;
    if (ew_uses_z(op)) { z = ld(3u, i); }
    var o = 0.0;
    if (ew_reads_o(op, ow)) { o = ld(2u, i); }
    return ew_val(op, x, y, z, o, F(0u), F(1u), ow);
}

// Any op (u0); dk_ewop_<code> are this kernel with OPC = code.
@compute @workgroup_size(256)
fn dk_ew(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let op = ew_op();
    var y0 = 0.0;
    if (ew_y_scalar(op)) { y0 = ld(1u, 0u); }
    let n = N();
    if (pairs_out(2u)) {
        let o = P(2u) >> 2u;
        let t = TG(2u);
        for (var j = gid.x; j < n / 2u; j += nwg.x * 256u) {
            H[o + j] = pack16(ew_at(op, 2u * j, y0), ew_at(op, 2u * j + 1u, y0), t);
        }
        if ((n & 1u) != 0u && gid.x == 0u) { st(2u, n - 1u, ew_at(op, n - 1u, y0)); }
        return;
    }
    for (var i = gid.x; i < n; i += nwg.x * 256u) { st(2u, i, ew_at(op, i, y0)); }
}

// ── Reductions (256 threads per workgroup, one atomic per workgroup) ─────────
var<workgroup> red: array<f32, 256>;
// The workgroup's total of v, returned to every thread (block_sum_d).
fn block_sum(v: f32, li: u32) -> f32 {
    workgroupBarrier();   // the previous call's readers are done with red
    red[li] = v;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w >>= 1u) {
        if (li < w) { red[li] += red[li + w]; }
        workgroupBarrier();
    }
    return red[0];
}
fn block_add(v: f32, k: u32, li: u32) {
    let s = block_sum(v, li);
    if (li == 0u) { atadd(k, 0u, s); }
}

// p0 = in, p1 = out[0] (zeroed by the caller); u0: 0 sum, 1 sum of squares
@compute @workgroup_size(256)
fn dk_sum(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u,
          @builtin(local_invocation_index) li: u32) {
    var acc = 0.0;
    let sq = U(0u) != 0u;
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let v = ld(0u, i);
        acc += select(v, v * v, sq);
    }
    block_add(acc, 1u, li);
}

// ── Rows and columns: u0 = cols ──────────────────────────────────────────────
// out[r,c] = x[r,c] + b[c]                                    n = rows*cols
@compute @workgroup_size(256)
fn dk_bias_rows(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) { st(2u, i, ld(0u, i) + ld(1u, i % C)); }
}
// out[r,c] = x[r,c] * g[c]                                    n = rows*cols
@compute @workgroup_size(256)
fn dk_mul_row(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) { st(2u, i, ld(0u, i) * ld(1u, i % C)); }
}
// gb[c] += sum_r d[r,c] (* x[r,c] when p2 != 0); u0 = cols, u1 = rows, u2 = S
// row slices; a thread per column               n = ceil(cols/256)*S*256
@compute @workgroup_size(256)
fn dk_colsum_acc(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                 @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let R = U(1u);
    let S = max(U(2u), 1u);
    let hx = P(2u) != 0u;
    for (var u = wg.x; u < ((C + 255u) / 256u) * S; u += nwg.x) {
        let c = (u / S) * 256u + li;
        let sl = u % S;
        if (c < C) {
            var s = 0.0;
            for (var r = sl; r < R; r += S) {
                var v = ld(0u, r * C + c);
                if (hx) { v *= ld(2u, r * C + c); }
                s += v;
            }
            if (S > 1u) { atadd(1u, c, s); } else { stf(1u, c, ldf(1u, c) + s); }
        }
    }
}
// dx[r,c] += d[r,c] * g[c]                                    n = rows*cols
@compute @workgroup_size(256)
fn dk_mul_row_acc(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) { accum(2u, i, ld(0u, i) * ld(1u, i % C)); }
}
// out[r] = sum_c x[r,c]                                       n = rows
@compute @workgroup_size(256)
fn dk_rowsum(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        var s = 0.0;
        for (var c = 0u; c < C; c++) { s += ld(0u, r * C + c); }
        st(1u, r, s);
    }
}
// dx[r,c] += g[r]                                             n = rows*cols
@compute @workgroup_size(256)
fn dk_rowbcast_acc(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) { accum(1u, i, ld(0u, i / C)); }
}
// out [rows, p+q] = x [rows, p] ++ y [rows, q]; u0 = p, u1 = q   n = rows*(p+q)
@compute @workgroup_size(256)
fn dk_concat(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Pp = U(0u);
    let Q = U(1u);
    let W = Pp + Q;
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / W;
        let c = i % W;
        if (c < Pp) { st(2u, i, ld(0u, r * Pp + c)); } else { st(2u, i, ld(1u, r * Q + c - Pp)); }
    }
}
// Columns [start, start+len) of a wide [rows, cols] tensor and a narrow one;
// u0 = cols, u1 = start, u2 = len, u3 = mode (0 slice, 1 slice backward,
// 2 concat backward)                                          n = rows*len
@compute @workgroup_size(256)
fn dk_slice(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let S = U(1u);
    let L = U(2u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / L;
        let c = i % L;
        let w = r * C + S + c;
        if (U(3u) == 1u) { st(1u, w, ld(1u, w) + ld(0u, i)); }
        else if (U(3u) == 2u) { accum(1u, i, ld(0u, w)); }
        else { st(1u, i, ld(0u, w)); }
    }
}
// out[i, :] = table[idx[i], :]; u0 = d                        n = m*d
@compute @workgroup_size(256)
fn dk_gather_rows(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let D = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / D;
        let c = i % D;
        st(2u, i, ld(0u, u32(ld(1u, r)) * D + c));
    }
}
// table.grad[idx[i], :] += g[i, :]                            n = m*d
@compute @workgroup_size(256)
fn dk_scatter_rows(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let D = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / D;
        let c = i % D;
        atadd(2u, u32(ld(1u, r)) * D + c, ld(0u, i));
    }
}

// ── Attention masks (attention.h) ────────────────────────────────────────────
fn attn_vis(flags: u32, w: u32, pos: i32, j: i32) -> bool {
    if ((flags & 1u) != 0u && j > pos) { return false; }
    if ((flags & 2u) != 0u && j <= pos - i32(w)) { return false; }
    return true;
}

// ── Softmax family: one thread per row, u0 = cols ────────────────────────────
// u1: 0 softmax, 1 log-softmax. A masked softmax: u2 = mask flags, u3 =
// window, u4 = (int) keys - queries.
@compute @workgroup_size(256)
fn dk_softmax_rows(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        let fl = U(2u);
        if (fl != 0u) {
            let pos = i32(r) + i32(U(4u));
            var mx = -3.0e38;
            for (var c = 0u; c < C; c++) { if (attn_vis(fl, U(3u), pos, i32(c))) { mx = max(mx, ld(0u, b + c)); } }
            var s = 0.0;
            for (var c = 0u; c < C; c++) { if (attn_vis(fl, U(3u), pos, i32(c))) { s += exp(ld(0u, b + c) - mx); } }
            let inv = select(0.0, 1.0 / s, s > 0.0);
            for (var c = 0u; c < C; c++) {
                var v = 0.0;
                if (attn_vis(fl, U(3u), pos, i32(c))) { v = exp(ld(0u, b + c) - mx) * inv; }
                st(1u, b + c, v);
            }
            continue;
        }
        var mx = ld(0u, b);
        for (var c = 1u; c < C; c++) { mx = max(mx, ld(0u, b + c)); }
        var s = 0.0;
        for (var c = 0u; c < C; c++) { s += exp(ld(0u, b + c) - mx); }
        if (U(1u) != 0u) {
            let l = mx + log(s);
            for (var c = 0u; c < C; c++) { st(1u, b + c, ld(0u, b + c) - l); }
        } else {
            let inv = 1.0 / s;
            for (var c = 0u; c < C; c++) { st(1u, b + c, exp(ld(0u, b + c) - mx) * inv); }
        }
    }
}
// dx += y * (g - sum(g*y)) (softmax), dx += g - exp(y) * sum(g) (log-softmax)
// p0 = g, p1 = y, p2 = dx
@compute @workgroup_size(256)
fn dk_softmax_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        var s = 0.0;
        if (U(1u) != 0u) {
            for (var c = 0u; c < C; c++) { s += ld(0u, b + c); }
            for (var c = 0u; c < C; c++) { st(2u, b + c, ld(2u, b + c) + ld(0u, b + c) - exp(ld(1u, b + c)) * s); }
        } else {
            for (var c = 0u; c < C; c++) { s += ld(0u, b + c) * ld(1u, b + c); }
            for (var c = 0u; c < C; c++) { st(2u, b + c, ld(2u, b + c) + ld(1u, b + c) * (ld(0u, b + c) - s)); }
        }
    }
}

// ── Transpose ────────────────────────────────────────────────────────────────
// out [C, R] = in [R, C]^T, or with u2 = 1 out += it; u0 = R, u1 = C   n = R*C
@compute @workgroup_size(256)
fn dk_transpose(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let R = U(0u);
    let C = U(1u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / C;
        let c = i % C;
        let v = ld(0u, i);
        if (U(2u) != 0u) { accum(1u, c * R + r, v); } else { st(1u, c * R + r, v); }
    }
}

// ── Fused multi-head attention (attention.h) ─────────────────────────────────
// FlashAttention-2's tiling as device_kernels.cu's dk_attn_*: a workgroup of
// ATT_T x ATT_T threads takes one head's tile of ATT_T rows and walks the
// other side in tiles of ATT_T through workgroup memory. Arguments: u0 = S,
// u1 = L, u2 = H, u3 = D, u4 = E, u5 = flags, u6 = window, f0 = scale;
// p4 = lse, p5 = dout, p7 = D (rowsum(dout * out)).
override ATT_T: u32 = 8u;
override ATT_MAXD: u32 = 128u;
const ATT_HID = -3.0e38;
const ATT_PER_MAX = 16u;   // ATT_MAXD / ATT_T for either build
var<workgroup> aQ: array<f32, ATT_T * (ATT_MAXD + 1u)>;
var<workgroup> aK: array<f32, ATT_T * (ATT_MAXD + 1u)>;
var<workgroup> aV: array<f32, ATT_T * (ATT_MAXD + 1u)>;
var<workgroup> aG: array<f32, ATT_T * (ATT_MAXD + 1u)>;
var<workgroup> aP: array<f32, ATT_T * (ATT_T + 1u)>;
var<workgroup> aD: array<f32, ATT_T * (ATT_T + 1u)>;
var<workgroup> aRd: array<f32, ATT_T>;
var<workgroup> aRl: array<f32, ATT_T>;

// The key range a tile of rows [i0, i0 + ATT_T) can see: [lo, hi).
fn attn_range(i0: u32) -> vec2i {
    let off = i32(U(1u)) - i32(U(0u));
    var lo = 0;
    var hi = i32(U(1u));
    if ((U(5u) & 1u) != 0u) { let m = i32(i0) + i32(ATT_T) - 1 + off + 1; if (m < hi) { hi = m; } }
    if ((U(5u) & 2u) != 0u) { let m = i32(i0) + off - i32(U(6u)) + 1; if (m > lo) { lo = m; } }
    if (lo < 0) { lo = 0; }
    return vec2i(lo, hi);
}
// rows [r0, r0 + ATT_T) of argument k ([rows, H*W]), head h -> tile 'which'
// (0 aQ, 1 aK, 2 aV, 3 aG), 0 past the end
fn attn_load(which: u32, k: u32, r0: u32, rows: u32, Hh: u32, h: u32, W: u32, li: u32) {
    let LD = ATT_MAXD + 1u;
    for (var q = li; q < ATT_T * W; q += ATT_T * ATT_T) {
        let r = q / W;
        let c = q % W;
        var v = 0.0;
        if (r0 + r < rows) { v = ld(k, ((r0 + r) * Hh + h) * W + c); }
        let at = r * LD + c;
        switch which {
            case 0u: { aQ[at] = v; }
            case 1u: { aK[at] = v; }
            case 2u: { aV[at] = v; }
            default: { aG[at] = v; }
        }
    }
}
// p0 q, p1 k, p2 v, p3 out, p4 lse (written)
@compute @workgroup_size(ATT_T * ATT_T)
fn dk_attn_fwd(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
               @builtin(num_workgroups) nwg: vec3u) {
    let S = U(0u);
    let L = U(1u);
    let Hh = U(2u);
    let D = U(3u);
    let E = U(4u);
    let fl = U(5u) & 3u;
    let off = i32(L) - i32(S);
    let tiles = (S + ATT_T - 1u) / ATT_T;
    let r = li / ATT_T;
    let ln = li % ATT_T;
    let LD = ATT_MAXD + 1u;
    let PL = ATT_T + 1u;
    let PER = ATT_MAXD / ATT_T;
    for (var u = wg.x; u < Hh * tiles; u += nwg.x) {
        let h = u / tiles;
        let i0 = (u % tiles) * ATT_T;
        let i = i0 + r;
        attn_load(0u, 0u, i0, S, Hh, h, D, li);
        var acc: array<f32, ATT_PER_MAX>;
        var m = ATT_HID;
        var l = 0.0;
        let rg = attn_range(i0);
        for (var j0 = (rg.x / i32(ATT_T)) * i32(ATT_T); j0 < rg.y; j0 += i32(ATT_T)) {
            workgroupBarrier();
            attn_load(1u, 1u, u32(j0), L, Hh, h, D, li);
            attn_load(2u, 2u, u32(j0), L, Hh, h, E, li);
            workgroupBarrier();
            let j = u32(j0) + ln;
            var sc = ATT_HID;
            if (i < S && j < L && attn_vis(fl, U(6u), i32(i) + off, i32(j))) {
                var s = 0.0;
                for (var d = 0u; d < D; d++) { s += aQ[r * LD + d] * aK[ln * LD + d]; }
                sc = s * F(0u);
            }
            aP[r * PL + ln] = sc;
            workgroupBarrier();
            var mt = ATT_HID;
            for (var c = 0u; c < ATT_T; c++) { mt = max(mt, aP[r * PL + c]); }
            if (mt > ATT_HID) {
                let mn = max(m, mt);
                let cr = exp(m - mn);
                var pv: array<f32, ATT_PER_MAX>;
                var ps = 0.0;
                for (var c = 0u; c < ATT_T; c++) {
                    let sv = aP[r * PL + c];
                    var p = 0.0;
                    if (sv > ATT_HID) { p = exp(sv - mn); }
                    pv[c] = p;
                    ps += p;
                }
                l = l * cr + ps;
                for (var t = 0u; t < PER; t++) {
                    let e = ln + t * ATT_T;
                    if (e < E) {
                        var x = acc[t] * cr;
                        for (var c = 0u; c < ATT_T; c++) { x += pv[c] * aV[c * LD + e]; }
                        acc[t] = x;
                    }
                }
                m = mn;
            }
        }
        if (i < S) {
            var inv = 0.0;
            if (l > 0.0) { inv = 1.0 / l; }
            let ob = (i * Hh + h) * E;
            for (var t = 0u; t < PER; t++) { let e = ln + t * ATT_T; if (e < E) { st(3u, ob + e, acc[t] * inv); } }
            if (ln == 0u) {
                var lse = 3.0e38;   // no visible key: every p is 0
                if (l > 0.0) { lse = m + log(l); }
                stf(4u, h * S + i, lse);
            }
        }
        workgroupBarrier();
    }
}
// p0 q, p1 k, p2 v, p3 out, p6 dq (+)= scale * sum_j ds_ij k_j (0: none),
// ds_ij = p_ij (dout_i . v_j - D_i)
@compute @workgroup_size(ATT_T * ATT_T)
fn dk_attn_bwd_q(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                 @builtin(num_workgroups) nwg: vec3u) {
    let S = U(0u);
    let L = U(1u);
    let Hh = U(2u);
    let D = U(3u);
    let E = U(4u);
    let fl = U(5u) & 3u;
    let off = i32(L) - i32(S);
    let tiles = (S + ATT_T - 1u) / ATT_T;
    let r = li / ATT_T;
    let ln = li % ATT_T;
    let LD = ATT_MAXD + 1u;
    let PL = ATT_T + 1u;
    let PER = ATT_MAXD / ATT_T;
    for (var u = wg.x; u < Hh * tiles; u += nwg.x) {
        let h = u / tiles;
        let i0 = (u % tiles) * ATT_T;
        let i = i0 + r;
        attn_load(3u, 5u, i0, S, Hh, h, E, li);
        attn_load(1u, 3u, i0, S, Hh, h, E, li);   // the output rows, for D (aK is free until the key tiles)
        workgroupBarrier();
        var part = 0.0;
        for (var e = ln; e < E; e += ATT_T) { part += aG[r * LD + e] * aK[r * LD + e]; }
        aD[r * PL + ln] = part;
        workgroupBarrier();
        if (ln == 0u) {
            var Di = 0.0;
            for (var c = 0u; c < ATT_T; c++) { Di += aD[r * PL + c]; }
            aRd[r] = Di;
            var rl = 0.0;
            if (i < S) { rl = ldf(4u, h * S + i); stf(7u, h * S + i, Di); }
            aRl[r] = rl;
        }
        if (P(6u) == 0u) { workgroupBarrier(); continue; }
        attn_load(0u, 0u, i0, S, Hh, h, D, li);
        var dq: array<f32, ATT_PER_MAX>;
        let rg = attn_range(i0);
        for (var j0 = (rg.x / i32(ATT_T)) * i32(ATT_T); j0 < rg.y; j0 += i32(ATT_T)) {
            workgroupBarrier();
            attn_load(1u, 1u, u32(j0), L, Hh, h, D, li);
            attn_load(2u, 2u, u32(j0), L, Hh, h, E, li);
            workgroupBarrier();
            let j = u32(j0) + ln;
            var ds = 0.0;
            if (i < S && j < L && attn_vis(fl, U(6u), i32(i) + off, i32(j))) {
                var s = 0.0;
                var dp = 0.0;
                for (var d = 0u; d < D; d++) { s += aQ[r * LD + d] * aK[ln * LD + d]; }
                for (var e = 0u; e < E; e++) { dp += aG[r * LD + e] * aV[ln * LD + e]; }
                ds = exp(s * F(0u) - aRl[r]) * (dp - aRd[r]);
            }
            aD[r * PL + ln] = ds;
            workgroupBarrier();
            for (var t = 0u; t < PER; t++) {
                let d = ln + t * ATT_T;
                if (d < D) {
                    var x = dq[t];
                    for (var c = 0u; c < ATT_T; c++) { x += aD[r * PL + c] * aK[c * LD + d]; }
                    dq[t] = x;
                }
            }
        }
        if (i < S) {
            let ob = (i * Hh + h) * D;
            for (var t = 0u; t < PER; t++) {
                let d = ln + t * ATT_T;
                if (d < D) {
                    if ((U(5u) & 4u) != 0u) { st(6u, ob + d, F(0u) * dq[t]); }
                    else { st(6u, ob + d, ld(6u, ob + d) + F(0u) * dq[t]); }
                }
            }
        }
        workgroupBarrier();
    }
}
// p0 q, p1 k, p2 v, p3 dk (+)= scale * sum_i ds_ij q_i, p6 dv (+)= sum_i p_ij
// dout_i (either 0: none); after dk_attn_bwd_q. A workgroup takes ATT_T keys.
@compute @workgroup_size(ATT_T * ATT_T)
fn dk_attn_bwd_kv(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                  @builtin(num_workgroups) nwg: vec3u) {
    let S = U(0u);
    let L = U(1u);
    let Hh = U(2u);
    let D = U(3u);
    let E = U(4u);
    let fl = U(5u) & 3u;
    let off = i32(L) - i32(S);
    let tiles = (L + ATT_T - 1u) / ATT_T;
    let r = li / ATT_T;
    let ln = li % ATT_T;
    let LD = ATT_MAXD + 1u;
    let PL = ATT_T + 1u;
    let PER = ATT_MAXD / ATT_T;
    for (var u = wg.x; u < Hh * tiles; u += nwg.x) {
        let h = u / tiles;
        let j0 = (u % tiles) * ATT_T;
        let j = j0 + r;
        attn_load(1u, 1u, j0, L, Hh, h, D, li);
        attn_load(2u, 2u, j0, L, Hh, h, E, li);
        var dk: array<f32, ATT_PER_MAX>;
        var dv: array<f32, ATT_PER_MAX>;
        // the queries that can see keys [j0, j0 + ATT_T): [qlo, qhi)
        var qlo = 0;
        var qhi = i32(S);
        if ((fl & 1u) != 0u) { let m2 = i32(j0) - off; if (m2 > qlo) { qlo = m2; } }
        if ((fl & 2u) != 0u) { let m2 = i32(j0) + i32(ATT_T) - 1 - off + i32(U(6u)); if (m2 < qhi) { qhi = m2; } }
        if (qlo < 0) { qlo = 0; }
        for (var i0 = (qlo / i32(ATT_T)) * i32(ATT_T); i0 < qhi; i0 += i32(ATT_T)) {
            workgroupBarrier();
            attn_load(0u, 0u, u32(i0), S, Hh, h, D, li);
            attn_load(3u, 5u, u32(i0), S, Hh, h, E, li);
            if (li < ATT_T) {
                let iq = u32(i0) + li;
                var rl = 0.0;
                var rd = 0.0;
                if (iq < S) { rl = ldf(4u, h * S + iq); rd = ldf(7u, h * S + iq); }
                aRl[li] = rl;
                aRd[li] = rd;
            }
            workgroupBarrier();
            // thread (key r, query ln)
            let i = u32(i0) + ln;
            var p = 0.0;
            var ds = 0.0;
            if (i < S && j < L && attn_vis(fl, U(6u), i32(i) + off, i32(j))) {
                var s = 0.0;
                var dp = 0.0;
                for (var d = 0u; d < D; d++) { s += aQ[ln * LD + d] * aK[r * LD + d]; }
                for (var e = 0u; e < E; e++) { dp += aG[ln * LD + e] * aV[r * LD + e]; }
                p = exp(s * F(0u) - aRl[ln]);
                ds = p * (dp - aRd[ln]);
            }
            aP[r * PL + ln] = p;
            aD[r * PL + ln] = ds;
            workgroupBarrier();
            for (var t = 0u; t < PER; t++) {
                let d = ln + t * ATT_T;
                if (d < E) {
                    var x = dv[t];
                    for (var c = 0u; c < ATT_T; c++) { x += aP[r * PL + c] * aG[c * LD + d]; }
                    dv[t] = x;
                }
                if (d < D) {
                    var x = dk[t];
                    for (var c = 0u; c < ATT_T; c++) { x += aD[r * PL + c] * aQ[c * LD + d]; }
                    dk[t] = x;
                }
            }
        }
        if (j < L) {
            if (P(3u) != 0u) {
                let ob = (j * Hh + h) * D;
                for (var t = 0u; t < PER; t++) {
                    let d = ln + t * ATT_T;
                    if (d < D) {
                        if ((U(5u) & 8u) != 0u) { st(3u, ob + d, F(0u) * dk[t]); }
                        else { st(3u, ob + d, ld(3u, ob + d) + F(0u) * dk[t]); }
                    }
                }
            }
            if (P(6u) != 0u) {
                let ob = (j * Hh + h) * E;
                for (var t = 0u; t < PER; t++) {
                    let e = ln + t * ATT_T;
                    if (e < E) {
                        if ((U(5u) & 16u) != 0u) { st(6u, ob + e, dv[t]); }
                        else { st(6u, ob + e, ld(6u, ob + e) + dv[t]); }
                    }
                }
            }
        }
        workgroupBarrier();
    }
}

// ── Fused losses -> loss[0] (zeroed by the caller), averaged ─────────────────
// Cross-entropy on class indices: p0 = logits [rows, C], p1 = labels, p2 =
// loss[0]; u0 = C; f0 = 1/rows                                n = rows
@compute @workgroup_size(256)
fn dk_ce_fwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u,
             @builtin(local_invocation_index) li: u32) {
    let C = U(0u);
    var acc = 0.0;
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        var mx = ld(0u, b);
        for (var c = 1u; c < C; c++) { mx = max(mx, ld(0u, b + c)); }
        var s = 0.0;
        for (var c = 0u; c < C; c++) { s += exp(ld(0u, b + c) - mx); }
        let lab = u32(ld(1u, r));
        acc += (mx + log(s) - ld(0u, b + lab)) * F(0u);
    }
    block_add(acc, 2u, li);
}
// dlogits += seed[0] * (softmax - onehot) / rows; p3 = seed
@compute @workgroup_size(256)
fn dk_ce_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        let sc = ld(3u, 0u) * F(0u);
        var mx = ld(0u, b);
        for (var c = 1u; c < C; c++) { mx = max(mx, ld(0u, b + c)); }
        var s = 0.0;
        for (var c = 0u; c < C; c++) { s += exp(ld(0u, b + c) - mx); }
        let lab = u32(ld(1u, r));
        let inv = 1.0 / s;
        for (var c = 0u; c < C; c++) { accum(2u, b + c, sc * (exp(ld(0u, b + c) - mx) * inv - select(0.0, 1.0, c == lab))); }
    }
}
// Elementwise losses: u0 = 0 BCE on logits, 1 Huber (f1 = delta); p0 = x,
// p1 = target, p2 = loss[0], f0 = 1/n                          n = elements
@compute @workgroup_size(256)
fn dk_ewloss_fwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u,
                 @builtin(local_invocation_index) li: u32) {
    var acc = 0.0;
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let x = ld(0u, i);
        let t = ld(1u, i);
        var v = 0.0;
        if (U(0u) == 0u) { v = max(x, 0.0) - x * t + log(1.0 + exp(-abs(x))); }
        else {
            let d = x - t;
            let ad = abs(d);
            let k = F(1u);
            v = select(k * (ad - 0.5 * k), 0.5 * d * d, ad <= k);
        }
        acc += v * F(0u);
    }
    block_add(acc, 2u, li);
}
// dx += seed[0] / n * (sigmoid(x) - t) | clamp(x - t, -delta, delta); p2 = dx, p3 = seed
@compute @workgroup_size(256)
fn dk_ewloss_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let sc = ld(3u, 0u) * F(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let x = ld(0u, i);
        let t = ld(1u, i);
        var g = 0.0;
        if (U(0u) == 0u) { g = sig_f(x) - t; } else { let k = F(1u); g = min(max(x - t, -k), k); }
        accum(2u, i, sc * g);
    }
}

// ── Layer norm: one thread per row, u0 = cols, f0 = eps ──────────────────────
// p0 = x, p1 = xhat out, p2 = inv std [rows]
@compute @workgroup_size(256)
fn dk_layernorm(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        var m = 0.0;
        for (var c = 0u; c < C; c++) { m += ld(0u, b + c); }
        m /= f32(C);
        var v = 0.0;
        for (var c = 0u; c < C; c++) { let d = ld(0u, b + c) - m; v += d * d; }
        let is = inverseSqrt(v / f32(C) + F(0u));
        st(2u, r, is);
        for (var c = 0u; c < C; c++) { st(1u, b + c, (ld(0u, b + c) - m) * is); }
    }
}
// dx += inv * (g - mean(g) - xhat * mean(g * xhat)); p0 = g, p1 = xhat, p2 = inv, p3 = dx
@compute @workgroup_size(256)
fn dk_layernorm_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    for (var r = gid.x; r < N(); r += nwg.x * 256u) {
        let b = r * C;
        var mg = 0.0;
        var mgy = 0.0;
        for (var c = 0u; c < C; c++) { let g = ld(0u, b + c); mg += g; mgy += g * ld(1u, b + c); }
        mg /= f32(C);
        mgy /= f32(C);
        let is = ld(2u, r);
        for (var c = 0u; c < C; c++) {
            st(3u, b + c, ld(3u, b + c) + is * (ld(0u, b + c) - mg - ld(1u, b + c) * mgy));
        }
    }
}

// ── Optimizer step over one parameter (nn.h's OPT_* arithmetic) ─────────────
// p0 = w, p1 = g, p2 = m, p3 = v, p4 = scale[0] (gradient clip factor, or 0 =
// none); u0 = kind; f0 lr, f1 wd, f2 momentum, f3 beta1, f4 beta2, f5 eps,
// f6 1/bc1, f7 1/sqrt(bc2). w, g, m, v are word indices of fp32 buffers.
const OPT_SGD = 0;
const OPT_MOMENTUM = 1;
const OPT_NESTEROV = 2;
const OPT_RMSPROP = 3;
const OPT_ADAM = 4;
const OPT_ADAMW = 5;
const OPT_ADAGRAD = 6;
const OPT_LION = 7;
fn opt_one(w: u32, gr: u32, m: u32, v: u32, i: u32, cs: f32) {
    let kind = i32(U(0u));
    let lr = F(0u);
    let wd = F(1u);
    let mu = F(2u);
    let b1 = F(3u);
    let b2 = F(4u);
    let eps = F(5u);
    let g0 = rdw(gr + i) * cs;
    let wi = rdw(w + i);
    if (kind == OPT_SGD) { wrw(w + i, wi - lr * (g0 + wd * wi)); }
    else if (kind == OPT_MOMENTUM || kind == OPT_NESTEROV) {
        let g = g0 + wd * wi;
        let mi = mu * rdw(m + i) + g;
        wrw(m + i, mi);
        wrw(w + i, wi - lr * select(mi, g + mu * mi, kind == OPT_NESTEROV));
    } else if (kind == OPT_ADAGRAD) {
        let g = g0 + wd * wi;
        let vi = rdw(v + i) + g * g;
        wrw(v + i, vi);
        wrw(w + i, wi - lr * g / (sqrt(vi) + eps));
    } else if (kind == OPT_LION) {
        let mi = rdw(m + i);
        let c = b1 * mi + (1.0 - b1) * g0;
        let sg = select(select(0.0, -1.0, c < 0.0), 1.0, c > 0.0);
        wrw(w + i, wi - lr * (sg + wd * wi));
        wrw(m + i, b2 * mi + (1.0 - b2) * g0);
    } else if (kind == OPT_RMSPROP) {
        let g = g0 + wd * wi;
        let vi = b2 * rdw(v + i) + (1.0 - b2) * g * g;
        wrw(v + i, vi);
        wrw(w + i, wi - lr * g / (sqrt(vi) + eps));
    } else {
        let l2 = select(wd, 0.0, kind == OPT_ADAMW);
        let dec = select(0.0, wd, kind == OPT_ADAMW);
        let g = g0 + l2 * wi;
        let mi = b1 * rdw(m + i) + (1.0 - b1) * g;
        let vi = b2 * rdw(v + i) + (1.0 - b2) * g * g;
        wrw(m + i, mi);
        wrw(v + i, vi);
        wrw(w + i, wi - lr * ((mi * F(6u)) / (sqrt(vi) * F(7u) + eps) + dec * wi));
    }
}
// p5 (when set): the loss scale's state; an overflowed step changes nothing.
@compute @workgroup_size(256)
fn dk_opt(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    if (P(5u) != 0u && ld(5u, 2u) != 0.0) { return; }
    var cs = 1.0;
    if (P(4u) != 0u) { cs = ld(4u, 0u); }
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        opt_one(P(0u) >> 2u, P(1u) >> 2u, P(2u) >> 2u, P(3u) >> 2u, i, cs);
    }
}
// Every parameter's step in one launch: p0 = a table of u1 entries {w, g, m,
// v, n, start} (64-bit; their low words: byte offsets and counts)
@compute @workgroup_size(256)
fn dk_opt_multi(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let T = P(0u) >> 2u;
    let cnt = U(1u);
    if (P(5u) != 0u && ld(5u, 2u) != 0.0) { return; }
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        var lo = 0u;
        var hi = cnt - 1u;
        while (lo < hi) {
            let mid = (lo + hi + 1u) / 2u;
            if (H[T + mid * 12u + 10u] <= i) { lo = mid; } else { hi = mid - 1u; }
        }
        let e = T + lo * 12u;
        opt_one(H[e] >> 2u, H[e + 2u] >> 2u, H[e + 4u] >> 2u, H[e + 6u] >> 2u, i - H[e + 10u], 1.0);
    }
}
// scale[0] = sumsq[0] > maxNorm^2 ? maxNorm / sqrt(sumsq[0]) : 1; p0 = sumsq,
// p1 = scale, f0 = maxNorm
@compute @workgroup_size(256)
fn dk_clip_scale(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x == 0u) {
        let n = sqrt(ld(0u, 0u));
        st(1u, 0u, select(1.0, F(0u) / n, n > F(0u) && n > 0.0));
    }
}

// ── Convolution helpers (convnet.h's layout: [batch, C*H*W] rows, NCHW) ─────
// u0 C, u1 H, u2 W, u3 kH, u4 kW, u5 OH, u6 OW; f0 strideH, f1 strideW, f2
// padH, f3 padW, f4 dilH, f5 dilW, f6 = B (integers carried as floats).
// col [C*kH*kW, B*OH*OW]; p0 = x [B, C*H*W], p1 = col     n = C*kH*kW*B*OH*OW
struct ConvGeo { C: u32, Hh: u32, W: u32, KH: u32, KW: u32, OH: u32, OW: u32, B: u32,
                 sh: i32, sw: i32, ph: i32, pw: i32, dh: i32, dw: i32 }
fn conv_geo() -> ConvGeo {
    return ConvGeo(U(0u), U(1u), U(2u), U(3u), U(4u), U(5u), U(6u), u32(F(6u)),
                   i32(F(0u)), i32(F(1u)), i32(F(2u)), i32(F(3u)), i32(F(4u)), i32(F(5u)));
}
// The input element of col entry i, or -1 outside the image.
fn col_src(g: ConvGeo, i: u32) -> i32 {
    let OP = g.OH * g.OW;
    let BP = g.B * OP;
    let q = i % BP;
    let r = i / BP;
    let b = q / OP;
    let p = q % OP;
    let oh = p / g.OW;
    let ow = p % g.OW;
    let kw = r % g.KW;
    let kh = (r / g.KW) % g.KH;
    let c = r / (g.KW * g.KH);
    let y = i32(oh) * g.sh - g.ph + i32(kh) * g.dh;
    let x = i32(ow) * g.sw - g.pw + i32(kw) * g.dw;
    if (y < 0 || y >= i32(g.Hh) || x < 0 || x >= i32(g.W)) { return -1; }
    return i32(((b * g.C + c) * g.Hh + u32(y)) * g.W + u32(x));
}
@compute @workgroup_size(256)
fn dk_im2col(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let g = conv_geo();
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let s = col_src(g, i);
        var v = 0.0;
        if (s >= 0) { v = ld(0u, u32(s)); }
        st(1u, i, v);
    }
}
// x [B, C*H*W] += col2im(col): the adjoint of dk_im2col (atomics: windows overlap)
@compute @workgroup_size(256)
fn dk_col2im(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let g = conv_geo();
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let s = col_src(g, i);
        if (s >= 0) { atadd(1u, u32(s), ld(0u, i)); }
    }
}
// out [B2, A, P] (+)= in [A, B2, P] with the first two axes swapped; u0 A,
// u1 B2, u2 P, u3 = 1 to accumulate                          n = A*B2*P
@compute @workgroup_size(256)
fn dk_swap01(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Aa = U(0u);
    let B2 = U(1u);
    let Pp = U(2u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let p = i % Pp;
        let t = i / Pp;
        let bi = t % B2;
        let ai = t / B2;
        let o = (bi * Aa + ai) * Pp + p;
        if (U(3u) != 0u) { st(1u, o, ld(1u, o) + ld(0u, i)); } else { st(1u, o, ld(0u, i)); }
    }
}
// The channel of value i in [batch, C, P].
fn chan_of(i: u32, C: u32, Pp: u32) -> u32 { return (i / Pp) % C; }
// out[b, c, p] += bias[c]; u0 = channels, u1 = plane            n = batch*C*plane
@compute @workgroup_size(256)
fn dk_bias_channels(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let Pp = U(1u);
    let n = N();
    if (pairs_out(0u) && (Pp & 1u) == 0u && (n & 1u) == 0u) {
        // a pair shares its channel (even planes)
        let o = P(0u) >> 2u;
        let t = TG(0u);
        for (var j = gid.x; j < n / 2u; j += nwg.x * 256u) {
            let b = ldf(1u, chan_of(2u * j, C, Pp));
            let w = H[o + j];
            H[o + j] = pack16(h2f(w & 0xffffu, t) + b, h2f(w >> 16u, t) + b, t);
        }
        return;
    }
    for (var i = gid.x; i < n; i += nwg.x * 256u) { st(0u, i, ld(0u, i) + ld(1u, chan_of(i, C, Pp))); }
}
// The index of the q-th value of channel c.
fn chan_at(q: u32, c: u32, C: u32, Pp: u32) -> u32 {
    let b = q / Pp;
    return (b * C + c) * Pp + (q - b * Pp);
}
// gb[c] += sum over batch and plane of d; u0 = C, u1 = plane, u2 = batch,
// u3 = S slices: slice s takes the 256-value runs r with r % S == s   n = C*S*256
@compute @workgroup_size(256)
fn dk_bias_channels_bwd(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                        @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let Pp = U(1u);
    let M = U(2u) * U(1u);
    let S = U(3u);
    for (var u = wg.x; u < C * S; u += nwg.x) {
        let c = u / S;
        let sl = u % S;
        var s = 0.0;
        for (var q = sl * 256u + li; q < M; q += S * 256u) { s += ld(0u, chan_at(q, c, C, Pp)); }
        s = block_sum(s, li);
        if (li == 0u) { atadd(1u, c, s); }
    }
}

// ── Batch / group normalization (convnet.h's layout [batch, C*plane]) ───────
// Batch norm statistics, pass 1: per (channel, slice) the count, mean and
// sum of squared deviations in p1 [C*S*3] (floats). p0 x; u0 C, u1 plane,
// u2 batch, u3 S                                              n = C*S*256
@compute @workgroup_size(256)
fn dk_bn_partial(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                 @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let Pp = U(1u);
    let M = U(2u) * U(1u);
    let S = U(3u);
    for (var u = wg.x; u < C * S; u += nwg.x) {
        let c = u / S;
        let sl = u % S;
        var s = 0.0;
        var cnt = 0.0;
        for (var q = sl * 256u + li; q < M; q += S * 256u) { s += ld(0u, chan_at(q, c, C, Pp)); cnt += 1.0; }
        cnt = block_sum(cnt, li);
        let tot = block_sum(s, li);
        var mu = 0.0;
        if (cnt > 0.0) { mu = tot / cnt; }
        var sq = 0.0;
        for (var q = sl * 256u + li; q < M; q += S * 256u) { let d = ld(0u, chan_at(q, c, C, Pp)) - mu; sq += d * d; }
        sq = block_sum(sq, li);
        if (li == 0u) { stf(1u, u * 3u, cnt); stf(1u, u * 3u + 1u, mu); stf(1u, u * 3u + 2u, sq); }
    }
}
// Pass 2, one thread per channel: the slices combined (Chan et al.), the
// running statistics updated (training), mean and invstd. p0 partials, p1
// mean[C], p2 invstd[C], p3 running mean, p4 running var; u0 C, u3 S,
// u4 training; f0 momentum, f1 eps                              n = C
@compute @workgroup_size(256)
fn dk_bn_finish(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let S = U(3u);
    for (var c = gid.x; c < N(); c += nwg.x * 256u) {
        var mean = 0.0;
        var vr = 0.0;
        if (U(4u) != 0u) {
            var n = 0.0;
            var mu = 0.0;
            var m2 = 0.0;
            for (var s = 0u; s < S; s++) {
                let p = (c * S + s) * 3u;
                let pn = ldf(0u, p);
                if (pn == 0.0) { continue; }
                let nn = n + pn;
                let d = ldf(0u, p + 1u) - mu;
                mu += d * pn / nn;
                m2 += ldf(0u, p + 2u) + d * d * n * pn / nn;
                n = nn;
            }
            mean = mu;
            vr = m2 / n;
            var unb = vr;
            if (n > 1.0) { unb = m2 / (n - 1.0); }
            st(3u, c, (1.0 - F(0u)) * ld(3u, c) + F(0u) * mean);
            st(4u, c, (1.0 - F(0u)) * ld(4u, c) + F(0u) * unb);
        } else {
            mean = ld(3u, c);
            vr = ld(4u, c);
        }
        st(1u, c, mean);
        st(2u, c, 1.0 / sqrt(vr + F(1u)));
    }
}
// Group norm statistics per (image, group) unit of M contiguous values:
// p0 x, p1 mean[U], p2 invstd[U]; u0 M, u1 U; f0 eps            n = U*256
@compute @workgroup_size(256)
fn dk_gn_stats(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
               @builtin(num_workgroups) nwg: vec3u) {
    let M = U(0u);
    for (var u = wg.x; u < U(1u); u += nwg.x) {
        let b = u * M;
        var s = 0.0;
        for (var q = li; q < M; q += 256u) { s += ld(0u, b + q); }
        let mu = block_sum(s, li) / f32(M);
        var sq = 0.0;
        for (var q = li; q < M; q += 256u) { let d = ld(0u, b + q) - mu; sq += d * d; }
        sq = block_sum(sq, li);
        if (li == 0u) { st(1u, u, mu); st(2u, u, 1.0 / sqrt(sq / f32(M) + F(0u))); }
    }
}
// The statistic of value i: its channel (batch norm, G = 0) or its (image, group).
fn norm_stat(i: u32, C: u32, Pp: u32, G: u32) -> u32 {
    let t = i / Pp;
    let img = t / C;
    let c = t - img * C;
    if (G == 0u) { return c; }
    return img * G + c / (C / G);
}
// A normalization fused with the activation after it: u5 = 0 none, 1 ReLU,
// 2 leaky ReLU (slope f1); the backward kernels take its output in p7.
fn fused_act(v: f32) -> f32 {
    if (U(5u) == 0u || v > 0.0) { return v; }
    return select(F(1u) * v, 0.0, U(5u) == 1u);
}
fn fused_dy(i: u32) -> f32 {
    let dy = ld(0u, i);
    if (U(5u) == 0u || ld(7u, i) > 0.0) { return dy; }
    return select(F(1u) * dy, 0.0, U(5u) == 1u);
}
// out = gamma[c] (x - mean) invstd + beta[c]: p0 x, p1 out, p2 mean, p3 invstd,
// p4 gamma, p5 beta; u0 C, u1 plane, u2 groups (0: batch norm)  n = values
fn norm_apply_at(i: u32) -> f32 {
    let C = U(0u);
    let Pp = U(1u);
    let c = chan_of(i, C, Pp);
    let s = norm_stat(i, C, Pp, U(2u));
    return fused_act(ld(4u, c) * (ld(0u, i) - ld(2u, s)) * ld(3u, s) + ld(5u, c));
}
@compute @workgroup_size(256)
fn dk_norm_apply(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let n = N();
    if (pairs_out(1u)) {
        let o = P(1u) >> 2u;
        let t = TG(1u);
        for (var j = gid.x; j < n / 2u; j += nwg.x * 256u) { H[o + j] = pack16(norm_apply_at(2u * j), norm_apply_at(2u * j + 1u), t); }
        if ((n & 1u) != 0u && gid.x == 0u) { st(1u, n - 1u, norm_apply_at(n - 1u)); }
        return;
    }
    for (var i = gid.x; i < n; i += nwg.x * 256u) { st(1u, i, norm_apply_at(i)); }
}
// Per channel: dgamma[c] += sum dy xh, dbeta[c] += sum dy, and (p6, when
// set) sums[c] += sum dy, sums[C + c] += sum dy xh. p0 dy, p1 x, p2 mean,
// p3 invstd, p4 dgamma, p5 dbeta (either may be 0); u0 C, u1 plane,
// u2 groups, u3 batch, u4 S slices                            n = C*S*256
@compute @workgroup_size(256)
fn dk_norm_sums(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
                @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let Pp = U(1u);
    let G = U(2u);
    let M = U(3u) * U(1u);
    let S = U(4u);
    for (var u = wg.x; u < C * S; u += nwg.x) {
        let c = u / S;
        let sl = u % S;
        var sd = 0.0;
        var sx = 0.0;
        for (var q = sl * 256u + li; q < M; q += S * 256u) {
            let i = chan_at(q, c, C, Pp);
            var s = c;
            if (G != 0u) { s = norm_stat(i, C, Pp, G); }
            let dy = fused_dy(i);
            sd += dy;
            sx += dy * (ld(1u, i) - ld(2u, s)) * ld(3u, s);
        }
        sd = block_sum(sd, li);
        sx = block_sum(sx, li);
        if (li == 0u) {
            if (P(4u) != 0u) { atadd(4u, c, sx); }
            if (P(5u) != 0u) { atadd(5u, c, sd); }
            if (P(6u) != 0u) { atadd(6u, c, sd); atadd(6u, C + c, sx); }
        }
    }
}
// Batch norm: dx += gamma invstd (dy - m1 - xh m2), m1 = sum dy / N, m2 =
// sum dy xh / N in training (0 in eval). p0 dy, p1 x, p2 mean, p3 invstd,
// p4 gamma, p5 sums, p6 dx; u0 C, u1 plane, u2 training; f0 1/N   n = values
fn bn_bwd_dx_at(i: u32) -> f32 {
    let C = U(0u);
    let c = chan_of(i, C, U(1u));
    let inv = ld(3u, c);
    var m1 = 0.0;
    var m2 = 0.0;
    if (U(2u) != 0u) { m1 = ld(5u, c) * F(0u); m2 = ld(5u, C + c) * F(0u); }
    let xh = (ld(1u, i) - ld(2u, c)) * inv;
    var v = ld(4u, c) * inv * (fused_dy(i) - m1 - xh * m2);
    if (U(6u) == 0u) { v += ld(6u, i); }
    return v;
}
@compute @workgroup_size(256)
fn dk_bn_bwd_dx(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let n = N();
    if (pairs_out(6u)) {
        let o = P(6u) >> 2u;
        let t = TG(6u);
        for (var j = gid.x; j < n / 2u; j += nwg.x * 256u) { H[o + j] = pack16(bn_bwd_dx_at(2u * j), bn_bwd_dx_at(2u * j + 1u), t); }
        if ((n & 1u) != 0u && gid.x == 0u) { st(6u, n - 1u, bn_bwd_dx_at(n - 1u)); }
        return;
    }
    for (var i = gid.x; i < n; i += nwg.x * 256u) { st(6u, i, bn_bwd_dx_at(i)); }
}
// Group norm: per (image, group) unit, dxh = dy gamma[c],
// dx += invstd (dxh - mean(dxh) - xh mean(dxh xh)). p0 dy, p1 x, p2 mean,
// p3 invstd, p4 gamma, p5 dx; u0 C, u1 plane, u2 groups, u3 units   n = U*256
@compute @workgroup_size(256)
fn dk_gn_bwd(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
             @builtin(num_workgroups) nwg: vec3u) {
    let C = U(0u);
    let Pp = U(1u);
    let G = U(2u);
    let Cg = C / G;
    let M = Cg * Pp;
    for (var u = wg.x; u < U(3u); u += nwg.x) {
        let b = u * M;
        let g = u % G;
        let mean = ld(2u, u);
        let inv = ld(3u, u);
        var s1 = 0.0;
        var s2 = 0.0;
        for (var q = li; q < M; q += 256u) {
            let dxh = fused_dy(b + q) * ld(4u, g * Cg + q / Pp);
            s1 += dxh;
            s2 += dxh * (ld(1u, b + q) - mean) * inv;
        }
        let m1 = block_sum(s1, li) / f32(M);
        let m2 = block_sum(s2, li) / f32(M);
        for (var q = li; q < M; q += 256u) {
            let dxh = fused_dy(b + q) * ld(4u, g * Cg + q / Pp);
            let xh = (ld(1u, b + q) - mean) * inv;
            st(5u, b + q, ld(5u, b + q) + inv * (dxh - m1 - xh * m2));
        }
    }
}

// ── Pooling ([batch, C*H*W] planes) ──────────────────────────────────────────
// u0 C, u1 H, u2 W, u3 oH, u4 oW, u5 mode (0 max, 1 avg over the padded
// window, 2 avg over the part inside the image); f0..f5 kH, kW, sH, sW, pH, pW.
struct PoolWin { h0: u32, h1: u32, w0: u32, w1: u32, padded: u32 }
fn pool_win(oh: u32, ow: u32) -> PoolWin {
    let Hh = i32(U(1u));
    let W = i32(U(2u));
    let kH = i32(F(0u));
    let kW = i32(F(1u));
    let sH = i32(F(2u));
    let sW = i32(F(3u));
    let pH = i32(F(4u));
    let pW = i32(F(5u));
    let h0 = i32(oh) * sH - pH;
    let w0 = i32(ow) * sW - pW;
    let h1 = h0 + kH;
    let w1 = w0 + kW;
    let hp = min(h1, Hh + pH);
    let wp = min(w1, W + pW);
    return PoolWin(u32(max(h0, 0)), u32(min(h1, Hh)), u32(max(w0, 0)), u32(min(w1, W)), u32((hp - h0) * (wp - w0)));
}
// The (first) largest value's index in window r of plane base (element offset).
fn pool_argmax(base: u32, r: PoolWin) -> u32 {
    let W = U(2u);
    var best = r.h0 * W + r.w0;
    var bv = ld(0u, base + best);
    for (var y = r.h0; y < r.h1; y++) {
        for (var x = r.w0; x < r.w1; x++) {
            let v = ld(0u, base + y * W + x);
            if (v > bv) { bv = v; best = y * W + x; }
        }
    }
    return best;
}
fn pool_div(r: PoolWin) -> f32 {
    if (U(5u) == 1u) { return f32(r.padded); }
    return f32((r.h1 - r.h0) * (r.w1 - r.w0));
}
// p0 x, p1 out [batch, C*oH*oW] (forward); p0 x, p1 dout, p2 dx (backward:
// u6 = 1, atomics)                                            n = batch*C*oH*oW
@compute @workgroup_size(256)
fn dk_pool(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Hh = U(1u);
    let W = U(2u);
    let oH = U(3u);
    let oW = U(4u);
    let mode = U(5u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let t = i / oW;
        let ow = i - t * oW;
        let pl = t / oH;
        let oh = t - pl * oH;
        let r = pool_win(oh, ow);
        let base = pl * Hh * W;
        if (mode == 0u) {
            let best = pool_argmax(base, r);
            if (U(6u) != 0u) { atadd(2u, base + best, ld(1u, i)); } else { st(1u, i, ld(0u, base + best)); }
        } else {
            let div = pool_div(r);
            if (U(6u) != 0u) {
                let share = ld(1u, i) / div;
                for (var y = r.h0; y < r.h1; y++) { for (var x = r.w0; x < r.w1; x++) { atadd(2u, base + y * W + x, share); } }
            } else {
                var acc = 0.0;
                for (var y = r.h0; y < r.h1; y++) { for (var x = r.w0; x < r.w1; x++) { acc += ld(0u, base + y * W + x); } }
                st(1u, i, acc / div);
            }
        }
    }
}
// The pooling's backward as a gather (no atomics, any element type). p0 x,
// p1 dout, p2 dx; geometry as dk_pool; u6 = 1: dx was an unwritten zero
// gradient                                                    n = planes*H*W
@compute @workgroup_size(256)
fn dk_pool_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Hh = U(1u);
    let W = U(2u);
    let oH = U(3u);
    let oW = U(4u);
    let mode = U(5u);
    let kH = i32(F(0u));
    let kW = i32(F(1u));
    let sH = i32(F(2u));
    let sW = i32(F(3u));
    let pH = i32(F(4u));
    let pW = i32(F(5u));
    let tiled = kH == sH && kW == sW && pH == 0 && pW == 0;
    if (tiled && (U(6u) == 0u || (oH * u32(kH) == Hh && oW * u32(kW) == W))) {
        // non-overlapping windows partition the inputs: a thread per window
        let planes = N() / (Hh * W);
        for (var i = gid.x; i < planes * oH * oW; i += nwg.x * 256u) {
            let t = i / oW;
            let ow = i - t * oW;
            let pl = t / oH;
            let oh = t - pl * oH;
            let r = pool_win(oh, ow);
            let base = pl * Hh * W;
            let d = ld(1u, i);
            if (mode == 0u) {
                let best = pool_argmax(base, r);
                if (U(6u) != 0u) {
                    for (var yy = r.h0; yy < r.h1; yy++) {
                        for (var xx = r.w0; xx < r.w1; xx++) { st(2u, base + yy * W + xx, select(0.0, d, yy * W + xx == best)); }
                    }
                } else { st(2u, base + best, ld(2u, base + best) + d); }
            } else {
                let share = d / pool_div(r);
                for (var yy = r.h0; yy < r.h1; yy++) { for (var xx = r.w0; xx < r.w1; xx++) { accum(2u, base + yy * W + xx, share); } }
            }
        }
        return;
    }
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let t = i / W;
        let x = i - t * W;
        let pl = t / Hh;
        let y = t - pl * Hh;
        let base = pl * Hh * W;
        let dbase = pl * oH * oW;
        var oh0: i32;
        var oh1: i32;
        var ow0: i32;
        var ow1: i32;
        if (tiled) {
            // non-overlapping windows: exactly one holds (y, x)
            oh0 = i32(y / u32(kH)); oh1 = oh0;
            ow0 = i32(x / u32(kW)); ow1 = ow0;
        } else {
            oh0 = (i32(y) + pH - kH + sH) / sH; oh1 = (i32(y) + pH) / sH;   // windows over row y
            ow0 = (i32(x) + pW - kW + sW) / sW; ow1 = (i32(x) + pW) / sW;
            if (i32(y) + pH - kH + 1 < 0) { oh0 = 0; }
            if (i32(x) + pW - kW + 1 < 0) { ow0 = 0; }
        }
        if (oh1 > i32(oH) - 1) { oh1 = i32(oH) - 1; }
        if (ow1 > i32(oW) - 1) { ow1 = i32(oW) - 1; }
        var s = 0.0;
        for (var oh = oh0; oh <= oh1; oh++) {
            for (var ow = ow0; ow <= ow1; ow++) {
                let r = pool_win(u32(oh), u32(ow));
                if (y < r.h0 || y >= r.h1 || x < r.w0 || x >= r.w1) { continue; }
                let d = ld(1u, dbase + u32(oh) * oW + u32(ow));
                if (mode == 0u) {
                    if (pool_argmax(base, r) == y * W + x) { s += d; }
                } else {
                    s += d / pool_div(r);
                }
            }
        }
        accum(2u, i, s);
    }
}
// Global pooling, one plane of u0 values per unit: p0 x, p1 out [planes];
// backward (u2 = 1): p1 dout, p2 dx. u1 = 0 max, 1 avg           n = planes*256
@compute @workgroup_size(256)
fn dk_gpool(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u,
            @builtin(num_workgroups) nwg: vec3u) {
    let Pp = U(0u);
    for (var pl = wg.x; pl < N() / 256u; pl += nwg.x) {
        let b = pl * Pp;
        if (U(1u) != 0u) {
            if (U(2u) != 0u) {
                let share = ld(1u, pl) / f32(Pp);
                for (var q = li; q < Pp; q += 256u) { accum(2u, b + q, share); }
            } else {
                var s = 0.0;
                for (var q = li; q < Pp; q += 256u) { s += ld(0u, b + q); }
                s = block_sum(s, li);
                if (li == 0u) { st(1u, pl, s / f32(Pp)); }
            }
        } else if (li == 0u) {
            var best = 0u;
            for (var q = 1u; q < Pp; q++) { if (ld(0u, b + q) > ld(0u, b + best)) { best = q; } }
            if (U(2u) != 0u) { st(2u, b + best, ld(2u, b + best) + ld(1u, pl)); } else { st(1u, pl, ld(0u, b + best)); }
        }
    }
}

// ── Upsampling and resizing (planes of H x W -> oH x oW) ─────────────────────
// Nearest by an integer factor u2: forward (u3 = 0) p0 x, p1 out, one thread
// per output; backward (u3 = 1) p0 dout, p1 dx, one per input. u0 W, u1 H
@compute @workgroup_size(256)
fn dk_upsample(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let W = U(0u);
    let Hh = U(1u);
    let f = U(2u);
    let oW = W * f;
    let oH = Hh * f;
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        if (U(3u) == 0u) {
            let ow = i % oW;
            let oh = (i / oW) % oH;
            let pl = i / (oW * oH);
            st(1u, i, ld(0u, (pl * Hh + oh / f) * W + ow / f));
        } else {
            let w = i % W;
            let h = (i / W) % Hh;
            let pl = i / (W * Hh);
            var s = 0.0;
            for (var y = 0u; y < f; y++) { for (var x = 0u; x < f; x++) { s += ld(0u, (pl * oH + h * f + y) * oW + w * f + x); } }
            accum(1u, i, s);
        }
    }
}
// Up to four taps per axis (unused ones weigh 0).
struct Taps { i0: u32, i1: u32, i2: u32, i3: u32, w0: f32, w1: f32, w2: f32, w3: f32 }
fn rs_src(o: u32, inn: u32, out: u32, align: u32, clampZero: bool) -> f32 {
    if (align != 0u) {
        if (out <= 1u) { return 0.0; }
        return f32(o) * f32(inn - 1u) / f32(out - 1u);
    }
    var s = (f32(o) + 0.5) * f32(inn) / f32(out) - 0.5;
    if (clampZero && s < 0.0) { s = 0.0; }
    return s;
}
fn rs_clamp(j: i32, inn: u32) -> u32 {
    if (j < 0) { return 0u; }
    if (j >= i32(inn)) { return inn - 1u; }
    return u32(j);
}
fn rs_taps(o: u32, inn: u32, out: u32, mode: u32, align: u32) -> Taps {
    var t = Taps(0u, 0u, 0u, 0u, 0.0, 0.0, 0.0, 0.0);
    if (mode == 0u) {
        let i = u32(floor(f32(o) * (f32(inn) / f32(out))));
        let c = min(i, inn - 1u);
        t.i0 = c; t.i1 = c; t.i2 = c; t.i3 = c; t.w0 = 1.0;
    } else if (mode == 1u) {
        let s = rs_src(o, inn, out, align, true);
        let i0 = min(u32(s), inn - 1u);
        let l = s - f32(i0);
        let i1 = select(inn - 1u, i0 + 1u, i0 + 1u < inn);
        t.i0 = i0; t.i1 = i1; t.i2 = i1; t.i3 = i1;
        t.w0 = 1.0 - l; t.w1 = l;
    } else {
        let s = rs_src(o, inn, out, align, false);
        let f = floor(s);
        let x = s - f;
        let Aa = -0.75;
        let x1 = x + 1.0;
        let x2 = 1.0 - x;
        let x3 = 2.0 - x;
        t.w0 = ((Aa * x1 - 5.0 * Aa) * x1 + 8.0 * Aa) * x1 - 4.0 * Aa;
        t.w1 = ((Aa + 2.0) * x - (Aa + 3.0)) * x * x + 1.0;
        t.w2 = ((Aa + 2.0) * x2 - (Aa + 3.0)) * x2 * x2 + 1.0;
        t.w3 = ((Aa * x3 - 5.0 * Aa) * x3 + 8.0 * Aa) * x3 - 4.0 * Aa;
        let i = i32(f);
        t.i0 = rs_clamp(i - 1, inn); t.i1 = rs_clamp(i, inn); t.i2 = rs_clamp(i + 1, inn); t.i3 = rs_clamp(i + 2, inn);
    }
    return t;
}
// sum_k w_k row[i_k] (row: an element offset into argument 0)
fn rs_dot(t: Taps, row: u32) -> f32 {
    var s = 0.0;
    if (t.w0 != 0.0) { s += t.w0 * ld(0u, row + t.i0); }
    if (t.w1 != 0.0) { s += t.w1 * ld(0u, row + t.i1); }
    if (t.w2 != 0.0) { s += t.w2 * ld(0u, row + t.i2); }
    if (t.w3 != 0.0) { s += t.w3 * ld(0u, row + t.i3); }
    return s;
}
// convnet.h's RESIZE_* (u4), alignCorners u5: p0 x, p1 out. u0 H, u1 W, u2 oH,
// u3 oW                                                       n = planes*oH*oW
@compute @workgroup_size(256)
fn dk_resize(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Hh = U(0u);
    let W = U(1u);
    let oH = U(2u);
    let oW = U(3u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let ow = i % oW;
        let oh = (i / oW) % oH;
        let pl = i / (oW * oH);
        let ty = rs_taps(oh, Hh, oH, U(4u), U(5u));
        let tx = rs_taps(ow, W, oW, U(4u), U(5u));
        let img = pl * Hh * W;
        var acc = 0.0;
        if (ty.w0 != 0.0) { acc += ty.w0 * rs_dot(tx, img + ty.i0 * W); }
        if (ty.w1 != 0.0) { acc += ty.w1 * rs_dot(tx, img + ty.i1 * W); }
        if (ty.w2 != 0.0) { acc += ty.w2 * rs_dot(tx, img + ty.i2 * W); }
        if (ty.w3 != 0.0) { acc += ty.w3 * rs_dot(tx, img + ty.i3 * W); }
        st(1u, i, acc);
    }
}
// The weight with which output o's taps read input i along one axis (0: none).
fn rs_weight(o: u32, i: u32, inn: u32, out: u32, mode: u32, align: u32) -> f32 {
    let t = rs_taps(o, inn, out, mode, align);
    return select(0.0, t.w0, t.i0 == i) + select(0.0, t.w1, t.i1 == i) + select(0.0, t.w2, t.i2 == i) +
           select(0.0, t.w3, t.i3 == i);
}
// The outputs that can read input i: [lo, hi] (see device_kernels.cu).
fn rs_range(i: u32, inn: u32, out: u32, mode: u32, align: u32) -> vec2i {
    var a0: f32;
    var a1: f32;
    let lo_s = select(f32(i) - 2.0, f32(i) - 1.0, mode == 1u);
    let hi_s = select(f32(i) + 2.0, f32(i) + 1.0, mode == 1u);
    if (mode == 0u) {
        let r = f32(out) / f32(inn);
        a0 = f32(i) * r;
        a1 = f32(i + 1u) * r;
    } else if (align != 0u) {
        var r = f32(out);
        if (inn > 1u) { r = f32(out - 1u) / f32(inn - 1u); }
        a0 = lo_s * r;
        a1 = hi_s * r;
    } else {
        let r = f32(out) / f32(inn);
        a0 = (lo_s + 0.5) * r - 0.5;
        a1 = (hi_s + 0.5) * r - 0.5;
    }
    let l = i32(floor(a0)) - 1;
    let h = i32(ceil(a1)) + 1;
    return vec2i(max(l, 0), min(h, i32(out) - 1));
}
// The resize's backward as two gathers (no atomics), one axis at a time:
// pass 0 (f0 = 0): t[pl, oy, x] = sum over ox reading x of dout[pl, oy, ox] wx;
// pass 1 (f0 = 1): dx[pl, y, x] (+)= sum over oy reading y of t[pl, oy, x] wy.
// p0 source, p1 destination; u0 H, u1 W, u2 oH, u3 oW, u4 mode, u5 align
@compute @workgroup_size(256)
fn dk_resize_bwd(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let Hh = U(0u);
    let W = U(1u);
    let oH = U(2u);
    let oW = U(3u);
    let mode = U(4u);
    let align = U(5u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let x = i % W;
        let r = i / W;
        var s = 0.0;
        if (F(0u) == 0.0) {
            // r = pl * oH + oy: a row of dout
            let q = rs_range(x, W, oW, mode, align);
            for (var ox = q.x; ox <= q.y; ox++) {
                let w = rs_weight(u32(ox), x, W, oW, mode, align);
                if (w != 0.0) { s += w * ld(0u, r * oW + u32(ox)); }
            }
            st(1u, i, s);
        } else {
            let y = r % Hh;
            let pl = r / Hh;
            let q = rs_range(y, Hh, oH, mode, align);
            for (var oy = q.x; oy <= q.y; oy++) {
                let w = rs_weight(u32(oy), y, Hh, oH, mode, align);
                if (w != 0.0) { s += w * ld(0u, pl * oH * W + x + u32(oy) * W); }
            }
            accum(1u, i, s);
        }
    }
}

// ── The dynamic loss scale ───────────────────────────────────────────────────
// State (4 floats): [scale, 1/scale, overflow flag, good steps].
// g *= 1/scale, the flag raised on a non-finite value: p0 g, p1 state   n = values
@compute @workgroup_size(256)
fn dk_amp_unscale(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let inv = ld(1u, 1u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let v = ld(0u, i) * inv;
        if (!finite(v)) { st(1u, 2u, 1.0); }
        st(0u, i, v);
    }
}
// Every gradient of a table at once: p0 = u1 entries {g, n, start} (64-bit;
// low words), p1 state                                        n = total values
@compute @workgroup_size(256)
fn dk_amp_unscale_multi(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let T = P(0u) >> 2u;
    let cnt = U(1u);
    let inv = ld(1u, 1u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        var lo = 0u;
        var hi = cnt - 1u;
        while (lo < hi) {
            let mid = (lo + hi + 1u) / 2u;
            if (H[T + mid * 6u + 4u] <= i) { lo = mid; } else { hi = mid - 1u; }
        }
        let g = (H[T + lo * 6u] >> 2u) + i - H[T + lo * 6u + 4u];
        let v = rdw(g) * inv;
        if (!finite(v)) { st(1u, 2u, 1.0); }
        wrw(g, v);
    }
}
// After a step: an overflow halves the scale, u0 good steps in a row double
// it; the flag is cleared. p0 state                             n = 1
@compute @workgroup_size(256)
fn dk_amp_update(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x == 0u) {
        if (ld(0u, 2u) != 0.0) { st(0u, 0u, max(ld(0u, 0u) * 0.5, 1.0)); st(0u, 3u, 0.0); }
        else {
            st(0u, 3u, ld(0u, 3u) + 1.0);
            if (ld(0u, 3u) >= f32(U(0u))) { st(0u, 0u, min(ld(0u, 0u) * 2.0, 16777216.0)); st(0u, 3u, 0.0); }
        }
        st(0u, 1u, 1.0 / ld(0u, 0u));
        st(0u, 2u, 0.0);
    }
}

// ── Sparse (CSR) x dense: sparse.h ───────────────────────────────────────────
// out [rows, n] (+)= S . X; p0 rowPtr, p1 colIdx (float indices), p2 values,
// p3 X, p4 out; u0 = n, u1 = 1 to accumulate                  n = rows*ncols
@compute @workgroup_size(256)
fn dk_spmm(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let NC = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let r = i / NC;
        let j = i % NC;
        let lo = u32(ld(0u, r));
        let hi = u32(ld(0u, r + 1u));
        var s = 0.0;
        for (var e = lo; e < hi; e++) { s += ld(2u, e) * ld(3u, u32(ld(1u, e)) * NC + j); }
        if (U(1u) != 0u) { st(4u, i, ld(4u, i) + s); } else { st(4u, i, s); }
    }
}
// dX[cols, n] += S^T . G: one thread per (nonzero, column); p5 = row of
// each nonzero                                                n = nnz*ncols
@compute @workgroup_size(256)
fn dk_spmm_tgrad(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let NC = U(0u);
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let e = i / NC;
        let j = i % NC;
        let r = u32(ld(5u, e));
        let col = u32(ld(1u, e));
        atadd(4u, col * NC + j, ld(2u, e) * ld(3u, r * NC + j));
    }
}
// dvalues[e] += G[row(e), :] . X[col(e), :]; p1 colIdx, p3 X, p4 G, p5 row
// of each nonzero, p6 dvalues                                 n = nnz
@compute @workgroup_size(256)
fn dk_spmm_vgrad(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let NC = U(0u);
    for (var e = gid.x; e < N(); e += nwg.x * 256u) {
        let r = u32(ld(5u, e));
        let col = u32(ld(1u, e));
        var s = 0.0;
        for (var j = 0u; j < NC; j++) { s += ld(4u, r * NC + j) * ld(3u, col * NC + j); }
        st(6u, e, ld(6u, e) + s);
    }
}

// ── WebGPU only: copies, GEMMs, convolutions ─────────────────────────────────
// dst = src for n 32-bit words (bit-exact): p0 src, p2 dst, u7 = n. Either
// may start mid-word (a 16-bit view): then it moves 2n halves.
@compute @workgroup_size(256)
fn wk_copy(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let s = P(0u);
    let d = P(2u);
    if (((s | d) & 3u) == 0u) {
        for (var i = gid.x; i < N(); i += nwg.x * 256u) { H[(d >> 2u) + i] = H[(s >> 2u) + i]; }
        return;
    }
    for (var i = gid.x; i < 2u * N(); i += nwg.x * 256u) {
        let a = s + 2u * i;
        let h = (H[a >> 2u] >> ((a & 2u) * 8u)) & 0xffffu;
        let b = d + 2u * i;
        let sh = (b & 2u) * 8u;
        atomicAnd(&HA[b >> 2u], ~(0xffffu << sh));
        atomicOr(&HA[b >> 2u], h << sh);
    }
}

// gpu_webgpu.h's elementwise ops with no DK_* kernel: p0 a, p1 b, p2 out,
// u0 = 0 a / b, 1 pow(a, b), 2 sqrt(a)                        n = elements
@compute @workgroup_size(256)
fn wk_map(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    for (var i = gid.x; i < N(); i += nwg.x * 256u) {
        let a = ld(0u, i);
        var v = 0.0;
        if (U(0u) == 0u) { v = a / ld(1u, i); }
        else if (U(0u) == 1u) { v = pow(a, ld(1u, i)); }
        else { v = sqrt(a); }
        st(2u, i, v);
    }
}

// A strided batch of c_i = alpha op(a_i) op(b_i) + beta c_i (device.h's
// DevGemm), row-major. Words: 0 a, 1 b, 2 c (byte offsets), 3 m, 4 n, 5 k,
// 6 lda, 7 ldb, 8 ldc, 9 sa, 10 sb, 11 sc (batch strides in elements),
// 12 alpha, 13 beta, 14 the first batch index of this dispatch. The
// pipeline constants fix the transposes and the element types (GAT: a and
// b's, GCT: c's), accumulation is fp32. A workgroup of 256 threads takes a
// 64 x 64 tile of c, k in steps of 16 through workgroup memory, each thread
// a 4 x 4 block of it; the grid is (n / 64, m / 64, batch). The tiles are
// scalar arrays: threads storing different components of one vec4 in
// workgroup memory race on Metal (the store rewrites the whole vector).
override GTA: bool = false;
override GTB: bool = false;
override GAT: u32 = 0u;
override GCT: u32 = 0u;
var<workgroup> gA: array<f32, 1024>;   // [k 16][m 64]
var<workgroup> gB: array<f32, 1024>;   // [k 16][n 64]
fn gld(base: u32, t: u32, i: u32) -> f32 {
    if (t == 0u) { return bitcast<f32>(H[(base >> 2u) + i]); }
    let a = base + 2u * i;
    return h2f((H[a >> 2u] >> ((a & 2u) * 8u)) & 0xffffu, t);
}
fn gst1(base: u32, t: u32, i: u32, v: f32) {
    if (t == 0u) { H[(base >> 2u) + i] = bitcast<u32>(v); return; }
    let a = base + 2u * i;
    let s = (a & 2u) * 8u;
    atomicAnd(&HA[a >> 2u], ~(0xffffu << s));
    atomicOr(&HA[a >> 2u], f2h(v, t) << s);
}
@compute @workgroup_size(256)
fn wk_gemm(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
    let M = aw(3u);
    let Nn = aw(4u);
    let K = aw(5u);
    let lda = aw(6u);
    let ldb = aw(7u);
    let ldc = aw(8u);
    let z = wg.z + aw(14u);
    let ea = select(2u, 4u, GAT == 0u);
    let ec = select(2u, 4u, GCT == 0u);
    let Ab = aw(0u) + z * aw(9u) * ea;
    let Bb = aw(1u) + z * aw(10u) * ea;
    let Cb = aw(2u) + z * aw(11u) * ec;
    let m0 = wg.y * 64u;
    let n0 = wg.x * 64u;
    let tx = li % 16u;
    let ty = li / 16u;
    var acc: array<vec4f, 4>;
    for (var k0 = 0u; k0 < K; k0 += 16u) {
        for (var q = 0u; q < 4u; q++) {
            let e = li + q * 256u;
            var mm: u32;
            var kk: u32;
            if (GTA) { mm = e % 64u; kk = e / 64u; } else { kk = e % 16u; mm = e / 16u; }
            let gm = m0 + mm;
            let gk = k0 + kk;
            var v = 0.0;
            if (gm < M && gk < K) { v = gld(Ab, GAT, select(gm * lda + gk, gk * lda + gm, GTA)); }
            gA[kk * 64u + mm] = v;
        }
        for (var q = 0u; q < 4u; q++) {
            let e = li + q * 256u;
            var nn: u32;
            var kk: u32;
            if (GTB) { kk = e % 16u; nn = e / 16u; } else { nn = e % 64u; kk = e / 64u; }
            let gn = n0 + nn;
            let gk = k0 + kk;
            var v = 0.0;
            if (gn < Nn && gk < K) { v = gld(Bb, GAT, select(gk * ldb + gn, gn * ldb + gk, GTB)); }
            gB[kk * 64u + nn] = v;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let ai = kk * 64u + ty * 4u;
            let bi = kk * 64u + tx * 4u;
            let a = vec4f(gA[ai], gA[ai + 1u], gA[ai + 2u], gA[ai + 3u]);
            let b = vec4f(gB[bi], gB[bi + 1u], gB[bi + 2u], gB[bi + 3u]);
            acc[0] += a.x * b;
            acc[1] += a.y * b;
            acc[2] += a.z * b;
            acc[3] += a.w * b;
        }
        workgroupBarrier();
    }
    let alpha = bitcast<f32>(aw(12u));
    let beta = bitcast<f32>(aw(13u));
    for (var i = 0u; i < 4u; i++) {
        let gm = m0 + ty * 4u + i;
        if (gm >= M) { break; }
        for (var j = 0u; j < 4u; j++) {
            let gn = n0 + tx * 4u + j;
            if (gn >= Nn) { break; }
            let idx = gm * ldc + gn;
            var v = alpha * acc[i][j];
            if (beta != 0.0) { v += beta * gld(Cb, GCT, idx); }   // beta 0: c is not read
            gst1(Cb, GCT, idx, v);
        }
    }
}

// Convolutions as implicit GEMMs (WebGpuBackend::conv: MIOpen / cuDNN's
// role), the column matrix gathered tile by tile instead of written out by
// im2col. In a plain convolution's terms -- input I [B, Ci, Hi, Wi], weights
// W [Co, Ci, kH, kW], output O [B, Co, Ho, Wo]; a transposed convolution is
// one with I and O swapped --
//   COP 0  O = conv(I, W):       m = co, n = (b, oy, ox), k = (ci, ky, kx)
//   COP 1  I = conv^T(O, W):     m = ci, n = (b, y, x),   k = (co, ky, kx),
//          a gather over the windows holding (y, x), no atomics
//   COP 2  W += O . I-columns:   m = co, n = (ci, ky, kx), k = (b, oy, ox),
//          split over k (grid z, kSplit each), added atomically (fp32)
// Words: 0 x (I), 1 w, 2 y (O) byte offsets, 3 B, 4 Ci, 5 Hi, 6 Wi, 7 Co,
// 8 Ho, 9 Wo, 10 kH, 11 kW, 12 sH, 13 sW, 14 pH, 15 pW, 16 dH, 17 dW,
// 18 kSplit. A workgroup of 256 threads takes a CBM x (4096 / CBM) tile of
// the product (64 x 64, 32 x 128 or 16 x 256: few output channels take the
// flat ones), k in steps of 16, each thread a 4 x 4 block; each thread
// loads the same column of the B tile every step, so its pixel is decoded
// once. CT is I, W and O's element type (COP 2: fp32).
override COP: u32 = 0u;
override CBM: u32 = 64u;
override CT: u32 = 0u;
var<workgroup> cA: array<f32, 16u * CBM>;            // [k 16][m CBM]
var<workgroup> cB: array<f32, 16u * (4096u / CBM)>;  // [k 16][n BN]
struct Cv { x: u32, w: u32, y: u32, B: u32, Ci: u32, Hi: u32, Wi: u32, Co: u32, Ho: u32, Wo: u32,
            kH: u32, kW: u32, sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32, kSplit: u32 }
fn cv_args() -> Cv {
    return Cv(aw(0u), aw(1u), aw(2u), aw(3u), aw(4u), aw(5u), aw(6u), aw(7u), aw(8u), aw(9u), aw(10u), aw(11u),
              aw(12u), aw(13u), aw(14u), aw(15u), aw(16u), aw(17u), aw(18u));
}
// A[m][k]: the weights (COP 0, 1) or the output gradient (COP 2)
fn cv_a(c: Cv, m: u32, k: u32) -> f32 {
    let KK = c.kH * c.kW;
    if (COP == 0u) { return gld(c.w, CT, m * c.Ci * KK + k); }
    if (COP == 1u) { return gld(c.w, CT, ((k / KK) * c.Ci + m) * KK + k % KK); }
    let Po = c.Ho * c.Wo;
    let b = k / Po;
    return gld(c.y, 0u, (b * c.Co + m) * Po + (k - b * Po));
}
// B[k][n] for the thread's column: its decoded pixel / tap (p0, p1, p2) and
// image base (ib, an element offset)
fn cv_b(c: Cv, k: u32, p0: i32, p1: i32, p2: u32, ib: u32) -> f32 {
    let KK = c.kH * c.kW;
    if (COP == 0u) {
        // p0, p1: the window's corner (oy sH - pH, ox sW - pW)
        let ci = k / KK;
        let r = k - ci * KK;
        let ky = r / c.kW;
        let kx = r - ky * c.kW;
        let y = p0 + i32(ky * c.dH);
        let x = p1 + i32(kx * c.dW);
        if (y < 0 || y >= i32(c.Hi) || x < 0 || x >= i32(c.Wi)) { return 0.0; }
        return gld(c.x, CT, ib + (ci * c.Hi + u32(y)) * c.Wi + u32(x));
    }
    if (COP == 1u) {
        // p0, p1: (y + pH, x + pW); the window at (oy, ox) holds (y, x) through tap (ky, kx)
        let co = k / KK;
        let r = k - co * KK;
        let ky = r / c.kW;
        let kx = r - ky * c.kW;
        let ny = p0 - i32(ky * c.dH);
        let nx = p1 - i32(kx * c.dW);
        if (ny < 0 || nx < 0) { return 0.0; }
        let oy = u32(ny) / c.sH;
        let ox = u32(nx) / c.sW;
        if (oy * c.sH != u32(ny) || ox * c.sW != u32(nx) || oy >= c.Ho || ox >= c.Wo) { return 0.0; }
        return gld(c.y, CT, ib + (co * c.Ho + oy) * c.Wo + ox);
    }
    // COP 2: p0, p1: the tap's offset (ky dH - pH, kx dW - pW), p2 = ci
    let Po = c.Ho * c.Wo;
    let b = k / Po;
    let op = k - b * Po;
    let oy = op / c.Wo;
    let ox = op - oy * c.Wo;
    let y = i32(oy * c.sH) + p0;
    let x = i32(ox * c.sW) + p1;
    if (y < 0 || y >= i32(c.Hi) || x < 0 || x >= i32(c.Wi)) { return 0.0; }
    return gld(c.x, 0u, ((b * c.Ci + p2) * c.Hi + u32(y)) * c.Wi + u32(x));
}
@compute @workgroup_size(256)
fn wk_conv(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
    let c = cv_args();
    let BN = 4096u / CBM;
    let KK = c.kH * c.kW;
    let Po = c.Ho * c.Wo;
    let Pi = c.Hi * c.Wi;
    var M = c.Co;
    var Ntot = c.B * Po;
    var kBeg = 0u;
    var kEnd = c.Ci * KK;
    if (COP == 1u) { M = c.Ci; Ntot = c.B * Pi; kEnd = c.Co * KK; }
    if (COP == 2u) { Ntot = c.Ci * KK; kBeg = wg.z * c.kSplit; kEnd = min(c.B * Po, kBeg + c.kSplit); }
    let m0 = wg.y * CBM;
    let n0 = wg.x * BN;
    // this thread's column of the B tile, decoded once
    let bc = li % BN;
    let n = n0 + bc;
    let nIn = n < Ntot;
    let nc = min(n, Ntot - 1u);
    var p0 = 0;
    var p1 = 0;
    var p2 = 0u;
    var ib = 0u;
    if (COP == 0u) {
        let b = nc / Po;
        let op = nc - b * Po;
        let oy = op / c.Wo;
        p0 = i32(oy * c.sH) - i32(c.pH);
        p1 = i32((op - oy * c.Wo) * c.sW) - i32(c.pW);
        ib = b * c.Ci * Pi;
    } else if (COP == 1u) {
        let b = nc / Pi;
        let ip = nc - b * Pi;
        let y = ip / c.Wi;
        p0 = i32(y + c.pH);
        p1 = i32(ip - y * c.Wi + c.pW);
        ib = b * c.Co * Po;
    } else {
        p2 = nc / KK;
        let r = nc - p2 * KK;
        let ky = r / c.kW;
        p0 = i32(ky * c.dH) - i32(c.pH);
        p1 = i32((r - ky * c.kW) * c.dW) - i32(c.pW);
    }
    let tx = li % (BN / 4u);
    let ty = li / (BN / 4u);
    var acc: array<vec4f, 4>;
    for (var k0 = kBeg; k0 < kEnd; k0 += 16u) {
        for (var q = 0u; q < CBM / 16u; q++) {
            let e = li + q * 256u;
            let kk = e % 16u;
            let mm = e / 16u;
            var v = 0.0;
            if (m0 + mm < M && k0 + kk < kEnd) { v = cv_a(c, m0 + mm, k0 + kk); }
            cA[kk * CBM + mm] = v;
        }
        for (var r = li / BN; r < 16u; r += 256u / BN) {
            var v = 0.0;
            if (nIn && k0 + r < kEnd) { v = cv_b(c, k0 + r, p0, p1, p2, ib); }
            cB[r * BN + bc] = v;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let ai = kk * CBM + ty * 4u;
            let bi = kk * BN + tx * 4u;
            let a = vec4f(cA[ai], cA[ai + 1u], cA[ai + 2u], cA[ai + 3u]);
            let b = vec4f(cB[bi], cB[bi + 1u], cB[bi + 2u], cB[bi + 3u]);
            acc[0] += a.x * b;
            acc[1] += a.y * b;
            acc[2] += a.z * b;
            acc[3] += a.w * b;
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i++) {
        let gm = m0 + ty * 4u + i;
        if (gm >= M) { break; }
        for (var j = 0u; j < 4u; j++) {
            let gn = n0 + tx * 4u + j;
            if (gn >= Ntot) { break; }
            let v = acc[i][j];
            if (COP == 0u) {
                let b = gn / Po;
                gst1(c.y, CT, (b * c.Co + gm) * Po + gn - b * Po, v);
            } else if (COP == 1u) {
                let b = gn / Pi;
                gst1(c.x, CT, (b * c.Ci + gm) * Pi + gn - b * Pi, v);
            } else {
                atomic_addf((c.w >> 2u) + gm * Ntot + gn, v);
            }
        }
    }
}
