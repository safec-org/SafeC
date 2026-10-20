#pragma once
// SafeC Standard Library — vectorized math over float arrays.
//
// y[i] = f(x[i]) for exp, tanh and the logistic sigmoid, written as
// branch-free polynomial code the compiler turns into SIMD on any target
// (SSE/AVX, NEON, WebAssembly SIMD128). Measured error against the exact
// result: under 1 ulp for exp, 1.3 ulp for tanh, 2 ulp for sigmoid,
// subnormal results included; NaN propagates and overflow gives infinity,
// as with libm. Results can differ from exp_f / tanh_f in the last bit.
//
// Arrays of 32768 or more elements are split across std/sync/parallel.h's
// thread pool, so these are not for real-time code with large arrays;
// shorter arrays run on the calling thread. x and y may be the same array.
namespace std {

void vm_exp_f32(const &float x, &float y, unsigned long n);
void vm_tanh_f32(const &float x, &float y, unsigned long n);
void vm_sigmoid_f32(const &float x, &float y, unsigned long n);

} // namespace std
