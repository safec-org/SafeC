// SafeC Standard Library — Window Functions
//
// Symmetric N-point analysis windows for spectral work (tapering a frame
// before an FFT to reduce spectral leakage) and FIR filter design (window
// method). All four write into a caller-provided 'w' array of length n;
// none allocate.
#pragma once

namespace std {

void window_rectangular([]double w); // w[i] = 1 (no tapering — included for a uniform interface, and as an FFT/leakage baseline to compare the others against)
void window_hann([]double w);         // w[i] = 0.5*(1 - cos(2*pi*i/(n-1)))
void window_hamming([]double w);       // w[i] = 0.54 - 0.46*cos(2*pi*i/(n-1))
void window_blackman([]double w);      // w[i] = 0.42 - 0.5*cos(2*pi*i/(n-1)) + 0.08*cos(4*pi*i/(n-1))

// Applies window 'w' to signal 'x' in place: x[i] *= w[i], for i in [0,n).
void window_apply([]double x, []const double w);

// Single precision: the same windows (computed in double, rounded once).
void window_rectangular_f32([]float w);
void window_hann_f32([]float w);
void window_hamming_f32([]float w);
void window_blackman_f32([]float w);
void window_apply_f32([]float x, []const float w);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
void window_rectangular_n_(&double w, unsigned long n);
void window_hann_n_(&double w, unsigned long n);
void window_hamming_n_(&double w, unsigned long n);
void window_blackman_n_(&double w, unsigned long n);
void window_rectangular_f32_n_(&float w, unsigned long n);
void window_hann_f32_n_(&float w, unsigned long n);
void window_hamming_f32_n_(&float w, unsigned long n);
void window_blackman_f32_n_(&float w, unsigned long n);
void window_apply_n_(&double x, const &double w, unsigned long n);
void window_apply_f32_n_(&float x, const &float w, unsigned long n);

} // namespace std
