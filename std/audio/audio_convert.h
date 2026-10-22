// SafeC Standard Library — audio sample-format and layout conversion.
//
// The conversions std/audio's streams do between the format/layout a
// callback asked for and the one the device runs (see audio.h), exposed
// for application use: integer <-> float scaling, packed 24-bit, (de)
// interleaving, and the bridge to std/dsp's Q8.24 Fixed samples and its
// lock-free AudioBuffer. All of them are allocation-free and lock-free —
// safe inside an audio callback.
//
// Scaling: an N-bit integer sample s maps to s / 2^(N-1) (so -2^(N-1) is
// exactly -1.0); float -> integer multiplies by 2^(N-1) - 1, rounds to
// nearest and clamps, so out-of-range floats saturate instead of wrapping.
// Dithered variants add triangular (TPDF) dither of +-1 LSB before
// rounding, from a caller-held xorshift state.
#pragma once
#include <std/dsp/fixed.h>
#include <std/dsp/audio_buffer.h>

namespace std {

// ── element-wise (n samples) ─────────────────────────────────────────────
void audio_f32_to_i16(const &float in, &short out, unsigned long n);
void audio_i16_to_f32(const &short in, &float out, unsigned long n);
void audio_f32_to_i32(const &float in, &int out, unsigned long n);
void audio_i32_to_f32(const &int in, &float out, unsigned long n);
void audio_f32_to_i8(const &float in, &signed char out, unsigned long n);
void audio_i8_to_f32(const &signed char in, &float out, unsigned long n);
// 24-bit packed little-endian: 3 bytes per sample in 'out' / 'in'.
void audio_f32_to_i24(const &float in, &unsigned char out, unsigned long n);
void audio_i24_to_f32(const &unsigned char in, &float out, unsigned long n);
void audio_f32_to_f64(const &float in, &double out, unsigned long n);
void audio_f64_to_f32(const &double in, &float out, unsigned long n);
void audio_i16_to_i32(const &short in, &int out, unsigned long n);
void audio_i32_to_i16(const &int in, &short out, unsigned long n);

// TPDF-dithered float -> 16-bit. 'state' is any non-zero seed, updated.
void audio_f32_to_i16_dither(const &float in, &short out, unsigned long n, &unsigned int state);

// One sample of any AUDIO_FORMAT_* at byte offset 0 of 'p', as double, and back.
double audio_sample_read_raw_(const void* p, unsigned int format);
void   audio_sample_write_raw_(void* p, unsigned int format, double v);

// Any format to any format, n samples (same layout).
void audio_convert_raw_(void* dst, unsigned int dst_format, const void* src, unsigned int src_format,
                        unsigned long n);

// ── layout ───────────────────────────────────────────────────────────────
// planar: channel-major ('frames' samples per channel, back to back).
void audio_interleave_f32(const &float planar, &float out, unsigned long frames, int channels);
void audio_deinterleave_f32(const &float in, &float planar, unsigned long frames, int channels);

// The general copy streams use: 'channels' channels of 'frames' frames,
// from 'src' (src_channels per frame, src_interleaved, src_format,
// starting at channel src_first) to 'dst' (likewise). Channels of 'dst'
// outside [dst_first, dst_first + channels) are left untouched.
void audio_copy_frames_raw_(void* dst, unsigned int dst_format, int dst_channels, int dst_interleaved, int dst_first,
                            const void* src, unsigned int src_format, int src_channels, int src_interleaved,
                            int src_first, int channels, unsigned long frames);

// ── gain / mixing helpers ────────────────────────────────────────────────
void  audio_gain_f32(&float buf, unsigned long n, float gain);
void  audio_mix_f32(&float dst, const &float src, unsigned long n, float gain);
// Peak absolute value of n samples.
float audio_peak_f32(const &float buf, unsigned long n);

// ── std/dsp interop (Q8.24 Fixed, AudioBuffer) ───────────────────────────
void audio_f32_to_fixed(const &float in, &Fixed out, unsigned long n);
void audio_fixed_to_f32(const &Fixed in, &float out, unsigned long n);
// Writes/reads 'frames' interleaved float frames to/from a dsp AudioBuffer
// (its channel count must match), converting on the way in fixed-size
// chunks on the stack. Returns frames transferred (fewer when full/empty).
unsigned long audio_buffer_write_f32(&AudioBuffer buf, const &float in, unsigned long frames);
unsigned long audio_buffer_read_f32(&AudioBuffer buf, &float out, unsigned long frames);

} // namespace std
