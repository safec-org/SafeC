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
//
// Buffers are slices: every function works on as many samples (or frames)
// as both its input and its output hold, and returns that count — a short
// output buffer truncates, it never overruns. 24-bit samples are S24
// (std/audio/s24.h); a byte buffer of any format is []unsigned char.
#pragma once
#include <std/audio/s24.h>
#include <std/dsp/fixed.h>
#include <std/dsp/audio_buffer.h>

namespace std {

// ── element-wise: min(in.len, out.len) samples; returns that count ──────
unsigned long audio_f32_to_i16([]const float in, []short out);
unsigned long audio_i16_to_f32([]const short in, []float out);
unsigned long audio_f32_to_i32([]const float in, []int out);
unsigned long audio_i32_to_f32([]const int in, []float out);
unsigned long audio_f32_to_i8([]const float in, []signed char out);
unsigned long audio_i8_to_f32([]const signed char in, []float out);
unsigned long audio_f32_to_i24([]const float in, []S24 out);
unsigned long audio_i24_to_f32([]const S24 in, []float out);
unsigned long audio_f32_to_f64([]const float in, []double out);
unsigned long audio_f64_to_f32([]const double in, []float out);
unsigned long audio_i16_to_i32([]const short in, []int out);
unsigned long audio_i32_to_i16([]const int in, []short out);

// TPDF-dithered float -> 16-bit. 'state' is any non-zero seed, updated.
unsigned long audio_f32_to_i16_dither([]const float in, []short out, &unsigned int state);

// One sample of any AUDIO_FORMAT_* at the start of 'bytes', as double
// (0.0 if 'bytes' is shorter than one sample), and back (nothing written
// if it is).
double audio_sample_read([]const unsigned char bytes, unsigned int format);
void   audio_sample_write([]unsigned char bytes, unsigned int format, double v);

// Any format to any format (same layout): as many whole samples as both
// buffers hold. Returns the samples converted.
unsigned long audio_convert([]unsigned char dst, unsigned int dst_format,
                            []const unsigned char src, unsigned int src_format);

// ── layout ───────────────────────────────────────────────────────────────
// planar: channel-major ('frames' samples per channel, back to back).
// frames = min(planar.len, out.len) / channels; returns it.
unsigned long audio_interleave_f32([]const float planar, []float out, int channels);
unsigned long audio_deinterleave_f32([]const float in, []float planar, int channels);

// The general copy streams use: 'channels' channels of 'frames' frames,
// from 'src' (src_channels per frame, src_interleaved, src_format,
// starting at channel src_first) to 'dst' (likewise). Channels of 'dst'
// outside [dst_first, dst_first + channels) are left untouched. Returns
// 'frames', or 0 (copying nothing) when a buffer is smaller than
// frames * its channels samples or the channel ranges don't fit.
unsigned long audio_copy_frames([]unsigned char dst, unsigned int dst_format, int dst_channels,
                                int dst_interleaved, int dst_first,
                                []const unsigned char src, unsigned int src_format, int src_channels,
                                int src_interleaved, int src_first, int channels, unsigned long frames);

// ── gain / mixing helpers ────────────────────────────────────────────────
void          audio_gain_f32([]float buf, float gain);
// dst[i] += src[i] * gain for min(dst.len, src.len) samples; returns that.
unsigned long audio_mix_f32([]float dst, []const float src, float gain);
// Peak absolute value.
float         audio_peak_f32([]const float buf);

// ── std/dsp interop (Q8.24 Fixed, AudioBuffer) ───────────────────────────
unsigned long audio_f32_to_fixed([]const float in, []Fixed out);
unsigned long audio_fixed_to_f32([]const Fixed in, []float out);
// Writes in.len / channels interleaved float frames to a dsp AudioBuffer,
// or reads up to out.len / channels frames from it, converting in
// fixed-size chunks on the stack. Returns frames transferred (fewer when
// full/empty).
unsigned long audio_buffer_write_f32(&AudioBuffer buf, []const float in);
unsigned long audio_buffer_read_f32(&AudioBuffer buf, []float out);

} // namespace std
