// SafeC Standard Library — S24: one packed 24-bit audio sample.
//
// AUDIO_FORMAT_I24 buffers hold signed 24-bit samples packed in 3 bytes,
// little-endian (no padding byte). S24 is exactly that layout —
// sizeof(struct S24) == 3, alignment 1 — so a packed buffer is a []S24:
//
//   []S24 out = io.output_i24();             // frames * channels samples
//   out[i] = S24::from_f32(0.5f);
//   int raw = out[i].value();                // -8388608 .. 8388607
//
// and audio_convert.h's audio_f32_to_i24 / audio_i24_to_f32 convert whole
// slices of them. Freestanding-safe.
#pragma once

#define S24_MIN (-8388608)
#define S24_MAX 8388607

namespace std {

struct S24 {
    unsigned char b[3];   // two's complement, least significant byte first

    // The sample with value 'v', clamped to [S24_MIN, S24_MAX].
    static struct S24 of(int v);
    // A float in [-1, 1] scaled by 2^23 - 1, rounded to nearest and clamped
    // (out-of-range values saturate).
    static struct S24 from_f32(float v);

    int   value() const;          // sign-extended, S24_MIN .. S24_MAX
    float to_f32() const;         // value() / 2^23: S24_MIN is exactly -1.0
    void  set(int v);             // clamped, as of()
};

} // namespace std
