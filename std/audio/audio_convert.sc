// SafeC Standard Library — audio sample-format and layout conversion
// (see audio_convert.h). Allocation-free and lock-free throughout.
#pragma once
#include <std/audio/audio_convert.h>
#include <std/audio/audio.h>

namespace std {

static long long audio_round_clamp_(double v, double lo, double hi) {
    if (v != v) return 0LL;               // NaN
    if (v <= lo) return (long long)lo;
    if (v >= hi) return (long long)hi;
    if (v >= 0.0) return (long long)(v + 0.5);
    return 0LL - (long long)(0.5 - v);
}

void audio_f32_to_i16(const &float in, &short out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL)
            out[i] = (short)audio_round_clamp_((double)in[i] * 32767.0, -32768.0, 32767.0);
    }
}
void audio_i16_to_f32(const &short in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (float)((double)in[i] / 32768.0);
    }
}
void audio_f32_to_i32(const &float in, &int out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL)
            out[i] = (int)audio_round_clamp_((double)in[i] * 2147483647.0, -2147483648.0, 2147483647.0);
    }
}
void audio_i32_to_f32(const &int in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (float)((double)in[i] / 2147483648.0);
    }
}
void audio_f32_to_i8(const &float in, &signed char out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL)
            out[i] = (signed char)audio_round_clamp_((double)in[i] * 127.0, -128.0, 127.0);
    }
}
void audio_i8_to_f32(const &signed char in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (float)((double)in[i] / 128.0);
    }
}
void audio_f32_to_i24(const &float in, &unsigned char out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL)
            audio_sample_write_raw_((void*)&out[i * 3UL], AUDIO_FORMAT_I24, (double)in[i]);
    }
}
void audio_i24_to_f32(const &unsigned char in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL)
            out[i] = (float)audio_sample_read_raw_((const void*)&in[i * 3UL], AUDIO_FORMAT_I24);
    }
}
void audio_f32_to_f64(const &float in, &double out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (double)in[i];
    }
}
void audio_f64_to_f32(const &double in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (float)in[i];
    }
}
void audio_i16_to_i32(const &short in, &int out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (int)in[i] * 65536;
    }
}
void audio_i32_to_i16(const &int in, &short out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            int v = in[i];
            // Round to nearest by adding half an LSB of the 16-bit result.
            long long r = ((long long)v + 32768LL) >> 16;
            if (r > 32767LL) r = 32767LL;
            out[i] = (short)r;
        }
    }
}

static unsigned int audio_xorshift_(&unsigned int s) {
    unsigned int x = *s;
    if (x == 0U) x = 0x9E3779B9U;
    x = x ^ (x << 13);
    x = x ^ (x >> 17);
    x = x ^ (x << 5);
    *s = x;
    return x;
}

void audio_f32_to_i16_dither(const &float in, &short out, unsigned long n, &unsigned int state) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            // TPDF: the sum of two uniform [-0.5, 0.5) LSB values.
            double a = (double)(audio_xorshift_(state) >> 8) / 16777216.0 - 0.5;
            double b = (double)(audio_xorshift_(state) >> 8) / 16777216.0 - 0.5;
            out[i] = (short)audio_round_clamp_((double)in[i] * 32767.0 + a + b, -32768.0, 32767.0);
        }
    }
}

double audio_sample_read_raw_(const void* p, unsigned int format) {
    unsafe {
        if (format == AUDIO_FORMAT_F32) return (double)*(const float*)p;
        if (format == AUDIO_FORMAT_F64) return *(const double*)p;
        if (format == AUDIO_FORMAT_I16) return (double)*(const short*)p / 32768.0;
        if (format == AUDIO_FORMAT_I32) return (double)*(const int*)p / 2147483648.0;
        if (format == AUDIO_FORMAT_I8) return (double)*(const signed char*)p / 128.0;
        if (format == AUDIO_FORMAT_I24) {
            const unsigned char* b = (const unsigned char*)p;
            int v = (int)b[0] | ((int)b[1] << 8) | ((int)b[2] << 16);
            if ((v & 0x800000) != 0) v = v - 0x1000000;
            return (double)v / 8388608.0;
        }
    }
    return 0.0;
}

void audio_sample_write_raw_(void* p, unsigned int format, double v) {
    unsafe {
        if (format == AUDIO_FORMAT_F32) { *(float*)p = (float)v; return; }
        if (format == AUDIO_FORMAT_F64) { *(double*)p = v; return; }
        if (format == AUDIO_FORMAT_I16) { *(short*)p = (short)audio_round_clamp_(v * 32767.0, -32768.0, 32767.0); return; }
        if (format == AUDIO_FORMAT_I32) {
            *(int*)p = (int)audio_round_clamp_(v * 2147483647.0, -2147483648.0, 2147483647.0);
            return;
        }
        if (format == AUDIO_FORMAT_I8) { *(signed char*)p = (signed char)audio_round_clamp_(v * 127.0, -128.0, 127.0); return; }
        if (format == AUDIO_FORMAT_I24) {
            long long s = audio_round_clamp_(v * 8388607.0, -8388608.0, 8388607.0);
            unsigned char* b = (unsigned char*)p;
            unsigned int u = (unsigned int)(s & 0xFFFFFFLL);
            b[0] = (unsigned char)(u & 0xFFU);
            b[1] = (unsigned char)((u >> 8) & 0xFFU);
            b[2] = (unsigned char)((u >> 16) & 0xFFU);
            return;
        }
    }
}

void audio_convert_raw_(void* dst, unsigned int dst_format, const void* src, unsigned int src_format,
                        unsigned long n) {
    int db = audio_format_bytes(dst_format);
    int sb = audio_format_bytes(src_format);
    if (db == 0 || sb == 0) return;
    unsafe {
        if (dst_format == src_format) {
            unsigned char* d = (unsigned char*)dst;
            const unsigned char* s = (const unsigned char*)src;
            unsigned long bytes = n * (unsigned long)db;
            for (unsigned long i = 0UL; i < bytes; i = i + 1UL) d[i] = s[i];
            return;
        }
        if (src_format == AUDIO_FORMAT_F32 && dst_format == AUDIO_FORMAT_I16) {
            audio_f32_to_i16((const &float)src, (&short)dst, n);
            return;
        }
        if (src_format == AUDIO_FORMAT_I16 && dst_format == AUDIO_FORMAT_F32) {
            audio_i16_to_f32((const &short)src, (&float)dst, n);
            return;
        }
        const unsigned char* s = (const unsigned char*)src;
        unsigned char* d = (unsigned char*)dst;
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            double v = audio_sample_read_raw_((const void*)(s + i * (unsigned long)sb), src_format);
            audio_sample_write_raw_((void*)(d + i * (unsigned long)db), dst_format, v);
        }
    }
}

void audio_interleave_f32(const &float planar, &float out, unsigned long frames, int channels) {
    unsigned long c = (unsigned long)channels;
    unsafe {
        for (unsigned long ch = 0UL; ch < c; ch = ch + 1UL)
            for (unsigned long f = 0UL; f < frames; f = f + 1UL)
                out[f * c + ch] = planar[ch * frames + f];
    }
}

void audio_deinterleave_f32(const &float in, &float planar, unsigned long frames, int channels) {
    unsigned long c = (unsigned long)channels;
    unsafe {
        for (unsigned long ch = 0UL; ch < c; ch = ch + 1UL)
            for (unsigned long f = 0UL; f < frames; f = f + 1UL)
                planar[ch * frames + f] = in[f * c + ch];
    }
}

void audio_copy_frames_raw_(void* dst, unsigned int dst_format, int dst_channels, int dst_interleaved, int dst_first,
                            const void* src, unsigned int src_format, int src_channels, int src_interleaved,
                            int src_first, int channels, unsigned long frames) {
    unsigned long db = (unsigned long)audio_format_bytes(dst_format);
    unsigned long sb = (unsigned long)audio_format_bytes(src_format);
    if (db == 0UL || sb == 0UL || channels <= 0) return;
    unsigned long dc = (unsigned long)dst_channels;
    unsigned long scn = (unsigned long)src_channels;
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        // Whole-buffer fast path: identical layout and format, no offsets.
        if (dst_format == src_format && dst_channels == channels && src_channels == channels &&
            dst_first == 0 && src_first == 0 && (dst_interleaved == src_interleaved || channels == 1)) {
            unsigned long bytes = frames * (unsigned long)channels * db;
            for (unsigned long i = 0UL; i < bytes; i = i + 1UL) d[i] = s[i];
            return;
        }
        for (int ch = 0; ch < channels; ch = ch + 1) {
            unsigned long dch = (unsigned long)(dst_first + ch);
            unsigned long sch = (unsigned long)(src_first + ch);
            // Sample index of (frame f, channel c) = interleaved ? f*C + c : c*frames + f.
            unsigned long dstep = dst_interleaved ? dc : 1UL;
            unsigned long sstep = src_interleaved ? scn : 1UL;
            unsigned long di = dst_interleaved ? dch : dch * frames;
            unsigned long si = src_interleaved ? sch : sch * frames;
            for (unsigned long f = 0UL; f < frames; f = f + 1UL) {
                if (dst_format == src_format) {
                    for (unsigned long b = 0UL; b < db; b = b + 1UL) d[di * db + b] = s[si * sb + b];
                } else {
                    double v = audio_sample_read_raw_((const void*)(s + si * sb), src_format);
                    audio_sample_write_raw_((void*)(d + di * db), dst_format, v);
                }
                di = di + dstep;
                si = si + sstep;
            }
        }
    }
}

void audio_gain_f32(&float buf, unsigned long n, float gain) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) buf[i] = buf[i] * gain;
    }
}

void audio_mix_f32(&float dst, const &float src, unsigned long n, float gain) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) dst[i] = dst[i] + src[i] * gain;
    }
}

float audio_peak_f32(const &float buf, unsigned long n) {
    float p = 0.0f;
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            float v = buf[i];
            if (v < 0.0f) v = -v;
            if (v > p) p = v;
        }
    }
    return p;
}

void audio_f32_to_fixed(const &float in, &Fixed out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            double v = (double)in[i] * 16777216.0;
            out[i] = (Fixed)(int)audio_round_clamp_(v, -2147483648.0, 2147483647.0);
        }
    }
}

void audio_fixed_to_f32(const &Fixed in, &float out, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = (float)((double)(int)in[i] / 16777216.0);
    }
}

#define AUDIO_FIXED_CHUNK_ 256UL

unsigned long audio_buffer_write_f32(&AudioBuffer buf, const &float in, unsigned long frames) {
    Fixed tmp[256];
    unsigned long ch = buf.channels;
    if (ch == 0UL) return 0UL;
    unsigned long per = AUDIO_FIXED_CHUNK_ / ch;
    if (per == 0UL) return 0UL;
    unsigned long done = 0UL;
    while (done < frames) {
        unsigned long n = frames - done;
        if (n > per) n = per;
        unsigned long room = buf.writable_frames();
        if (n > room) n = room;
        if (n == 0UL) break;
        unsafe {
            audio_f32_to_fixed((const &float)&in[done * ch], &tmp[0], n * ch);
        }
        unsigned long w = buf.write_frames(&tmp[0], n);
        done = done + w;
        if (w < n) break;
    }
    return done;
}

unsigned long audio_buffer_read_f32(&AudioBuffer buf, &float out, unsigned long frames) {
    Fixed tmp[256];
    unsigned long ch = buf.channels;
    if (ch == 0UL) return 0UL;
    unsigned long per = AUDIO_FIXED_CHUNK_ / ch;
    if (per == 0UL) return 0UL;
    unsigned long done = 0UL;
    while (done < frames) {
        unsigned long n = frames - done;
        if (n > per) n = per;
        unsigned long r = buf.read_frames(&tmp[0], n);
        if (r == 0UL) break;
        unsafe {
            audio_fixed_to_f32(&tmp[0], (&float)&out[done * ch], r * ch);
        }
        done = done + r;
        if (r < n) break;
    }
    return done;
}

} // namespace std
