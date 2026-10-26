// std::audio — real-time audio and MIDI I/O: sample-format conversion, the
// dummy API (deterministic through AudioStream::pump(), and on its timer
// thread), the API registry and errors, MIDI messages/parser/encoder and
// the dummy MIDI loopback; on Linux ALSA (zero-device probe, the "null" and
// "file" PCMs, a capture from a file through an ALSA config), the ALSA
// sequencer when /dev/snd/seq exists, and JACK audio + MIDI loopback
// against a running server — or one this test starts with
// 'jackd -d dummy' when jackd is installed. Parts whose system piece is
// missing print SKIP and pass.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/audio/audio.h>
#include <std/audio/audio_convert.h>
#include <std/audio/audio_sys.h>
#include <std/audio/midi.h>

extern int printf(const char* fmt, ...);
extern int system(const char* cmd);
extern int setenv(const char* name, const char* value, int overwrite);
extern void* fopen(const char* path, const char* mode);
extern unsigned long fread(void* buf, unsigned long size, unsigned long n, void* f);
extern unsigned long fwrite(const void* buf, unsigned long size, unsigned long n, void* f);
extern int fclose(void* f);
extern int access(const char* path, int mode);

static float absf_(float v) { return v < (float)0.0 ? -v : v; }

// ── conversion ───────────────────────────────────────────────────────────

void t_convert_formats() {
    float f[6];
    f[0] = (float)0.0; f[1] = (float)0.5; f[2] = (float)-0.5; f[3] = (float)1.0; f[4] = (float)-1.0; f[5] = (float)2.0;
    short s[6];
    audio_f32_to_i16(&f[0], &s[0], 6UL);
    ASSERT_EQ(s[0], 0);
    ASSERT_EQ(s[1], 16384);              // 0.5 * 32767 rounded
    ASSERT_EQ(s[2], -16384);
    ASSERT_EQ(s[3], 32767);
    ASSERT_EQ(s[4], -32767);
    ASSERT_EQ(s[5], 32767);              // out of range saturates
    float back[6];
    audio_i16_to_f32(&s[0], &back[0], 6UL);
    ASSERT_TRUE(absf_(back[1] - (float)0.5) < (float)0.0001);
    short mn = (short)-32768;
    float mnf = (float)0.0;
    audio_i16_to_f32(&mn, &mnf, 1UL);
    ASSERT_TRUE(mnf == (float)-1.0);

    unsigned char p24[18];
    audio_f32_to_i24(&f[0], &p24[0], 6UL);
    ASSERT_EQ(p24[3], 0x00); ASSERT_EQ(p24[4], 0x00); ASSERT_EQ(p24[5], 0x40);   // 0.5 -> 0x400000 (rounded)
    ASSERT_EQ(p24[9], 0xFF); ASSERT_EQ(p24[10], 0xFF); ASSERT_EQ(p24[11], 0x7F); // 1.0 -> 0x7FFFFF
    float f24[6];
    audio_i24_to_f32(&p24[0], &f24[0], 6UL);
    ASSERT_TRUE(absf_(f24[2] + (float)0.5) < (float)0.000001);
    ASSERT_TRUE(absf_(f24[4] + (float)1.0) < (float)0.000001);

    int i32[6];
    audio_f32_to_i32(&f[0], &i32[0], 6UL);
    ASSERT_EQ(i32[3], 2147483647);
    ASSERT_EQ(i32[5], 2147483647);
    float f32b[6];
    audio_i32_to_f32(&i32[0], &f32b[0], 6UL);
    ASSERT_TRUE(absf_(f32b[2] + (float)0.5) < (float)0.000001);

    double d[6];
    audio_f32_to_f64(&f[0], &d[0], 6UL);
    ASSERT_TRUE(d[1] == 0.5);
    signed char b8[6];
    audio_f32_to_i8(&f[0], &b8[0], 6UL);
    ASSERT_EQ(b8[3], 127);
    ASSERT_EQ(b8[4], -127);
    short sh[2]; sh[0] = (short)1000; sh[1] = (short)-1000;
    int wide[2];
    audio_i16_to_i32(&sh[0], &wide[0], 2UL);
    ASSERT_EQ(wide[0], 1000 * 65536);
    short narrow[2];
    audio_i32_to_i16(&wide[0], &narrow[0], 2UL);
    ASSERT_EQ(narrow[0], 1000);
    ASSERT_EQ(narrow[1], -1000);

    // The generic path agrees with the specialized ones.
    short s2[6];
    unsafe { audio_convert_raw_((void*)&s2[0], AUDIO_FORMAT_I16, (const void*)&f[0], AUDIO_FORMAT_F32, 6UL); }
    for (int i = 0; i < 6; i = i + 1) ASSERT_EQ(s2[i], s[i]);
    double d2[6];
    unsafe { audio_convert_raw_((void*)&d2[0], AUDIO_FORMAT_F64, (const void*)&p24[0], AUDIO_FORMAT_I24, 6UL); }
    ASSERT_TRUE(d2[3] > 0.99999 && d2[3] <= 1.0);
}

void t_convert_dither_layout() {
    // Dither: within one LSB of the undithered value, and not constant.
    float f[64];
    for (int i = 0; i < 64; i = i + 1) f[i] = (float)0.25;
    short s[64];
    unsigned int seed = 12345U;
    audio_f32_to_i16_dither(&f[0], &s[0], 64UL, &seed);
    int lo = 99999; int hi = -99999;
    for (int i = 0; i < 64; i = i + 1) {
        int v = (int)s[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    ASSERT_TRUE(lo >= 8191 && hi <= 8193 && hi > lo);

    float planar[6];        // 2 channels x 3 frames: L0 L1 L2 R0 R1 R2
    planar[0] = (float)1.0; planar[1] = (float)2.0; planar[2] = (float)3.0;
    planar[3] = (float)-1.0; planar[4] = (float)-2.0; planar[5] = (float)-3.0;
    float inter[6];
    audio_interleave_f32(&planar[0], &inter[0], 3UL, 2);
    ASSERT_TRUE(inter[0] == (float)1.0 && inter[1] == (float)-1.0 && inter[4] == (float)3.0 && inter[5] == (float)-3.0);
    float back[6];
    audio_deinterleave_f32(&inter[0], &back[0], 3UL, 2);
    for (int i = 0; i < 6; i = i + 1) ASSERT_TRUE(back[i] == planar[i]);

    // 2 interleaved f32 channels -> channels 1..2 of a 4-channel planar i16 buffer.
    short dev[12];
    for (int i = 0; i < 12; i = i + 1) dev[i] = (short)7;
    unsafe {
        audio_copy_frames_raw_((void*)&dev[0], AUDIO_FORMAT_I16, 4, 0, 1, (const void*)&inter[0], AUDIO_FORMAT_F32, 2, 1, 0,
                               2, 3UL);
    }
    ASSERT_EQ(dev[0], 7);                       // channel 0 untouched
    ASSERT_EQ(dev[3], 32767);                   // channel 1, frame 0: 1.0
    ASSERT_EQ(dev[6], -32767);                  // channel 2, frame 0: -1.0
    ASSERT_EQ(dev[9], 7);                       // channel 3 untouched

    float g[4];
    g[0] = (float)0.5; g[1] = (float)-0.25; g[2] = (float)1.0; g[3] = (float)0.0;
    audio_gain_f32(&g[0], 4UL, (float)2.0);
    ASSERT_TRUE(g[0] == (float)1.0 && g[1] == (float)-0.5);
    ASSERT_TRUE(audio_peak_f32(&g[0], 4UL) == (float)2.0);
    float m[4];
    for (int i = 0; i < 4; i = i + 1) m[i] = (float)1.0;
    audio_mix_f32(&m[0], &g[0], 4UL, (float)0.5);
    ASSERT_TRUE(m[0] == (float)1.5 && m[2] == (float)2.0);
}

void t_convert_dsp() {
    float f[4];
    f[0] = (float)0.5; f[1] = (float)-0.5; f[2] = (float)0.0; f[3] = (float)1.0;
    Fixed q[4];
    audio_f32_to_fixed(&f[0], &q[0], 4UL);
    ASSERT_EQ((int)q[0], 1 << 23);
    ASSERT_EQ((int)q[3], 1 << 24);
    float back[4];
    audio_fixed_to_f32(&q[0], &back[0], 4UL);
    for (int i = 0; i < 4; i = i + 1) ASSERT_TRUE(back[i] == f[i]);

    // A dsp AudioBuffer as the lock-free hand-off between a callback and the app.
    struct AudioBuffer ring = AudioBuffer::new(2UL);
    float frames[600];
    for (int i = 0; i < 600; i = i + 1) frames[i] = (float)(i % 50) / (float)100.0;
    ASSERT_EQ(audio_buffer_write_f32(&ring, &frames[0], 300UL), 300);
    float out[600];
    ASSERT_EQ(audio_buffer_read_f32(&ring, &out[0], 300UL), 300);
    int same = 1;
    for (int i = 0; i < 600; i = i + 1) if (absf_(out[i] - frames[i]) > (float)0.0000001) same = 0;
    ASSERT_TRUE(same);
    ASSERT_EQ(audio_buffer_read_f32(&ring, &out[0], 10UL), 0);
}

// ── API registry, devices, errors ────────────────────────────────────────

void t_api_registry() {
    int apis[AUDIO_API_COUNT];
    int n = audio_compiled_apis(&apis[0], AUDIO_API_COUNT);
    ASSERT_TRUE(n >= 1);
    ASSERT_EQ(apis[n - 1], AUDIO_API_DUMMY);
    ASSERT_TRUE(audio_api_compiled(AUDIO_API_DUMMY));
#if defined(__linux__)
    ASSERT_TRUE(audio_api_compiled(AUDIO_API_ALSA) && audio_api_compiled(AUDIO_API_JACK));
    ASSERT_FALSE(audio_api_compiled(AUDIO_API_WASAPI));
#endif
    ASSERT_EQ(audio_api_by_name("alsa"), AUDIO_API_ALSA);
    ASSERT_EQ(audio_api_by_name("dummy"), AUDIO_API_DUMMY);
    ASSERT_EQ(audio_api_by_name("nope"), AUDIO_API_UNSPECIFIED);
    ASSERT_STR_EQ(audio_api_display_name(AUDIO_API_COREAUDIO), "Core Audio");
    ASSERT_EQ(audio_format_bytes(AUDIO_FORMAT_I24), 3);
    ASSERT_STR_EQ(audio_format_name(AUDIO_FORMAT_F64), "f64");
    ASSERT_STR_EQ(audio_error_name(AUDIO_ERR_NO_DEVICES), "no devices");

    // UNSPECIFIED always opens something (with zero devices if need be).
    struct Result<Audio, AudioError> any = Audio::open(AUDIO_API_UNSPECIFIED);
    ASSERT_TRUE(any.ok());
    struct Audio a = any.unwrap_or(Audio::closed());
    ASSERT_TRUE(a.is_open());
    printf("    default API here: %s, %u device(s)\n", audio_api_name(a.api()), a.device_count());
    a.close();
    unsafe {   // close() leaves it closed and empty
        ASSERT_FALSE(a.is_open());
        ASSERT_EQ(a.device_count(), 0);
    }
#if !defined(_WIN32)
    struct Result<Audio, AudioError> bad = Audio::open(AUDIO_API_WASAPI);
    ASSERT_FALSE(bad.ok());
    match (bad.get_err()) { case some(e): ASSERT_EQ(e.code, AUDIO_ERR_UNSUPPORTED); case null: ASSERT_TRUE(0); }
    bad.drop();
#endif
}

void t_dummy_devices() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    ASSERT_EQ(a.device_count(), 2);
    struct AudioDeviceInfo d0 = a.device_info(0U).unwrap_or(AudioDeviceInfo::none());
    struct AudioDeviceInfo d1 = a.device_info(1U).unwrap_or(AudioDeviceInfo::none());
    ASSERT_STR_EQ(&d0.name[0], "Dummy Duplex");
    ASSERT_EQ(d0.output_channels, 8);
    ASSERT_EQ(d0.duplex_channels, 8);
    ASSERT_TRUE(d0.is_default_output && d0.is_default_input);
    ASSERT_TRUE(d0.supports_rate(44100U) && d0.supports_rate(192000U) && !d0.supports_rate(12345U));
    ASSERT_EQ(d0.preferred_sample_rate, 48000);
    ASSERT_TRUE((d0.native_formats & AUDIO_FORMAT_I24) != 0U);
    ASSERT_EQ(d1.input_channels, 0);
    ASSERT_TRUE(d0.id != 0U && d1.id != d0.id);
    ASSERT_EQ(a.default_output_device(), d0.id);
    // Ids survive a rescan.
    ASSERT_TRUE(a.rescan().ok());
    struct AudioDeviceInfo again = a.device_by_id(d1.id).unwrap_or(AudioDeviceInfo::none());
    ASSERT_STR_EQ(&again.name[0], "Dummy Output");
    ASSERT_FALSE(a.device_info(5U).ok());
    ASSERT_FALSE(a.device_by_id(999U).ok());
    // The rate list is sorted.
    for (int i = 1; i < d0.sample_rate_count; i = i + 1) ASSERT_TRUE(d0.sample_rates[i - 1] < d0.sample_rates[i]);
    a.close();
}

struct Rec {
    int calls;
    int stop_after;           // AUDIO_CB_STOP on this call (0 = never)
    int abort_after;
    unsigned int status;
    unsigned int frames;
    double last_time;
    float first_in;           // input[0] of the last period
    float in_sum;
    int bad;
};

static struct Rec rec_new_() {
    struct Rec r;
    unsafe { safe_memset_raw_((void*)&r, 0, sizeof(struct Rec)); }
    return r;
}

// Out: left = calls*1000 + frame, right = -(that), as f32 interleaved.
int rec_f32(&AudioIo io, ?&Rec rp) {
    match (rp) {
        case some(r): {
            r.calls = r.calls + 1;
            r.status = r.status | io.status;
            r.frames = io.frames;
            r.last_time = io.stream_time;
            struct Slice<float> in = io.input_f32();
            if (in.len > 0UL) {
                r.first_in = *in.at(0UL);
                r.in_sum = (float)0.0;
                for (unsigned long i = 0UL; i < in.len; i = i + 1UL) r.in_sum = r.in_sum + *in.at(i);
            }
            struct Slice<float> o = io.output_f32();
            for (unsigned long f = 0UL; f < (unsigned long)io.frames && o.len > 0UL; f = f + 1UL) {
                float v = (float)(r.calls * 1000 + (int)f) / (float)100000.0;
                o.set(f * 2UL, v);
                o.set(f * 2UL + 1UL, -v);
            }
            if (r.calls == r.stop_after) return AUDIO_CB_STOP;
            if (r.calls == r.abort_after) return AUDIO_CB_ABORT;
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

void t_dummy_manual() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    struct AudioStreamParams p = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    struct AudioStreamOptions o = AudioStreamOptions::with_flags(AUDIO_FLAG_DUMMY_MANUAL);
    o.buffer_count = 4U;
    struct Rec r = rec_new_();
    unsigned int frames = 128U;
    struct Result<AudioStream, AudioError> sr =
        a.open_stream((?&AudioStreamParams)&p, (?&AudioStreamParams)&p, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32,
                      (?&Rec)&r, (?&AudioStreamOptions)&o);
    ASSERT_TRUE(sr.ok());
    struct AudioStream s = sr.unwrap_or(AudioStream::closed());
    ASSERT_EQ(frames, 128);
    ASSERT_EQ(s.buffer_frames(), 128);
    ASSERT_EQ(s.sample_rate(), 48000);
    ASSERT_EQ(s.latency(), 2 * 128 * 4);
    ASSERT_EQ(s.state(), AUDIO_STREAM_STOPPED);
    ASSERT_EQ(s.pump(), -1);                       // not started
    ASSERT_TRUE(s.start().ok());
    ASSERT_TRUE(s.is_running());
    ASSERT_EQ(s.pump(), AUDIO_CB_CONTINUE);
    ASSERT_TRUE(r.first_in == (float)0.0);         // nothing played yet
    ASSERT_EQ(s.pump(), AUDIO_CB_CONTINUE);
    // Loopback: this period's input is the previous period's output.
    ASSERT_TRUE(r.first_in == (float)(1 * 1000 + 0) / (float)100000.0);
    ASSERT_TRUE(absf_(r.in_sum) < (float)0.0001);  // left + right cancel
    ASSERT_TRUE(r.last_time == 128.0 / 48000.0);
    ASSERT_TRUE(s.stream_time() == 256.0 / 48000.0);
    s.set_stream_time(10.0);
    s.pump();
    ASSERT_TRUE(r.last_time == 10.0);
    // A second start is a warning, not an error.
    ASSERT_EQ(s.start().code, AUDIO_ERR_WARNING);
    r.stop_after = r.calls + 2;
    ASSERT_EQ(s.pump(), AUDIO_CB_CONTINUE);
    ASSERT_EQ(s.pump(), AUDIO_CB_STOP);
    ASSERT_FALSE(s.is_running());
    ASSERT_EQ(s.pump(), -1);
    ASSERT_EQ(s.stop().code, AUDIO_ERR_WARNING);   // already stopped by the callback
    // Restart, then abort from the callback.
    ASSERT_TRUE(s.start().ok());
    r.stop_after = 0;
    r.abort_after = r.calls + 1;
    ASSERT_EQ(s.pump(), AUDIO_CB_ABORT);
    ASSERT_FALSE(s.is_running());
    ASSERT_TRUE(s.last_error().ok());
    s.close();
    unsafe {   // a closed stream still answers (close() leaves it in the closed state)
        ASSERT_FALSE(s.is_open());
        ASSERT_EQ(s.state(), AUDIO_STREAM_CLOSED);
    }
    a.close();
}

// i16, non-interleaved, channel offsets: the callback sees its own layout.
int rec_i16_planar(&AudioIo io, ?&Rec rp) {
    match (rp) {
        case some(r): {
            r.calls = r.calls + 1;
            if (io.interleaved || io.format != AUDIO_FORMAT_I16) r.bad = r.bad + 1;
            if (!io.output_f32().is_empty()) r.bad = r.bad + 1;     // other formats' views are empty
            if (!io.output_channel_f32(0).is_empty()) r.bad = r.bad + 1;
            struct Slice<short> in = io.input_i16();
            // Channel 1 of the input (planar: second half) is the loopback of
            // what we wrote to output channel 1 last period.
            if (in.len == 2UL * (unsigned long)io.frames) r.first_in = (float)*in.at((unsigned long)io.frames);
            struct Slice<short> o = io.output_i16();
            for (unsigned long f = 0UL; f < (unsigned long)io.frames; f = f + 1UL) {
                o.set(f, (short)1000);                                   // channel 0
                o.set((unsigned long)io.frames + f, (short)(r.calls * 10)); // channel 1
            }
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

void t_dummy_layouts() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    struct AudioStreamParams p = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    p.first_channel = 3;                          // device channels 3 and 4
    struct AudioStreamOptions o = AudioStreamOptions::with_flags(AUDIO_FLAG_DUMMY_MANUAL | AUDIO_FLAG_NONINTERLEAVED);
    struct Rec r = rec_new_();
    unsigned int frames = 64U;
    struct Result<AudioStream, AudioError> sr =
        a.open_stream((?&AudioStreamParams)&p, (?&AudioStreamParams)&p, AUDIO_FORMAT_I16, 44100U, &frames, rec_i16_planar,
                      (?&Rec)&r, (?&AudioStreamOptions)&o);
    ASSERT_TRUE(sr.ok());
    struct AudioStream s = sr.unwrap_or(AudioStream::closed());
    s.start();
    s.pump();
    s.pump();
    s.pump();
    ASSERT_EQ(r.bad, 0);
    // i16 -> device f32 -> loopback -> i16 again: exact for these values.
    ASSERT_TRUE(r.first_in == (float)20.0);
    s.close();

    // f64 and i24 streams convert through the device format too.
    struct AudioStreamParams mono = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 1);
    struct AudioStreamOptions man = AudioStreamOptions::with_flags(AUDIO_FLAG_DUMMY_MANUAL | AUDIO_FLAG_MINIMIZE_LATENCY);
    struct Rec r2 = rec_new_();
    frames = 1000U;
    sr = a.open_stream((?&AudioStreamParams)&mono, null, AUDIO_FORMAT_F64, 96000U, &frames, rec_f32, (?&Rec)&r2,
                       (?&AudioStreamOptions)&man);
    ASSERT_TRUE(sr.ok());
    ASSERT_EQ(frames, 64);                         // MINIMIZE_LATENCY
    s = sr.unwrap_or(AudioStream::closed());
    s.start();
    s.pump();
    ASSERT_EQ(r2.calls, 1);
    s.close();
    a.close();
}

// Input-only: the dummy device captures a 440 Hz sine at half scale.
void t_dummy_input() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    struct AudioStreamParams p = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 1);
    struct AudioStreamOptions o = AudioStreamOptions::with_flags(AUDIO_FLAG_DUMMY_MANUAL);
    struct Rec r = rec_new_();
    unsigned int frames = 480U;
    struct Result<AudioStream, AudioError> sr =
        a.open_stream(null, (?&AudioStreamParams)&p, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r,
                      (?&AudioStreamOptions)&o);
    struct AudioStream s = sr.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(s.is_open());
    ASSERT_EQ(s.output_latency(), 0);
    s.start();
    s.pump();                // 480 frames = 4.4 cycles of 440 Hz
    ASSERT_TRUE(r.first_in == (float)0.0);
    ASSERT_TRUE(r.in_sum > (float)-50.0 && r.in_sum < (float)50.0);
    s.close();
    a.close();
}

void t_dummy_thread() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    struct AudioStreamParams p = AudioStreamParams::of(a.default_output_device(), 2);
    struct Rec r = rec_new_();
    r.stop_after = 12;
    unsigned int frames = 64U;
    struct AudioStreamOptions o = AudioStreamOptions::with_flags(AUDIO_FLAG_SCHEDULE_REALTIME);
    struct Result<AudioStream, AudioError> sr =
        a.open_stream((?&AudioStreamParams)&p, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r,
                      (?&AudioStreamOptions)&o);
    struct AudioStream s = sr.unwrap_or(AudioStream::closed());
    long long t0 = audio_now_ns_();
    ASSERT_TRUE(s.start().ok());
    ASSERT_EQ(s.pump(), -1);                       // not a manual stream
    while (s.is_running() && audio_now_ns_() - t0 < 5000000000LL) audio_sleep_us_(1000LL);
    long long ms = (audio_now_ns_() - t0) / 1000000LL;
    ASSERT_FALSE(s.is_running());
    ASSERT_EQ(r.calls, 12);
    // 12 periods of 64 frames at 48 kHz: 16 ms of audio, paced in real time.
    ASSERT_TRUE(ms >= 10LL);
    ASSERT_TRUE(s.stream_time() == 768.0 / 48000.0);
    // Run again and stop from the control thread.
    r.stop_after = 0;
    ASSERT_TRUE(s.start().ok());
    audio_sleep_us_(20000LL);
    ASSERT_TRUE(s.stop().ok());
    ASSERT_FALSE(s.is_running());
    int calls = r.calls;
    audio_sleep_us_(10000LL);
    ASSERT_EQ(r.calls, calls);                     // nothing after stop()
    ASSERT_TRUE(s.abort().failed() == 0);          // abort on a stopped stream: a warning
    s.close();
    unsafe { s.close(); }                          // idempotent (a second close does nothing)
    a.close();
}

void t_stream_errors() {
    struct Audio a = Audio::open(AUDIO_API_DUMMY).unwrap_or(Audio::closed());
    struct AudioStreamParams p = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    struct Rec r = rec_new_();
    unsigned int frames = 256U;
    struct Result<AudioStream, AudioError> e1 = a.open_stream(null, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e1);
    match (e1.get_err()) { case some(e): ASSERT_EQ(e.code, AUDIO_ERR_INVALID_PARAM); case null: ASSERT_TRUE(0); }
    struct Result<AudioStream, AudioError> e2 = a.open_stream((?&AudioStreamParams)&p, null, 0x40U, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e2);
    ASSERT_FALSE(e2.ok());
    struct AudioStreamParams nine = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 9);
    struct Result<AudioStream, AudioError> e3 = a.open_stream((?&AudioStreamParams)&nine, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e3);
    match (e3.get_err()) {
        case some(e): { ASSERT_EQ(e.code, AUDIO_ERR_INVALID_PARAM); ASSERT_TRUE(audio_strprefix_(&e.message[0], "audio: device has too few channels")); }
        case null: ASSERT_TRUE(0);
    }
    struct AudioStreamParams gone = AudioStreamParams::of(4242U, 1);
    struct Result<AudioStream, AudioError> e4 = a.open_stream((?&AudioStreamParams)&gone, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e4);
    match (e4.get_err()) { case some(e): ASSERT_EQ(e.code, AUDIO_ERR_INVALID_DEVICE); case null: ASSERT_TRUE(0); }
    // Input from the output-only device.
    struct AudioStreamParams outOnly = AudioStreamParams::of(a.device_info(1U).unwrap_or(AudioDeviceInfo::none()).id, 1);
    struct Result<AudioStream, AudioError> e5 = a.open_stream(null, (?&AudioStreamParams)&outOnly, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e5);
    ASSERT_FALSE(e5.ok());
    struct AudioStream closed = AudioStream::closed();
    ASSERT_EQ(closed.start().code, AUDIO_ERR_INVALID_USE);
    ASSERT_EQ(closed.stop().code, AUDIO_ERR_INVALID_USE);
    ASSERT_TRUE(closed.stream_time() == 0.0);
    closed.close();
    struct Audio na = Audio::closed();
    struct Result<AudioStream, AudioError> e6 = na.open_stream((?&AudioStreamParams)&p, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r, null); defer drop(e6);
    match (e6.get_err()) { case some(e): ASSERT_EQ(e.code, AUDIO_ERR_INVALID_USE); case null: ASSERT_TRUE(0); }
    struct AudioError m = AudioError::make_code(AUDIO_ERR_SYSTEM, "x failed", -42LL);
    ASSERT_STR_EQ(&m.message[0], "x failed (-42)");
    ASSERT_TRUE(m.failed());
    na.close();
    a.close();
}

// ── ALSA (Linux) ─────────────────────────────────────────────────────────

static int g_alsa_config_ = 0;

#if defined(__linux__)
// An ALSA config: the system one plus a capture PCM reading a file.
static void alsa_test_config_() {
    int ok = 0;
    unsafe { ok = access("/usr/share/alsa/alsa.conf", 4) == 0; }
    if (!ok) return;
    void* f = (void*)0;
    unsafe { f = fopen("/tmp/safec_audio_test_in.raw", "wb"); }
    if (f == (void*)0) return;
    short ramp[4096];
    for (int i = 0; i < 4096; i = i + 1) ramp[i] = (short)i;
    unsafe { fwrite((const void*)&ramp[0], 2UL, 4096UL, f); fclose(f); }
    unsafe { f = fopen("/tmp/safec_audio_test.conf", "wb"); }
    if (f == (void*)0) return;
    const char* conf = "</usr/share/alsa/alsa.conf>\n"
                       "pcm.safec_capture_test {\n type file\n slave.pcm null\n file \"/dev/null\"\n"
                       " infile \"/tmp/safec_audio_test_in.raw\"\n format raw\n}\n";
    unsafe {
        fwrite((const void*)conf, 1UL, audio_strlen_(conf), f);
        fclose(f);
        setenv("ALSA_CONFIG_PATH", "/tmp/safec_audio_test.conf", 1);
    }
    g_alsa_config_ = 1;
}

static int alsa_present_() {
    struct Result<Audio, AudioError> r = Audio::open(AUDIO_API_ALSA);
    if (!r.ok()) {
        match (r.get_err()) { case some(e): printf("    SKIP alsa: %s\n", e.message); case null: {} }
        r.drop();
        return 0;
    }
    struct Audio a = r.unwrap_or(Audio::closed());
    a.close();
    return 1;
}

static long file_size_(const &char path) {
    void* f = (void*)0;
    unsafe { f = fopen(path, "rb"); }
    if (f == (void*)0) return -1L;
    unsigned char buf[4096];
    long total = 0L;
    while (1) {
        unsigned long n = 0UL;
        unsafe { n = fread((void*)&buf[0], 1UL, 4096UL, f); }
        if (n == 0UL) break;
        total = total + (long)n;
    }
    unsafe { fclose(f); }
    return total;
}

int alsa_cap(&AudioIo io, ?&Rec rp) {
    match (rp) {
        case some(r): {
            r.calls = r.calls + 1;
            struct Slice<short> in = io.input_i16();
            for (unsigned long i = 0UL; i < in.len; i = i + 1UL) {
                // The capture file holds the ramp 0, 1, 2, ... (i16 mono).
                int want = (r.calls - 1) * (int)io.frames + (int)i;
                if ((int)*in.at(i) != want) r.bad = r.bad + 1;
            }
            if (r.calls == r.stop_after) return AUDIO_CB_STOP;
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

void t_alsa() {
    if (!alsa_present_()) return;
    struct Audio a = Audio::open(AUDIO_API_ALSA).unwrap_or(Audio::closed());
    printf("    alsa: %u device(s)\n", a.device_count());
    for (unsigned int i = 0U; i < a.device_count(); i = i + 1U) {
        struct AudioDeviceInfo d = a.device_info(i).unwrap_or(AudioDeviceInfo::none());
        ASSERT_TRUE(d.output_channels > 0 || d.input_channels > 0);
    }
    // Playback into the 'file' plugin: everything the callbacks wrote lands in the file.
    unsafe { system("rm -f /tmp/safec_alsa_test_out.raw"); }
    struct AudioStreamParams p = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    struct AudioStreamOptions o = AudioStreamOptions::none();
    audio_strcpy_(&o.device_name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "file:FILE=/tmp/safec_alsa_test_out.raw,FORMAT=raw");
    struct Rec r = rec_new_();
    r.stop_after = 10;
    unsigned int frames = 256U;
    struct Result<AudioStream, AudioError> sr =
        a.open_stream((?&AudioStreamParams)&p, null, AUDIO_FORMAT_F32, 48000U, &frames, rec_f32, (?&Rec)&r,
                      (?&AudioStreamOptions)&o);
    match (sr.get_err()) { case some(e): printf("    alsa file: %s\n", e.message); case null: {} }
    ASSERT_TRUE(sr.ok());
    struct AudioStream s = sr.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(frames >= 64U);
    ASSERT_TRUE(s.latency() >= (long)frames);
    ASSERT_TRUE(s.start().ok());
    long long t0 = audio_now_ns_();
    while (s.is_running() && audio_now_ns_() - t0 < 5000000000LL) audio_sleep_us_(1000LL);
    ASSERT_EQ(r.calls, 10);
    s.close();
    // Pre-roll (buffer - period) of silence + 10 periods, 2 channels x f32
    // (the null PCM takes f32 as is).
    long bytes = file_size_("/tmp/safec_alsa_test_out.raw");
    long period = (long)frames * 2L * 4L;
    ASSERT_TRUE(bytes >= 10L * period && bytes % period == 0L);

    // Duplex on the null PCM (both directions; ALSA links the two PCMs).
    struct AudioStreamOptions dn = AudioStreamOptions::none();
    audio_strcpy_(&dn.device_name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "null");
    struct Rec rd = rec_new_();
    rd.stop_after = 6;
    frames = 128U;
    sr = a.open_stream((?&AudioStreamParams)&p, (?&AudioStreamParams)&p, AUDIO_FORMAT_F32, 44100U, &frames, rec_f32,
                       (?&Rec)&rd, (?&AudioStreamOptions)&dn);
    match (sr.get_err()) { case some(e): printf("    alsa duplex: %s\n", e.message); case null: {} }
    ASSERT_TRUE(sr.ok());
    s = sr.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(s.latency() > 0L && s.input_latency() > 0L);
    ASSERT_TRUE(s.start().ok());
    t0 = audio_now_ns_();
    while (s.is_running() && audio_now_ns_() - t0 < 5000000000LL) audio_sleep_us_(1000LL);
    ASSERT_EQ(rd.calls, 6);
    ASSERT_TRUE(rd.in_sum == (float)0.0);      // the null PCM captures silence
    // Restart after the callback stopped it, then abort from the control thread.
    rd.stop_after = 0;
    ASSERT_TRUE(s.start().ok());
    audio_sleep_us_(20000LL);
    ASSERT_TRUE(s.abort().ok());
    ASSERT_FALSE(s.is_running());
    s.close();

    // Capture from a file, through the ALSA config main() installed
    // (ALSA_CONFIG_PATH: the system config plus a 'file' PCM whose infile
    // holds an i16 ramp).
    if (!g_alsa_config_) { printf("    SKIP alsa capture: no /usr/share/alsa/alsa.conf to extend\n"); a.close(); return; }
    struct AudioStreamParams ip = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 1);
    struct AudioStreamOptions io = AudioStreamOptions::none();
    audio_strcpy_(&io.device_name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "safec_capture_test");
    struct Rec rc = rec_new_();
    rc.stop_after = 4;
    frames = 256U;
    sr = a.open_stream(null, (?&AudioStreamParams)&ip, AUDIO_FORMAT_I16, 48000U, &frames, alsa_cap, (?&Rec)&rc,
                       (?&AudioStreamOptions)&io);
    match (sr.get_err()) { case some(e): printf("    alsa capture: %s\n", e.message); case null: {} }
    ASSERT_TRUE(sr.ok());
    s = sr.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(s.start().ok());
    t0 = audio_now_ns_();
    while (s.is_running() && audio_now_ns_() - t0 < 5000000000LL) audio_sleep_us_(1000LL);
    ASSERT_EQ(rc.calls, 4);
    ASSERT_EQ(rc.bad, 0);                 // the samples are the file's ramp, in order
    s.close();
    a.close();
}
#endif

// ── JACK ─────────────────────────────────────────────────────────────────

static int g_jack_started_ = 0;

// A JACK server: one already running, else 'jackd -d dummy' if installed.
static int jack_up_() {
#if defined(__linux__) || defined(__APPLE__)
    struct Result<Audio, AudioError> r = Audio::open(AUDIO_API_JACK);
    if (r.ok()) { struct Audio a = r.unwrap_or(Audio::closed()); a.close(); return 1; }
    r.drop();
    if (g_jack_started_) return 0;
    int has = 0;
    unsafe { has = system("command -v jackd >/dev/null 2>&1") == 0; }
    if (!has) { printf("    SKIP jack: no server running and no jackd to start\n"); return 0; }
    unsafe {
        setenv("JACK_DEFAULT_SERVER", "safec_audio_test", 1);
        system("jackd -n safec_audio_test -d dummy -r 48000 -p 256 >/dev/null 2>&1 & echo $! > /tmp/safec_audio_test_jackd.pid");
    }
    g_jack_started_ = 1;
    for (int i = 0; i < 100; i = i + 1) {
        audio_sleep_us_(50000LL);
        struct Result<Audio, AudioError> r2 = Audio::open(AUDIO_API_JACK);
        if (r2.ok()) { struct Audio a = r2.unwrap_or(Audio::closed()); a.close(); return 1; }
        r2.drop();
    }
    printf("    SKIP jack: jackd didn't come up\n");
#endif
    return 0;
}

struct Ramp { int calls; float next; int ok; int bad; int seen; };

int jack_gen(&AudioIo io, ?&Ramp gp) {
    match (gp) {
        case some(g): {
            g.calls = g.calls + 1;
            struct Slice<float> o = io.output_f32();
            for (unsigned long f = 0UL; f < (unsigned long)io.frames; f = f + 1UL) {
                g.next = g.next + (float)(1.0 / 65536.0);
                if (g.next > (float)0.5) g.next = (float)(1.0 / 65536.0);
                o.set(f * 2UL, g.next);
                o.set(f * 2UL + 1UL, -g.next);
            }
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

int jack_sink(&AudioIo io, ?&Ramp sp) {
    match (sp) {
        case some(s): {
            s.calls = s.calls + 1;
            struct Slice<float> in = io.input_f32();
            for (unsigned long f = 0UL; f < (unsigned long)io.frames; f = f + 1UL) {
                float l = *in.at(f * 2UL);
                float r = *in.at(f * 2UL + 1UL);
                if (l == (float)0.0 && r == (float)0.0) continue;    // before the ports were connected
                float d = l - s.next;
                if (r == -l && ((d > (float)0.0 && d < (float)0.0001) || l <= (float)(1.0 / 65536.0) || s.seen == 0)) s.ok = s.ok + 1;
                else s.bad = s.bad + 1;
                s.next = l;
                s.seen = 1;
            }
            if (s.calls >= 40) return AUDIO_CB_STOP;
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

void t_jack_audio() {
    if (!jack_up_()) return;
    struct Audio a = Audio::open(AUDIO_API_JACK).unwrap_or(Audio::closed());
    ASSERT_TRUE(a.device_count() >= 1U);
    struct AudioDeviceInfo sys = a.device_by_id(a.default_output_device()).unwrap_or(AudioDeviceInfo::none());
    unsigned int rate = sys.preferred_sample_rate;
    printf("    jack: %u device(s), '%s' %d out / %d in at %u Hz\n", a.device_count(), sys.name, sys.output_channels,
           sys.input_channels, rate);
    // Wrong rate: refused.
    struct Ramp g; g.calls = 0; g.next = (float)0.0; g.ok = 0; g.bad = 0; g.seen = 0;
    struct Ramp k; k.calls = 0; k.next = (float)0.0; k.ok = 0; k.bad = 0; k.seen = 0;
    struct AudioStreamParams op = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    unsigned int frames = 0U;
    struct AudioStreamOptions o1 = AudioStreamOptions::with_flags(AUDIO_FLAG_JACK_DONT_CONNECT);
    audio_strcpy_(&o1.stream_name[0], (unsigned long)AUDIO_STREAM_NAME_MAX, "safec_src");
    struct Result<AudioStream, AudioError> bad = a.open_stream((?&AudioStreamParams)&op, null, AUDIO_FORMAT_F32, rate + 1U,
                                                               &frames, jack_gen, (?&Ramp)&g, (?&AudioStreamOptions)&o1);
    match (bad.get_err()) { case some(e): ASSERT_EQ(e.code, AUDIO_ERR_INVALID_PARAM); case null: ASSERT_TRUE(0); }
    bad.drop();
    // A generator stream, and a capture stream connected to it: a loopback through the JACK graph.
    struct Result<AudioStream, AudioError> r1 = a.open_stream((?&AudioStreamParams)&op, null, AUDIO_FORMAT_F32, rate, &frames,
                                                              jack_gen, (?&Ramp)&g, (?&AudioStreamOptions)&o1);
    ASSERT_TRUE(r1.ok());
    struct AudioStream src = r1.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(frames > 0U);
    ASSERT_TRUE(a.rescan().ok());
    int found = 0;
    for (unsigned int i = 0U; i < a.device_count(); i = i + 1U) {
        struct AudioDeviceInfo d = a.device_info(i).unwrap_or(AudioDeviceInfo::none());
        if (audio_strprefix_(&d.name[0], "safec_src")) { found = 1; ASSERT_EQ(d.input_channels, 2); }
    }
    ASSERT_TRUE(found);           // our own client is a device now
    struct AudioStreamOptions o2 = AudioStreamOptions::with_flags(AUDIO_FLAG_NONINTERLEAVED);
    o2.flags = 0U;
    audio_strcpy_(&o2.stream_name[0], (unsigned long)AUDIO_STREAM_NAME_MAX, "safec_sink");
    audio_strcpy_(&o2.device_name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "safec_src");
    struct AudioStreamParams ip = AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    struct Result<AudioStream, AudioError> r2 = a.open_stream(null, (?&AudioStreamParams)&ip, AUDIO_FORMAT_F32, rate, &frames,
                                                              jack_sink, (?&Ramp)&k, (?&AudioStreamOptions)&o2);
    ASSERT_TRUE(r2.ok());
    struct AudioStream sink = r2.unwrap_or(AudioStream::closed());
    ASSERT_TRUE(src.start().ok());
    ASSERT_TRUE(sink.start().ok());
    long long t0 = audio_now_ns_();
    while (sink.is_running() && audio_now_ns_() - t0 < 10000000000LL) audio_sleep_us_(2000LL);
    printf("    jack loopback: %d samples ok, %d bad, %d generator / %d sink callbacks\n", k.ok, k.bad, g.calls, k.calls);
    ASSERT_FALSE(sink.is_running());
    ASSERT_TRUE(k.ok > (int)frames * 10);
    ASSERT_TRUE(k.bad <= 2);
    ASSERT_TRUE(src.stop().ok());
    sink.close();
    src.close();
    a.close();
}

// ── MIDI ─────────────────────────────────────────────────────────────────

void t_midi_messages() {
    struct MidiMsg on = MidiMsg::note_on(3, 60, 100);
    ASSERT_EQ(on.len, 3);
    ASSERT_EQ(on.status(), 0x93);
    ASSERT_EQ(on.type(), MIDI_NOTE_ON);
    ASSERT_EQ(on.channel(), 3);
    ASSERT_EQ(on.data1(), 60);
    ASSERT_EQ(on.data2(), 100);
    ASSERT_TRUE(on.is_note_on() && !on.is_note_off());
    ASSERT_TRUE(MidiMsg::note_on(0, 60, 0).is_note_off());
    ASSERT_TRUE(MidiMsg::note_off(0, 60, 64).is_note_off());
    ASSERT_EQ(MidiMsg::note_on(17, 200, -5).data[0], 0x91);       // channel masked, data clamped
    ASSERT_EQ(MidiMsg::note_on(17, 200, -5).data[1], 127);
    ASSERT_EQ(MidiMsg::note_on(17, 200, -5).data[2], 0);
    struct MidiMsg pc = MidiMsg::program_change(9, 42);
    ASSERT_EQ(pc.len, 2);
    ASSERT_EQ(pc.data2(), -1);
    struct MidiMsg pb = MidiMsg::pitch_bend(0, 8191);
    ASSERT_EQ(pb.data[1], 0x7F);
    ASSERT_EQ(pb.data[2], 0x7F);
    ASSERT_EQ(pb.pitch_bend_value(), 8191);
    ASSERT_EQ(MidiMsg::pitch_bend(0, -8192).pitch_bend_value(), -8192);
    ASSERT_EQ(MidiMsg::pitch_bend(0, 0).data[2], 0x40);
    ASSERT_EQ(MidiMsg::control_change(1, 7, 127).status(), 0xB1);
    ASSERT_EQ(MidiMsg::system(MIDI_CLOCK).len, 1);
    ASSERT_EQ(MidiMsg::system(MIDI_CLOCK).channel(), -1);
    ASSERT_EQ(MidiMsg::song_position(300).data[1], 300 & 0x7F);
    ASSERT_EQ(MidiMsg::song_position(300).data[2], 300 >> 7);
    ASSERT_EQ(midi_message_length(0x90), 3);
    ASSERT_EQ(midi_message_length(0xC5), 2);
    ASSERT_EQ(midi_message_length(0xF0), 0);
    ASSERT_EQ(midi_message_length(0xF1), 2);
    ASSERT_EQ(midi_message_length(0xF8), 1);
    ASSERT_EQ(midi_message_length(0xF4), -1);
    ASSERT_EQ(midi_message_length(0x40), -1);
    ASSERT_TRUE(midi_is_realtime(0xFE) && !midi_is_realtime(0xF7));
    unsigned char sx[8];
    unsigned char pl[3]; pl[0] = (unsigned char)0x7E; pl[1] = (unsigned char)0x7F; pl[2] = (unsigned char)0x06;
    ASSERT_EQ(midi_build_sysex(&sx[0], 8UL, &pl[0], 3UL), 5);
    ASSERT_EQ(sx[0], 0xF0);
    ASSERT_EQ(sx[4], 0xF7);
    ASSERT_EQ(midi_build_sysex(&sx[0], 4UL, &pl[0], 3UL), -1);
    pl[1] = (unsigned char)0x80;
    ASSERT_EQ(midi_build_sysex(&sx[0], 8UL, &pl[0], 3UL), -1);
    ASSERT_EQ(midi_14bit(0x7F, 0x7F), 16383);
}

void t_midi_parser() {
    unsigned char sysex[8];
    struct MidiParser p = MidiParser::new((?&unsigned char)&sysex[0], 8UL);
    // Running status: one status byte, three notes.
    unsigned char stream[7];
    stream[0] = (unsigned char)0x90; stream[1] = (unsigned char)60; stream[2] = (unsigned char)100;
    stream[3] = (unsigned char)62; stream[4] = (unsigned char)101; stream[5] = (unsigned char)64; stream[6] = (unsigned char)0;
    int msgs = 0;
    int lastNote = 0;
    for (int i = 0; i < 7; i = i + 1) {
        if (p.feed((int)stream[i]) == MIDI_PARSE_MESSAGE) {
            msgs = msgs + 1;
            struct MidiMsg m = p.message();
            ASSERT_EQ(m.status(), 0x90);
            lastNote = m.data1();
        }
    }
    ASSERT_EQ(msgs, 3);
    ASSERT_EQ(lastNote, 64);
    // A clock byte in the middle of a message stands alone; the message completes.
    ASSERT_EQ(p.feed(0xB0), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(7), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(0xF8), MIDI_PARSE_MESSAGE);
    ASSERT_EQ(p.message().status(), 0xF8);
    ASSERT_EQ(p.feed(80), MIDI_PARSE_MESSAGE);
    ASSERT_EQ(p.message().data2(), 80);
    // Sysex, with a real-time byte inside it.
    ASSERT_EQ(p.feed(0xF0), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(0x41), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(0xFE), MIDI_PARSE_MESSAGE);
    ASSERT_EQ(p.feed(0x10), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(0xF7), MIDI_PARSE_SYSEX);
    ASSERT_EQ(p.sysex_length(), 4);
    ASSERT_EQ(sysex[1], 0x41);
    ASSERT_EQ(sysex[3], 0xF7);
    // Running status was cancelled by the sysex: a stray data byte is an error.
    ASSERT_EQ(p.feed(0x22), MIDI_PARSE_ERROR);
    // A sysex longer than the buffer is dropped.
    p.feed(0xF0);
    for (int i = 0; i < 20; i = i + 1) p.feed(1);
    ASSERT_EQ(p.feed(0xF7), MIDI_PARSE_ERROR);
    // An unterminated sysex ends at the next status byte, which still parses.
    p.feed(0xF0); p.feed(5);
    ASSERT_EQ(p.feed(0xC2), MIDI_PARSE_NONE);
    ASSERT_EQ(p.feed(9), MIDI_PARSE_MESSAGE);
    ASSERT_EQ(p.message().type(), MIDI_PROGRAM_CHANGE);
    ASSERT_EQ(p.feed(0xF6), MIDI_PARSE_MESSAGE);      // tune request: one byte
    p.reset();
    ASSERT_EQ(p.feed(60), MIDI_PARSE_ERROR);

    // Encoder: running status drops repeated status bytes.
    struct MidiEncoder e = MidiEncoder::new(1);
    unsigned char out[16];
    struct MidiMsg a = MidiMsg::note_on(0, 60, 1);
    struct MidiMsg b = MidiMsg::note_on(0, 61, 1);
    struct MidiMsg c = MidiMsg::note_on(1, 61, 1);
    ASSERT_EQ(e.encode(&a.data[0], 3UL, &out[0], 16UL), 3);
    ASSERT_EQ(e.encode(&b.data[0], 3UL, &out[0], 16UL), 2);
    ASSERT_EQ(out[0], 61);
    ASSERT_EQ(e.encode(&c.data[0], 3UL, &out[0], 16UL), 3);
    struct MidiMsg clk = MidiMsg::system(MIDI_CLOCK);
    ASSERT_EQ(e.encode(&clk.data[0], 1UL, &out[0], 16UL), 1);       // real-time keeps running status
    ASSERT_EQ(e.encode(&c.data[0], 3UL, &out[0], 16UL), 2);
    struct MidiMsg tr = MidiMsg::system(MIDI_TUNE_REQUEST);
    e.encode(&tr.data[0], 1UL, &out[0], 16UL);                       // system common cancels it
    ASSERT_EQ(e.encode(&c.data[0], 3UL, &out[0], 16UL), 3);
    ASSERT_EQ(e.encode(&c.data[0], 3UL, &out[0], 1UL), -1);
    struct MidiEncoder plain = MidiEncoder::new(0);
    plain.encode(&a.data[0], 3UL, &out[0], 16UL);
    ASSERT_EQ(plain.encode(&a.data[0], 3UL, &out[0], 16UL), 3);
}

struct MidiMon { int n; int notes; int sysex_len; double last_t; double total_delta; int bad_order; };

void midi_mon(&MidiEvent ev, ?&MidiMon mp) {
    match (mp) {
        case some(m): {
            m.n = m.n + 1;
            if (ev.short_msg().is_note_on()) m.notes = m.notes + 1;
            if (ev.status() == MIDI_SYSEX) m.sysex_len = (int)ev.bytes.len;
            if (ev.timestamp < m.last_t) m.bad_order = m.bad_order + 1;
            m.last_t = ev.timestamp;
            m.total_delta = m.total_delta + ev.delta;
        }
        case null: {}
    }
}

// Sends through 'out', receives on 'in' (both already opened), via the queue
// and then the callback.
static void midi_roundtrip_(&MidiIn in, &MidiOut out, int waitMs) {
    in.ignore_types(0, 1, 1);
    ASSERT_TRUE(out.send(MidiMsg::note_on(0, 60, 100)).ok());
    ASSERT_TRUE(out.send(MidiMsg::control_change(5, 7, 64)).ok());
    unsigned char sx[64];
    unsigned char pl[40];
    for (int i = 0; i < 40; i = i + 1) pl[i] = (unsigned char)i;
    long sl = midi_build_sysex(&sx[0], 64UL, &pl[0], 40UL);
    ASSERT_TRUE(out.send_message(&sx[0], (unsigned long)sl).ok());
    ASSERT_TRUE(out.send(MidiMsg::system(MIDI_CLOCK)).ok());        // filtered
    ASSERT_TRUE(out.send(MidiMsg::system(MIDI_ACTIVE_SENSING)).ok()); // filtered
    ASSERT_TRUE(out.send(MidiMsg::pitch_bend(2, -100)).ok());
    unsigned char buf[64];
    double t = 0.0;
    double lastT = -1.0;
    int got = 0;
    long n = 0L;
    long long t0 = audio_now_ns_();
    while (got < 4 && audio_now_ns_() - t0 < (long long)waitMs * 1000000LL) {
        n = in.get_message(&buf[0], 64UL, &t);
        if (n == 0L) { audio_sleep_us_(1000LL); continue; }
        if (got == 0) { ASSERT_EQ(n, 3); ASSERT_EQ(buf[0], 0x90); ASSERT_EQ(buf[1], 60); }
        if (got == 1) { ASSERT_EQ(n, 3); ASSERT_EQ(buf[0], 0xB5); }
        if (got == 2) { ASSERT_EQ(n, sl); ASSERT_EQ(buf[0], 0xF0); ASSERT_EQ(buf[20], 19); ASSERT_EQ(buf[(int)sl - 1], 0xF7); }
        if (got == 3) { ASSERT_EQ(n, 3); ASSERT_EQ(buf[0], 0xE2); }
        ASSERT_TRUE(t >= lastT && t >= 0.0);
        lastT = t;
        got = got + 1;
    }
    ASSERT_EQ(got, 4);
    ASSERT_EQ(in.get_message(&buf[0], 64UL, &t), 0);
    // A buffer too small: the size comes back negative and the message stays.
    out.send_message(&sx[0], (unsigned long)sl);
    t0 = audio_now_ns_();
    n = 0L;
    while (n == 0L && audio_now_ns_() - t0 < (long long)waitMs * 1000000LL) { n = in.get_message(&buf[0], 8UL, &t); if (n == 0L) audio_sleep_us_(1000LL); }
    ASSERT_EQ(n, 0L - sl);
    ASSERT_EQ(in.get_message(&buf[0], 64UL, &t), sl);
    // Callback mode.
    struct MidiMon mon;
    mon.n = 0; mon.notes = 0; mon.sysex_len = 0; mon.last_t = 0.0; mon.total_delta = 0.0; mon.bad_order = 0;
    in.set_callback(midi_mon, (?&MidiMon)&mon);
    for (int i = 0; i < 20; i = i + 1) out.send(MidiMsg::note_on(1, 40 + i, 90));
    out.send_message(&sx[0], (unsigned long)sl);
    t0 = audio_now_ns_();
    while (mon.n < 21 && audio_now_ns_() - t0 < (long long)waitMs * 1000000LL) audio_sleep_us_(1000LL);
    // The filter applies to what arrives from now on.
    in.ignore_types(1, 1, 1);
    out.send_message(&sx[0], (unsigned long)sl);
    audio_sleep_us_((long long)waitMs * 100LL);
    ASSERT_EQ(mon.n, 21);
    ASSERT_EQ(mon.notes, 20);
    ASSERT_EQ(mon.sysex_len, (int)sl);
    ASSERT_EQ(mon.bad_order, 0);
    ASSERT_TRUE(mon.total_delta >= 0.0);
    in.cancel_callback();
    ASSERT_EQ(in.dropped(), 0);
}

void t_midi_dummy() {
    int apis[MIDI_API_COUNT];
    int na = midi_compiled_apis(&apis[0], MIDI_API_COUNT);
    ASSERT_EQ(apis[na - 1], MIDI_API_DUMMY);
    ASSERT_EQ(midi_api_by_name("winmm"), MIDI_API_WINMM);
    ASSERT_STR_EQ(midi_api_display_name(MIDI_API_COREMIDI), "Core MIDI");

    struct MidiOut out = MidiOut::open(MIDI_API_DUMMY, "out").unwrap_or(MidiOut::closed());
    struct MidiIn in = MidiIn::open(MIDI_API_DUMMY, "in", 0UL).unwrap_or(MidiIn::closed());
    ASSERT_TRUE(out.is_open() && in.is_open());
    ASSERT_EQ(in.port_count(), 1);
    struct MidiPortInfo p0;
    p0.index = 0U; p0.name[0] = (char)0;
    p0 = in.port_info(0U).unwrap_or(p0);
    ASSERT_STR_EQ(&p0.name[0], "Dummy Loopback");
    ASSERT_FALSE(in.port_info(3U).ok());
    ASSERT_EQ(out.send(MidiMsg::note_on(0, 1, 1)).code, AUDIO_ERR_INVALID_USE);   // no port yet
    // Through the built-in loopback port.
    ASSERT_TRUE(out.open_port(0U, "o").ok());
    ASSERT_TRUE(in.open_port(0U, "i").ok());
    ASSERT_TRUE(in.is_port_open());
    ASSERT_EQ(in.open_port(0U, "again").code, AUDIO_ERR_INVALID_USE);
    midi_roundtrip_(&in, &out, 1000);
    in.close_port();
    out.close_port();
    // Through a virtual port the MidiOut publishes.
    ASSERT_TRUE(out.open_virtual_port("SafeC virtual").ok());
    ASSERT_EQ(in.port_count(), 2);
    int idx = -1;
    for (unsigned int i = 0U; i < in.port_count(); i = i + 1U) {
        struct MidiPortInfo pi;
        pi.index = 0U; pi.name[0] = (char)0;
        pi = in.port_info(i).unwrap_or(pi);
        if (audio_streq_(&pi.name[0], "SafeC virtual")) idx = (int)i;
    }
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(in.open_port((unsigned int)idx, "i").ok());
    midi_roundtrip_(&in, &out, 1000);
    in.close_port();
    out.close_port();
    ASSERT_EQ(in.port_count(), 1);                   // the virtual port is gone
    // A tiny queue overflows: dropped messages are counted.
    struct MidiIn small = MidiIn::open(MIDI_API_DUMMY, "small", 64UL).unwrap_or(MidiIn::closed());
    small.open_port(0U, "s");
    out.open_port(0U, "o");
    for (int i = 0; i < 20; i = i + 1) out.send(MidiMsg::note_on(0, i, 1));
    ASSERT_TRUE(small.dropped() > 0UL);
    unsigned char b[4];
    double t = 0.0;
    int kept = 0;
    while (small.get_message(&b[0], 4UL, &t) > 0L) kept = kept + 1;
    ASSERT_EQ((unsigned long)kept + small.dropped(), 20UL);
    small.close();
    in.close();
    out.close();
    unsafe { ASSERT_FALSE(in.is_open()); }   // close() leaves it closed
    // UNSPECIFIED opens something on any machine.
    struct Result<MidiIn, AudioError> any = MidiIn::open(MIDI_API_UNSPECIFIED, "any", 0UL);
    ASSERT_TRUE(any.ok());
    struct MidiIn ai = any.unwrap_or(MidiIn::closed());
    printf("    default MIDI API here: %s, %u input port(s)\n", midi_api_name(ai.api()), ai.port_count());
    ai.close();
}

void t_midi_system() {
#if defined(__linux__)
    // ALSA sequencer: a loopback through our own virtual port when there is a sequencer.
    struct Result<MidiOut, AudioError> ro = MidiOut::open(MIDI_API_ALSA, "safec_seq_out");
    if (!ro.ok()) {
        match (ro.get_err()) { case some(e): { ASSERT_EQ(e.code, AUDIO_ERR_UNAVAILABLE); printf("    SKIP alsa seq: %s\n", e.message); } case null: {} }
        ro.drop();
    } else {
        struct MidiOut out = ro.unwrap_or(MidiOut::closed());
        struct MidiIn in = MidiIn::open(MIDI_API_ALSA, "safec_seq_in", 0UL).unwrap_or(MidiIn::closed());
        ASSERT_TRUE(out.open_virtual_port("vout").ok());
        int idx = -1;
        for (unsigned int i = 0U; i < in.port_count(); i = i + 1U) {
            struct MidiPortInfo pi;
            pi.index = 0U; pi.name[0] = (char)0;
            pi = in.port_info(i).unwrap_or(pi);
            if (audio_strprefix_(&pi.name[0], "safec_seq_out:vout")) idx = (int)i;
        }
        ASSERT_TRUE(idx >= 0);
        if (idx >= 0) {
            ASSERT_TRUE(in.open_port((unsigned int)idx, "in").ok());
            midi_roundtrip_(&in, &out, 2000);
        }
        in.close();
        out.close();
    }
#endif
    // JACK MIDI: our MidiOut's virtual port, read by a MidiIn.
    if (!jack_up_()) return;
    struct MidiOut jo = MidiOut::open(MIDI_API_JACK, "safec_jmidi_out").unwrap_or(MidiOut::closed());
    struct MidiIn ji = MidiIn::open(MIDI_API_JACK, "safec_jmidi_in", 0UL).unwrap_or(MidiIn::closed());
    ASSERT_TRUE(jo.is_open() && ji.is_open());
    ASSERT_TRUE(jo.open_virtual_port("out").ok());
    int jidx = -1;
    for (unsigned int i = 0U; i < ji.port_count(); i = i + 1U) {
        struct MidiPortInfo pi;
        pi.index = 0U; pi.name[0] = (char)0;
        pi = ji.port_info(i).unwrap_or(pi);
        if (audio_streq_(&pi.name[0], "safec_jmidi_out:out")) jidx = (int)i;
    }
    ASSERT_TRUE(jidx >= 0);
    if (jidx >= 0) {
        ASSERT_TRUE(ji.open_port((unsigned int)jidx, "in").ok());
        midi_roundtrip_(&ji, &jo, 2000);
    }
    ji.close();
    jo.close();
}

int main() {
#if defined(__linux__)
    alsa_test_config_();       // before anything loads libasound's config
#endif
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("convert: formats", t_convert_formats);
        t.add("convert: dither, layouts, gain", t_convert_dither_layout);
        t.add("convert: dsp Fixed / AudioBuffer", t_convert_dsp);
        t.add("api registry, default open", t_api_registry);
        t.add("dummy: devices", t_dummy_devices);
        t.add("dummy: manual pump, loopback, stop/abort", t_dummy_manual);
        t.add("dummy: i16 planar, offsets, f64", t_dummy_layouts);
        t.add("dummy: input-only sine", t_dummy_input);
        t.add("dummy: timer thread", t_dummy_thread);
        t.add("stream errors", t_stream_errors);
#if defined(__linux__)
        t.add("alsa: probe, null/file PCMs", t_alsa);
#endif
        t.add("jack: audio loopback", t_jack_audio);
        t.add("midi: messages", t_midi_messages);
        t.add("midi: parser, encoder", t_midi_parser);
        t.add("midi: dummy loopback, virtual ports, queue", t_midi_dummy);
        t.add("midi: alsa sequencer, jack", t_midi_system);
    }
    t.run();
    t.print_summary();
    if (g_jack_started_) { unsafe { system("kill $(cat /tmp/safec_audio_test_jackd.pid) >/dev/null 2>&1"); } }
    return t.all_passed() ? 0 : 1;
}
