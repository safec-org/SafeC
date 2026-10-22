// SafeC Standard Library — audio backend: dummy (no device). See audio.h.
//
// Two fake devices — "Dummy Duplex" (8 in / 8 out) and "Dummy Output"
// (2 out) — that accept every format and the standard rates. A running
// stream calls the callback once per period from a timer thread paced by
// the monotonic clock, so a program sees the same callback cadence as on
// real hardware: for tests and CI machines without sound hardware. A late
// wake-up (more than one period behind) is reported as
// AUDIO_STATUS_OUTPUT_UNDERFLOW / _INPUT_OVERFLOW, as a real device would.
//
// What the "device" plays back is discarded; what it captures is
// deterministic: in a duplex stream, each period's input is the previous
// period's output (a loopback cable, channel for channel), and in an
// input-only stream a 440 Hz sine at half scale on every channel.
//
// With AUDIO_FLAG_DUMMY_MANUAL there is no thread: AudioStream::pump() runs
// one period on the caller's thread — the deterministic way to test a
// callback, and the only mode on targets without threads (wasm32).
#pragma once
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>

namespace std {

struct AudioDummyStream_ {
    int manual;
    double phase;            // input sine phase (radians)
    int have_previous;       // a previous output period exists (loopback)
};

static int audio_dummy_available_() { return 1; }

static void audio_dummy_fill_info_(&AudioDeviceInfo d, const &char name, int outs, int ins) {
    audio_strcpy_(&d.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, name);
    d.output_channels = outs;
    d.input_channels = ins;
    d.duplex_channels = outs < ins ? outs : ins;
    d.native_formats = AUDIO_FORMAT_I8 | AUDIO_FORMAT_I16 | AUDIO_FORMAT_I24 | AUDIO_FORMAT_I32 |
                       AUDIO_FORMAT_F32 | AUDIO_FORMAT_F64;
    for (int i = 0; i < audio_standard_rate_count_(); i = i + 1)
        audio_device_add_rate_(d, audio_standard_rate_(i));
    d.preferred_sample_rate = 48000U;
}

static int audio_dummy_probe_(&AudioHost_ h) {
    struct AudioDeviceInfo a = audio_device_info_blank_();
    audio_dummy_fill_info_(&a, "Dummy Duplex", 8, 8);
    a.is_default_output = 1;
    a.is_default_input = 1;
    ?&AudioDeviceRec_ ra = audio_host_add_(h, &a, "dummy:duplex", 0LL);
    struct AudioDeviceInfo b = audio_device_info_blank_();
    audio_dummy_fill_info_(&b, "Dummy Output", 2, 0);
    ?&AudioDeviceRec_ rb = audio_host_add_(h, &b, "dummy:output", 1LL);
    match (ra) {
        case some(r): { h.default_out = r.info.id; h.default_in = r.info.id; }
        case null: { return audio_fail_(&h.error, AUDIO_ERR_NO_MEMORY, "dummy: out of memory"); }
    }
    if (rb.is_null()) return audio_fail_(&h.error, AUDIO_ERR_NO_MEMORY, "dummy: out of memory");
    return AUDIO_OK;
}

static void audio_dummy_host_close_(&AudioHost_ h) { }

static ?&AudioDummyStream_ audio_dummy_of_(&AudioStreamState_ st) {
    unsafe { return (?&AudioDummyStream_)st.backend; }
}

static int audio_dummy_open_(&AudioHost_ h, &AudioStreamState_ st) {
    if (st.buffer_frames == 0U) st.buffer_frames = 256U;
    if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U) st.buffer_frames = 64U;
    if (st.buffer_frames > 65536U) st.buffer_frames = 65536U;
    st.max_frames = st.buffer_frames;
    if (st.buffer_count == 0U) st.buffer_count = 2U;
    for (int d = 0; d < 2; d = d + 1) {
        int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
        if (!on) continue;
        // The "hardware" runs f32 interleaved at the requested width.
        st.device_format[d] = AUDIO_FORMAT_F32;
        st.device_channels[d] = st.user_channels[d] + st.first_channel[d];
        st.device_interleaved[d] = 1;
        st.latency[d] = (long)st.buffer_frames * (long)st.buffer_count;
    }
    ?&AudioDummyStream_ dp = null;
    unsafe { dp = (?&AudioDummyStream_)alloc_zeroed_raw_(sizeof(struct AudioDummyStream_)); }
    match (dp) {
        case some(d): {
            d.manual = (st.flags & AUDIO_FLAG_DUMMY_MANUAL) != 0U;
#if defined(__wasm__)
            d.manual = 1;
#endif
            unsafe { st.backend = (void*)d; }
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "dummy: out of memory");
}

// sin(x) for x in [0, 2pi], no libm (wasm32 has none): fold to
// [-pi/2, pi/2] and a degree-11 Taylor polynomial (error < 1e-7 there).
static double audio_dummy_sin_(double x) {
    double pi = 3.14159265358979323846;
    double y = x;
    if (y > pi) y = y - 2.0 * pi;           // [-pi, pi]
    if (y > pi / 2.0) y = pi - y;
    if (y < 0.0 - pi / 2.0) y = 0.0 - pi - y;
    double y2 = y * y;
    return y * (1.0 - y2 / 6.0 * (1.0 - y2 / 20.0 * (1.0 - y2 / 42.0 * (1.0 - y2 / 72.0 * (1.0 - y2 / 110.0)))));
}

// What the device "captured" for the coming period.
static void audio_dummy_capture_(&AudioStreamState_ st, &AudioDummyStream_ d, unsigned int frames) {
    if (!st.mode_in) return;
    unsigned long ch = (unsigned long)st.device_channels[AUDIO_DIR_IN];
    unsafe {
        float* in = (float*)st.device_buf[AUDIO_DIR_IN];
        if (st.mode_out) {
            // Loopback: last period's output, channel for channel.
            float* out = (float*)st.device_buf[AUDIO_DIR_OUT];
            unsigned long och = (unsigned long)st.device_channels[AUDIO_DIR_OUT];
            for (unsigned long f = 0UL; f < (unsigned long)frames; f = f + 1UL)
                for (unsigned long c = 0UL; c < ch; c = c + 1UL)
                    in[f * ch + c] = (d.have_previous && c < och) ? out[f * och + c] : (float)0.0;
            return;
        }
        double step = 2.0 * 3.14159265358979323846 * 440.0 / (double)st.sample_rate;
        for (unsigned long f = 0UL; f < (unsigned long)frames; f = f + 1UL) {
            float v = (float)(0.5 * audio_dummy_sin_(d.phase));
            d.phase = d.phase + step;
            if (d.phase > 6.283185307179586) d.phase = d.phase - 6.283185307179586;
            for (unsigned long c = 0UL; c < ch; c = c + 1UL) in[f * ch + c] = v;
        }
    }
}

// One period: capture, callback (+ conversions), "play".
static int audio_dummy_period_(&AudioStreamState_ st, &AudioDummyStream_ d) {
    audio_dummy_capture_(st, d, st.buffer_frames);
    int r = audio_stream_process_(st, st.buffer_frames);
    d.have_previous = 1;
    return r;
}

static void audio_dummy_thread_(&AudioStreamState_ st) {
    match (audio_dummy_of_(st)) {
        case some(d): {
            long long period_ns = (long long)st.buffer_frames * 1000000000LL / (long long)st.sample_rate;
            long long next = audio_now_ns_();
            while (audio_stream_stop_requested_(st) == 0) {
                int r = audio_dummy_period_(st, d);
                if (r != AUDIO_CB_CONTINUE) break;
                next = next + period_ns;
                long long now = audio_now_ns_();
                if (now > next + period_ns) {
                    // More than a period late: the device would have under-/overrun.
                    unsigned int s = 0U;
                    if (st.mode_out) s = s | AUDIO_STATUS_OUTPUT_UNDERFLOW;
                    if (st.mode_in) s = s | AUDIO_STATUS_INPUT_OVERFLOW;
                    audio_stream_flag_status_(st, s);
                    next = now;
                } else if (next > now) {
                    audio_sleep_us_((next - now) / 1000LL);
                }
            }
        }
        case null: {}
    }
    audio_stream_mark_stopped_(st, null);
}

static int audio_dummy_start_(&AudioStreamState_ st) {
    match (audio_dummy_of_(st)) {
        case some(d): {
            d.have_previous = 0;
            if (d.manual) return AUDIO_OK;
            return audio_stream_thread_start_(st, audio_dummy_thread_);
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "dummy: stream not open");
}

static int audio_dummy_stop_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    return AUDIO_OK;
}

static void audio_dummy_close_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    unsafe { dealloc_raw_(st.backend); st.backend = (void*)0; }
}

int audio_dummy_pump_(&AudioStreamState_ st) {
    match (audio_dummy_of_(st)) {
        case some(d): {
            if (!d.manual) return -1;
            int state = 0;
            unsafe { state = atomic_load((_Atomic int*)&st.state); }
            if (state != AUDIO_STREAM_RUNNING) return -1;
            int r = audio_dummy_period_(st, d);
            if (r != AUDIO_CB_CONTINUE) audio_stream_mark_stopped_(st, null);
            return r;
        }
        case null: {}
    }
    return -1;
}

static struct AudioBackendVt_ g_audio_dummy_vt_;
static int g_audio_dummy_vt_ready_ = 0;

&static AudioBackendVt_ audio_dummy_backend_() {
    if (!g_audio_dummy_vt_ready_) {
        g_audio_dummy_vt_.api = AUDIO_API_DUMMY;
        g_audio_dummy_vt_.available = audio_dummy_available_;
        g_audio_dummy_vt_.probe = audio_dummy_probe_;
        g_audio_dummy_vt_.host_close = audio_dummy_host_close_;
        g_audio_dummy_vt_.open = audio_dummy_open_;
        g_audio_dummy_vt_.start = audio_dummy_start_;
        g_audio_dummy_vt_.stop = audio_dummy_stop_;
        g_audio_dummy_vt_.abort = audio_dummy_stop_;
        g_audio_dummy_vt_.close = audio_dummy_close_;
        g_audio_dummy_vt_ready_ = 1;
    }
    return &g_audio_dummy_vt_;
}

} // namespace std
