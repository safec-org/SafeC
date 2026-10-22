// SafeC Standard Library — audio backend: Web Audio (wasm32). See audio.h.
//
// A wasm32 module has no audio API of its own: every operation is an
// import from the JS glue std ships next to this file,
// std/audio/audio_web.js (load it, then pass SafeCAudio.imports(env) as the
// module's import object and SafeCAudio.attach(instance) once instantiated;
// see that file's header). The glue opens an AudioContext at the stream's
// rate and drives the callback either from an AudioWorklet — the worklet
// asks the main thread for each period through its MessagePort and plays
// from a small queue of rendered periods (AudioStreamOptions.buffer_count,
// default 3) — or, where AudioWorklet is missing, from a
// ScriptProcessorNode whose onaudioprocess calls straight into the module.
// Either way the callback runs on the page's main thread, in periods of the
// stream's buffer size; Web Audio's buffers are non-interleaved float32 and
// std converts to the stream's layout.
//
// Devices: one "Web Audio" device, output stereo and — when the page
// called SafeCAudio.enableInput() (getUserMedia) before probing — the
// microphone's channels as input. Browsers start an AudioContext only from
// a user gesture: open and start a stream from a click handler (or call
// SafeCAudio.resume() from one).
#pragma once
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define AUDIO_WEB_MAX_STREAMS 8

namespace std {

// ── imports (std/audio/audio_web.js) ─────────────────────────────────────
extern int  std_audio_web_available();
extern int  std_audio_web_default_rate();
extern int  std_audio_web_input_channels();
extern int  std_audio_web_output_channels();
// Opens the context and node for stream 'handle'; the period actually
// used (frames), or < 0 on failure.
extern int  std_audio_web_open(int handle, int outChannels, int inChannels, int rate, int frames, int buffers);
extern int  std_audio_web_start(int handle);
extern void std_audio_web_stop(int handle);
extern void std_audio_web_close(int handle);
extern int  std_audio_web_latency(int handle);

static struct AudioStreamState_* g_audio_web_streams_[AUDIO_WEB_MAX_STREAMS];

struct AudioWebStream_ {
    int handle;
    int opened;
};

static ?&AudioWebStream_ audio_web_of_(&AudioStreamState_ st) {
    unsafe { return (?&AudioWebStream_)st.backend; }
}

// ── exports (called by the glue) ─────────────────────────────────────────

// One period for stream 'handle': the glue wrote the input planes into
// std_audio_web_buffer(handle, 1) first and reads the output planes from
// std_audio_web_buffer(handle, 0) after. Returns AUDIO_CB_*; a stopped
// stream renders silence and returns AUDIO_CB_STOP.
__attribute__((export_name("std_audio_web_process")))
int audio_web_process(int handle, int frames) {
    if (handle < 0 || handle >= AUDIO_WEB_MAX_STREAMS) return AUDIO_CB_ABORT;
    unsafe {
        struct AudioStreamState_* st = g_audio_web_streams_[handle];
        if (st == (struct AudioStreamState_*)0) return AUDIO_CB_ABORT;
        unsigned int n = (unsigned int)frames;
        if (n > st->max_frames) n = st->max_frames;
        if (atomic_load((_Atomic int*)&st->state) != AUDIO_STREAM_RUNNING || st->stop_request != 0) {
            audio_stream_silence_((&AudioStreamState_)st, n);
            return AUDIO_CB_STOP;
        }
        int r = audio_stream_process_((&AudioStreamState_)st, n);
        if (r != AUDIO_CB_CONTINUE) audio_stream_mark_stopped_((&AudioStreamState_)st, null);
        return r;
    }
}

// The planar float32 buffer of direction 'dir' (0 out, 1 in): channels *
// max_frames floats, channel c at c * frames (frames = this period's).
__attribute__((export_name("std_audio_web_buffer")))
float* audio_web_buffer(int handle, int dir) {
    if (handle < 0 || handle >= AUDIO_WEB_MAX_STREAMS || dir < 0 || dir > 1) return (float*)0;
    unsafe {
        struct AudioStreamState_* st = g_audio_web_streams_[handle];
        if (st == (struct AudioStreamState_*)0) return (float*)0;
        return (float*)st->device_buf[dir];
    }
}

// The glue saw a glitch (a worklet queue that ran dry, dropped input).
__attribute__((export_name("std_audio_web_xrun")))
void audio_web_xrun(int handle, int flags) {
    if (handle < 0 || handle >= AUDIO_WEB_MAX_STREAMS) return;
    unsafe {
        struct AudioStreamState_* st = g_audio_web_streams_[handle];
        if (st != (struct AudioStreamState_*)0) audio_stream_flag_status_((&AudioStreamState_)st, (unsigned int)flags);
    }
}

// ── the backend ──────────────────────────────────────────────────────────

static int audio_webaudio_available_() {
    unsafe { return std_audio_web_available(); }
}

static int audio_webaudio_probe_(&AudioHost_ h) {
    if (!audio_webaudio_available_()) return AUDIO_OK;     // no AudioContext: zero devices
    struct AudioDeviceInfo info = audio_device_info_blank_();
    audio_strcpy_(&info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "Web Audio");
    unsafe {
        info.output_channels = std_audio_web_output_channels();
        info.input_channels = std_audio_web_input_channels();
        info.preferred_sample_rate = (unsigned int)std_audio_web_default_rate();
    }
    info.duplex_channels = info.output_channels < info.input_channels ? info.output_channels : info.input_channels;
    info.native_formats = AUDIO_FORMAT_F32;
    // An AudioContext can be created at any rate from 8 to 96 kHz.
    for (int i = 0; i < audio_standard_rate_count_(); i = i + 1) {
        unsigned int r = audio_standard_rate_(i);
        if (r >= 8000U && r <= 96000U) audio_device_add_rate_(&info, r);
    }
    audio_device_add_rate_(&info, info.preferred_sample_rate);
    info.is_default_output = info.output_channels > 0;
    info.is_default_input = info.input_channels > 0;
    match (audio_host_add_(h, &info, "webaudio:default", 0LL)) {
        case some(r): {
            if (r.info.is_default_output) h.default_out = r.info.id;
            if (r.info.is_default_input) h.default_in = r.info.id;
        }
        case null: { return audio_fail_(&h.error, AUDIO_ERR_NO_MEMORY, "webaudio: out of memory"); }
    }
    return AUDIO_OK;
}

static void audio_webaudio_host_close_(&AudioHost_ h) { }

static int audio_webaudio_open_(&AudioHost_ h, &AudioStreamState_ st) {
    int handle = -1;
    unsafe {
        for (int i = 0; i < AUDIO_WEB_MAX_STREAMS && handle < 0; i = i + 1)
            if (g_audio_web_streams_[i] == (struct AudioStreamState_*)0) handle = i;
    }
    if (handle < 0) return audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "webaudio: too many streams");
    ?&AudioWebStream_ wp = null;
    unsafe { wp = (?&AudioWebStream_)alloc_zeroed_raw_(sizeof(struct AudioWebStream_)); }
    match (wp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "webaudio: out of memory"); }
        case some(w): {
            w.handle = handle;
            int outCh = st.mode_out ? st.user_channels[AUDIO_DIR_OUT] + st.first_channel[AUDIO_DIR_OUT] : 0;
            int inCh = st.mode_in ? st.user_channels[AUDIO_DIR_IN] + st.first_channel[AUDIO_DIR_IN] : 0;
            int frames = (int)(st.buffer_frames == 0U ? 512U : st.buffer_frames);
            if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U) frames = 256;
            int nbuf = (int)(st.buffer_count == 0U ? 3U : st.buffer_count);
            int got = 0;
            unsafe { got = std_audio_web_open(handle, outCh, inCh, (int)st.sample_rate, frames, nbuf); }
            if (got <= 0) {
                unsafe { dealloc_raw_((void*)w); }
                return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "webaudio: could not open an AudioContext", (long long)got);
            }
            w.opened = 1;
            st.buffer_frames = (unsigned int)got;
            st.max_frames = (unsigned int)got;
            st.buffer_count = (unsigned int)nbuf;
            long lat = 0L;
            unsafe { lat = (long)std_audio_web_latency(handle); }
            for (int d = 0; d < 2; d = d + 1) {
                int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
                if (!on) continue;
                st.device_format[d] = AUDIO_FORMAT_F32;
                st.device_channels[d] = d == AUDIO_DIR_OUT ? outCh : inCh;
                st.device_interleaved[d] = 0;
                st.latency[d] = lat;
            }
            unsafe {
                st.backend = (void*)w;
                g_audio_web_streams_[handle] = (struct AudioStreamState_*)st;
            }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

static int audio_webaudio_start_(&AudioStreamState_ st) {
    match (audio_web_of_(st)) {
        case some(w): {
            int rc = 0;
            unsafe { rc = std_audio_web_start(w.handle); }
            if (rc != 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "webaudio: could not start", (long long)rc);
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "webaudio: stream not open");
}

static int audio_webaudio_stop_(&AudioStreamState_ st) {
    match (audio_web_of_(st)) {
        case some(w): { unsafe { std_audio_web_stop(w.handle); } }
        case null: {}
    }
    return AUDIO_OK;
}

static void audio_webaudio_close_(&AudioStreamState_ st) {
    match (audio_web_of_(st)) {
        case some(w): {
            unsafe {
                std_audio_web_close(w.handle);
                g_audio_web_streams_[w.handle] = (struct AudioStreamState_*)0;
                dealloc_raw_((void*)w);
            }
        }
        case null: {}
    }
    unsafe { st.backend = (void*)0; }
}

static struct AudioBackendVt_ g_audio_webaudio_vt_;
static int g_audio_webaudio_vt_ready_ = 0;

&static AudioBackendVt_ audio_webaudio_backend_() {
    if (!g_audio_webaudio_vt_ready_) {
        g_audio_webaudio_vt_.api = AUDIO_API_WEBAUDIO;
        g_audio_webaudio_vt_.available = audio_webaudio_available_;
        g_audio_webaudio_vt_.probe = audio_webaudio_probe_;
        g_audio_webaudio_vt_.host_close = audio_webaudio_host_close_;
        g_audio_webaudio_vt_.open = audio_webaudio_open_;
        g_audio_webaudio_vt_.start = audio_webaudio_start_;
        g_audio_webaudio_vt_.stop = audio_webaudio_stop_;
        g_audio_webaudio_vt_.abort = audio_webaudio_stop_;
        g_audio_webaudio_vt_.close = audio_webaudio_close_;
        g_audio_webaudio_vt_ready_ = 1;
    }
    return &g_audio_webaudio_vt_;
}

} // namespace std
