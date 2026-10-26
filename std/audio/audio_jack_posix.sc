// SafeC Standard Library — audio backend: JACK (Linux, macOS). See audio.h.
//
// libjack (libjack.so.0 / libjack.0.dylib) is loaded with dlopen() on first
// use and the server is never auto-started (JackNoStartServer): without a
// running server the API reports AUDIO_ERR_UNAVAILABLE, and open(AUDIO_API_
// UNSPECIFIED) moves on. Devices are JACK clients with audio ports —
// "system" (the hardware) and every application, including other SafeC
// streams — so one stream can play into another. A device's output
// channels are the client's input ports, its input channels its output
// ports. JACK runs at one sample rate and buffer size for the whole graph:
// a stream must ask for the server's rate, and gets its buffer size.
//
// Each stream is its own JACK client (named AudioStreamOptions.stream_name,
// "SafeC" by default) with ports out_1.. / in_1..; start() activates it and
// connects port k to the device's port first_channel + k, unless
// AUDIO_FLAG_JACK_DONT_CONNECT. The callback runs in JACK's process thread
// (already real-time). JACK's buffers are non-interleaved f32; std
// converts to the stream's layout. Server xruns arrive as
// AUDIO_STATUS_INPUT_OVERFLOW | AUDIO_STATUS_OUTPUT_UNDERFLOW.
#pragma once
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>
#include <std/audio/jack_lib.h>
#include <std/mem.h>

namespace std {

static int audio_jack_available_() {
    if (!audio_jack_load_()) return 0;
    JackClient_* c = audio_jack_client_open_("SafeC probe");
    if (c == (JackClient_*)0) return 0;
    unsafe { g_std_audio_jack_.client_close(c); }
    return 1;
}

// Client name of a full port name ("system:playback_1" -> "system").
static void audio_jack_client_of_(&char out, unsigned long cap, const &char port) {
    audio_strcpy_(out, cap, port);
    unsafe {
        char* p = (char*)out;
        for (unsigned long i = 0UL; i < cap && p[i] != (char)0; i = i + 1UL) {
            if (p[i] == ':') { p[i] = (char)0; return; }
        }
    }
}

// Ports of client 'client' with JACK flag 'flag' (JackPortIsInput/Output),
// in JACK's order. The list must be freed with g_std_audio_jack_.free.
static const char** audio_jack_ports_of_(JackClient_* jc, const &char client, unsigned long flag, const &char type) {
    char pattern[AUDIO_DEVICE_KEY_MAX];
    audio_strcpy_(&pattern[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, "^");
    // Escape regex metacharacters in the client name.
    unsafe {
        unsigned long n = audio_strlen_(client);
        unsigned long k = 1UL;
        for (unsigned long i = 0UL; i < n && k + 3UL < (unsigned long)AUDIO_DEVICE_KEY_MAX; i = i + 1UL) {
            char ch = client[i];
            if (ch == '.' || ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == '*' || ch == '+' ||
                ch == '?' || ch == '^' || ch == '$' || ch == '|' || ch == '\\' || ch == '{' || ch == '}') {
                pattern[k] = '\\';
                k = k + 1UL;
            }
            pattern[k] = ch;
            k = k + 1UL;
        }
        pattern[k] = (char)0;
        audio_strcat_(&pattern[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, ":");
        return g_std_audio_jack_.get_ports(jc, (const char*)&pattern[0], (const char*)type, flag);
    }
}

static int audio_jack_count_(const char** list) {
    int n = 0;
    unsafe {
        if (list == (const char**)0) return 0;
        while (list[n] != (const char*)0) n = n + 1;
    }
    return n;
}

static int audio_jack_probe_(&AudioHost_ h) {
    if (!audio_jack_load_()) return audio_fail_(&h.error, AUDIO_ERR_UNAVAILABLE, "jack: libjack not found");
    JackClient_* jc = audio_jack_client_open_("SafeC probe");
    if (jc == (JackClient_*)0) return audio_fail_(&h.error, AUDIO_ERR_UNAVAILABLE, "jack: server not running");
    unsigned int rate = 0U;
    unsafe {
        rate = g_std_audio_jack_.get_sample_rate(jc);
        const char** all = g_std_audio_jack_.get_ports(jc, (const char*)0, (const char*)AUDIO_JACK_AUDIO_TYPE, 0UL);
        int n = audio_jack_count_(all);
        // One device per client, in the order JACK lists their ports.
        for (int i = 0; i < n; i = i + 1) {
            char client[AUDIO_DEVICE_NAME_MAX];
            audio_jack_client_of_(&client[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, (const &char)all[i]);
            int seen = 0;
            for (int j = 0; j < i && !seen; j = j + 1) {
                char other[AUDIO_DEVICE_NAME_MAX];
                audio_jack_client_of_(&other[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, (const &char)all[j]);
                if (audio_streq_(&client[0], &other[0])) seen = 1;
            }
            if (seen) continue;
            const char** ins = audio_jack_ports_of_(jc, &client[0], AUDIO_JACK_IS_INPUT, AUDIO_JACK_AUDIO_TYPE);
            const char** outs = audio_jack_ports_of_(jc, &client[0], AUDIO_JACK_IS_OUTPUT, AUDIO_JACK_AUDIO_TYPE);
            struct AudioDeviceInfo info = audio_device_info_blank_();
            audio_strcpy_(&info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, &client[0]);
            info.output_channels = audio_jack_count_(ins);     // we write into its input ports
            info.input_channels = audio_jack_count_(outs);     // and read its output ports
            info.duplex_channels = info.output_channels < info.input_channels ? info.output_channels : info.input_channels;
            info.native_formats = AUDIO_FORMAT_F32;
            audio_device_add_rate_(&info, rate);
            info.preferred_sample_rate = rate;
            int isSystem = audio_streq_(&client[0], "system");
            info.is_default_output = isSystem && info.output_channels > 0;
            info.is_default_input = isSystem && info.input_channels > 0;
            if (ins != (const char**)0) g_std_audio_jack_.free(ins);
            if (outs != (const char**)0) g_std_audio_jack_.free(outs);
            match (audio_host_add_(h, &info, &client[0], 0LL)) {
                case some(r): {
                    if (r.info.is_default_output) h.default_out = r.info.id;
                    if (r.info.is_default_input) h.default_in = r.info.id;
                }
                case null: {}
            }
        }
        if (all != (const char**)0) g_std_audio_jack_.free(all);
        g_std_audio_jack_.client_close(jc);
    }
    return AUDIO_OK;
}

static void audio_jack_host_close_(&AudioHost_ h) { }

// ── streams ──────────────────────────────────────────────────────────────

#define AUDIO_JACK_MAX_PORTS 64

struct JackStream_ {
    JackClient_* client;
    JackPort_* ports[2][AUDIO_JACK_MAX_PORTS];
    int nports[2];
    int offset[2];               // the stream's first_channel per direction
    int active;
    int stopped;                 // (rt) the process callback outputs silence
    int bad_size;                // (rt) JACK's buffer grew past max_frames
    char target[2][AUDIO_DEVICE_KEY_MAX];   // client to connect to
    struct AudioStreamState_* st;
};

static ?&JackStream_ audio_jack_of_(&AudioStreamState_ st) {
    unsafe { return (?&JackStream_)st.backend; }
}

static int audio_jack_load_flag_(&int p) {
    unsafe { return atomic_load((_Atomic int*)p); }
}

// JACK's process thread.
static int audio_jack_process_(unsigned int nframes, ?&Erased arg) {
    unsafe {
        struct JackStream_* j = (struct JackStream_*)arg;
        struct AudioStreamState_* st = j->st;
        int running = atomic_load((_Atomic int*)&st->state) == AUDIO_STREAM_RUNNING &&
                      atomic_load((_Atomic int*)&j->stopped) == 0;
        if (running && nframes > st->max_frames) {
            atomic_store((_Atomic int*)&j->bad_size, 1);
            audio_stream_set_rt_error_((&AudioStreamState_)st, AUDIO_ERR_UNSUPPORTED, "jack: buffer size grew while running");
            running = 0;
        }
        if (!running) {
            for (int p = 0; p < j->nports[AUDIO_DIR_OUT]; p = p + 1) {
                float* b = (float*)g_std_audio_jack_.port_get_buffer(j->ports[AUDIO_DIR_OUT][p], nframes);
                for (unsigned int f = 0U; f < nframes; f = f + 1U) b[f] = (float)0.0;
            }
            return 0;
        }
        unsigned long n = (unsigned long)nframes;
        if (st->mode_in) {
            float* dst = (float*)st->device_buf[AUDIO_DIR_IN].ptr;
            for (int p = 0; p < j->nports[AUDIO_DIR_IN]; p = p + 1) {
                const float* b = (const float*)g_std_audio_jack_.port_get_buffer(j->ports[AUDIO_DIR_IN][p], nframes);
                for (unsigned long f = 0UL; f < n; f = f + 1UL) dst[(unsigned long)p * n + f] = b[f];
            }
        }
        int r = audio_stream_process_((&AudioStreamState_)st, nframes);
        if (st->mode_out) {
            const float* src = (const float*)st->device_buf[AUDIO_DIR_OUT].ptr;
            for (int p = 0; p < j->nports[AUDIO_DIR_OUT]; p = p + 1) {
                float* b = (float*)g_std_audio_jack_.port_get_buffer(j->ports[AUDIO_DIR_OUT][p], nframes);
                for (unsigned long f = 0UL; f < n; f = f + 1UL) b[f] = src[(unsigned long)p * n + f];
            }
        }
        if (r != AUDIO_CB_CONTINUE) {
            // JACK can't deactivate from its own thread: go silent and
            // report stopped; stop()/close() deactivate the client.
            atomic_store((_Atomic int*)&j->stopped, 1);
            audio_stream_mark_stopped_((&AudioStreamState_)st, null);
        }
    }
    return 0;
}

static int audio_jack_xrun_(?&Erased arg) {
    unsafe {
        struct JackStream_* j = (struct JackStream_*)arg;
        audio_stream_flag_status_((&AudioStreamState_)j->st, AUDIO_STATUS_INPUT_OVERFLOW | AUDIO_STATUS_OUTPUT_UNDERFLOW);
    }
    return 0;
}

static void audio_jack_shutdown_(?&Erased arg) {
    unsafe {
        struct JackStream_* j = (struct JackStream_*)arg;
        atomic_store((_Atomic int*)&j->stopped, 1);
        audio_stream_set_rt_error_((&AudioStreamState_)j->st, AUDIO_ERR_SYSTEM, "jack: the server shut down");
        audio_stream_mark_stopped_((&AudioStreamState_)j->st, null);
    }
}

static void audio_jack_release_(&JackStream_ j) {
    unsafe {
        if (j.client != (JackClient_*)0) {
            if (j.active) g_std_audio_jack_.deactivate(j.client);
            g_std_audio_jack_.client_close(j.client);
        }
        j.client = (JackClient_*)0;
        j.active = 0;
    }
}

static int audio_jack_open_(&AudioHost_ h, &AudioStreamState_ st) {
    if (!audio_jack_load_()) return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "jack: libjack not found");
    ?&JackStream_ jp = null;
    unsafe { jp = (?&JackStream_)alloc_zeroed_raw_(sizeof(struct JackStream_)); }
    match (jp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "jack: out of memory"); }
        case some(j): {
            unsafe { j.st = (struct AudioStreamState_*)st; }
            j.client = audio_jack_client_open_(&st.stream_name[0]);
            if (j.client == (JackClient_*)0) {
                unsafe { dealloc((?&heap JackStream_)j); }
                return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "jack: server not running");
            }
            int rc = AUDIO_OK;
            unsafe {
                unsigned int rate = g_std_audio_jack_.get_sample_rate(j.client);
                if (rate != st.sample_rate) {
                    rc = audio_fail_code_(&st.error, AUDIO_ERR_INVALID_PARAM, "jack: the server runs at another sample rate",
                                          (long long)rate);
                }
                unsigned int bs = g_std_audio_jack_.get_buffer_size(j.client);
                st.buffer_frames = bs;
                st.max_frames = bs;
                st.buffer_count = 2U;
                for (int d = 0; d < 2 && rc == AUDIO_OK; d = d + 1) {
                    int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
                    if (!on) continue;
                    int n = st.user_channels[d];
                    if (n > AUDIO_JACK_MAX_PORTS) { rc = audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "jack: too many channels"); break; }
                    for (int p = 0; p < n; p = p + 1) {
                        char pname[32];
                        audio_strcpy_(&pname[0], 32UL, d == AUDIO_DIR_OUT ? "out_" : "in_");
                        audio_strcat_int_(&pname[0], 32UL, (long long)(p + 1));
                        JackPort_* port = g_std_audio_jack_.port_register(j.client, (const char*)&pname[0], (const char*)AUDIO_JACK_AUDIO_TYPE,
                                                           d == AUDIO_DIR_OUT ? AUDIO_JACK_IS_OUTPUT : AUDIO_JACK_IS_INPUT, 0UL);
                        if (port == (JackPort_*)0) { rc = audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "jack: could not register a port"); break; }
                        j.ports[d][p] = port;
                        j.nports[d] = p + 1;
                    }
                    // JACK buffers: one f32 plane per port; the channel offset
                    // picks which device ports we connect to, not a buffer offset.
                    j.offset[d] = st.first_channel[d];
                    st.first_channel[d] = 0;
                    st.device_format[d] = AUDIO_FORMAT_F32;
                    st.device_channels[d] = n;
                    st.device_interleaved[d] = 0;
                    st.latency[d] = (long)bs;
                    if (st.device_name[0] != (char)0) audio_strcpy_(&j.target[d][0], (unsigned long)AUDIO_DEVICE_KEY_MAX, &st.device_name[0]);
                    else audio_strcpy_(&j.target[d][0], (unsigned long)AUDIO_DEVICE_KEY_MAX, &st.device_key[d][0]);
                }
                if (rc == AUDIO_OK) {
                    g_std_audio_jack_.set_process_callback(j.client, audio_jack_process_, (?&Erased)j);
                    g_std_audio_jack_.set_xrun_callback(j.client, audio_jack_xrun_, (?&Erased)j);
                    g_std_audio_jack_.on_shutdown(j.client, audio_jack_shutdown_, (?&Erased)j);
                }
            }
            if (rc != AUDIO_OK) {
                audio_jack_release_(j);
                unsafe { dealloc((?&heap JackStream_)j); }
                return rc;
            }
            unsafe { st.backend = (?&heap AudioStreamImpl_)j; }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

// Connects our ports to the target client's, from the offset on.
static int audio_jack_connect_(&AudioStreamState_ st, &JackStream_ j) {
    if ((st.flags & AUDIO_FLAG_JACK_DONT_CONNECT) != 0U) return AUDIO_OK;
    for (int d = 0; d < 2; d = d + 1) {
        if (j.nports[d] == 0 || j.target[d][0] == (char)0) continue;
        unsafe {
            unsigned long flag = d == AUDIO_DIR_OUT ? AUDIO_JACK_IS_INPUT : AUDIO_JACK_IS_OUTPUT;
            const char** targets = audio_jack_ports_of_(j.client, &j.target[d][0], flag, AUDIO_JACK_AUDIO_TYPE);
            int n = audio_jack_count_(targets);
            for (int p = 0; p < j.nports[d]; p = p + 1) {
                int t = j.offset[d] + p;
                if (t >= n) break;
                const char* mine = g_std_audio_jack_.port_name(j.ports[d][p]);
                int rc = d == AUDIO_DIR_OUT ? g_std_audio_jack_.connect(j.client, mine, targets[t])
                                            : g_std_audio_jack_.connect(j.client, targets[t], mine);
                (void)rc;   // EEXIST (already connected) is fine
            }
            // Latency: what JACK reports for our first port's connections.
            if (j.nports[d] > 0) {
                unsigned int range[2];
                range[0] = 0U; range[1] = 0U;
                g_std_audio_jack_.port_get_latency_range(j.ports[d][0], d == AUDIO_DIR_OUT ? 1 : 0, &range[0]);
                st.latency[d] = (long)st.buffer_frames + (long)range[1];
            }
            if (targets != (const char**)0) g_std_audio_jack_.free(targets);
        }
    }
    return AUDIO_OK;
}

static int audio_jack_start_(&AudioStreamState_ st) {
    match (audio_jack_of_(st)) {
        case some(j): {
            unsafe {
                atomic_store((_Atomic int*)&j.stopped, 0);
                atomic_store((_Atomic int*)&j.bad_size, 0);
            }
            if (j.active) return AUDIO_OK;    // restarted after the callback stopped it
            int rc = 0;
            unsafe { rc = g_std_audio_jack_.activate(j.client); }
            if (rc != 0) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "jack: could not activate the client");
            j.active = 1;
            return audio_jack_connect_(st, j);
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "jack: stream not open");
}

static int audio_jack_stop_(&AudioStreamState_ st) {
    match (audio_jack_of_(st)) {
        case some(j): {
            unsafe { atomic_store((_Atomic int*)&j.stopped, 1); }
            if (j.active) {
                // jack_deactivate waits for the running cycle: the last
                // buffer the callback produced is played.
                unsafe { g_std_audio_jack_.deactivate(j.client); }
                j.active = 0;
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_OK;
}

static void audio_jack_close_(&AudioStreamState_ st) {
    match (audio_jack_of_(st)) {
        case some(j): {
            audio_jack_release_(j);
            unsafe { dealloc((?&heap JackStream_)j); }
        }
        case null: {}
    }
    unsafe { st.backend = null; }
}

static struct AudioBackendVt_ g_audio_jack_vt_;
static int g_audio_jack_vt_ready_ = 0;

&static AudioBackendVt_ audio_jack_backend_() {
    if (!g_audio_jack_vt_ready_) {
        g_audio_jack_vt_.api = AUDIO_API_JACK;
        g_audio_jack_vt_.available = audio_jack_available_;
        g_audio_jack_vt_.probe = audio_jack_probe_;
        g_audio_jack_vt_.host_close = audio_jack_host_close_;
        g_audio_jack_vt_.open = audio_jack_open_;
        g_audio_jack_vt_.start = audio_jack_start_;
        g_audio_jack_vt_.stop = audio_jack_stop_;
        g_audio_jack_vt_.abort = audio_jack_stop_;
        g_audio_jack_vt_.close = audio_jack_close_;
        g_audio_jack_vt_ready_ = 1;
    }
    return &g_audio_jack_vt_;
}

} // namespace std
