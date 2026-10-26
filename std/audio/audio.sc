// SafeC Standard Library — real-time audio I/O: the API-independent core
// (see audio.h): the API registry, device lists, stream setup, the
// real-time conversion/callback step every backend calls, stream control.
// safeguard: implements <std/audio/audio_backend.h>
#pragma once
#include <std/audio/audio.h>
#include <std/audio/audio_backend.h>
#include <std/audio/audio_convert.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>
#include <std/thread.h>

namespace std {

// ── formats & API names ──────────────────────────────────────────────────

int audio_format_bytes(unsigned int format) {
    if (format == AUDIO_FORMAT_I8) return 1;
    if (format == AUDIO_FORMAT_I16) return 2;
    if (format == AUDIO_FORMAT_I24) return 3;
    if (format == AUDIO_FORMAT_I32) return 4;
    if (format == AUDIO_FORMAT_F32) return 4;
    if (format == AUDIO_FORMAT_F64) return 8;
    return 0;
}

const &char audio_format_name(unsigned int format) {
    if (format == AUDIO_FORMAT_I8) return "i8";
    if (format == AUDIO_FORMAT_I16) return "i16";
    if (format == AUDIO_FORMAT_I24) return "i24";
    if (format == AUDIO_FORMAT_I32) return "i32";
    if (format == AUDIO_FORMAT_F32) return "f32";
    if (format == AUDIO_FORMAT_F64) return "f64";
    return "";
}

const &char audio_api_name(int api) {
    if (api == AUDIO_API_ALSA) return "alsa";
    if (api == AUDIO_API_JACK) return "jack";
    if (api == AUDIO_API_COREAUDIO) return "coreaudio";
    if (api == AUDIO_API_WASAPI) return "wasapi";
    if (api == AUDIO_API_WEBAUDIO) return "webaudio";
    if (api == AUDIO_API_DUMMY) return "dummy";
    return "";
}

const &char audio_api_display_name(int api) {
    if (api == AUDIO_API_ALSA) return "ALSA";
    if (api == AUDIO_API_JACK) return "JACK";
    if (api == AUDIO_API_COREAUDIO) return "Core Audio";
    if (api == AUDIO_API_WASAPI) return "WASAPI";
    if (api == AUDIO_API_WEBAUDIO) return "Web Audio";
    if (api == AUDIO_API_DUMMY) return "Dummy";
    return "";
}

int audio_api_by_name(const &char name) {
    for (int a = 1; a < AUDIO_API_COUNT; a = a + 1)
        if (audio_streq_(name, audio_api_name(a))) return a;
    return AUDIO_API_UNSPECIFIED;
}

// ── the registry ─────────────────────────────────────────────────────────

// The compiled backends, in preference order.
static int audio_backends_(&?&AudioBackendVt_ out, int cap) {
    int n = 0;
#if defined(__APPLE__)
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_coreaudio_backend_(); } n = n + 1; }
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_jack_backend_(); } n = n + 1; }
#endif
#if defined(__linux__)
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_alsa_backend_(); } n = n + 1; }
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_jack_backend_(); } n = n + 1; }
#endif
#if defined(_WIN32)
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_wasapi_backend_(); } n = n + 1; }
#endif
#if defined(__wasm__)
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_webaudio_backend_(); } n = n + 1; }
#endif
    if (n < cap) { unsafe { out[n] = (?&AudioBackendVt_)audio_dummy_backend_(); } n = n + 1; }
    return n;
}

static ?&AudioBackendVt_ audio_backend_for_(int api) {
    ?&AudioBackendVt_ list[AUDIO_API_COUNT];
    int n = audio_backends_(&list[0], AUDIO_API_COUNT);
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): { if (vt.api == api) return (?&AudioBackendVt_)vt; }
            case null: {}
        }
    }
    return null;
}

int audio_compiled_apis([]int out) {
    ?&AudioBackendVt_ list[AUDIO_API_COUNT];
    int n = audio_backends_(&list[0], AUDIO_API_COUNT);
    int k = 0;
    for (int i = 0; i < n && (unsigned long)k < out.len; i = i + 1) {
        match (list[i]) {
            case some(vt): { out[k] = vt.api; k = k + 1; }
            case null: {}
        }
    }
    return k;
}

int audio_api_compiled(int api) { return !audio_backend_for_(api).is_null(); }

static int audio_backend_available_(&AudioBackendVt_ vt) {
    int r = 0;
    unsafe { r = vt.available(); }
    return r;
}

int audio_default_api() {
    ?&char env = audio_getenv_("SAFEC_AUDIO_API");
    match (env) {
        case some(e): {
            int a = audio_api_by_name(e);
            if (a != AUDIO_API_UNSPECIFIED && audio_api_compiled(a)) return a;
        }
        case null: {}
    }
    ?&AudioBackendVt_ list[AUDIO_API_COUNT];
    int n = audio_backends_(&list[0], AUDIO_API_COUNT);
    // JACK only when its server is up (it's a choice the user made then).
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): {
                if (vt.api == AUDIO_API_JACK && audio_backend_available_(vt)) return AUDIO_API_JACK;
            }
            case null: {}
        }
    }
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): {
                if (vt.api != AUDIO_API_JACK && vt.api != AUDIO_API_DUMMY) return vt.api;
            }
            case null: {}
        }
    }
    return AUDIO_API_DUMMY;
}

// ── device lists ─────────────────────────────────────────────────────────

int AudioDeviceInfo::supports_rate(unsigned int rate) const {
    for (int i = 0; i < self.sample_rate_count && i < AUDIO_MAX_SAMPLE_RATES; i = i + 1)
        if (self.sample_rates[i] == rate) return 1;
    return 0;
}

struct AudioDeviceInfo AudioDeviceInfo::none() {
    struct AudioDeviceInfo d;
    unsafe { safe_memset_raw_((void*)&d, 0, sizeof(struct AudioDeviceInfo)); }
    return d;
}

struct AudioDeviceInfo audio_device_info_blank_() { return AudioDeviceInfo::none(); }

void audio_device_add_rate_(&AudioDeviceInfo info, unsigned int rate) {
    int n = info.sample_rate_count;
    for (int i = 0; i < n; i = i + 1) if (info.sample_rates[i] == rate) return;
    if (n >= AUDIO_MAX_SAMPLE_RATES) return;
    int at = n;
    while (at > 0 && info.sample_rates[at - 1] > rate) {
        info.sample_rates[at] = info.sample_rates[at - 1];
        at = at - 1;
    }
    info.sample_rates[at] = rate;
    info.sample_rate_count = n + 1;
}

static const unsigned int kAudioStdRates_[14] = {
    4000U, 5512U, 8000U, 9600U, 11025U, 16000U, 22050U, 32000U, 44100U, 48000U, 88200U, 96000U,
    176400U, 192000U
};

int audio_standard_rate_count_() { return 14; }
unsigned int audio_standard_rate_(int i) {
    if (i >= 0 && i < 14) return kAudioStdRates_[i];
    return 0U;
}


void audio_host_clear_(&AudioHost_ h) {
    h.count = 0U;
    h.default_out = AUDIO_DEFAULT_DEVICE;
    h.default_in = AUDIO_DEFAULT_DEVICE;
}

static unsigned int audio_host_id_for_(&AudioHost_ h, const &char key) {
    unsafe {
        for (unsigned int i = 0U; i < h.known_count; i = i + 1U)
            if (audio_streq_(&h.known[i].key[0], key)) return h.known[i].id;
        if (h.known_count >= h.known_cap) {
            unsigned int ncap = h.known_cap == 0U ? 8U : h.known_cap * 2U;
            void* nk = realloc_buf_raw_((void*)h.known, (unsigned long)ncap * sizeof(struct AudioKnownId_));
            if (nk == (void*)0) return 0U;
            h.known = (struct AudioKnownId_*)nk;
            h.known_cap = ncap;
        }
        h.next_id = h.next_id + 1U;
        struct AudioKnownId_* k = &h.known[h.known_count];
        audio_strcpy_(&k->key[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, key);
        k->id = h.next_id;
        h.known_count = h.known_count + 1U;
        return h.next_id;
    }
}

?&AudioDeviceRec_ audio_host_add_(&AudioHost_ h, &AudioDeviceInfo info, const &char key, long long native) {
    unsigned int id = audio_host_id_for_(h, key);
    if (id == 0U) return null;
    unsafe {
        if (h.count >= h.cap) {
            unsigned int ncap = h.cap == 0U ? 8U : h.cap * 2U;
            void* nd = realloc_buf_raw_((void*)h.devs, (unsigned long)ncap * sizeof(struct AudioDeviceRec_));
            if (nd == (void*)0) return null;
            h.devs = (struct AudioDeviceRec_*)nd;
            h.cap = ncap;
        }
        struct AudioDeviceRec_* r = &h.devs[h.count];
        r->info = *info;
        r->info.id = id;
        audio_strcpy_(&r->key[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, key);
        r->native = native;
        h.count = h.count + 1U;
        return (?&AudioDeviceRec_)r;
    }
}

static ?&AudioDeviceRec_ audio_host_find_(&AudioHost_ h, unsigned int id) {
    unsafe {
        for (unsigned int i = 0U; i < h.count; i = i + 1U)
            if (h.devs[i].info.id == id) return (?&AudioDeviceRec_)&h.devs[i];
    }
    return null;
}

// ── Audio ────────────────────────────────────────────────────────────────

struct Audio Audio::closed() {
    struct Audio a;
    a.api_id = AUDIO_API_UNSPECIFIED;
    a.impl = null;
    return a;
}

static ?&AudioHost_ audio_host_of_(const &Audio a) {
    return a.impl;
}

static int audio_host_probe_(&AudioHost_ h) {
    ?&AudioBackendVt_ vt = audio_backend_for_(h.api);
    match (vt) {
        case some(v): {
            audio_host_clear_(h);
            h.error = AudioError::none();
            int rc = 0;
            unsafe { rc = v.probe(h); }
            return rc;
        }
        case null: {}
    }
    return audio_fail_(&h.error, AUDIO_ERR_UNSUPPORTED, "audio API not compiled in");
}

static struct Result<Audio, AudioError> audio_open_api_(int api) {
    ?&AudioBackendVt_ vt = audio_backend_for_(api);
    if (vt.is_null()) {
        struct AudioError e = AudioError::make2(AUDIO_ERR_UNSUPPORTED, "audio API not compiled in",
                                                audio_api_name(api));
        return Result::new_err(e);
    }
    ?&AudioHost_ hp = null;
    unsafe { hp = (?&AudioHost_)alloc_zeroed_raw_(sizeof(struct AudioHost_)); }
    match (hp) {
        case some(h): {
            h.api = api;
            int rc = audio_host_probe_(h);
            if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) {
                struct AudioError e = h.error;
                unsafe {
                    match (vt) { case some(v): { v.host_close(h); } case null: {} }
                    dealloc_raw_((void*)h.devs);
                    dealloc_raw_((void*)h.known);
                    dealloc_raw_((void*)h);
                }
                return Result::new_err(e);
            }
            struct Audio a;
            a.api_id = api;
            a.impl = h;
            return Result::new_ok(a);
        }
        case null: {}
    }
    return Result::new_err(AudioError::make(AUDIO_ERR_NO_MEMORY, "out of memory"));
}

struct Result<Audio, AudioError> Audio::open(int api) {
    if (api != AUDIO_API_UNSPECIFIED) return audio_open_api_(api);
    // An explicit SAFEC_AUDIO_API wins.
    ?&char env = audio_getenv_("SAFEC_AUDIO_API");
    match (env) {
        case some(e): {
            int a = audio_api_by_name(e);
            if (a != AUDIO_API_UNSPECIFIED && audio_api_compiled(a)) return audio_open_api_(a);
        }
        case null: {}
    }
    // The default API, then the first other one with devices, then the
    // first that opened at all (no devices), then dummy.
    int first = audio_default_api();
    struct Result<Audio, AudioError> r = audio_open_api_(first);
    int firstHasDevices = 0;
    match (r.get_ok()) { case some(a): { firstHasDevices = a.device_count() > 0U; } case null: {} }
    if (firstHasDevices) return r;
    ?&AudioBackendVt_ list[AUDIO_API_COUNT];
    int n = audio_backends_(&list[0], AUDIO_API_COUNT);
    for (int i = 0; i < n; i = i + 1) {
        int api2 = 0;
        int avail = 0;
        match (list[i]) {
            case some(vt): { api2 = vt.api; avail = audio_backend_available_(vt); }
            case null: {}
        }
        if (api2 == first || api2 == AUDIO_API_DUMMY || !avail) continue;
        struct Result<Audio, AudioError> r2 = audio_open_api_(api2);
        int hasDevices = 0;
        match (r2.get_ok()) { case some(a2): { hasDevices = a2.device_count() > 0U; } case null: {} }
        if (hasDevices) {
            r.drop();   // the first API's Audio (opened, no devices)
            return r2;
        }
        r2.drop();
    }
    if (r.ok()) return r;
    r.drop();
    return audio_open_api_(AUDIO_API_DUMMY);
}

int Audio::is_open() const { return !self.impl.is_null(); }
int Audio::api() const { return self.api_id; }

struct AudioError Audio::rescan() {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) {
        case some(h): {
            int rc = audio_host_probe_(h);
            if (rc != AUDIO_OK) return h.error;
            return AudioError::none();
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: not open");
}

unsigned int Audio::device_count() const {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) { case some(h): return h.count; case null: {} }
    return 0U;
}

struct Result<AudioDeviceInfo, AudioError> Audio::device_info(unsigned int index) const {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) {
        case some(h): {
            if (index < h.count) {
                struct AudioDeviceInfo info;
                unsafe { info = h.devs[index].info; }
                return Result::new_ok(info);
            }
            return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_DEVICE, "audio: device index out of range"));
        }
        case null: {}
    }
    return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_USE, "audio: not open"));
}

struct Result<AudioDeviceInfo, AudioError> Audio::device_by_id(unsigned int id) const {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) {
        case some(h): {
            match (audio_host_find_(h, id)) {
                case some(r): return Result::new_ok(r.info);
                case null: {}
            }
            return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_DEVICE, "audio: no device with that id"));
        }
        case null: {}
    }
    return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_USE, "audio: not open"));
}

unsigned int Audio::default_output_device() const {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) { case some(h): return h.default_out; case null: {} }
    return AUDIO_DEFAULT_DEVICE;
}

unsigned int Audio::default_input_device() const {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) { case some(h): return h.default_in; case null: {} }
    return AUDIO_DEFAULT_DEVICE;
}

void Audio::close() { self.drop(); }

void Audio::drop() {
    ?&AudioHost_ hp = audio_host_of_(self);
    match (hp) {
        case some(h): {
            match (audio_backend_for_(h.api)) {
                case some(v): { unsafe { v.host_close(h); } }
                case null: {}
            }
            unsafe {
                dealloc_raw_((void*)h.devs);
                dealloc_raw_((void*)h.known);
                dealloc_raw_((void*)h);
            }
        }
        case null: {}
    }
    self.impl = null;
    self.api_id = AUDIO_API_UNSPECIFIED;
}

// ── parameters ───────────────────────────────────────────────────────────

struct AudioStreamParams AudioStreamParams::of(unsigned int device, int channels) {
    struct AudioStreamParams p;
    p.device = device;
    p.channels = channels;
    p.first_channel = 0;
    return p;
}

struct AudioStreamOptions AudioStreamOptions::none() {
    struct AudioStreamOptions o;
    unsafe { safe_memset_raw_((void*)&o, 0, sizeof(struct AudioStreamOptions)); }
    return o;
}

struct AudioStreamOptions AudioStreamOptions::with_flags(unsigned int flags) {
    struct AudioStreamOptions o = AudioStreamOptions::none();
    o.flags = flags;
    return o;
}

// ── opening a stream ─────────────────────────────────────────────────────

static int audio_format_valid_(unsigned int f) { return audio_format_bytes(f) != 0; }

static struct Result<AudioStream, AudioError> audio_stream_err_(int code, const &char msg) {
    return Result::new_err(AudioError::make(code, msg));
}

// Resolves one direction's device against the host's list.
static int audio_resolve_dir_(&AudioHost_ h, &AudioStreamState_ st, int dir, &AudioStreamParams p, &AudioError err) {
    if (p.channels < 1) return audio_fail_(err, AUDIO_ERR_INVALID_PARAM, "audio: channels must be >= 1");
    if (p.first_channel < 0) return audio_fail_(err, AUDIO_ERR_INVALID_PARAM, "audio: first_channel must be >= 0");
    st.user_channels[dir] = p.channels;
    st.first_channel[dir] = p.first_channel;
    if (st.device_name[0] != (char)0) {
        st.device_id[dir] = p.device;      // the backend opens device_name; the id is informational
        return AUDIO_OK;
    }
    unsigned int id = p.device;
    if (id == AUDIO_DEFAULT_DEVICE) id = dir == AUDIO_DIR_OUT ? h.default_out : h.default_in;
    if (id == AUDIO_DEFAULT_DEVICE) {
        // No default: the first device that has channels in this direction.
        unsafe {
            for (unsigned int i = 0U; i < h.count; i = i + 1U) {
                int ch = dir == AUDIO_DIR_OUT ? h.devs[i].info.output_channels : h.devs[i].info.input_channels;
                if (ch > 0) { id = h.devs[i].info.id; break; }
            }
        }
    }
    if (id == AUDIO_DEFAULT_DEVICE)
        return audio_fail_(err, AUDIO_ERR_NO_DEVICES,
                           dir == AUDIO_DIR_OUT ? "audio: no output device" : "audio: no input device");
    match (audio_host_find_(h, id)) {
        case some(r): {
            int have = dir == AUDIO_DIR_OUT ? r.info.output_channels : r.info.input_channels;
            if (p.channels + p.first_channel > have)
                return audio_fail2_(err, AUDIO_ERR_INVALID_PARAM, "audio: device has too few channels",
                                    &r.info.name[0]);
            st.device_id[dir] = id;
            audio_strcpy_(&st.device_key[dir][0], (unsigned long)AUDIO_DEVICE_KEY_MAX, &r.key[0]);
            st.device_native[dir] = r.native;
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(err, AUDIO_ERR_INVALID_DEVICE, "audio: no device with that id");
}

static void audio_stream_free_buffers_(&AudioStreamState_ st) {
    unsafe {
        for (int d = 0; d < 2; d = d + 1) {
            if (st.device_buf[d] != st.user_buf[d]) dealloc_raw_(st.device_buf[d]);
            dealloc_raw_(st.user_buf[d]);
            st.device_buf[d] = (void*)0;
            st.user_buf[d] = (void*)0;
        }
    }
}

// After the backend's open(): decide per direction whether the callback's
// layout differs from the device's, and allocate both buffers.
static int audio_stream_alloc_buffers_(&AudioStreamState_ st) {
    if (st.max_frames < st.buffer_frames) st.max_frames = st.buffer_frames;
    for (int d = 0; d < 2; d = d + 1) {
        int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
        if (!on) continue;
        if (st.device_channels[d] == 0) st.device_channels[d] = st.user_channels[d] + st.first_channel[d];
        if (st.device_format[d] == 0U) st.device_format[d] = st.user_format;
        unsigned long frames = (unsigned long)st.max_frames;
        unsigned long ub = frames * (unsigned long)st.user_channels[d] * (unsigned long)audio_format_bytes(st.user_format);
        unsigned long db = frames * (unsigned long)st.device_channels[d] *
                           (unsigned long)audio_format_bytes(st.device_format[d]);
        int same = st.device_format[d] == st.user_format && st.device_channels[d] == st.user_channels[d] &&
                   st.first_channel[d] == 0 &&
                   (st.device_interleaved[d] == st.user_interleaved || st.user_channels[d] == 1);
        st.convert[d] = !same;
        unsafe {
            st.user_buf[d] = alloc_zeroed_raw_(ub == 0UL ? 16UL : ub);
            if (st.user_buf[d] == (void*)0) return AUDIO_ERR_NO_MEMORY;
            if (same) st.device_buf[d] = st.user_buf[d];
            else {
                st.device_buf[d] = alloc_zeroed_raw_(db == 0UL ? 16UL : db);
                if (st.device_buf[d] == (void*)0) return AUDIO_ERR_NO_MEMORY;
            }
        }
    }
    return AUDIO_OK;
}

static ?&AudioBackendVt_ audio_stream_vt_(&AudioStreamState_ st) {
    unsafe { return (?&AudioBackendVt_)st.vt; }
}

struct Result<AudioStream, AudioError> Audio::open_stream_raw_(?&AudioStreamParams output, ?&AudioStreamParams input,
                                                               unsigned int format, unsigned int sample_rate,
                                                               &unsigned int frames, AudioCallbackRaw_ callback,
                                                               ?&Erased user, ?&AudioStreamOptions opts) {
    ?&AudioHost_ hp = audio_host_of_(self);
    if (hp.is_null()) return audio_stream_err_(AUDIO_ERR_INVALID_USE, "audio: not open");
    if (output.is_null() && input.is_null())
        return audio_stream_err_(AUDIO_ERR_INVALID_PARAM, "audio: a stream needs output and/or input parameters");
    if (!audio_format_valid_(format)) return audio_stream_err_(AUDIO_ERR_INVALID_PARAM, "audio: invalid sample format");
    if (sample_rate == 0U) return audio_stream_err_(AUDIO_ERR_INVALID_PARAM, "audio: sample rate must be > 0");
    ?&AudioBackendVt_ vtp = audio_backend_for_(self.api_id);
    if (vtp.is_null()) return audio_stream_err_(AUDIO_ERR_UNSUPPORTED, "audio: API not compiled in");

    ?&AudioStreamState_ sp = null;
    unsafe { sp = (?&AudioStreamState_)alloc_zeroed_raw_(sizeof(struct AudioStreamState_)); }
    match (sp) {
        case null: { return audio_stream_err_(AUDIO_ERR_NO_MEMORY, "audio: out of memory"); }
        case some(st): {
            match (hp) {
                case null: {}
                case some(h): {
                    match (vtp) {
                        case null: {}
                        case some(vt): {
                            st.api = self.api_id;
                            unsafe { st.vt = (void*)vt; }
                            st.user_format = format;
                            st.sample_rate = sample_rate;
                            st.buffer_frames = *frames;
                            st.callback = callback;
                            st.user = user;
                            st.state = AUDIO_STREAM_CLOSED;
                            st.user_interleaved = 1;
                            match (opts) {
                                case some(o): {
                                    st.flags = o.flags;
                                    st.buffer_count = o.buffer_count;
                                    st.priority = o.priority;
                                    audio_strcpy_(&st.stream_name[0], (unsigned long)AUDIO_STREAM_NAME_MAX, &o.stream_name[0]);
                                    audio_strcpy_(&st.device_name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, &o.device_name[0]);
                                }
                                case null: {}
                            }
                            if ((st.flags & AUDIO_FLAG_NONINTERLEAVED) != 0U) st.user_interleaved = 0;
                            if (st.stream_name[0] == (char)0)
                                audio_strcpy_(&st.stream_name[0], (unsigned long)AUDIO_STREAM_NAME_MAX, "SafeC");
                            struct AudioError err = AudioError::none();
                            int rc = AUDIO_OK;
                            match (output) {
                                case some(p): { st.mode_out = 1; rc = audio_resolve_dir_(h, st, AUDIO_DIR_OUT, p, &err); }
                                case null: {}
                            }
                            if (rc == AUDIO_OK) {
                                match (input) {
                                    case some(p): { st.mode_in = 1; rc = audio_resolve_dir_(h, st, AUDIO_DIR_IN, p, &err); }
                                    case null: {}
                                }
                            }
                            if (rc == AUDIO_OK) {
                                st.error = AudioError::none();
                                unsafe { rc = vt.open(h, st); }
                                if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) err = st.error;
                                else {
                                    rc = audio_stream_alloc_buffers_(st);
                                    if (rc != AUDIO_OK) {
                                        err = AudioError::make(rc, "audio: out of memory");
                                        unsafe { vt.close(st); }
                                    }
                                }
                            }
                            if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) {
                                audio_stream_free_buffers_(st);
                                unsafe { dealloc_raw_((void*)st); }
                                return Result::new_err(err);
                            }
                            *frames = st.buffer_frames;
                            unsafe { atomic_store((_Atomic int*)&st.state, AUDIO_STREAM_STOPPED); }
                            struct AudioStream s;
                            s.impl = st;
                            return Result::new_ok(s);
                        }
                    }
                }
            }
            unsafe { dealloc_raw_((void*)st); }
        }
    }
    return audio_stream_err_(AUDIO_ERR_INVALID_USE, "audio: not open");
}

// ── the real-time step ───────────────────────────────────────────────────

unsigned long audio_stream_device_frame_bytes_(&AudioStreamState_ st, int dir) {
    return (unsigned long)st.device_channels[dir] * (unsigned long)audio_format_bytes(st.device_format[dir]);
}

void audio_stream_flag_status_(&AudioStreamState_ st, unsigned int status) {
    unsafe { atomic_fetch_or((_Atomic unsigned int*)&st.pending_status, status); }
}

int audio_stream_stop_requested_(&AudioStreamState_ st) {
    unsafe { return atomic_load((_Atomic int*)&st.stop_request); }
}

void audio_stream_request_stop_(&AudioStreamState_ st, int how) {
    unsafe {
        // ABORT overrides STOP, never the other way round.
        int cur = atomic_load((_Atomic int*)&st.stop_request);
        if (how > cur) atomic_store((_Atomic int*)&st.stop_request, how);
    }
}

void audio_stream_mark_stopped_(&AudioStreamState_ st, ?&AudioError err) {
    match (err) {
        case some(e): { if (e.failed()) audio_stream_set_rt_error_(st, e.code, &e.message[0]); }
        case null: {}
    }
    unsafe { atomic_store((_Atomic int*)&st.state, AUDIO_STREAM_STOPPED); }
}

void audio_stream_set_rt_error_(&AudioStreamState_ st, int code, const &char message) {
    unsafe {
        int expected = 0;
        // Claim the slot first (2 = being written), then publish (1).
        if (atomic_compare_exchange_strong((_Atomic int*)&st.rt_error_set, &expected, 2)) {
            st.rt_error.code = code;
            audio_strcpy_(&st.rt_error.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, message);
            atomic_store((_Atomic int*)&st.rt_error_set, 1);
        }
    }
}

void audio_stream_silence_(&AudioStreamState_ st, unsigned int frames) {
    if (!st.mode_out) return;
    unsigned long n = (unsigned long)frames * audio_stream_device_frame_bytes_(st, AUDIO_DIR_OUT);
    unsafe {
        unsigned char* p = (unsigned char*)st.device_buf[AUDIO_DIR_OUT];
        for (unsigned long i = 0UL; i < n; i = i + 1UL) p[i] = (unsigned char)0;
    }
}

// The bytes of one period of direction 'd''s user / device buffer.
static []unsigned char audio_user_bytes_(&AudioStreamState_ st, int d, unsigned int frames) {
    unsigned long n = (unsigned long)frames * (unsigned long)st.user_channels[d] *
                      (unsigned long)audio_format_bytes(st.user_format);
    unsafe { return ((unsigned char*)st.user_buf[d])[0..n]; }
}
static []unsigned char audio_device_bytes_(&AudioStreamState_ st, int d, unsigned int frames) {
    unsigned long n = (unsigned long)frames * audio_stream_device_frame_bytes_(st, d);
    unsafe { return ((unsigned char*)st.device_buf[d])[0..n]; }
}

int audio_stream_process_(&AudioStreamState_ st, unsigned int frames) {
    unsigned int n = frames;
    if (n > st.max_frames) n = st.max_frames;
    if (st.mode_in && st.convert[AUDIO_DIR_IN]) {
        audio_copy_frames(audio_user_bytes_(st, AUDIO_DIR_IN, n), st.user_format, st.user_channels[AUDIO_DIR_IN],
                          st.user_interleaved, 0,
                          audio_device_bytes_(st, AUDIO_DIR_IN, n), st.device_format[AUDIO_DIR_IN],
                          st.device_channels[AUDIO_DIR_IN], st.device_interleaved[AUDIO_DIR_IN],
                          st.first_channel[AUDIO_DIR_IN], st.user_channels[AUDIO_DIR_IN],
                          (unsigned long)n);
    }
    struct AudioIo io;
    unsigned char none_[1];
    io.output_ = none_[0..0];
    io.input_ = none_[0..0];
    if (st.mode_out) io.output_ = audio_user_bytes_(st, AUDIO_DIR_OUT, n);
    if (st.mode_in) io.input_ = audio_user_bytes_(st, AUDIO_DIR_IN, n);
    unsafe {
        long long done = atomic_load((_Atomic long long*)&st.frames_done);
        io.stream_time = (double)(done + st.frames_offset) / (double)st.sample_rate;
        io.status = atomic_exchange((_Atomic unsigned int*)&st.pending_status, 0U);
    }
    io.frames = n;
    io.output_channels = st.mode_out ? st.user_channels[AUDIO_DIR_OUT] : 0;
    io.input_channels = st.mode_in ? st.user_channels[AUDIO_DIR_IN] : 0;
    io.format = st.user_format;
    io.interleaved = st.user_interleaved;
    io.sample_rate = st.sample_rate;
    int r = AUDIO_CB_CONTINUE;
    r = st.callback(&io, st.user);
    if (st.mode_out && st.convert[AUDIO_DIR_OUT]) {
        []unsigned char dev = audio_device_bytes_(st, AUDIO_DIR_OUT, n);
        // Device channels outside the stream's range stay silent.
        if (st.device_channels[AUDIO_DIR_OUT] != st.user_channels[AUDIO_DIR_OUT]) {
            for (unsigned long i = 0UL; i < dev.len; i = i + 1UL) dev[i] = (unsigned char)0;
        }
        audio_copy_frames(dev, st.device_format[AUDIO_DIR_OUT],
                          st.device_channels[AUDIO_DIR_OUT], st.device_interleaved[AUDIO_DIR_OUT],
                          st.first_channel[AUDIO_DIR_OUT],
                          audio_user_bytes_(st, AUDIO_DIR_OUT, n), st.user_format,
                          st.user_channels[AUDIO_DIR_OUT], st.user_interleaved, 0,
                          st.user_channels[AUDIO_DIR_OUT], (unsigned long)n);
    }
    unsafe { atomic_fetch_add((_Atomic long long*)&st.frames_done, (long long)n); }
    if (r == AUDIO_CB_STOP || r == AUDIO_CB_ABORT) audio_stream_request_stop_(st, r);
    else if (r != AUDIO_CB_CONTINUE) {
        audio_stream_request_stop_(st, AUDIO_CB_ABORT);
        r = AUDIO_CB_ABORT;
    }
    return r;
}

// ── stream threads ───────────────────────────────────────────────────────

typedef fn void(&AudioStreamState_) AudioThreadBody_;

static ?&AudioStreamState_ audio_thread_tramp_(?&AudioStreamState_ arg) {
    match (arg) {
        case some(st): {
            if ((st.flags & AUDIO_FLAG_SCHEDULE_REALTIME) != 0U) audio_thread_make_realtime_(st.priority);
            unsafe {
                AudioThreadBody_ body = (AudioThreadBody_)st.thread_body;
                body(st);
            }
        }
        case null: {}
    }
    return null;
}

void audio_stream_thread_join_(&AudioStreamState_ st) {
#if !defined(__wasm__)
    if (st.thread_live) {
        thread_join(st.thread);
        st.thread_live = 0;
    }
#endif
}

int audio_stream_thread_start_(&AudioStreamState_ st, fn void(&AudioStreamState_) body) {
#if defined(__wasm__)
    return audio_fail_(&st.error, AUDIO_ERR_THREAD, "audio: no threads on this target (use AUDIO_FLAG_DUMMY_MANUAL)");
#else
    audio_stream_thread_join_(st);
    unsafe { st.thread_body = (void*)body; }
    unsigned long long tid = 0ULL;
    int rc = 0;
    unsafe { rc = thread_create(&tid, audio_thread_tramp_, (?&AudioStreamState_)st); }
    if (rc != 0) return audio_fail_(&st.error, AUDIO_ERR_THREAD, "audio: could not start the stream thread");
    st.thread = tid;
    st.thread_live = 1;
    return AUDIO_OK;
#endif
}

// ── AudioStream ──────────────────────────────────────────────────────────

struct AudioStream AudioStream::closed() {
    struct AudioStream s;
    s.impl = null;
    return s;
}

static ?&AudioStreamState_ audio_state_of_(const &AudioStream s) {
    return s.impl;
}

static int audio_state_load_(&AudioStreamState_ st) {
    unsafe { return atomic_load((_Atomic int*)&st.state); }
}

struct AudioError AudioStream::start() {
    match (audio_state_of_(self)) {
        case some(st): {
            int s = audio_state_load_(st);
            if (s == AUDIO_STREAM_RUNNING) return AudioError::make(AUDIO_ERR_WARNING, "audio: stream already running");
            if (s != AUDIO_STREAM_STOPPED) return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: stream not open");
            audio_stream_thread_join_(st);
            unsafe {
                atomic_store((_Atomic int*)&st.stop_request, 0);
                atomic_store((_Atomic unsigned int*)&st.pending_status, 0U);
                atomic_store((_Atomic int*)&st.rt_error_set, 0);
                atomic_store((_Atomic int*)&st.state, AUDIO_STREAM_RUNNING);
            }
            st.error = AudioError::none();
            int rc = AUDIO_OK;
            match (audio_stream_vt_(st)) {
                case some(vt): { unsafe { rc = vt.start(st); } }
                case null: { rc = AUDIO_ERR_INVALID_USE; }
            }
            if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) {
                unsafe { atomic_store((_Atomic int*)&st.state, AUDIO_STREAM_STOPPED); }
                return st.error;
            }
            return AudioError::none();
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: stream not open");
}

static struct AudioError audio_stream_halt_(const &AudioStream s, int how) {
    match (audio_state_of_(s)) {
        case some(st): {
            int state = audio_state_load_(st);
            if (state == AUDIO_STREAM_CLOSED) return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: stream not open");
            audio_stream_request_stop_(st, how);
            int rc = AUDIO_OK;
            st.error = AudioError::none();
            // A stream that already stopped itself still has its thread /
            // device to settle: the backend's stop/abort handles both.
            match (audio_stream_vt_(st)) {
                case some(vt): {
                    unsafe { rc = how == AUDIO_CB_ABORT ? vt.abort(st) : vt.stop(st); }
                }
                case null: {}
            }
            audio_stream_thread_join_(st);
            unsafe { atomic_store((_Atomic int*)&st.state, AUDIO_STREAM_STOPPED); }
            if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) return st.error;
            if (state != AUDIO_STREAM_RUNNING) return AudioError::make(AUDIO_ERR_WARNING, "audio: stream already stopped");
            return AudioError::none();
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: stream not open");
}

struct AudioError AudioStream::stop() { return audio_stream_halt_(self, AUDIO_CB_STOP); }
struct AudioError AudioStream::abort() { return audio_stream_halt_(self, AUDIO_CB_ABORT); }

void AudioStream::close() { self.drop(); }

void AudioStream::drop() {
    match (audio_state_of_(self)) {
        case some(st): {
            if (audio_state_load_(st) != AUDIO_STREAM_CLOSED) {
                audio_stream_halt_(self, AUDIO_CB_ABORT);
                match (audio_stream_vt_(st)) {
                    case some(vt): { unsafe { vt.close(st); } }
                    case null: {}
                }
            }
            audio_stream_free_buffers_(st);
            unsafe { dealloc_raw_((void*)st); }
        }
        case null: {}
    }
    self.impl = null;
}

int AudioStream::is_open() const { return !self.impl.is_null(); }

int AudioStream::is_running() const {
    match (audio_state_of_(self)) {
        case some(st): return audio_state_load_(st) == AUDIO_STREAM_RUNNING;
        case null: {}
    }
    return 0;
}

int AudioStream::state() const {
    match (audio_state_of_(self)) {
        case some(st): return audio_state_load_(st);
        case null: {}
    }
    return AUDIO_STREAM_CLOSED;
}

double AudioStream::stream_time() const {
    match (audio_state_of_(self)) {
        case some(st): {
            long long done = 0LL;
            unsafe { done = atomic_load((_Atomic long long*)&st.frames_done); }
            if (st.sample_rate == 0U) return 0.0;
            return (double)(done + st.frames_offset) / (double)st.sample_rate;
        }
        case null: {}
    }
    return 0.0;
}

void AudioStream::set_stream_time(double seconds) {
    match (audio_state_of_(self)) {
        case some(st): {
            long long done = 0LL;
            unsafe { done = atomic_load((_Atomic long long*)&st.frames_done); }
            st.frames_offset = (long long)(seconds * (double)st.sample_rate) - done;
        }
        case null: {}
    }
}

long AudioStream::output_latency() const {
    match (audio_state_of_(self)) { case some(st): return st.mode_out ? st.latency[AUDIO_DIR_OUT] : 0L; case null: {} }
    return 0L;
}
long AudioStream::input_latency() const {
    match (audio_state_of_(self)) { case some(st): return st.mode_in ? st.latency[AUDIO_DIR_IN] : 0L; case null: {} }
    return 0L;
}
long AudioStream::latency() const { return self.output_latency() + self.input_latency(); }

unsigned int AudioStream::sample_rate() const {
    match (audio_state_of_(self)) { case some(st): return st.sample_rate; case null: {} }
    return 0U;
}
unsigned int AudioStream::buffer_frames() const {
    match (audio_state_of_(self)) { case some(st): return st.buffer_frames; case null: {} }
    return 0U;
}
int AudioStream::api() const {
    match (audio_state_of_(self)) { case some(st): return st.api; case null: {} }
    return AUDIO_API_UNSPECIFIED;
}

struct AudioError AudioStream::last_error() const {
    match (audio_state_of_(self)) {
        case some(st): {
            int set = 0;
            unsafe { set = atomic_load((_Atomic int*)&st.rt_error_set); }
            if (set == 1) return st.rt_error;
            return AudioError::none();
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "audio: stream not open");
}

// The dummy backend's manual step (audio_dummy.sc).
int audio_dummy_pump_(&AudioStreamState_ st);

int AudioStream::pump() {
    match (audio_state_of_(self)) {
        case some(st): {
            if (st.api != AUDIO_API_DUMMY) return -1;
            return audio_dummy_pump_(st);
        }
        case null: {}
    }
    return -1;
}

// ── AudioIo ──────────────────────────────────────────────────────────────

static unsigned long audio_io_samples_(const &AudioIo io, int out) {
    return (unsigned long)io.frames * (unsigned long)(out ? io.output_channels : io.input_channels);
}

// 'b' viewed as n samples of an E: empty unless the stream's format is
// 'format' and b holds them.
generic<E>
static []E audio_io_view_(const &AudioIo io, []unsigned char b, int out, unsigned int format) {
    unsigned long n = audio_io_samples_(io, out);
    if (io.format != format || b.len < n * sizeof(E)) n = 0UL;
    unsafe { return ((E*)b.ptr)[0..n]; }
}

[]float AudioIo::output_f32() const { []float v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_F32); return v; }
[]const float AudioIo::input_f32() const { []float v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_F32); return v; }
[]double AudioIo::output_f64() const { []double v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_F64); return v; }
[]const double AudioIo::input_f64() const { []double v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_F64); return v; }
[]short AudioIo::output_i16() const { []short v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_I16); return v; }
[]const short AudioIo::input_i16() const { []short v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_I16); return v; }
[]int AudioIo::output_i32() const { []int v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_I32); return v; }
[]const int AudioIo::input_i32() const { []int v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_I32); return v; }
[]signed char AudioIo::output_i8() const { []signed char v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_I8); return v; }
[]const signed char AudioIo::input_i8() const { []signed char v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_I8); return v; }
[]S24 AudioIo::output_i24() const { []S24 v = audio_io_view_(self, self.output_, 1, AUDIO_FORMAT_I24); return v; }
[]const S24 AudioIo::input_i24() const { []S24 v = audio_io_view_(self, self.input_, 0, AUDIO_FORMAT_I24); return v; }

[]unsigned char AudioIo::output_bytes() const { return self.output_; }
[]const unsigned char AudioIo::input_bytes() const { return self.input_; }

[]float AudioIo::output_channel_f32(int channel) const {
    []float all = self.output_f32();
    if (self.interleaved || channel < 0 || channel >= self.output_channels || all.len == 0UL) return all[0..0];
    unsigned long f = (unsigned long)self.frames;
    return all[(unsigned long)channel * f..(unsigned long)(channel + 1) * f];
}
[]const float AudioIo::input_channel_f32(int channel) const {
    []const float all = self.input_f32();
    if (self.interleaved || channel < 0 || channel >= self.input_channels || all.len == 0UL) return all[0..0];
    unsigned long f = (unsigned long)self.frames;
    return all[(unsigned long)channel * f..(unsigned long)(channel + 1) * f];
}
void AudioIo::clear_output() const {
    []unsigned char b = self.output_;
    for (unsigned long i = 0UL; i < b.len; i = i + 1UL) b[i] = (unsigned char)0;
}

} // namespace std
