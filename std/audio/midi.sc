// SafeC Standard Library — real-time MIDI I/O: the API-independent core
// (see midi.h): the API registry, message helpers, the parser/encoder,
// the lock-free input queue and the MidiIn/MidiOut front ends.
// safeguard: implements <std/audio/midi_backend.h>
#pragma once
#include <std/audio/midi.h>
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

namespace std {

// ── API names & registry ─────────────────────────────────────────────────

const &char midi_api_name(int api) {
    if (api == MIDI_API_ALSA) return "alsa";
    if (api == MIDI_API_JACK) return "jack";
    if (api == MIDI_API_COREMIDI) return "coremidi";
    if (api == MIDI_API_WINMM) return "winmm";
    if (api == MIDI_API_WEBMIDI) return "webmidi";
    if (api == MIDI_API_DUMMY) return "dummy";
    return "";
}

const &char midi_api_display_name(int api) {
    if (api == MIDI_API_ALSA) return "ALSA";
    if (api == MIDI_API_JACK) return "JACK";
    if (api == MIDI_API_COREMIDI) return "Core MIDI";
    if (api == MIDI_API_WINMM) return "Windows MultiMedia";
    if (api == MIDI_API_WEBMIDI) return "Web MIDI";
    if (api == MIDI_API_DUMMY) return "Dummy";
    return "";
}

int midi_api_by_name(const &char name) {
    for (int a = 1; a < MIDI_API_COUNT; a = a + 1)
        if (audio_streq_(name, midi_api_name(a))) return a;
    return MIDI_API_UNSPECIFIED;
}

static int midi_backends_(&?&static MidiBackendVt_ out, int cap) {
    int n = 0;
#if defined(__APPLE__)
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_coremidi_backend_(); } n = n + 1; }
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_jack_backend_(); } n = n + 1; }
#endif
#if defined(__linux__)
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_alsa_backend_(); } n = n + 1; }
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_jack_backend_(); } n = n + 1; }
#endif
#if defined(_WIN32)
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_winmm_backend_(); } n = n + 1; }
#endif
#if defined(__wasm__)
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_webmidi_backend_(); } n = n + 1; }
#endif
    if (n < cap) { unsafe { out[n] = (?&static MidiBackendVt_)midi_dummy_backend_(); } n = n + 1; }
    return n;
}

static ?&static MidiBackendVt_ midi_backend_for_(int api) {
    ?&static MidiBackendVt_ list[MIDI_API_COUNT];
    int n = midi_backends_(&list[0], MIDI_API_COUNT);
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): { if (vt.api == api) return (?&static MidiBackendVt_)vt; }
            case null: {}
        }
    }
    return null;
}

int midi_compiled_apis([]int out) {
    ?&static MidiBackendVt_ list[MIDI_API_COUNT];
    int n = midi_backends_(&list[0], MIDI_API_COUNT);
    int k = 0;
    for (int i = 0; i < n && (unsigned long)k < out.len; i = i + 1) {
        match (list[i]) {
            case some(vt): { out[k] = vt.api; k = k + 1; }
            case null: {}
        }
    }
    return k;
}

int midi_api_compiled(int api) { return !midi_backend_for_(api).is_null(); }

static int midi_available_(&MidiBackendVt_ vt) {
    int r = 0;
    unsafe { r = vt.available(); }
    return r;
}

int midi_default_api() {
    ?&char env = audio_getenv_("SAFEC_MIDI_API");
    match (env) {
        case some(e): {
            int a = midi_api_by_name(e);
            if (a != MIDI_API_UNSPECIFIED && midi_api_compiled(a)) return a;
        }
        case null: {}
    }
    ?&static MidiBackendVt_ list[MIDI_API_COUNT];
    int n = midi_backends_(&list[0], MIDI_API_COUNT);
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): { if (vt.api == MIDI_API_JACK && midi_available_(vt)) return MIDI_API_JACK; }
            case null: {}
        }
    }
    for (int i = 0; i < n; i = i + 1) {
        match (list[i]) {
            case some(vt): { if (vt.api != MIDI_API_JACK && vt.api != MIDI_API_DUMMY) return vt.api; }
            case null: {}
        }
    }
    return MIDI_API_DUMMY;
}

// ── message helpers ──────────────────────────────────────────────────────

static int midi_clamp7_(int v) {
    if (v < 0) return 0;
    if (v > 127) return 127;
    return v;
}

static struct MidiMsg midi_msg3_(int status, int a, int b) {
    struct MidiMsg m;
    m.data[0] = (unsigned char)(status & 0xFF);
    m.data[1] = (unsigned char)midi_clamp7_(a);
    m.data[2] = (unsigned char)midi_clamp7_(b);
    m.len = 3;
    return m;
}

static struct MidiMsg midi_msg2_(int status, int a) {
    struct MidiMsg m;
    m.data[0] = (unsigned char)(status & 0xFF);
    m.data[1] = (unsigned char)midi_clamp7_(a);
    m.data[2] = (unsigned char)0;
    m.len = 2;
    return m;
}

static int midi_ch_(int channel) { return channel & 0x0F; }

struct MidiMsg MidiMsg::note_on(int channel, int note, int velocity) {
    return midi_msg3_(MIDI_NOTE_ON | midi_ch_(channel), note, velocity);
}
struct MidiMsg MidiMsg::note_off(int channel, int note, int velocity) {
    return midi_msg3_(MIDI_NOTE_OFF | midi_ch_(channel), note, velocity);
}
struct MidiMsg MidiMsg::poly_pressure(int channel, int note, int pressure) {
    return midi_msg3_(MIDI_POLY_PRESSURE | midi_ch_(channel), note, pressure);
}
struct MidiMsg MidiMsg::control_change(int channel, int controller, int value) {
    return midi_msg3_(MIDI_CONTROL_CHANGE | midi_ch_(channel), controller, value);
}
struct MidiMsg MidiMsg::program_change(int channel, int program) {
    return midi_msg2_(MIDI_PROGRAM_CHANGE | midi_ch_(channel), program);
}
struct MidiMsg MidiMsg::channel_pressure(int channel, int pressure) {
    return midi_msg2_(MIDI_CHANNEL_PRESSURE | midi_ch_(channel), pressure);
}
struct MidiMsg MidiMsg::pitch_bend(int channel, int value) {
    int v = value + 8192;
    if (v < 0) v = 0;
    if (v > 16383) v = 16383;
    return midi_msg3_(MIDI_PITCH_BEND | midi_ch_(channel), v & 0x7F, (v >> 7) & 0x7F);
}
struct MidiMsg MidiMsg::song_position(int beats) {
    int v = beats;
    if (v < 0) v = 0;
    if (v > 16383) v = 16383;
    return midi_msg3_(MIDI_SONG_POSITION, v & 0x7F, (v >> 7) & 0x7F);
}
struct MidiMsg MidiMsg::song_select(int song) { return midi_msg2_(MIDI_SONG_SELECT, song); }
struct MidiMsg MidiMsg::system(int status) {
    struct MidiMsg m;
    m.data[0] = (unsigned char)(status & 0xFF);
    m.data[1] = (unsigned char)0;
    m.data[2] = (unsigned char)0;
    m.len = 1;
    return m;
}
struct MidiMsg MidiMsg::of([]const unsigned char bytes) {
    struct MidiMsg m;
    m.data[0] = (unsigned char)0; m.data[1] = (unsigned char)0; m.data[2] = (unsigned char)0;
    m.len = 0;
    unsigned long n = bytes.len;
    if (n == 0UL || n > 3UL) return m;
    for (unsigned long i = 0UL; i < n; i = i + 1UL) m.data[i] = bytes[i];
    m.len = (int)n;
    return m;
}

[]const unsigned char MidiMsg::bytes() const {
    unsigned long n = self.len < 0 ? 0UL : (unsigned long)self.len;
    if (n > 3UL) n = 3UL;
    return self.data[0..n];
}

int MidiMsg::status() const { return self.len > 0 ? (int)self.data[0] : 0; }
int MidiMsg::type() const {
    int s = self.status();
    if (s >= 0x80 && s < 0xF0) return s & 0xF0;
    return s;
}
int MidiMsg::channel() const {
    int s = self.status();
    if (s >= 0x80 && s < 0xF0) return s & 0x0F;
    return -1;
}
int MidiMsg::data1() const { return self.len > 1 ? (int)self.data[1] : -1; }
int MidiMsg::data2() const { return self.len > 2 ? (int)self.data[2] : -1; }
int MidiMsg::is_note_on() const { return self.type() == MIDI_NOTE_ON && self.len == 3 && self.data[2] > (unsigned char)0; }
int MidiMsg::is_note_off() const {
    int t = self.type();
    return self.len == 3 && (t == MIDI_NOTE_OFF || (t == MIDI_NOTE_ON && self.data[2] == (unsigned char)0));
}
int MidiMsg::pitch_bend_value() const {
    if (self.type() != MIDI_PITCH_BEND || self.len != 3) return 0;
    return midi_14bit((int)self.data[1], (int)self.data[2]) - 8192;
}

int midi_message_length(int status) {
    if (status < 0x80 || status > 0xFF) return -1;
    if (status < 0xF0) {
        int t = status & 0xF0;
        if (t == MIDI_PROGRAM_CHANGE || t == MIDI_CHANNEL_PRESSURE) return 2;
        return 3;
    }
    if (status == MIDI_SYSEX) return 0;
    if (status == MIDI_TIME_CODE || status == MIDI_SONG_SELECT) return 2;
    if (status == MIDI_SONG_POSITION) return 3;
    if (status == 0xF4 || status == 0xF5 || status == 0xF9 || status == 0xFD) return -1;  // undefined
    return 1;   // F6, F7, real-time
}

int midi_is_realtime(int status) { return status >= 0xF8 && status <= 0xFF; }

long midi_build_sysex([]unsigned char out, []const unsigned char payload) {
    unsigned long n = payload.len;
    if (out.len < n + 2UL) return -1L;
    for (unsigned long i = 0UL; i < n; i = i + 1UL) if (payload[i] >= (unsigned char)0x80) return -1L;
    out[0] = (unsigned char)MIDI_SYSEX;
    for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i + 1UL] = payload[i];
    out[n + 1UL] = (unsigned char)MIDI_SYSEX_END;
    return (long)(n + 2UL);
}

int midi_14bit(int lsb, int msb) { return (lsb & 0x7F) | ((msb & 0x7F) << 7); }

// ── parser ───────────────────────────────────────────────────────────────

struct MidiParser MidiParser::new([]unsigned char sysex_buf) {
    struct MidiParser p;
    p.running = (unsigned char)0;
    p.msg[0] = (unsigned char)0; p.msg[1] = (unsigned char)0; p.msg[2] = (unsigned char)0;
    p.have = 0;
    p.need = 0;
    p.last = MidiMsg::system(0);
    p.last.len = 0;
    p.sysex = sysex_buf;
    p.sysex_len = 0UL;
    p.in_sysex = 0;
    p.sysex_overflow = 0;
    return p;
}

void MidiParser::reset() {
    self.running = (unsigned char)0;
    self.have = 0;
    self.need = 0;
    self.sysex_len = 0UL;
    self.in_sysex = 0;
    self.sysex_overflow = 0;
}

static void midi_parser_sysex_put_(&MidiParser p, int byte) {
    if (p.sysex_len < p.sysex.len) {
        p.sysex[p.sysex_len] = (unsigned char)byte;
        p.sysex_len = p.sysex_len + 1UL;
    } else {
        p.sysex_overflow = 1;
    }
}

int MidiParser::feed(int byte) {
    int b = byte & 0xFF;
    // Real-time bytes stand alone, anywhere — even inside a sysex.
    if (b >= 0xF8) {
        if (b == 0xF9 || b == 0xFD) return MIDI_PARSE_NONE;   // undefined: ignore
        self.last = MidiMsg::system(b);
        return MIDI_PARSE_MESSAGE;
    }
    if (self.in_sysex) {
        if (b == MIDI_SYSEX_END) {
            midi_parser_sysex_put_(self, b);
            self.in_sysex = 0;
            if (self.sysex_overflow || self.sysex.len == 0UL) { self.sysex_overflow = 0; return MIDI_PARSE_ERROR; }
            return MIDI_PARSE_SYSEX;
        }
        if (b < 0x80) { midi_parser_sysex_put_(self, b); return MIDI_PARSE_NONE; }
        // Another status byte ends the sysex unterminated: drop it, then
        // parse the byte normally.
        self.in_sysex = 0;
        self.sysex_overflow = 0;
    }
    if (b >= 0x80) {
        if (b == MIDI_SYSEX) {
            self.in_sysex = 1;
            self.sysex_len = 0UL;
            self.sysex_overflow = 0;
            self.running = (unsigned char)0;
            midi_parser_sysex_put_(self, b);
            return MIDI_PARSE_NONE;
        }
        int len = midi_message_length(b);
        if (len < 0 || b == MIDI_SYSEX_END) {           // undefined status or a stray F7
            self.running = (unsigned char)0;
            self.have = 0;
            return MIDI_PARSE_ERROR;
        }
        // Channel messages set running status; system common clears it.
        self.running = b < 0xF0 ? (unsigned char)b : (unsigned char)0;
        self.msg[0] = (unsigned char)b;
        self.have = 1;
        self.need = len;
        if (len == 1) {
            self.last = MidiMsg::system(b);
            self.have = 0;
            return MIDI_PARSE_MESSAGE;
        }
        return MIDI_PARSE_NONE;
    }
    // A data byte.
    if (self.have == 0) {
        if (self.running == (unsigned char)0) return MIDI_PARSE_ERROR;
        self.msg[0] = self.running;
        self.have = 1;
        self.need = midi_message_length((int)self.running);
    }
    self.msg[self.have] = (unsigned char)b;
    self.have = self.have + 1;
    if (self.have >= self.need) {
        self.last = MidiMsg::of(self.msg[0..(unsigned long)self.need]);
        self.have = 0;
        return MIDI_PARSE_MESSAGE;
    }
    return MIDI_PARSE_NONE;
}

struct MidiMsg MidiParser::message() const { return self.last; }
unsigned long MidiParser::sysex_length() const { return self.sysex_len; }
[]const unsigned char MidiParser::sysex_message() const {
    unsigned long n = self.sysex_len < self.sysex.len ? self.sysex_len : self.sysex.len;
    return self.sysex[0..n];
}

// ── encoder ──────────────────────────────────────────────────────────────

struct MidiEncoder MidiEncoder::new(int use_running_status) {
    struct MidiEncoder e;
    e.running = (unsigned char)0;
    e.use_running_status = use_running_status;
    return e;
}

void MidiEncoder::reset() { self.running = (unsigned char)0; }

long MidiEncoder::encode([]const unsigned char msg, []unsigned char out) {
    unsigned long len = msg.len;
    if (len == 0UL) return 0L;
    int status = (int)msg[0];
    unsigned long skip = 0UL;
    if (status >= 0x80 && status < 0xF0) {
        if (self.use_running_status && (int)self.running == status) skip = 1UL;
        self.running = (unsigned char)status;
    } else if (status >= 0xF0 && status < 0xF8) {
        self.running = (unsigned char)0;     // system common cancels running status
    }
    unsigned long n = len - skip;
    if (n > out.len) return -1L;
    for (unsigned long i = 0UL; i < n; i = i + 1UL) out[i] = msg[i + skip];
    return (long)n;
}

// ── MidiEvent ────────────────────────────────────────────────────────────

int MidiEvent::status() const {
    if (self.bytes.len == 0UL) return 0;
    return (int)self.bytes[0];
}

struct MidiMsg MidiEvent::short_msg() const {
    struct MidiMsg m = MidiMsg::system(0);
    m.len = 0;
    if (self.bytes.len == 0UL || self.bytes.len > 3UL) return m;
    return MidiMsg::of(self.bytes);
}

// ── the input path (real-time) ───────────────────────────────────────────

double midi_in_now_(&MidiPortState_ st) {
    return (double)(audio_now_ns_() - st.t0_ns) / 1000000000.0;
}

void midi_in_reset_clock_(&MidiPortState_ st) {
    st.t0_ns = audio_now_ns_();
    st.have_last = 0;
    st.last_time = 0.0;
    st.parser.reset();
}

static int midi_filtered_(int ignore, int status) {
    if ((ignore & 1) != 0 && (status == MIDI_SYSEX)) return 1;
    if ((ignore & 2) != 0 && (status == MIDI_CLOCK || status == MIDI_TIME_CODE)) return 1;
    if ((ignore & 4) != 0 && status == MIDI_ACTIVE_SENSING) return 1;
    return 0;
}

static void midi_ring_put_(&MidiPortState_ st, unsigned long at, const &unsigned char src, unsigned long n) {
    unsafe {
        unsigned long mask = st.ring_cap - 1UL;
        for (unsigned long i = 0UL; i < n; i = i + 1UL) st.ring[(at + i) & mask] = src[i];
    }
}

static void midi_ring_get_(&MidiPortState_ st, unsigned long at, &unsigned char dst, unsigned long n) {
    unsafe {
        unsigned long mask = st.ring_cap - 1UL;
        for (unsigned long i = 0UL; i < n; i = i + 1UL) dst[i] = st.ring[(at + i) & mask];
    }
}

static void midi_count_drop_(&MidiPortState_ st) {
    unsafe { atomic_fetch_add((_Atomic unsigned long long*)&st.dropped_count, 1ULL); }
}

void midi_in_deliver_(&MidiPortState_ st, const &unsigned char msg, unsigned long len, double time) {
    if (len == 0UL) return;
    int status = 0;
    int ignore = 0;
    unsafe {
        status = (int)msg[0];
        ignore = atomic_load((_Atomic int*)&st.ignore);
    }
    if (midi_filtered_(ignore, status)) return;
    double t = time < 0.0 ? midi_in_now_(st) : time;
    double delta = st.have_last ? t - st.last_time : 0.0;
    if (delta < 0.0) delta = 0.0;
    st.last_time = t;
    st.have_last = 1;
    MidiCallbackSlot_ cb = null;
    unsafe { cb = atomic_load((MidiCallbackSlot_*)&st.callback); }
    match (cb) {
        case some(f): {
            struct MidiEvent ev;
            ev.timestamp = t;
            ev.delta = delta;
            ?&Erased user = null;
            unsafe {
                ev.bytes = ((const unsigned char*)msg)[0..len];
                user = atomic_load((MidiUserSlot_*)&st.user);
            }
            f(&ev, user);
            return;
        }
        case null: {}
    }
    // The queue: [u32 len][f64 time][len bytes].
    if (st.ring_cap == 0UL) { midi_count_drop_(st); return; }
    unsigned long need = 12UL + len;
    unsigned long head = st.ring_head;
    unsigned long tail = 0UL;
    unsafe { tail = atomic_load((_Atomic unsigned long*)&st.ring_tail); }
    if (need > st.ring_cap - (head - tail)) { midi_count_drop_(st); return; }
    unsigned int l32 = (unsigned int)len;
    unsafe {
        midi_ring_put_(st, head, (const &unsigned char)&l32, 4UL);
        midi_ring_put_(st, head + 4UL, (const &unsigned char)&t, 8UL);
        midi_ring_put_(st, head + 12UL, msg, len);
        atomic_store((_Atomic unsigned long*)&st.ring_head, head + need);
    }
}

void midi_in_feed_(&MidiPortState_ st, const &unsigned char bytes, unsigned long n, double time) {
    for (unsigned long i = 0UL; i < n; i = i + 1UL) {
        int b = 0;
        unsafe { b = (int)bytes[i]; }
        int r = st.parser.feed(b);
        if (r == MIDI_PARSE_MESSAGE) {
            struct MidiMsg m = st.parser.message();
            midi_in_deliver_(st, &m.data[0], (unsigned long)m.len, time);
        } else if (r == MIDI_PARSE_SYSEX) {
            unsafe { midi_in_deliver_(st, (const &unsigned char)st.sysex_buf, st.parser.sysex_len, time); }
        } else if (r == MIDI_PARSE_ERROR && b == MIDI_SYSEX_END) {
            midi_count_drop_(st);    // a sysex longer than MIDI_MAX_SYSEX
        }
    }
}

// ── MidiIn / MidiOut ─────────────────────────────────────────────────────

static ?&MidiPortState_ midi_state_of_(?&MidiPortState_ impl) {
    return impl;
}

static ?&static MidiBackendVt_ midi_vt_of_(&MidiPortState_ st) {
    return st.vt;
}

static void midi_state_free_(&MidiPortState_ st) {
    unsafe {
        dealloc(st.ring);
        dealloc(st.sysex_buf);
        dealloc((?&heap MidiPortState_)st);
    }
}

// Opens a client of one API; the state, or null with 'err' set.
static ?&MidiPortState_ midi_client_open_(int api, int input, const &char name, unsigned long queue, &AudioError err) {
    ?&static MidiBackendVt_ vtp = midi_backend_for_(api);
    match (vtp) {
        case null: {
            *err = AudioError::make2(AUDIO_ERR_UNSUPPORTED, "midi: API not compiled in", midi_api_name(api));
            return null;
        }
        case some(vt): {
            ?&MidiPortState_ sp = null;
            unsafe { sp = (?&MidiPortState_)alloc_zeroed_raw_(sizeof(struct MidiPortState_)); }
            match (sp) {
                case null: {}
                case some(st): {
                    st.api = api;
                    st.vt = vt;
                    st.is_input = input;
                    audio_strcpy_(&st.client_name[0], (unsigned long)MIDI_CLIENT_NAME_MAX, name);
                    if (st.client_name[0] == (char)0)
                        audio_strcpy_(&st.client_name[0], (unsigned long)MIDI_CLIENT_NAME_MAX, "SafeC");
                    st.ignore = 7;
                    int ok = 1;
                    if (input) {
                        unsigned long cap = 256UL;
                        unsigned long want = queue == 0UL ? MIDI_DEFAULT_QUEUE_BYTES : queue;
                        while (cap < want && cap < (1UL << 30)) cap = cap * 2UL;
                        unsafe {
                            st.ring = (unsigned char*)alloc_zeroed_raw_(cap);
                            st.sysex_buf = (unsigned char*)alloc_zeroed_raw_(MIDI_MAX_SYSEX);
                            if (st.ring == (unsigned char*)0 || st.sysex_buf == (unsigned char*)0) ok = 0;
                            unsafe {
                                st.parser = MidiParser::new(st.sysex_buf[0..(st.sysex_buf == (unsigned char*)0 ? 0UL : MIDI_MAX_SYSEX)]);
                            }
                        }
                        st.ring_cap = cap;
                    }
                    if (!ok) {
                        midi_state_free_(st);
                        *err = AudioError::make(AUDIO_ERR_NO_MEMORY, "midi: out of memory");
                        return null;
                    }
                    st.error = AudioError::none();
                    st.t0_ns = audio_now_ns_();
                    int rc = 0;
                    unsafe { rc = vt.client_open(st); }
                    if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) {
                        *err = st.error;
                        midi_state_free_(st);
                        return null;
                    }
                    return (?&MidiPortState_)st;
                }
            }
            *err = AudioError::make(AUDIO_ERR_NO_MEMORY, "midi: out of memory");
            return null;
        }
    }
    return null;
}

// UNSPECIFIED: the default API, then the other available ones, then dummy.
static ?&MidiPortState_ midi_client_open_any_(int api, int input, const &char name, unsigned long queue,
                                              &int chosen, &AudioError err) {
    if (api != MIDI_API_UNSPECIFIED) {
        *chosen = api;
        return midi_client_open_(api, input, name, queue, err);
    }
    int first = midi_default_api();
    *chosen = first;
    ?&MidiPortState_ st = midi_client_open_(first, input, name, queue, err);
    if (!st.is_null()) return st;
    struct AudioError firstErr = *err;
    ?&static MidiBackendVt_ list[MIDI_API_COUNT];
    int n = midi_backends_(&list[0], MIDI_API_COUNT);
    for (int i = 0; i < n; i = i + 1) {
        int a = 0;
        match (list[i]) { case some(vt): { a = vt.api; } case null: {} }
        if (a == first || a == MIDI_API_UNSPECIFIED) continue;
        *chosen = a;
        st = midi_client_open_(a, input, name, queue, err);
        if (!st.is_null()) return st;
    }
    *err = firstErr;
    return null;
}

struct Result<MidiIn, AudioError> MidiIn::open(int api, const &char client_name, unsigned long queue_bytes) {
    struct AudioError err = AudioError::none();
    int chosen = api;
    ?&MidiPortState_ st = midi_client_open_any_(api, 1, client_name, queue_bytes, &chosen, &err);
    if (st.is_null()) return Result::new_err(err);
    struct MidiIn m;
    m.api_id = chosen;
    m.impl = st;
    return Result::new_ok(m);
}

struct Result<MidiOut, AudioError> MidiOut::open(int api, const &char client_name) {
    struct AudioError err = AudioError::none();
    int chosen = api;
    ?&MidiPortState_ st = midi_client_open_any_(api, 0, client_name, 0UL, &chosen, &err);
    if (st.is_null()) return Result::new_err(err);
    struct MidiOut m;
    m.api_id = chosen;
    m.impl = st;
    return Result::new_ok(m);
}

struct MidiIn MidiIn::closed() {
    struct MidiIn m;
    m.api_id = MIDI_API_UNSPECIFIED;
    m.impl = null;
    return m;
}

struct MidiOut MidiOut::closed() {
    struct MidiOut m;
    m.api_id = MIDI_API_UNSPECIFIED;
    m.impl = null;
    return m;
}

static unsigned int midi_port_count_(?&MidiPortState_ impl) {
    match (midi_state_of_(impl)) {
        case some(st): {
            match (midi_vt_of_(st)) {
                case some(vt): {
                    unsigned int n = 0U;
                    unsafe { n = vt.port_count(st); }
                    return n;
                }
                case null: {}
            }
        }
        case null: {}
    }
    return 0U;
}

static struct Result<MidiPortInfo, AudioError> midi_port_info_(?&MidiPortState_ impl, unsigned int index) {
    match (midi_state_of_(impl)) {
        case some(st): {
            match (midi_vt_of_(st)) {
                case some(vt): {
                    struct MidiPortInfo info;
                    info.index = index;
                    info.name[0] = (char)0;
                    int rc = 0;
                    unsafe { rc = vt.port_name(st, index, &info.name[0], (unsigned long)MIDI_PORT_NAME_MAX); }
                    if (rc != AUDIO_OK) return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_DEVICE, "midi: no port with that index"));
                    return Result::new_ok(info);
                }
                case null: {}
            }
        }
        case null: {}
    }
    return Result::new_err(AudioError::make(AUDIO_ERR_INVALID_USE, "midi: not open"));
}

static struct AudioError midi_open_port_(?&MidiPortState_ impl, int isVirtual, unsigned int index, const &char name) {
    match (midi_state_of_(impl)) {
        case some(st): {
            if (st.port_open) return AudioError::make(AUDIO_ERR_INVALID_USE, "midi: a port is already open (close_port() first)");
            match (midi_vt_of_(st)) {
                case some(vt): {
                    char pname[MIDI_PORT_NAME_MAX];
                    audio_strcpy_(&pname[0], (unsigned long)MIDI_PORT_NAME_MAX, name);
                    if (pname[0] == (char)0)
                        audio_strcpy_(&pname[0], (unsigned long)MIDI_PORT_NAME_MAX, st.is_input ? "input" : "output");
                    if (st.is_input) midi_in_reset_clock_(st);
                    st.error = AudioError::none();
                    int rc = 0;
                    unsafe { rc = isVirtual ? vt.open_virtual(st, &pname[0]) : vt.open_port(st, index, &pname[0]); }
                    if (rc != AUDIO_OK && rc != AUDIO_ERR_WARNING) return st.error;
                    st.port_open = 1;
                    return AudioError::none();
                }
                case null: {}
            }
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "midi: not open");
}

static void midi_close_port_(?&MidiPortState_ impl) {
    match (midi_state_of_(impl)) {
        case some(st): {
            if (!st.port_open) return;
            match (midi_vt_of_(st)) {
                case some(vt): { unsafe { vt.close_port(st); } }
                case null: {}
            }
            st.port_open = 0;
        }
        case null: {}
    }
}

static void midi_close_(?&MidiPortState_ impl) {
    midi_close_port_(impl);
    match (midi_state_of_(impl)) {
        case some(st): {
            match (midi_vt_of_(st)) {
                case some(vt): { unsafe { vt.client_close(st); } }
                case null: {}
            }
            midi_state_free_(st);
        }
        case null: {}
    }
}

int MidiIn::is_open() const { return !self.impl.is_null(); }
int MidiIn::api() const { return self.api_id; }
unsigned int MidiIn::port_count() { return midi_port_count_(self.impl); }
struct Result<MidiPortInfo, AudioError> MidiIn::port_info(unsigned int index) { return midi_port_info_(self.impl, index); }
struct AudioError MidiIn::open_port(unsigned int index, const &char port_name) {
    return midi_open_port_(self.impl, 0, index, port_name);
}
struct AudioError MidiIn::open_virtual_port(const &char port_name) {
    return midi_open_port_(self.impl, 1, 0U, port_name);
}
void MidiIn::close_port() { midi_close_port_(self.impl); }
int MidiIn::is_port_open() const {
    match (midi_state_of_(self.impl)) { case some(st): return st.port_open; case null: {} }
    return 0;
}

void MidiIn::set_callback_raw_(MidiCallbackRaw_ callback, ?&Erased user) {
    match (midi_state_of_(self.impl)) {
        case some(st): {
            unsafe {
                atomic_store((MidiCallbackSlot_*)&st.callback, (MidiCallbackSlot_)null);
                atomic_store((MidiUserSlot_*)&st.user, user);
                atomic_store((MidiCallbackSlot_*)&st.callback, (MidiCallbackSlot_)callback);
            }
        }
        case null: {}
    }
}

void MidiIn::cancel_callback() {
    match (midi_state_of_(self.impl)) {
        case some(st): { unsafe { atomic_store((MidiCallbackSlot_*)&st.callback, (MidiCallbackSlot_)null); } }
        case null: {}
    }
}

void MidiIn::ignore_types(int sysex, int timing, int active_sensing) {
    int v = (sysex ? 1 : 0) | (timing ? 2 : 0) | (active_sensing ? 4 : 0);
    match (midi_state_of_(self.impl)) {
        case some(st): { unsafe { atomic_store((_Atomic int*)&st.ignore, v); } }
        case null: {}
    }
}

long MidiIn::get_message([]unsigned char buf, &double timestamp) {
    match (midi_state_of_(self.impl)) {
        case some(st): {
            if (st.ring_cap == 0UL) return 0L;
            unsigned long tail = st.ring_tail;
            unsigned long head = 0UL;
            unsafe { head = atomic_load((_Atomic unsigned long*)&st.ring_head); }
            if (head == tail) return 0L;
            unsigned int l32 = 0U;
            double t = 0.0;
            unsafe {
                midi_ring_get_(st, tail, (&unsigned char)&l32, 4UL);
                midi_ring_get_(st, tail + 4UL, (&unsigned char)&t, 8UL);
            }
            unsigned long len = (unsigned long)l32;
            if (len > buf.len) return 0L - (long)len;
            if (len > 0UL) unsafe { midi_ring_get_(st, tail + 12UL, (&unsigned char)&buf[0], len); }
            *timestamp = t;
            unsafe { atomic_store((_Atomic unsigned long*)&st.ring_tail, tail + 12UL + len); }
            return (long)len;
        }
        case null: {}
    }
    return 0L;
}

unsigned long MidiIn::dropped() const {
    match (midi_state_of_(self.impl)) {
        case some(st): {
            unsigned long long d = 0ULL;
            unsafe { d = atomic_load((_Atomic unsigned long long*)&st.dropped_count); }
            return (unsigned long)d;
        }
        case null: {}
    }
    return 0UL;
}

void MidiIn::close() { self.drop(); }

void MidiIn::drop() {
    midi_close_(self.impl);
    self.impl = null;
    self.api_id = MIDI_API_UNSPECIFIED;
}

int MidiOut::is_open() const { return !self.impl.is_null(); }
int MidiOut::api() const { return self.api_id; }
unsigned int MidiOut::port_count() { return midi_port_count_(self.impl); }
struct Result<MidiPortInfo, AudioError> MidiOut::port_info(unsigned int index) { return midi_port_info_(self.impl, index); }
struct AudioError MidiOut::open_port(unsigned int index, const &char port_name) {
    return midi_open_port_(self.impl, 0, index, port_name);
}
struct AudioError MidiOut::open_virtual_port(const &char port_name) {
    return midi_open_port_(self.impl, 1, 0U, port_name);
}
void MidiOut::close_port() { midi_close_port_(self.impl); }
int MidiOut::is_port_open() const {
    match (midi_state_of_(self.impl)) { case some(st): return st.port_open; case null: {} }
    return 0;
}

struct AudioError MidiOut::send_message([]const unsigned char msg) {
    unsigned long len = msg.len;
    match (midi_state_of_(self.impl)) {
        case some(st): {
            if (!st.port_open) return AudioError::make(AUDIO_ERR_INVALID_USE, "midi: no port open");
            if (len == 0UL) return AudioError::make(AUDIO_ERR_INVALID_PARAM, "midi: empty message");
            int first = (int)msg[0];
            if (first < 0x80) return AudioError::make(AUDIO_ERR_INVALID_PARAM, "midi: a message starts with a status byte");
            match (midi_vt_of_(st)) {
                case some(vt): {
                    st.error = AudioError::none();
                    int rc = 0;
                    unsafe { rc = vt.send(st, (const &unsigned char)&msg[0], len); }
                    if (rc != AUDIO_OK) return st.error;
                    return AudioError::none();
                }
                case null: {}
            }
        }
        case null: {}
    }
    return AudioError::make(AUDIO_ERR_INVALID_USE, "midi: not open");
}

struct AudioError MidiOut::send(struct MidiMsg msg) {
    if (msg.len <= 0 || msg.len > 3) return AudioError::make(AUDIO_ERR_INVALID_PARAM, "midi: empty message");
    return self.send_message(msg.bytes());
}

void MidiOut::close() { self.drop(); }

void MidiOut::drop() {
    midi_close_(self.impl);
    self.impl = null;
    self.api_id = MIDI_API_UNSPECIFIED;
}

} // namespace std
