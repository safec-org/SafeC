// SafeC Standard Library — MIDI backend: Web MIDI (wasm32). See midi.h.
//
// Every operation is an import from std/audio/audio_web.js. The page asks
// for MIDI access first — await SafeCAudio.enableMIDI({ sysex }) (the
// browser's navigator.requestMIDIAccess, which may show a permission
// prompt) — then the module sees the MIDIAccess's inputs and outputs as
// ports, in the browser's order, named "<name>" (+ " (<manufacturer>)").
// Input messages arrive in the page's event loop: the glue copies each one
// into this module's per-port scratch buffer and calls
// std_midi_web_receive(), which runs the callback or queues the message.
// Web MIDI has no virtual ports (AUDIO_ERR_UNSUPPORTED).
#pragma once
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define MIDI_WEB_MAX_PORTS 16

namespace std {

// ── imports (std/audio/audio_web.js) ─────────────────────────────────────
extern int  std_midi_web_available();
extern int  std_midi_web_port_count(int input);
// Writes the port's UTF-8 name (NUL-terminated) into buf; its length, or -1.
extern int  std_midi_web_port_name(int input, int index, char* buf, int cap);
extern int  std_midi_web_open(int handle, int input, int index);
extern int  std_midi_web_send(int handle, const unsigned char* data, int len, double timestamp);
extern void std_midi_web_close(int handle);

struct MidiWebPort_ {
    int handle;
    int open;
    unsigned char* scratch;     // MIDI_MAX_SYSEX bytes the glue writes a message into
};

static struct MidiPortState_* g_midi_web_ports_[MIDI_WEB_MAX_PORTS];
static struct MidiWebPort_* g_midi_web_priv_[MIDI_WEB_MAX_PORTS];

// ── exports (called by the glue) ─────────────────────────────────────────

// Where the glue writes an incoming message of 'len' bytes for 'handle'.
__attribute__((export_name("std_midi_web_scratch")))
unsigned char* midi_web_scratch(int handle, int len) {
    if (handle < 0 || handle >= MIDI_WEB_MAX_PORTS || len <= 0 || (unsigned long)len > MIDI_MAX_SYSEX)
        return (unsigned char*)0;
    unsafe {
        struct MidiWebPort_* p = g_midi_web_priv_[handle];
        if (p == (struct MidiWebPort_*)0) return (unsigned char*)0;
        return p->scratch;
    }
}

// A complete message (in the scratch buffer) and its DOMHighResTimeStamp.
__attribute__((export_name("std_midi_web_receive")))
void midi_web_receive(int handle, int len, double timestamp_ms) {
    if (handle < 0 || handle >= MIDI_WEB_MAX_PORTS || len <= 0) return;
    unsafe {
        struct MidiPortState_* st = g_midi_web_ports_[handle];
        struct MidiWebPort_* p = g_midi_web_priv_[handle];
        if (st == (struct MidiPortState_*)0 || p == (struct MidiWebPort_*)0) return;
        double t = (double)(audio_now_ns_() - st->t0_ns) / 1000000000.0;
        midi_in_deliver_((&MidiPortState_)st, (const &unsigned char)p->scratch, (unsigned long)len, t);
    }
}

// ── the backend ──────────────────────────────────────────────────────────

static int midi_webmidi_available_() {
    unsafe { return std_midi_web_available(); }
}

static ?&MidiWebPort_ midi_web_of_(&MidiPortState_ st) {
    unsafe { return (?&MidiWebPort_)st.backend; }
}

static int midi_webmidi_client_open_(&MidiPortState_ st) {
    if (!midi_webmidi_available_())
        return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "webmidi: no MIDI access (call SafeCAudio.enableMIDI() first)");
    int handle = -1;
    unsafe {
        for (int i = 0; i < MIDI_WEB_MAX_PORTS && handle < 0; i = i + 1)
            if (g_midi_web_priv_[i] == (struct MidiWebPort_*)0) handle = i;
    }
    if (handle < 0) return audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "webmidi: too many clients");
    ?&MidiWebPort_ pp = null;
    unsafe { pp = (?&MidiWebPort_)alloc_zeroed_raw_(sizeof(struct MidiWebPort_)); }
    match (pp) {
        case null: {}
        case some(p): {
            p.handle = handle;
            int ok = 1;
            unsafe {
                if (st.is_input) {
                    p.scratch = (unsigned char*)alloc_zeroed_raw_(MIDI_MAX_SYSEX);
                    if (p.scratch == (unsigned char*)0) ok = 0;
                }
                if (ok) {
                    st.backend = (?&heap MidiPortImpl_)p;
                    g_midi_web_priv_[handle] = (struct MidiWebPort_*)p;
                    g_midi_web_ports_[handle] = (struct MidiPortState_*)st;
                } else {
                    dealloc((?&heap MidiWebPort_)p);
                }
            }
            if (ok) return AUDIO_OK;
        }
    }
    return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "webmidi: out of memory");
}

static unsigned int midi_webmidi_port_count_(&MidiPortState_ st) {
    int n = 0;
    unsafe { n = std_midi_web_port_count(st.is_input); }
    return n < 0 ? 0U : (unsigned int)n;
}

static int midi_webmidi_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    int n = 0;
    unsafe { n = std_midi_web_port_name(st.is_input, (int)index, (char*)out, (int)cap); }
    return n < 0 ? AUDIO_ERR_INVALID_DEVICE : AUDIO_OK;
}

static int midi_webmidi_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (midi_web_of_(st)) {
        case some(p): {
            int rc = 0;
            unsafe { rc = std_midi_web_open(p.handle, st.is_input, (int)index); }
            if (rc != 0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "webmidi: no MIDI port with that index");
            p.open = 1;
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_webmidi_open_virtual_(&MidiPortState_ st, const &char name) {
    return audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "webmidi: no virtual MIDI ports in Web MIDI");
}

static void midi_webmidi_close_port_(&MidiPortState_ st) {
    match (midi_web_of_(st)) {
        case some(p): {
            if (p.open) { unsafe { std_midi_web_close(p.handle); } }
            p.open = 0;
        }
        case null: {}
    }
}

static int midi_webmidi_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (midi_web_of_(st)) {
        case some(p): {
            int rc = 0;
            unsafe { rc = std_midi_web_send(p.handle, (const unsigned char*)msg, (int)len, 0.0); }
            if (rc != 0) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "webmidi: send failed (sysex needs enableMIDI({ sysex: true }))");
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_webmidi_client_close_(&MidiPortState_ st) {
    match (midi_web_of_(st)) {
        case some(p): {
            midi_webmidi_close_port_(st);
            unsafe {
                g_midi_web_ports_[p.handle] = (struct MidiPortState_*)0;
                g_midi_web_priv_[p.handle] = (struct MidiWebPort_*)0;
                dealloc(p.scratch);
                dealloc((?&heap MidiWebPort_)p);
                st.backend = null;
            }
        }
        case null: {}
    }
}

static struct MidiBackendVt_ g_midi_webmidi_vt_;
static int g_midi_webmidi_vt_ready_ = 0;

&static MidiBackendVt_ midi_webmidi_backend_() {
    if (!g_midi_webmidi_vt_ready_) {
        g_midi_webmidi_vt_.api = MIDI_API_WEBMIDI;
        g_midi_webmidi_vt_.available = midi_webmidi_available_;
        g_midi_webmidi_vt_.client_open = midi_webmidi_client_open_;
        g_midi_webmidi_vt_.port_count = midi_webmidi_port_count_;
        g_midi_webmidi_vt_.port_name = midi_webmidi_port_name_;
        g_midi_webmidi_vt_.open_port = midi_webmidi_open_port_;
        g_midi_webmidi_vt_.open_virtual = midi_webmidi_open_virtual_;
        g_midi_webmidi_vt_.close_port = midi_webmidi_close_port_;
        g_midi_webmidi_vt_.send = midi_webmidi_send_;
        g_midi_webmidi_vt_.client_close = midi_webmidi_client_close_;
        g_midi_webmidi_vt_ready_ = 1;
    }
    return &g_midi_webmidi_vt_;
}

} // namespace std
