// SafeC Standard Library — MIDI backend: JACK MIDI (Linux, macOS). See midi.h.
//
// libjack is loaded at run time (jack_lib.h) and the server is never
// auto-started: without a running server the API reports
// AUDIO_ERR_UNAVAILABLE. Each MidiIn/MidiOut is its own JACK client (named
// after the client name given to open()) with one MIDI port; ports are the
// other clients' MIDI ports, by full JACK name ("a2j:Midi Through [14]
// (capture): ...", "synth:midi_in"). Input is read in JACK's process
// thread — the callback runs there — and output is handed to that thread
// through a lock-free queue, written at the start of the next cycle.
#pragma once
#include <std/audio/midi_backend.h>
#include <std/audio/jack_lib.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define MIDI_JACK_OUT_QUEUE 16384UL

namespace std {

struct JackMidi_ {
    JackClient_* client;
    JackPort_* port;
    int active;
    char own_prefix[MIDI_CLIENT_NAME_MAX + 2];
    // output: [u32 len][bytes] records from send() to the process thread
    unsigned char* q;
    unsigned long q_head;     // (atomic) producer: send()
    unsigned long q_tail;     // (atomic) consumer: process
    struct MidiPortState_* st;
};

static int midi_jack_available_() {
    if (!audio_jack_load_()) return 0;
    JackClient_* c = audio_jack_client_open_("SafeC probe");
    if (c == (JackClient_*)0) return 0;
    unsafe { g_std_audio_jack_.client_close(c); }
    return 1;
}

static ?&JackMidi_ midi_jack_of_(&MidiPortState_ st) {
    unsafe { return (?&JackMidi_)st.backend; }
}

static void midi_jack_q_get_(&JackMidi_ j, unsigned long at, &unsigned char dst, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) dst[i] = j.q[(at + i) & (MIDI_JACK_OUT_QUEUE - 1UL)];
    }
}

static int midi_jack_process_(unsigned int nframes, ?&Erased arg) {
    unsafe {
        struct JackMidi_* j = (struct JackMidi_*)arg;
        struct MidiPortState_* st = j->st;
        if (j->port == (JackPort_*)0) return 0;
        JackMidiBuf_* buf = (JackMidiBuf_*)g_std_audio_jack_.port_get_buffer(j->port, nframes);
        if (st->is_input) {
            unsigned int n = g_std_audio_jack_.midi_get_event_count(buf);
            for (unsigned int i = 0U; i < n; i = i + 1U) {
                struct JackMidiEvent_ ev;
                if (g_std_audio_jack_.midi_event_get(&ev, buf, i) == 0 && ev.size > (size_t)0)
                    midi_in_feed_((&MidiPortState_)st, (const &unsigned char)ev.buffer, (unsigned long)ev.size, -1.0);
            }
            return 0;
        }
        g_std_audio_jack_.midi_clear_buffer(buf);
        unsigned long tail = j->q_tail;
        unsigned long head = atomic_load((_Atomic unsigned long*)&j->q_head);
        unsigned char tmp[1024];
        while (tail != head) {
            unsigned int len = 0U;
            midi_jack_q_get_((&JackMidi_)j, tail, (&unsigned char)&len, 4UL);
            unsigned long l = (unsigned long)len;
            if (l <= 1024UL) {
                midi_jack_q_get_((&JackMidi_)j, tail + 4UL, &tmp[0], l);
                g_std_audio_jack_.midi_event_write(buf, 0U, (const unsigned char*)&tmp[0], (size_t)l);
            }
            tail = tail + 4UL + l;
        }
        atomic_store((_Atomic unsigned long*)&j->q_tail, tail);
    }
    return 0;
}

static int midi_jack_client_open_(&MidiPortState_ st) {
    if (!audio_jack_load_()) return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "jack: libjack not found");
    ?&JackMidi_ jp = null;
    unsafe { jp = (?&JackMidi_)alloc_zeroed_raw_(sizeof(struct JackMidi_)); }
    match (jp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "jack: out of memory"); }
        case some(j): {
            j.client = audio_jack_client_open_(&st.client_name[0]);
            if (j.client == (JackClient_*)0) {
                unsafe { dealloc((?&heap JackMidi_)j); }
                return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "jack: server not running");
            }
            unsafe {
                j.st = (struct MidiPortState_*)st;
                if (!st.is_input) {
                    j.q = (unsigned char*)alloc_zeroed_raw_(MIDI_JACK_OUT_QUEUE);
                    if (j.q == (unsigned char*)0) {
                        g_std_audio_jack_.client_close(j.client);
                        dealloc((?&heap JackMidi_)j);
                        return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "jack: out of memory");
                    }
                }
                g_std_audio_jack_.set_process_callback(j.client, midi_jack_process_, (?&Erased)j);
                // Our own ports start with "<actual client name>:" (JACK
                // renames a client whose name is taken).
                audio_strcpy_(&j.own_prefix[0], (unsigned long)(MIDI_CLIENT_NAME_MAX + 2),
                              (const ?&char)g_std_audio_jack_.get_client_name(j.client));
                audio_strcat_(&j.own_prefix[0], (unsigned long)(MIDI_CLIENT_NAME_MAX + 2), ":");
                st.backend = (?&heap MidiPortImpl_)j;
            }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

// Other clients' MIDI ports in our direction (null-terminated, jack_free it).
static const char** midi_jack_ports_(&MidiPortState_ st, &JackMidi_ j) {
    unsafe {
        return g_std_audio_jack_.get_ports(j.client, (const char*)0, (const char*)AUDIO_JACK_MIDI_TYPE,
                                           st.is_input ? AUDIO_JACK_IS_OUTPUT : AUDIO_JACK_IS_INPUT);
    }
}

// The index-th port that isn't ours; null if none.
static const char* midi_jack_port_at_(&JackMidi_ j, const char** list, unsigned int index, &unsigned int count) {
    unsigned int k = 0U;
    unsafe {
        if (list == (const char**)0) { *count = 0U; return (const char*)0; }
        const char* found = (const char*)0;
        for (int i = 0; list[i] != (const char*)0; i = i + 1) {
            if (audio_strprefix_((const &char)list[i], &j.own_prefix[0])) continue;
            if (k == index && found == (const char*)0) found = list[i];
            k = k + 1U;
        }
        *count = k;
        return found;
    }
}

static unsigned int midi_jack_port_count_(&MidiPortState_ st) {
    match (midi_jack_of_(st)) {
        case some(j): {
            const char** list = midi_jack_ports_(st, j);
            unsigned int n = 0U;
            midi_jack_port_at_(j, list, 0xFFFFFFFFU, &n);
            unsafe { if (list != (const char**)0) g_std_audio_jack_.free(list); }
            return n;
        }
        case null: {}
    }
    return 0U;
}

static int midi_jack_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    match (midi_jack_of_(st)) {
        case some(j): {
            const char** list = midi_jack_ports_(st, j);
            unsigned int n = 0U;
            const char* p = midi_jack_port_at_(j, list, index, &n);
            int rc = AUDIO_ERR_INVALID_DEVICE;
            unsafe {
                if (p != (const char*)0) { audio_strcpy_(out, cap, (const ?&char)p); rc = AUDIO_OK; }
                if (list != (const char**)0) g_std_audio_jack_.free(list);
            }
            return rc;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_jack_make_port_(&MidiPortState_ st, &JackMidi_ j, const &char name) {
    unsafe {
        j.port = g_std_audio_jack_.port_register(j.client, name, (const char*)AUDIO_JACK_MIDI_TYPE,
                                                 st.is_input ? AUDIO_JACK_IS_INPUT : AUDIO_JACK_IS_OUTPUT, 0UL);
        if (j.port == (JackPort_*)0) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "jack: could not register a MIDI port");
        if (!j.active) {
            if (g_std_audio_jack_.activate(j.client) != 0) {
                g_std_audio_jack_.port_unregister(j.client, j.port);
                j.port = (JackPort_*)0;
                return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "jack: could not activate the client");
            }
            j.active = 1;
        }
    }
    return AUDIO_OK;
}

static int midi_jack_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (midi_jack_of_(st)) {
        case some(j): {
            char target[MIDI_PORT_NAME_MAX];
            if (midi_jack_port_name_(st, index, &target[0], (unsigned long)MIDI_PORT_NAME_MAX) != AUDIO_OK)
                return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "jack: no MIDI port with that index");
            int rc = midi_jack_make_port_(st, j, name);
            if (rc != AUDIO_OK) return rc;
            unsafe {
                const char* mine = g_std_audio_jack_.port_name(j.port);
                int e = st.is_input ? g_std_audio_jack_.connect(j.client, (const char*)&target[0], mine)
                                    : g_std_audio_jack_.connect(j.client, mine, (const char*)&target[0]);
                if (e != 0 && e != 17) {   // 17: EEXIST, already connected
                    return audio_fail2_(&st.error, AUDIO_ERR_SYSTEM, "jack: could not connect to", &target[0]);
                }
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_jack_open_virtual_(&MidiPortState_ st, const &char name) {
    match (midi_jack_of_(st)) {
        case some(j): { return midi_jack_make_port_(st, j, name); }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_jack_close_port_(&MidiPortState_ st) {
    match (midi_jack_of_(st)) {
        case some(j): {
            unsafe {
                if (j.active) { g_std_audio_jack_.deactivate(j.client); j.active = 0; }
                if (j.port != (JackPort_*)0) { g_std_audio_jack_.port_unregister(j.client, j.port); j.port = (JackPort_*)0; }
            }
        }
        case null: {}
    }
}

static void midi_jack_q_put_(&JackMidi_ j, unsigned long at, const &unsigned char src, unsigned long n) {
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) j.q[(at + i) & (MIDI_JACK_OUT_QUEUE - 1UL)] = src[i];
    }
}

static int midi_jack_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (midi_jack_of_(st)) {
        case some(j): {
            if (j.port == (JackPort_*)0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "jack: no port open");
            if (len > 1024UL) return audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "jack: message longer than 1024 bytes");
            unsigned long head = j.q_head;
            unsigned long tail = 0UL;
            unsafe { tail = atomic_load((_Atomic unsigned long*)&j.q_tail); }
            if (4UL + len > MIDI_JACK_OUT_QUEUE - (head - tail))
                return audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "jack: MIDI output queue full");
            unsigned int l32 = (unsigned int)len;
            unsafe {
                midi_jack_q_put_(j, head, (const &unsigned char)&l32, 4UL);
                midi_jack_q_put_(j, head + 4UL, msg, len);
                atomic_store((_Atomic unsigned long*)&j.q_head, head + 4UL + len);
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_jack_client_close_(&MidiPortState_ st) {
    match (midi_jack_of_(st)) {
        case some(j): {
            midi_jack_close_port_(st);
            unsafe {
                g_std_audio_jack_.client_close(j.client);
                dealloc(j.q);
                dealloc((?&heap JackMidi_)j);
                st.backend = null;
            }
        }
        case null: {}
    }
}

static struct MidiBackendVt_ g_midi_jack_vt_;
static int g_midi_jack_vt_ready_ = 0;

&static MidiBackendVt_ midi_jack_backend_() {
    if (!g_midi_jack_vt_ready_) {
        g_midi_jack_vt_.api = MIDI_API_JACK;
        g_midi_jack_vt_.available = midi_jack_available_;
        g_midi_jack_vt_.client_open = midi_jack_client_open_;
        g_midi_jack_vt_.port_count = midi_jack_port_count_;
        g_midi_jack_vt_.port_name = midi_jack_port_name_;
        g_midi_jack_vt_.open_port = midi_jack_open_port_;
        g_midi_jack_vt_.open_virtual = midi_jack_open_virtual_;
        g_midi_jack_vt_.close_port = midi_jack_close_port_;
        g_midi_jack_vt_.send = midi_jack_send_;
        g_midi_jack_vt_.client_close = midi_jack_client_close_;
        g_midi_jack_vt_ready_ = 1;
    }
    return &g_midi_jack_vt_;
}

} // namespace std
