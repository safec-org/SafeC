// SafeC Standard Library — MIDI backend: dummy (an in-process loopback). See midi.h.
//
// No system MIDI at all: a process-wide set of "buses" connects this
// program's own MidiOut and MidiIn clients, which is what tests (and
// programs on machines without MIDI) need. Port 0 in both directions is
// "Dummy Loopback": everything any MidiOut on it sends reaches every MidiIn
// on it. Virtual ports work as on ALSA/Core MIDI: MidiOut::open_virtual_port
// ("x") adds an input port "x" for MidiIns to open, MidiIn::
// open_virtual_port("y") an output port "y" for MidiOuts. Delivery is
// synchronous — the receiving callback runs on the sending thread — and
// lock-free: up to MIDI_DUMMY_BUSES buses with MIDI_DUMMY_LISTENERS
// receivers each.
#pragma once
#include <std/mem.h>
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>

#define MIDI_DUMMY_BUSES     16
#define MIDI_DUMMY_LISTENERS 16
#define MIDI_DUMMY_KIND_FREE    0
#define MIDI_DUMMY_KIND_LOOP    1   // the built-in loopback (bus 0)
#define MIDI_DUMMY_KIND_OUT_VIRTUAL 2   // a MidiOut's virtual port: an input port for others
#define MIDI_DUMMY_KIND_IN_VIRTUAL  3   // a MidiIn's virtual port: an output port for others

namespace std {

struct MidiDummyBus_ {
    int kind;                                     // (atomic) MIDI_DUMMY_KIND_*
    char name[MIDI_PORT_NAME_MAX];
    void* listeners[MIDI_DUMMY_LISTENERS];        // (atomic) MidiPortState_*
    int busy;                                     // (atomic) senders delivering right now
};

static struct MidiDummyBus_ g_midi_dummy_bus_[MIDI_DUMMY_BUSES];

// Per client: the bus it sends to / listens on / owns.
struct MidiDummyClient_ {
    int bus;          // -1 none
    int slot;         // listener slot (inputs)
    int owns;         // the bus is this client's virtual port
};

static int midi_dummy_kind_(int b) {
    unsafe { return atomic_load((_Atomic int*)&g_midi_dummy_bus_[b].kind); }
}

static void midi_dummy_init_() {
    if (midi_dummy_kind_(0) != MIDI_DUMMY_KIND_FREE) return;
    audio_strcpy_(&g_midi_dummy_bus_[0].name[0], (unsigned long)MIDI_PORT_NAME_MAX, "Dummy Loopback");
    unsafe {
        int expected = MIDI_DUMMY_KIND_FREE;
        atomic_compare_exchange_strong((_Atomic int*)&g_midi_dummy_bus_[0].kind, &expected, MIDI_DUMMY_KIND_LOOP);
    }
}

static int midi_dummy_available_() { return 1; }

static ?&MidiDummyClient_ midi_dummy_of_(&MidiPortState_ st) {
    unsafe { return (?&MidiDummyClient_)st.backend; }
}

static int midi_dummy_client_open_(&MidiPortState_ st) {
    midi_dummy_init_();
    ?&MidiDummyClient_ cp = null;
    unsafe { cp = (?&MidiDummyClient_)alloc_zeroed_raw_(sizeof(struct MidiDummyClient_)); }
    match (cp) {
        case some(c): {
            c.bus = -1;
            c.slot = -1;
            unsafe { st.backend = (void*)c; }
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "dummy midi: out of memory");
}

// The bus behind port 'index' as an input (1) or output (0) sees them:
// inputs list the loopback and the MidiOuts' virtual ports, outputs the
// loopback and the MidiIns'.
static int midi_dummy_bus_at_(int input, unsigned int index) {
    unsigned int k = 0U;
    for (int b = 0; b < MIDI_DUMMY_BUSES; b = b + 1) {
        int kind = midi_dummy_kind_(b);
        int listed = kind == MIDI_DUMMY_KIND_LOOP ||
                     (input && kind == MIDI_DUMMY_KIND_OUT_VIRTUAL) || (!input && kind == MIDI_DUMMY_KIND_IN_VIRTUAL);
        if (!listed) continue;
        if (k == index) return b;
        k = k + 1U;
    }
    return -1;
}

static unsigned int midi_dummy_port_count_(&MidiPortState_ st) {
    unsigned int n = 0U;
    while (midi_dummy_bus_at_(st.is_input, n) >= 0) n = n + 1U;
    return n;
}

static int midi_dummy_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    int b = midi_dummy_bus_at_(st.is_input, index);
    if (b < 0) return AUDIO_ERR_INVALID_DEVICE;
    audio_strcpy_(out, cap, &g_midi_dummy_bus_[b].name[0]);
    return AUDIO_OK;
}

static int midi_dummy_listen_(&MidiPortState_ st, &MidiDummyClient_ c, int b) {
    for (int i = 0; i < MIDI_DUMMY_LISTENERS; i = i + 1) {
        unsafe {
            void* expected = (void*)0;
            if (atomic_compare_exchange_strong((_Atomic void**)&g_midi_dummy_bus_[b].listeners[i], &expected, (void*)st)) {
                c.bus = b;
                c.slot = i;
                return AUDIO_OK;
            }
        }
    }
    return audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "dummy midi: too many listeners on this port");
}

static int midi_dummy_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (midi_dummy_of_(st)) {
        case some(c): {
            int b = midi_dummy_bus_at_(st.is_input, index);
            if (b < 0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "dummy midi: no port with that index");
            c.owns = 0;
            if (st.is_input) return midi_dummy_listen_(st, c, b);
            c.bus = b;
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_dummy_open_virtual_(&MidiPortState_ st, const &char name) {
    match (midi_dummy_of_(st)) {
        case some(c): {
            int kind = st.is_input ? MIDI_DUMMY_KIND_IN_VIRTUAL : MIDI_DUMMY_KIND_OUT_VIRTUAL;
            for (int b = 1; b < MIDI_DUMMY_BUSES; b = b + 1) {
                int expected = MIDI_DUMMY_KIND_FREE;
                int got = 0;
                // Claim the bus (kind -1 = being set up), name it, publish it.
                unsafe { got = atomic_compare_exchange_strong((_Atomic int*)&g_midi_dummy_bus_[b].kind, &expected, -1); }
                if (!got) continue;
                audio_strcpy_(&g_midi_dummy_bus_[b].name[0], (unsigned long)MIDI_PORT_NAME_MAX, name);
                c.owns = 1;
                c.bus = b;
                if (st.is_input) {
                    int rc = midi_dummy_listen_(st, c, b);
                    if (rc != AUDIO_OK) {
                        unsafe { atomic_store((_Atomic int*)&g_midi_dummy_bus_[b].kind, MIDI_DUMMY_KIND_FREE); }
                        return rc;
                    }
                }
                unsafe { atomic_store((_Atomic int*)&g_midi_dummy_bus_[b].kind, kind); }
                return AUDIO_OK;
            }
            return audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "dummy midi: too many virtual ports");
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_dummy_wait_idle_(int b) {
    while (1) {
        int busy = 0;
        unsafe { busy = atomic_load((_Atomic int*)&g_midi_dummy_bus_[b].busy); }
        if (busy == 0) return;
        audio_sleep_us_(50LL);
    }
}

static void midi_dummy_close_port_(&MidiPortState_ st) {
    match (midi_dummy_of_(st)) {
        case some(c): {
            int b = c.bus;
            if (b < 0) return;
            if (c.slot >= 0) {
                unsafe { atomic_store((_Atomic void**)&g_midi_dummy_bus_[b].listeners[c.slot], (void*)0); }
            }
            if (c.owns) {
                // Disconnect everyone else listening on our virtual port too.
                unsafe {
                    atomic_store((_Atomic int*)&g_midi_dummy_bus_[b].kind, -1);
                    for (int i = 0; i < MIDI_DUMMY_LISTENERS; i = i + 1)
                        atomic_store((_Atomic void**)&g_midi_dummy_bus_[b].listeners[i], (void*)0);
                }
            }
            // A sender may still be delivering to us: wait it out.
            midi_dummy_wait_idle_(b);
            if (c.owns) {
                unsafe { atomic_store((_Atomic int*)&g_midi_dummy_bus_[b].kind, MIDI_DUMMY_KIND_FREE); }
            }
            c.bus = -1;
            c.slot = -1;
            c.owns = 0;
        }
        case null: {}
    }
}

static int midi_dummy_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (midi_dummy_of_(st)) {
        case some(c): {
            int b = c.bus;
            if (b < 0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "dummy midi: no port open");
            unsafe {
                atomic_fetch_add((_Atomic int*)&g_midi_dummy_bus_[b].busy, 1);
                for (int i = 0; i < MIDI_DUMMY_LISTENERS; i = i + 1) {
                    void* l = atomic_load((_Atomic void**)&g_midi_dummy_bus_[b].listeners[i]);
                    if (l != (void*)0) midi_in_feed_((&MidiPortState_)l, msg, len, -1.0);
                }
                atomic_fetch_sub((_Atomic int*)&g_midi_dummy_bus_[b].busy, 1);
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_dummy_client_close_(&MidiPortState_ st) {
    unsafe {
        dealloc_raw_(st.backend);
        st.backend = (void*)0;
    }
}

static struct MidiBackendVt_ g_midi_dummy_vt_;
static int g_midi_dummy_vt_ready_ = 0;

&static MidiBackendVt_ midi_dummy_backend_() {
    if (!g_midi_dummy_vt_ready_) {
        g_midi_dummy_vt_.api = MIDI_API_DUMMY;
        g_midi_dummy_vt_.available = midi_dummy_available_;
        g_midi_dummy_vt_.client_open = midi_dummy_client_open_;
        g_midi_dummy_vt_.port_count = midi_dummy_port_count_;
        g_midi_dummy_vt_.port_name = midi_dummy_port_name_;
        g_midi_dummy_vt_.open_port = midi_dummy_open_port_;
        g_midi_dummy_vt_.open_virtual = midi_dummy_open_virtual_;
        g_midi_dummy_vt_.close_port = midi_dummy_close_port_;
        g_midi_dummy_vt_.send = midi_dummy_send_;
        g_midi_dummy_vt_.client_close = midi_dummy_client_close_;
        g_midi_dummy_vt_ready_ = 1;
    }
    return &g_midi_dummy_vt_;
}

} // namespace std
