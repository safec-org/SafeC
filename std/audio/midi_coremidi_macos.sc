// SafeC Standard Library — MIDI backend: Core MIDI (macOS). See midi.h.
//
// Each MidiIn/MidiOut is a MIDIClient named after open()'s client name.
// Input ports are the system's MIDI sources, output ports its
// destinations, named by kMIDIPropertyDisplayName (falling back to
// kMIDIPropertyName). open_port() makes an input port connected to the
// source (MIDIInputPortCreate + MIDIPortConnectSource) or an output port
// sending to the destination (MIDIOutputPortCreate + MIDISend);
// open_virtual_port() publishes a destination (MidiIn: MIDIDestinationCreate)
// or a source (MidiOut: MIDISourceCreate + MIDIReceived) other apps see.
// Input packets arrive on Core MIDI's high-priority thread and go through
// std's parser (a packet may hold several messages, or part of a sysex).
//
// Built against the CoreMIDI and CoreFoundation frameworks; MIDIObjectRef
// & co. are UInt32, ItemCount is unsigned long, and the MIDIPacketList
// layout (4-byte packed; packets 4-aligned on arm64) follows <CoreMIDI/MIDIServices.h>.
// safeguard: link macos -framework CoreMIDI -framework CoreFoundation
#pragma once
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define CM_UTF8 0x08000100U   // kCFStringEncodingUTF8

namespace std {

extern int  MIDIClientCreate(const void* name, void* notifyProc, void* notifyRefCon, unsigned int* outClient);
extern int  MIDIClientDispose(unsigned int client);
extern int  MIDIInputPortCreate(unsigned int client, const void* name, void* readProc, void* refCon, unsigned int* outPort);
extern int  MIDIOutputPortCreate(unsigned int client, const void* name, unsigned int* outPort);
extern int  MIDIPortDispose(unsigned int port);
extern int  MIDIPortConnectSource(unsigned int port, unsigned int source, void* connRefCon);
extern int  MIDIPortDisconnectSource(unsigned int port, unsigned int source);
extern int  MIDIDestinationCreate(unsigned int client, const void* name, void* readProc, void* refCon, unsigned int* outDest);
extern int  MIDISourceCreate(unsigned int client, const void* name, unsigned int* outSrc);
extern int  MIDIEndpointDispose(unsigned int endpoint);
extern unsigned long MIDIGetNumberOfSources();
extern unsigned int  MIDIGetSource(unsigned long index);
extern unsigned long MIDIGetNumberOfDestinations();
extern unsigned int  MIDIGetDestination(unsigned long index);
extern int  MIDIObjectGetStringProperty(unsigned int obj, const void* propertyID, void** outStr);
extern void* MIDIPacketListInit(void* list);
extern void* MIDIPacketListAdd(void* list, unsigned long listSize, void* cur, unsigned long long time,
                               unsigned long nData, const unsigned char* data);
extern int  MIDISend(unsigned int port, unsigned int dest, const void* list);
extern int  MIDIReceived(unsigned int src, const void* list);
extern const void* kMIDIPropertyName;
extern const void* kMIDIPropertyDisplayName;
extern void* CFStringCreateWithCString(const void* alloc, const char* str, unsigned int encoding);
extern unsigned char CFStringGetCString(const void* str, char* buf, long size, unsigned int encoding);
extern void CFRelease(const void* cf);

struct CoreMidi_ {
    unsigned int client;
    unsigned int port;          // input/output port (0 none)
    unsigned int endpoint;      // connected source / destination, or our virtual endpoint
    int is_virtual;
    struct MidiPortState_* st;
};

static int midi_coremidi_available_() { return 1; }

static ?&CoreMidi_ cm_of_(&MidiPortState_ st) {
    unsafe { return (?&CoreMidi_)st.backend; }
}

static void* cm_cfstring_(const &char s) {
    unsafe { return CFStringCreateWithCString((const void*)0, (const char*)s, CM_UTF8); }
}

// The MIDIReadProc: walk the packet list, feed each packet's bytes.
static void cm_read_proc_(const unsigned char* list, void* refCon, void* srcRefCon) {
    unsafe {
        struct CoreMidi_* c = (struct CoreMidi_*)refCon;
        unsigned int n = *(const unsigned int*)list;
        const unsigned char* p = list + 4;                  // MIDIPacket packet[]
        for (unsigned int i = 0U; i < n; i = i + 1U) {
            unsigned short len = *(const unsigned short*)(p + 8);   // after the 8-byte timeStamp
            midi_in_feed_((&MidiPortState_)c->st, (const &unsigned char)(p + 10), (unsigned long)len, -1.0);
#if defined(__aarch64__)
            p = (const unsigned char*)(((unsigned long)(p + 10 + len) + 3UL) & ~3UL);
#else
            p = p + 10 + len;
#endif
        }
    }
}

static int midi_coremidi_client_open_(&MidiPortState_ st) {
    ?&CoreMidi_ cp = null;
    unsafe { cp = (?&CoreMidi_)alloc_zeroed_raw_(sizeof(struct CoreMidi_)); }
    match (cp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "coremidi: out of memory"); }
        case some(c): {
            unsafe {
                c.st = (struct MidiPortState_*)st;
                void* name = cm_cfstring_(&st.client_name[0]);
                int e = MIDIClientCreate((const void*)name, (void*)0, (void*)0, &c.client);
                if (name != (void*)0) CFRelease((const void*)name);
                if (e != 0) {
                    dealloc_raw_((void*)c);
                    return audio_fail_code_(&st.error, AUDIO_ERR_UNAVAILABLE, "coremidi: MIDIClientCreate", (long long)e);
                }
                st.backend = (void*)c;
            }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

static unsigned int cm_endpoint_at_(int input, unsigned int index) {
    unsafe {
        if (input) {
            if ((unsigned long)index >= MIDIGetNumberOfSources()) return 0U;
            return MIDIGetSource((unsigned long)index);
        }
        if ((unsigned long)index >= MIDIGetNumberOfDestinations()) return 0U;
        return MIDIGetDestination((unsigned long)index);
    }
}

static unsigned int midi_coremidi_port_count_(&MidiPortState_ st) {
    unsafe { return (unsigned int)(st.is_input ? MIDIGetNumberOfSources() : MIDIGetNumberOfDestinations()); }
}

static int midi_coremidi_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    unsigned int ep = cm_endpoint_at_(st.is_input, index);
    if (ep == 0U) return AUDIO_ERR_INVALID_DEVICE;
    unsafe {
        void* s = (void*)0;
        if (MIDIObjectGetStringProperty(ep, kMIDIPropertyDisplayName, &s) != 0 || s == (void*)0)
            MIDIObjectGetStringProperty(ep, kMIDIPropertyName, &s);
        audio_strcpy_(out, cap, "Core MIDI port");
        if (s != (void*)0) {
            CFStringGetCString((const void*)s, (char*)out, (long)cap, CM_UTF8);
            CFRelease((const void*)s);
        }
    }
    return AUDIO_OK;
}

static int midi_coremidi_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (cm_of_(st)) {
        case some(c): {
            unsigned int ep = cm_endpoint_at_(st.is_input, index);
            if (ep == 0U) return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "coremidi: no MIDI port with that index");
            unsafe {
                void* pname = cm_cfstring_(name);
                int e = st.is_input ? MIDIInputPortCreate(c.client, (const void*)pname, (void*)cm_read_proc_, (void*)c, &c.port)
                                    : MIDIOutputPortCreate(c.client, (const void*)pname, &c.port);
                if (pname != (void*)0) CFRelease((const void*)pname);
                if (e != 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coremidi: could not create a port", (long long)e);
                if (st.is_input) {
                    e = MIDIPortConnectSource(c.port, ep, (void*)0);
                    if (e != 0) {
                        MIDIPortDispose(c.port);
                        c.port = 0U;
                        return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coremidi: MIDIPortConnectSource", (long long)e);
                    }
                }
                c.endpoint = ep;
                c.is_virtual = 0;
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_coremidi_open_virtual_(&MidiPortState_ st, const &char name) {
    match (cm_of_(st)) {
        case some(c): {
            unsafe {
                void* pname = cm_cfstring_(name);
                int e = st.is_input ? MIDIDestinationCreate(c.client, (const void*)pname, (void*)cm_read_proc_, (void*)c, &c.endpoint)
                                    : MIDISourceCreate(c.client, (const void*)pname, &c.endpoint);
                if (pname != (void*)0) CFRelease((const void*)pname);
                if (e != 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coremidi: could not create a virtual port", (long long)e);
                c.is_virtual = 1;
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_coremidi_close_port_(&MidiPortState_ st) {
    match (cm_of_(st)) {
        case some(c): {
            unsafe {
                if (c.is_virtual) {
                    if (c.endpoint != 0U) MIDIEndpointDispose(c.endpoint);
                } else if (c.port != 0U) {
                    if (st.is_input && c.endpoint != 0U) MIDIPortDisconnectSource(c.port, c.endpoint);
                    MIDIPortDispose(c.port);
                }
            }
            c.port = 0U;
            c.endpoint = 0U;
            c.is_virtual = 0;
        }
        case null: {}
    }
}

static int midi_coremidi_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (cm_of_(st)) {
        case some(c): {
            if (c.endpoint == 0U) return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "coremidi: no port open");
            // A packet list big enough for the message (stack for short ones).
            unsigned long size = len + 128UL;
            unsigned char local[512];
            int rc = AUDIO_OK;
            unsafe {
                unsigned char* list = (unsigned char*)&local[0];
                if (size > 512UL) {
                    list = (unsigned char*)alloc_raw_(size);
                    if (list == (unsigned char*)0) return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "coremidi: out of memory");
                }
                void* pkt = MIDIPacketListInit((void*)list);
                pkt = MIDIPacketListAdd((void*)list, size, pkt, 0ULL, len, (const unsigned char*)msg);
                if (pkt == (void*)0) rc = audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "coremidi: message too long");
                else {
                    int e = c.is_virtual ? MIDIReceived(c.endpoint, (const void*)list)
                                         : MIDISend(c.port, c.endpoint, (const void*)list);
                    if (e != 0) rc = audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coremidi: MIDISend", (long long)e);
                }
                if (list != (unsigned char*)&local[0]) dealloc_raw_((void*)list);
            }
            return rc;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_coremidi_client_close_(&MidiPortState_ st) {
    match (cm_of_(st)) {
        case some(c): {
            midi_coremidi_close_port_(st);
            unsafe {
                MIDIClientDispose(c.client);
                dealloc_raw_((void*)c);
                st.backend = (void*)0;
            }
        }
        case null: {}
    }
}

static struct MidiBackendVt_ g_midi_coremidi_vt_;
static int g_midi_coremidi_vt_ready_ = 0;

&static MidiBackendVt_ midi_coremidi_backend_() {
    if (!g_midi_coremidi_vt_ready_) {
        g_midi_coremidi_vt_.api = MIDI_API_COREMIDI;
        g_midi_coremidi_vt_.available = midi_coremidi_available_;
        g_midi_coremidi_vt_.client_open = midi_coremidi_client_open_;
        g_midi_coremidi_vt_.port_count = midi_coremidi_port_count_;
        g_midi_coremidi_vt_.port_name = midi_coremidi_port_name_;
        g_midi_coremidi_vt_.open_port = midi_coremidi_open_port_;
        g_midi_coremidi_vt_.open_virtual = midi_coremidi_open_virtual_;
        g_midi_coremidi_vt_.close_port = midi_coremidi_close_port_;
        g_midi_coremidi_vt_.send = midi_coremidi_send_;
        g_midi_coremidi_vt_.client_close = midi_coremidi_client_close_;
        g_midi_coremidi_vt_ready_ = 1;
    }
    return &g_midi_coremidi_vt_;
}

} // namespace std
