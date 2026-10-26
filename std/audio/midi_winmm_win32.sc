// SafeC Standard Library — MIDI backend: Windows MultiMedia (WinMM). See midi.h.
//
// Ports are the WinMM MIDI devices (midiInGetNumDevs / midiOutGetNumDevs),
// named by their caps' szPname with the device index appended when two
// share a name. WinMM has no virtual ports: open_virtual_port() returns
// AUDIO_ERR_UNSUPPORTED (a loopback driver such as loopMIDI provides
// them system-wide). Input arrives on WinMM's callback thread
// (CALLBACK_FUNCTION): MIM_DATA short messages are unpacked, MIM_LONGDATA
// sysex buffers (MIDI_WINMM_SYSEX_BUFFERS of MIDI_WINMM_SYSEX_SIZE bytes,
// re-queued as they come back) go through std's parser. Output uses
// midiOutShortMsg for short messages and a prepared MIDIHDR with
// midiOutLongMsg (waiting for it to finish) for sysex.
//
// Structures follow <mmsystem.h> for 64-bit Windows: MIDIINCAPSA is 44
// bytes, MIDIOUTCAPSA 52, MIDIHDR 120; DWORD_PTR is pointer-sized.
// safeguard: link windows -lwinmm
#pragma once
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define WMM_CALLBACK_FUNCTION 0x00030000U
#define WMM_CALLBACK_NULL     0U
#define WMM_MIM_DATA          0x3C3U
#define WMM_MIM_LONGDATA      0x3C4U
#define WMM_MIM_ERROR         0x3C5U
#define WMM_MIM_LONGERROR     0x3C6U
#define WMM_MIM_MOREDATA      0x3CCU
#define WMM_STILLPLAYING      65
#define MIDI_WINMM_SYSEX_BUFFERS 4
#define MIDI_WINMM_SYSEX_SIZE    1024U

namespace std {

struct WmmInCaps_ {
    unsigned short mid;
    unsigned short pid;
    unsigned int   driver_version;
    char           name[32];
    unsigned int   support;
};

struct WmmOutCaps_ {
    unsigned short mid;
    unsigned short pid;
    unsigned int   driver_version;
    char           name[32];
    unsigned short technology;
    unsigned short voices;
    unsigned short notes;
    unsigned short channel_mask;
    unsigned int   support;
};

struct WmmHdr_ {
    char*          data;
    unsigned int   buffer_length;
    unsigned int   bytes_recorded;
    unsigned long long user;
    unsigned int   flags;
    void*          next;
    unsigned long long reserved;
    unsigned int   offset;
    unsigned long long reserved8[8];
};

static_assert(sizeof(struct WmmInCaps_) == 44, "MIDIINCAPSA is 44 bytes");
static_assert(sizeof(struct WmmOutCaps_) == 52, "MIDIOUTCAPSA is 52 bytes");
static_assert(sizeof(struct WmmHdr_) == 120, "MIDIHDR is 120 bytes on 64-bit Windows");

extern unsigned int midiInGetNumDevs();
extern unsigned int midiOutGetNumDevs();
extern unsigned int midiInGetDevCapsA(unsigned long long id, WmmInCaps_* caps, unsigned int size);
extern unsigned int midiOutGetDevCapsA(unsigned long long id, WmmOutCaps_* caps, unsigned int size);
extern unsigned int midiInOpen(void** handle, unsigned int id, unsigned long long callback, unsigned long long instance, unsigned int flags);
extern unsigned int midiInClose(void* h);
extern unsigned int midiInStart(void* h);
extern unsigned int midiInStop(void* h);
extern unsigned int midiInReset(void* h);
extern unsigned int midiInPrepareHeader(void* h, WmmHdr_* hdr, unsigned int size);
extern unsigned int midiInUnprepareHeader(void* h, WmmHdr_* hdr, unsigned int size);
extern unsigned int midiInAddBuffer(void* h, WmmHdr_* hdr, unsigned int size);
extern unsigned int midiOutOpen(void** handle, unsigned int id, unsigned long long callback, unsigned long long instance, unsigned int flags);
extern unsigned int midiOutClose(void* h);
extern unsigned int midiOutReset(void* h);
extern unsigned int midiOutShortMsg(void* h, unsigned int msg);
extern unsigned int midiOutLongMsg(void* h, WmmHdr_* hdr, unsigned int size);
extern unsigned int midiOutPrepareHeader(void* h, WmmHdr_* hdr, unsigned int size);
extern unsigned int midiOutUnprepareHeader(void* h, WmmHdr_* hdr, unsigned int size);
extern void Sleep(unsigned int ms);

struct WinMidi_ {
    void* handle;
    int closing;                                    // (atomic) don't re-queue sysex buffers
    struct WmmHdr_ hdr[MIDI_WINMM_SYSEX_BUFFERS];
    unsigned char* sysex;                           // MIDI_WINMM_SYSEX_BUFFERS * MIDI_WINMM_SYSEX_SIZE
    struct MidiPortState_* st;
};

static int midi_winmm_available_() { return 1; }

static ?&WinMidi_ wmm_of_(&MidiPortState_ st) {
    unsafe { return (?&WinMidi_)st.backend; }
}

static int midi_winmm_client_open_(&MidiPortState_ st) {
    ?&WinMidi_ wp = null;
    unsafe { wp = (?&WinMidi_)alloc_zeroed_raw_(sizeof(struct WinMidi_)); }
    match (wp) {
        case null: {}
        case some(w): {
            int ok = 1;
            unsafe {
                w.st = (struct MidiPortState_*)st;
                if (st.is_input) {
                    w.sysex = (unsigned char*)alloc_zeroed_raw_((unsigned long)(MIDI_WINMM_SYSEX_BUFFERS * (int)MIDI_WINMM_SYSEX_SIZE));
                    if (w.sysex == (unsigned char*)0) ok = 0;
                }
                if (ok) st.backend = (?&heap MidiPortImpl_)w;
                else dealloc(w);
            }
            if (ok) return AUDIO_OK;
        }
    }
    return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "winmm: out of memory");
}

static unsigned int midi_winmm_port_count_(&MidiPortState_ st) {
    unsafe { return st.is_input ? midiInGetNumDevs() : midiOutGetNumDevs(); }
}

static int wmm_raw_name_(int input, unsigned int index, &char out, unsigned long cap) {
    unsafe {
        if (input) {
            struct WmmInCaps_ c;
            safe_memset_obj(&c, 0);
            if (midiInGetDevCapsA((unsigned long long)index, &c, (unsigned int)sizeof(struct WmmInCaps_)) != 0U) return 0;
            c.name[31] = (char)0;
            audio_strcpy_(out, cap, &c.name[0]);
        } else {
            struct WmmOutCaps_ c;
            safe_memset_obj(&c, 0);
            if (midiOutGetDevCapsA((unsigned long long)index, &c, (unsigned int)sizeof(struct WmmOutCaps_)) != 0U) return 0;
            c.name[31] = (char)0;
            audio_strcpy_(out, cap, &c.name[0]);
        }
    }
    return 1;
}

static int midi_winmm_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    if (index >= midi_winmm_port_count_(st)) return AUDIO_ERR_INVALID_DEVICE;
    if (!wmm_raw_name_(st.is_input, index, out, cap)) return AUDIO_ERR_SYSTEM;
    // Disambiguate devices with the same name, as RtMidi does.
    unsigned int n = midi_winmm_port_count_(st);
    for (unsigned int i = 0U; i < n; i = i + 1U) {
        if (i == index) continue;
        char other[64];
        if (wmm_raw_name_(st.is_input, i, &other[0], 64UL) && audio_streq_(&other[0], out)) {
            audio_strcat_(out, cap, " ");
            audio_strcat_int_(out, cap, (long long)index);
            break;
        }
    }
    return AUDIO_OK;
}

// WinMM's MidiInProc (its callback thread).
static void wmm_in_proc_(void* h, unsigned int msg, unsigned long long instance, unsigned long long p1, unsigned long long p2) {
    unsafe {
        struct WinMidi_* w = (struct WinMidi_*)instance;
        if (w == (struct WinMidi_*)0) return;
        struct MidiPortState_* st = w->st;
        double t = (double)p2 / 1000.0;                // ms since midiInStart
        if (msg == WMM_MIM_DATA || msg == WMM_MIM_MOREDATA) {
            unsigned char b[3];
            b[0] = (unsigned char)(p1 & 0xFFULL);
            b[1] = (unsigned char)((p1 >> 8) & 0xFFULL);
            b[2] = (unsigned char)((p1 >> 16) & 0xFFULL);
            int len = midi_message_length((int)b[0]);
            if (len < 1 || len > 3) return;
            midi_in_deliver_((&MidiPortState_)st, &b[0], (unsigned long)len, t);
            return;
        }
        if (msg == WMM_MIM_LONGDATA || msg == WMM_MIM_LONGERROR) {
            struct WmmHdr_* hdr = (struct WmmHdr_*)p1;
            if (hdr->bytes_recorded > 0U)
                midi_in_feed_((&MidiPortState_)st, (const &unsigned char)hdr->data, (unsigned long)hdr->bytes_recorded, t);
            // Hand the buffer back for the next sysex chunk (as RtMidi does
            // from its callback), unless the port is closing.
            if (atomic_load((_Atomic int*)&w->closing) == 0)
                midiInAddBuffer(w->handle, hdr, (unsigned int)sizeof(struct WmmHdr_));
        }
    }
}

static int midi_winmm_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (wmm_of_(st)) {
        case some(w): {
            if (index >= midi_winmm_port_count_(st))
                return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "winmm: no MIDI port with that index");
            unsigned int r = 0U;
            unsafe {
                if (!st.is_input) {
                    r = midiOutOpen(&w.handle, index, 0ULL, 0ULL, WMM_CALLBACK_NULL);
                    if (r != 0U) return audio_fail_code_(&st.error, AUDIO_ERR_DEVICE_BUSY, "winmm: midiOutOpen", (long long)r);
                    return AUDIO_OK;
                }
                atomic_store((_Atomic int*)&w.closing, 0);
                r = midiInOpen(&w.handle, index, (unsigned long long)(void*)wmm_in_proc_, (unsigned long long)(void*)w,
                               WMM_CALLBACK_FUNCTION);
                if (r != 0U) return audio_fail_code_(&st.error, AUDIO_ERR_DEVICE_BUSY, "winmm: midiInOpen", (long long)r);
                for (int i = 0; i < MIDI_WINMM_SYSEX_BUFFERS; i = i + 1) {
                    safe_memset_obj(&w.hdr[i], 0);
                    w.hdr[i].data = (char*)(w.sysex + (unsigned long)i * (unsigned long)MIDI_WINMM_SYSEX_SIZE);
                    w.hdr[i].buffer_length = MIDI_WINMM_SYSEX_SIZE;
                    midiInPrepareHeader(w.handle, &w.hdr[i], (unsigned int)sizeof(struct WmmHdr_));
                    midiInAddBuffer(w.handle, &w.hdr[i], (unsigned int)sizeof(struct WmmHdr_));
                }
                r = midiInStart(w.handle);
                if (r != 0U) {
                    midiInClose(w.handle);
                    w.handle = (void*)0;
                    return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "winmm: midiInStart", (long long)r);
                }
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_winmm_open_virtual_(&MidiPortState_ st, const &char name) {
    return audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "winmm: no virtual MIDI ports on Windows");
}

static void midi_winmm_close_port_(&MidiPortState_ st) {
    match (wmm_of_(st)) {
        case some(w): {
            unsafe {
                if (w.handle == (void*)0) return;
                if (st.is_input) {
                    atomic_store((_Atomic int*)&w.closing, 1);
                    midiInStop(w.handle);
                    midiInReset(w.handle);         // returns the sysex buffers
                    for (int i = 0; i < MIDI_WINMM_SYSEX_BUFFERS; i = i + 1)
                        midiInUnprepareHeader(w.handle, &w.hdr[i], (unsigned int)sizeof(struct WmmHdr_));
                    midiInClose(w.handle);
                } else {
                    midiOutReset(w.handle);
                    midiOutClose(w.handle);
                }
                w.handle = (void*)0;
            }
        }
        case null: {}
    }
}

static int midi_winmm_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (wmm_of_(st)) {
        case some(w): {
            if (w.handle == (void*)0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "winmm: no port open");
            unsigned int r = 0U;
            unsafe {
                int status = (int)msg[0];
                if (status != MIDI_SYSEX && len <= 3UL) {
                    unsigned int word = (unsigned int)msg[0];
                    if (len > 1UL) word = word | ((unsigned int)msg[1] << 8);
                    if (len > 2UL) word = word | ((unsigned int)msg[2] << 16);
                    r = midiOutShortMsg(w.handle, word);
                    if (r != 0U) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "winmm: midiOutShortMsg", (long long)r);
                    return AUDIO_OK;
                }
                // Sysex: a prepared header, sent and waited for.
                char* copy = (char*)alloc_raw_(len);
                if (copy == (char*)0) return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "winmm: out of memory");
                safe_memcpy_raw_((void*)copy, (const void*)msg, len);
                struct WmmHdr_ hdr;
                safe_memset_obj(&hdr, 0);
                hdr.data = copy;
                hdr.buffer_length = (unsigned int)len;
                hdr.bytes_recorded = (unsigned int)len;
                r = midiOutPrepareHeader(w.handle, &hdr, (unsigned int)sizeof(struct WmmHdr_));
                if (r == 0U) r = midiOutLongMsg(w.handle, &hdr, (unsigned int)sizeof(struct WmmHdr_));
                if (r == 0U) {
                    while (midiOutUnprepareHeader(w.handle, &hdr, (unsigned int)sizeof(struct WmmHdr_)) == (unsigned int)WMM_STILLPLAYING)
                        Sleep(1U);
                }
                dealloc(copy);
                if (r != 0U) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "winmm: midiOutLongMsg", (long long)r);
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_winmm_client_close_(&MidiPortState_ st) {
    match (wmm_of_(st)) {
        case some(w): {
            midi_winmm_close_port_(st);
            unsafe {
                dealloc(w.sysex);
                dealloc(w);
                st.backend = null;
            }
        }
        case null: {}
    }
}

static struct MidiBackendVt_ g_midi_winmm_vt_;
static int g_midi_winmm_vt_ready_ = 0;

&static MidiBackendVt_ midi_winmm_backend_() {
    if (!g_midi_winmm_vt_ready_) {
        g_midi_winmm_vt_.api = MIDI_API_WINMM;
        g_midi_winmm_vt_.available = midi_winmm_available_;
        g_midi_winmm_vt_.client_open = midi_winmm_client_open_;
        g_midi_winmm_vt_.port_count = midi_winmm_port_count_;
        g_midi_winmm_vt_.port_name = midi_winmm_port_name_;
        g_midi_winmm_vt_.open_port = midi_winmm_open_port_;
        g_midi_winmm_vt_.open_virtual = midi_winmm_open_virtual_;
        g_midi_winmm_vt_.close_port = midi_winmm_close_port_;
        g_midi_winmm_vt_.send = midi_winmm_send_;
        g_midi_winmm_vt_.client_close = midi_winmm_client_close_;
        g_midi_winmm_vt_ready_ = 1;
    }
    return &g_midi_winmm_vt_;
}

} // namespace std
