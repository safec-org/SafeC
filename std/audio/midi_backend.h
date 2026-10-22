// SafeC Standard Library — MIDI: the interface between midi.sc and the
// per-API backends (midi_<api>[_<os>].sc). std-private.
//
// A MidiIn/MidiOut is one MidiPortState_. Backends list the system's ports,
// open/close one and send; for input they hand complete messages (or raw
// byte streams) to midi_in_deliver_() / midi_in_feed_() on whatever thread
// they receive on — those apply the ignore filter, then call the user's
// callback or push to the lock-free queue, without allocating or locking.
#pragma once
#include <std/audio/midi.h>

namespace std {

struct MidiPortState_ {
    int api;
    void* vt;                         // the MidiBackendVt_
    int is_input;
    char client_name[MIDI_CLIENT_NAME_MAX];
    int port_open;
    struct AudioError error;          // the last control-thread error

    // ── input ────────────────────────────────────────────────────────────
    unsigned char* ring;              // queue: [u32 length][f64 time][bytes], wrapping
    unsigned long ring_cap;           // bytes (power of two)
    unsigned long ring_head;          // (rt) producer's write offset, monotonic
    unsigned long ring_tail;          // consumer's read offset, monotonic
    void* callback;                   // (rt) MidiCallbackRaw_, null = queue
    void* user;                       // (rt)
    int ignore;                       // (rt) bit 1 sysex, 2 timing, 4 sensing
    long long t0_ns;                  // when the port opened
    double last_time;                 // (rt, producer only) previous message's time
    int have_last;
    unsigned long long dropped_count; // (rt)
    struct MidiParser parser;         // for byte-stream backends (producer only)
    unsigned char* sysex_buf;         // the parser's buffer (MIDI_MAX_SYSEX)

    void* backend;                    // backend-private
};

struct MidiBackendVt_ {
    int api;
    fn int() available;
    fn int(&MidiPortState_) client_open;                          // connect to the API as client_name
    fn unsigned int(&MidiPortState_) port_count;
    fn int(&MidiPortState_, unsigned int, &char, unsigned long) port_name;   // 0 = ok
    fn int(&MidiPortState_, unsigned int, const &char) open_port; // index, our port's name
    fn int(&MidiPortState_, const &char) open_virtual;
    fn void(&MidiPortState_) close_port;
    fn int(&MidiPortState_, const &unsigned char, unsigned long) send;
    fn void(&MidiPortState_) client_close;
};

&static MidiBackendVt_ midi_dummy_backend_();
#if defined(__linux__)
&static MidiBackendVt_ midi_alsa_backend_();
#endif
#if defined(__linux__) || defined(__APPLE__)
&static MidiBackendVt_ midi_jack_backend_();
#endif
#if defined(__APPLE__)
&static MidiBackendVt_ midi_coremidi_backend_();
#endif
#if defined(_WIN32)
&static MidiBackendVt_ midi_winmm_backend_();
#endif
#if defined(__wasm__)
&static MidiBackendVt_ midi_webmidi_backend_();
#endif

// ── for backends' input path (real-time safe) ────────────────────────────
// One complete message received at 'time' (seconds since the port opened;
// < 0 = now).
void midi_in_deliver_(&MidiPortState_ st, const &unsigned char msg, unsigned long len, double time);
// Raw MIDI bytes (any split, running status, interleaved real-time):
// parsed with st.parser and delivered message by message.
void midi_in_feed_(&MidiPortState_ st, const &unsigned char bytes, unsigned long n, double time);
// Seconds since the port opened.
double midi_in_now_(&MidiPortState_ st);
// (Re)starts the port clock and the parser — call from open_port/open_virtual.
void midi_in_reset_clock_(&MidiPortState_ st);

} // namespace std
