// SafeC Standard Library — real-time MIDI I/O (in the spirit of RtMidi).
//
//   API                 platform        backend file
//   MIDI_API_ALSA       Linux           midi_alsa_linux.sc     (ALSA sequencer, libasound loaded at run time)
//   MIDI_API_JACK       Linux, macOS    midi_jack_posix.sc     (JACK MIDI, libjack loaded at run time)
//   MIDI_API_COREMIDI   macOS           midi_coremidi_macos.sc
//   MIDI_API_WINMM      Windows         midi_winmm_win32.sc    (the Windows multimedia MIDI API)
//   MIDI_API_WEBMIDI    wasm32          midi_webmidi_wasm.sc   (+ std/audio/audio_web.js)
//   MIDI_API_DUMMY      everywhere      midi_dummy.sc          (an in-process loopback, for tests)
//
// A MidiIn or MidiOut is one client of an API. It lists the system's ports,
// connects to one (open_port) or creates its own that other programs
// connect to (open_virtual_port: ALSA, JACK, Core MIDI and dummy — WinMM
// and Web MIDI have no virtual ports and return AUDIO_ERR_UNSUPPORTED).
//
//   struct Result<MidiIn, AudioError> r = std::MidiIn::open(MIDI_API_UNSPECIFIED, "monitor", 0UL);
//   struct MidiIn in = r.unwrap_or(std::MidiIn::closed());
//   in.open_port(0U, "in");
//   in.set_callback(on_midi, (?&State)&state);      // or poll: in.get_message(...)
//
//   struct MidiOut out = std::MidiOut::open(MIDI_API_UNSPECIFIED, "synth").unwrap_or(std::MidiOut::closed());
//   out.open_port(0U, "out");
//   out.send(std::MidiMsg::note_on(0, 60, 100));
//
// Input arrives on the backend's thread (the ALSA sequencer thread std
// starts, JACK's process thread, Core MIDI's and WinMM's callback threads,
// the browser's event loop). With a callback set, it is called there with
// each complete message — sysex reassembled, running status expanded — and
// must follow the same real-time rules as an audio callback (see audio.h):
// no allocation, no locks, no blocking. Without one, messages go into a
// lock-free single-producer/single-consumer queue (sized at open, default
// MIDI_DEFAULT_QUEUE_BYTES) that get_message() drains from any one thread;
// a message that doesn't fit is dropped and counted (dropped()). Sysex,
// MIDI clock/time code and active sensing are filtered out by default
// (ignore_types(1, 1, 1)), as RtMidi does.
//
// Timestamps are seconds since the port was opened (monotonic); each event
// also carries the time since the previous delivered message.
#pragma once
#include <std/audio/audio_error.h>
#include <std/collections/slice.h>
#include <std/result.h>

#define MIDI_API_UNSPECIFIED 0
#define MIDI_API_ALSA        1
#define MIDI_API_JACK        2
#define MIDI_API_COREMIDI    3
#define MIDI_API_WINMM       4
#define MIDI_API_WEBMIDI     5
#define MIDI_API_DUMMY       6
#define MIDI_API_COUNT       7

#define MIDI_PORT_NAME_MAX       128
#define MIDI_CLIENT_NAME_MAX     64
#define MIDI_DEFAULT_QUEUE_BYTES 16384UL
#define MIDI_MAX_SYSEX           65536UL   // longest sysex std reassembles

// ── Status bytes ─────────────────────────────────────────────────────────
#define MIDI_NOTE_OFF          0x80
#define MIDI_NOTE_ON           0x90
#define MIDI_POLY_PRESSURE     0xA0
#define MIDI_CONTROL_CHANGE    0xB0
#define MIDI_PROGRAM_CHANGE    0xC0
#define MIDI_CHANNEL_PRESSURE  0xD0
#define MIDI_PITCH_BEND        0xE0
#define MIDI_SYSEX             0xF0
#define MIDI_TIME_CODE         0xF1
#define MIDI_SONG_POSITION     0xF2
#define MIDI_SONG_SELECT       0xF3
#define MIDI_TUNE_REQUEST      0xF6
#define MIDI_SYSEX_END         0xF7
#define MIDI_CLOCK             0xF8
#define MIDI_START             0xFA
#define MIDI_CONTINUE          0xFB
#define MIDI_STOP              0xFC
#define MIDI_ACTIVE_SENSING    0xFE
#define MIDI_SYSTEM_RESET      0xFF

// MidiParser::feed() results.
#define MIDI_PARSE_NONE    0   // byte consumed, nothing complete yet
#define MIDI_PARSE_MESSAGE 1   // a channel / system common / real-time message: message()
#define MIDI_PARSE_SYSEX   2   // a complete sysex (F0 .. F7) in the sysex buffer
#define MIDI_PARSE_ERROR   3   // a stray data byte or a sysex too long for the buffer (dropped)

namespace std {

// A MIDI message of up to 3 bytes: everything but sysex.
struct MidiMsg {
    unsigned char data[3];
    int           len;

    static struct MidiMsg note_on(int channel, int note, int velocity);
    static struct MidiMsg note_off(int channel, int note, int velocity);
    static struct MidiMsg poly_pressure(int channel, int note, int pressure);
    static struct MidiMsg control_change(int channel, int controller, int value);
    static struct MidiMsg program_change(int channel, int program);
    static struct MidiMsg channel_pressure(int channel, int pressure);
    // value: -8192 .. 8191 (0 = centre)
    static struct MidiMsg pitch_bend(int channel, int value);
    static struct MidiMsg song_position(int beats);     // 0 .. 16383 sixteenth notes
    static struct MidiMsg song_select(int song);
    // A one-byte system message: MIDI_CLOCK, MIDI_START, MIDI_STOP, ...
    static struct MidiMsg system(int status);
    // The first 1..3 bytes of 'bytes' as a message (len 0 if n is 0 or > 3).
    static struct MidiMsg of(const &unsigned char bytes, unsigned long n);

    int status() const;       // the status byte (0 for an empty message)
    int type() const;         // status & 0xF0 for channel messages, else the status
    int channel() const;      // 0..15 for channel messages, -1 otherwise
    int data1() const;        // first data byte (note, controller, program), -1 if none
    int data2() const;        // second data byte (velocity, value), -1 if none
    int is_note_on() const;   // note on with velocity > 0
    int is_note_off() const;  // note off, or note on with velocity 0
    int pitch_bend_value() const;   // -8192 .. 8191 for a pitch bend, 0 otherwise
};

// Expected length of a message starting with 'status': 1..3, 0 for sysex
// (variable), -1 for a data byte (< 0x80) or an undefined status.
int midi_message_length(int status);
// Whether 'status' is a real-time byte (0xF8..0xFF), which may appear
// between the bytes of any other message.
int midi_is_realtime(int status);
// F0 <payload> F7 into 'out'; the length written, or -1 if it doesn't fit
// or the payload has a byte >= 0x80.
long midi_build_sysex(&unsigned char out, unsigned long cap, const &unsigned char payload, unsigned long n);
// 14-bit value from LSB/MSB data bytes, and back.
int  midi_14bit(int lsb, int msb);

// A byte-stream parser: running status, real-time bytes interleaved with
// other messages, sysex reassembly into a caller-owned buffer. Allocation-
// free; safe on a real-time thread.
struct MidiParser {
    unsigned char running;      // current running status (0 = none)
    unsigned char msg[3];
    int           have;         // bytes in msg
    int           need;         // bytes msg needs
    struct MidiMsg last;        // the last complete short message
    ?&unsigned char sysex;      // sysex buffer (may be null: sysex dropped)
    unsigned long sysex_cap;
    unsigned long sysex_len;
    int           in_sysex;
    int           sysex_overflow;

    // 'sysex_buf'/'cap' receive reassembled sysex messages (F0 .. F7).
    static struct MidiParser new(?&unsigned char sysex_buf, unsigned long cap);
    // One byte in; MIDI_PARSE_*.
    int  feed(int byte);
    struct MidiMsg message() const;
    unsigned long sysex_length() const;
    void reset();
};

// The reverse: writes messages for a byte stream, dropping the status byte
// when running status allows (channel messages with the previous status).
struct MidiEncoder {
    unsigned char running;
    int           use_running_status;

    static struct MidiEncoder new(int use_running_status);
    // Bytes written to 'out', or -1 if 'cap' is too small.
    long encode(const &unsigned char msg, unsigned long len, &unsigned char out, unsigned long cap);
    void reset();
};

// One input message, as a callback sees it. 'bytes' is valid only during
// the callback.
struct MidiEvent {
    double                     timestamp;   // seconds since the port was opened
    double                     delta;       // seconds since the previous message
    struct Slice<unsigned char> bytes;      // the complete message

    int            status() const;
    // The message as a MidiMsg (len 0 for sysex).
    struct MidiMsg short_msg() const;
};

struct MidiPortInfo {
    unsigned int index;
    char         name[MIDI_PORT_NAME_MAX];
};

// The input callback.
typedef fn void(&MidiEvent, void*) MidiCallbackRaw_;

struct MidiIn {
    int    api_id;
    ?&void impl;     // std-private; null when closed

    // A client of 'api' (MIDI_API_UNSPECIFIED: midi_default_api()) named
    // 'client_name' (what other programs see); 'queue_bytes' sizes the
    // polling queue (0 = MIDI_DEFAULT_QUEUE_BYTES).
    static struct Result<MidiIn, AudioError> open(int api, const &char client_name, unsigned long queue_bytes);
    static struct MidiIn closed();

    int  is_open() const;
    int  api() const;
    unsigned int port_count();
    struct Result<MidiPortInfo, AudioError> port_info(unsigned int index);
    // Connects to the index-th port; 'port_name' names our end.
    struct AudioError open_port(unsigned int index, const &char port_name);
    // Creates a port other programs can connect to.
    struct AudioError open_virtual_port(const &char port_name);
    void close_port();
    int  is_port_open() const;

    generic<T> void set_callback(fn void(&MidiEvent, ?&T) callback, ?&T user);
    void set_callback_raw_(void* callback, void* user);
    // Back to the polling queue.
    void cancel_callback();
    // Which message kinds to drop (non-zero = drop): sysex, clock/time code, active sensing.
    void ignore_types(int sysex, int timing, int active_sensing);

    // Polling: copies the next queued message into 'buf' and returns its
    // length (its timestamp into 'timestamp'); 0 when the queue is empty;
    // -(length) when 'buf' is too small (the message stays queued).
    long get_message(&unsigned char buf, unsigned long cap, &double timestamp);
    // Messages lost to a full queue or an oversized sysex.
    unsigned long dropped() const;
    void drop();

    void close() consume;   // = drop()
};

struct MidiOut {
    int    api_id;
    ?&void impl;

    static struct Result<MidiOut, AudioError> open(int api, const &char client_name);
    static struct MidiOut closed();

    int  is_open() const;
    int  api() const;
    unsigned int port_count();
    struct Result<MidiPortInfo, AudioError> port_info(unsigned int index);
    struct AudioError open_port(unsigned int index, const &char port_name);
    struct AudioError open_virtual_port(const &char port_name);
    void close_port();
    int  is_port_open() const;

    // Sends one complete message (a short message or a whole sysex).
    struct AudioError send_message(const &unsigned char msg, unsigned long len);
    struct AudioError send(struct MidiMsg msg);
    void drop();

    void close() consume;   // = drop()
};

generic<T>
void MidiIn::set_callback(fn void(&MidiEvent, ?&T) callback, ?&T user) {
    unsafe { self.set_callback_raw_((void*)callback, (void*)user); }
}

// The APIs compiled into this build, in preference order.
int midi_compiled_apis(&int out, int cap);
int midi_api_compiled(int api);
// Core MIDI, WinMM, Web MIDI, ALSA (JACK when its server is running); the
// SAFEC_MIDI_API environment variable ("alsa", "jack", "dummy", ...) overrides.
int midi_default_api();
const &char midi_api_name(int api);
const &char midi_api_display_name(int api);
int midi_api_by_name(const &char name);

} // namespace std
