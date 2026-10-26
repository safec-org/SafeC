// SafeC Standard Library — audio & MIDI: libjack, loaded at run time.
//
// std-private: the JACK functions std/audio's JACK backends (audio_jack_posix.sc
// for audio, midi_jack_posix.sc for MIDI) call, looked up with dlsym() from
// libjack.so.0 (libjack.0.dylib on macOS) so builds don't link libjack and
// run without it. Function signatures follow <jack/jack.h> and
// <jack/midiport.h>; jack_nframes_t is uint32, size_t is pointer width.
#pragma once
#include <std/stddef.h>
#include <std/erased.h>

#define AUDIO_JACK_AUDIO_TYPE "32 bit float mono audio"
#define AUDIO_JACK_MIDI_TYPE  "8 bit raw midi"
#define AUDIO_JACK_IS_INPUT  1UL
#define AUDIO_JACK_IS_OUTPUT 2UL
#define AUDIO_JACK_NO_START_SERVER 1

namespace std {

// jack_midi_event_t
struct JackMidiEvent_ {
    unsigned int time;
    size_t size;
    unsigned char* buffer;
};

// jack_client_t, jack_port_t and a MIDI port buffer (opaque).
struct JackClient_  { char opaque_; };
struct JackPort_    { char opaque_; };
struct JackMidiBuf_ { char opaque_; };

// The callbacks a client registers, with the 'arg' it registered them with
// (each backend's stream state, erased: see std/erased.h).
typedef fn int(unsigned int, ?&Erased) JackProcessFn_;   // JackProcessCallback
typedef fn int(?&Erased) JackXrunFn_;                    // JackXRunCallback
typedef fn void(?&Erased) JackShutdownFn_;               // JackShutdownCallback
typedef fn void(const char*) JackMsgFn_;                 // error/info message sink

typedef fn JackClient_*(const char*, int, int*, ...) JackClientOpen_;
typedef fn int(JackClient_*) JackClient1_;                          // close, activate, deactivate
typedef fn unsigned int(JackClient_*) JackClientU_;                 // get_sample_rate, get_buffer_size
typedef fn JackPort_*(JackClient_*, const char*, const char*, unsigned long, unsigned long) JackPortRegister_;
typedef fn int(JackClient_*, JackPort_*) JackPortUnregister_;
// jack_port_get_buffer: f32 samples for an audio port, a JackMidiBuf_ for a MIDI one.
typedef fn unsigned char*(JackPort_*, unsigned int) JackPortGetBuffer_;
typedef fn const char*(const JackPort_*) JackPortName_;
typedef fn int(JackClient_*, JackProcessFn_, ?&Erased) JackSetProcess_;
typedef fn int(JackClient_*, JackXrunFn_, ?&Erased) JackSetXrun_;
typedef fn void(JackClient_*, JackShutdownFn_, ?&Erased) JackOnShutdown_;
typedef fn const char**(JackClient_*, const char*, const char*, unsigned long) JackGetPorts_;
typedef fn int(JackClient_*, const char*, const char*) JackConnect_;
typedef fn void(const char**) JackFree_;                            // jack_free of a jack_get_ports list
typedef fn void(JackPort_*, int, unsigned int*) JackLatencyRange_;
typedef fn void(JackMsgFn_) JackSetMsgFn_;
typedef fn unsigned int(JackMidiBuf_*) JackMidiCount_;
typedef fn int(&JackMidiEvent_, JackMidiBuf_*, unsigned int) JackMidiGet_;
typedef fn void(JackMidiBuf_*) JackMidiClear_;
typedef fn int(JackMidiBuf_*, unsigned int, const unsigned char*, size_t) JackMidiWrite_;
typedef fn unsigned int(const JackClient_*) JackFrameTime_;         // jack_last_frame_time, jack_frame_time
typedef fn const char*(JackClient_*) JackClientName_;

struct JackLib_ {
    int loaded;      // 0 not tried, 1 ok, -1 unavailable
    JackClientOpen_ client_open;
    JackClient1_ client_close;
    JackClient1_ activate;
    JackClient1_ deactivate;
    JackClientU_ get_sample_rate;
    JackClientU_ get_buffer_size;
    JackPortRegister_ port_register;
    JackPortUnregister_ port_unregister;
    JackPortGetBuffer_ port_get_buffer;
    JackPortName_ port_name;
    JackSetProcess_ set_process_callback;
    JackSetXrun_ set_xrun_callback;
    JackOnShutdown_ on_shutdown;
    JackGetPorts_ get_ports;
    JackConnect_ connect;
    JackConnect_ disconnect;
    JackFree_ free;
    JackLatencyRange_ port_get_latency_range;
    JackSetMsgFn_ set_error_function;
    JackSetMsgFn_ set_info_function;
    JackMidiCount_ midi_get_event_count;
    JackMidiGet_ midi_event_get;
    JackMidiClear_ midi_clear_buffer;
    JackMidiWrite_ midi_event_write;
    JackFrameTime_ frame_time;
    JackClientName_ get_client_name;
};

// The loaded library (fields valid after audio_jack_load_() returned 1).
extern struct JackLib_ g_std_audio_jack_;

// Loads libjack once; 1 when it and every function above are there.
int audio_jack_load_();
// jack_client_open(name, JackNoStartServer): a client, or null when the
// server isn't running (or refused the name).
JackClient_* audio_jack_client_open_(const &char name);

} // namespace std
