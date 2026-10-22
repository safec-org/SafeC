// SafeC Standard Library — audio & MIDI: libjack, loaded at run time.
//
// std-private: the JACK functions std/audio's JACK backends (audio_jack_posix.sc
// for audio, midi_jack_posix.sc for MIDI) call, looked up with dlsym() from
// libjack.so.0 (libjack.0.dylib on macOS) so builds don't link libjack and
// run without it. Function signatures follow <jack/jack.h> and
// <jack/midiport.h>; jack_nframes_t is uint32, size_t is pointer width.
#pragma once
#include <std/stddef.h>

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

typedef fn void*(const char*, int, int*, ...) JackClientOpen_;
typedef fn int(void*) JackClient1_;                                 // close, activate, deactivate
typedef fn unsigned int(void*) JackClientU_;                        // get_sample_rate, get_buffer_size
typedef fn void*(void*, const char*, const char*, unsigned long, unsigned long) JackPortRegister_;
typedef fn int(void*, void*) JackPortUnregister_;
typedef fn void*(void*, unsigned int) JackPortGetBuffer_;
typedef fn const char*(const void*) JackPortName_;
typedef fn int(void*, void*, void*) JackSetCallback_;               // set_process_callback, set_xrun_callback
typedef fn void(void*, void*, void*) JackOnShutdown_;
typedef fn const char**(void*, const char*, const char*, unsigned long) JackGetPorts_;
typedef fn int(void*, const char*, const char*) JackConnect_;
typedef fn void(void*) JackFree_;
typedef fn void(void*, int, unsigned int*) JackLatencyRange_;
typedef fn void(void*) JackSetMsgFn_;
typedef fn unsigned int(void*) JackMidiCount_;
typedef fn int(&JackMidiEvent_, void*, unsigned int) JackMidiGet_;
typedef fn void(void*) JackMidiClear_;
typedef fn int(void*, unsigned int, const unsigned char*, size_t) JackMidiWrite_;
typedef fn unsigned int(const void*) JackFrameTime_;                // jack_last_frame_time, jack_frame_time
typedef fn const char*(void*) JackClientName_;

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
    JackSetCallback_ set_process_callback;
    JackSetCallback_ set_xrun_callback;
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
void* audio_jack_client_open_(const &char name);

} // namespace std
