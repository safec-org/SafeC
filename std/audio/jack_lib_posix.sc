// SafeC Standard Library — audio & MIDI: libjack, loaded at run time (see
// jack_lib.h).
#pragma once
#include <std/audio/jack_lib.h>
#include <std/audio/audio_sys.h>

namespace std {

struct JackLib_ g_std_audio_jack_;

// libjack prints "cannot connect to server" & co. itself; std reports
// through AudioError instead.
static void audio_jack_quiet_(const char* msg) { }

static fn void() audio_jack_sym_(?&AudioLib_ lib, const &char name, &int ok) {
    return audio_dlsym_or_(lib, name, ok);
}

int audio_jack_load_() {
    if (g_std_audio_jack_.loaded != 0) return g_std_audio_jack_.loaded > 0;
    g_std_audio_jack_.loaded = -1;
#if defined(__APPLE__)
    ?&AudioLib_ lib = audio_dlopen2_("libjack.0.dylib", "/usr/local/lib/libjack.0.dylib");
    if (lib.is_null()) lib = audio_dlopen_("/opt/homebrew/lib/libjack.0.dylib");
#else
    ?&AudioLib_ lib = audio_dlopen2_("libjack.so.0", "libjack.so");
#endif
    if (lib.is_null()) return 0;
    int ok = 1;
    unsafe {
        g_std_audio_jack_.client_open = (JackClientOpen_)audio_jack_sym_(lib, "jack_client_open", &ok);
        g_std_audio_jack_.client_close = (JackClient1_)audio_jack_sym_(lib, "jack_client_close", &ok);
        g_std_audio_jack_.activate = (JackClient1_)audio_jack_sym_(lib, "jack_activate", &ok);
        g_std_audio_jack_.deactivate = (JackClient1_)audio_jack_sym_(lib, "jack_deactivate", &ok);
        g_std_audio_jack_.get_sample_rate = (JackClientU_)audio_jack_sym_(lib, "jack_get_sample_rate", &ok);
        g_std_audio_jack_.get_buffer_size = (JackClientU_)audio_jack_sym_(lib, "jack_get_buffer_size", &ok);
        g_std_audio_jack_.port_register = (JackPortRegister_)audio_jack_sym_(lib, "jack_port_register", &ok);
        g_std_audio_jack_.port_unregister = (JackPortUnregister_)audio_jack_sym_(lib, "jack_port_unregister", &ok);
        g_std_audio_jack_.port_get_buffer = (JackPortGetBuffer_)audio_jack_sym_(lib, "jack_port_get_buffer", &ok);
        g_std_audio_jack_.port_name = (JackPortName_)audio_jack_sym_(lib, "jack_port_name", &ok);
        g_std_audio_jack_.set_process_callback = (JackSetProcess_)audio_jack_sym_(lib, "jack_set_process_callback", &ok);
        g_std_audio_jack_.set_xrun_callback = (JackSetXrun_)audio_jack_sym_(lib, "jack_set_xrun_callback", &ok);
        g_std_audio_jack_.on_shutdown = (JackOnShutdown_)audio_jack_sym_(lib, "jack_on_shutdown", &ok);
        g_std_audio_jack_.get_ports = (JackGetPorts_)audio_jack_sym_(lib, "jack_get_ports", &ok);
        g_std_audio_jack_.connect = (JackConnect_)audio_jack_sym_(lib, "jack_connect", &ok);
        g_std_audio_jack_.disconnect = (JackConnect_)audio_jack_sym_(lib, "jack_disconnect", &ok);
        g_std_audio_jack_.free = (JackFree_)audio_jack_sym_(lib, "jack_free", &ok);
        g_std_audio_jack_.port_get_latency_range = (JackLatencyRange_)audio_jack_sym_(lib, "jack_port_get_latency_range", &ok);
        g_std_audio_jack_.set_error_function = (JackSetMsgFn_)audio_jack_sym_(lib, "jack_set_error_function", &ok);
        g_std_audio_jack_.set_info_function = (JackSetMsgFn_)audio_jack_sym_(lib, "jack_set_info_function", &ok);
        g_std_audio_jack_.midi_get_event_count = (JackMidiCount_)audio_jack_sym_(lib, "jack_midi_get_event_count", &ok);
        g_std_audio_jack_.midi_event_get = (JackMidiGet_)audio_jack_sym_(lib, "jack_midi_event_get", &ok);
        g_std_audio_jack_.midi_clear_buffer = (JackMidiClear_)audio_jack_sym_(lib, "jack_midi_clear_buffer", &ok);
        g_std_audio_jack_.midi_event_write = (JackMidiWrite_)audio_jack_sym_(lib, "jack_midi_event_write", &ok);
        g_std_audio_jack_.frame_time = (JackFrameTime_)audio_jack_sym_(lib, "jack_frame_time", &ok);
        g_std_audio_jack_.get_client_name = (JackClientName_)audio_jack_sym_(lib, "jack_get_client_name", &ok);
        if (!ok) return 0;
        g_std_audio_jack_.set_error_function(audio_jack_quiet_);
        g_std_audio_jack_.set_info_function(audio_jack_quiet_);
    }
    g_std_audio_jack_.loaded = 1;
    return 1;
}

JackClient_* audio_jack_client_open_(const &char name) {
    if (!audio_jack_load_()) return (JackClient_*)0;
    int status = 0;
    unsafe { return g_std_audio_jack_.client_open(name, AUDIO_JACK_NO_START_SERVER, &status); }
}

} // namespace std
