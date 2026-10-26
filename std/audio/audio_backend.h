// SafeC Standard Library — audio: the interface between audio.sc and the
// per-API backends (audio_<api>[_<os>].sc). std-private.
//
// Each backend is one AudioBackendVt_ (a table of functions) audio.sc picks
// by API id. A backend fills an AudioHost_'s device list in probe(), and
// opens/starts/stops an AudioStreamState_ that audio.sc allocated and filled
// with the request. On its real-time path it only calls
// audio_stream_process_() (convert + callback + convert, allocation- and
// lock-free) and the atomic flag helpers below.
#pragma once
#include <std/audio/audio.h>

#define AUDIO_DIR_OUT 0
#define AUDIO_DIR_IN  1
#define AUDIO_DEVICE_KEY_MAX 256

namespace std {

// A device a probe found: the public info plus what the backend needs to
// open it again (an ALSA PCM name, a JACK client name, a Core Audio
// AudioDeviceID, a WASAPI endpoint id).
struct AudioDeviceRec_ {
    struct AudioDeviceInfo info;
    char      key[AUDIO_DEVICE_KEY_MAX];
    long long native;
};

// A device key a probe has seen and the id it got (ids stay stable across
// rescans: a device that comes back gets its old id).
struct AudioKnownId_ {
    char key[AUDIO_DEVICE_KEY_MAX];
    unsigned int id;
};

struct AudioHost_ {
    int api;
    struct AudioDeviceRec_* devs;
    unsigned int count;
    unsigned int cap;
    struct AudioKnownId_* known;
    unsigned int known_count;
    unsigned int known_cap;
    unsigned int next_id;        // ids handed out so far
    unsigned int default_out;    // device ids, AUDIO_DEFAULT_DEVICE if none
    unsigned int default_in;
    void* backend;               // backend-private, released by host_close
    struct AudioError error;
};

// One stream. Fields marked (req) are the request, set by audio.sc before
// the backend's open(); (be) are the backend's to set in open(); (rt) are
// touched on the real-time path, atomically.
struct AudioStreamState_ {
    int api;
    void* vt;                         // the AudioBackendVt_
    int mode_out;                     // (req) output direction requested
    int mode_in;                      // (req) input direction requested
    unsigned int user_format;         // (req) AUDIO_FORMAT_*
    int user_interleaved;             // (req)
    int user_channels[2];             // (req)
    int first_channel[2];             // (req) device channel offset
    unsigned int device_id[2];        // (req) resolved ids (0 if device_name used)
    char device_key[2][AUDIO_DEVICE_KEY_MAX]; // (req) the device records' keys
    long long device_native[2];       // (req)
    unsigned int sample_rate;         // (req) requested -> (be) actual
    unsigned int buffer_frames;       // (req) requested (0 = default) -> (be) actual
    unsigned int max_frames;          // (be) the largest period process() may get (>= buffer_frames)
    unsigned int buffer_count;        // (req) -> (be)
    unsigned int flags;               // (req) AUDIO_FLAG_*
    int priority;                     // (req)
    char stream_name[AUDIO_STREAM_NAME_MAX];   // (req)
    char device_name[AUDIO_DEVICE_NAME_MAX];   // (req) options.device_name

    unsigned int device_format[2];    // (be) the device's sample format
    int device_channels[2];           // (be) channels the device was opened with
    int device_interleaved[2];        // (be)
    long latency[2];                  // (be) frames

    int convert[2];                   // set after open: user != device layout
    void* user_buf[2];                // what the callback sees
    void* device_buf[2];              // what the backend reads/writes (== user_buf if !convert)

    AudioCallbackRaw_ callback;       // the user's fn int(&AudioIo, ?&T), erased
    ?&Erased user;

    int state;                        // (rt) AUDIO_STREAM_*
    int stop_request;                 // (rt) 0, AUDIO_CB_STOP or AUDIO_CB_ABORT (from control or callback)
    unsigned int pending_status;      // (rt) AUDIO_STATUS_* the backend observed
    long long frames_done;            // (rt) frames processed
    long long frames_offset;          // set_stream_time()
    int rt_error_set;                 // (rt) 1 once rt_error holds an error
    struct AudioError rt_error;       // written once by the stream thread before rt_error_set
    unsigned long long thread;        // a stream thread std started (ALSA, WASAPI, dummy)
    int thread_live;                  // it hasn't been joined yet
    void* thread_body;                // fn void(&AudioStreamState_) it runs
    void* backend;                    // backend-private
    struct AudioError error;          // control-thread errors (open/start/stop)
};

// What a backend implements. Each returns AUDIO_OK or an AUDIO_ERR_* code,
// with the message in host.error / st.error.
struct AudioBackendVt_ {
    int api;
    fn int() available;                                   // can this API work here at all (lib/server present)?
    fn int(&AudioHost_) probe;                            // (re)fill the device list
    fn void(&AudioHost_) host_close;
    fn int(&AudioHost_, &AudioStreamState_) open;         // open the device(s); set the (be) fields
    fn int(&AudioStreamState_) start;
    fn int(&AudioStreamState_) stop;                      // drain, then stop
    fn int(&AudioStreamState_) abort;                     // stop now
    fn void(&AudioStreamState_) close;                    // release (stream is stopped)
};

// The backends compiled into this build (defined by each backend file).
&static AudioBackendVt_ audio_dummy_backend_();
#if defined(__linux__)
&static AudioBackendVt_ audio_alsa_backend_();
#endif
#if defined(__linux__) || defined(__APPLE__)
&static AudioBackendVt_ audio_jack_backend_();
#endif
#if defined(__APPLE__)
&static AudioBackendVt_ audio_coreaudio_backend_();
#endif
#if defined(_WIN32)
&static AudioBackendVt_ audio_wasapi_backend_();
#endif
#if defined(__wasm__)
&static AudioBackendVt_ audio_webaudio_backend_();
#endif

// ── for backends' probe() ────────────────────────────────────────────────
// Empties the list (ids are kept per key for the next adds).
void audio_host_clear_(&AudioHost_ h);
// Appends a device (the caller filled info except id); returns its record,
// null on OOM. A key seen in an earlier probe gets its old id back.
?&AudioDeviceRec_ audio_host_add_(&AudioHost_ h, &AudioDeviceInfo info, const &char key, long long native);
// A blank AudioDeviceInfo (no channels, no rates).
struct AudioDeviceInfo audio_device_info_blank_();
// Adds 'rate' to info's rate list (kept ascending, no duplicates).
void audio_device_add_rate_(&AudioDeviceInfo info, unsigned int rate);
// The standard rates std probes: 8000 .. 192000.
int audio_standard_rate_count_();
unsigned int audio_standard_rate_(int i);

// ── for backends' real-time path ─────────────────────────────────────────
// Runs one period of 'frames' (<= max_frames): device_buf[IN] -> user_buf[IN]
// conversion, the callback, user_buf[OUT] -> device_buf[OUT]. Returns the
// callback's AUDIO_CB_* (a stop/abort is also latched into stop_request).
int audio_stream_process_(&AudioStreamState_ st, unsigned int frames);
// Zeroes device_buf[OUT] for 'frames' frames (an underrun's silence).
void audio_stream_silence_(&AudioStreamState_ st, unsigned int frames);
// ORs AUDIO_STATUS_* into the next callback's status.
void audio_stream_flag_status_(&AudioStreamState_ st, unsigned int status);
// Latest stop request (0, AUDIO_CB_STOP or AUDIO_CB_ABORT).
int  audio_stream_stop_requested_(&AudioStreamState_ st);
void audio_stream_request_stop_(&AudioStreamState_ st, int how);
// The stream ended on its own (callback stop, device error): state ->
// STOPPED. 'err' may be null.
void audio_stream_mark_stopped_(&AudioStreamState_ st, ?&AudioError err);
// Records an error raised on the stream thread (first one wins).
void audio_stream_set_rt_error_(&AudioStreamState_ st, int code, const &char message);
// Bytes per frame of device_buf[dir].
unsigned long audio_stream_device_frame_bytes_(&AudioStreamState_ st, int dir);

// ── stream threads (ALSA, WASAPI, dummy) ─────────────────────────────────
// Starts body(st) on a new thread (made real-time when the stream asked);
// joins a previous, finished one first.
int  audio_stream_thread_start_(&AudioStreamState_ st, fn void(&AudioStreamState_) body);
// Joins the stream thread if one is live. Never from the stream thread.
void audio_stream_thread_join_(&AudioStreamState_ st);

} // namespace std
