// SafeC Standard Library — real-time audio I/O (in the spirit of RtAudio).
//
// One API over every platform's native audio system:
//
//   API                  platform        backend file
//   AUDIO_API_ALSA       Linux           audio_alsa_linux.sc      (libasound, loaded at run time)
//   AUDIO_API_JACK       Linux, macOS    audio_jack_posix.sc      (libjack, loaded at run time)
//   AUDIO_API_COREAUDIO  macOS           audio_coreaudio_macos.sc (Core Audio HAL: AudioObject + IOProc)
//   AUDIO_API_WASAPI     Windows         audio_wasapi_win32.sc    (shared + exclusive, event-driven)
//   AUDIO_API_WEBAUDIO   wasm32          audio_webaudio_wasm.sc   (+ the JS glue std/audio/audio_web.js)
//   AUDIO_API_DUMMY      everywhere      audio_dummy.sc           (no device: a timer thread, for tests/CI)
//
// safeguard builds the backends for the target and links what they need;
// ALSA and JACK are dlopen()ed, so a program runs (and probes zero devices)
// on a machine without them. audio_compiled_apis() lists what a build has.
//
//   struct Result<Audio, AudioError> r = std::Audio::open(AUDIO_API_UNSPECIFIED);
//   struct Audio a = r.unwrap_or(std::Audio::closed());
//   for (unsigned int i = 0U; i < a.device_count(); i = i + 1U) { ... a.device_info(i) ... }
//
//   struct AudioStreamParams out = std::AudioStreamParams::of(a.default_output_device(), 2);
//   unsigned int frames = 256U;
//   struct Result<AudioStream, AudioError> s =
//       a.open_stream((?&AudioStreamParams)&out, null, AUDIO_FORMAT_F32, 48000U, &frames,
//                     render, (?&Synth)&synth, null);
//   ... st.start(); ... st.stop(); st.close(); a.close();
//
// with the callback
//
//   int render(&AudioIo io, ?&Synth s) {
//       []float out = io.output_f32();                 // frames * channels, interleaved
//       ... fill it ...
//       return AUDIO_CB_CONTINUE;                      // or AUDIO_CB_STOP / AUDIO_CB_ABORT
//   }
//
// ── Real-time discipline ─────────────────────────────────────────────────
// The callback runs on the backend's real-time thread (an ALSA/WASAPI/dummy
// thread std starts, the JACK process thread, the Core Audio IOProc
// thread, the browser's audio rendering). Everything std does on that path
// is allocation-free and lock-free: every buffer (format conversion,
// interleaving, channel offset) is allocated by open_stream(), the stream's
// state is read and written with atomics only, and the status flags
// (AUDIO_STATUS_*) and stream time are plain atomic words. Keep your
// callback the same: no malloc/new, no locks, no blocking I/O, no printf —
// hand data to other threads through a lock-free queue (std/dsp/
// audio_buffer.h's AudioBuffer, std/sync's ring buffers). start()/stop()/
// close() are control-thread calls; they may block (joining the stream
// thread, draining the device) and must not be called from the callback —
// return AUDIO_CB_STOP or AUDIO_CB_ABORT from it instead.
//
// ── Buffers ──────────────────────────────────────────────────────────────
// The callback sees buffers in the format and layout the stream was opened
// with, whatever the device uses natively: std converts sample formats
// (i16/i24/i32/f32/f64), (de)interleaves and applies the channel offsets
// on its side. Interleaved (the default): frame-major, 'channels' samples per
// frame. With AUDIO_FLAG_NONINTERLEAVED: channel-major, 'frames' samples per
// channel. AUDIO_FORMAT_I24 is packed 3-byte little-endian samples: S24
// (std/audio/s24.h), reached through output_i24()/input_i24().
#pragma once
#include <std/audio/audio_error.h>
#include <std/audio/s24.h>
#include <std/erased.h>
#include <std/result.h>

// ── APIs ─────────────────────────────────────────────────────────────────
#define AUDIO_API_UNSPECIFIED 0   // open(): the first compiled API that works
#define AUDIO_API_ALSA        1
#define AUDIO_API_JACK        2
#define AUDIO_API_COREAUDIO   3
#define AUDIO_API_WASAPI      4
#define AUDIO_API_WEBAUDIO    5
#define AUDIO_API_DUMMY       6
#define AUDIO_API_COUNT       7

// ── Sample formats (bit flags: AudioDeviceInfo.native_formats is a set) ──
#define AUDIO_FORMAT_I8   0x01U   // signed 8-bit
#define AUDIO_FORMAT_I16  0x02U   // signed 16-bit
#define AUDIO_FORMAT_I24  0x04U   // signed 24-bit, packed 3 bytes, little-endian
#define AUDIO_FORMAT_I32  0x08U   // signed 32-bit
#define AUDIO_FORMAT_F32  0x10U   // float, nominal range [-1, 1]
#define AUDIO_FORMAT_F64  0x20U   // double, nominal range [-1, 1]

// ── Stream flags (AudioStreamOptions.flags) ──────────────────────────────
#define AUDIO_FLAG_NONINTERLEAVED   0x01U  // callback buffers channel-major
#define AUDIO_FLAG_MINIMIZE_LATENCY 0x02U  // smallest buffer the device allows
#define AUDIO_FLAG_EXCLUSIVE        0x04U  // WASAPI exclusive / Core Audio hog mode / ALSA hw: device
#define AUDIO_FLAG_SCHEDULE_REALTIME 0x08U // raise the stream thread to real-time priority
#define AUDIO_FLAG_ALSA_USE_DEFAULT 0x10U  // ALSA: open the "default" PCM for the device
#define AUDIO_FLAG_JACK_DONT_CONNECT 0x20U // JACK: register ports, don't connect them
#define AUDIO_FLAG_DUMMY_MANUAL     0x40U  // dummy: no timer thread; drive it with AudioStream::pump()

// ── Callback status (AudioIo.status) ─────────────────────────────────────
#define AUDIO_STATUS_INPUT_OVERFLOW   0x1U // input data was lost since the last callback
#define AUDIO_STATUS_OUTPUT_UNDERFLOW 0x2U // the device ran dry since the last callback

// ── Callback return values ───────────────────────────────────────────────
#define AUDIO_CB_CONTINUE 0   // keep running
#define AUDIO_CB_STOP     1   // play this buffer and what's queued, then stop
#define AUDIO_CB_ABORT    2   // stop now, dropping what's queued

// ── Stream states ────────────────────────────────────────────────────────
#define AUDIO_STREAM_CLOSED  0
#define AUDIO_STREAM_STOPPED 1
#define AUDIO_STREAM_RUNNING 2

// open_stream() 'device' meaning "the API's default device" for that direction.
#define AUDIO_DEFAULT_DEVICE 0U

#define AUDIO_DEVICE_NAME_MAX 128
#define AUDIO_MAX_SAMPLE_RATES 16
#define AUDIO_STREAM_NAME_MAX 64

namespace std {

// One audio device, as a probe saw it.
struct AudioDeviceInfo {
    unsigned int id;                       // what open_stream() takes (never 0)
    char         name[AUDIO_DEVICE_NAME_MAX];
    int          output_channels;          // max playback channels
    int          input_channels;           // max capture channels
    int          duplex_channels;          // max channels usable in both directions at once
    int          is_default_output;
    int          is_default_input;
    unsigned int native_formats;           // AUDIO_FORMAT_* the device takes without conversion
    unsigned int sample_rates[AUDIO_MAX_SAMPLE_RATES]; // supported rates, ascending
    int          sample_rate_count;
    unsigned int preferred_sample_rate;    // the device's current/natural rate

    // Whether 'rate' is in sample_rates.
    int supports_rate(unsigned int rate) const;
    // An empty record (id 0, no channels), e.g. for Result::unwrap_or().
    static struct AudioDeviceInfo none();
};

// One direction of a stream.
struct AudioStreamParams {
    unsigned int device;        // AudioDeviceInfo.id, or AUDIO_DEFAULT_DEVICE
    int          channels;      // >= 1
    int          first_channel; // offset into the device's channels (0 = first)

    static struct AudioStreamParams of(unsigned int device, int channels);
};

// Everything else a stream can be asked for. Zero-initialized (none())
// means: interleaved, the API's default buffering, normal priority.
struct AudioStreamOptions {
    unsigned int flags;         // AUDIO_FLAG_*
    unsigned int buffer_count;  // periods/buffers the device queues (0 = API default)
    int          priority;      // with AUDIO_FLAG_SCHEDULE_REALTIME: 1..99 (0 = API default)
    char         stream_name[AUDIO_STREAM_NAME_MAX]; // JACK client name, WebAudio label
    // An API-specific device to open instead of the 'device' ids: an ALSA
    // PCM name ("hw:1,0", "null", a ~/.asoundrc plugin), or for JACK the
    // client whose ports to connect to ("system"). "" = use the ids.
    char         device_name[AUDIO_DEVICE_NAME_MAX];

    static struct AudioStreamOptions none();
    static struct AudioStreamOptions with_flags(unsigned int flags);
};

// What the callback gets: the buffers for this period plus its context.
struct AudioIo {
    []unsigned char output_;    // the user-format output buffer's bytes (empty for input-only)
    []unsigned char input_;     // the user-format input buffer's bytes (empty for output-only)
    unsigned int frames;        // frames in this period
    double       stream_time;   // seconds of audio processed before this period
    unsigned int status;        // AUDIO_STATUS_* since the last callback
    int          output_channels;
    int          input_channels;
    unsigned int format;        // AUDIO_FORMAT_*
    int          interleaved;   // 0 with AUDIO_FLAG_NONINTERLEAVED
    unsigned int sample_rate;

    // The buffers, typed: empty when the stream's format is another one.
    // frames * channels samples; interleaved or channel-major as opened.
    // The input buffers are read-only.
    []float                 output_f32() const;
    []const float           input_f32() const;
    []double                output_f64() const;
    []const double          input_f64() const;
    []short                 output_i16() const;
    []const short           input_i16() const;
    []int                   output_i32() const;
    []const int             input_i32() const;
    []signed char           output_i8() const;
    []const signed char     input_i8() const;
    []S24                   output_i24() const;
    []const S24             input_i24() const;
    // Raw bytes, any format.
    []unsigned char         output_bytes() const;
    []const unsigned char   input_bytes() const;
    // One channel of a non-interleaved f32 stream ('frames' samples);
    // empty when interleaved, out of range or another format.
    []float                 output_channel_f32(int channel) const;
    []const float           input_channel_f32(int channel) const;
    // Zeroes the output buffer.
    void                    clear_output() const;
};

// The callback, erased (see std/erased.h): return AUDIO_CB_CONTINUE / _STOP / _ABORT.
typedef fn int(&AudioIo, ?&Erased) AudioCallbackRaw_;

// An open stream: a handle onto state open_stream() allocated. Copies refer
// to the same stream; close() it once.
struct AudioStream {
    ?&struct AudioStreamState_ impl;    // std-private; null when closed

    // A closed (never-opened) stream.
    static struct AudioStream closed();

    struct AudioError start();
    // Plays out what's queued, then stops (the device drains).
    struct AudioError stop();
    // Stops now, dropping what's queued.
    struct AudioError abort();
    // Stops (abort) if running and releases everything. Idempotent.
    void drop();
    void close() consume;   // = drop()

    int  is_open() const;
    // 1 while started — also 0 after the callback returned STOP/ABORT or
    // the device failed (see last_error()).
    int  is_running() const;
    int  state() const;                       // AUDIO_STREAM_*
    // Seconds of audio processed since open (frames / sample rate).
    double stream_time() const;
    void   set_stream_time(double seconds);
    // Total latency in frames (input + output) the API reports; 0 if unknown.
    long   latency() const;
    long   output_latency() const;
    long   input_latency() const;
    unsigned int sample_rate() const;         // the rate actually running
    unsigned int buffer_frames() const;       // frames per callback
    int    api() const;
    // The error that stopped the stream from its own thread (device lost,
    // xrun recovery failed), or none().
    struct AudioError last_error() const;

    // AUDIO_FLAG_DUMMY_MANUAL streams only: runs one period on the calling
    // thread — the deterministic way to drive the dummy API in tests (and
    // the only one on targets without threads). Returns the callback's
    // value, or -1 when the stream isn't a manual dummy stream or isn't running.
    int pump();
};

// A connection to one audio API: its device list and the streams it opens.
struct Audio {
    int  api_id;    // AUDIO_API_*, 0 when closed
    ?&struct AudioHost_ impl;    // std-private; null when closed

    // Opens 'api' (AUDIO_API_UNSPECIFIED: audio_default_api(), then the
    // other compiled APIs, then dummy) and probes its devices. A machine
    // with no sound hardware still opens: device_count() is 0.
    static struct Result<Audio, AudioError> open(int api);
    static struct Audio closed();

    int  is_open() const;
    int  api() const;
    // Re-enumerates the devices (after a hot-plug).
    struct AudioError rescan();
    unsigned int device_count() const;
    // The 'index'-th device (0 <= index < device_count()).
    struct Result<AudioDeviceInfo, AudioError> device_info(unsigned int index) const;
    // The device with this id.
    struct Result<AudioDeviceInfo, AudioError> device_by_id(unsigned int id) const;
    // Ids of the default devices, AUDIO_DEFAULT_DEVICE if none.
    unsigned int default_output_device() const;
    unsigned int default_input_device() const;

    // Opens a stream: output, input or both (duplex; give both params).
    // 'frames' is the requested frames per callback (0 = the API's default)
    // and receives what the device actually uses. 'opts' may be null. The
    // callback gets 'user' back as its second argument. The callback runs on
    // the audio thread while 'user' stays reachable from the caller's, so T
    // must be Send + Sync (plain data, atomics, a Mutex; no raw pointers,
    // stack references or !Send handles such as GUI windows); writes shared
    // with other threads still need atomics or a lock.
    generic<T: Send + Sync>
    struct Result<AudioStream, AudioError> open_stream(?&AudioStreamParams output, ?&AudioStreamParams input,
                                                       unsigned int format, unsigned int sample_rate,
                                                       &unsigned int frames,
                                                       fn int(&AudioIo, ?&T) callback, ?&T user,
                                                       ?&AudioStreamOptions opts);
    struct Result<AudioStream, AudioError> open_stream_raw_(?&AudioStreamParams output, ?&AudioStreamParams input,
                                                            unsigned int format, unsigned int sample_rate,
                                                            &unsigned int frames, AudioCallbackRaw_ callback,
                                                            ?&Erased user, ?&AudioStreamOptions opts);

    // Releases the device list. Streams opened from it stay valid.
    void drop();
    void close() consume;   // = drop()
};

generic<T: Send + Sync>
struct Result<AudioStream, AudioError> Audio::open_stream(?&AudioStreamParams output, ?&AudioStreamParams input,
                                                          unsigned int format, unsigned int sample_rate,
                                                          &unsigned int frames,
                                                          fn int(&AudioIo, ?&T) callback, ?&T user,
                                                          ?&AudioStreamOptions opts) {
    unsafe {
        return self.open_stream_raw_(output, input, format, sample_rate, frames,
                                     (AudioCallbackRaw_)callback, (?&Erased)user, opts);
    }
}

// ── API helpers ──────────────────────────────────────────────────────────

// The APIs compiled into this build, in preference order; returns how many
// were written to 'out' (up to out.len).
int audio_compiled_apis([]int out);
int audio_api_compiled(int api);
// The platform's usual API: Core Audio, WASAPI, WebAudio, ALSA (JACK when
// its server is running and SAFEC_AUDIO_API isn't set otherwise); the
// SAFEC_AUDIO_API environment variable ("alsa", "jack", "dummy", ...) overrides.
int audio_default_api();
// Short lowercase name ("alsa") / display name ("ALSA"); "" for unknown.
const &char audio_api_name(int api);
const &char audio_api_display_name(int api);
// AUDIO_API_* for a short name, AUDIO_API_UNSPECIFIED if none matches.
int audio_api_by_name(const &char name);

// Bytes per sample of an AUDIO_FORMAT_* (0 for an unknown/multi-bit value).
int audio_format_bytes(unsigned int format);
const &char audio_format_name(unsigned int format);

} // namespace std
