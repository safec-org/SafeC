// std/audio on wasm32: a module the Node harness (run.mjs) drives through
// std/audio/audio_web.js with stand-ins for AudioContext, AudioWorklet and
// Web MIDI. Each export returns >= 0 on success, a negative step number on
// failure.
#include <std/audio/audio.h>
#include <std/audio/midi.h>

struct Synth { int calls; int last_frames; unsigned int status; };
static struct Synth g_synth;
static struct Audio g_audio;
static struct AudioStream g_stream;
static struct MidiIn g_in;
static struct MidiOut g_out;

// Frame f of each period: left = f + 1, right = -(f + 1) (exact in float).
int render(&AudioIo io, ?&Synth sp) {
    match (sp) {
        case some(s): {
            s.calls = s.calls + 1;
            s.last_frames = (int)io.frames;
            s.status = s.status | io.status;
            []float o = io.output_f32();
            for (unsigned long f = 0UL; f < (unsigned long)io.frames; f = f + 1UL) {
                o[f * 2UL] = (float)(f + 1UL);
                o[f * 2UL + 1UL] = (float)0.0 - (float)(f + 1UL);
            }
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

__attribute__((export_name("test_open")))
int test_open(int frames_wanted) {
    g_synth.calls = 0;
    g_synth.status = 0U;
    struct Result<Audio, AudioError> r = std::Audio::open(AUDIO_API_UNSPECIFIED);
    if (!r.ok()) { r.drop(); return -1; }
    g_audio = r.unwrap_or(std::Audio::closed());
    if (g_audio.api() != AUDIO_API_WEBAUDIO) return -2;
    if (g_audio.device_count() != 1U) return -3;
    match (g_audio.device_info(0U).get_ok()) {
        case some(d): { if (d.output_channels != 2 || !d.supports_rate(48000U)) return -4; }
        case null: { return -4; }
    }
    struct AudioStreamParams op = std::AudioStreamParams::of(g_audio.default_output_device(), 2);
    unsigned int frames = (unsigned int)frames_wanted;
    struct Result<AudioStream, AudioError> sr =
        g_audio.open_stream((?&AudioStreamParams)&op, null, AUDIO_FORMAT_F32, 48000U, &frames, render,
                            (?&Synth)&g_synth, null);
    if (!sr.ok()) { sr.drop(); return -5; }
    g_stream = sr.unwrap_or(std::AudioStream::closed());
    if (g_stream.start().failed()) return -6;
    if (!g_stream.is_running()) return -7;
    return (int)frames;
}

static double g_in_sum = 0.0;
static int g_in_calls = 0;

int capture(&AudioIo io, ?&Synth sp) {
    g_in_calls = g_in_calls + 1;
    []const short in = io.input_i16();         // f32 from the browser, i16 here
    for (unsigned long i = 0UL; i < in.len; i = i + 1UL) g_in_sum = g_in_sum + (double)in[i];
    return AUDIO_CB_CONTINUE;
}

// An input-only stream (the page called SafeCAudio.enableInput()).
__attribute__((export_name("test_open_input")))
int test_open_input() {
    g_in_sum = 0.0;
    g_in_calls = 0;
    struct Result<Audio, AudioError> r = std::Audio::open(AUDIO_API_WEBAUDIO);
    if (!r.ok()) { r.drop(); return -1; }
    g_audio = r.unwrap_or(std::Audio::closed());
    if (g_audio.default_input_device() == AUDIO_DEFAULT_DEVICE) return -2;
    struct AudioStreamParams ip = std::AudioStreamParams::of(g_audio.default_input_device(), 1);
    unsigned int frames = 128U;
    struct Result<AudioStream, AudioError> sr =
        g_audio.open_stream(null, (?&AudioStreamParams)&ip, AUDIO_FORMAT_I16, 48000U, &frames, capture,
                            (?&Synth)&g_synth, null);
    if (!sr.ok()) { sr.drop(); return -3; }
    g_stream = sr.unwrap_or(std::AudioStream::closed());
    if (g_stream.start().failed()) return -4;
    return (int)frames;
}

__attribute__((export_name("test_in_calls")))
int test_in_calls() { return g_in_calls; }

__attribute__((export_name("test_in_sum")))
double test_in_sum() { return g_in_sum; }

__attribute__((export_name("test_calls")))
int test_calls() { return g_synth.calls; }

__attribute__((export_name("test_status")))
int test_status() { return (int)g_synth.status; }

__attribute__((export_name("test_close")))
int test_close() {
    if (g_stream.stop().failed()) return -1;
    if (g_stream.is_running()) return -2;
    g_stream.close();
    g_audio.close();
    return 0;
}

__attribute__((export_name("test_midi_open")))
int test_midi_open() {
    struct Result<MidiIn, AudioError> ri = std::MidiIn::open(MIDI_API_UNSPECIFIED, "check", 0UL);
    if (!ri.ok()) { ri.drop(); return -1; }
    g_in = ri.unwrap_or(std::MidiIn::closed());
    if (g_in.api() != MIDI_API_WEBMIDI) return -2;
    if (g_in.port_count() != 1U) return -3;
    match (g_in.port_info(0U).get_ok()) {
        case some(p): { if (!std::audio_streq_(&p.name[0], "Fake In (SafeC)")) return -4; }
        case null: { return -4; }
    }
    if (g_in.open_port(0U, "in").failed()) return -5;
    if (!g_in.open_virtual_port("v").failed()) return -6;      // a port is open already
    struct Result<MidiOut, AudioError> ro = std::MidiOut::open(MIDI_API_WEBMIDI, "check");
    if (!ro.ok()) { ro.drop(); return -7; }
    g_out = ro.unwrap_or(std::MidiOut::closed());
    if (g_out.open_virtual_port("v").code != AUDIO_ERR_UNSUPPORTED) return -8;
    if (g_out.open_port(0U, "out").failed()) return -9;
    if (g_out.send(std::MidiMsg::note_on(3, 64, 99)).failed()) return -10;
    return 0;
}

// The next queued message: (status << 16) | (data1 << 8) | data2 for a
// 3-byte message, 0 if none.
__attribute__((export_name("test_midi_poll")))
int test_midi_poll() {
    unsigned char buf[16];
    double t = 0.0;
    long n = g_in.get_message(buf[0..16], &t);
    if (n != 3L) return (int)n;
    return ((int)buf[0] << 16) | ((int)buf[1] << 8) | (int)buf[2];
}

__attribute__((export_name("test_midi_close")))
int test_midi_close() {
    g_in.close();
    g_out.close();
    return 0;
}

int main() { return 0; }
