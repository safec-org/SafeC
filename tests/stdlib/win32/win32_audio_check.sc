// std/audio on Windows: WASAPI and WinMM through their real APIs (run
// under Wine by run.sh, or natively). Probes both, opens every WASAPI
// render endpoint it finds for a few callbacks, sends through every WinMM
// output port, and checks the dummy APIs work alongside.
#include <std/audio/audio.h>
#include <std/audio/audio_sys.h>
#include <std/audio/midi.h>

extern int printf(const char* fmt, ...);

static int g_fails = 0;
static void check(int ok, const &char what) {
    unsafe { printf("  %s  %s\n", ok ? "PASS" : "FAIL", what); }
    if (!ok) g_fails = g_fails + 1;
}

struct Count { int calls; int in_frames; };
int count_cb(&AudioIo io, ?&Count cp) {
    match (cp) {
        case some(c): {
            c.calls = c.calls + 1;
            if (io.input_channels > 0) c.in_frames = c.in_frames + (int)(io.input_bytes().len / (unsigned long)(audio_format_bytes(io.format) * io.input_channels));
            struct Slice<float> o = io.output_f32();
            for (unsigned long i = 0UL; i < o.len; i = i + 1UL) o.set(i, (float)0.0);
            if (c.calls >= 8) return AUDIO_CB_STOP;
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

int main() {
    check(audio_api_compiled(AUDIO_API_WASAPI) && !audio_api_compiled(AUDIO_API_ALSA), "WASAPI compiled in, not ALSA");
    check(audio_default_api() == AUDIO_API_WASAPI, "WASAPI is the default API");
    struct Result<Audio, AudioError> r = std::Audio::open(AUDIO_API_WASAPI);
    match (r.get_err()) { case some(e): printf("    wasapi: %s\n", e.message); case null: {} }
    check(r.ok(), "WASAPI probes (CoCreateInstance, EnumAudioEndpoints, Activate, GetMixFormat)");
    struct Audio a = r.unwrap_or(std::Audio::closed());
    printf("    %u WASAPI endpoint(s)\n", a.device_count());
    for (unsigned int i = 0U; i < a.device_count(); i = i + 1U) {
        struct AudioDeviceInfo d = a.device_info(i).unwrap_or(std::AudioDeviceInfo::none());
        printf("    [%u] %s: %d out, %d in, %u Hz, formats 0x%x%s%s\n", d.id, d.name, d.output_channels,
               d.input_channels, d.preferred_sample_rate, d.native_formats,
               d.is_default_output ? " (default out)" : "", d.is_default_input ? " (default in)" : "");
        if (d.output_channels < 1) continue;
        for (int mode = 0; mode < 2; mode = mode + 1) {
            struct Count c; c.calls = 0; c.in_frames = 0;
            struct AudioStreamParams op = std::AudioStreamParams::of(d.id, 1);
            struct AudioStreamOptions o = std::AudioStreamOptions::with_flags(mode == 1 ? AUDIO_FLAG_EXCLUSIVE : 0U);
            unsigned int frames = 0U;
            struct Result<AudioStream, AudioError> sr =
                a.open_stream((?&AudioStreamParams)&op, null, AUDIO_FORMAT_F32, d.preferred_sample_rate, &frames, count_cb,
                              (?&Count)&c, (?&AudioStreamOptions)&o);
            match (sr.get_err()) { case some(e): printf("    %s: %s\n", mode == 1 ? "exclusive" : "shared", e.message); case null: {} }
            if (!sr.ok()) { check(mode == 1, "shared-mode stream opens"); continue; }
            struct AudioStream s = sr.unwrap_or(std::AudioStream::closed());
            printf("    %s: %u frames, latency %ld\n", mode == 1 ? "exclusive" : "shared", frames, s.latency());
            struct AudioError e = s.start();
            check(e.ok(), "stream starts");
            long long t0 = std::audio_now_ns_();
            while (s.is_running() && std::audio_now_ns_() - t0 < 5000000000LL) std::audio_sleep_us_(2000LL);
            printf("    %d callbacks, last error '%s'\n", c.calls, s.last_error().message);
            check(c.calls == 8, "callbacks ran until AUDIO_CB_STOP");
            s.close();
        }
    }
    // Input-only and duplex on the default endpoints.
    unsigned int defIn = a.default_input_device();
    unsigned int defOut = a.default_output_device();
    if (defIn != AUDIO_DEFAULT_DEVICE) {
        for (int duplex = 0; duplex < 2; duplex = duplex + 1) {
            if (duplex && defOut == AUDIO_DEFAULT_DEVICE) continue;
            struct Count c; c.calls = 0; c.in_frames = 0;
            struct AudioStreamParams ip = std::AudioStreamParams::of(defIn, 1);
            struct AudioStreamParams op = std::AudioStreamParams::of(defOut, 1);
            unsigned int frames = 256U;
            struct AudioDeviceInfo di = a.device_by_id(defIn).unwrap_or(std::AudioDeviceInfo::none());
            ?&AudioStreamParams outp = null;
            if (duplex) outp = (?&AudioStreamParams)&op;
            struct Result<AudioStream, AudioError> sr =
                a.open_stream(outp, (?&AudioStreamParams)&ip, AUDIO_FORMAT_I16,
                              di.preferred_sample_rate, &frames, count_cb, (?&Count)&c, null);
            match (sr.get_err()) { case some(e): printf("    %s: %s\n", duplex ? "duplex" : "input", e.message); case null: {} }
            check(sr.ok(), duplex ? "duplex stream opens (render + capture endpoints)" : "input stream opens");
            if (!sr.ok()) continue;
            struct AudioStream s = sr.unwrap_or(std::AudioStream::closed());
            check(s.start().ok(), "stream starts");
            long long t0 = std::audio_now_ns_();
            while (s.is_running() && std::audio_now_ns_() - t0 < 5000000000LL) std::audio_sleep_us_(2000LL);
            printf("    %s: %u frames, %d callbacks, input %d frames\n", duplex ? "duplex" : "input", frames, c.calls, c.in_frames);
            check(c.calls == 8, "capture callbacks ran until AUDIO_CB_STOP");
            s.close();
        }
    }
    a.close();

    check(midi_default_api() == MIDI_API_WINMM, "WinMM is the default MIDI API");
    struct Result<MidiOut, AudioError> ro = std::MidiOut::open(MIDI_API_WINMM, "check");
    check(ro.ok(), "WinMM output client");
    struct MidiOut out = ro.unwrap_or(std::MidiOut::closed());
    check(out.open_virtual_port("v").code == AUDIO_ERR_UNSUPPORTED, "no virtual ports on WinMM");
    printf("    %u WinMM output port(s)\n", out.port_count());
    for (unsigned int i = 0U; i < out.port_count(); i = i + 1U) {
        struct MidiPortInfo pi;
        pi.index = 0U; pi.name[0] = (char)0;
        pi = out.port_info(i).unwrap_or(pi);
        printf("    out [%u] %s\n", i, pi.name);
        struct AudioError e = out.open_port(i, "o");
        if (e.failed()) { printf("      open: %s\n", e.message); continue; }
        check(out.send(std::MidiMsg::note_on(0, 60, 1)).ok(), "midiOutShortMsg");
        unsigned char sx[6]; sx[0] = (unsigned char)0xF0; sx[1] = (unsigned char)0x7E; sx[2] = (unsigned char)0x7F;
        sx[3] = (unsigned char)0x09; sx[4] = (unsigned char)0x01; sx[5] = (unsigned char)0xF7;   // GM on
        struct AudioError se = out.send_message(&sx[0], 6UL);
        printf("      sysex: '%s'\n", se.message);
        out.send(std::MidiMsg::note_off(0, 60, 0));
        out.close_port();
    }
    out.close();
    struct Result<MidiIn, AudioError> ri = std::MidiIn::open(MIDI_API_WINMM, "check", 0UL);
    check(ri.ok(), "WinMM input client");
    struct MidiIn in = ri.unwrap_or(std::MidiIn::closed());
    printf("    %u WinMM input port(s)\n", in.port_count());
    for (unsigned int i = 0U; i < in.port_count(); i = i + 1U) {
        struct AudioError e = in.open_port(i, "i");
        printf("    in [%u] open: '%s'\n", i, e.message);
        in.close_port();
    }
    in.close();

    // The dummy APIs work on Windows too (a thread std starts).
    struct Audio d = std::Audio::open(AUDIO_API_DUMMY).unwrap_or(std::Audio::closed());
    struct Count c; c.calls = 0; c.in_frames = 0;
    struct AudioStreamParams p = std::AudioStreamParams::of(AUDIO_DEFAULT_DEVICE, 2);
    unsigned int frames = 128U;
    struct AudioStream s = d.open_stream((?&AudioStreamParams)&p, null, AUDIO_FORMAT_F32, 48000U, &frames, count_cb,
                                         (?&Count)&c, null).unwrap_or(std::AudioStream::closed());
    s.start();
    long long t0 = std::audio_now_ns_();
    while (s.is_running() && std::audio_now_ns_() - t0 < 5000000000LL) std::audio_sleep_us_(2000LL);
    check(c.calls == 8, "dummy stream on its thread");
    s.close();
    d.close();
    printf("win32 audio: %s\n", g_fails == 0 ? "all passed" : "FAILED");
    return g_fails == 0 ? 0 : 1;
}
