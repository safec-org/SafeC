// audio_demo — std/audio in three commands:
//
//   audio_demo list  [--api NAME]                    the APIs, audio devices and MIDI ports
//   audio_demo sine  [--api NAME] [--device ID] [--freq HZ] [--seconds S] [--rate HZ]
//   audio_demo midi  [--api NAME] [--port N | --virtual] [--seconds S]
//
// --api takes an audio or MIDI API name (alsa, jack, coreaudio, wasapi,
// coremidi, winmm, dummy, ...); without it each command uses the
// platform's default. 'sine' plays a 440 Hz tone on the default output (a
// short fade in and out, so no clicks); 'midi' prints every message the
// chosen input port receives — or, with --virtual, what other programs send
// to a port named "audio_demo" it creates. 'audio_demo sine --api dummy'
// runs anywhere, sound card or not.
#include <std/audio/audio.h>
#include <std/audio/audio_sys.h>
#include <std/audio/midi.h>
#include <std/math.h>

extern int printf(const char* fmt, ...);
extern double atof(const char* s);
extern int atoi(const char* s);

struct Args {
    const char* command;
    const char* api;
    unsigned int device;
    double freq;
    double seconds;
    unsigned int rate;
    int port;
    int virtual_port;
};

static int streq(const char* a, const char* b) {
    unsafe { return std::audio_streq_((const ?&char)a, (const ?&char)b); }
}

static struct Args parse(int argc, char** argv) {
    struct Args a;
    a.command = "list";
    a.api = (const char*)0;
    a.device = AUDIO_DEFAULT_DEVICE;
    a.freq = 440.0;
    a.seconds = -1.0;
    a.rate = 48000U;
    a.port = 0;
    a.virtual_port = 0;
    unsafe {
        int i = 1;
        if (argc > 1 && argv[1][0] != '-') { a.command = (const char*)argv[1]; i = 2; }
        while (i < argc) {
            const char* k = (const char*)argv[i];
            const char* v = i + 1 < argc ? (const char*)argv[i + 1] : (const char*)"";
            if (streq(k, "--api")) { a.api = v; i = i + 1; }
            else if (streq(k, "--device")) { a.device = (unsigned int)atoi(v); i = i + 1; }
            else if (streq(k, "--freq")) { a.freq = atof(v); i = i + 1; }
            else if (streq(k, "--seconds")) { a.seconds = atof(v); i = i + 1; }
            else if (streq(k, "--rate")) { a.rate = (unsigned int)atoi(v); i = i + 1; }
            else if (streq(k, "--port")) { a.port = atoi(v); i = i + 1; }
            else if (streq(k, "--virtual")) { a.virtual_port = 1; }
            else printf("audio_demo: ignoring '%s'\n", k);
            i = i + 1;
        }
    }
    return a;
}

static int audio_api_of(&Args a) {
    if (a.api == (const char*)0) return AUDIO_API_UNSPECIFIED;
    unsafe { return std::audio_api_by_name((const &char)a.api); }
}

static int midi_api_of(&Args a) {
    if (a.api == (const char*)0) return MIDI_API_UNSPECIFIED;
    int m = 0;
    unsafe { m = std::midi_api_by_name((const &char)a.api); }
    // "alsa", "jack", "dummy" name both; an audio-only name means the default MIDI API.
    return m;
}

static void print_formats(unsigned int f) {
    unsigned int all[6];
    all[0] = AUDIO_FORMAT_I8; all[1] = AUDIO_FORMAT_I16; all[2] = AUDIO_FORMAT_I24;
    all[3] = AUDIO_FORMAT_I32; all[4] = AUDIO_FORMAT_F32; all[5] = AUDIO_FORMAT_F64;
    for (int i = 0; i < 6; i = i + 1) if ((f & all[i]) != 0U) printf(" %s", std::audio_format_name(all[i]));
}

static void list_audio(int api) {
    struct Result<Audio, AudioError> r = std::Audio::open(api);
    match (r.get_err()) {
        case some(e): { printf("  %s: %s\n", std::audio_api_display_name(api), e.message); return; }
        case null: {}
    }
    struct Audio a = r.unwrap_or(std::Audio::closed());
    printf("  %s: %u device(s)\n", std::audio_api_display_name(a.api()), a.device_count());
    for (unsigned int i = 0U; i < a.device_count(); i = i + 1U) {
        struct AudioDeviceInfo d = a.device_info(i).unwrap_or(std::AudioDeviceInfo::none());
        printf("    [%u] %s%s%s\n", d.id, d.name, d.is_default_output ? "  (default output)" : "",
               d.is_default_input ? "  (default input)" : "");
        printf("        %d out, %d in, %d duplex; formats", d.output_channels, d.input_channels, d.duplex_channels);
        print_formats(d.native_formats);
        printf("; rates");
        for (int k = 0; k < d.sample_rate_count; k = k + 1) printf(" %u", d.sample_rates[k]);
        printf(" (preferred %u)\n", d.preferred_sample_rate);
    }
    a.close();
}

static void list_midi(int api) {
    struct Result<MidiIn, AudioError> ri = std::MidiIn::open(api, "audio_demo", 0UL);
    match (ri.get_err()) {
        case some(e): { printf("  %s: %s\n", std::midi_api_display_name(api), e.message); return; }
        case null: {}
    }
    struct MidiIn in = ri.unwrap_or(std::MidiIn::closed());
    struct MidiOut out = std::MidiOut::open(in.api(), "audio_demo").unwrap_or(std::MidiOut::closed());
    printf("  %s: %u input port(s), %u output port(s)\n", std::midi_api_display_name(in.api()), in.port_count(),
           out.port_count());
    for (unsigned int i = 0U; i < in.port_count(); i = i + 1U) {
        struct MidiPortInfo p;
        p.index = i; p.name[0] = (char)0;
        p = in.port_info(i).unwrap_or(p);
        printf("    in  [%u] %s\n", i, p.name);
    }
    for (unsigned int i = 0U; i < out.port_count(); i = i + 1U) {
        struct MidiPortInfo p;
        p.index = i; p.name[0] = (char)0;
        p = out.port_info(i).unwrap_or(p);
        printf("    out [%u] %s\n", i, p.name);
    }
    in.close();
    out.close();
}

static int cmd_list(&Args a) {
    int apis[AUDIO_API_COUNT];
    int n = std::audio_compiled_apis(apis[0..AUDIO_API_COUNT]);
    printf("audio APIs compiled in:");
    for (int i = 0; i < n; i = i + 1) printf(" %s", std::audio_api_name(apis[i]));
    printf(" (default: %s)\n", std::audio_api_name(std::audio_default_api()));
    int want = audio_api_of(a);
    for (int i = 0; i < n; i = i + 1) if (want == AUDIO_API_UNSPECIFIED || want == apis[i]) list_audio(apis[i]);
    int mapis[MIDI_API_COUNT];
    int m = std::midi_compiled_apis(mapis[0..MIDI_API_COUNT]);
    printf("MIDI APIs compiled in:");
    for (int i = 0; i < m; i = i + 1) printf(" %s", std::midi_api_name(mapis[i]));
    printf(" (default: %s)\n", std::midi_api_name(std::midi_default_api()));
    int mwant = midi_api_of(a);
    for (int i = 0; i < m; i = i + 1) if (mwant == MIDI_API_UNSPECIFIED || mwant == mapis[i]) list_midi(mapis[i]);
    return 0;
}

// ── sine ─────────────────────────────────────────────────────────────────

struct Tone {
    double phase;          // radians
    double step;           // radians per frame
    long   frame;          // frames rendered
    long   total;          // frames to render before stopping
    long   fade;           // fade in/out length in frames
    unsigned int xruns;
};

// The real-time callback: no allocation, no locks, no I/O.
int render_tone(&AudioIo io, ?&Tone tp) {
    match (tp) {
        case some(t): {
            if (io.status != 0U) t.xruns = t.xruns + 1U;
            []float out = io.output_f32();
            int ch = io.output_channels;
            for (unsigned int f = 0U; f < io.frames; f = f + 1U) {
                double gain = 0.2;
                long left = t.total - t.frame;
                if (t.frame < t.fade) gain = gain * (double)t.frame / (double)t.fade;
                if (left < t.fade) gain = gain * (double)(left < 0L ? 0L : left) / (double)t.fade;
                float v = (float)(gain * std::sin_d(t.phase));
                for (int c = 0; c < ch; c = c + 1) out[(unsigned long)(f * (unsigned int)ch) + (unsigned long)c] = v;
                t.phase = t.phase + t.step;
                if (t.phase > 6.283185307179586) t.phase = t.phase - 6.283185307179586;
                t.frame = t.frame + 1L;
            }
            if (t.frame >= t.total) return AUDIO_CB_STOP;      // play out what's queued, then stop
        }
        case null: {}
    }
    return AUDIO_CB_CONTINUE;
}

static int cmd_sine(&Args a) {
    struct Result<Audio, AudioError> r = std::Audio::open(audio_api_of(a));
    match (r.get_err()) { case some(e): { printf("audio_demo: %s\n", e.message); return 1; } case null: {} }
    struct Audio audio = r.unwrap_or(std::Audio::closed());
    unsigned int dev = a.device != AUDIO_DEFAULT_DEVICE ? a.device : audio.default_output_device();
    struct AudioDeviceInfo info = audio.device_by_id(dev).unwrap_or(std::AudioDeviceInfo::none());
    if (info.id == 0U) {
        printf("audio_demo: no output device on %s (try --api dummy)\n", std::audio_api_display_name(audio.api()));
        audio.close();
        return 1;
    }
    unsigned int rate = a.rate;
    if (!info.supports_rate(rate)) rate = info.preferred_sample_rate;
    int channels = info.output_channels >= 2 ? 2 : 1;
    struct Tone tone;
    tone.phase = 0.0;
    tone.step = 2.0 * 3.141592653589793 * a.freq / (double)rate;
    tone.frame = 0L;
    tone.total = (long)((a.seconds < 0.0 ? 2.0 : a.seconds) * (double)rate);
    tone.fade = (long)rate / 50L;
    tone.xruns = 0U;
    struct AudioStreamParams out = std::AudioStreamParams::of(info.id, channels);
    struct AudioStreamOptions opts = std::AudioStreamOptions::with_flags(AUDIO_FLAG_SCHEDULE_REALTIME);
    std::audio_strcpy_(&opts.stream_name[0], (unsigned long)AUDIO_STREAM_NAME_MAX, "audio_demo");
    unsigned int frames = 256U;
    struct Result<AudioStream, AudioError> sr =
        audio.open_stream((?&AudioStreamParams)&out, null, AUDIO_FORMAT_F32, rate, &frames, render_tone,
                          (?&Tone)&tone, (?&AudioStreamOptions)&opts);
    match (sr.get_err()) { case some(e): { printf("audio_demo: %s\n", e.message); audio.close(); return 1; } case null: {} }
    struct AudioStream s = sr.unwrap_or(std::AudioStream::closed());
    printf("playing %.0f Hz on '%s' (%s): %d ch, %u Hz, %u frames/callback, latency %ld frames\n", a.freq, info.name,
           std::audio_api_display_name(audio.api()), channels, s.sample_rate(), frames, s.latency());
    struct AudioError e = s.start();
    if (e.failed()) { printf("audio_demo: %s\n", e.message); s.close(); audio.close(); return 1; }
    while (s.is_running()) std::audio_sleep_us_(10000LL);
    struct AudioError late = s.last_error();
    if (late.failed()) printf("audio_demo: the stream stopped: %s\n", late.message);
    printf("done: %.2f s of audio, %u callback(s) reported an xrun\n", s.stream_time(), tone.xruns);
    s.close();
    audio.close();
    return 0;
}

// ── MIDI monitor ─────────────────────────────────────────────────────────

static void print_message([]const unsigned char b, double t) {
    long n = (long)b.len;
    {
        printf("%10.4f  ", t);
        for (long i = 0L; i < n && i < 12L; i = i + 1L) printf("%02X ", (unsigned int)b[i]);
        if (n > 12L) printf("... (%ld bytes)", n);
        struct MidiMsg m = std::MidiMsg::of(b);
        int type = m.type();
        if (m.is_note_on()) printf("  note on  ch %d note %d vel %d", m.channel() + 1, m.data1(), m.data2());
        else if (m.is_note_off()) printf("  note off ch %d note %d", m.channel() + 1, m.data1());
        else if (type == MIDI_CONTROL_CHANGE) printf("  CC       ch %d #%d = %d", m.channel() + 1, m.data1(), m.data2());
        else if (type == MIDI_PROGRAM_CHANGE) printf("  program  ch %d %d", m.channel() + 1, m.data1());
        else if (type == MIDI_PITCH_BEND) printf("  bend     ch %d %d", m.channel() + 1, m.pitch_bend_value());
        else if (b[0] == (unsigned char)MIDI_SYSEX) printf("  sysex");
        printf("\n");
    }
}

static unsigned char g_buf[65536];

static int cmd_midi(&Args a) {
    struct Result<MidiIn, AudioError> r = std::MidiIn::open(midi_api_of(a), "audio_demo", 0UL);
    match (r.get_err()) { case some(e): { printf("audio_demo: %s\n", e.message); return 1; } case null: {} }
    struct MidiIn in = r.unwrap_or(std::MidiIn::closed());
    struct AudioError e = std::AudioError::none();
    if (a.virtual_port || in.port_count() == 0U) {
        e = in.open_virtual_port("audio_demo");
        if (e.ok()) printf("listening on virtual port 'audio_demo' (%s)\n", std::midi_api_display_name(in.api()));
    } else {
        struct MidiPortInfo p;
        p.index = 0U; p.name[0] = (char)0;
        p = in.port_info((unsigned int)a.port).unwrap_or(p);
        e = in.open_port((unsigned int)a.port, "audio_demo in");
        if (e.ok()) printf("listening on '%s' (%s)\n", p.name, std::midi_api_display_name(in.api()));
    }
    if (e.failed()) { printf("audio_demo: %s\n", e.message); in.close(); return 1; }
    in.ignore_types(0, 1, 1);          // show sysex; drop clock and active sensing
    double seconds = a.seconds < 0.0 ? 10.0 : a.seconds;
    long long end = std::audio_now_ns_() + (long long)(seconds * 1000000000.0);
    int count = 0;
    while (std::audio_now_ns_() < end) {
        double t = 0.0;
        // g_buf holds the longest message std reassembles (MIDI_MAX_SYSEX).
        long n = in.get_message(g_buf[0..MIDI_MAX_SYSEX], &t);
        if (n > 0L) { print_message(g_buf[0..(unsigned long)n], t); count = count + 1; continue; }
        std::audio_sleep_us_(2000LL);
    }
    printf("%d message(s), %lu dropped\n", count, in.dropped());
    in.close();
    return 0;
}

int main(int argc, char** argv) {
    struct Args a = parse(argc, argv);
    if (streq(a.command, "list")) return cmd_list(&a);
    if (streq(a.command, "sine")) return cmd_sine(&a);
    if (streq(a.command, "midi")) return cmd_midi(&a);
    printf("usage: audio_demo [list|sine|midi] [--api NAME] [--device ID] [--freq HZ] [--seconds S] [--rate HZ]\n"
           "                  [--port N | --virtual]\n");
    return 2;
}
