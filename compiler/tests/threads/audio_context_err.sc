// SAFEC-ARGS: -I %root/std -I %root --compat-preprocessor
// An audio / MIDI callback's context is reached from the API's thread
// while its owner still holds it: it must be Send + Sync.
#include <std/audio/audio.h>
#include <std/audio/midi.h>
struct Plain { int frames; float gain; };
struct Raw { int frames; float* scratch; };
static int cb_plain(&AudioIo io, ?&Plain p) { return AUDIO_CB_CONTINUE; }
static int cb_raw(&AudioIo io, ?&Raw p) { return AUDIO_CB_CONTINUE; }
static void on_midi(&MidiEvent ev, ?&Raw p) { }

void streams(&Audio a, &MidiIn in) {
    unsigned int frames = 0U;
    struct Plain pl; pl.frames = 0; pl.gain = 1.0f;
    struct Raw rw; rw.frames = 0; rw.scratch = (float*)0;
    struct Result<AudioStream, AudioError> ok = a.open_stream(null, null, AUDIO_FORMAT_F32, 48000U, &frames, cb_plain, (?&Plain)&pl, null);
    drop(ok);
    struct Result<AudioStream, AudioError> bad = a.open_stream(null, null, AUDIO_FORMAT_F32, 48000U, &frames, cb_raw, (?&Raw)&rw, null);   // EXPECT-ERROR: type 'Raw' does not satisfy constraint 'Send'
    drop(bad);
    in.set_callback(on_midi, (?&Raw)&rw);   // EXPECT-ERROR: type 'Raw' does not satisfy constraint 'Send'
    // ... and neither is it Sync (once for each call above):
    // EXPECT-ERROR: type 'Raw' does not satisfy constraint 'Sync'
    // EXPECT-ERROR: type 'Raw' does not satisfy constraint 'Sync'
}
