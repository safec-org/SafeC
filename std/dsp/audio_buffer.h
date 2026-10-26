// SafeC Standard Library — Lock-Free Audio Ring Buffer
// Stereo (or multi-channel) interleaved Q8.24 samples; SPSC safe.
// Specialized version of RingBuffer for DSP use — operates in frames.
// Freestanding-safe.
#pragma once
#include <std/dsp/fixed.h>

#define AUDIO_MAX_CHANNELS  8
#define AUDIO_BUF_FRAMES    512   // default frame capacity (override before include)

namespace std {

struct AudioBuffer {
    Fixed         data[AUDIO_BUF_FRAMES * AUDIO_MAX_CHANNELS];
    unsigned long cap_frames;     // capacity in frames
    unsigned long channels;       // number of channels (1=mono, 2=stereo, …)
    volatile unsigned long head;  // write head (frames)
    volatile unsigned long tail;  // read head  (frames)

    // Frames available to read.
    unsigned long readable_frames() const;

    // Frames available to write.
    unsigned long writable_frames() const;

    // Buffers are interleaved frames: a slice of k * channels samples
    // holds k frames (a partial frame at the end is ignored).

    // Write the frames in `src` (as many as fit).  Returns frames written.
    unsigned long write_frames([]const Fixed src);

    // Read up to dst.len / channels frames into `dst`.  Returns frames read.
    unsigned long read_frames([]Fixed dst);

    // Like read_frames, without consuming.
    unsigned long peek_frames([]Fixed dst) const;

    // Mix (add) the frames in `src` into the next writable frames.
    // Used for audio mixing; does not advance head. Returns frames mixed.
    unsigned long mix_frames([]const Fixed src);

    // Clear (zero) all frames.
    void  clear();

    int   is_empty() const;
    int   is_full() const;

    // An empty buffer of 'channels' channels (capacity = AUDIO_BUF_FRAMES
    // frames), zeroed.
    static struct AudioBuffer new(unsigned long channels);
};

} // namespace std
