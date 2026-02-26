// SafeC Standard Library — Lock-Free Audio Ring Buffer
// Stereo (or multi-channel) interleaved Q16.16 samples; SPSC safe.
// Specialized version of RingBuffer for DSP use — operates in frames.
// Freestanding-safe.
#pragma once
#include "fixed.h"

#define AUDIO_MAX_CHANNELS  8
#define AUDIO_BUF_FRAMES    512   // default frame capacity (override before include)

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

    // Write `n` interleaved frames from `src`.  Returns frames written.
    unsigned long write_frames(const Fixed* src, unsigned long n);

    // Read `n` interleaved frames into `dst`.  Returns frames read.
    unsigned long read_frames(Fixed* dst, unsigned long n);

    // Peek at `n` frames without consuming.
    unsigned long peek_frames(Fixed* dst, unsigned long n) const;

    // Mix (add) `n` frames from `src` into the next writable frames.
    // Used for audio mixing; does not advance head.
    void  mix_frames(const Fixed* src, unsigned long n);

    // Clear (zero) all frames.
    void  clear();

    int   is_empty() const;
    int   is_full() const;
};

// Initialise with given channel count (capacity = AUDIO_BUF_FRAMES).
struct AudioBuffer audio_buf_init(unsigned long channels);
