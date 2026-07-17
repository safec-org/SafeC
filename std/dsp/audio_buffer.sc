// SafeC Standard Library — Audio Buffer Implementation
#pragma once
#include <std/dsp/audio_buffer.h>

namespace std {

inline struct AudioBuffer audio_buf_init(unsigned long channels) {
    struct AudioBuffer ab;
    ab.cap_frames = (unsigned long)AUDIO_BUF_FRAMES;
    ab.channels   = channels;
    ab.head       = (unsigned long)0;
    ab.tail       = (unsigned long)0;
    unsigned long total = (unsigned long)AUDIO_BUF_FRAMES * channels;
    unsigned long i = (unsigned long)0;
    while (i < total) {
        unsafe { ab.data[i] = (Fixed)0; }
        i = i + (unsigned long)1;
    }
    return ab;
}

inline unsigned long AudioBuffer::readable_frames() const {
    return self.head - self.tail;
}

inline unsigned long AudioBuffer::writable_frames() const {
    return self.cap_frames - (self.head - self.tail);
}

inline int AudioBuffer::is_empty() const {
    if (self.head == self.tail) { return 1; }
    return 0;
}

inline int AudioBuffer::is_full() const {
    if ((self.head - self.tail) >= self.cap_frames) { return 1; }
    return 0;
}

inline unsigned long AudioBuffer::write_frames(&stack Fixed src, unsigned long n) {
    unsigned long space = self.writable_frames();
    if (n > space) { n = space; }
    unsigned long mask = self.cap_frames - (unsigned long)1;
    unsigned long i = (unsigned long)0;
    while (i < n) {
        unsigned long slot = ((self.head + i) & mask) * self.channels;
        unsigned long ch = (unsigned long)0;
        while (ch < self.channels) {
            unsafe {
                Fixed* ps = (Fixed*)src;
                self.data[slot + ch] = ps[i * self.channels + ch];
            }
            ch = ch + (unsigned long)1;
        }
        i = i + (unsigned long)1;
    }
    unsafe { asm volatile ("" : : : "memory"); }
    self.head = self.head + n;
    return n;
}

inline unsigned long AudioBuffer::read_frames(&stack Fixed dst, unsigned long n) {
    unsigned long avail = self.readable_frames();
    if (n > avail) { n = avail; }
    unsigned long mask = self.cap_frames - (unsigned long)1;
    unsigned long i = (unsigned long)0;
    while (i < n) {
        unsigned long slot = ((self.tail + i) & mask) * self.channels;
        unsigned long ch = (unsigned long)0;
        while (ch < self.channels) {
            unsafe {
                Fixed* pd = (Fixed*)dst;
                pd[i * self.channels + ch] = self.data[slot + ch];
            }
            ch = ch + (unsigned long)1;
        }
        i = i + (unsigned long)1;
    }
    unsafe { asm volatile ("" : : : "memory"); }
    self.tail = self.tail + n;
    return n;
}

inline unsigned long AudioBuffer::peek_frames(&stack Fixed dst, unsigned long n) const {
    unsigned long avail = self.readable_frames();
    if (n > avail) { n = avail; }
    unsigned long mask = self.cap_frames - (unsigned long)1;
    unsigned long i = (unsigned long)0;
    while (i < n) {
        unsigned long slot = ((self.tail + i) & mask) * self.channels;
        unsigned long ch = (unsigned long)0;
        while (ch < self.channels) {
            unsafe {
                Fixed* pd = (Fixed*)dst;
                pd[i * self.channels + ch] = self.data[slot + ch];
            }
            ch = ch + (unsigned long)1;
        }
        i = i + (unsigned long)1;
    }
    return n;
}

inline void AudioBuffer::mix_frames(&stack Fixed src, unsigned long n) {
    unsigned long space = self.writable_frames();
    if (n > space) { n = space; }
    unsigned long mask = self.cap_frames - (unsigned long)1;
    unsigned long i = (unsigned long)0;
    while (i < n) {
        unsigned long slot = ((self.head + i) & mask) * self.channels;
        unsigned long ch = (unsigned long)0;
        while (ch < self.channels) {
            unsafe {
                Fixed* ps = (Fixed*)src;
                self.data[slot + ch] = fixed_add(self.data[slot + ch],
                                                  ps[i * self.channels + ch]);
            }
            ch = ch + (unsigned long)1;
        }
        i = i + (unsigned long)1;
    }
}

inline void AudioBuffer::clear() {
    self.head = (unsigned long)0;
    self.tail = (unsigned long)0;
    unsigned long total = self.cap_frames * self.channels;
    unsigned long i = (unsigned long)0;
    while (i < total) {
        unsafe { self.data[i] = (Fixed)0; }
        i = i + (unsigned long)1;
    }
}

} // namespace std
