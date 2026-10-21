// SafeC Standard Library — Physical Frame Allocator
#pragma once
#include <std/kernel/frame.h>

namespace std {

inline void FrameAllocator::init(unsigned long total_frames) {
    // The bitmap tracks at most FRAME_BITMAP_SIZE * 32 frames.
    unsigned long maxFrames = (unsigned long)FRAME_BITMAP_SIZE * (unsigned long)32;
    if (total_frames > maxFrames) { total_frames = maxFrames; }
    self.total_frames = total_frames;
    self.used_frames  = (unsigned long)0;
    self.base         = (unsigned long)0;
    int i = 0;
    while (i < FRAME_BITMAP_SIZE) {
        self.bitmap[i] = (unsigned int)0;
        i = i + 1;
    }
}

inline int FrameAllocator::ctz32_(unsigned int x) const {
    if (x == (unsigned int)0) { return 32; }
    int n = 0;
    while ((x & (unsigned int)1) == (unsigned int)0) {
        x = x >> 1;
        n = n + 1;
    }
    return n;
}

inline void FrameAllocator::init_at(unsigned long base_phys, unsigned long total_frames) {
    self.init(total_frames);
    self.base = base_phys & ~4095UL;
}

inline unsigned long FrameAllocator::frame_addr(unsigned long frame) const {
    return self.base + frame * 4096UL;
}

inline long long FrameAllocator::alloc() {
    // Round up: a partial last word still holds frames.
    unsigned long max_idx = (self.total_frames + (unsigned long)31) / (unsigned long)32;
    if (max_idx > (unsigned long)FRAME_BITMAP_SIZE) { max_idx = (unsigned long)FRAME_BITMAP_SIZE; }

    unsigned long i = (unsigned long)0;
    while (i < max_idx) {
        if (self.bitmap[i] != (unsigned int)0xFFFFFFFF) {
            unsigned int inv = ~self.bitmap[i];
            int bit = self.ctz32_(inv);
            unsigned long frame = i * (unsigned long)32 + (unsigned long)bit;
            if (frame >= self.total_frames) { return (long long)-1; }
            self.bitmap[i]  = self.bitmap[i] | ((unsigned int)1 << bit);
            self.used_frames = self.used_frames + (unsigned long)1;
            return (long long)frame;
        }
        i = i + (unsigned long)1;
    }
    return (long long)-1;
}

inline void FrameAllocator::free(unsigned long frame) {
    if (frame >= self.total_frames) { return; }
    unsigned long idx = frame / (unsigned long)32;
    int bit = (int)(frame & (unsigned long)31);
    if ((self.bitmap[idx] & ((unsigned int)1 << bit)) == (unsigned int)0) { return; }  // already free
    self.bitmap[idx]  = self.bitmap[idx] & ~((unsigned int)1 << bit);
    self.used_frames  = self.used_frames - (unsigned long)1;
}

inline int FrameAllocator::is_used(unsigned long frame) const {
    if (frame >= self.total_frames) { return 0; }
    unsigned long idx = frame / (unsigned long)32;
    int bit = (int)(frame & (unsigned long)31);
    if ((self.bitmap[idx] & ((unsigned int)1 << bit)) != (unsigned int)0) { return 1; }
    return 0;
}

inline void FrameAllocator::mark_range(unsigned long start, unsigned long count) {
    unsigned long i = (unsigned long)0;
    while (i < count) {
        unsigned long frame = start + i;
        if (frame >= self.total_frames) { return; }
        unsigned long idx = frame / (unsigned long)32;
        int bit = (int)(frame & (unsigned long)31);
        if ((self.bitmap[idx] & ((unsigned int)1 << bit)) == (unsigned int)0) {
            self.bitmap[idx]  = self.bitmap[idx] | ((unsigned int)1 << bit);
            self.used_frames  = self.used_frames + (unsigned long)1;
        }
        i = i + (unsigned long)1;
    }
}

inline unsigned long FrameAllocator::free_count() const {
    return self.total_frames - self.used_frames;
}

} // namespace std
