// SafeC Standard Library — Physical Frame Allocator
#pragma once
#include "frame.h"

void FrameAllocator::init(unsigned long total_frames) {
    self.total_frames = total_frames;
    self.used_frames  = (unsigned long)0;
    int i = 0;
    while (i < 4096) {
        self.bitmap[i] = (unsigned int)0;
        i = i + 1;
    }
}

int FrameAllocator::ctz32_(unsigned int x) const {
    if (x == (unsigned int)0) { return 32; }
    int n = 0;
    while ((x & (unsigned int)1) == (unsigned int)0) {
        x = x >> 1;
        n = n + 1;
    }
    return n;
}

long long FrameAllocator::alloc() {
    unsigned long max_idx = self.total_frames / (unsigned long)32;
    if (max_idx > (unsigned long)4096) { max_idx = (unsigned long)4096; }

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

void FrameAllocator::free(unsigned long frame) {
    if (frame >= self.total_frames) { return; }
    unsigned long idx = frame / (unsigned long)32;
    int bit = (int)(frame & (unsigned long)31);
    self.bitmap[idx]  = self.bitmap[idx] & ~((unsigned int)1 << bit);
    self.used_frames  = self.used_frames - (unsigned long)1;
}

int FrameAllocator::is_used(unsigned long frame) const {
    if (frame >= self.total_frames) { return 0; }
    unsigned long idx = frame / (unsigned long)32;
    int bit = (int)(frame & (unsigned long)31);
    if ((self.bitmap[idx] & ((unsigned int)1 << bit)) != (unsigned int)0) { return 1; }
    return 0;
}

void FrameAllocator::mark_range(unsigned long start, unsigned long count) {
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

unsigned long FrameAllocator::free_count() const {
    return self.total_frames - self.used_frames;
}
