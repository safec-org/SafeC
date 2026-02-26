// SafeC Standard Library — Physical Frame Allocator
// Bitmap-based physical memory frame allocator. Freestanding-safe.
#pragma once

// Each bit represents one 4K frame. 1 = allocated, 0 = free.
// 4096 unsigned int entries = 4096 * 32 = 131072 frames = 512 MB.
#define FRAME_BITMAP_SIZE 4096

struct FrameAllocator {
    unsigned int  bitmap[FRAME_BITMAP_SIZE];
    unsigned long total_frames;  // total number of physical frames
    unsigned long used_frames;   // currently allocated frames

    // Initialize for `total_frames` frames (clears bitmap).
    void          init(unsigned long total_frames);

    // Allocate a single frame. Returns frame number, or -1 if OOM.
    long long     alloc();

    // Free a frame by frame number.
    void          free(unsigned long frame);

    // Check if a frame is allocated. Returns 1 if allocated.
    int           is_used(unsigned long frame) const;

    // Mark a range of frames as used (e.g., for kernel image).
    void          mark_range(unsigned long start, unsigned long count);

    // Return the number of free frames.
    unsigned long free_count() const;

    // Internal: count trailing zeros in a 32-bit word.
    int           ctz32_(unsigned int x) const;
};
