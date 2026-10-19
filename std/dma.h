// SafeC Standard Library — DMA-Safe Memory Abstractions
// Provides cache-coherent buffer management for DMA transfers.
// Freestanding-safe; works with any allocator or static storage.
#pragma once

#define DMA_ALIGN  64   // typical DMA burst-alignment (cache line)

// A DMA-coherent buffer descriptor.
namespace std {

struct DmaBuffer {
    void*         virt;       // virtual address (CPU view)
    unsigned long phys;       // physical address (device view, 0 if identity-mapped)
    unsigned long size;       // buffer size in bytes
    int           owner;      // 0 = CPU owns, 1 = device owns

    // Mark buffer as owned by device (flush CPU cache → DRAM before transfer).
    void  to_device();

    // Mark buffer as owned by CPU (invalidate cache after DMA completes).
    void  to_cpu();

    // Is the buffer currently safe to access from the CPU?
    int   cpu_accessible() const;

    // Return physical address of byte at `offset`.
    unsigned long phys_at(unsigned long offset) const;
};

// Initialise a DmaBuffer wrapping an existing allocation.
// `phys` = 0 means physical == virtual (identity-mapped / non-MMU systems).
struct DmaBuffer dma_buf_wrap(void* virt, unsigned long phys, unsigned long size);

// A DMA channel descriptor, generic over the driver context type T its
// callbacks take. One driver owns a channel for its whole lifetime, so the
// context can be pinned to one T per channel; it is '?&T' because a driver
// with no per-transfer state passes none.
generic<T> struct DmaChannel {
    unsigned int  id;
    int           busy;

    fn int(?&T, unsigned long, unsigned long, unsigned long) start_fn;  // (ctx, src, dst, len) → 0 on success
    fn int(?&T)   poll_fn;    // (ctx) → 1 when complete, 0 while busy
    int           has_poll;   // 0: wait() returns at once (synchronous engines)
    ?&T           ctx;

    // Start a memory-to-memory DMA transfer.
    int  start(unsigned long src_phys, unsigned long dst_phys, unsigned long len);

    // Poll until the channel is idle.  Returns 0 on success, -1 on timeout.
    int  wait(unsigned int max_polls);

    // Is this channel currently busy?
    int  is_busy() const;
};

generic<T>
struct DmaChannel<T> dma_channel_new(unsigned int id,
                                     fn int(?&T, unsigned long, unsigned long, unsigned long) start_fn,
                                     fn int(?&T) poll_fn, ?&T ctx) {
    struct DmaChannel<T> ch;
    ch.id = id;
    ch.busy = 0;
    ch.start_fn = start_fn;
    ch.poll_fn = poll_fn;
    ch.has_poll = 1;
    ch.ctx = ctx;
    return ch;
}

generic<T>
int DmaChannel::start(unsigned long src_phys, unsigned long dst_phys, unsigned long len) {
    if (self.busy != 0) return -1;
    int rc = self.start_fn(self.ctx, src_phys, dst_phys, len);
    if (rc == 0) self.busy = 1;
    return rc;
}

generic<T>
int DmaChannel::wait(unsigned int max_polls) {
    if (!self.has_poll) { self.busy = 0; return 0; }
    unsigned int i = 0U;
    while (i < max_polls) {
        if (self.poll_fn(self.ctx) != 0) { self.busy = 0; return 0; }
        i = i + 1U;
    }
    return -1;
}

generic<T>
int DmaChannel::is_busy() const { return self.busy; }

// Scatter-gather entry.
struct DmaSgEntry {
    unsigned long phys;
    unsigned long len;
};

#define DMA_SG_MAX  16

struct DmaSgList {
    struct DmaSgEntry entries[DMA_SG_MAX];
    int               count;

    void  add(unsigned long phys, unsigned long len);
    void  clear();
    unsigned long total_len() const;
};

} // namespace std
