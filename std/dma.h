// SafeC Standard Library — DMA-Safe Memory Abstractions
// Provides cache-coherent buffer management for DMA transfers.
// Freestanding-safe; works with any allocator or static storage.
#pragma once

#define DMA_ALIGN  64   // typical DMA burst-alignment (cache line)

// A DMA-coherent buffer descriptor.
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

// A simple DMA channel descriptor (driver fills function pointers).
struct DmaChannel {
    unsigned int  id;
    int           busy;

    // start_fn(ctx, src_phys, dst_phys, len) → 0 on success.
    void*         start_fn;
    // poll_fn(ctx) → 1 when complete, 0 while busy.
    void*         poll_fn;
    void*         ctx;

    // Start a memory-to-memory DMA transfer.
    int  start(unsigned long src_phys, unsigned long dst_phys, unsigned long len);

    // Poll until the channel is idle.  Returns 0 on success, -1 on timeout.
    int  wait(unsigned int max_polls);

    // Is this channel currently busy?
    int  is_busy() const;
};

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
