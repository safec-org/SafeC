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

// A simple DMA channel descriptor (driver fills function pointers).
//
// Generic over 'T', the driver context type start_fn/poll_fn are called
// with: unlike std/gui's Widget (one shared tree holding many widgets
// with unrelated context types at once — see README's "Outliving
// references" section) or std/dsp's TimerWheel/std/fs's BlockDevice/
// std/rpc's handler table (same shape: one process-wide, multi-tenant
// collection), a single DmaChannel is owned by exactly one driver for
// its whole lifetime, so there's no heterogeneity to lose by pinning
// 'ctx' to one concrete type per channel — a real 'T*'/'&T' fits, not
// just a 'void*' erasure. 'ctx' is '?&T' rather than '&T' because a
// driver with no per-transfer state legitimately passes none (see
// start()/wait() below, which treat a null ctx as "no context needed"
// the same way they already treat a null start_fn/poll_fn as "unset").
// start_fn/poll_fn stay 'void*' (function pointers, not object
// pointers — matches std/gui/gui_widget.h's WidgetCallback fields, cast
// back to a real 'fn' type only at the call site, same documented
// workaround).
generic<T> struct DmaChannel {
    unsigned int  id;
    int           busy;

    // start_fn(ctx, src_phys, dst_phys, len) → 0 on success.
    void*         start_fn;
    // poll_fn(ctx) → 1 when complete, 0 while busy.
    void*         poll_fn;
    ?&T           ctx;

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

} // namespace std
