// SafeC Standard Library — Block Device Abstraction
// Freestanding-safe virtual block device interface.
#pragma once

#define BLOCK_SIZE  512   // bytes per sector (standard)

// Abstract block device.  Implementors fill in read_fn/write_fn/ctx.
struct BlockDevice {
    unsigned long  sector_count;   // total sectors on device
    unsigned long  sector_size;    // bytes per sector (usually 512)

    // read_fn(ctx, lba, buf, count) → 0 on success
    void*          read_fn;
    // write_fn(ctx, lba, buf, count) → 0 on success
    void*          write_fn;
    void*          ctx;            // driver context

    // Read `count` sectors starting at `lba` into `buf`.
    int  read(unsigned long lba, &stack unsigned char buf, unsigned long count);

    // Write `count` sectors from `buf` to `lba`.
    int  write(unsigned long lba, const &stack unsigned char buf, unsigned long count);

    // Is the device valid / initialised?
    int  valid() const;
};
