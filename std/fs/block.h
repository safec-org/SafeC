// SafeC Standard Library — Block Device Abstraction
// Freestanding-safe virtual block device interface.
#pragma once

#define BLOCK_SIZE  512   // bytes per sector (standard)

// Abstract block device.  Implementors fill in read_fn/write_fn/ctx.
namespace std {

struct BlockDevice {
    unsigned long  sector_count;   // total sectors on device
    unsigned long  sector_size;    // bytes per sector (usually 512)

    // Driver callbacks and context, stored type-erased: one VFS can mount
    // devices from different drivers at once, so BlockDevice itself isn't
    // generic. BlockDevice::new<T> is the only writer and always
    // pairs fn(&T, ...) callbacks with a &T context.
    void*          read_fn;        // fn int(&T, lba, &unsigned char buf, count) → 0 on success
    void*          write_fn;       // fn int(&T, lba, const &unsigned char buf, count)
    void*          ctx;            // &T

    // A device of `sector_count` sectors of `sector_size` bytes whose
    // read()/write() call read(ctx, ...)/write(ctx, ...) (0 on success).
    generic<T>
    static struct BlockDevice new(&T ctx, unsigned long sector_count, unsigned long sector_size,
                                  fn int(&T, unsigned long, &unsigned char, unsigned long) read,
                                  fn int(&T, unsigned long, const &unsigned char, unsigned long) write);

    // Read `count` sectors starting at `lba` into `buf`.
    int  read(unsigned long lba, &unsigned char buf, unsigned long count);

    // Write `count` sectors from `buf` to `lba`.
    int  write(unsigned long lba, const &unsigned char buf, unsigned long count);

    // Is the device valid / initialised?
    int  valid() const;
};

generic<T>
struct BlockDevice BlockDevice::new(&T ctx, unsigned long sector_count, unsigned long sector_size,
                                    fn int(&T, unsigned long, &unsigned char, unsigned long) read,
                                    fn int(&T, unsigned long, const &unsigned char, unsigned long) write) {
    struct BlockDevice dev;
    dev.sector_count = sector_count;
    dev.sector_size  = sector_size;
    unsafe {
        dev.read_fn  = (void*)read;
        dev.write_fn = (void*)write;
        dev.ctx      = (void*)ctx;
    }
    return dev;
}

} // namespace std
