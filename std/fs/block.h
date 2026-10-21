// SafeC Standard Library — Block Device Abstraction
// Freestanding-safe virtual block device interface.
#pragma once

#define BLOCK_SIZE  512   // bytes per sector (standard)

// Abstract block device over a driver's context type.
namespace std {

// What a block driver provides: sector reads and writes (sectors of the
// device's sector_size), 0 on success. BlockDevice::new takes any such
// type, and a BlockDevice has the same methods, so it is one too.
trait BlockDriver {
    int read(unsigned long lba, &unsigned char buf, unsigned long count);
    int write(unsigned long lba, const &unsigned char buf, unsigned long count);
}

struct BlockDevice {
    unsigned long  sector_count;   // total sectors on device
    unsigned long  sector_size;    // bytes per sector (usually 512)

    // The driver's methods and context, stored type-erased: one VFS can
    // mount devices from different drivers at once, so BlockDevice itself
    // isn't generic. BlockDevice::new<T> is the only writer and always
    // pairs T's thunks with a &T context.
    void*          read_fn;        // fn int(&T, lba, &unsigned char buf, count) → 0 on success
    void*          write_fn;       // fn int(&T, lba, const &unsigned char buf, count)
    void*          ctx;            // &T

    // A device of `sector_count` sectors of `sector_size` bytes whose
    // read()/write() call drv.read(...)/drv.write(...). drv must outlive it.
    generic<T: BlockDriver>
    static struct BlockDevice new(&T drv, unsigned long sector_count, unsigned long sector_size);

    // Read `count` sectors starting at `lba` into `buf`.
    int  read(unsigned long lba, &unsigned char buf, unsigned long count);

    // Write `count` sectors from `buf` to `lba`.
    int  write(unsigned long lba, const &unsigned char buf, unsigned long count);

    // Is the device valid / initialised?
    int  valid() const;
};

generic<T: BlockDriver>
int block_read_thunk_(&T drv, unsigned long lba, &unsigned char buf, unsigned long count) {
    return drv.read(lba, buf, count);
}

generic<T: BlockDriver>
int block_write_thunk_(&T drv, unsigned long lba, const &unsigned char buf, unsigned long count) {
    return drv.write(lba, buf, count);
}

generic<T: BlockDriver>
struct BlockDevice BlockDevice::new(&T drv, unsigned long sector_count, unsigned long sector_size) {
    struct BlockDevice dev;
    dev.sector_count = sector_count;
    dev.sector_size  = sector_size;
    fn int(&T, unsigned long, &unsigned char, unsigned long) r = block_read_thunk_;
    fn int(&T, unsigned long, const &unsigned char, unsigned long) w = block_write_thunk_;
    unsafe {
        dev.read_fn  = (void*)r;
        dev.write_fn = (void*)w;
        dev.ctx      = (void*)drv;
    }
    return dev;
}

} // namespace std
