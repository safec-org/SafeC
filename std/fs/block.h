// SafeC Standard Library — Block Device Abstraction
// Freestanding-safe virtual block device interface.
#pragma once
#include <std/erased.h>

#define BLOCK_SIZE  512   // bytes per sector (standard)

// Abstract block device over a driver's context type.
namespace std {

// What a block driver provides: whole-sector reads and writes at a
// sector address, 0 on success. A buffer covers buf.len / sector_size
// sectors (the device's sector_size), and its length is a multiple of it.
// BlockDevice::new takes any such type, and a BlockDevice has the same
// methods, so it is one too.
trait BlockDriver {
    int read(unsigned long lba, []unsigned char buf);
    int write(unsigned long lba, []const unsigned char buf);
}

struct BlockDevice {
    unsigned long  sector_count;   // total sectors on device
    unsigned long  sector_size;    // bytes per sector (usually 512)

    // The driver's methods and context, stored type-erased (std/erased.h):
    // one VFS can mount devices from different drivers at once, so
    // BlockDevice itself isn't generic. BlockDevice::new<T> is the only
    // writer and always pairs T's thunks with a &T context.
    ?fn int(&Erased, unsigned long, []unsigned char) read_fn;
    ?fn int(&Erased, unsigned long, []const unsigned char) write_fn;
    ?&Erased       ctx;            // the &T

    // A device of `sector_count` sectors of `sector_size` bytes whose
    // read()/write() call drv.read(...)/drv.write(...). drv must outlive it.
    generic<T: BlockDriver>
    static struct BlockDevice new(&T drv, unsigned long sector_count, unsigned long sector_size);

    // Reads buf.len / sector_size sectors starting at `lba` into `buf`.
    // -1 when buf is empty or not a whole number of sectors.
    int  read(unsigned long lba, []unsigned char buf);

    // Writes the buf.len / sector_size sectors of `buf` at `lba` (same rule).
    int  write(unsigned long lba, []const unsigned char buf);

    // Is the device valid / initialised?
    int  valid() const;
};

generic<T: BlockDriver>
int block_read_thunk_(&T drv, unsigned long lba, []unsigned char buf) {
    return drv.read(lba, buf);
}

generic<T: BlockDriver>
int block_write_thunk_(&T drv, unsigned long lba, []const unsigned char buf) {
    return drv.write(lba, buf);
}

generic<T: BlockDriver>
struct BlockDevice BlockDevice::new(&T drv, unsigned long sector_count, unsigned long sector_size) {
    struct BlockDevice dev;
    dev.sector_count = sector_count;
    dev.sector_size  = sector_size;
    fn int(&T, unsigned long, []unsigned char) r = block_read_thunk_;
    fn int(&T, unsigned long, []const unsigned char) w = block_write_thunk_;
    unsafe {
        dev.read_fn  = (?fn int(&Erased, unsigned long, []unsigned char))r;
        dev.write_fn = (?fn int(&Erased, unsigned long, []const unsigned char))w;
        dev.ctx      = (?&Erased)drv;
    }
    return dev;
}

} // namespace std
