// SafeC Standard Library — Block Device Implementation
#pragma once
#include <std/fs/block.h>

namespace std {

static int block_whole_sectors_(&BlockDevice d, unsigned long len) {
    return d.sector_size != 0UL && len != 0UL && len % d.sector_size == 0UL;
}

inline int BlockDevice::read(unsigned long lba, []unsigned char buf) {
    if (!block_whole_sectors_(self, buf.len)) { return -1; }
    match (self.read_fn) {
        case some(f): {
            match (self.ctx) { case some(c): return f(c, lba, buf); case null: { } }
        }
        case null: { }
    }
    return -1;
}

inline int BlockDevice::write(unsigned long lba, []const unsigned char buf) {
    if (!block_whole_sectors_(self, buf.len)) { return -1; }
    match (self.write_fn) {
        case some(f): {
            match (self.ctx) { case some(c): return f(c, lba, buf); case null: { } }
        }
        case null: { }
    }
    return -1;
}

inline int BlockDevice::valid() const {
    if (!self.read_fn.is_null() && self.sector_count > (unsigned long)0) { return 1; }
    return 0;
}

} // namespace std
