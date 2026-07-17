// SafeC Standard Library — Block Device Implementation
#pragma once
#include <std/fs/block.h>

namespace std {

inline int BlockDevice::read(unsigned long lba, &stack unsigned char buf, unsigned long count) {
    if (self.read_fn == (void*)0) { return -1; }
    unsafe {
        fn int(void*, unsigned long, unsigned char*, unsigned long) func =
            (fn int(void*, unsigned long, unsigned char*, unsigned long))self.read_fn;
        return func(self.ctx, lba, (unsigned char*)buf, count);
    }
    return -1;
}

inline int BlockDevice::write(unsigned long lba, const &stack unsigned char buf, unsigned long count) {
    if (self.write_fn == (void*)0) { return -1; }
    unsafe {
        fn int(void*, unsigned long, const unsigned char*, unsigned long) func =
            (fn int(void*, unsigned long, const unsigned char*, unsigned long))self.write_fn;
        return func(self.ctx, lba, (const unsigned char*)buf, count);
    }
    return -1;
}

inline int BlockDevice::valid() const {
    if (self.read_fn != (void*)0 && self.sector_count > (unsigned long)0) { return 1; }
    return 0;
}

} // namespace std
