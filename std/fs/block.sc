// SafeC Standard Library — Block Device Implementation
#pragma once
#include "block.h"

int BlockDevice::read(unsigned long lba, &stack unsigned char buf, unsigned long count) {
    if (self.read_fn == (void*)0) { return -1; }
    unsafe {
        int (*fn)(void*, unsigned long, unsigned char*, unsigned long) =
            (int (*)(void*, unsigned long, unsigned char*, unsigned long))self.read_fn;
        return fn(self.ctx, lba, (unsigned char*)buf, count);
    }
    return -1;
}

int BlockDevice::write(unsigned long lba, const &stack unsigned char buf, unsigned long count) {
    if (self.write_fn == (void*)0) { return -1; }
    unsafe {
        int (*fn)(void*, unsigned long, const unsigned char*, unsigned long) =
            (int (*)(void*, unsigned long, const unsigned char*, unsigned long))self.write_fn;
        return fn(self.ctx, lba, (const unsigned char*)buf, count);
    }
    return -1;
}

int BlockDevice::valid() const {
    if (self.read_fn != (void*)0 && self.sector_count > (unsigned long)0) { return 1; }
    return 0;
}
