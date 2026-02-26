// SafeC Standard Library — Page Table Management
// Page entry format: bits [12:63] = physical address (4K aligned), bits [0:11] = flags
#pragma once
#include "paging.h"

void PageTable::init() {
    int i = 0;
    while (i < 512) {
        self.entries[i].value = (unsigned long)0;
        i = i + 1;
    }
}

void PageTable::map(int idx, unsigned long phys_addr, unsigned int flags) {
    if (idx < 0 || idx >= 512) { return; }
    unsigned long entry = (phys_addr & ~(unsigned long)0xFFF) | (unsigned long)flags;
    self.entries[idx].value = entry;
}

void PageTable::unmap(int idx) {
    if (idx < 0 || idx >= 512) { return; }
    self.entries[idx].value = (unsigned long)0;
}

int PageTable::is_present(int idx) const {
    if (idx < 0 || idx >= 512) { return 0; }
    if ((self.entries[idx].value & (unsigned long)1) != (unsigned long)0) { return 1; }
    return 0;
}

unsigned long PageTable::get_phys(int idx) const {
    if (idx < 0 || idx >= 512) { return (unsigned long)0; }
    if ((self.entries[idx].value & (unsigned long)1) == (unsigned long)0) { return (unsigned long)0; }
    return self.entries[idx].value & ~(unsigned long)0xFFF;
}

unsigned int PageTable::get_flags(int idx) const {
    if (idx < 0 || idx >= 512) { return (unsigned int)0; }
    return (unsigned int)(self.entries[idx].value & (unsigned long)0xFFF);
}

void PageTable::set_flags(int idx, unsigned int flags) {
    if (idx < 0 || idx >= 512) { return; }
    unsigned long phys = self.entries[idx].value & ~(unsigned long)0xFFF;
    self.entries[idx].value = phys | (unsigned long)flags;
}
