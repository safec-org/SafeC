#pragma once
// SafeC Standard Library — HAL register access helpers (internal).
// Byte-offset volatile accesses at the width each register needs; used by
// the chip implementations in std/hal/*.sc.
namespace std {

static unsigned int hal_rd32_(unsigned long base, unsigned long off) {
    unsafe { return volatile_load((unsigned int*)(base + off)); }
}
static void hal_wr32_(unsigned long base, unsigned long off, unsigned int v) {
    unsafe { volatile_store((unsigned int*)(base + off), v); }
}
static unsigned int hal_rd8_(unsigned long base, unsigned long off) {
    unsafe { return (unsigned int)volatile_load((unsigned char*)(base + off)); }
}
static void hal_wr8_(unsigned long base, unsigned long off, unsigned int v) {
    unsafe { volatile_store((unsigned char*)(base + off), (unsigned char)(v & 0xFFu)); }
}
static void hal_set32_(unsigned long base, unsigned long off, unsigned int m) {
    hal_wr32_(base, off, hal_rd32_(base, off) | m);
}
static void hal_clr32_(unsigned long base, unsigned long off, unsigned int m) {
    hal_wr32_(base, off, hal_rd32_(base, off) & ~m);
}
// Polls until (reg & mask) == want; 1 if it got there, 0 after 'spins' reads.
static int hal_wait32_(unsigned long base, unsigned long off, unsigned int mask,
                       unsigned int want, int spins) {
    while (spins > 0) {
        if ((hal_rd32_(base, off) & mask) == want) { return 1; }
        spins = spins - 1;
    }
    return 0;
}

} // namespace std
