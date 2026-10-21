// qemu-system-i386 (multiboot): MmuX86_32 with paging really on, then
// MmuX86_64 tables entered from long mode by start_x86.S's 64-bit stub.
// Output goes to COM1 (port 0x3F8).
#include <std/kernel/mmu.h>
#include <std/kernel/frame.h>

#define IMAGE    0x00100000UL   // image + stack
#define POOL32   0x00400000UL
#define POOL64   0x00600000UL
#define DATA     0x00500000UL   // keep in sync with start_x86.S
#define ALIAS32  0x40000000UL
#define ALIAS64  0x0000004000000000UL

extern void qemu_outb(int port, int value);
extern void qemu_paging_on();
extern void qemu_paging_off();
extern void qemu_long_mode(unsigned int pml4);

static struct FrameAllocator g_frames;

static void say(const &char s) {
    unsigned long i = 0UL;
    unsafe { while (s[i] != (char)0) { qemu_outb(0x3F8, (int)s[i]); i = i + 1UL; } }
}
static void finish(int ok) {
    say(ok ? "QEMU-TEST PASS\n" : "QEMU-TEST FAIL\n");
    qemu_outb(0xF4, 0);
    while (1) { }
}

static unsigned long fresh_root_(unsigned long pool) {
    g_frames.init_at(pool, 32UL);
    unsigned long root = g_frames.frame_addr((unsigned long)g_frames.alloc());
    unsafe {
        unsigned char* r = (unsigned char*)root;
        unsigned long k = 0UL;
        while (k < 4096UL) { r[k] = (unsigned char)0; k = k + 1UL; }
    }
    return root;
}

void qemu_main() {
    say("uart: COM1 ok\n");
    unsafe { ((unsigned int*)DATA)[0] = 0xC0FFEEu; }

    // 32-bit paging (no PAE).
    unsigned long root = fresh_root_(POOL32);
    struct MmuContext<MmuX86_32> m = MmuContext::new(root, &g_frames, 0UL);
    int ok = 1;
    unsigned long a = IMAGE;
    while (a < POOL64 + 32UL * 4096UL) { ok = ok & m.map(a, a, (unsigned int)(MMU_WRITE | MMU_EXEC)); a = a + 4096UL; }
    ok = ok & m.map(ALIAS32, DATA, (unsigned int)MMU_WRITE);
    if (!ok || m.activate() != 1) { say("mmu: x86 32-bit setup failed\n"); finish(0); }
    qemu_paging_on();
    unsafe {
        unsigned int* alias = (unsigned int*)ALIAS32;
        if (alias[0] != 0xC0FFEEu) { say("mmu: x86 32-bit alias read wrong\n"); finish(0); }
        alias[1] = 0x77u;
        if (((unsigned int*)DATA)[1] != 0x77u) { say("mmu: x86 32-bit alias write missed\n"); finish(0); }
    }
    say("mmu: x86 32-bit paging reads and writes through the alias\n");
    qemu_paging_off();

    // 4-level paging, entered from the 64-bit stub.
    root = fresh_root_(POOL64);
    struct MmuContext<MmuX86_64> l = MmuContext::new(root, &g_frames, 0UL);
    a = IMAGE;
    while (a < POOL64 + 32UL * 4096UL) { ok = ok & l.map(a, a, (unsigned int)(MMU_WRITE | MMU_EXEC)); a = a + 4096UL; }
    ok = ok & l.map(DATA, DATA, (unsigned int)MMU_WRITE);
    ok = ok & l.map(ALIAS64, DATA, (unsigned int)MMU_WRITE);
    unsigned long got = 0UL;
    if (!ok || !l.walk(ALIAS64, &got) || got != DATA) { say("mmu: x86-64 table build failed\n"); finish(0); }
    say("mmu: x86-64 tables built, entering long mode\n");
    qemu_long_mode((unsigned int)root);
}
