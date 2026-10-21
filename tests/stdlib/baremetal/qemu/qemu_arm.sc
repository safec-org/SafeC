// QEMU ARM virt (AArch64 or ARMv7-A): PL011 UART + the native MMU format
// (MmuAArch64 / MmuArmV7) with the MMU really on: the program keeps running
// from identity-mapped RAM and reads a page through an alias mapping.
#include <std/hal/uart.h>
#include <std/kernel/mmu.h>
#include <std/kernel/frame.h>

#define UART0    0x09000000UL
#define IMAGE    0x40000000UL   // image + stack: identity-mapped
#define POOL     0x40400000UL   // page-table frames (16 KiB aligned for ARMv7's L1)
#define DATA     0x40500000UL
#define ALIAS    0x70000000UL

extern void qemu_exit(int failed);
extern void qemu_mmu_on();
extern void qemu_mmu_enable();

static struct FrameAllocator g_frames;
static struct Uart<Pl011> g_uart;

static void say(const &char s) { g_uart.write_str(s); }
static void say_hex(unsigned long v) {
    char buf[19];
    buf[0] = '0'; buf[1] = 'x';
    int n = (int)sizeof(unsigned long) * 2;
    int i = 0;
    while (i < n) {
        unsigned long nib = (v >> (unsigned long)((n - 1 - i) * 4)) & 15UL;
        buf[2 + i] = (char)(nib < 10UL ? 48UL + nib : 87UL + nib);
        i = i + 1;
    }
    buf[2 + n] = (char)0;
    say(&buf[0]);
}
static void finish(int ok) {
    say(ok ? "QEMU-TEST PASS\n" : "QEMU-TEST FAIL\n");
    qemu_exit(ok ? 0 : 1);
}

void qemu_trap(unsigned long syndrome, unsigned long pc, unsigned long addr) {
    say("trap: syndrome="); say_hex(syndrome); say(" pc="); say_hex(pc); say(" addr="); say_hex(addr); say("\n");
    finish(0);
}

void qemu_main() {
    unsafe { g_uart = Uart::new((&volatile unsigned int)UART0, 115200u, 24000000u); }
    say("uart: PL011 ok\n");

    g_frames.init_at(POOL, 64UL);
    g_frames.mark_range(0UL, 4UL);                  // 16 KiB root (ARMv7 L1 size)
    unsigned long root = g_frames.frame_addr(0UL);
    unsafe {
        unsigned char* r = (unsigned char*)root;
        unsigned long k = 0UL;
        while (k < 16384UL) { r[k] = (unsigned char)0; k = k + 1UL; }
    }
    struct MmuContext<MmuNative> m = MmuContext::new(root, &g_frames, 0UL);
    int ok = 1;
    unsigned long a = IMAGE;
    while (a < POOL + 64UL * 4096UL) {
        ok = ok & m.map(a, a, (unsigned int)(MMU_WRITE | MMU_EXEC));
        a = a + 4096UL;
    }
    ok = ok & m.map(DATA, DATA, (unsigned int)MMU_WRITE);
    ok = ok & m.map(UART0, UART0, (unsigned int)(MMU_WRITE | MMU_DEVICE));
    ok = ok & m.map(ALIAS, DATA, (unsigned int)MMU_WRITE);
    unsigned long got = 0UL;
    if (!ok || !m.walk(ALIAS + 4UL, &got) || got != DATA + 4UL) { say("mmu: table build failed\n"); finish(0); }
    unsafe { ((unsigned int*)DATA)[0] = 0xC0FFEEu; }
    say("mmu: tables built, turning the MMU on\n");
    qemu_mmu_on();
    if (m.activate() != 1) { say("mmu: activate did not run\n"); finish(0); }
    qemu_mmu_enable();
    say("mmu: on, still running from identity-mapped RAM\n");
    unsafe {
        unsigned int* alias = (unsigned int*)ALIAS;
        if (alias[0] != 0xC0FFEEu) { say("mmu: alias read wrong\n"); finish(0); }
        alias[1] = 0x1234u;
        if (((unsigned int*)DATA)[1] != 0x1234u) { say("mmu: alias write missed\n"); finish(0); }
    }
    say("mmu: reads and writes through the alias\n");
    if (m.tlb_flush_page(ALIAS) != 1 || m.tlb_flush_all() != 1) { say("mmu: flush did not run\n"); finish(0); }
    finish(1);
}
