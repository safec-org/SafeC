// QEMU RISC-V virt: Ns16550 UART + the native MMU format (Sv39 on RV64,
// Sv32 on RV32), with translation really on: S-mode reads a page through
// an alias mapping built by MmuContext.
#include <std/hal/uart.h>
#include <std/kernel/mmu.h>
#include <std/kernel/frame.h>

#define UART0    0x10000000UL
#define TESTDEV  0x00100000UL   // sifive_test: 0x5555 = pass, 0x3333 = fail
#define IMAGE    0x80000000UL   // image + stack: identity-mapped
#define POOL     0x80400000UL   // page-table frames
#define DATA     0x80500000UL   // physical page behind the alias
#define ALIAS    0x40000000UL

extern void qemu_setup_pmp();
extern void qemu_enter_s(fn void() entry);

static struct FrameAllocator g_frames;
static struct Uart<Ns16550> g_uart;

static void say(const &char s) { g_uart.write_str(s); }

static void say_hex(unsigned long v) {
    char buf[19];
    buf[0] = '0'; buf[1] = 'x';
    int i = 0;
    while (i < 16) {
        unsigned long nib = (v >> (unsigned long)((15 - i) * 4)) & 15UL;
        buf[2 + i] = (char)(nib < 10UL ? 48UL + nib : 87UL + nib);
        i = i + 1;
    }
    buf[18] = (char)0;
    say(&buf[0]);
}

static void finish(int ok) {
    say(ok ? "QEMU-TEST PASS\n" : "QEMU-TEST FAIL\n");
    unsafe { volatile_store((unsigned int*)TESTDEV, ok ? 0x5555u : 0x13333u); }
    while (1) { }
}

void qemu_trap(unsigned long cause, unsigned long epc, unsigned long tval) {
    say("trap: mcause="); say_hex(cause); say(" mepc="); say_hex(epc); say(" mtval="); say_hex(tval); say("\n");
    finish(0);
}

// Runs in S-mode with satp on.
void s_main() {
    unsafe {
        unsigned int* alias = (unsigned int*)ALIAS;
        unsigned int* data = (unsigned int*)DATA;
        if (alias[0] != 0xC0FFEEu) { say("mmu: alias read wrong\n"); finish(0); }
        alias[1] = 0x1234u;
        if (data[1] != 0x1234u) { say("mmu: alias write missed\n"); finish(0); }
    }
    say("mmu: S-mode reads and writes through the alias\n");
    finish(1);
}

void qemu_main() {
    unsafe { g_uart = Uart::new((&volatile unsigned int)UART0, 115200u, 3686400u); }
    say("uart: Ns16550 ok\n");

    g_frames.init_at(POOL, 64UL);
    long long rootFrame = g_frames.alloc();
    unsigned long root = g_frames.frame_addr((unsigned long)rootFrame);
    unsafe {
        unsigned char* r = (unsigned char*)root;
        unsigned long k = 0UL;
        while (k < 4096UL) { r[k] = (unsigned char)0; k = k + 1UL; }
    }
    struct MmuContext<MmuNative> m = MmuContext::new(root, &g_frames, 0UL);
    int ok = 1;
    unsigned long a = IMAGE;
    while (a < POOL + 64UL * 4096UL) {             // image, stack and the table pool
        ok = ok & m.map(a, a, (unsigned int)(MMU_WRITE | MMU_EXEC));
        a = a + 4096UL;
    }
    ok = ok & m.map(DATA, DATA, (unsigned int)MMU_WRITE);
    ok = ok & m.map(UART0, UART0, (unsigned int)(MMU_WRITE | MMU_DEVICE));
    ok = ok & m.map(TESTDEV, TESTDEV, (unsigned int)(MMU_WRITE | MMU_DEVICE));
    ok = ok & m.map(ALIAS, DATA, (unsigned int)MMU_WRITE);
    unsigned long got = 0UL;
    if (!ok || !m.walk(ALIAS + 4UL, &got) || got != DATA + 4UL) { say("mmu: table build failed\n"); finish(0); }
    unsafe { ((unsigned int*)DATA)[0] = 0xC0FFEEu; }
    say("mmu: tables built, switching to S-mode\n");
    qemu_setup_pmp();
    if (m.activate() != 1) { say("mmu: activate did not run\n"); finish(0); }
    qemu_enter_s(s_main);
}
