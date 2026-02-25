// baremetal.sc — bare-metal / freestanding programming demo
// Compile: ./build/safec examples/baremetal.sc --freestanding --emit-llvm -o /dev/null

// ── Section attribute: place data in named sections ─────────────────────────
section(".rodata") const long MAGIC = 0xDEADBEEF;
section(".bss") volatile int MMIO_STATUS;

// ── Pure function: no side effects, readonly ────────────────────────────────
pure int square(int x) {
    return x * x;
}

pure int add(int a, int b) {
    return a + b;
}

// ── Interrupt handler: void(void), ISR calling convention ───────────────────
section(".isr_vector") interrupt void timer_isr() {
    // In a real bare-metal system, this would acknowledge the interrupt
    // and update state. For demo, just a volatile store.
    unsafe {
        volatile_store((int*)0x40000000, 1);
    }
}

// ── Noreturn function: never returns ────────────────────────────────────────
noreturn void hang() {
    unsafe {
        asm volatile ("" ::: "memory");
    }
}

// ── Naked function: body is only asm, no prologue/epilogue ──────────────────
naked void _start() {
    asm volatile ("nop");
}

// ── Volatile MMIO access ────────────────────────────────────────────────────
void mmio_write(int* addr, int val) {
    unsafe {
        volatile_store(addr, val);
    }
}

int mmio_read(int* addr) {
    unsafe {
        return volatile_load(addr);
    }
}

// ── Inline assembly usage ───────────────────────────────────────────────────
void memory_barrier() {
    unsafe {
        asm volatile ("" ::: "memory");
    }
}
