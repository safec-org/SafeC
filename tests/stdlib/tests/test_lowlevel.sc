// Freestanding building blocks, exercised hosted: std::interrupt (bitfield,
// mmio, isr, vector_table, clock), std::kernel (frame, paging, process,
// scheduler, syscall, ipc), std::hal drivers against RAM-backed registers,
// std::debug (coverage, perf), dma, node_pool, and allocator edge cases.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/dma.h>
#include <std/alloc/tlsf.h>
#include <std/alloc/pool.h>
#include <std/alloc/slab.h>
#include <std/collections/node_pool.h>
#include <std/interrupt/bitfield.h>
#include <std/interrupt/mmio.h>
#include <std/interrupt/isr.h>
#include <std/interrupt/vector_table.h>
#include <std/interrupt/clock.h>
#include <std/kernel/frame.h>
#include <std/kernel/paging.h>
#include <std/kernel/process.h>
#include <std/kernel/scheduler.h>
#include <std/kernel/syscall.h>
#include <std/kernel/ipc.h>
#include <std/hal/gpio.h>
#include <std/hal/uart.h>
#include <std/hal/spi.h>
#include <std/hal/i2c.h>
#include <std/hal/timer.h>
#include <std/hal/watchdog.h>
#include <std/debug/coverage.h>
#include <std/debug/perf.h>

void t_bitfield() {
    ASSERT_EQ(bf_mask32(4, 7), 0xF0);
    test_assert_eq_u(bf_mask32(0, 31), 0xFFFFFFFFULL, "full-width mask", __FILE__, __LINE__);
    test_assert_eq_u(bf_extract32(0xDEADBEEFu, 0, 31), 0xDEADBEEFULL, "full-width extract", __FILE__, __LINE__);
    ASSERT_EQ(bf_extract32(0xABCDu, 4, 11), 0xBC);
    ASSERT_EQ(bf_insert32(0u, 8, 15, 0x1FFu), 0xFF00);    // field truncated to its width
    ASSERT_EQ(bf_set32(0u, 0, 3), 0xF);
    ASSERT_EQ(bf_clear32(0xFFu, 0, 3), 0xF0);
    ASSERT_EQ(bf_toggle32(0x0Fu, 0, 7), 0xF0);
    ASSERT_TRUE(bf_test32(0x10u, 4));
    ASSERT_FALSE(bf_test32(0x10u, 3));
    ASSERT_EQ(bf_popcount32(0xF0F0u), 8);
}

void t_mmio() {
    unsigned int regs[2];
    regs[0] = 0x11223344u;
    regs[1] = 0x55667788u;
    unsafe {
        &volatile unsigned int r0 = (&volatile unsigned int)&regs[0];
        ASSERT_EQ(mmio_read32(r0), 0x11223344);
        mmio_write32(r0, 0xA0B0C0D0u);
        mmio_set_bits32(r0, 0xFu);
        mmio_clear_bits32(r0, 0xF0000000u);
        ASSERT_EQ(regs[0], 0x00B0C0DF);
        mmio_write_field32(r0, 8, 15, 0x5Au);
        ASSERT_EQ(mmio_read_field32(r0, 8, 15), 0x5A);
        mmio_write_field32(r0, 0, 31, 0x12345678u);       // full-width field
        ASSERT_EQ(regs[0], 0x12345678);
        test_assert_eq_u(mmio_read_field32(r0, 0, 31), 0x12345678ULL, "full-width field", __FILE__, __LINE__);

        // 8/16-bit registers: only that register changes (little-endian host).
        regs[0] = 0x11223344u;
        &volatile unsigned char b1 = (&volatile unsigned char)((unsigned char*)&regs[0] + 1);
        ASSERT_EQ(mmio_read8(b1), 0x33);
        mmio_write8(b1, 0xAAu);
        ASSERT_EQ(regs[0], 0x1122AA44);
        ASSERT_EQ(regs[1], 0x55667788);
        &volatile unsigned short h1 = (&volatile unsigned short)((unsigned short*)&regs[0] + 1);
        ASSERT_EQ(mmio_read16(h1), 0x1122);
        mmio_write16(h1, 0xBEEFu);
        ASSERT_EQ(regs[0], 0xBEEFAA44);
        ASSERT_EQ(regs[1], 0x55667788);

        struct MmioReg reg = mmio_reg((&volatile unsigned int)&regs[1]);
        reg.write32(0u);
        reg.set_bits(0x30u);
        reg.write_field(0, 3, 0x9u);
        ASSERT_EQ(reg.read32(), 0x39);
        ASSERT_EQ(reg.read_field(4, 5), 3);
        reg.clear_bits(0x10u);
        ASSERT_EQ(regs[1], 0x29);
    }
}

static int gIrqHits = 0;
static int gDefaultHits = 0;
static void irq_a() { gIrqHits = gIrqHits + 1; }
static void irq_b() { gIrqHits = gIrqHits + 10; }
static void irq_default() { gDefaultHits = gDefaultHits + 1; }

void t_isr_vtable() {
    struct IsrTable t = isr_init();
    ASSERT_EQ(t.count, 0);
    ASSERT_TRUE(t.register_(5, irq_a));
    ASSERT_FALSE(t.register_(ISR_MAX, irq_a));
    ASSERT_FALSE(t.register_(-1, irq_a));
    ASSERT_EQ(t.count, 1);
    ASSERT_TRUE(t.dispatch(5));
    ASSERT_FALSE(t.dispatch(6));
    ASSERT_EQ(gIrqHits, 1);
    t.unregister_(5);
    ASSERT_EQ(t.count, 0);
    ASSERT_FALSE(t.dispatch(5));
    irq_disable();   // hosted: compiler barriers only
    irq_enable();

    gIrqHits = 0;
    vtable_init();
    vtable.set_default(irq_default);   // reaches slots vtable_init filled
    vtable.dispatch(7);
    ASSERT_EQ(gDefaultHits, 1);
    vtable.install(3, irq_b);
    ASSERT_TRUE(vtable.get(3) == irq_b);
    vtable.dispatch(3);
    ASSERT_EQ(gIrqHits, 10);
    vtable.remove(3);
    vtable.dispatch(3);
    ASSERT_EQ(gDefaultHits, 2);
    vtable.dispatch(VTABLE_MAX_VECTORS + 5);   // out of range: the default, not a hang
    ASSERT_EQ(gDefaultHits, 3);
    ASSERT_TRUE(vtable.get(-1) == (IrqHandler)0);
}

void t_clock() {
    unsigned int regs[4];
    regs[0] = 0u; regs[1] = 0u; regs[2] = 0u;
    regs[3] = (2u << 2) | (1u << 25);   // status: PLL selected + PLL locked
    unsafe {
        struct ClockConfig c = clock_init((&volatile unsigned int)&regs[0]);
        ASSERT_EQ(c.get_freq(), 8000000);
        c.set_source(CLK_SRC_PLL);
        ASSERT_EQ(regs[0] & 3u, 2);
        ASSERT_TRUE(c.configure_pll(9u, 2u));
        ASSERT_EQ(regs[1], 9u | (2u << 8));
        ASSERT_EQ(c.get_freq(), 36000000);
        ASSERT_FALSE(c.configure_pll(1u, 0u));
        c.set_ahb_div(2u);
        c.set_apb1_div(4u);
        c.set_apb2_div(3u);
        ASSERT_EQ(regs[2], (2u << 4) | (4u << 8) | (3u << 11));
        ASSERT_EQ(c.get_freq(), 18000000);
    }
}

void t_frames_paging() {
    unsafe {
        ?&heap FrameAllocator fq = (?&heap FrameAllocator)alloc_raw_(sizeof(struct FrameAllocator));
        if (fq.is_null()) { ASSERT_TRUE(0); return; }
        &FrameAllocator f = std::expect_ref(fq, "frames");
        f.init(10UL);                    // not a multiple of 32
        int i = 0;
        int ok = 1;
        while (i < 10) { if (f.alloc() != (long long)i) ok = 0; i = i + 1; }
        ASSERT_TRUE(ok);
        ASSERT_EQ(f.alloc(), -1);
        ASSERT_EQ(f.free_count(), 0);
        f.free(4UL);
        f.free(4UL);                     // already free: no double count
        ASSERT_EQ(f.free_count(), 1);
        ASSERT_FALSE(f.is_used(4UL));
        ASSERT_EQ(f.alloc(), 4);
        f.init(100UL);
        f.mark_range(0UL, 40UL);
        ASSERT_EQ(f.free_count(), 60);
        ASSERT_TRUE(f.is_used(39UL));
        ASSERT_EQ(f.alloc(), 40);
        f.init(1000000UL);               // more than the bitmap tracks: clamped
        ASSERT_EQ(f.total_frames, (unsigned long)FRAME_BITMAP_SIZE * 32UL);
        dealloc_raw_((void*)f);

        ?&heap PageTable tq = (?&heap PageTable)alloc_raw_(sizeof(struct PageTable));
        if (tq.is_null()) { ASSERT_TRUE(0); return; }
        &PageTable t = std::expect_ref(tq, "page table");
        t.init();
        ASSERT_FALSE(t.is_present(3));
        t.map(3, 0x12345678UL, (unsigned int)(PAGE_PRESENT | PAGE_WRITABLE));
        ASSERT_TRUE(t.is_present(3));
        ASSERT_EQ(t.get_phys(3), 0x12345000);
        ASSERT_EQ(t.get_flags(3), PAGE_PRESENT | PAGE_WRITABLE);
        t.set_flags(3, (unsigned int)(PAGE_PRESENT | PAGE_USER));
        ASSERT_EQ(t.get_phys(3), 0x12345000);
        ASSERT_EQ(t.get_flags(3), PAGE_PRESENT | PAGE_USER);
        t.unmap(3);
        ASSERT_EQ(t.get_phys(3), 0);
        ASSERT_FALSE(t.is_present(PAGE_TABLE_SIZE));
        dealloc_raw_((void*)t);
    }
}

void t_scheduler() {
    unsafe {
        ?&heap Scheduler sq = (?&heap Scheduler)alloc_raw_(sizeof(struct Scheduler));
        if (sq.is_null()) { ASSERT_TRUE(0); return; }
        &Scheduler s = std::expect_ref(sq, "sched");
        *s = sched_init();
        struct PCB hi = pcb_init(1, 0x1000UL, 0x8000UL, 0UL);
        hi.set_priority(10);
        struct PCB lo = pcb_init(2, 0x2000UL, 0x9000UL, 0UL);
        lo.set_priority(1);
        ASSERT_EQ(s.spawn_proc(&hi), 0);
        ASSERT_EQ(s.spawn_proc(&lo), 1);
        ASSERT_EQ(s.ready_count(), 2);
        ASSERT_EQ(s.next(), 0);
        ASSERT_EQ(s.procs[0].state, PROC_RUNNING);
        ASSERT_EQ(s.yield(), 0);        // the higher priority keeps winning
        struct PCB hi2 = pcb_init(3, 0x3000UL, 0xA000UL, 0UL);
        hi2.set_priority(10);
        ASSERT_EQ(s.spawn_proc(&hi2), 2);
        ASSERT_EQ(s.yield(), 2);        // equal priorities round-robin
        ASSERT_EQ(s.yield(), 0);
        s.block_current();
        ASSERT_EQ(s.next(), 2);
        s.block_current();
        ASSERT_EQ(s.next(), 1);         // only the low-priority one is ready
        s.unblock(0);
        ASSERT_EQ(s.yield(), 0);
        s.procs[1].exit(7);
        ASSERT_EQ(s.procs[1].state, PROC_ZOMBIE);
        ASSERT_EQ(s.procs[1].exit_code, 7);
        s.remove(1);                    // before current: current shifts down
        ASSERT_EQ(s.count, 2);
        ASSERT_EQ(s.current, 0);
        s.remove(0);                    // the current one
        ASSERT_EQ(s.current, -1);
        ASSERT_EQ(s.procs[0].pid, 3);
        s.procs[0].save_context(0x1234UL, 0x5678UL);
        ASSERT_EQ(s.procs[0].stack_ptr, 0x1234);
        dealloc_raw_((void*)s);
    }
}

static long long sys_add(long long a, long long b, long long c) { return a + b + c; }

void t_syscall_mailbox() {
    unsafe {
        struct SyscallTable t = syscall_init();
        ASSERT_TRUE(t.register_(4, sys_add));
        ASSERT_FALSE(t.register_(SYSCALL_MAX, sys_add));
        ASSERT_TRUE(t.is_registered(4));
        ASSERT_EQ(t.dispatch(4, 1LL, 2LL, 3LL), 6);
        ASSERT_EQ(t.dispatch(5, 1LL, 2LL, 3LL), -1);
        t.unregister_(4);
        ASSERT_FALSE(t.is_registered(4));
        ASSERT_EQ(t.count, 0);

        ?&heap Mailbox mq = (?&heap Mailbox)alloc_raw_(sizeof(struct Mailbox));
        if (mq.is_null()) { ASSERT_TRUE(0); return; }
        &Mailbox mb = std::expect_ref(mq, "mailbox");
        *mb = mailbox_init(9);
        ASSERT_FALSE(mb.has_msg());
        ASSERT_TRUE(mb.send(1, 42, (const ?&unsigned char)"hi", 3UL));
        ASSERT_TRUE(mb.send(2, 43, null, 0UL));
        ASSERT_EQ(mb.length(), 2);
        struct Message m;
        ASSERT_TRUE(mb.peek(&m));
        ASSERT_EQ(m.type, 42);
        ASSERT_TRUE(mb.recv(&m));
        ASSERT_EQ(m.sender_pid, 1);
        ASSERT_EQ(m.size, 3);
        ASSERT_STR_EQ((const char*)m.payload, "hi");
        ASSERT_TRUE(mb.recv(&m));
        ASSERT_EQ(m.size, 0);
        ASSERT_FALSE(mb.recv(&m));
        int i = 0;
        while (i < MAILBOX_CAPACITY) { mb.send(1, i, null, 0UL); i = i + 1; }
        ASSERT_FALSE(mb.send(1, 0, null, 0UL));     // full
        mb.clear();
        ASSERT_EQ(mb.length(), 0);
        dealloc_raw_((void*)mb);
    }
}

// HAL drivers against a RAM-backed register block (the MMIO accessors are
// plain volatile loads/stores, so ordinary memory stands in for a device).
void t_hal() {
    unsigned int r[8];
    int i = 0;
    while (i < 8) { r[i] = 0u; i = i + 1; }
    unsafe {
        &volatile unsigned int base = (&volatile unsigned int)&r[0];

        struct GpioPin p = gpio_init(base, 3, GPIO_OUTPUT);
        ASSERT_EQ(r[0], 8);
        p.write(1);
        ASSERT_EQ(r[1], 8);
        p.toggle();
        ASSERT_EQ(r[1], 0);
        r[2] = 8u;
        ASSERT_EQ(p.read(), 1);
        p.set_pull(GPIO_PULL_UP);
        ASSERT_EQ(r[3], 8);
        p.set_direction(GPIO_INPUT);
        ASSERT_EQ(r[0], 0);

        r[1] = 2u;                                     // TX ready
        struct Uart u = uart_init(base, 115200u);
        ASSERT_EQ(r[2], 115200);
        ASSERT_TRUE(u.tx_ready());
        u.write_str("ok");
        ASSERT_EQ(r[0], 'k');
        r[1] = 3u;
        ASSERT_EQ(u.read_byte(), 'k');

        r[0] = 0u; r[1] = 2u; r[2] = 0u; r[3] = 0u;   // TX ready, not busy
        struct SpiDevice s = spi_init(base, SPI_MODE_3);
        ASSERT_EQ(r[2], 0x83);
        ASSERT_EQ(s.transfer((unsigned char)0x5A), 0x5A);   // the data register echoes
        unsigned char tx[3];
        unsigned char rx[3];
        tx[0] = (unsigned char)1; tx[1] = (unsigned char)2; tx[2] = (unsigned char)3;
        s.write(&tx[0], (?&unsigned char)&rx[0], 3UL);
        ASSERT_EQ(rx[2], 3);
        s.write(&tx[0], null, 3UL);                    // rx is optional
        s.read(&rx[0], 2UL);
        ASSERT_EQ(rx[0], 0xFF);
        s.cs_assert();
        ASSERT_EQ(r[3], 0);
        s.cs_deassert();
        ASSERT_EQ(r[3], 1);

        r[0] = 0u; r[1] = 2u; r[2] = 0u; r[3] = 0u; r[4] = 0u;   // idle + ACK
        struct I2cBus bus = i2c_init(base, 400000u);
        ASSERT_EQ(r[4], 400000);
        unsigned char data[2];
        data[0] = (unsigned char)0x10; data[1] = (unsigned char)0x20;
        ASSERT_EQ(bus.write((unsigned char)0x50, &data[0], 2UL), I2C_OK);
        ASSERT_EQ(r[3], 0x50 << 1);
        ASSERT_EQ(r[2], 2);                            // ended with STOP
        ASSERT_TRUE(bus.probe((unsigned char)0x50));
        r[1] = 0u;                                     // NACK
        ASSERT_EQ(bus.write((unsigned char)0x51, &data[0], 2UL), I2C_NACK);
        ASSERT_EQ(r[2], 2);                            // STOP still sent on error
        r[1] = 4u;                                     // bus error
        ASSERT_EQ(bus.read((unsigned char)0x51, &data[0], 2UL), I2C_ERROR);
        ASSERT_EQ(r[2], 2);

        i = 0;
        while (i < 8) { r[i] = 0u; i = i + 1; }
        struct Timer tm = timer_init(base, 99u);
        ASSERT_EQ(r[1], 99);
        tm.set_period(1000u);
        ASSERT_EQ(r[2], 1000);
        tm.start();
        ASSERT_EQ(r[0] & 1u, 1);
        r[3] = 77u;
        ASSERT_EQ(tm.read(), 77);
        r[4] = 1u;
        ASSERT_TRUE(tm.flag_set());
        tm.clear_flag();
        ASSERT_FALSE(tm.flag_set());
        tm.stop();
        ASSERT_EQ(r[0] & 1u, 0);

        struct Watchdog w = watchdog_init(base, 500u);
        ASSERT_EQ(r[1], 500);
        w.enable();
        ASSERT_EQ(r[0] & 1u, 1);
        w.feed();
        ASSERT_EQ(r[2], 0x6969);
        r[3] = 1u;
        ASSERT_TRUE(w.caused_reset());
    }
}

void t_debug() {
    unsafe {
        coverage_init();
        int a = coverage.register_site("x.sc", 10);
        int b = coverage.register_site("x.sc", 20);
        ASSERT_EQ(a, 0);
        ASSERT_EQ(b, 1);
        coverage.hit(a);
        coverage.hit(a);
        coverage.hit(99);                      // ignored
        ASSERT_EQ(coverage.get(a), 2);
        ASSERT_EQ(coverage.covered_count(), 1);
        ASSERT_EQ(coverage.coverage_pct(), 50);
        coverage.reset();
        ASSERT_EQ(coverage.covered_count(), 0);
    }
    unsigned long long t0 = perf_read_ns();
    struct PerfCounter c = perf_counter_init();
    c.start();
    volatile int spin = 0;
    while (spin < 100000) { spin = spin + 1; }
    c.stop();
    ASSERT_TRUE(perf_read_ns() > t0);
    ASSERT_TRUE(c.ticks() == c.end_val - c.start_val);
    if (c.freq_hz != 0ULL) { ASSERT_TRUE(c.ns() > 0ULL); }
}

void t_dma() {
    unsigned char buf[200];
    unsafe {
        struct DmaBuffer d = dma_buf_wrap(&buf[3], 0UL, 130UL);   // unaligned start
        ASSERT_EQ(d.phys, (unsigned long)&buf[3]);
        ASSERT_TRUE(d.cpu_accessible());
        d.to_device();
        ASSERT_FALSE(d.cpu_accessible());
        d.to_cpu();
        ASSERT_TRUE(d.cpu_accessible());
        ASSERT_EQ(d.phys_at(10UL), (unsigned long)&buf[13]);
        struct DmaBuffer e = dma_buf_wrap(&buf[0], 0x8000UL, 64UL);
        ASSERT_EQ(e.phys_at(4UL), 0x8004);
    }
    struct DmaSgList sg;
    sg.clear();
    sg.add(0x1000UL, 100UL);
    sg.add(0x2000UL, 50UL);
    ASSERT_EQ(sg.total_len(), 150);
    int i = 0;
    while (i < DMA_SG_MAX + 4) { sg.add(0UL, 1UL); i = i + 1; }
    ASSERT_EQ(sg.count, DMA_SG_MAX);
}

struct PoolNode {
    long long value;
    ?&heap PoolNode next;
};

void t_node_pool() {
    struct NodePool<PoolNode> p = node_pool_new();
    ?&heap PoolNode first = p.take();
    ASSERT_FALSE(first.is_null());
    int i = 0;
    int ok = 1;
    while (i < 100) {
        match (p.take()) {
            case null: { ok = 0; }
            case some(n): { n.value = (long long)i; }
        }
        i = i + 1;
    }
    ASSERT_TRUE(ok);
    match (first) {
        case null: { }
        case some(n): {
            p.give(n);
            ?&heap PoolNode again = p.take();      // recycled first
            unsafe { ASSERT_TRUE((void*)again == (void*)n); }
        }
    }
    p.free_all();
    ASSERT_TRUE(p.chunks.is_null());
    ASSERT_FALSE(p.take().is_null());
    p.free_all();
}

void t_alloc_edges() {
    unsafe {
        // TLSF: a block that exactly fits the request's size class is found
        // (the search used to skip that class and report OOM).
        struct TlsfAllocator t = tlsf_new(4096UL);
        ?&heap unsigned char blocks[64];
        int n = 0;
        while (n < 64) {
            blocks[n] = (?&heap unsigned char)t.alloc(64UL);
            if (blocks[n].is_null()) break;
            n = n + 1;
        }
        ASSERT_TRUE(n > 4);
        ASSERT_TRUE((void*)t.alloc(64UL) == (void*)0);      // full
        match (blocks[2]) {                                  // neighbours stay used
            case null: { ASSERT_TRUE(0); }
            case some(blk): { t.free((&heap void)blk); }
        }
        void* again = (void*)t.alloc(64UL);
        ASSERT_TRUE(again != (void*)0);
        ASSERT_TRUE(again == (void*)blocks[2]);
        t.destroy();

        // Pool over a buffer too small for one header: empty, nothing written.
        unsigned char tiny[32];
        int i = 0;
        while (i < 32) { tiny[i] = (unsigned char)0xEE; i = i + 1; }
        struct PoolAllocator p = pool_init(&tiny[0], 10UL);
        ASSERT_TRUE((void*)p.alloc(1UL) == (void*)0);
        ASSERT_EQ(p.available(), 0);
        int untouched = 1;
        i = 0;
        while (i < 32) { if (tiny[i] != (unsigned char)0xEE) untouched = 0; i = i + 1; }
        ASSERT_TRUE(untouched);

        // Slab with zero slots.
        unsigned char sbuf[16];
        struct SlabAllocator s = slab_init(&sbuf[0], 16UL, 0UL);
        ASSERT_TRUE((void*)s.alloc() == (void*)0);
        ASSERT_EQ(s.available(), 0);

        // Sizes whose header would wrap around fail instead of returning a tiny block.
        ASSERT_TRUE(alloc_raw_(~0UL - 4UL) == (void*)0);
        ASSERT_TRUE(alloc_zeroed_raw_(~0UL) == (void*)0);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("bitfield", t_bitfield);
        t.add("mmio", t_mmio);
        t.add("isr/vector_table", t_isr_vtable);
        t.add("clock", t_clock);
        t.add("frame/paging", t_frames_paging);
        t.add("process/scheduler", t_scheduler);
        t.add("syscall/mailbox", t_syscall_mailbox);
        t.add("hal drivers", t_hal);
        t.add("coverage/perf", t_debug);
        t.add("dma", t_dma);
        t.add("node_pool", t_node_pool);
        t.add("allocator edge cases", t_alloc_edges);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
