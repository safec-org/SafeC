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
#include <std/kernel/mmu.h>
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

        struct MmioReg reg = MmioReg::at((&volatile unsigned int)&regs[1]);
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
    struct IsrTable t = IsrTable::new();
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
        struct ClockConfig c = ClockConfig::new((&volatile unsigned int)&regs[0]);
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
        *s = Scheduler::new();
        struct PCB hi = PCB::new(1, 0x1000UL, 0x8000UL, 0UL);
        hi.set_priority(10);
        struct PCB lo = PCB::new(2, 0x2000UL, 0x9000UL, 0UL);
        lo.set_priority(1);
        ASSERT_EQ(s.spawn_proc(&hi), 0);
        ASSERT_EQ(s.spawn_proc(&lo), 1);
        ASSERT_EQ(s.ready_count(), 2);
        ASSERT_EQ(s.next(), 0);
        ASSERT_EQ(s.procs[0].state, PROC_RUNNING);
        ASSERT_EQ(s.yield(), 0);        // the higher priority keeps winning
        struct PCB hi2 = PCB::new(3, 0x3000UL, 0xA000UL, 0UL);
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

// ── MMU: real page-table formats, built in a RAM arena ───────────────────────
// Physical frame N lives at g_mmu_base + N * 4096 (phys_offset), so the
// tables the code writes are the exact bytes a CPU would walk.
#define MMU_TEST_FRAMES 64UL
static struct FrameAllocator g_mmu_frames;
static unsigned long g_mmu_base = 0UL;

static void mmu_reset_() {
    unsafe {
        if (g_mmu_base == 0UL) {
            unsigned long raw = (unsigned long)alloc_zeroed_raw_(MMU_TEST_FRAMES * 4096UL + 16384UL);
            g_mmu_base = (raw + 16383UL) & ~16383UL;        // 16 KiB aligned (ARMv7 L1)
        }
        safe_memset_raw_((void*)g_mmu_base, 0, MMU_TEST_FRAMES * 4096UL);
    }
    g_mmu_frames.init(MMU_TEST_FRAMES);
    g_mmu_frames.mark_range(0UL, 4UL);                    // frames 0-3: the root table
}

// Map, walk, re-walk through a second page in the same tables, unmap.
// Returns the number of frames the first map took for new tables.
generic<F>
static long long mmu_roundtrip_(&MmuContext<F> m, unsigned long virt, unsigned long phys) {
    unsigned long before = m.frames.free_count();
    if (!m.map(virt, phys, (unsigned int)(MMU_WRITE | MMU_USER))) { return -1LL; }
    unsigned long used = before - m.frames.free_count();
    unsigned long got = 0UL;
    if (!m.walk(virt + 0x123UL, &got) || got != phys + 0x123UL) { return -2LL; }
    if (!m.map(virt + 4096UL, phys + 0x10000UL, (unsigned int)MMU_EXEC)) { return -3LL; }
    if (m.frames.free_count() != before - used) { return -4LL; }          // tables shared
    if (!m.walk(virt + 4096UL, &got) || got != phys + 0x10000UL) { return -5LL; }
    if (m.walk(virt + 0x200000UL, &got)) { return -6LL; }                 // never mapped
    m.unmap(virt);
    if (m.walk(virt, &got)) { return -7LL; }
    if (!m.walk(virt + 4096UL, &got)) { return -8LL; }
    if (m.activate() != 0 || m.tlb_flush_all() != 0 || m.tlb_flush_page(virt) != 0) { return -9LL; }
    m.map(virt, phys, (unsigned int)(MMU_WRITE | MMU_USER));
    return (long long)used;
}

void t_mmu() {
    unsafe {
        unsigned long long e = 0ULL;

        mmu_reset_();
        struct MmuContext<MmuX86_64> x = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&x, 0x00007F1234567000UL, 0x0000000ABCDE0000UL), 3);   // PDPT, PD, PT
        ASSERT_EQ(x.entry(0x00007F1234567000UL, &e), 4);
        test_assert_eq_u(e, 0x0000000ABCDE0000ULL | 0x67ULL | 0x8000000000000000ULL,
                         "x86-64 leaf: P RW US A D, NX", __FILE__, __LINE__);
        x.entry(0x00007F1234568000UL, &e);
        test_assert_eq_u(e, 0x0000000ABCDF0000ULL | 0x21ULL, "x86-64 exec leaf: P A, no NX", __FILE__, __LINE__);
        x.map(0x10000UL, 0xFEE00000UL, (unsigned int)(MMU_WRITE | MMU_DEVICE | MMU_EXEC));
        x.entry(0x10000UL, &e);
        test_assert_eq_u(e & 0x18ULL, 0x18ULL, "x86-64 device: PCD PWT", __FILE__, __LINE__);
        // A 2 MiB page written by hand: walk follows it, map refuses to split it.
        unsigned long long* pml4 = (unsigned long long*)g_mmu_base;
        unsigned long long* pdpt = (unsigned long long*)(g_mmu_base + (unsigned long)(pml4[0] & 0xFFFFFFFFF000ULL));
        unsigned long long* pd = (unsigned long long*)(g_mmu_base + (unsigned long)(pdpt[0] & 0xFFFFFFFFF000ULL));
        pd[5] = 0x40000000ULL | 0x83ULL;                 // PS | RW | P at 10 MiB
        unsigned long got = 0UL;
        ASSERT_TRUE(x.walk(0xA12345UL, &got));
        ASSERT_EQ(got, 0x40012345UL);
        ASSERT_FALSE(x.map(0xA01000UL, 0x1000UL, 0u));

        mmu_reset_();
        struct MmuContext<MmuX86_32> x32 = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&x32, 0xC0401000UL, 0x00800000UL), 1);
        x32.entry(0xC0401000UL, &e);
        test_assert_eq_u(e, 0x00800000ULL | 0x67ULL, "x86 32-bit leaf", __FILE__, __LINE__);
        unsigned int* pde = (unsigned int*)g_mmu_base;
        ASSERT_EQ(pde[0xC0401000UL >> 22], (unsigned int)(g_mmu_frames.is_used(4UL) ? 4UL * 4096UL : 0UL) | 7u);

        mmu_reset_();
        struct MmuContext<MmuSv39> sv = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&sv, 0x0000003F80201000UL, 0x0000000080400000UL), 2);
        sv.entry(0x0000003F80201000UL, &e);
        test_assert_eq_u(e, ((0x80400000ULL >> 12ULL) << 10ULL) | 0xD7ULL, "Sv39 leaf: V R W U A D", __FILE__, __LINE__);
        sv.entry(0x0000003F80202000UL, &e);
        test_assert_eq_u(e & 0xFFULL, 0xCBULL, "Sv39 exec leaf: V R X A D", __FILE__, __LINE__);
        // A 1 GiB superpage written by hand at root index 2.
        unsigned long long* svroot = (unsigned long long*)g_mmu_base;
        svroot[2] = ((0xC0000000ULL >> 12ULL) << 10ULL) | 0xCFULL;
        ASSERT_TRUE(sv.walk(0x80054321UL, &got));
        ASSERT_EQ(got, 0xC0054321UL);

        mmu_reset_();
        struct MmuContext<MmuSv32> s32 = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&s32, 0x80401000UL, 0x81000000UL), 1);
        s32.entry(0x80401000UL, &e);
        test_assert_eq_u(e, ((0x81000000ULL >> 12ULL) << 10ULL) | 0xD7ULL, "Sv32 leaf", __FILE__, __LINE__);

        mmu_reset_();
        struct MmuContext<MmuAArch64> a64 = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&a64, 0x0000FFFF40201000UL, 0x0000000040000000UL), 3);
        a64.entry(0x0000FFFF40201000UL, &e);
        test_assert_eq_u(e, 0x40000000ULL | 0x3ULL | 0x400ULL | 0x300ULL | 0x40ULL | 0x0060000000000000ULL,
                         "AArch64 page: AF ISH EL0 RW, PXN UXN", __FILE__, __LINE__);
        a64.entry(0x0000FFFF40202000UL, &e);
        test_assert_eq_u(e & 0x00600000000000C0ULL, 0x0040000000000080ULL, "AArch64 kernel exec: RO, UXN only", __FILE__, __LINE__);
        a64.map(0x1000UL, 0x09000000UL, (unsigned int)(MMU_WRITE | MMU_DEVICE));
        a64.entry(0x1000UL, &e);
        test_assert_eq_u(e & 0x1CULL, 0x4ULL, "AArch64 device: AttrIndx 1", __FILE__, __LINE__);

        mmu_reset_();
        struct MmuContext<MmuArmV7> v7 = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_EQ(mmu_roundtrip_(&v7, 0xC0101000UL, 0x60000000UL), 1);
        v7.entry(0xC0101000UL, &e);
        test_assert_eq_u(e, 0x60000000ULL | 0x2ULL | 0x1ULL | 0x44CULL | 0x30ULL, "ARMv7 small page: XN, WB S, AP=011", __FILE__, __LINE__);
        v7.entry(0xC0102000UL, &e);
        test_assert_eq_u(e & 0x231ULL, 0x210ULL, "ARMv7 kernel exec: AP=101, no XN", __FILE__, __LINE__);
        unsigned int* l1 = (unsigned int*)g_mmu_base;
        l1[0x800] = 0x90000000u | 2u;                     // 1 MiB section at 2 GiB
        ASSERT_TRUE(v7.walk(0x80012345UL, &got));
        ASSERT_EQ(got, 0x90012345UL);

        // Out of frames: map fails cleanly.
        mmu_reset_();
        g_mmu_frames.mark_range(4UL, MMU_TEST_FRAMES - 4UL);
        struct MmuContext<MmuSv39> full = MmuContext::new(0UL, &g_mmu_frames, g_mmu_base);
        ASSERT_FALSE(full.map(0x1000UL, 0x2000UL, 0u));
    }
}

static long long sys_add(long long a, long long b, long long c) { return a + b + c; }

void t_syscall_mailbox() {
    unsafe {
        struct SyscallTable t = SyscallTable::new();
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
        *mb = Mailbox::new(9);
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
static void zero_regs_(&unsigned int r, int n) {
    int i = 0;
    while (i < n) { unsafe { r[i] = 0u; } i = i + 1; }
}

void t_hal() {
    unsigned int r[16];
    zero_regs_(&r[0], 16);
    unsafe {
        &volatile unsigned int base = (&volatile unsigned int)&r[0];

        // ── generic chips ──
        struct GpioPin<GpioGeneric> p = GpioPin::new(base, 3, GPIO_OUTPUT);
        ASSERT_EQ(r[0], 8);
        p.write(1);
        ASSERT_EQ(r[1], 8);
        p.toggle();
        ASSERT_EQ(r[1], 0);
        r[2] = 8u;
        ASSERT_EQ(p.read(), 1);
        p.set_pull(GPIO_PULL_UP);
        ASSERT_EQ(r[3], 8);
        ASSERT_EQ(r[4], 8);                            // pull-select: up
        p.set_pull(GPIO_PULL_DOWN);
        ASSERT_EQ(r[3], 8);
        ASSERT_EQ(r[4], 0);                            // ...now distinguishable from up
        p.set_pull(GPIO_PULL_NONE);
        ASSERT_EQ(r[3], 0);
        p.set_direction(GPIO_INPUT);
        ASSERT_EQ(r[0], 0);

        zero_regs_(&r[0], 16);
        r[1] = 2u;                                     // TX ready
        struct Uart<UartGeneric> u = Uart::new(base, 115200u, 0u);
        ASSERT_EQ(r[2], 115200);
        ASSERT_TRUE(u.tx_ready());
        u.write_str("ok");
        ASSERT_EQ(r[0], 'k');
        r[1] = 3u;
        ASSERT_EQ(u.read_byte(), 'k');

        zero_regs_(&r[0], 16);
        r[1] = 2u;                                     // TX ready, not busy
        struct SpiDevice<SpiGeneric> s = SpiDevice::new(base, SPI_MODE_3);
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

        zero_regs_(&r[0], 16);
        r[1] = 2u;                                     // idle + ACK
        struct I2cBus<I2cGeneric> bus = I2cBus::new(base, 400000u, 0u);
        ASSERT_EQ(r[4], 400000);
        unsigned char data[2];
        data[0] = (unsigned char)0x10; data[1] = (unsigned char)0x20;
        ASSERT_EQ(bus.write((unsigned char)0x50, &data[0], 2UL), I2C_OK);
        ASSERT_EQ(r[3], 0x50 << 1);
        ASSERT_EQ(r[2], 2);                            // ended with STOP
        ASSERT_TRUE(bus.probe((unsigned char)0x50));
        unsigned char rd[1];
        ASSERT_EQ(bus.write_read((unsigned char)0x50, &data[0], 1UL, &rd[0], 1UL), I2C_OK);
        r[1] = 0u;                                     // NACK
        ASSERT_EQ(bus.write((unsigned char)0x51, &data[0], 2UL), I2C_NACK);
        ASSERT_EQ(r[2], 2);                            // STOP still sent on error
        r[1] = 4u;                                     // bus error
        ASSERT_EQ(bus.read((unsigned char)0x51, &data[0], 2UL), I2C_ERROR);
        ASSERT_EQ(r[2], 2);

        zero_regs_(&r[0], 16);
        struct Timer<TimerGeneric> tm = Timer::new(base, 99u);
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

        zero_regs_(&r[0], 16);
        struct Watchdog<WatchdogGeneric> w = Watchdog::new(base, 500u);
        ASSERT_EQ(r[1], 500);
        w.enable();
        ASSERT_EQ(r[0] & 1u, 1);
        w.feed();
        ASSERT_EQ(r[2], 0x6969);
        r[3] = 1u;
        ASSERT_TRUE(w.caused_reset());
    }
}

// The real-chip layouts write the documented registers (offsets in bytes).
void t_hal_chips() {
    unsigned int r[16];
    unsafe {
        &volatile unsigned int base = (&volatile unsigned int)&r[0];

        // STM32 GPIO: MODER +0, PUPDR +0xC, IDR +0x10, ODR +0x14, BSRR +0x18
        zero_regs_(&r[0], 16);
        r[0] = 0xFFFFFFFFu;
        struct GpioPin<Stm32F4Gpio> led = GpioPin::new(base, 5, GPIO_OUTPUT);
        ASSERT_EQ(r[0], 0xFFFFF7FFu);                  // MODER5 = 01
        led.write(1);
        ASSERT_EQ(r[6], 1u << 5);                      // BSRR set
        led.write(0);
        ASSERT_EQ(r[6], 1u << 21);                     // BSRR reset
        r[5] = 1u << 5;                                // ODR high -> toggle resets
        led.toggle();
        ASSERT_EQ(r[6], 1u << 21);
        led.set_pull(GPIO_PULL_DOWN);
        ASSERT_EQ(r[3], 2u << 10);
        r[4] = 1u << 5;
        ASSERT_EQ(led.read(), 1);

        // PL011: DR +0, FR +0x18, IBRD +0x24, FBRD +0x28, LCR_H +0x2C, CR +0x30
        zero_regs_(&r[0], 16);
        struct Uart<Pl011> pl = Uart::new(base, 115200u, 24000000u);
        ASSERT_EQ(r[9], 13);                           // 24 MHz / (16 * 115200) = 13.02
        ASSERT_EQ(r[10], 1);                           // fraction 0.02 * 64, rounded
        ASSERT_EQ(r[11], 0x70);
        ASSERT_EQ(r[12], 0x301);
        r[6] = 0x20u;                                  // TXFF
        ASSERT_FALSE(pl.tx_ready());
        r[6] = 0x00u;
        pl.write_byte((unsigned char)'Z');
        ASSERT_EQ(r[0], 'Z');
        ASSERT_TRUE(pl.rx_ready());
        r[6] = 0x10u;                                  // RXFE
        ASSERT_FALSE(pl.rx_ready());

        // 16550: byte registers THR +0, IER +1, FCR +2, LCR +3, LSR +5
        zero_regs_(&r[0], 16);
        unsigned char* b = (unsigned char*)&r[0];
        struct Uart<Ns16550> ns = Uart::new(base, 115200u, 1843200u);
        ASSERT_EQ(b[0], 1);                            // divisor latch low (DLAB set at the time)
        ASSERT_EQ(b[3], 0x03);                         // 8N1, DLAB cleared
        ASSERT_EQ(b[2], 0x07);
        b[5] = (unsigned char)0x21;                    // THRE | DR
        ns.write_byte((unsigned char)'Q');
        ASSERT_EQ(b[0], 'Q');
        ASSERT_EQ(ns.read_byte(), 'Q');

        // STM32 USART: SR +0, DR +4, BRR +8, CR1 +0xC
        zero_regs_(&r[0], 16);
        struct Uart<Stm32F4Usart> su = Uart::new(base, 115200u, 42000000u);
        ASSERT_EQ(r[2], 365);                          // 42 MHz / 115200 = 364.6
        ASSERT_EQ(r[3], 0x200C);
        r[0] = 0x80u;
        su.write_byte((unsigned char)'S');
        ASSERT_EQ(r[1], 'S');

        // STM32 SPI: CR1 +0, SR +8, DR +0xC
        zero_regs_(&r[0], 16);
        r[2] = 3u;                                     // TXE | RXNE
        struct SpiDevice<Stm32F4Spi> sp = SpiDevice::new(base, SPI_MODE_1);
        ASSERT_EQ(r[0], 0x1u | 0x4u | 0x18u | 0x300u | 0x40u);
        ASSERT_EQ(sp.transfer((unsigned char)0xA5), 0xA5);

        // STM32 I2C: SR1 +0x14 with SB|ADDR|BTF|RXNE|TXE all set; DR +0x10
        zero_regs_(&r[0], 16);
        struct I2cBus<Stm32F4I2c> si = I2cBus::new(base, 100000u, 42000000u);
        ASSERT_EQ(r[1], 42);                           // CR2.FREQ
        ASSERT_EQ(r[7], 210);                          // CCR = 42 MHz / 200 kHz
        ASSERT_EQ(r[8], 43);                           // TRISE
        ASSERT_EQ(r[0], 1);                            // PE
        r[5] = 0xC7u;
        unsigned char two[2];
        two[0] = (unsigned char)0x11; two[1] = (unsigned char)0x22;
        ASSERT_EQ(si.write((unsigned char)0x3C, &two[0], 2UL), I2C_OK);
        ASSERT_EQ(r[4], 0x22);                         // last byte written
        ASSERT_TRUE((r[0] & 0x200u) != 0u);            // STOP
        ASSERT_EQ(si.read((unsigned char)0x3C, &two[0], 2UL), I2C_OK);
        ASSERT_EQ(two[1], (0x3C << 1) | 1);            // RAM "DR" still holds the address byte
        ASSERT_TRUE((r[0] & 0x400u) == 0u);            // ACK cleared for the last byte
        r[5] = 0x400u;                                 // AF: not acknowledged
        ASSERT_EQ(si.write((unsigned char)0x3C, &two[0], 1UL), I2C_NACK);
        ASSERT_FALSE(si.probe((unsigned char)0x3C));

        // STM32 TIM: CR1 +0, SR +0x10, EGR +0x14, CNT +0x24, PSC +0x28, ARR +0x2C
        zero_regs_(&r[0], 16);
        struct Timer<Stm32F4Tim> st = Timer::new(base, 8399u);
        ASSERT_EQ(r[10], 8399);
        ASSERT_EQ(r[5], 1);                            // UG
        st.set_period(999u);
        ASSERT_EQ(r[11], 999);
        st.start();
        ASSERT_EQ(r[0] & 1u, 1);
        r[4] = 1u;
        ASSERT_TRUE(st.flag_set());
        st.clear_flag();
        ASSERT_EQ(r[4], 0xFFFFFFFEu);                  // rc_w0: write 0 to UIF only
        r[9] = 123u;
        ASSERT_EQ(st.read(), 123);

        // STM32 IWDG: KR +0, PR +4, RLR +8, SR +0xC
        zero_regs_(&r[0], 16);
        struct Watchdog<Stm32F4Iwdg> wd = Watchdog::new(base, 0x1FFFu);
        ASSERT_EQ(r[1], 6);
        ASSERT_EQ(r[2], 0xFFF);                        // 12-bit reload
        ASSERT_EQ(r[0], 0xAAAA);
        wd.enable();
        ASSERT_EQ(r[0], 0xCCCC);
        wd.feed();
        ASSERT_EQ(r[0], 0xAAAA);
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
    struct PerfCounter c = PerfCounter::new();
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
        struct DmaBuffer d = DmaBuffer::wrap(&buf[3], 0UL, 130UL);   // unaligned start
        ASSERT_EQ(d.phys, (unsigned long)&buf[3]);
        ASSERT_TRUE(d.cpu_accessible());
        d.to_device();
        ASSERT_FALSE(d.cpu_accessible());
        d.to_cpu();
        ASSERT_TRUE(d.cpu_accessible());
        ASSERT_EQ(d.phys_at(10UL), (unsigned long)&buf[13]);
        struct DmaBuffer e = DmaBuffer::wrap(&buf[0], 0x8000UL, 64UL);
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
    struct NodePool<PoolNode> p = NodePool::new();
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
        struct TlsfAllocator t = TlsfAllocator::new(4096UL);
        ?&heap unsigned char blocks[64];
        int n = 0;
        while (n < 64) {
            blocks[n] = (?&heap unsigned char)t.alloc(64UL, 8UL);
            if (blocks[n].is_null()) break;
            n = n + 1;
        }
        ASSERT_TRUE(n > 4);
        ASSERT_TRUE(t.alloc(64UL, 8UL).is_null());          // full
        match (blocks[2]) {                                  // neighbours stay used
            case null: { ASSERT_TRUE(0); }
            case some(blk): { t.dealloc((?&heap void)blk); }
        }
        void* again = (void*)t.alloc(64UL, 8UL);
        ASSERT_TRUE(again != (void*)0);
        ASSERT_TRUE(again == (void*)blocks[2]);
        t.destroy();

        // Pool over a buffer too small for one header: empty, nothing written.
        unsigned char tiny[32];
        int i = 0;
        while (i < 32) { tiny[i] = (unsigned char)0xEE; i = i + 1; }
        struct PoolAllocator p = PoolAllocator::over(&tiny[0], 10UL);
        ASSERT_TRUE(p.alloc(1UL, 1UL).is_null());
        ASSERT_EQ(p.available(), 0);
        int untouched = 1;
        i = 0;
        while (i < 32) { if (tiny[i] != (unsigned char)0xEE) untouched = 0; i = i + 1; }
        ASSERT_TRUE(untouched);

        // Slab with zero slots.
        unsigned char sbuf[16];
        struct SlabAllocator s = SlabAllocator::over(&sbuf[0], 16UL, 0UL);
        ASSERT_TRUE(s.alloc(16UL, 1UL).is_null());
        ASSERT_EQ(s.available(), 0);

        // Sizes whose header would wrap around fail instead of returning a tiny block.
        ASSERT_TRUE(alloc_raw_(~0UL - 4UL) == (void*)0);
        ASSERT_TRUE(alloc_zeroed_raw_(~0UL) == (void*)0);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("bitfield", t_bitfield);
        t.add("mmio", t_mmio);
        t.add("isr/vector_table", t_isr_vtable);
        t.add("clock", t_clock);
        t.add("frame/paging", t_frames_paging);
        t.add("mmu page-table formats", t_mmu);
        t.add("process/scheduler", t_scheduler);
        t.add("syscall/mailbox", t_syscall_mailbox);
        t.add("hal drivers (generic chips)", t_hal);
        t.add("hal real-chip layouts", t_hal_chips);
        t.add("coverage/perf", t_debug);
        t.add("dma", t_dma);
        t.add("node_pool", t_node_pool);
        t.add("allocator edge cases", t_alloc_edges);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
