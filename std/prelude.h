// SafeC Standard Library — Prelude
// Include this single file to pull in all standard module declarations.
#pragma once

// ── Fundamental types and constants ──────────────────────────────────────────
#include "stddef.h"
#include "stdint.h"
#include "stdbool.h"
#include "limits.h"
#include "float.h"
#include "inttypes.h"

// ── Core ─────────────────────────────────────────────────────────────────────
#include "mem.h"
#include "io.h"
#include "io_file.h"
#include "str.h"
#include "fmt.h"
#include "convert.h"
#include "heap.h"
#include "result.h"

// ── Math & numeric ────────────────────────────────────────────────────────────
#include "math.h"
#include "complex.h"
#include "bit.h"

// ── Character ─────────────────────────────────────────────────────────────────
#include "ctype.h"

// ── Assertions ────────────────────────────────────────────────────────────────
#include "assert.h"

// ── Checked arithmetic (C23) ──────────────────────────────────────────────────
#include "stdckdint.h"

// ── System & OS ───────────────────────────────────────────────────────────────
#include "sys.h"
#include "errno.h"
#include "signal.h"
#include "time.h"
#include "locale.h"
#include "fenv.h"

// ── Concurrency ───────────────────────────────────────────────────────────────
#include "atomic.h"
#include "thread.h"

// ── Collections ───────────────────────────────────────────────────────────────
#include "collections/slice.h"
#include "collections/vec.h"
#include "collections/string.h"
#include "collections/stack.h"
#include "collections/queue.h"
#include "collections/list.h"
#include "collections/map.h"
#include "collections/bst.h"
#include "collections/btree.h"
#include "collections/ringbuffer.h"
#include "collections/static_vec.h"

// ── Allocators ────────────────────────────────────────────────────────────
#include "alloc/bump.h"
#include "alloc/slab.h"
#include "alloc/pool.h"
#include "alloc/tlsf.h"

// ── Synchronization ──────────────────────────────────────────────────────
#include "sync/spinlock.h"
#include "sync/lockfree.h"
#include "sync/task.h"
#include "sync/thread_bare.h"

// ── Testing framework ────────────────────────────────────────────────────
#include "test/test.h"
#include "test/bench.h"
#include "test/fuzz.h"

// ── Security / Cryptography ───────────────────────────────────────────────
#include "crypto/crypto.h"

// ── Networking ────────────────────────────────────────────────────────────
#include "net/net.h"

// ── Filesystem ────────────────────────────────────────────────────────────
#include "fs/fs.h"

// ── DSP & Real-Time ───────────────────────────────────────────────────────
#include "dsp/dsp_all.h"

// ── Debugging & Profiling ─────────────────────────────────────────────────
#include "debug/debug.h"

// ── Panic & Logging ───────────────────────────────────────────────────────
#include "panic.h"
#include "log.h"

// ── DMA ───────────────────────────────────────────────────────────────────
#include "dma.h"

// ── Hardware Abstraction (freestanding) ──────────────────────────────────
#ifdef __SAFEC_FREESTANDING__
#include "hal/gpio.h"
#include "hal/uart.h"
#include "hal/spi.h"
#include "hal/i2c.h"
#include "hal/timer.h"
#include "hal/watchdog.h"

// ── Interrupt & Register Access (freestanding) ──────────────────────────
#include "interrupt/mmio.h"
#include "interrupt/bitfield.h"
#include "interrupt/isr.h"
#include "interrupt/clock.h"
#include "interrupt/vector_table.h"

// ── Kernel Data Structures (freestanding) ───────────────────────────────
#include "kernel/paging.h"
#include "kernel/frame.h"
#include "kernel/mmu.h"
#include "kernel/process.h"
#include "kernel/scheduler.h"
#include "kernel/ipc.h"
#include "kernel/syscall.h"
#endif
