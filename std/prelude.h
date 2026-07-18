// SafeC Standard Library — Prelude
// Include this single file to pull in all standard module declarations.
#pragma once

// ── Fundamental types and constants ──────────────────────────────────────────
#include <std/stddef.h>
#include <std/stdint.h>
#include <std/stdbool.h>
#include <std/limits.h>
#include <std/float.h>
#include <std/inttypes.h>

// ── Core ─────────────────────────────────────────────────────────────────────
#include <std/mem.h>
#include <std/io.h>
#include <std/io_file.h>
#include <std/str.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/heap.h>
#include <std/result.h>

// ── Math & numeric ────────────────────────────────────────────────────────────
#include <std/math.h>
#include <std/complex.h>
#include <std/bit.h>

// ── Character ─────────────────────────────────────────────────────────────────
#include <std/ctype.h>

// ── Assertions ────────────────────────────────────────────────────────────────
#include <std/assert.h>

// ── Checked arithmetic (C23) ──────────────────────────────────────────────────
#include <std/stdckdint.h>

// ── System & OS ───────────────────────────────────────────────────────────────
#include <std/sys.h>
#include <std/errno.h>
#include <std/signal.h>
#include <std/time.h>
#include <std/locale.h>
#include <std/fenv.h>

// ── Concurrency ───────────────────────────────────────────────────────────────
#include <std/atomic.h>
#include <std/thread.h>

// ── Collections ───────────────────────────────────────────────────────────────
#include <std/collections/slice.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
#include <std/collections/stack.h>
#include <std/collections/queue.h>
#include <std/collections/list.h>
#include <std/collections/map.h>
#include <std/collections/bst.h>
#include <std/collections/btree.h>
#include <std/collections/ringbuffer.h>
#include <std/collections/static_vec.h>

// ── Allocators ────────────────────────────────────────────────────────────
#include <std/alloc/bump.h>
#include <std/alloc/slab.h>
#include <std/alloc/pool.h>
#include <std/alloc/tlsf.h>

// ── Synchronization ──────────────────────────────────────────────────────
#include <std/sync/spinlock.h>
#include <std/sync/lockfree.h>
#include <std/sync/channel.h>
#include <std/sync/mpsc.h>
#include <std/sync/task.h>
#include <std/sync/thread_bare.h>

// ── Inter-process communication ──────────────────────────────────────────
#include <std/ipc/pipe.h>
#include <std/ipc/uds.h>

// ── Testing framework ────────────────────────────────────────────────────
#include <std/test/test.h>
#include <std/test/bench.h>
#include <std/test/fuzz.h>

// ── Security / Cryptography ───────────────────────────────────────────────
#include <std/crypto/crypto.h>

// ── Networking ────────────────────────────────────────────────────────────
#include <std/net/net.h>

// ── Filesystem ────────────────────────────────────────────────────────────
#include <std/fs/fs.h>

// ── Serialization ─────────────────────────────────────────────────────────
#include <std/serial/serial.h>

// ── DSP & Real-Time ───────────────────────────────────────────────────────
#include <std/dsp/dsp_all.h>

// ── Debugging & Profiling ─────────────────────────────────────────────────
#include <std/debug/debug.h>

// ── Panic & Logging ───────────────────────────────────────────────────────
#include <std/panic.h>
#include <std/log.h>

// ── DMA ───────────────────────────────────────────────────────────────────
#include <std/dma.h>

// ── Hardware Abstraction (freestanding) ──────────────────────────────────
#ifdef __SAFEC_FREESTANDING__
#include <std/hal/gpio.h>
#include <std/hal/uart.h>
#include <std/hal/spi.h>
#include <std/hal/i2c.h>
#include <std/hal/timer.h>
#include <std/hal/watchdog.h>

// ── Interrupt & Register Access (freestanding) ──────────────────────────
#include <std/interrupt/mmio.h>
#include <std/interrupt/bitfield.h>
#include <std/interrupt/isr.h>
#include <std/interrupt/clock.h>
#include <std/interrupt/vector_table.h>

// ── Kernel Data Structures (freestanding) ───────────────────────────────
#include <std/kernel/paging.h>
#include <std/kernel/frame.h>
#include <std/kernel/mmu.h>
#include <std/kernel/process.h>
#include <std/kernel/scheduler.h>
#include <std/kernel/ipc.h>
#include <std/kernel/syscall.h>
#endif
