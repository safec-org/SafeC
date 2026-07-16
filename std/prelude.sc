// SafeC Standard Library — Prelude (implementation aggregator)
// Including this file compiles all standard module implementations in one unit.
// For project builds, safeguard compiles each .sc module separately
// and links them as a static library (libsafec_std.a).
#pragma once

// ── Fundamental types (header-only, no .sc needed) ───────────────────────────
#include <std/stddef.h>
#include <std/stdint.h>
#include <std/stdbool.h>
#include <std/limits.h>
#include <std/float.h>
#include <std/inttypes.h>

// ── Core ─────────────────────────────────────────────────────────────────────
#include <std/mem.sc>
#include <std/io.sc>
#include <std/io_file.sc>
#include <std/str.sc>
#include <std/fmt.sc>
#include <std/convert.sc>

// ── Math & numeric ────────────────────────────────────────────────────────────
#include <std/math.sc>
#include <std/complex.sc>
#include <std/bit.sc>

// ── Character ─────────────────────────────────────────────────────────────────
#include <std/ctype.sc>

// ── Assertions ────────────────────────────────────────────────────────────────
#include <std/assert.sc>

// ── Checked arithmetic (C23) ──────────────────────────────────────────────────
#include <std/stdckdint.sc>

// ── System & OS ───────────────────────────────────────────────────────────────
#include <std/sys.sc>
#include <std/errno.sc>
#include <std/signal.sc>
#include <std/time.sc>
#include <std/locale.sc>
#include <std/fenv.sc>

// ── Concurrency ───────────────────────────────────────────────────────────────
#include <std/atomic.sc>
#include <std/thread.sc>

// ── Collections ───────────────────────────────────────────────────────────────
#include <std/collections/slice.sc>
#include <std/collections/vec.sc>
#include <std/collections/string.sc>
#include <std/collections/stack.sc>
#include <std/collections/queue.sc>
#include <std/collections/list.sc>
#include <std/collections/map.sc>
#include <std/collections/bst.sc>
