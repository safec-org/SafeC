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
#include <std/mem.h>
#include <std/io.h>
#include <std/io_file.h>
#include <std/str.h>
#include <std/fmt.h>
#include <std/convert.h>

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
