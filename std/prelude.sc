// SafeC Standard Library — Prelude (implementation aggregator)
// Including this file compiles all standard module implementations in one unit.
// For project builds, safeguard compiles each .sc module separately
// and links them as a static library (libsafec_std.a).
#pragma once

// ── Fundamental types (header-only, no .sc needed) ───────────────────────────
#include "stddef.h"
#include "stdint.h"
#include "stdbool.h"
#include "limits.h"
#include "float.h"
#include "inttypes.h"

// ── Core ─────────────────────────────────────────────────────────────────────
#include "mem.sc"
#include "io.sc"
#include "io_file.sc"
#include "str.sc"
#include "fmt.sc"
#include "convert.sc"

// ── Math & numeric ────────────────────────────────────────────────────────────
#include "math.sc"
#include "complex.sc"
#include "bit.sc"

// ── Character ─────────────────────────────────────────────────────────────────
#include "ctype.sc"

// ── Assertions ────────────────────────────────────────────────────────────────
#include "assert.sc"

// ── Checked arithmetic (C23) ──────────────────────────────────────────────────
#include "stdckdint.sc"

// ── System & OS ───────────────────────────────────────────────────────────────
#include "sys.sc"
#include "errno.sc"
#include "signal.sc"
#include "time.sc"
#include "locale.sc"
#include "fenv.sc"

// ── Concurrency ───────────────────────────────────────────────────────────────
#include "atomic.sc"
#include "thread.sc"

// ── Collections ───────────────────────────────────────────────────────────────
#include "collections/slice.sc"
#include "collections/vec.sc"
#include "collections/string.sc"
#include "collections/stack.sc"
#include "collections/queue.sc"
#include "collections/list.sc"
#include "collections/map.sc"
#include "collections/bst.sc"
