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
