# SafeC

> **A deterministic, region-aware, compile-time-first systems programming language.**

SafeC is an evolution of C — preserving C ABI compatibility — while enforcing memory safety, type safety, and real-time determinism at compile time with zero runtime overhead.

**[Documentation](https://safec-org.github.io/safec-docs/guide/introduction.html)** | **[Language Reference](https://safec-org.github.io/safec-docs/reference/types.html)** | **[Standard Library](https://safec-org.github.io/safec-docs/stdlib/)**

---

## Quick Start

Installs a prebuilt `safec`/`safeguard`/`sc-lsp` release from GitHub Releases — no LLVM or C++ toolchain needed on the machine you're installing to:

```bash
curl -fsSL https://raw.githubusercontent.com/safec-org/SafeC/main/install.sh | bash
```
```powershell
irm https://raw.githubusercontent.com/safec-org/SafeC/main/install.ps1 | iex
```

```bash
bash install.sh --prefix=/opt/safec   # or specify install directory
bash install.sh --version=v0.2.0      # or pin a specific release
```

macOS/Linux publish `arm64`/`x86_64` binaries; Windows publishes `x86_64`. See [Releases](https://github.com/safec-org/SafeC/releases) for the full asset list, or `.github/workflows/release.yml` for how they're built. To build the compiler, stdlib, `safeguard`, or `sc-lsp` from source instead (e.g. for a platform the release workflow doesn't publish, or from an unreleased commit), see each component's own `CMakeLists.txt` under `compiler/`, `safeguard/`, and the sibling `sc-language-server` repo.

Then create a project and run:

```bash
safeguard new hello && cd hello
safeguard run
```

```c
// hello.sc
extern int printf(const char* fmt, ...);

region AudioPool { capacity: 65536 }

struct Sample { float left; float right; };

int main() {
    // Region-safe reference
    int x = 42;
    &stack int ref = &x;

    // Arena allocation — deterministic, no malloc
    &arena<AudioPool> Sample s = new<AudioPool> Sample;
    s.left = 0.5;
    s.right = -0.3;
    printf("L=%.2f R=%.2f\n", s.left, s.right);
    arena_reset<AudioPool>();

    // Bounds-checked slice
    int arr[5];
    arr[0] = 10; arr[1] = 20; arr[2] = 30;
    []int sl = arr[0..3];
    printf("slice len = %ld\n", sl.len);

    return 0;
}
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Region-Based Memory Safety** | `stack`, `heap`, `arena<R>`, `static` regions enforced at compile time. No GC, no runtime metadata. |
| **Zero Hidden Cost** | No implicit allocations, no hidden runtime, no background GC, no implicit exceptions. |
| **C ABI Compatible** | C struct layout, C calling conventions, native `#include <stdio.h>`. Link SafeC objects into any C or C++ project. |
| **Full C Superset** | `__attribute__`, C-style function pointers, bitfields, designated initializers, flexible array members, anonymous struct/union, compound literals, `_Generic`, VLAs (gated to `unsafe{}`). |
| **Compile-Time First** | `const fn`/`consteval`/`constinit`, `static_assert`, `if const`, generics via monomorphization. |
| **`namespace`** | C++-style `namespace std { ... }` blocks — qualified lookup, mangled linkage, unqualified-fallback resolution. Types are not namespaced by design. |
| **Native SIMD** | `vec<T, N>` — a first-class vector type lowering directly to LLVM vector IR; portable `std::simd` library with per-ISA convenience headers for x86_64 (SSE/AVX), AArch64 (NEON), RISC-V (RVV), WebAssembly (SIMD128), SPIR-V, ARM Cortex-M (MVE + DSP extension), CUDA (NVPTX), and ROCm (AMDGPU). |
| **Multi-Target Codegen** | `--target <triple>` cross-compiles to any LLVM-supported architecture/OS — see the target matrix below. |
| **Bare-Metal Ready** | `--freestanding` mode, `naked`/`interrupt` functions, inline assembly, volatile MMIO, atomic built-ins, ARM Cortex-M HAL (NVIC/SysTick/SCB) and DSP-extension intrinsics. |
| **Modern Language** | Generics, struct methods, operator overloading, pattern matching, optional types, slices, defer, tuples, typed channels. |
| **Borrow Checker** | Mutable-XOR-shared aliasing discipline with non-lexical lifetimes (NLL). |
| **Standard Library** | 30+ modules: mem, io, str, math, collections (vec, map, list, bst, ...), thread, atomic, simd, serialization (JSON/XML/HTML), a cooperative task scheduler + real-time I/O reactor, and more. |
| **Package Manager** | `safeguard` — project scaffolding, dependency resolution, build orchestration, mixed SafeC/C/C++ compilation and linking, plus `format` (reindenter), `lint` (static analysis), `check` (fast compile-only), and `test` (build + run `tests/`). |
| **LSP Support** | Language server sharing the compiler's own frontend (always in sync with the language); diagnostics, hover, completion, go-to-definition; VS Code extension (`.sc` and `.h`), Neovim, Emacs. |

---

## Design Principles

1. **Determinism** — same input, same output; no hidden runtime variance
2. **Zero hidden cost** — every operation has a visible, predictable cost
3. **Explicit control** — no implicit memory, no implicit narrowing conversions, no implicit error handling (safe *widening* between numeric types is the one deliberate exception — see [Numeric Widening](#numeric-widening) below)
4. **C ABI compatibility** — C struct layout, C calling conventions, C linkage by default
5. **Compile-time first** — anything knowable at compile time is resolved at compile time

---

## Safety Model

SafeC enforces seven safety properties entirely at compile time:

| Property | Guarantee |
|----------|-----------|
| **Spatial safety** | Bounds-checked array/slice access (static when possible, runtime otherwise) |
| **Temporal safety** | No use-after-free — region escape analysis prevents dangling references |
| **Aliasing discipline** | One mutable XOR many shared references (borrow checker with NLL) |
| **Region escape** | References cannot outlive their region; inter-procedural analysis at call sites |
| **Data race freedom** | `spawn` rejects mutable non-static refs; channels for message passing |
| **Null safety** | References are non-null by default. Pointers, nullable references (`?&region T`, and the region-less `?&T` outliving reference — see below), and `?T` optionals cannot be dereferenced, member-accessed, or force-unwrapped directly — only `is_null()`/`is_none()`, `.default(fallback)`, `match`, or an explicit `unsafe {}` block can read one |
| **Determinism** | No hidden allocator, no GC, no implicit nondeterminism in the safe fragment |

Inside `unsafe {}`, all properties become the programmer's responsibility. The boundary is explicit and auditable — raw-pointer aliasing (WAR/RAW/WAW hazards) and raw-pointer subscript bounds are two examples of *not* being tracked in `unsafe` code, by design, the same way real C leaves them to the programmer there.

Heap allocator misuse — double-free, use-after-free, freeing a pointer a different allocator produced, and freeing NULL — sits partly inside and partly outside the table above. The common, single-variable case (`&heap T p = new T; ... std::dealloc(p);`) *is* caught at compile time: `p`'s use-after-free and double-free are tracked with the same flow-sensitive generation-counter analysis as arena references (if/else branches checked independently, loop bodies pre-scanned so a free on one iteration is visible to code textually before it on the next — see [Memory & Regions](https://safec-org.github.io/safec-docs/reference/memory.html#compile-time-use-after-free-and-double-free-checking)). That check is intra-procedural and syntactic, though — it only recognizes `std::dealloc(p)` with a bare identifier argument, and doesn't follow a pointer across aliases or function-call boundaries, so it can't be a general solution (whether two arbitrary pointers alias is undecidable at scale). `std::alloc`/`std::dealloc` catch the rest at runtime instead, for a fixed, small cost: every allocation carries a 16-byte header tagging it live/freed, `dealloc()`/`realloc_buf()` check the tag and abort with a diagnostic on a double-free or a pointer `alloc()` never returned, `dealloc(NULL)` is a safe no-op, and a bounded quarantine (freed blocks aren't handed back to the system allocator for a further ~64 frees) keeps that check reliable even though this platform's allocator overwrites a freed block's first bytes almost immediately. `std/alloc/pool.h`/`slab.h`/`tlsf.h`'s allocators get the equivalent runtime check using their own existing per-block metadata; freeing a pointer through the *wrong* allocator's free function (mismatched free) is also runtime-only, since the compiler doesn't track which allocator produced a given pointer. Memory leaks aren't checked at all, at compile time or runtime — general leak detection needs whole-program ownership/escape analysis; pair every allocation with `defer std::dealloc(...)` to make cleanup structurally hard to forget instead.

Formal treatment uses syntactic type safety (Wright & Felleisen) with region semantics
drawn from Oxide (Weiss et al., 2019) and Cyclone (Grossman et al., 2002).

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `compiler/` | SafeC → LLVM IR compiler (C++17, ~10K LOC) |
| `std/` | Standard library modules (.h declarations + .sc implementations), including `std/simd/` (portable `vec<T,N>`-based SIMD + per-ISA convenience headers) and `std/hal/` (per-architecture bare-metal HAL) |
| `safeguard/` | Package manager and build system — compiles and links mixed SafeC/C/C++ projects |

### Compiler Pipeline

```
Source (.sc) → Preprocessor → Lexer → Parser → AST → Sema → ConstEval → CodeGen → LLVM IR
```

### Compiler Flags

```
safec <input.sc> [options]

Output:
  -o <file>                Output file
  --emit-llvm              Emit LLVM IR text
  --dump-ast               Dump AST and exit
  --dump-pp                Dump preprocessed source and exit

Compilation:
  --no-sema                Skip semantic analysis
  --no-consteval           Skip const-eval pass
  --freestanding           Freestanding mode (no hosted headers, warns on stdlib)
  --compat-preprocessor    Enable full C preprocessor
  --target <triple>        Cross-target LLVM triple (default: host). See
                            "Target Support" below for the verified matrix.

Preprocessor:
  -I <dir>                 Add include search path
  -D NAME[=VALUE]          Define preprocessor macro
  --no-import-c-headers    Disable automatic C header import via clang

Debug:
  --g lines                Line-table debug info
  --g full                 Full debug info (lines + local variables)

Incremental:
  --no-incremental         Disable file-level bitcode cache
  --cache-dir <dir>        Cache directory (default: .safec_cache)
  --clear-cache            Clear cached .bc files
```

---

## Target Support

`--target <triple>` selects any LLVM-registered target; SafeC hardcodes
nothing architecture-specific — `vec<T,N>` and the rest of codegen lower to
portable LLVM IR, and the target's own backend does instruction selection.
Verified with real generated machine code (not just accepted input) across:

| OS | Architectures |
|---|---|
| macOS | x86_64, AArch64 |
| Linux | x86_64, x86, AArch64, Aarch32 (ARMv7), RV64, RV32 |
| Windows (MSVC) | x86_64, x86, AArch64 |
| iOS | AArch64 (device + simulator) |
| Android | AArch64, Aarch32, x86_64, x86 |
| FreeBSD | x86_64, AArch64 |
| Bare metal (`--freestanding`) | ARM Cortex-M (Thumb/Thumb2), RV32, RV64, AArch64 |
| Portable / GPU | WebAssembly, SPIR-V, CUDA (NVPTX), ROCm (AMDGPU) |

**Metal Shading Language is not supported** — Apple's Metal compiler has no
LLVM backend upstream (it's a separate, closed toolchain), unlike NVPTX/
AMDGPU/SPIR-V which are real LLVM targets. The only practical interop path
from SPIR-V output is a third-party translator (e.g. SPIRV-Cross), not
something `safec` does itself.

ARM Cortex-M gets first-class treatment: cross-compiles cleanly with
`--freestanding`, has a HAL (`std/hal/cortex_m.sc` — NVIC, SysTick, SCB),
and exposes the DSP extension's packed-SIMD/saturating instructions
(SADD16, SMLAD, USAD8, SSAT, ...) as `std::dsp_*` builtins on M4/M7, plus
real MVE vector codegen through `vec<T,N>` on M55/M85.

---

## Language at a Glance

### Regions and References

```c
&stack int ref = &x;                      // stack reference (lexical scope)
&heap float hp = alloc(sizeof(float));    // heap reference (explicit free)
&arena<AudioPool> Sample s = new<AudioPool> Sample;  // arena reference
&static Config cfg = &global_config;      // static reference (program lifetime)
?&stack Node next;                        // nullable reference
?&Handle outliving;                       // outliving reference — no region at all
&Point sumPoint(&Point p);                // non-nullable, region left to the caller

auto x = compute_something();             // type inferred from the initializer
```

An arena region (`region AudioPool { capacity: 65536 }`) lazily allocates its backing buffer on first `new<AudioPool>`. `arena_reset<AudioPool>()` rewinds the bump pointer for reuse (the buffer itself stays allocated); `arena_destroy<AudioPool>()` actually frees the buffer — safe to call more than once (a no-op past the first call) or before any allocation has happened, and a later `new<AudioPool>` after destroying transparently re-allocates.

#### Outliving references (`&T` / `?&T` with no region)

Leaving off the region qualifier — `&T` (non-nullable) or `?&T` (nullable) — gives a reference with **no declared or tracked region**. Two overlapping reasons to reach for it:

1. **Crossing the `extern` boundary.** `extern` declarations (SafeC's only linkage form — always C ABI, no name mangling) use plain `T*` for pointer parameters/returns, same as C. Most of the time a raw pointer only needs to survive the one call it's used in, so an ordinary `unsafe {}`-gated `T*` is enough. But sometimes a pointer crossing that boundary is meant to be *retained* by the C side past the current function returning — a callback's registered context, an opaque handle a C API hands you and expects back later — and neither `&stack T` (dies with this scope, can't be handed to something that outlives it) nor `&heap`/`&static T` (both claim an ownership/lifetime guarantee SafeC has no way to verify across an ABI boundary it doesn't control) are the right fit for that value while it's on the SafeC side.
2. **Not committing to a region for its own sake.** Most functions that just read or write through a reference for the duration of one call don't actually care whether the caller's value is `&stack`, `&heap`, `&static`, or `&arena` — picking one specific region as the parameter type only narrows who can call it, for no real safety benefit. Default to the region-less form unless the parameter's own semantics genuinely pin it to one lifetime (e.g. a callee that *stores* the reference past its own return needs `&heap`/`&static`, not `&stack`).

Nullable vs. non-nullable is the ordinary, orthogonal choice here too — use `?&T` only where the value can genuinely be absent, `&T` otherwise:

```c
extern int c_register_callback(struct Widget* w);   // conceptually retains 'w'

int use_outliving(?&Widget w) {
    match (w) {
        case null: return -1;
        case some(v): return v.value;   // 'v' is a plain 'Widget' value, no unsafe
    }
}

struct Widget local;
?&Widget wref = &local;        // &stack T → ?&T: implicit, no unsafe, no region check
use_outliving(wref);
c_register_callback(wref);     // ?&T → T*: implicit, no unsafe
```

A raw `T*` (from any extern call) converts to `&T`/`?&T` implicitly, a reference of *any* region converts to either implicitly (`&stack Widget` → `&Widget`/`?&Widget`, no cast), and `&T`/`?&T` converts back to `T*`/`void*` implicitly — no `unsafe` in any direction, and no region bookkeeping, because the region-less form doesn't claim a region to check: `Region::Extern` is exempt from every escape/lifetime check by construction (see `Type.h`'s `Region::Extern` for the full rationale). A non-nullable `&T` derefs/member-accesses freely, same as any other non-nullable reference; reading a nullable `?&T` out still goes through the same grammar as `?&stack T`: `match`, `is_null()`, `.default(fallback)`, or an `unsafe`-gated `!`.

This is deliberately narrow in one direction: `&T`/`?&T` needs a concrete `T`, so it doesn't help with the (common, and intentionally type-erased) `void* userData` callback-context pattern — that stays `void*` on purpose, since the whole point there is that the caller can stash *any* type, and a single collection instance is expected to hold many unrelated types at once (see `std/gui/gui_widget.h`'s `Widget.userData` for the fuller writeup, and `std/dma.h`'s `generic<T> struct DmaChannel` for the case where a concrete `T` — genericized per instantiation — *does* fit). `&T`/`?&T` is for the case where the pointee's type is known and fixed but its region isn't, either because it came from outside SafeC's tracking or because the function simply doesn't need to care.

### Generics

```c
generic<T: Numeric> T add(T a, T b) { return a + b; }
int r = add(3, 4);          // monomorphized to add_int
double d = add(1.0, 2.0);   // monomorphized to add_double
```

### Traits

A trait declares a method signature set; a struct satisfies it **structurally** — just by defining methods with matching names/signatures, no `impl Trait for Type` block. Traits are used as generic bounds (`generic<T: Drawable>`), checked at the call site during monomorphization.

```c
trait Drawable {
    void draw() const;
}

struct Circle {
    double radius;
    void draw() const;   // satisfies Drawable structurally — nothing else needed
}
void Circle::draw() const { printf("circle r=%.1f\n", self.radius); }

generic<T: Drawable> void render(T shape) { shape.draw(); }
render(circle);   // OK: Circle has a matching draw() const
```

Built-in traits usable as bounds: `Eq`, `Ord`, `Add`, `Sub`, `Mul`, `Div`, `Numeric` (int/float family), plus the structural `Indexed`/`Pointer` traits satisfied automatically by array/slice and pointer/reference types.

### Newtype and Tuples

`newtype` wraps a base type in a distinct one — same representation, but not implicitly interchangeable with the base type or other newtypes over it (this is how `std::dsp`'s `Fixed` Q8.24 type keeps itself from being accidentally mixed with a plain `int`):

```c
newtype UserId = int;
UserId id = (UserId)42;   // explicit cast required both ways
```

Tuples are `tuple(T1, T2, ...)`, constructed with a parenthesized literal and accessed by position (`.0`, `.1`, ...):

```c
tuple(int, double) p = (42, 3.14);
int x = p.0;
double y = p.1;
```

### Numeric Widening

Binary operators (`+`, `*`, comparisons, ...) require both operands to
already be the same type — SafeC never does C's "usual arithmetic
conversions" to silently reconcile mismatched signedness. Assignment and
argument-passing are more permissive: a value implicitly *widens* to any
larger numeric type, including crossing from integer to floating-point
(`int` → `long`, `float` → `double`, `int` → `double`). Anything that
isn't a pure widening — narrowing (`long` → `int`), or float → integer in
either direction — requires an explicit cast:

```c
int x = 42;
double dx = x;              // implicit: int -> double widens safely

long long big = 100LL;
int small = (int)big;       // explicit cast required: this narrows
double pi = 3.14;
int truncated = (int)pi;    // explicit cast required: float -> int narrows
```

### Namespaces

```c
namespace std {
    void log(const char* msg);
}
std::log("hello");   // or just log("hello") — unqualified fallback resolves too
```

### Native SIMD

```c
#include <std/simd/simd.h>   // portable f32x4/i32x4/... type aliases

vec<float, 4> a = {1.0, 2.0, 3.0, 4.0};
vec<float, 4> b = simd_splat_f32x4(10.0);
vec<float, 4> c = a + b;      // lowers directly to LLVM vector IR —
                               // real SSE/NEON/RVV/SIMD128 instructions
                               // depending on --target, no intrinsics needed
```

### Struct Methods and Operator Overloading

```c
struct Vec2 {
    double x; double y;
    double length() const;
    Vec2 operator+(Vec2 other) const;
};
double Vec2::length() const { return self.x * self.x + self.y * self.y; }
Vec2 Vec2::operator+(Vec2 other) const {
    Vec2 r; r.x = self.x + other.x; r.y = self.y + other.y; return r;
}
```

### Pattern Matching

```c
match (status) {
    case 200:      printf("OK\n");
    case 404:      printf("Not Found\n");
    case 400..499: printf("Client Error\n");
    default:       printf("Unknown\n");
}
```

### Optional Types, Nullable References, and `try`

A plain `T` implicitly wraps to `?T` ("present"); `null` is the empty case for both `?T` and `?&region T`. Reading one back requires `is_null()`/`is_none()`, `.default(fallback)`, `match`, or `unsafe {}` — direct dereference/member-access/`!` force-unwrap outside those is a compile error.

```c
?int find(int *arr, int n, int target) {
    int i = 0;
    while (i < n) {
        int v; unsafe { v = arr[i]; }
        if (v == target) return v;   // implicit T -> ?T wrap
        i = i + 1;
    }
    return null;                     // the empty case
}

int val = try find(arr, n, 42);      // propagates the empty case to the caller
int safe = find(arr, n, 42).default(-1);

match (find(arr, n, 42)) {
    case none:    printf("not found\n");
    case some(v): printf("found %d\n", v);
}
```

### Defer

```c
FILE *f = fopen(path, "r");
defer fclose(f);
char *buf = malloc(4096);
defer free(buf);
// both run in LIFO order at function exit
```

`auto p = malloc_defer(64);` is sugar for exactly the `alloc()`+`defer dealloc()` pair above in one statement — `malloc_defer` isn't a real callable (nothing to look up or link), it's recognized only as a variable declaration's initializer and expands to those two statements at parse time. Declares as `void*` (like `alloc()` itself), so `auto` or an explicit `void*` are the only valid declared types.

### Concurrency

```c
void* worker(void* arg) { printf("hello from thread\n"); return (void*)0; }
long long h = spawn(worker, 0);
join(h);
```

Channels — `chan_create`/`chan_send`/`chan_recv`/`chan_close` are compiler built-ins (no `#include` needed), a bounded blocking MPMC channel backed by `std/sync/channel.h`'s runtime; `std/sync/channel.h` itself adds a typed generic wrapper (`chan_send_t<T>`/`chan_recv_t<T>`) over the same untyped `void*` handle:

```c
void* ch = chan_create(4);      // capacity 4
int v = 42;
unsafe { chan_send(ch, (void*)&v); }
int out;
unsafe { chan_recv(ch, (void*)&out); }  // out == 42
chan_close(ch);
```

For single-consumer fan-in from multiple producer threads without the channel runtime's blocking/condvar overhead, `std/sync/mpsc.h` provides a lock-free-ish spinlock-guarded bounded ring buffer (`MpscQueue`/`mpsc_new`/`mpsc_enqueue_t`/`mpsc_dequeue_t`); `std/sync/lockfree.h`'s `LFQueue` is the single-producer/single-consumer case with no locking at all. For communicating with a *different process* rather than another thread, see `std/ipc/pipe.h` (anonymous pipes, typically parent/child after `fork()`) and `std/ipc/uds.h` (named Unix domain sockets, for unrelated processes).

### Bare-Metal

```c
naked void _start() { asm volatile ("mov $stack_top, %rsp"); }
interrupt void timer_isr() { unsafe { volatile_store((int*)0x40000000, 1); } }
section(".rodata") const int MAGIC = 42;
```

---

### Known Limitations

- `switch`/`case` with fall-through not supported (use `match`)
- Metal Shading Language has no LLVM backend upstream — not reachable via `--target` (see "Target Support")
- `thread_local` / `_Thread_local` / `__thread` supported on globals and static locals
- Calling convention annotations: `__stdcall`, `__cdecl`, `__fastcall`

---

## Non-Goals

- Async/await
- Dynamic dispatch / vtables
- Garbage collection
- Reflection / RTTI
- Implicit move semantics
- Automatic destructors
- Exceptions

---

## License

Released under the MIT License. Copyright 2024-2026 SafeC Contributors.
