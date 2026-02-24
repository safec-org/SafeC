# SafeC

> **SafeC is a deterministic, region-aware, compile-time-first systems programming language.**
> It is designed as an evolution of C — preserving C ABI compatibility — while enforcing memory safety, type safety, and real-time determinism.

SafeC targets systems domains where hidden allocations, hidden runtime costs, and silent undefined behavior are unacceptable:

* Operating systems and microkernels
* Embedded firmware and bare-metal RTOS
* Real-time audio and DSP engines
* Kernel subsystems and security-critical infrastructure
* Deterministic simulation systems

---

# 1. Design Philosophy

SafeC is built on five non-negotiable principles:

1. **Determinism** — same input always produces same output; no hidden runtime variance
2. **Zero hidden cost** — every operation has a visible, predictable cost
3. **Explicit control** — no implicit memory, no implicit conversion, no implicit error handling
4. **C ABI compatibility** — C struct layout, C calling conventions, and C linkage by default
5. **Compile-time-first design** — anything knowable at compile time must be resolved at compile time

SafeC does not assume a runtime.
SafeC does not assume an ecosystem.
SafeC does not require dynamic memory.

It assumes you are building systems.

---

# 2. Determinism Contract

SafeC explicitly guarantees:

* **No hidden heap allocation**
* **No hidden runtime**
* **No hidden panics** (unless opted-in via explicit bounds checking)
* **No implicit exceptions**
* **No background garbage collection**
* **No implicit destructor calls**

If memory is allocated, it is explicit.
If a check exists, it is visible in the source.
If a failure occurs, it is predictable.

This makes SafeC suitable for hard real-time DSP, kernel-space execution, bare-metal firmware, and safety-critical systems.

---

# 3. Memory Model: Region-Based Safety

SafeC enforces memory safety through **region-aware references**.
Instead of raw pointers, SafeC references encode the memory region, lifetime constraints, mutability, and nullability.

---

## 3.1 Memory Regions

SafeC defines four primary memory regions:

| Region        | Lifetime              | Allocation            |
| ------------- | --------------------- | --------------------- |
| `stack`       | Lexical scope         | Automatic             |
| `static`      | Program lifetime      | Static storage        |
| `heap`        | Dynamic               | Explicit allocation   |
| `arena<R>`    | User-defined region   | Region allocator      |

---

## 3.2 Region-Qualified References

```c
&stack int           // non-null reference into stack memory
&heap float          // non-null reference into heap memory
&static Config       // non-null reference into static storage
&arena<AudioPool> Frame  // non-null reference into an arena region
```

These references:

* Cannot outlive their region
* Cannot escape to a longer-lived region
* Cannot be implicitly converted across regions

Nullable form uses `?&`:

```c
?&stack Node next;   // nullable stack reference
```

Null must be explicit; references are non-null by default.

---

## 3.3 Region Lifetime Rules

1. `stack` region lifetime = enclosing lexical scope
2. `static` region lifetime = entire program lifetime
3. `heap` region lifetime = until explicit `free`
4. `arena<R>` lifetime = until the region is destroyed

A reference cannot escape to a region with a longer lifetime. The compiler enforces this statically:

```c
// COMPILE ERROR: returning a stack reference outlives the stack frame
&stack int danger() {
    int x = 42;
    return &x;  // error: x is destroyed when danger() returns
}
```

---

## 3.4 Arena Regions

User-defined regions enable high-performance, deterministic memory:

```c
region AudioPool {
    capacity: 4096
}

&arena<AudioPool> AudioFrame buffer;
```

Deallocating the region invalidates all contained references at once, with no per-object bookkeeping. Ideal for audio block processing, game frame memory, and kernel subsystems.

---

# 4. Type System

---

## 4.1 Core Type Categories

1. Primitive types (`int`, `float`, `bool`, `char`, sized variants)
2. Struct types (value types, C-layout compatible)
3. Union types (tagged unions supported)
4. Function types
5. Region-qualified reference types
6. Generic types (monomorphized at compile time)

---

## 4.2 No Implicit Conversions

SafeC does not allow:

* Silent integer widening
* Implicit pointer conversions
* Hidden heap promotion

All conversions are explicit casts.

---

## 4.3 Value vs Reference Semantics

* Structs are value types — assignment copies the struct.
* References are explicit and always region-qualified.
* No hidden move semantics.
* No implicit destructor behavior.

---

## 4.4 Generics

SafeC generics use the `generic` keyword with optional trait constraints:

```c
generic<T: Numeric> T add(T a, T b) {
    return a + b;
}
```

Generics are:

* Compile-time only
* Fully monomorphized (one copy per concrete type instantiation)
* Zero runtime overhead
* No dynamic dispatch, no vtables

Traits are constraint contracts resolved entirely at compile time.

---

## 4.5 Tagged Unions

```c
generic<T, E> union Result {
    T ok;
    E err;
}
```

Representation: a discriminant enum field plus a payload union.
Pattern matching is supported.
No implicit destructor logic is injected.

---

# 5. Compile-Time-First Design

SafeC's compile-time execution engine is a core part of the semantic model — not an optimization hint.

> If something can be known at compile time, it should be known at compile time.

---

## 5.1 Const Functions

A `const` function may execute at compile time or runtime:

```c
const int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

const int FACT_10 = factorial(10);  // evaluated at compile time → 3628800
```

If called in a const context → evaluated at compile time.
If called at runtime → executed as a normal function.

---

## 5.2 Consteval Functions

A `consteval` function must execute at compile time. Calling it in a runtime context is a compile error:

```c
consteval int pow2(int x) {
    return 1 << x;
}

const int PAGE_SIZE = pow2(12);  // 4096, compile time only
```

---

## 5.3 Const Contexts

An expression is required to evaluate at compile time when it appears in:

* `const` variable initializer
* Array length expression
* `static_assert` condition
* Enum value
* Type parameter or generic constraint
* `consteval` function body
* Switch case label
* `if const` branch condition

---

## 5.4 Static Assertions

```c
static_assert(sizeof(AudioFrame) == 64, "AudioFrame must be 64 bytes");
static_assert(FACT_10 == 3628800, "10! must be 3628800");
```

`static_assert` conditions must be compile-time constant expressions.
A false condition is a compile error, not a runtime check.

---

## 5.5 Compile-Time Conditionals

```c
if const (WORD_SIZE == 8) {
    // 64-bit path — selected at compile time
    printf("64-bit architecture\n");
} else {
    // 32-bit path — discarded at compile time
    printf("32-bit architecture\n");
}
```

The unselected branch is eliminated before type-checking and codegen.
This replaces preprocessor `#if` for architecture-specific code paths.

---

## 5.6 Compile-Time Execution Rules

Compile-time functions must:

* Not mutate global state
* Not perform I/O
* Not call non-const functions (unless those are also `const`/`consteval`)
* Not allocate dynamic memory
* Not perform unsafe operations

The compiler enforces termination via:

* Recursion depth limit: 256 frames
* Loop iteration limit: 1,000,000 iterations
* Total instruction budget: 10,000,000 steps

Exceeding any limit is a compile-time error.

---

## 5.7 ABI Guarantee

Const-evaluated results become literal data in the output object.
The compiler never injects hidden runtime initialization for `const` globals.

---

# 6. Preprocessor Model

SafeC includes a preprocessor for C interoperability and build configuration.
It is a compatibility layer, not a metaprogramming system.

---

## 6.1 Design Principles

1. **Deterministic** — same inputs always produce same expansion
2. **No hidden semantics** — no type-oblivious substitution
3. **No region bypass** — preprocessing cannot circumvent region safety
4. **C interoperability** — for existing C headers and build flags

---

## 6.2 Supported Directives

**File inclusion:**

```c
#include "file.h"
#include <system.h>
```

`#pragma once` is supported.

**Object-like macros** (restricted to constant expressions in safe mode):

```c
#define BUFFER_SIZE   256
#define VERSION_MAJOR 1
```

**Conditional compilation:**

```c
#if VERSION >= 100
#ifdef SAFEC_PLATFORM_MACOS
#ifndef DEBUG_LEVEL
#else
#elif DEBUG_LEVEL >= 1
#endif
```

**Undefinition:**

```c
#undef TEMP_MACRO
```

**Built-in identifiers:**

```c
__FILE__   // current source file name
__LINE__   // current line number
```

`__TIME__` and `__DATE__` are disabled (break build reproducibility).

---

## 6.3 Safe Mode vs Compatibility Mode

**Safe mode (default):**

* Function-like macros disabled
* Token pasting (`##`) disabled
* Stringification (`#`) disabled
* Object-like macros limited to constant expressions

**Compatibility mode** (`--compat-preprocessor` flag):

* Full C macro system enabled
* For interfacing with legacy C headers
* For transitional migration from C

---

## 6.4 Compile-Time-First Replacements

Instead of preprocessor constants:

```c
// Old: #define SIZE 128
const int SIZE = 128;
```

Instead of function-like macros:

```c
// Old: #define SQR(x) ((x)*(x))
generic<T: Numeric> T sqr(T x) { return x * x; }
```

Instead of preprocessor conditionals where possible:

```c
// Old: #if ARCH == 64
if const (WORD_SIZE == 8) { ... }
```

---

## 6.5 Interaction with Region System

The preprocessor must not:

* Generate region-qualified references implicitly
* Bypass region typing rules
* Inject unsafe dereferences
* Hide lifetime violations

All region semantics are enforced after preprocessing, in the semantic analysis phase.

---

# 7. Safety Analysis

---

## 7.1 Definite Initialization

The compiler rejects use-before-initialization and conditional initialization gaps:

```c
int x;
printf("%d", x);  // error: x used before initialization
```

---

## 7.2 Region Escape Analysis

The compiler detects:

* Stack references escaping their scope
* Arena references escaping their region
* Heap references used after free

---

## 7.3 Aliasing Rules

For safe references:

* Two mutable references to the same object in overlapping lifetime → compile error
* Immutable aliasing is allowed
* Raw pointer aliasing is only allowed inside `unsafe {}`

---

## 7.4 Nullability Enforcement

For `?&T` (nullable):

* Dereference requires a null check on all paths before the dereference site
* Flow-sensitive null elimination is applied

For `&T` (non-null):

* Must be provably non-null at construction

---

## 7.5 Bounds Safety

Arrays are bounds-aware:

* Compile-time bounds proofs applied where possible
* Runtime bounds checks inserted when necessary
* Checks disabled inside `unsafe {}`

Runtime bounds checks are explicit in the IR — no hidden runtime machinery.

---

# 8. Unsafe Boundary Model

```c
unsafe {
    Header* hdr = (Header*)raw_ptr;
    if (hdr != null) {
        result = hdr->length;
    }
}
```

Inside `unsafe {}`:

* Raw pointer operations allowed
* Bounds checks disabled
* Lifetime checks disabled
* Region rules bypassed

Raw values produced inside `unsafe {}` cannot flow directly into safe references.
Conversion requires an explicit safety assertion where the programmer accepts responsibility.

---

## 8.1 Safe vs Unsafe Domains

SafeC defines two semantic domains:

```
Safe domain   : region + alias rules enforced
Unsafe domain : raw pointer semantics allowed
```

Transition is explicit via `unsafe {}`. There are no implicit transitions.

---

## 8.2 Lexical Unsafe Tracking

During semantic analysis, each scope carries an `unsafe_mode` flag:

1. Entering `unsafe {}` sets `unsafe_mode = true`
2. Leaving restores the previous state
3. Unsafe does not propagate outside its lexical scope
4. Nested unsafe blocks do not change behavior

---

## 8.3 Pointer Interpretation by Mode

Outside unsafe:

```
T &stack     ≠    T*
T &heap      ≠    T*
T &arena<R>  ≠    T*
```

Inside unsafe — *Temporary Raw Interpretation*:

```
T &region  may decay to  T*
```

This requires unsafe context, does not change ownership, does not erase the region type outside the block, and introduces no runtime conversion.

---

## 8.4 Unsafe Escape

`unsafe escape` permanently removes region guarantees for the affected reference:

```c
unsafe escape {
    global_ptr = x;   // region lifetime no longer tracked
}
```

* Region lifetime no longer tracked
* Alias safety no longer enforced
* Full responsibility transferred to the programmer

This is the only legal region break.

---

## 8.5 Formal Safety Boundary

```
S = region-checked domain
U = raw pointer domain

unsafe {}         → temporary S → U interpretation
unsafe escape {}  → permanent S → U conversion
```

No implicit transitions exist.

---

## 8.6 Alias Rules Inside Unsafe

Unsafe does **not** disable alias checking.

Before entering unsafe: alias graph must be valid.

Inside unsafe, multiple raw aliases may exist — the compiler assumes potential aliasing and restricts alias-based optimizations accordingly:

```c
unsafe {
    int *p = x;
    int *q = x;   // allowed; compiler assumes p aliases q
}
```

After leaving unsafe: no illegal alias may persist; the borrow graph must remain consistent.

---

# 9. C Interoperability (FFI)

SafeC provides zero-cost, explicit, region-safe interoperability with C-compatible foreign code. FFI preserves all SafeC guarantees by default.

---

## 9.1 FFI Function Declarations

Foreign functions must be declared using **raw C types**. Region-qualified references are not allowed in `extern` signatures:

```c
// Correct: raw C pointer types
extern int printf(char* fmt, ...);
extern void* malloc(long size);
extern void  free(void* ptr);
```

Region qualifiers belong to the SafeC caller, not to the extern declaration.

---

## 9.2 Calling FFI Functions

The rule depends on the region of the argument being passed:

**`&static T` — always safe, no `unsafe {}` required:**

```c
// String literals are &static char — safe to pass to char*
printf("hello\n");          // OK, no unsafe needed
printf("value: %d\n", x);  // OK
```

**`&stack T`, `&heap T`, `&arena<R> T` — require `unsafe {}`:**

```c
char &stack buffer = ...;

unsafe {
    fgets(buffer, 256, stdin);  // OK inside unsafe
}

fgets(buffer, 256, stdin);      // compile error: outside unsafe
```

---

## 9.3 Lifetime Containment Rule

The compiler verifies that the region of every reference passed to a C function outlives the call:

```
lifetime(argument) ≥ lifetime(call)
```

* The region must outlive the entire call
* Temporary inner-scope regions cannot be passed to escaping C code

---

## 9.4 Escape Analysis in Unsafe

Inside `unsafe`, temporary raw interpretation does **not** allow region escape:

```c
int &stack x = ...;

unsafe {
    global_ptr = x;     // compile error: stack ref stored in persistent location
}
```

Escape requires explicit acknowledgement:

```c
unsafe escape {
    global_ptr = x;     // programmer accepts full responsibility
}
```

---

## 9.5 Region-Specific FFI Rules

| Region        | May Pass to C | Escape | Constraint                        |
| ------------- | ------------- | ------ | --------------------------------- |
| `&static`     | Yes           | Yes    | Always safe, no `unsafe` required |
| `&stack`      | Yes           | No     | Must not escape; must outlive call|
| `&heap`       | Yes           | Yes    | Ownership rules apply             |
| `&arena<R>`   | Yes           | Cond.  | Arena lifetime must allow         |

---

## 9.6 Receiving Raw Pointers from C

Raw pointers returned from C functions must be handled inside `unsafe {}`. Region assignment is explicit:

```c
unsafe {
    void* raw = malloc(128);
    int &heap p = from_raw(raw);  // explicit region declaration
}
```

* Region must be declared explicitly
* Ownership intent must be clear
* The compiler does not infer allocation behavior

No automatic wrapping.

---

## 9.7 Optimizer Behavior

Because region references erase to raw pointers at codegen, FFI introduces:

* No fat pointer
* No runtime checks
* No ABI difference
* Full compatibility with C calling conventions

Inside `unsafe`: compiler assumes aliasing; alias-based optimizations are restricted.
Outside `unsafe`: full region-based alias analysis and optimization apply.

---

## 9.8 Compiler FFI Responsibilities

The compiler must:

* Track unsafe lexically
* Enforce alias graph validity before unsafe
* Enforce lifetime containment on FFI calls
* Detect region escape
* Prevent implicit region leakage
* Erase region metadata at codegen

The compiler must not:

* Insert runtime guards
* Change ABI
* Infer ownership automatically

---

## 9.9 FFI Summary

| Feature            | Safe Mode | Unsafe Mode              |
| ------------------ | --------- | ------------------------ |
| Region enforcement | Yes       | Temporarily relaxed      |
| Alias checking     | Yes       | Yes                      |
| Raw pointer usage  | No        | Yes                      |
| Escape allowed     | No        | Only via `unsafe escape` |
| Runtime overhead   | None      | None                     |

SafeC is safe by default. Unsafe is visible. Region escape is explicit. FFI is powerful but never silent.

---

# 10. Error Handling Philosophy

SafeC supports multiple styles without mandating one:

**C-style error codes:**

```c
int result = do_something();
if (result != 0) { /* handle error */ }
```

**Tagged union Result-style:**

```c
ResultInt r = safe_div(100, 7);
if (r.is_ok) printf("result: %d\n", r.value);
```

No implicit exceptions. No stack unwinding. Error propagation is explicit.

---

# 11. Comparison with Other Languages

| Feature                | C      | C++         | Rust          | Zig        | SafeC                 |
| ---------------------- | ------ | ----------- | ------------- | ---------- | ----------------------|
| Memory Safety          | ❌     | ❌           | ✅            | ❌         | ✅                     |
| C ABI Compatibility    | Native | Native      | FFI           | Native     | Native                |
| Hidden Runtime         | ❌     | ⚠️           | ⚠️            | ❌         | ❌                     |
| Compile-Time Execution | ❌     | `constexpr` | `const fn`    | `comptime` | Compile-time-first     |
| GC                     | ❌     | ❌           | ❌            | ❌         | ❌                     |
| Unsafe Escape          | N/A    | N/A         | `unsafe`      | Manual     | `unsafe {}`           |
| Region Model           | ❌     | ❌           | Borrow-based  | ❌         | Explicit region-based |
| Preprocessor Control   | Full   | Full        | None          | Limited    | Disciplined subset    |

SafeC is not a Rust clone, not a C++ competitor, and not a scripting language. It is a systems language.

---

# 12. Compiler Architecture

## 12.1 Pipeline

```
Source (.sc)
    ↓
Preprocessor        — #include, #define, #ifdef, #pragma once
    ↓
Lexer               — tokenization with SafeC keyword set
    ↓
Parser              — recursive-descent, C grammar + SafeC extensions
    ↓
AST
    ↓
Semantic Analysis   — name resolution, type checking, region escape analysis
    ↓
Const-Eval Engine   — evaluate const/consteval at compile time
    ↓
Code Generation     — LLVM IR with region metadata (noalias, nonnull, etc.)
    ↓
LLVM Backend        — object code / bitcode
```

No runtime injection phase.
No implicit transformation inserting hidden allocations.

---

## 12.2 Key Design Decisions

* **References lower to `ptr`** with `noalias`, `nonnull`, `dereferenceable` attributes
* **Regions are compile-time only** — zero runtime region metadata
* **Const globals are folded** — evaluated initializers become literal IR constants
* **`if const` branches are dead-code-eliminated** before codegen
* **`consteval` is enforced** — calling in runtime context is a compile error
* **Struct layout** follows C ABI rules exactly (no hidden padding reordering)

---

# 13. Build & Usage

## 13.1 Prerequisites

* CMake ≥ 3.20
* C++17 compiler (Clang ≥ 14, GCC ≥ 12, or MSVC 2022)
* LLVM ≥ 17 (with CMake config files)

## 13.2 Building

```bash
# Set LLVM_DIR if LLVM is not on the default search path:
#   macOS (Homebrew):   export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
#   Ubuntu/Debian:      export LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
#   Windows (vcpkg):    set LLVM_DIR=<vcpkg_root>/installed/x64-windows/share/llvm

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release [-DLLVM_DIR=$LLVM_DIR]
cmake --build . --parallel
```

The compiler binary is `build/safec`.

## 13.3 Compiler Flags

```
safec <input.sc> [options]

  -o <file>                Output file (default: stdout for IR)
  --emit-llvm              Emit LLVM IR text instead of bitcode
  --dump-ast               Dump AST and exit (no codegen)
  --dump-pp                Dump preprocessed source and exit
  --no-sema                Skip semantic analysis
  --no-consteval           Skip const-eval pass
  --compat-preprocessor    Enable full C preprocessor (function-like macros, ## , #)
  -I <dir>                 Add directory to include search path
  -D NAME[=VALUE]          Define preprocessor macro
  -v                       Verbose output
```

## 13.4 Example Workflow

```bash
# Compile to LLVM IR
./build/safec examples/hello.sc --emit-llvm -o hello.ll

# Compile IR to native binary (using system clang)
clang hello.ll -o hello

# Run
./hello
```

---

# 14. Project Status

The SafeC compiler is a working prototype implementing:

| Milestone                        | Status      |
| -------------------------------- | ----------- |
| Lexer & tokenizer                | ✅ Complete |
| Recursive-descent parser         | ✅ Complete |
| Region-aware type system         | ✅ Complete |
| Semantic analysis & region checks| ✅ Complete |
| Preprocessor (safe + compat mode)| ✅ Complete |
| Const-eval engine                | ✅ Complete |
| LLVM IR codegen                  | ✅ Complete |
| C interoperability (FFI)         | ✅ Complete |
| Bounds check insertion           | ✅ Complete |
| Arena region escape analysis     | ✅ Complete |
| Alias / borrow checker           | ✅ Complete |
| Generics monomorphization        | ✅ Complete |
| Arena region runtime allocator   | 🔲 Future   |
| Concurrency model                | 🔲 Future   |

---

# 15. Non-Goals (Initial Version)

* Async/await
* Full concurrency model
* Dynamic dispatch / vtables
* Garbage collection
* Reflection
* Implicit move semantics
* Automatic destructors
* Standard library (use C stdlib via `extern` declarations)

---

# 16. Long-Term Vision

After the core compiler is stable:

* Region isolation for safe concurrency
* Verified kernel mode compilation target
* Deterministic lock-free primitives
* Formal safety proof model
* Static effect system

SafeC aims to demonstrate that determinism and safety are not mutually exclusive,
that compile-time design can eliminate runtime cost,
and that C compatibility does not require sacrificing correctness.

SafeC is an exploration of what C could have become — if memory safety and compile-time reasoning were first-class citizens from the start.

---

# 17. Identity

SafeC is:

* **Not Rust** — no borrow checker, no ownership transfer, explicit region annotation
* **Not C++** — no hidden constructors/destructors, no exceptions, no virtual dispatch
* **Not Zig** — different memory model, different compile-time system
* **Not a scripting language**

It is:

> A deterministic, ABI-stable, region-aware evolution of C for serious systems work.

The compiler must reflect that identity at every design decision.
