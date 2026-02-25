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

## 4.6 Slice Types `[]T`

A slice is a fat pointer — a pair `{ T*, i64 }` — that carries a base address and element count.

```c
[]int buf;   // fat pointer: {int*, i64}  (parse-only; codegen to be added)
```

Slice types are parsed and type-checked. Codegen ABI is planned for a future milestone.

---

## 4.7 Optional Types `?T`

An optional wraps a value together with a presence bit: `{ T, i1 }`.
It is the zero-overhead equivalent of a nullable pointer — for non-pointer types.

```c
?int find_first(int *arr, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return arr[i];
    }
    return (int)0;  // empty optional
}

// Unwrap with 'try' — propagates None to caller if empty:
int val = try find_first(arr, n, 42);
```

Representation: `llvm struct { T, i1 }` where `i1` is the has-value bit.

---

## 4.8 Result Types

`Result<T, E>` is a stdlib convention using tagged unions.  The language has no special syntax for it yet; use:

```c
struct Result_int { bool ok; int value; int err; };
```

Full `?`-propagation sugar for `Result<T,E>` is planned after pattern matching matures.

---

## 4.9 Distinct Types

Planned: `newtype` wraps an existing type with a fresh identity for stronger type safety.

```c
// Future syntax (not yet implemented):
newtype FileDescriptor = int;
```

---

## 4.10 Function Pointer Types `fn`

The `fn` keyword declares a **type-safe function pointer** — no raw `*`, no hidden casting.
Syntax: `fn ReturnType(ParamTypes...) name`

```c
// Declare a function pointer variable
fn int(int, double) compute;

// Assign any compatible function
int add_one(int x) { return x + 1; }
fn int(int) transform = add_one;
int result = transform(5);   // indirect call → 6

// As a function parameter
void apply(fn int(int) op, int x) {
    printf("result: %d\n", op(x));
}

// Reassignable — behaves like a C function pointer
fn void(void) on_exit;
on_exit = cleanup;
on_exit();
```

The `fn` type lowers to an opaque `ptr` in LLVM IR — **zero runtime overhead** compared to a raw C function pointer.
Unlike raw C `int (*fp)(int)`, the `fn` type is never implicitly cast to a different signature.

**`&static` function references.** Every named function has type `&static fn_type` — a
static-lifetime reference to its function type.  This means passing a function by name
is always safe (no dangling pointer risk) and requires no `unsafe{}` block.

```c
// Range patterns with fn pointer in a dispatch table:
fn void(int) handlers[3] = { do_quit, do_help, do_noop };
```

---

# 5. Object Model

SafeC objects provide deterministic layout, pure static dispatch, and zero hidden runtime cost. All object behavior is resolved entirely at compile time.

---

## 5.1 Struct Layout

Every SafeC struct is a contiguous, metadata-free block of memory whose size is determined solely by its declared fields. Methods, traits, and operators never affect object size.

```c
struct Foo {
    double a;
    double b;
};

// sizeof(Foo) = 2 * sizeof(double) — no hidden fields
```

No vtable pointer, no type tag, and no hidden destructor pointer is injected.

---

## 5.2 Value Semantics

Structs are plain value types:

* Assignment is a bitwise copy — no constructor or destructor is called.
* No implicit move semantics.
* No hidden heap promotion on assignment.

```c
Foo x = y;   // memory copy; no hidden side effects
```

---

## 5.3 Methods

Methods are declared inside the struct body and defined outside using qualified syntax.

**Declaration:**

```c
struct Point {
    double x;
    double y;

    double length() const;               // read-only receiver
    double dot(double ox, double oy) const;
    void   scale(double s);              // mutating receiver
};
```

**Definition:**

```c
double Point::length() const {
    return self.x * self.x + self.y * self.y;
}

void Point::scale(double s) {
    self.x = self.x * s;
    self.y = self.y * s;
}
```

**Lowering rule** — methods lower to free functions at codegen:

| SafeC form | Lowered form |
| ---------- | ------------ |
| `R T::m(P...) const` | `R T_m(const T* self, P...)` |
| `R T::m(P...)` | `R T_m(T* self, P...)` |
| `x.m(args)` call | `T_m(&x, args)` |

`self` is the implicit receiver reference. In a `const` method it is a read-only non-null reference (`noalias nonnull const T*`); in a non-`const` method it is mutable (`noalias nonnull T*`). Accessing `self.field` in safe code requires no `unsafe {}` block.

Method declarations do **not** alter the struct's memory layout.

---

## 5.4 Static Dispatch

All method calls are resolved entirely at compile time based on:

* Receiver type
* Method name
* Const qualifier
* Parameter types (for future overload resolution)

No vtable lookup. No runtime polymorphism. Dynamic dispatch is not supported in safe code.

---

## 5.5 Prohibited Object Features

SafeC explicitly forbids:

| Feature | Reason |
| ------- | ------ |
| Virtual functions | Runtime dispatch violates determinism |
| Vtables | Hidden metadata, hidden allocation |
| Inheritance | Implicit layout coupling |
| Hidden constructors | No-surprise initialization |
| Hidden destructors | No hidden resource cleanup |
| RTTI | Runtime type metadata |
| Implicit heap allocation | Violates zero-hidden-cost guarantee |

Composition and generics replace inheritance.

---

## 5.6 Packed Structs

`packed struct` removes all alignment padding, guaranteeing byte-for-byte layout.
Useful for protocol headers, serialization formats, and hardware register maps.

```c
packed struct EthernetHeader {
    unsigned char dst[6];
    unsigned char src[6];
    unsigned short ethertype;
};
// sizeof(EthernetHeader) == 14  (no padding)
```

LLVM representation: `<{ ... }>` (angle brackets indicate packed layout).

---

## 5.7 Alignment Annotations

Planned: `align(N)` annotation on struct fields and variables.

```c
// Future syntax (not yet implemented):
struct AlignedBuffer {
    align(64) char data[64];
};
```

---

## 5.8 `must_use` Keyword

Prefix a function declaration with `must_use` to require callers to use its return value.
Discarding the return emits a compile-time warning.
The syntax is a keyword prefix — no bracket wrappers — consistent with C-style qualifiers.

```c
must_use int open_file(const char *path);

open_file("foo.txt");           // warning: return value should not be ignored
int fd = open_file("foo.txt");  // OK

must_use int compute(int x) { return x * x + 1; }
```

---

# 6. Compile-Time-First Design

SafeC's compile-time execution engine is a core part of the semantic model — not an optimization hint.

> If something can be known at compile time, it should be known at compile time.

---

## 6.1 Const Functions

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

## 6.2 Consteval Functions

A `consteval` function must execute at compile time. Calling it in a runtime context is a compile error:

```c
consteval int pow2(int x) {
    return 1 << x;
}

const int PAGE_SIZE = pow2(12);  // 4096, compile time only
```

---

## 6.3 Const Contexts

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

## 6.4 Static Assertions

```c
static_assert(sizeof(AudioFrame) == 64, "AudioFrame must be 64 bytes");
static_assert(FACT_10 == 3628800, "10! must be 3628800");
```

`static_assert` conditions must be compile-time constant expressions.
A false condition is a compile error, not a runtime check.

---

## 6.5 Compile-Time Conditionals

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

## 6.6 Compile-Time Execution Rules

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

## 6.7 ABI Guarantee

Const-evaluated results become literal data in the output object.
The compiler never injects hidden runtime initialization for `const` globals.

---

# 7. Preprocessor Model

SafeC includes a preprocessor for C interoperability and build configuration.
It is a compatibility layer, not a metaprogramming system.

---

## 7.1 Design Principles

1. **Deterministic** — same inputs always produce same expansion
2. **No hidden semantics** — no type-oblivious substitution
3. **No region bypass** — preprocessing cannot circumvent region safety
4. **C interoperability** — for existing C headers and build flags

---

## 7.2 Supported Directives

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

## 7.3 Safe Mode vs Compatibility Mode

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

## 7.4 Compile-Time-First Replacements

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

## 7.5 Interaction with Region System

The preprocessor must not:

* Generate region-qualified references implicitly
* Bypass region typing rules
* Inject unsafe dereferences
* Hide lifetime violations

All region semantics are enforced after preprocessing, in the semantic analysis phase.

---

# 8. Safety Analysis

---

## 8.1 Definite Initialization

The compiler rejects use-before-initialization and conditional initialization gaps:

```c
int x;
printf("%d", x);  // error: x used before initialization
```

---

## 8.2 Region Escape Analysis

The compiler detects:

* Stack references escaping their scope
* Arena references escaping their region
* Heap references used after free

---

## 8.3 Aliasing Rules

For safe references:

* Two mutable references to the same object in overlapping lifetime → compile error
* Immutable aliasing is allowed
* Raw pointer aliasing is only allowed inside `unsafe {}`

---

## 8.4 Nullability Enforcement

For `?&T` (nullable):

* Dereference requires a null check on all paths before the dereference site
* Flow-sensitive null elimination is applied

For `&T` (non-null):

* Must be provably non-null at construction

---

## 8.5 Bounds Safety

Arrays are bounds-aware:

* Compile-time bounds proofs applied where possible
* Runtime bounds checks inserted when necessary
* Checks disabled inside `unsafe {}`

Runtime bounds checks are explicit in the IR — no hidden runtime machinery.

---

# 9. Unsafe Boundary Model

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

## 9.1 Safe vs Unsafe Domains

SafeC defines two semantic domains:

```
Safe domain   : region + alias rules enforced
Unsafe domain : raw pointer semantics allowed
```

Transition is explicit via `unsafe {}`. There are no implicit transitions.

---

## 9.2 Lexical Unsafe Tracking

During semantic analysis, each scope carries an `unsafe_mode` flag:

1. Entering `unsafe {}` sets `unsafe_mode = true`
2. Leaving restores the previous state
3. Unsafe does not propagate outside its lexical scope
4. Nested unsafe blocks do not change behavior

---

## 9.3 Pointer Interpretation by Mode

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

## 9.4 Unsafe Escape

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

## 9.5 Formal Safety Boundary

```
S = region-checked domain
U = raw pointer domain

unsafe {}         → temporary S → U interpretation
unsafe escape {}  → permanent S → U conversion
```

No implicit transitions exist.

---

## 9.6 Alias Rules Inside Unsafe

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

# 10. C Interoperability (FFI)

SafeC provides zero-cost, explicit, region-safe interoperability with C-compatible foreign code. FFI preserves all SafeC guarantees by default.

---

## 10.1 FFI Function Declarations

Foreign functions must be declared using **raw C types**. Region-qualified references are not allowed in `extern` signatures:

```c
// Correct: raw C pointer types
extern int printf(char* fmt, ...);
extern void* malloc(long size);
extern void  free(void* ptr);
```

Region qualifiers belong to the SafeC caller, not to the extern declaration.

---

## 10.2 Calling FFI Functions

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

## 10.3 Lifetime Containment Rule

The compiler verifies that the region of every reference passed to a C function outlives the call:

```
lifetime(argument) ≥ lifetime(call)
```

* The region must outlive the entire call
* Temporary inner-scope regions cannot be passed to escaping C code

---

## 10.4 Escape Analysis in Unsafe

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

## 10.5 Region-Specific FFI Rules

| Region        | May Pass to C | Escape | Constraint                        |
| ------------- | ------------- | ------ | --------------------------------- |
| `&static`     | Yes           | Yes    | Always safe, no `unsafe` required |
| `&stack`      | Yes           | No     | Must not escape; must outlive call|
| `&heap`       | Yes           | Yes    | Ownership rules apply             |
| `&arena<R>`   | Yes           | Cond.  | Arena lifetime must allow         |

---

## 10.6 Receiving Raw Pointers from C

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

## 10.7 Optimizer Behavior

Because region references erase to raw pointers at codegen, FFI introduces:

* No fat pointer
* No runtime checks
* No ABI difference
* Full compatibility with C calling conventions

Inside `unsafe`: compiler assumes aliasing; alias-based optimizations are restricted.
Outside `unsafe`: full region-based alias analysis and optimization apply.

---

## 10.8 Compiler FFI Responsibilities

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

## 10.9 FFI Summary

| Feature            | Safe Mode | Unsafe Mode              |
| ------------------ | --------- | ------------------------ |
| Region enforcement | Yes       | Temporarily relaxed      |
| Alias checking     | Yes       | Yes                      |
| Raw pointer usage  | No        | Yes                      |
| Escape allowed     | No        | Only via `unsafe escape` |
| Runtime overhead   | None      | None                     |

SafeC is safe by default. Unsafe is visible. Region escape is explicit. FFI is powerful but never silent.

---

# 11. Error Handling Philosophy

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

# 11.5 Advanced Control Flow

SafeC extends C's control flow with several safety-oriented and ergonomic constructs.

---

## 11.5.1 `defer` / `errdefer`

`defer` schedules a statement to run when the enclosing function exits, in **LIFO order**.
`errdefer` runs only on the error path (when `try` propagated a null optional).

```c
void process_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    defer fclose(f);              // always runs on exit

    char *buf = malloc(4096);
    defer free(buf);              // runs second (LIFO)

    // ... use f and buf ...
}   // fclose then free run here in LIFO order
```

- Multiple defers fire in LIFO order at any return point.
- `errdefer` fires only when `try` causes an early return.
- Defer bodies are ordinary statements; they execute in the same scope.

---

## 11.5.2 `match` Statement vs `switch`

SafeC has **both** `switch` and `match`. They serve different purposes:

| Feature             | `switch` (C-compatible)              | `match` (SafeC)                        |
|---------------------|--------------------------------------|----------------------------------------|
| Fall-through        | Yes (requires `break`)               | No — each arm is independent           |
| Exhaustiveness      | No warning                           | Warning if no `default` arm            |
| Pattern types       | Constant integer `case x:`           | Integer, char, range (`N..M`), OR (`,`)|
| Wildcard            | `default:`                           | `default:`                             |
| Syntax              | `case x: ... break;`                 | `case x, y:` or `case N..M:`          |

Use `switch` for C-compatible code with fall-through behavior.
Use `match` for exhaustive, structured pattern matching without fall-through.

`match` uses the same `case` / `default` / `:` keywords as `switch`, but with no fall-through
and with range + comma-OR patterns:

```c
// switch — C-compatible, explicit break, no exhaustiveness warning
switch (cmd) {
    case 'q': running = 0; break;
    case 'h': show_help(); break;
    default:  printf("unknown command\n");
}

// match — no fall-through, exhaustiveness-warned, range + OR patterns
match (status_code) {
    case 200:          printf("OK\n");
    case 404:          printf("Not Found\n");
    case 400..499:     printf("Client Error\n");
    case 500..599:     printf("Server Error\n");
    default:           printf("Unknown\n");
}
```

Multiple patterns per arm with `,` (logical OR):

```c
match (ch) {
    case 'a', 'e', 'i', 'o', 'u':  vowels++;
    default:                         consonants++;
}
```

The compiler warns if no `default` arm is present (possible non-exhaustive match).

---

## 11.5.3 Labeled `break` / `continue`

Labels allow breaking or continuing an outer loop from inside a nested loop.

```c
outer: for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        if (arr[i][j] == target) {
            found = i * M + j;
            break outer;     // exits both loops
        }
    }
}
```

```c
outer: for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        if (i == j) continue outer;  // skip to next i
        process(i, j);
    }
}
```

---

## 11.5.4 `try` Operator

`try` unwraps an `?T` optional.  If the value is empty, it immediately returns a zero-initialized optional from the enclosing function.

```c
?int safe_index(int *arr, int len, int i) {
    if (i < 0 || i >= len) return (int)0;  // None
    return arr[i];
}

int double_at(int *arr, int len, int i) {
    int v = try safe_index(arr, len, i);  // propagates None if out of range
    return v * 2;
}
```

`try` is a **zero-cost abstraction**: the compiler lowers it to a single flag check and conditional branch — identical to what you would write by hand.
No runtime exception machinery, no heap allocation, no RTTI.

---

## 11.5.5 Overflow Modes (Documented)

Integer overflow behavior is configurable per expression (planned):

```c
// Future syntax:
x +| y    // wrapping add
x +% y    // saturating add
x +^ y    // panic on overflow (debug builds only)
```

Default (`x + y`) is undefined on overflow for signed, wrapping for unsigned (matches C ABI).

---

## 11.6 Concurrency: `spawn` / `join`

SafeC provides lightweight thread spawning via `spawn` and `join`, backed by pthreads.

```c
// Thread function: must have signature void*(void*)
void* worker(void* arg) {
    printf("hello from thread\n");
    return (void*)0;
}

int main() {
    long long h = spawn(worker, 0);   // start thread
    join(h);                           // wait for completion
    return 0;
}
```

**Rules:**

* `spawn(fn, arg)` requires `fn` to be a **`&static` function reference** — only named functions
  with static lifetime may be used as thread entry points.  This is enforced by Sema at
  compile time.
* The thread function must have signature `void*(void*)` (pthreads convention).
* `arg` is the `void*` argument passed to the function; an integer `0` is implicitly
  converted to a null pointer.
* `spawn` returns a `long long` handle (the underlying `pthread_t`).
* `join(handle)` blocks until the thread completes (wraps `pthread_join`).
* Link with `-lpthread` when producing a native binary.

At the IR level, `spawn` lowers directly to a `pthread_create` call — no wrapper functions,
no hidden allocations.

---

# 12. Comparison with Other Languages

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

# 13. Compiler Architecture

## 13.1 Pipeline

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

## 13.2 Key Design Decisions

* **References lower to `ptr`** with `noalias`, `nonnull`, `dereferenceable` attributes
* **Regions are compile-time only** — zero runtime region metadata
* **Const globals are folded** — evaluated initializers become literal IR constants
* **`if const` branches are dead-code-eliminated** before codegen
* **`consteval` is enforced** — calling in runtime context is a compile error
* **Struct layout** follows C ABI rules exactly (no hidden padding reordering)

---

# 14. Build & Usage

## 14.1 Prerequisites

* CMake ≥ 3.20
* C++17 compiler (Clang ≥ 14, GCC ≥ 12, or MSVC 2022)
* LLVM ≥ 17 (with CMake config files)

## 14.2 Building

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

## 14.3 Compiler Flags

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

Incremental compilation (on by default):
  --no-incremental         Disable file-level bitcode cache
  --cache-dir <dir>        Cache directory (default: .safec_cache)
  --clear-cache            Delete all cached .bc files from the cache directory and exit

Debug info:
  --g lines                Emit line-table debug info (DISubprogram + DILocation per statement)
  --g full                 Emit full debug info (lines + DILocalVariable + dbg.declare per local)
```

## 14.4 Example Workflow

```bash
# Compile to LLVM IR
./build/safec examples/hello.sc --emit-llvm -o hello.ll

# Compile IR to native binary (using system clang)
clang hello.ll -o hello

# Run
./hello
```

### Incremental compilation

Incremental compilation is **on by default**. The compiler hashes the preprocessed
source with FNV-1a and caches the compiled bitcode in `.safec_cache/`. On unchanged
input the full pipeline is skipped.

```bash
# First build — compiles and writes bitcode to .safec_cache/
./build/safec src/main.sc --emit-llvm -o main.ll -v
# [safec] Cache written: .safec_cache/4750194f67521a03_main.sc.bc

# Second build on unchanged source — cache hit, skips full pipeline
./build/safec src/main.sc --emit-llvm -o main.ll -v
# [safec] Cache hit: .safec_cache/4750194f67521a03_main.sc.bc

# Disable the cache for a one-off forced recompile
./build/safec src/main.sc --emit-llvm -o main.ll --no-incremental

# Clear the cache
./build/safec src/main.sc --clear-cache
```

### Debug info

```bash
# Emit line tables (function locations + per-statement source map)
./build/safec examples/methods.sc --emit-llvm --g lines -o methods.ll

# Emit full debug info (lines + local variable declarations)
./build/safec examples/methods.sc --emit-llvm --g full -o methods_dbg.ll

# Link with native debugger support and inspect with lldb
clang methods_dbg.ll -g -o methods_dbg
lldb ./methods_dbg
```

---

# 15. Project Status

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
| Struct methods (static dispatch) | ✅ Complete |
| Standard library (`std/`)        | ✅ Complete |
| `safeguard` package manager      | ✅ Complete |
| Arena region runtime allocator   | ✅ Complete |
| Operator overloading             | ✅ Complete |
| Tuple lowering                   | ✅ Complete |
| Concurrency model (spawn/join)   | ✅ Complete |
| Debug info (DWARF)               | ✅ Complete |
| Incremental compilation          | ✅ Complete |
| `defer` / `errdefer`             | ✅ Complete |
| `match` statement (C-style)      | ✅ Complete |
| Labeled `break` / `continue`     | ✅ Complete |
| `must_use` keyword               | ✅ Complete |
| `packed struct`                  | ✅ Complete |
| `?T` optional type               | ✅ Complete |
| `try` operator (optional unwrap) | ✅ Complete |
| `fn` function pointer type       | ✅ Complete |
| Language server (LSP)            | ✅ Complete |
| Self-hosting                     | 🔲 Planned  |

---

# 16. Non-Goals (Initial Version)

* Async/await
* Full concurrency model
* Dynamic dispatch / vtables
* Garbage collection
* Reflection
* Implicit move semantics
* Automatic destructors
* Standard library (use C stdlib via `extern` declarations)

---

# 17. Long-Term Vision

SafeC is not just a safer C compiler — it is a statement about the direction
systems programming should go.  The milestones below trace a path from today's
working prototype to a production-grade language.

## 17.1 Near-Term (Compiler Completeness)

These build directly on the current implementation:

| Feature | Description |
| ------- | ----------- |
| ~~**Arena runtime allocator**~~ | ✅ Done — `new<R> T` bump-pointer allocator with `arena_reset<R>()`. |
| ~~**Operator overloading**~~ | ✅ Done — `operator +`, `operator ==`, etc. as method calls via `T::operator op`; trait enforcement on generics. |
| ~~**Tuple lowering**~~ | ✅ Done — `tuple(T, U)` lowers to anonymous structs; `spawn(fn, arg)` / `join(h)` via pthreads with C-style function pointers. |
| ~~**Debug info (DWARF)**~~ | ✅ Done — `--g lines` emits DISubprogram + per-statement DILocation; `--g full` adds DILocalVariable + `dbg.declare` per local. |
| ~~**Incremental compilation**~~ | ✅ Done — on by default; caches per-file bitcode keyed by FNV-1a hash of preprocessed source; `--no-incremental`, `--cache-dir`, and `--clear-cache` flags. |
| ~~**Function pointer types**~~ | ✅ Done — `fn` type syntax + `&static` function references; `spawn(fn, arg)` validates `&static` at compile time. |

## 17.2 Mid-Term (Language Expressiveness)

SafeC needs to be pleasant to write, not just theoretically safe:

| Feature | Description |
| ------- | ----------- |
| ~~**Pattern matching**~~ | ✅ Done — `match (expr) { pattern => stmt, ... }` with integer literals, ranges (`N..M`), wildcards (`_`), alternations (`\|`). Enum-value patterns planned. |
| ~~**`defer` / `errdefer`**~~ | ✅ Done — LIFO deferred cleanup; `errdefer` runs only on error path. |
| ~~**`try` / optional types**~~ | ✅ Done — `?T` optional as `{T, i1}`; `try` unwraps or early-returns. |
| **Error propagation (`Result<T,E>`)** | Full `Result<T, E>` stdlib type with `?`-propagation sugar requires slice + richer match; planned. |
| **Compile-time reflection** | `typeof(expr)`, `sizeof(T)`, `alignof(T)`, and limited struct field enumeration for generic algorithms. |
| **Variadic generics** | `generic<T...>` for type-safe heterogeneous argument lists. |

## 17.3 Long-Term (Systems Reach)

The ambitions that motivate SafeC's design:

| Feature | Description |
| ------- | ----------- |
| **Safe concurrency via region isolation** | Two threads may not hold mutable references to the same region simultaneously; enforced by the borrow checker at compile time, with zero runtime cost. |
| **Kernel-mode compilation target** | A `--freestanding` mode that disables all C-stdlib dependencies and generates position-independent, no-implicit-call IR suitable for kernel subsystems. |
| **Deterministic lock-free primitives** | `atomic<T>` with explicit memory-order annotations; no hidden mutex, no hidden spin. |
| **Formal safety model** | A machine-checkable proof (Lean 4 or Coq) that well-typed SafeC programs do not exhibit use-after-free, data races, or out-of-bounds accesses. |
| **Static effect system** | Track I/O, allocation, and non-termination effects in function signatures; pure functions verified at compile time. |
| **Language server (LSP)** | Real-time diagnostics, completion, and rename in any LSP-compatible editor — built on the same frontend passes. |
| **Self-hosting** | Rewrite `safec` in SafeC itself, bootstrapped from the LLVM-backed prototype. |

## 17.4 Package Manager & Build System (safeguard)

`safeguard` is the official package manager and build system for SafeC projects. It pairs
with the compiler to provide a complete, reproducible development workflow.

| Capability | Description |
| ---------- | ----------- |
| **Project scaffolding** | `safeguard new <name>` creates a `safeguard.toml` manifest, `src/`, and `examples/`. |
| **Dependency resolution** | Declarative `[dependencies]` table in `safeguard.toml`; content-addressed package cache. |
| **Build orchestration** | Drives `safec` per translation unit; links with LLVM `lld` or system linker; supports `--release` / `--debug` profiles. |
| **Standard library integration** | `std/` is a built-in package; individual modules (`io`, `mem`, `str`, `collections`, …) are imported by name. |
| **Workspace support** | Multi-crate monorepos with shared dependency resolution and incremental per-crate rebuilds. |
| **Reproducible builds** | Lockfile (`safeguard.lock`) pins exact package hashes; deterministic build graph with no network access after first resolution. |

`safeguard` is implemented in SafeC itself, demonstrating the language's suitability for
systems tooling — no hidden allocator, no GC, deterministic build graph execution.

## 17.5 The Core Thesis

SafeC aims to demonstrate three things:

1. **Determinism and safety are not mutually exclusive.**
   The borrow checker, region escape analysis, and bounds checking all operate
   entirely at compile time — there is no runtime safety overhead.

2. **Compile-time design eliminates runtime cost.**
   Generics are monomorphized, `if const` branches are dead-code-eliminated,
   regions carry no runtime metadata, and const-eval folds constants before
   codegen.  The resulting IR is as lean as hand-written C.

3. **C compatibility does not require sacrificing correctness.**
   Native C header import means every C library is immediately usable.
   C ABI compatibility means SafeC objects can be linked into any C project.
   The `unsafe{}` escape hatch provides explicit, auditable boundaries between
   the safe core and C-interop code.

SafeC is an exploration of what C could have become — if memory safety,
compile-time reasoning, and zero-cost abstractions were first-class citizens
from the start.

---

# 18. Identity

SafeC is:

* **Not Rust** — no borrow checker, no ownership transfer, explicit region annotation
* **Not C++** — no hidden constructors/destructors, no exceptions, no virtual dispatch
* **Not Zig** — different memory model, different compile-time system
* **Not a scripting language**

It is:

> A deterministic, ABI-stable, region-aware evolution of C for serious systems work.

The compiler must reflect that identity at every design decision.
