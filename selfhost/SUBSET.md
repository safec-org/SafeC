# The SafeC subset scc compiles

scc compiles one SafeC translation unit (a main file plus its `#include`s)
to a WebAssembly module. It follows the C++ `safec` compiler: the same
grammar, the same type rules and conversions, the same safety checks, and
the same diagnostic wording where scc reports the same error. When the two
compilers disagree on whether a program is valid, that is a bug, and the
differential tests (`tests/difftest.py`) look for exactly that.

This page lists what is in the subset, what is not, and where scc behaves
differently on purpose.

## Supported

**Preprocessor** (safec's `--compat-preprocessor` mode; scc always runs in
it, so `__SAFEC_COMPAT_PREPROCESSOR__` is defined):

- `#include "file"` and `#include <file>`. Bundled headers are listed under
  [Library](#library). The native `scc` also searches the main file's
  directory and `-I` directories.
- Object-like and function-like `#define`, `#` and `##`, `#undef`.
- `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`, `defined`.
- `#pragma once`, `#error`, `#warning`, `__LINE__` and `__FILE__`.
- Predefined macros: `__SCC__`, `__wasm__`, `__wasm32__` and
  `__SIZEOF_POINTER__` (4). The native `scc` also takes `-D NAME[=VALUE]`.

**Types**:

- `bool`, `char`, `signed char`, `unsigned char`, `short`, `int`, `long`,
  `long long` (signed and unsigned), `float`, `double`, and `int8_t` to
  `uint64_t`.
- Arrays, including multi-dimensional arrays and arrays sized by constant
  expressions.
- Raw pointers `T*`. They are restricted as in safec: dereferencing one,
  doing arithmetic on one, or casting between pointers and integers needs
  `unsafe`.
- References `&stack T`, `&heap T`, `&static T`, `&arena<R> T` and plain
  `&T`, with `const`.
- Nullable references `?&T`, optionals `?T`, slices `[]T`, tuples
  `tuple(int, double)`.
- Function types `fn R(P...)`, and C-style function pointer declarators.
- `struct`, `union`, `enum` (with an underlying type), tagged unions,
  `newtype`, `typedef`, `typeof`.

**Safety checks.** Each of these is a compile-time error, worded as in
safec:

- Integer conversions: implicit conversions only widen. Literals must fit
  the type they convert to. Binary operators need operands of the same type
  (safec's 8-bit promotions are the one exception).
- Region rules: a reference can't escape its region (to a global, to a
  longer-lived local, or in a return value). `&` of a local gives
  `&stack`. Arena references are tied to their region.
- Borrows: a mutable borrow excludes every other borrow while it is live.
- Nullable references can only be used after a check (`match`, `is_null`,
  `default`), and `?T` the same way.
- Use after `arena_reset`, `arena_destroy` or `arena_free_to`, and use
  after `std::dealloc`, through aliases, struct fields and calls, as
  safec's `MemoryFlow` analysis checks them; see [Differences](#differences).
- `unsafe { }` is needed for raw pointer operations and for unchecked
  indexing.
- Definite initialization, as in safec: flow-insensitive, and a global
  read before its declaration counts as "possibly uninitialized".
- `const` correctness, `must_use`, `noreturn`, `pure`, exhaustive `match`,
  trait bounds, and out-of-line method definitions matching their
  declarations.

**Runtime checks**: array, slice and `Slice<T>` indexing is bounds-checked,
arena allocations are checked against the region's capacity, and the stack
is checked for overflow. A failed check prints `runtime error: ...` and
exits with 134 (139 for stack overflow), the codes a native safec program
gets from the same failures.

**Declarations and statements**:

- Functions, including `inline`, `static` and `extern` ones. Variadic
  functions (`...`) can be declared and called. scc has no `va_list`: the
  runtime reads variadic arguments through the scc-only builtin
  `__va_ptr()`.
- Global variables with constant initializers. Struct, array and string
  initializers, designated initializers, compound literals.
- Methods: instance, `const` and `static`; defined inside the struct or out
  of line (`int S::get() const`).
- Operator overloading (`operator+`, `operator==`, ...).
- `namespace` (nested too), `::` qualification, `static_assert`,
  `const fn` / `consteval` evaluation, `if const`.
- Generics (`generic<T: Trait>`) on functions, structs and methods, compiled
  by monomorphization with safec's instance naming (`__safec_max_int`).
  Type arguments are inferred from the call's arguments or its expected
  type. Variadic packs `generic<T...>` work, with `T... args`, `args[i]`,
  forwarding `args` to another call, and `sizeof...(T)`.
- Traits, including supertraits and trait bounds.
- `if`/`else`, `while`, `do`/`while`, `for`, labeled loops with
  `break label` and `continue label`, `switch` (with fallthrough), and
  `match` on enums, tagged unions, optionals, nullable references and
  integers, as a statement and as an expression.
- `defer`, `errdefer`, and `try` on optional-returning calls.
- Regions: `region R { capacity: N }`, `new<R> T`, `arena_mark`,
  `arena_reset`, `arena_free_to`, `arena_destroy`. Heap allocation:
  `new T`, `std::alloc` and `std::dealloc`.
- `spawn(fn, arg)` and `join(h)`, and `fn_eval`.
- `__attribute__((export_name("x")))` exports a function from the module.

## Library

These headers are compiled into scc, so `#include` finds them without any
files.

- **C library subsets**, implemented by the scc runtime (`runtime/rt.sc`,
  which is SafeC compiled by scc into every program):
  - `stdio.h`: `printf`, `fprintf`, `sprintf`, `snprintf`, `puts`,
    `putchar`, `fputs`, `fputc`, `getchar`, `fgets` and `scanf`, on
    `stdin`, `stdout` and `stderr`. Number and float formatting are exact:
    `%f`, `%e` and `%g` print the same digits as glibc.
  - `stdlib.h`: `malloc`, `calloc`, `realloc`, `free`, `exit`, `abort`,
    `atoi`, `strtol`, `strtod`, `abs`, `rand` and `srand`.
  - `string.h`, `ctype.h`, `math.h` (fdlibm-style `exp`, `log`, `pow` and
    trigonometry), `stdbool.h`, `stdint.h`, `stddef.h`, `limits.h`,
    `assert.h`, and `time.h`, whose clocks are stubs.
- **SafeC std**: `std/mem.h`, `std/panic.h` and `std/collections/slice.h`.
  These are the real headers from `std/`, embedded when scc is built, with
  playground versions of their out-of-line functions (`stdshim/`).

## Not supported

These are rejected with an error that says scc doesn't support them:

- `goto` and statement labels (labeled loops do work).
- Inline `asm`.
- Bitfields.

These are not implemented, and the error is a parse or name error:

- SIMD vector types, `_Generic`, `thread_local`/`_Thread_local`, atomics,
  calling-convention keywords, `naked`, `section`, `interrupt`.
- The rest of the std library (`std/collections/vec.h`, `std/str.h`,
  `std/io.h`, ...), and anything else that needs an operating system:
  files, sockets, threads.
- More than one translation unit.

## Differences

- **Target**: wasm32. Pointers and references are 4 bytes, `long` is 8,
  `size_t` is `unsigned long`. `sizeof` of a type that contains pointers
  differs from a 64-bit native build.
- **Threads**: `spawn` runs the function to completion right away, and
  `join` returns its result. A program that needs real concurrency (one
  thread waiting for another) won't behave as it does natively.
- **Memory-flow analysis**: scc's `src/memflow.sc` follows the design of
  safec's `MemoryFlow`. It has three allocation tiers per site, per-cell
  "may dangle" bits, field and element cells, condition splitting, defers
  and errdefers, break and continue targets, and call summaries computed
  to a fixpoint (frees, reads, writes, returned values, resets and
  `free_to` through a passed mark). It is smaller than safec's: it tracks
  at most 64 objects and 384 cells per function, and paths 4 deep. Past
  those limits it stops tracking rather than guess, so it can miss an
  error but does not invent one. It also does not model `goto`. On safec's
  own `compiler/tests/lifetimes` suite, the two compilers agree on accept
  or reject for every test, and scc reports exactly safec's diagnostics on
  every `EXPECT-ERROR` test except the one that uses `goto`.
- **Undefined functions**: a function that is declared but never defined,
  and isn't in the runtime, is a compile error (`undefined function 'f'`)
  when the program calls it. Natively that would be a link error.
- **Capacity**: scc keeps everything in fixed-size tables. The browser
  build (`-D SCC_SMALL`) allows about 1 MB of source and 120,000 tokens,
  which is plenty for any playground program. The native build allows 4 MB
  of source and 400,000 tokens. Past the limit, scc reports `internal limit
  exceeded: <table>`.
- **Diagnostics**: scc stops after the first phase that reports errors
  (preprocessing, parsing, semantic analysis), as safec does. After the
  first error in a phase, scc can report a different set of follow-on
  errors than safec.
