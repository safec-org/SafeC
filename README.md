# SafeC

> **A deterministic, region-aware, compile-time-first systems programming language.**

SafeC is an evolution of C — preserving C ABI compatibility — while enforcing memory safety, type safety, and real-time determinism at compile time with zero runtime overhead.

**[Documentation](https://github.com/MinjaeKim/safec-docs)** | **[Language Reference](https://github.com/MinjaeKim/safec-docs/tree/main/reference)** | **[Standard Library](https://github.com/MinjaeKim/safec-docs/tree/main/stdlib)**

---

## Quick Start

```bash
# Build the compiler (requires LLVM 17+)
cd compiler
cmake -S . -B build -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
cmake --build build

# Compile and run a SafeC program
./build/safec hello.sc --emit-llvm -o hello.ll
clang hello.ll -o hello
./hello
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
| **C ABI Compatible** | C struct layout, C calling conventions, native `#include <stdio.h>`. Link SafeC objects into any C project. |
| **Compile-Time First** | `consteval` functions, `static_assert`, `if const`, generics via monomorphization. |
| **Bare-Metal Ready** | `--freestanding` mode, `naked`/`interrupt` functions, inline assembly, volatile MMIO, atomic built-ins. |
| **Modern Language** | Generics, struct methods, operator overloading, pattern matching, optional types, slices, defer, tuples, typed channels. |
| **Borrow Checker** | Mutable-XOR-shared aliasing discipline with non-lexical lifetimes (NLL). |
| **Standard Library** | 20+ modules: mem, io, str, math, collections (vec, map, list, bst, ...), thread, atomic. |
| **Package Manager** | `safeguard` — project scaffolding, dependency resolution, build orchestration. |
| **LSP Support** | Language server with diagnostics, hover, completion, go-to-definition; VS Code extension. |

---

## Design Principles

1. **Determinism** — same input, same output; no hidden runtime variance
2. **Zero hidden cost** — every operation has a visible, predictable cost
3. **Explicit control** — no implicit memory, no implicit conversions, no implicit error handling
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
| **Null safety** | References are non-null by default; `?T` optionals require explicit unwrap |
| **Determinism** | No hidden allocator, no GC, no implicit nondeterminism in the safe fragment |

Inside `unsafe {}`, all properties become the programmer's responsibility.
The boundary is explicit and auditable.

Formal treatment uses syntactic type safety (Wright & Felleisen) with region semantics
drawn from Oxide (Weiss et al., 2019) and Cyclone (Grossman et al., 2002).
Lean 4 proof structure is in `proofs/SafeCCore.lean`.

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `compiler/` | SafeC → LLVM IR compiler (C++17, ~10K LOC) |
| `std/` | Standard library modules (.h declarations + .sc implementations) |
| `safeguard/` | Package manager and build system |
| `proofs/` | Lean 4 formalization of the safety model |

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

## Language at a Glance

### Regions and References

```c
&stack int ref = &x;                      // stack reference (lexical scope)
&heap float hp = alloc(sizeof(float));    // heap reference (explicit free)
&arena<AudioPool> Sample s = new<AudioPool> Sample;  // arena reference
&static Config cfg = &global_config;      // static reference (program lifetime)
?&stack Node next;                        // nullable reference
```

### Generics

```c
generic<T: Numeric> T add(T a, T b) { return a + b; }
int r = add(3, 4);          // monomorphized to add_int
double d = add(1.0, 2.0);   // monomorphized to add_double
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

### Optional Types and `try`

```c
?int find(int *arr, int n, int target) {
    for (int i = 0; i < n; i++)
        if (arr[i] == target) return arr[i];
    return (int)0;  // None
}
int val = try find(arr, n, 42);  // propagates None on failure
```

### Defer

```c
FILE *f = fopen(path, "r");
defer fclose(f);
char *buf = malloc(4096);
defer free(buf);
// both run in LIFO order at function exit
```

### Concurrency

```c
void* worker(void* arg) { printf("hello from thread\n"); return (void*)0; }
long long h = spawn(worker, 0);
join(h);
```

### Bare-Metal

```c
naked void _start() { asm volatile ("mov $stack_top, %rsp"); }
interrupt void timer_isr() { unsafe { volatile_store((int*)0x40000000, 1); } }
section(".rodata") const int MAGIC = 42;
```

---

### Known Limitations

- `switch`/`case` with fall-through not supported (use `match`)
- Bitfield structs not supported (use `packed struct` + bit masking)
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
