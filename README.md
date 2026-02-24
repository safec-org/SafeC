# SafeC

> **SafeC is a deterministic, region-aware, compile-time-first systems programming language.**
> It is designed as an evolution of C — preserving C ABI compatibility — while enforcing memory safety, type safety, and real-time determinism.

SafeC does not seek to displace C from existing codebases.
It provides a safer path for new systems projects that would otherwise be written in C:

* Operating systems
* Embedded firmware
* Real-time audio engines
* Kernel subsystems
* Security-critical infrastructure
* Deterministic simulation systems

---

# 1. Design Philosophy

SafeC is built on five non-negotiable principles:

1. **Determinism**
2. **Zero hidden cost**
3. **Explicit control**
4. **C ABI compatibility**
5. **Compile-time-first design**

SafeC does not assume a runtime.
SafeC does not assume an ecosystem.
SafeC does not require dynamic memory.

It assumes you are building systems.

---

# 2. Determinism Contract

SafeC explicitly guarantees:

* **No hidden heap allocation**
* **No hidden runtime**
* **No hidden panics (unless opted-in)**
* **No implicit exceptions**
* **No background garbage collection**

If memory is allocated, it is explicit.
If a check exists, it is visible.
If a failure occurs, it is predictable.

This makes SafeC suitable for:

* Hard real-time DSP
* Kernel-space execution
* Bare-metal firmware
* Safety-critical systems

---

# 3. Memory Model: Region-Based Safety

SafeC enforces memory safety through **region-aware references**.

Instead of raw pointers, SafeC references encode:

* Memory region
* Lifetime constraints
* Mutability
* Nullability

---

## 3.1 Memory Regions

SafeC defines four primary memory regions:

| Region       | Lifetime            | Allocation          |
| ------------ | ------------------- | ------------------- |
| `stack`      | Lexical scope       | Automatic           |
| `static`     | Program lifetime    | Static storage      |
| `heap`       | Dynamic             | Explicit allocation |
| `arena<R>`   | User-defined region | Region allocator    |

---

## 3.2 Region-Qualified References

Examples:

```c
&stack int
&heap float
&static Config
&arena<AudioPool> AudioFrame
```

These references:

* Cannot outlive their region
* Cannot escape to longer-lived regions
* Cannot be implicitly converted across regions

---

## 3.3 Arena Regions

User-defined regions enable high-performance deterministic memory:

```c
region AudioPool {
    capacity: 4096  // maximum number of elements
}
```

Then:

```c
&arena<AudioPool> AudioFrame buffer;
```

Deallocating the region invalidates all contained references at once.

This is ideal for:

* Audio block processing
* Game frame memory
* Kernel subsystems
* DSP scratch buffers

---

# 4. Formal Safety Model

In safe mode, SafeC guarantees:

## 4.1 Memory Safety

* No use-after-free
* No double-free
* No dangling references
* No invalid region escape
* No null dereference (unless explicitly nullable)

## 4.2 Bounds Safety

* Arrays are bounds-aware
* Compile-time bounds checking when possible
* Runtime bounds checking when necessary
* Checks removable inside `unsafe {}`

## 4.3 Nullability Model

References are non-null by default:

```c
&int      // non-null
?&int     // nullable
```

Null must be explicit.

---

## 4.4 Unsafe Blocks

Unsafe operations must be explicitly marked:

```c
unsafe {
    raw = (int*)addr;
}
```

Inside `unsafe`:

* Bounds checks disabled
* Lifetime checks disabled
* Region rules bypassed

Raw pointers and region-unqualified values produced inside `unsafe {}` cannot be directly used in safe code. They must be converted to safe references through an explicit safety assertion — at which point the programmer takes responsibility for correctness, and the compiler enforces safe type rules from that point forward.

---

# 5. Type System Philosophy

SafeC types are:

* Explicit
* Predictable
* Zero-cost
* Region-aware

---

## 5.1 No Implicit Conversions

SafeC does not allow:

* Silent integer widening
* Implicit pointer conversions
* Hidden heap promotion

All conversions are explicit.

---

## 5.2 Value vs Reference Types

* Structs are value types.
* References are explicit.
* No hidden move semantics.
* No implicit destructor behavior.

---

## 5.3 Generics

SafeC generics use the `generic` keyword with optional trait constraints:

* Are monomorphized at compile time
* Produce zero runtime overhead
* Do not require dynamic dispatch

Example:

```c
generic<T: Numeric> T add(T a, T b)
```

Traits are constraint contracts, not runtime objects.

---

## 5.4 Tagged Unions

SafeC supports tagged unions for expressive type modeling:

```c
generic<T, E> union Result {
    T ok;
    E err;
}
```

Pattern matching is optional but supported.

SafeC does not mandate a specific error handling paradigm.

---

# 6. Error Handling Philosophy

SafeC supports multiple styles:

### C-style error codes

```c
int result = do_something();
if (result != 0) { ... }
```

### Tagged union Result-style

```c
Result<int, Error> value = divide(a, b);
```

No implicit exceptions.
No stack unwinding.
Error propagation is explicit.

---

# 7. Compile-Time-First Design

SafeC strongly encourages compile-time evaluation.

> If something can be known at compile time, it should be.

---

## 7.1 Compile-Time Functions

```c
const int factorial(int n)
```

If invoked in a constant context → evaluated at compile time.
Otherwise → runtime fallback.

---

## 7.2 Forced Compile-Time Execution

```c
consteval void generate_table()
```

Must execute at compile time or compilation fails.

---

## 7.3 Compile-Time Validation

```c
static_assert(sizeof(AudioFrame) == 64);
```

SafeC supports:

* Layout validation
* Alignment checks
* Compile-time bounds validation
* Generic specialization checks

---

## 7.4 Static Data Generation

Compile-time generated tables are placed in static storage:

```c
const float lookup[256] = generate_table();
```

No runtime initialization required.

---

# 8. Concurrency (Future Direction)

SafeC's region model enables:

* Region isolation across threads
* Deterministic ownership boundaries

Concurrency guarantees are future work.

---

# 9. Comparison with Other Languages

| Feature                | C      | C++         | Rust         | Zig        | SafeC                 |
| ---------------------- | ------ | ----------- | ------------ | ---------- | --------------------- |
| Memory Safety          | ❌     | ❌           | ✅           | ❌         | ✅                     |
| C ABI Compatibility    | Native | Native      | FFI          | Native     | Native                |
| Hidden Runtime         | ❌     | ⚠️           | ⚠️           | ❌         | ❌                     |
| Compile-Time Execution | ❌     | `constexpr` | `const fn`   | `comptime` | Compile-time-first    |
| GC                     | ❌     | ❌           | ❌           | ❌         | ❌                     |
| Unsafe Escape          | N/A    | N/A         | `unsafe`     | Manual     | `unsafe {}`           |
| Region Model           | ❌     | ❌           | Borrow-based | ❌         | Explicit region-based |

SafeC is positioned as:

> A deterministic, ABI-stable evolution of C for modern systems programming.

It is not a Rust clone.
It is not a C++ competitor.
It is not a scripting-friendly language.

It is a systems language.

---

# 10. Intended Use Cases

SafeC is designed for:

* Microkernels
* Embedded RTOS
* Audio DSP engines
* Low-latency infrastructure
* Hardware control systems
* Security-sensitive components

---

# 11. Project Status

SafeC is currently a language design project.
Compiler implementation is future work.

Planned milestones:

1. Formal grammar specification
2. Type system formalization
3. Region safety proof model
4. LLVM IR lowering design
5. Prototype compiler

---

# 12. Vision

SafeC aims to demonstrate that:

* Determinism and safety are not mutually exclusive.
* Compile-time design can eliminate runtime cost.
* C compatibility does not require sacrificing correctness.

SafeC is an exploration of what C could have become —
if memory safety and compile-time reasoning were first-class citizens.
