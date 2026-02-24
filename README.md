# SafeC

## Safe Embeddable Systems Language

### Warning: Currently a concept project, not yet implemented in any compiler, interpreter, or similar tool.

### Main Features

- C Superset
- Memory safety
- Type safety
- Modern features (generics, namespaces, closures, tuples, methods, operator overloading)
- LLVM Backend

### Concept

- A safe C superset to replace C in low-level areas such as operating system, web browser, and library programming.
- No built-in syntactic sugar — extend the language yourself!

### How do safe pointers work?

- A safe pointer consists of 3 × sizeof(void*) of memory on the stack: an address, a length, and a lifetime.
- Length prevents out-of-bounds access. If the index is greater than the length, it raises a bounds error.
- Lifetime prevents memory leaks. A single set of braces defines a block with a lifetime; a safe pointer defined inside a block is freed when the block ends, meaning the pointer's lifetime has expired. There are four ways to extend a pointer's lifetime:
  1. Return the pointer, which transfers ownership to the outer block.
  2. Move the pointer to a variable in the outer block.
  3. Copy the pointer to a variable in the outer block. This is not `memcpy()`; it only copies the stack data — the address and length — and reassigns the lifetime.
  4. Use the `static` keyword when declaring, which defers deallocation until the program ends or `free()` is called.
- example

```c
#include <salloc.h>

void pointer1()
{
    void& safe_pointer = (void&)salloc(sizeof(void) * 4); // safe pointer allocated
    return;
} // safe pointer freed

void& pointer2()
{
    void& safe_pointer = (void&)salloc(sizeof(void) * 4); // safe pointer allocated
    return safe_pointer; // moves ownership of safe_pointer to the caller
}

int main()
{
    pointer1(); // safe pointer allocated and deallocated.
    void& pointer = pointer2(); // safe pointer allocated and assigned to variable "pointer"
    return 0;
} // safe pointer in variable "pointer" freed.
```

### What is the generic-like feature?

- Generics in Java and C# use type erasure, which provides a single generic implementation for all types in a safer way than macros, at the cost of runtime overhead.
- Generics in C++ and Rust generate specialized code for every type used, achieving zero-cost abstraction at the expense of compilation time and binary size.
- Generic-like in SafeC only generates code for types instantiated in the library's source code, but third-party users can instantiate their own types in their libraries and programs using the provided header.
- Think of it as providing only the API, not the implementation.
- For C/C++ users, think of it as providing a header but not the source or a compiled library.
- Library developers can tell users how to instantiate their generic functions through comments or documentation.
- example

```c
typedef struct
{
    generic T;
    T a;
    T b;
} Foo;

generic T foo_implement(Foo& bar)
{
    return bar.a + bar.b;
    // If the generic type of bar doesn't match generic T,
    // or generic T doesn't have an operator override for add, it fails to compile.
}

generic T bar(T a, T b)
{
    return a + b;
    // If generic T doesn't have an operator override for add, it fails to compile.
}
```

### Other Modern Features

- A namespace is a named scope that separates one code block from another and can be accessed by name.

```c
int bar();

namespace Foo
{
    int bar();
}

int main()
{
    bar(); // use bar() outside the namespace.
    Foo::bar(); // use bar() in the namespace.
}
```

- A closure is an anonymous function defined inside another function.

```c
int Foo(int a, int b)
{
    return a + int (int x, int y)
    {
        return x * y;
    };
}
```

- A tuple is an anonymous struct that groups values without named fields.

```c
{ int, int } foo = { 1, 2 };
printf("%d, %d", foo.0, foo.1);
```

- A method is a function pointer in a struct with a default implementation.

```c
typedef struct foo
{
    double a;
    double b;
    double (bar*)(foo& self);
} Foo;

double Foo.bar(foo& self) default
{
    return self->a + self->b;
}

```

- Operator overloading allows the use of standard operators for a custom type, including arithmetic operators, logical operators, type casting, indexing, and more.

```c
typedef struct foo
{
    double a;
    double b;
    foo (add* override)(foo& x, foo& y);
} Foo;

Foo Foo.add(Foo& x, Foo& y) default
{
    return Foo { .a = x->a + y->a, .b = x->b + y->b };
}
```

### Other Safety Features

- Unless in an unsafe scope, all struct fields must be initialized before the object is used.
- Unless in an unsafe scope, a switch statement must exhaustively cover every possible value.
- Unless in an unsafe scope, raw pointers and unions are prohibited. Use safe pointers and unionums instead.
- Unless in an unsafe scope, type casting is forbidden unless the type defines a cast overload to the target type. (Primitive types may be cast freely.)

### Unsafe Scope

An unsafe scope (or unsafe block) exists to use C code directly or to optimize performance without overhead. Within an unsafe scope, all features restricted in safe code are allowed, provided they follow the C standard.

### Why not Rust?

- To be honest, I use Rust for most of my personal projects that do not interface with C/C++. But Rust presents a dilemma when used alongside C/C++. When interfacing through unsafe raw pointers, why should developers have to think in a completely different paradigm than the rest of their codebase? That was the first motivation for designing the SafeC project.
- When building the lowest levels of software, such as firmware and operating systems, much of the Rust standard library is unavailable without a C ABI or OS native libraries. You have to implement equivalents yourself or translate existing C code, which is inefficient.
- Enums in Rust are heavyweight. An enum in C is just a number, while an enum in Rust resembles a tagged union — a combination of a C union and a C enum — which makes it larger and slower.
- We drew inspiration for the concept of safety from Rust, but much of Rust's safety model assumes you are running on an existing OS within a multithreaded environment as defined by the Rust ABI. For embedded or operating system developers, these assumptions are unnecessary and can get in the way. So we removed those limitations and gave developers more freedom.
- Syntactic sugar in Rust differs from that in many modern languages such as Python and JavaScript. It is handled at compile time, so the runtime cost is eliminated, which is great. The trade-offs are longer compilation times and an unstable ABI. Rust's ABI is very unstable, so as a library developer, you must either release a new build for every version of Rust or open-source your code, which is not always desirable for proprietary library developers.
- For software developers on established platforms (Windows, macOS, Linux), however, we highly recommend Rust. Rust is even safer than SafeC, especially in multithreaded environments, and has syntactic sugar that will make development much easier. In most cases, you will not need to think about C ABI, OS APIs, intrinsics, or similar low-level concerns.

### Types

- auto
- numbers (signed or unsigned): char, short, int, long, float, double
- bool
- void
- enum
- union
- struct
- function (no dedicated syntax)
- raw pointer (\*)
- safe pointer (&)
- unionum (enum with union, like enum in Rust)

### Preprocessor syntax

- if
- elif
- else
- endif
- ifdef
- ifndef
- elifdef
- elifndef
- define
- undef
- include
- error
- warning
- pragma
- defined

### Other syntax

- //
- /\* \*/
- static
- const
- if
- else
- switch
- case
- for
- do
- while
- goto
- continue
- break
- typedef
- sizeof
- move
- malloc
- calloc
- salloc(safe alloc)
- realloc
- free
- return
- inline
- namespace
- extern
- generic
- override
- default
- unsafe

### Operator

- &
- \*
- !=
- !
- .
- ->
- ::
- ,
- ;
- \+
- \-
- \*
- \/
- \%
- &&
- ||
- &
- |
- ^
- <<
- \>\>
- ==
- \>=
- \>
- <=
- <
- =
- \"
- \'
- \`
- []
- {}
- ()

## Examples

### Hello World

```c
#include <stdio.h>

int main()
{
    printf("Hello World!\n");
    return 0;
}
```
