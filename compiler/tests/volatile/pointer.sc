// 'volatile T*' and 'volatile &T' keep the qualifier on the pointee type.

// Two reads through a pointer to volatile are two loads, not one.
unsigned int read_twice(volatile unsigned int* reg) {
    unsafe { return *reg + *reg; }
}
// CHECK-LABEL: define {{.*}}@read_twice(
// CHECK: load volatile i32
// CHECK: load volatile i32

// MMIO-style register write through a cast to pointer-to-volatile.
void mmio_write(unsigned long addr, unsigned int v) {
    unsafe {
        *(volatile unsigned int*)addr = v;
        *(volatile unsigned int*)addr = v;
    }
}
// CHECK-LABEL: define {{.*}}@mmio_write(
// CHECK: store volatile i32
// CHECK: store volatile i32

// 'unsigned int volatile*' (qualifier after the base type) means the same.
void spin_until_set(unsigned int volatile* flag) {
    unsafe { while (*flag == (unsigned int)0) { } }
}
// CHECK-LABEL: define {{.*}}@spin_until_set(
// CHECK: load volatile i32

// Subscript and compound assignment through a pointer to volatile.
void fill(volatile unsigned char* p, int n) {
    unsafe {
        for (int i = 0; i < n; i++) p[i] = (unsigned char)0xAA;
        p[0] += (unsigned char)1;
    }
}
// CHECK-LABEL: define {{.*}}@fill(
// CHECK: store volatile i8 -86
// CHECK: load volatile i8
// CHECK: store volatile i8

// Reference to volatile: member access through it is volatile.
struct Pair { unsigned int a; unsigned int b; };
void set_pair(volatile &Pair p) {
    p.a = (unsigned int)1;
    p.a = (unsigned int)2;
}
// CHECK-LABEL: define {{.*}}@set_pair(
// CHECK: store volatile i32 1
// CHECK: store volatile i32 2

// 'T* volatile p' qualifies the pointer variable, not the pointee: the
// pointer slot is accessed volatilely, the int it points at is not (both
// stores survive only because each one reloads 'p').
void ptr_itself(int* q) {
    int* volatile p = q;
    unsafe {
        *p = 1;
        *p = 2;
    }
}
// CHECK-LABEL: define {{.*}}@ptr_itself(
// CHECK: store volatile ptr
// CHECK: load volatile ptr
// CHECK: store i32 1,
// CHECK: load volatile ptr
// CHECK: store i32 2,
