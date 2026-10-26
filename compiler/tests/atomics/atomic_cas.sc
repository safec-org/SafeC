// atomic_cas(ptr, expected, desired) is 'bool atomic_cas(T *ptr, T exp,
// T des)': 'expected' is a value, not written back (that's
// atomic_compare_exchange_strong(ptr, &expected, desired)). Both values
// become the pointee type, so the cmpxchg covers the whole location.
// EXEC

bool cas_int(int* p, int expected, int desired) {
    unsafe { return atomic_cas(p, expected, desired); }
}
// CHECK-LABEL: define {{.*}}@cas_int(
// CHECK: cmpxchg ptr {{%[^,]+}}, i32 {{%[^,]+}}, i32 {{%[^ ]+}} seq_cst seq_cst
// CHECK: extractvalue { i32, i1 } {{%[^,]+}}, 1
// CHECK-NOT: store

// Literal and int operands on a 64-bit location: an i64 cmpxchg, not an
// i32 one on its low half.
bool cas_ll(long long* p, int desired) {
    unsafe { return atomic_cas(p, 0, desired); }
}
// CHECK-LABEL: define {{.*}}@cas_ll(
// CHECK: sext i32
// CHECK: cmpxchg ptr {{%[^,]+}}, i64 0, i64 {{%[^ ]+}} seq_cst seq_cst, align 8

bool cas_relaxed(int* p) {
    unsafe { return atomic_cas(p, 1, 2, "relaxed"); }
}
// CHECK-LABEL: define {{.*}}@cas_relaxed(
// CHECK: cmpxchg ptr {{%[^,]+}}, i32 1, i32 2 monotonic monotonic

// The C11 form updates 'expected' through its pointer on failure.
bool cas_strong(int* p, int* expected, int desired) {
    unsafe { return atomic_compare_exchange_strong(p, expected, desired); }
}
// CHECK-LABEL: define {{.*}}@cas_strong(
// CHECK: load i32, ptr
// CHECK: cmpxchg ptr
// CHECK: store i32

int main() {
    int x = 5;
    long long big = 4294967296LL;   // low 32 bits are 0
    int exp = 7;
    unsafe {
        if (!cas_int(&x, 5, 9)) return 1;      // success: 5 -> 9
        if (x != 9) return 2;
        if (cas_int(&x, 5, 3)) return 3;       // failure: x stays 9
        if (x != 9) return 4;
        if (cas_ll(&big, 1)) return 5;         // 2^32 != 0: must fail
        if (big != 4294967296LL) return 6;
        big = 0;
        if (!cas_ll(&big, 1)) return 7;
        if (big != 1) return 8;
        if (cas_strong(&x, &exp, 1)) return 9; // failure writes back 9
        if (exp != 9 || x != 9) return 10;
        if (!cas_strong(&x, &exp, 1)) return 11;
        if (x != 1 || exp != 9) return 12;
    }
    return 0;
}
