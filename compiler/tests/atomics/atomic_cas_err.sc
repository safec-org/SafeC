// atomic_cas takes 'expected' by value; passing its address (the C11
// atomic_compare_exchange_strong shape) used to type-check and then emit
// 'cmpxchg ptr %p, ptr %expected, i32 %d', which isn't valid IR.
bool by_address(int* p, int desired) {
    int expected = 0;
    unsafe { return atomic_cas(p, &expected, desired); } // EXPECT-ERROR: atomic_cas takes 'expected' by value
}

bool wrong_pointer(int* p, int* q) {
    unsafe { return atomic_cas(p, 0, q); } // EXPECT-ERROR: cannot convert 'desired'
}

bool not_a_pointer(int x) {
    return atomic_cas(x, 0, 1); // EXPECT-ERROR: first argument must be a pointer
}
