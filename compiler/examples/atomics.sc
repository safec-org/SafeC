// atomics.sc — atomic operations demo
// Compile: ./build/safec examples/atomics.sc --emit-llvm -o /dev/null

// Atomic fetch-add: returns old value, atomically adds
int increment_counter(int* counter) {
    return atomic_fetch_add(counter, 1);
}

// Atomic fetch-sub
int decrement_counter(int* counter) {
    return atomic_fetch_sub(counter, 1);
}

// Atomic load
int read_counter(int* counter) {
    return atomic_load(counter);
}

// Atomic store
void write_counter(int* counter, int val) {
    atomic_store(counter, val);
}

// Atomic compare-and-swap: returns true if exchange succeeded
bool try_set(int* ptr, int expected, int desired) {
    return atomic_cas(ptr, expected, desired);
}

// Atomic exchange
int swap_value(int* ptr, int new_val) {
    return atomic_exchange(ptr, new_val);
}

// Memory fence
void full_fence() {
    atomic_fence("seq_cst");
}

// Relaxed ordering example
int relaxed_load(int* ptr) {
    return atomic_load(ptr, "relaxed");
}

void release_store(int* ptr, int val) {
    atomic_store(ptr, val, "release");
}

// Run atomic operations through pointer-taking functions
void test_atomics(int* counter) {
    increment_counter(counter);
    increment_counter(counter);
    int val = read_counter(counter);

    bool ok = try_set(counter, 2, 42);
    int old = swap_value(counter, 100);
    full_fence();
}

int main() {
    return 0;
}
