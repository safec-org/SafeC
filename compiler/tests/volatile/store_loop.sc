// Volatile stores must survive -O2: dead-store elimination and loop deletion
// may not remove them, even when nothing reads the memory afterwards.

// The mem_zero_secure pattern: wipe a buffer through a pointer to volatile.
void wipe(&unsigned char buf, unsigned long n) {
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)buf;
        unsigned long i = (unsigned long)0;
        while (i < n) {
            p[i] = (unsigned char)0;
            i = i + (unsigned long)1;
        }
    }
}
// CHECK-LABEL: define {{.*}}@wipe(
// CHECK: store volatile i8 0

// A stack secret wiped before it goes out of scope. Without 'volatile' the
// whole wipe loop is dead and -O2 deletes it.
int use_secret(int seed) {
    unsigned char key[16];
    for (int i = 0; i < 16; i++) key[i] = (unsigned char)(seed + i);
    int sum = (int)key[3] + (int)key[7];
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)&key[0];
        for (int i = 0; i < 16; i++) p[i] = (unsigned char)0;
    }
    return sum;
}
// CHECK-LABEL: define {{.*}}@use_secret(
// CHECK: store volatile i8 0

// A volatile local: every store is kept, in order, and the read is real.
int local_stores() {
    volatile int x = 0;
    x = 1;
    x = 2;
    x += 3;
    x++;
    return x;
}
// CHECK-LABEL: define {{.*}}@local_stores(
// CHECK: store volatile i32 0
// CHECK: store volatile i32 1
// CHECK: store volatile i32 2
// CHECK: load volatile i32
// CHECK: store volatile i32
// CHECK: load volatile i32
// CHECK: store volatile i32
// CHECK: load volatile i32

// A volatile global: ++ and compound assignment are volatile
// read-modify-writes, not folded into one store.
volatile int g_counter;
void bump() {
    g_counter++;
    g_counter += 2;
    --g_counter;
}
// CHECK-LABEL: define {{.*}}@bump(
// CHECK: load volatile i32, ptr @g_counter
// CHECK: store volatile i32 {{.*}}@g_counter
// CHECK: load volatile i32, ptr @g_counter
// CHECK: store volatile i32 {{.*}}@g_counter
// CHECK: load volatile i32, ptr @g_counter
// CHECK: store volatile i32 {{.*}}@g_counter
