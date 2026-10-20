// volatile survives '&x', array decay, parameters and typedefs.

void store2(volatile unsigned int* p) {
    unsafe { *p = (unsigned int)1; *p = (unsigned int)2; }
}

// '&x' of a volatile local is '&volatile T': it binds to a 'volatile T*'
// parameter (and, per conversions_err.sc, not to a plain 'T*' one).
unsigned int addr_of_local() {
    volatile unsigned int x = (unsigned int)0;
    unsafe { store2(&x); }
    return x;
}
// CHECK-LABEL: define {{.*}}@addr_of_local(
// CHECK: store volatile i32 1
// CHECK: store volatile i32 2
// CHECK: load volatile i32

// A volatile array decays to a pointer to volatile.
volatile unsigned char g_buf[4];
void wipe_buf() {
    unsafe {
        volatile unsigned char* p = g_buf + 0;
        p[0] = (unsigned char)0;
        p[1] = (unsigned char)0;
    }
    g_buf[2] = (unsigned char)0;
    g_buf[2] = (unsigned char)1;
}
// CHECK-LABEL: define {{.*}}@wipe_buf(
// CHECK: store volatile i8 0
// CHECK: store volatile i8 0
// CHECK: store volatile i8 0
// CHECK: store volatile i8 1

// A volatile parameter: every access to its slot is volatile.
int param(volatile int n) {
    n = n + 1;
    n = n + 1;
    return n;
}
// CHECK-LABEL: define {{.*}}@param(
// CHECK: load volatile i32
// CHECK: store volatile i32
// CHECK: load volatile i32
// CHECK: store volatile i32

// A volatile array parameter: element accesses are volatile.
void param_array(volatile unsigned char buf[], int n) {
    unsafe { for (int i = 0; i < n; i++) buf[i] = (unsigned char)0; }
}
// CHECK-LABEL: define {{.*}}@param_array(
// CHECK: store volatile i8 0

// A volatile typedef applies to variables, fields and pointees using it.
typedef volatile unsigned int reg_t;
struct Uart { reg_t data; reg_t status; };
void uart_put(&Uart u, unsigned int c) {
    u.data = c;
    u.data = c;
}
// CHECK-LABEL: define {{.*}}@uart_put(
// CHECK: store volatile i32
// CHECK: store volatile i32

void typedef_ptr(reg_t* r) {
    unsafe { *r = (unsigned int)3; *r = (unsigned int)4; }
}
// CHECK-LABEL: define {{.*}}@typedef_ptr(
// CHECK: store volatile i32 3
// CHECK: store volatile i32 4

reg_t g_flag;
void typedef_global() {
    g_flag = (unsigned int)1;
    g_flag = (unsigned int)2;
}
// CHECK-LABEL: define {{.*}}@typedef_global(
// CHECK: store volatile i32 1
// CHECK: store volatile i32 2
