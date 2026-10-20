// Implicit conversions that would drop a volatile qualifier are rejected;
// they need an explicit cast, as in C.

extern void take(unsigned int* p);
volatile unsigned int g_reg;
volatile unsigned char g_buf[8];
typedef volatile unsigned int reg_t;

void ptr_to_plain(volatile unsigned int* vp) {
    unsafe { unsigned int* p = vp; } // EXPECT-ERROR: cannot convert 'volatile unsigned int*' to 'unsigned int*'
}
void ptr_arg(volatile unsigned int* vp) {
    unsafe { take(vp); } // EXPECT-ERROR: argument 1: cannot convert 'volatile unsigned int*' to 'unsigned int*'
}
void addr_of_volatile_global() {
    unsafe { take(&g_reg); } // EXPECT-ERROR: cannot convert '&static volatile unsigned int' to 'unsigned int*'
}
void addr_of_volatile_local() {
    volatile int x = 0;
    unsafe { int* p = &x; } // EXPECT-ERROR: cannot convert '&stack volatile int' to 'int*'
}
void volatile_array_decay() {
    unsafe { unsigned char* p = g_buf; } // EXPECT-ERROR: cannot convert 'volatile uint8[8]' to 'uint8*'
}
void nested(volatile unsigned int** vpp) {
    unsafe { unsigned int** pp = vpp; } // EXPECT-ERROR: cannot convert 'volatile unsigned int**' to 'unsigned int**'
}
void ref_to_plain(volatile &unsigned int r) {
    &unsigned int q = r; // EXPECT-ERROR: cannot convert '&volatile unsigned int' to '&unsigned int'
}
void typedef_ptr(reg_t* r) {
    unsafe { unsigned int* p = r; } // EXPECT-ERROR: cannot convert 'volatile unsigned int*' to 'unsigned int*'
}

// Still fine: adding volatile, and dropping it with an explicit cast.
void allowed(unsigned int* p, volatile unsigned int* vp) {
    unsafe {
        volatile unsigned int* a = p;
        unsigned int* b = (unsigned int*)vp;
        volatile unsigned int* c = &g_reg;
        volatile unsigned char* d = g_buf;
    }
}
