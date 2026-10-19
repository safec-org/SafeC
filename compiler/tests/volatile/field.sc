// A struct field declared 'volatile' gets volatile loads/stores; its
// non-volatile neighbours are still optimized normally.

struct Dev {
    volatile unsigned int ctrl;
    unsigned int shadow;
};

void poke(&Dev d) {
    d.ctrl = (unsigned int)1;
    d.ctrl = (unsigned int)2;
    d.ctrl |= (unsigned int)4;
    d.shadow = (unsigned int)1;
    d.shadow = (unsigned int)2;
}
// CHECK-LABEL: define {{.*}}@poke(
// CHECK: store volatile i32 1
// CHECK: store volatile i32 2
// CHECK: load volatile i32
// CHECK: store volatile i32
// CHECK-NOT: store i32 1,
// CHECK: store i32 2,

unsigned int poll(&Dev d) {
    unsigned int a = d.ctrl;
    unsigned int b = d.ctrl;
    return a + b;
}
// CHECK-LABEL: define {{.*}}@poll(
// CHECK: load volatile i32
// CHECK: load volatile i32

// Through a raw pointer and as a local struct value: still volatile.
void poke_ptr(Dev* d) {
    unsafe {
        d->ctrl = (unsigned int)7;
        d->ctrl++;
    }
}
// CHECK-LABEL: define {{.*}}@poke_ptr(
// CHECK: store volatile i32 7
// CHECK: load volatile i32
// CHECK: store volatile i32

unsigned int local_dev() {
    Dev d;
    d.ctrl = (unsigned int)9;
    d.shadow = (unsigned int)9;
    return d.shadow;
}
// CHECK-LABEL: define {{.*}}@local_dev(
// CHECK: store volatile i32 9
