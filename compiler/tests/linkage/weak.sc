// __attribute__((weak)) on an extern declaration makes an undefined
// reference that resolves to null when nothing defines the symbol, so code
// can use an optional library only when the program links it. On a
// definition it makes a weak definition another one may override.
extern int optional_hook(int x) __attribute__((weak));
// CHECK: declare extern_weak {{.*}}@optional_hook(

int call_if_linked(int x) {
    unsafe {
        if ((void*)optional_hook == (void*)0) return -1;
        return optional_hook(x);
    }
}
// CHECK-LABEL: define {{.*}}@call_if_linked(
// CHECK: icmp

__attribute__((weak)) int default_impl(int x) { return x + 1; }
// CHECK: define weak {{.*}}@default_impl(
