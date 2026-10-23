// SAFEC-ARGS: --target x86_64-pc-windows-msvc
// A monomorphized generic is linkonce_odr in a COMDAT of its own name, so
// the linker keeps one copy when several objects instantiate it. On
// Windows x64 a struct over 8 bytes is passed by reference and returned
// through sret, so the ABI pass rebuilds these functions; it must carry
// the COMDAT over. Without one, COFF emits a weak external with a
// per-object default and link.exe fails (LNK1227) when a program and the
// std archive both instantiate e.g. Vec<GuiCanvasState>::push.
struct Big { long a; long b; long c; };

generic<T> long take(T v, long k) __attribute__((noinline)) { return k; }
generic<T> T give(T v) __attribute__((noinline)) { return v; }

int main() {
    struct Big b;
    b.a = 1L; b.b = 2L; b.c = 3L;
    long n = take(b, 4L);
    struct Big c = give(b);
    return (int)(n + c.a) - 5;
}
// CHECK: $__safec_take_Big = comdat any
// CHECK: $__safec_give_Big = comdat any
// CHECK: define linkonce_odr {{.*}}@__safec_take_Big({{.*}}) {{.*}}comdat {
// CHECK: define linkonce_odr {{.*}}@__safec_give_Big({{.*}}) {{.*}}comdat {
