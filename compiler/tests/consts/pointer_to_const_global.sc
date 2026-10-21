// A leading 'const' on a pointer declaration qualifies the pointee, as for
// locals: the global (or array of pointers) itself stays writable, so it
// must not be emitted as an LLVM constant, which lets the optimizer drop
// the stores and fold every load to the initializer.
// CHECK: @last = {{.*}}global ptr null
// CHECK: @limit = {{.*}}constant i32 3
static const char* names[4];
const char* last;
const int limit = 3;

// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 3
int main() {
    static const char* cached;
    unsafe {
        names[1] = "hello";
        last = names[1];
        cached = last;
    }
    if (cached == (const char*)0) return 0;
    return limit;
}
