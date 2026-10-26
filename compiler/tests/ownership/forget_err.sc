// forget(x) gives a value up without dropping it: allowed only in unsafe.
struct Buf { int n; void drop(); };
void Buf::drop() { }
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }
void leak_on_purpose() {
    struct Buf b = mk(1);
    unsafe { forget(b); }
}
void not_unsafe() {
    struct Buf b = mk(1);
    forget(b);                            // EXPECT-ERROR: forget() leaks its argument without dropping it: write it inside 'unsafe { }'
}
int main() { return 0; }
