// A struct-body method declaration may name a typedef (or struct) declared
// after the struct; it resolves once everything is collected, so the
// out-of-line definition matches and calls type-check against it.
struct S { int x; void apply(Cb c); };
typedef fn void(&S) Cb;
void S::apply(Cb c) { c(self); }
static void bump(&S s) { s.x = s.x + 1; }

// CHECK-LABEL: define {{.*}}@main(
// CHECK: ret i32 2
int main() {
    struct S s;
    s.x = 1;
    s.apply(bump);
    return s.x;
}
