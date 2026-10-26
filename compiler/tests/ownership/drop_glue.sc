// drop(x) runs x's destructor glue: T::drop() for a Drop type, each owning
// member of an aggregate (fields, array elements, a present optional's
// value), nothing for a type that owns nothing. forget(x) emits nothing.
// needs_drop(T) is a compile-time constant.
extern int printf(const char* fmt, ...);
struct Buf { int n; void drop(); };   // drop() defined elsewhere: the calls stay at -O2
struct Pair { struct Buf a; int x; struct Buf b; };
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }

// CHECK-LABEL: define {{.*}}@drop_pair(
// CHECK: call void @Buf_drop(
// CHECK: call void @Buf_drop(
// CHECK: ret void
void drop_pair(struct Pair p) { drop(p); }

// CHECK-LABEL: define {{.*}}@drop_opt(
// CHECK: br i1
// CHECK: call void @Buf_drop(
void drop_opt(?struct Buf o) { drop(o); }

// CHECK-LABEL: define {{.*}}@drop_int(
// CHECK-NOT: call
// CHECK: ret void
void drop_int(int x) { drop(x); }

// CHECK-LABEL: define {{.*}}@forget_it(
// CHECK-NOT: Buf_drop
// CHECK: ret void
void forget_it(struct Buf b) { unsafe { forget(b); } }

generic<T> int owns(T v) { int r = needs_drop(T); drop(v); return r; }
// CHECK-LABEL: define {{.*}}@flags(
// CHECK: ret i32 1
int flags() { return owns(mk(1)) - owns(2); }

// A value matched out of an optional moves into the binding.
// CHECK-LABEL: define {{.*}}@take_opt(
// CHECK: call void @Buf_drop(
int take_opt(?struct Buf o) {
    match (o) {
        case some(b): { int n = b.n; b.drop(); return n; }
        case none: { }
    }
    return -1;
}

int main() { return flags(); }
