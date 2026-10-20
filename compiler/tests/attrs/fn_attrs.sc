// noinline / cold reach LLVM, also after a const method's 'const', and a
// generic function's instantiation keeps its attributes (always_inline,
// noinline, cold, target).
struct Box { int v; int get() const; };

int Box::get() const __attribute__((noinline, cold)) { return self.v; }
// CHECK: define i32 @Box_get(

generic<T>
T twice(T x) __attribute__((noinline, cold)) { return x + x; }

int use_twice(int x) { return twice(x); }

int use_box(const &Box b) { return b->get(); }
// CHECK: define linkonce_odr i32 @__safec_twice_int(
// CHECK: attributes #0 = { cold {{.*}}noinline
// CHECK: attributes #3 = { cold noinline }
