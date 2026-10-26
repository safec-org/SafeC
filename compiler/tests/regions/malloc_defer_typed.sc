// SAFEC-ARGS: -I %root/std -I %root --compat-preprocessor -O0
// malloc_defer(n) with a declared type is std::alloc(n) + a deferred
// std::dealloc(x), T taken from the declaration (std/mem.h's raw allocator
// returns bytes, which don't convert to a '?&heap T' implicitly); 'auto' and
// 'void*' keep the type-erased alloc_raw_/dealloc_raw_.
#include <std/mem.h>

// CHECK-LABEL: define {{.*}}@typed_ref(
// CHECK: call ptr @__safec_std_alloc_int(i64 64)
// CHECK: call void @__safec_std_dealloc_int(
int typed_ref() {
    ?&heap int p = malloc_defer(64UL);
    match (p) { case some(x): { *x = 1; } case null: {} }
    return 0;
}

// CHECK-LABEL: define {{.*}}@typed_raw(
// CHECK: call ptr @__safec_std_alloc_long(i64 8)
// CHECK: call void @__safec_std_dealloc_long(
int typed_raw() {
    unsafe {
        long* q = malloc_defer(8UL);
        *q = 2L;
    }
    return 0;
}

// CHECK-LABEL: define {{.*}}@erased(
// CHECK: call ptr @std_alloc_raw_(i64 16)
// CHECK: call ptr @std_alloc_raw_(i64 32)
// CHECK: call void @std_dealloc_raw_(
// CHECK: call void @std_dealloc_raw_(
int erased() {
    auto a = malloc_defer(16UL);
    unsafe { void* v = malloc_defer(32UL); }
    return 0;
}
