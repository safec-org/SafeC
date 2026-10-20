// 'new<R> T[n]' and 'new<R> T' check the region's remaining capacity at run
// time and abort instead of bumping past the end of the backing buffer. The
// array form also bounds the count so 'n * sizeof(T)' can't wrap.
region Small { capacity: 64 }

&arena<Small> int take(unsigned long n) { return new<Small> int[n]; }
// CHECK-LABEL: define {{.*}}@take(
// CHECK: arena.overcap
// CHECK: arena.count.over
// CHECK: call void @abort()

&arena<Small> long one() { return new<Small> long; }
// CHECK-LABEL: define {{.*}}@one(
// CHECK: arena.overcap
// CHECK: call void @abort()

int main() { take(4UL); one(); return 0; }
