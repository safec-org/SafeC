// Defers run where CodeGen runs them: at each return, the ones registered
// before it, last first; an errdefer only on a 'try' early return.
void dealloc(&heap int p);
?int lookup_value(int k);

// The errdefer frees on the 'try' failure path only; the normal path frees
// explicitly and the errdefer doesn't run again.
?int errdefer_only_on_error(int k) {
    &heap int p = new int;
    errdefer dealloc(p);
    int v = try lookup_value(k);
    *p = v;
    int out = *p;
    dealloc(p);
    return out;
}

// A defer registered after an early return doesn't run there.
int late_defer(int k) {
    &heap int p = new int;
    *p = k;
    if (k) { int x = *p; dealloc(p); return x; }
    defer dealloc(p);
    return *p;
}

// CHECK-LABEL: define {{.*}}@main(
int main() { return late_defer(0); }
