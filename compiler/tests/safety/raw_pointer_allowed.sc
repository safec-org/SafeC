// What safe code may still do with a raw pointer: hold it, compare it,
// hand it back to C, widen it to void*, and turn it into a *nullable*
// reference — which must then be null-checked before it can be used.
extern void* malloc(unsigned long n);
extern void  free(void* p);

int read_checked(int* q) {
    ?&int r = (?&int)q;              // region unknown, null still possible
    match (r) {
        case some(v): return *v;     // v: non-null reference
        case null:    return -1;
    }
    return -1;
}

int same(int* a, int* b)    { return a == b; }
int is_null(int* a)         { return a == null; }
void* widen(int* a)         { return (void*)a; }
int*  none()                { return (int*)0; }

int with_unsafe(int* q) {
    unsafe {
        &int r = (&int)q;            // non-null assertion: unsafe only
        return *(q + 1) + *r;
    }
}

int main() {
    void* m = malloc(4UL);
    free(m);
    return read_checked((int*)0) == -1 ? 0 : 1;
}
// CHECK-LABEL: define {{.*}}@read_checked(
// CHECK: icmp
