// extracted from safec-docs reference/concurrency.md (block 1)
extern int printf(const char* fmt, ...);
static int g_value = 42;

?&int worker(?&int arg) {
    match (arg) {
        case some(v): { printf("Worker received: %d\n", *v); }
        case null:    { }
    }
    return null;
}

int main() {
    long long handle = spawn(worker, &g_value);   // &static int: allowed
    join(handle);
    return 0;
}
