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
    // A detached-capable thread may only see static data; read-only here.
    long long handle = spawn(worker, (const &static int)&g_value);
    join(handle);
    return 0;
}
