// closures.sc — M4: Closure lowering
// Closures are lowered to internal LLVM functions.
// They can be passed to spawn to run on a thread.
extern int printf(const char *fmt, ...);

int main() {
    // No-param closure
    spawn || {
        printf("hello from closure\n");
    };

    printf("spawned OK\n");
    return 0;
}
