// M4: Closures and Spawn demo
extern int printf(const char *fmt, ...);

int main() {
    // No-param closure spawned on a new thread
    spawn || {
        printf("spawned closure called\n");
    };

    printf("spawn OK\n");
    return 0;
}
