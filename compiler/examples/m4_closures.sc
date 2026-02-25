// M4: Closures and Spawn demo
extern int printf(const char *fmt, ...);

int main() {
    // Test closure parsing - |params| { body }
    // Closures are lowered to function pointers
    spawn |x: int| {
        printf("spawned closure called\n");
    };
    
    printf("spawn OK\n");
    return 0;
}
