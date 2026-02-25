extern int printf(const char *fmt, ...);

generic<T...>
int safe_printf(const char *fmt, T... args) {
    return printf(fmt, args);
}

int main() {
    safe_printf("hello %d %f\n", 42, 3.14);
    return 0;
}
