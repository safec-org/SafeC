// variadic generic packs: T..., T... args, sizeof...(T), args[i], forwarding
#include <stdio.h>
generic<T...>
unsigned long count_args(T... args) { return sizeof...(T); }
generic<T...>
int sum3(T... args) { return args[0] + args[1] + args[2]; }
generic<T...>
int fwd(T... args) { return sum3(args); }
generic<T...>
void show(const char* fmt, T... args) { printf(fmt, args); }
int main() {
    printf("%lu %lu %lu\n", count_args(), count_args(1), count_args(1, 2.0, "x"));
    printf("%d %d\n", sum3(1, 2, 3), fwd(4, 5, 6));
    show("%d-%s\n", 7, "seven");
    return 0;
}
