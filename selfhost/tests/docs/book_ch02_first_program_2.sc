// extracted from safec-docs book/ch02-first-program.md (block 2)
#include <stdio.h>

int main() {
    printf("Celsius  Fahrenheit\n");
    for (int c = 0; c <= 100; c = c + 10) {
        double f = (double)c * 9.0 / 5.0 + 32.0;
        printf("%7d  %10.1f\n", c, f);
    }
    return 0;
}
