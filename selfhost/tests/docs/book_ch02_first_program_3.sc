// extracted from safec-docs book/ch02-first-program.md (block 3)
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        unsafe { printf("usage: %s <celsius>...\n", argv[0]); }
        return 1;
    }
    for (int i = 1; i < argc; i = i + 1) {
        double c;
        unsafe { c = atof(argv[i]); }
        double f = c * 9.0 / 5.0 + 32.0;
        printf("%.1f C = %.1f F\n", c, f);
    }
    return 0;
}
