// extracted from safec-docs reference/ffi.md (block 1)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    ?&char buf = malloc(256);              // raw void* -> ?&char: implicit
    match (buf) {
        case some(b): {
            sprintf(b, "Hello, %s!", "world");   // &char -> char*: implicit
            printf("%s\n", b);
        }
        case null: { }
    }
    free(buf);
    return 0;
}
