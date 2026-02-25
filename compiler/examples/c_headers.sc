// c_headers.sc — native C header import
//
// Demonstrates that SafeC can #include standard C headers directly.
// The compiler invokes clang behind the scenes to extract clean extern
// declarations; no manual `extern` stubs are needed anywhere.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // stdio.h: printf, puts
    printf("Hello from SafeC with native C headers!\n");
    puts("puts() works too.");

    // stdlib.h: malloc / free — raw pointer ops live in unsafe{}
    unsafe {
        int* buf = (int*)malloc((unsigned long)(4 * 4));
        buf[0] = 10;
        buf[1] = 20;
        buf[2] = 30;
        buf[3] = 40;
        printf("buf[2] = %d\n", buf[2]);
        free((void*)buf);
    }

    // string.h: strlen
    char* msg = "SafeC";
    printf("strlen(\"%s\") = %lu\n", msg, strlen(msg));

    return 0;
}
