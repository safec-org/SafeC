// Calls an Objective-C library through its C API (nslen.h).
#include <nslen.h>

extern int printf(const char* fmt, ...);

int main() {
    char upper[32];
    unsafe {
        long len = nslen_utf16_length("h\xc3\xa9llo");       // "héllo": 5 UTF-16 units, 6 bytes
        long n = nslen_uppercase("h\xc3\xa9llo", &upper[0], 32L);
        printf("length %ld, uppercase %s (%ld bytes)\n", len, &upper[0], n);
        if (len != 5L || n != 6L) return 1;
    }
    return 0;
}
