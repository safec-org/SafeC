#import <Foundation/Foundation.h>
#include <string.h>
#include "nslen.h"

long nslen_utf16_length(const char* s) {
    @autoreleasepool {
        NSString* str = [NSString stringWithUTF8String:s];
        return str ? (long)str.length : -1;
    }
}

long nslen_uppercase(const char* s, char* out, long cap) {
    @autoreleasepool {
        NSString* str = [NSString stringWithUTF8String:s];
        if (!str) return -1;
        const char* up = str.uppercaseString.UTF8String;
        long n = (long)strlen(up);
        if (cap > 0) {
            long k = n < cap - 1 ? n : cap - 1;
            memcpy(out, up, (size_t)k);
            out[k] = '\0';
        }
        return n;
    }
}
