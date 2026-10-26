// The scc runtime, compiled into every program (as the pseudo-file
// "<runtime>"). Host services come from three wasm imports; everything else
// (allocator, string functions, printf-style formatting, math) is SafeC.
extern int __host_write(int fd, const char* buf, int len);
extern int __host_read(int fd, char* buf, int len);
extern void __host_exit(int code);

void __rt_trap(int code) { __host_exit(code); }

void exit(int code) { __host_exit(code); }
void abort() { __host_exit(134); }

int putchar(int c) {
    char b[1];
    b[0] = (char)c;
    unsafe { __host_write(1, (const char*)&b[0], 1); }
    return c;
}

unsigned long strlen(const char* s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    return n;
}

int puts(const char* s) {
    unsafe { __host_write(1, s, (int)strlen(s)); }
    putchar(10);
    return 0;
}

char __rt_buf[512];
int __rt_len = 0;

void __rt_flush() {
    if (__rt_len > 0) {
        unsafe { __host_write(1, (const char*)&__rt_buf[0], __rt_len); }
        __rt_len = 0;
    }
}

void __rt_put(int c) {
    if (__rt_len >= 512) { __rt_flush(); }
    __rt_buf[__rt_len] = (char)c;
    __rt_len = __rt_len + 1;
}

void __rt_puts(const char* s) {
    unsafe {
        int i = 0;
        while (s[i] != (char)0) { __rt_put((int)s[i]); i = i + 1; }
    }
}

void __rt_putu(unsigned long v) {
    char d[24];
    int n = 0;
    if (v == 0UL) { __rt_put(48); return; }
    while (v > 0UL) { d[n] = (char)(48UL + v % 10UL); v = v / 10UL; n = n + 1; }
    while (n > 0) { n = n - 1; __rt_put((int)d[n]); }
}

int printf(const char* fmt, ...) {
    unsafe {
        unsigned char* ap = __va_ptr();
        int ai = 0;
        int i = 0;
        while (fmt[i] != (char)0) {
            char c = fmt[i];
            if (c != '%') { __rt_put((int)c); i = i + 1; continue; }
            i = i + 1;
            c = fmt[i];
            while (c == 'l') { i = i + 1; c = fmt[i]; }
            long* slot = (long*)(ap + 8 * ai);
            if (c == 'd' || c == 'i') {
                long v = *slot;
                if (v < 0L) { __rt_put(45); v = 0L - v; }
                __rt_putu((unsigned long)v);
                ai = ai + 1;
            } else if (c == 'u') {
                __rt_putu((unsigned long)*slot);
                ai = ai + 1;
            } else if (c == 's') {
                const char* s = (const char*)(int)*slot;
                __rt_puts(s);
                ai = ai + 1;
            } else if (c == 'c') {
                __rt_put((int)*slot);
                ai = ai + 1;
            } else if (c == '%') {
                __rt_put(37);
            }
            i = i + 1;
        }
    }
    __rt_flush();
    return 0;
}
