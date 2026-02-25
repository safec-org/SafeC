// SafeC Standard Library — I/O (C11/C17/C23 stdio coverage)
// Wraps stdio functions for formatted, raw, and interactive I/O.
// Explicit extern declarations are used because macOS stdio.h uses FILE*,
// __attribute__, and macros that SafeC's parser cannot handle directly.
#pragma once
#include "io.h"

// ── Explicit extern declarations for stdio functions ─────────────────────────
extern int   printf(const char* fmt, ...);
extern int   fprintf(void* stream, const char* fmt, ...);
extern int   fputs(const char* s, void* stream);
extern int   fputc(int c, void* stream);
extern int   fflush(void* stream);
extern int   puts(const char* s);
extern int   putchar(int c);
extern char* fgets(char* buf, int n, void* stream);
extern int   getchar();
extern int   ungetc(int c, void* stream);
extern int   scanf(const char* fmt, ...);
extern int   fscanf(void* stream, const char* fmt, ...);
extern int   sscanf(const char* buf, const char* fmt, ...);
extern int   snprintf(char* buf, unsigned long n, const char* fmt, ...);
extern int   remove(const char* path);
extern int   rename(const char* oldpath, const char* newpath);
extern void* tmpfile();
extern unsigned long strlen(const char* s);

// ── FILE* stream handles ──────────────────────────────────────────────────────
// On macOS/BSD, stdout/stderr/stdin are macros expanding to __stdoutp etc.
// We declare the underlying globals using void* to avoid needing FILE struct.
// TODO: add Linux support (#define via __iob_func or extern FILE* stdout)
extern void* __stdinp;
extern void* __stdoutp;
extern void* __stderrp;

// ── Basic stdout output ───────────────────────────────────────────────────────

// Print a string without a trailing newline.
void print(const char* s) {
    unsafe { fputs(s, __stdoutp); }
}

// Print a string followed by a newline.
void println(const char* s) {
    unsafe { puts(s); }
}

// Print a string to stderr without a trailing newline.
void eprint(const char* s) {
    unsafe { fputs(s, __stderrp); }
}

// Print a string to stderr followed by a newline.
void eprintln(const char* s) {
    unsafe {
        fputs(s, __stderrp);
        fputc('\n', __stderrp);
    }
}

// Print a single character to stdout.
void print_char(int c) {
    unsafe { putchar(c); }
}

// Print a signed 64-bit integer to stdout.
void print_int(long long v) {
    unsafe { printf("%lld", v); }
}

// Print a double to stdout.
void print_float(double v) {
    unsafe { printf("%g", v); }
}

// Print a signed 64-bit integer followed by newline.
void println_int(long long v) {
    unsafe { printf("%lld\n", v); }
}

// Print a double followed by newline.
void println_float(double v) {
    unsafe { printf("%g\n", v); }
}

// Flush stdout — call before program exit if using print() without println().
void flush_stdout() {
    unsafe { fflush(__stdoutp); }
}

// Flush stderr.
void flush_stderr() {
    unsafe { fflush(__stderrp); }
}

// ── Additional stdout output ───────────────────────────────────────────────────

void print_uint(unsigned long long v) {
    unsafe { printf("%llu", v); }
}

void print_hex(unsigned long long v) {
    unsafe { printf("0x%llx", v); }
}

void print_oct(unsigned long long v) {
    unsafe { printf("0%llo", v); }
}

void print_ptr(const void* p) {
    unsafe { printf("%p", p); }
}

void println_uint(unsigned long long v) {
    unsafe { printf("%llu\n", v); }
}

void println_char(int c) {
    unsafe {
        putchar(c);
        putchar('\n');
    }
}

// ── Stderr output ─────────────────────────────────────────────────────────────

void eprint_int(long long v) {
    unsafe { fprintf(__stderrp, "%lld", v); }
}

void eprint_float(double v) {
    unsafe { fprintf(__stderrp, "%g", v); }
}

// ── Buffer formatting (snprintf wrappers) ─────────────────────────────────────

int io_fmt_int(char* buf, int n, long long v) {
    unsafe { return snprintf(buf, (unsigned long)n, "%lld", v); }
}

int io_fmt_uint(char* buf, int n, unsigned long long v) {
    unsafe { return snprintf(buf, (unsigned long)n, "%llu", v); }
}

int io_fmt_float(char* buf, int n, double v) {
    unsafe { return snprintf(buf, (unsigned long)n, "%g", v); }
}

int io_fmt_float_prec(char* buf, int n, double v, int prec) {
    unsafe { return snprintf(buf, (unsigned long)n, "%.*f", prec, v); }
}

int io_fmt_hex(char* buf, int n, unsigned long long v) {
    unsafe { return snprintf(buf, (unsigned long)n, "0x%llx", v); }
}

int io_fmt_str(char* buf, int n, const char* s) {
    unsafe { return snprintf(buf, (unsigned long)n, "%s", s); }
}

// ── Input ─────────────────────────────────────────────────────────────────────

// Read a single character from stdin.  Returns the character or SC_EOF (-1).
int io_getchar() {
    unsafe { return getchar(); }
}

// Push a character back so the next read returns it.
int io_ungetc(int c) {
    unsafe { return ungetc(c, __stdinp); }
}

// Read a line (up to n-1 chars + NUL) from stdin.
// Returns the number of characters read (excluding NUL), or -1 on EOF/error.
int io_read_line(char* buf, int n) {
    if (n <= 0) { return -1; }
    unsafe {
        if (fgets(buf, n, __stdinp) == (char*)0) { return -1; }
        int len = (int)strlen(buf);
        // Strip trailing newline if present.
        if (len > 0 && buf[len - 1] == '\n') {
            len = len - 1;
            buf[len] = '\0';
        }
        return len;
    }
}

// Read one whitespace-delimited token from stdin into buf[0..n-1].
// Returns number of chars read, 0 = EOF before any token, -1 = buffer too small.
int io_read_token(char* buf, int n) {
    if (n <= 0) { return -1; }
    unsafe {
        int c = getchar();
        // Skip leading whitespace
        while (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            c = getchar();
        }
        if (c == -1) { return 0; }
        ungetc(c, __stdinp);
        char fmt[16];
        snprintf(fmt, (unsigned long)16, "%%%ds", n - 1);
        int read = scanf(fmt, buf);
        if (read == 1) { return (int)strlen(buf); }
        return 0;
    }
}

// Parse a signed decimal integer from stdin.  Returns 1 on success.
int io_scan_int(long long* out) {
    unsafe {
        int r = scanf("%lld", out);
        return r == 1;
    }
}

// Parse an unsigned decimal integer from stdin.  Returns 1 on success.
int io_scan_uint(unsigned long long* out) {
    unsafe {
        int r = scanf("%llu", out);
        return r == 1;
    }
}

// Parse a floating-point number from stdin.  Returns 1 on success.
int io_scan_float(double* out) {
    unsafe {
        int r = scanf("%lf", out);
        return r == 1;
    }
}

// ── File system ───────────────────────────────────────────────────────────────

// Delete a file.  Returns 0 on success, non-zero on error.
int io_remove(const char* path) {
    unsafe { return remove(path); }
}

// Rename / move a file.  Returns 0 on success, non-zero on error.
int io_rename(const char* old_path, const char* new_path) {
    unsafe { return rename(old_path, new_path); }
}

// Create a temporary file opened in "w+b" mode.  Returns handle or NULL.
void* io_tmpfile() {
    unsafe { return tmpfile(); }
}
