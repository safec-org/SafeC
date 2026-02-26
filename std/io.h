// SafeC Standard Library — I/O declarations (C11/C17/C23 stdio coverage)
#pragma once

// ── Seek / EOF constants ──────────────────────────────────────────────────────
#define SC_SEEK_SET  0
#define SC_SEEK_CUR  1
#define SC_SEEK_END  2
#define SC_EOF       (-1)

// ── Formatted output (stdout) ─────────────────────────────────────────────────
void print(const char* s);
void println(const char* s);
void print_char(int c);
void print_int(long long v);
void print_uint(unsigned long long v);
void print_float(double v);
void print_hex(unsigned long long v);
void print_oct(unsigned long long v);
void print_ptr(const void* p);
void println_int(long long v);
void println_uint(unsigned long long v);
void println_float(double v);
void println_char(int c);

// ── Formatted output (stderr) ─────────────────────────────────────────────────
void eprint(const char* s);
void eprintln(const char* s);
void eprint_int(long long v);
void eprint_float(double v);

// ── Formatted output to buffer (snprintf wrappers) ───────────────────────────
// All return the number of characters written (excluding NUL), or -1 on error.
int io_fmt_int(char* buf, int n, long long v);
int io_fmt_uint(char* buf, int n, unsigned long long v);
int io_fmt_float(char* buf, int n, double v);
int io_fmt_float_prec(char* buf, int n, double v, int prec);
int io_fmt_hex(char* buf, int n, unsigned long long v);
int io_fmt_str(char* buf, int n, const char* s);

// ── Input (stdin) ─────────────────────────────────────────────────────────────
// Read a single character from stdin.  Returns the character or SC_EOF.
int  io_getchar();

// Push a character back onto stdin so the next io_getchar/io_scan_* returns it.
int  io_ungetc(int c);

// Read a line (up to n-1 chars + NUL) from stdin into buf.
// Returns number of characters read (excluding NUL), or -1 on EOF/error.
int  io_read_line(char* buf, int n);

// Read a whitespace-delimited token from stdin.
// Returns number of characters read (0 = EOF before token, -1 = error).
int  io_read_token(char* buf, int n);

// Parse a signed decimal integer from stdin. Returns 1 on success, 0 on error.
int  io_scan_int(&stack long long out);

// Parse an unsigned decimal integer from stdin. Returns 1 on success.
int  io_scan_uint(&stack unsigned long long out);

// Parse a floating-point number from stdin. Returns 1 on success.
int  io_scan_float(&stack double out);

// ── File system operations ────────────────────────────────────────────────────
// Remove (delete) a file.  Returns 0 on success, non-zero on error.
int  io_remove(const char* path);

// Rename / move a file.  Returns 0 on success, non-zero on error.
int  io_rename(const char* old_path, const char* new_path);

// Create a temporary file; returns file handle or NULL on error.
void* io_tmpfile();

// ── Flush ─────────────────────────────────────────────────────────────────────
void flush_stdout();
void flush_stderr();
