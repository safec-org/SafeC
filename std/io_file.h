// SafeC Standard Library — File I/O (C11 <stdio.h> wrappers)
// Handles are '&File' references to an opaque type (the C FILE underneath is
// never exposed); file_open returns '?&File', null on failure.
#pragma once

// ── Open / close ──────────────────────────────────────────────────────────────

// Open a file.  mode: "r", "w", "a", "rb", "wb", etc.
// Returns a file handle, or null on failure.
namespace std {

struct File { int opaque_; };   // never constructed by SafeC code; only referenced

?&File file_open(const &char path, const &char mode);

// Close a file handle opened by file_open.  Returns 0 on success.
int   file_close(&File f);

// ── Read ──────────────────────────────────────────────────────────────────────

// Read at most n bytes into buf.  Returns bytes actually read (0 = EOF).
unsigned long file_read(&File f, &unsigned char buf, unsigned long n);

// Read a single character.  Returns -1 on EOF/error.
int  file_getc(&File f);

// Read up to n-1 characters into buf, NUL-terminate.  Returns buf or NULL.
?&char file_gets(&char buf, int n, &File f);

// ── Write ─────────────────────────────────────────────────────────────────────

// Write n bytes from buf.  Returns bytes actually written.
unsigned long file_write(&File f, const &unsigned char buf, unsigned long n);

// Write a single character.  Returns c on success, -1 on error.
int  file_putc(int c, &File f);

// Write a string (no newline).  Returns non-negative on success.
int  file_puts(const &char s, &File f);

// ── Seek / tell ───────────────────────────────────────────────────────────────

// Seek to byte offset from whence (0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END).
int  file_seek(&File f, long long offset, int whence);

// Return current file position, or -1 on error.
long long file_tell(&File f);

// Rewind to start of file.
void file_rewind(&File f);

// ── Status ────────────────────────────────────────────────────────────────────

// Return non-zero if the last operation hit EOF.
int  file_eof(&File f);

// Return non-zero if the file has an error indicator set.
int  file_error(&File f);

// Flush buffered writes.  Returns 0 on success.
int  file_flush(&File f);

// ── Convenience ───────────────────────────────────────────────────────────────

// Return the size of the file in bytes, or -1 on error.
// (Seeks to end, reads position, seeks back.)
long long file_size(&File f);

} // namespace std
