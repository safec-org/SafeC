// SafeC Standard Library — File I/O (C11 <stdio.h> wrappers)
// Handles are '&File' references to an opaque type (the C FILE underneath is
// never exposed); file_open returns '?&File', null on failure.
// Operations are methods on the handle: f.read(buf, n), f.write(buf, n),
// f.seek(off, whence), f.close(), ...
#pragma once

namespace std {

struct File {
    int opaque_;   // never constructed by SafeC code; only referenced

    // ── Close ─────────────────────────────────────────────────────────────────

    // Close a file handle opened by file_open.  Returns 0 on success.
    int  close();

    // ── Read ──────────────────────────────────────────────────────────────────

    // Read at most n bytes into buf.  Returns bytes actually read (0 = EOF).
    unsigned long read(&unsigned char buf, unsigned long n);

    // Read a single character.  Returns -1 on EOF/error.
    int  getc();

    // Read up to n-1 characters into buf, NUL-terminate.  Returns buf or NULL.
    ?&char gets(&char buf, int n);

    // ── Write ─────────────────────────────────────────────────────────────────

    // Write n bytes from buf.  Returns bytes actually written.
    unsigned long write(const &unsigned char buf, unsigned long n);

    // Write a single character.  Returns c on success, -1 on error.
    int  putc(int c);

    // Write a string (no newline).  Returns non-negative on success.
    int  puts(const &char s);

    // ── Seek / tell ───────────────────────────────────────────────────────────

    // Seek to byte offset from whence (0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END).
    int  seek(long long offset, int whence);

    // Return current file position, or -1 on error.
    long long tell();

    // Rewind to start of file.
    void rewind();

    // ── Status ────────────────────────────────────────────────────────────────

    // Return non-zero if the last operation hit EOF.
    int  eof();

    // Return non-zero if the file has an error indicator set.
    int  error();

    // Flush buffered writes.  Returns 0 on success.
    int  flush();

    // ── Convenience ───────────────────────────────────────────────────────────

    // Return the size of the file in bytes, or -1 on error.
    // (Seeks to end, reads position, seeks back.)
    long long size();
};

// Open a file.  mode: "r", "w", "a", "rb", "wb", etc.
// Returns a file handle, or null on failure.
?&File file_open(const &char path, const &char mode);

// Kept for std/ml until its refactor moves to the methods above; forward to
// f.close() / f.read() / f.write() / f.seek() / f.tell() / f.size().
int           file_close(&File f);
unsigned long file_read(&File f, &unsigned char buf, unsigned long n);
unsigned long file_write(&File f, const &unsigned char buf, unsigned long n);
int           file_seek(&File f, long long offset, int whence);
long long     file_tell(&File f);
long long     file_size(&File f);

} // namespace std
