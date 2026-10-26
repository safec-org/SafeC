// SafeC Standard Library — File I/O (C11 <stdio.h> wrappers)
// Handles are '&File' references to an opaque type (the C FILE underneath is
// never exposed); File::open returns '?&File', null on failure.
// Operations are methods on the handle: f.read(buf), f.write(buf) (byte
// slices),
// f.seek(off, whence), f.close(), ...
// File conforms to io_traits.h's SeekableStream (Reader, Writer, Seekable)
// and Flusher, so read_exact, write_all, copy_stream, stream_size and
// io_buf.h's BufReader / BufWriter work on it.
#pragma once
#include <std/collections/string.h>

namespace std {

struct File {
    int opaque_;   // never constructed by SafeC code; only referenced

    // ── Open ──────────────────────────────────────────────────────────────────

    // Open a file.  mode: "r", "w", "a", "rb", "wb", etc.
    // Returns a file handle, or null on failure.
    static ?&File open(const &char path, const &char mode);

    // ── Close ─────────────────────────────────────────────────────────────────

    // Close a file handle opened by File::open.  Returns 0 on success.
    int  close() consume;
    void drop();   // close(), ignoring the result

    // ── Read ──────────────────────────────────────────────────────────────────

    // Read at most buf.len bytes into buf.  Returns bytes actually read (0 = EOF),
    // or -1 when an error stopped it before any byte was read.
    long long read([]unsigned char buf);

    // Read a single character.  Returns -1 on EOF/error.
    int  getc();

    // Read a line (up to buf.len-1 characters) into buf, NUL-terminated.
    // Returns buf, or null at EOF/error (or when buf is empty).
    ?&char gets([]char buf);

    // ── Write ─────────────────────────────────────────────────────────────────

    // Write buf.  Returns bytes actually written (fewer than buf.len
    // on an error part-way), or -1 when an error stopped it before any byte.
    long long write([]const unsigned char buf);

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

    // ── Whole files ───────────────────────────────────────────────────────────

    // Appends the whole contents of the file at 'path' to 'out' (binary;
    // 'out' stays NUL-terminated). 1 on success, 0 if it can't be opened
    // or a read fails ('out' then holds whatever was read).
    static int read_to_string(const &char path, &String out);

    // Creates or truncates the file at 'path' and writes 'data' to it.
    // 1 when every byte was written and the file closed cleanly.
    static int write_bytes(const &char path, []const unsigned char data);
};

// Kept for std/ml until its refactor moves to the methods above; forward to
// File::open() / f.close() / f.read() / f.write() / f.seek() / f.tell() /
// f.size(). file_read / file_write keep their old unsigned results (an
// error reads as 0 bytes).
?&File        file_open(const &char path, const &char mode);
int           file_close(&File f);
unsigned long file_read(&File f, []unsigned char buf);
unsigned long file_write(&File f, []const unsigned char buf);
int           file_seek(&File f, long long offset, int whence);
long long     file_tell(&File f);
long long     file_size(&File f);

} // namespace std
