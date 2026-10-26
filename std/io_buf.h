#pragma once
// SafeC Standard Library — buffered streams over any Reader / Writer.
//
// BufReader<R> and BufWriter<W> borrow a stream (io_traits.h's Reader /
// Writer: File, Pipe, TcpStream, ...) and put an IO_BUF_SIZE-byte buffer in
// front of it: many small reads or writes become one call on the stream.
// Each is a Reader / Writer itself, so read_exact, write_all, copy_stream
// and any generic<R: Reader> code take it like the stream it wraps.
//
//   struct TcpStream s = TcpStream::from_fd(fd);
//   struct BufReader<TcpStream> in = BufReader::over(&s);
//   struct String head = String::new();
//   if (in.read_until(&head, "\r\n\r\n")) { ... }   // the header block
//   read_exact(&in, body[0..len]);                   // bytes after it
//
//   struct BufWriter<TcpStream> out = BufWriter::over(&s);
//   write_all(&out, hdr[0..6]);                      // buffered
//   write_all(&out, payload.as_bytes());
//   if (out.flush() != 0) { ... }                    // one write to s
//
// A BufWriter holds what was written until flush() (or until its buffer
// fills): call flush() before the stream is used directly or closed.
#include <std/io_traits.h>
#include <std/mem.h>
#include <std/collections/string.h>

#define IO_BUF_SIZE 4096

namespace std {

// ── BufReader ───────────────────────────────────────────────────────────────

generic<R: Reader>
struct BufReader {
    &R            inner;    // the stream read from (borrowed)
    unsigned long pos;      // next unread byte in buf
    unsigned long len;      // bytes in buf
    unsigned char buf[IO_BUF_SIZE];

    static struct BufReader<R> over(&R inner);

    // Reader: at most n bytes, from the buffer first. A read of at least
    // IO_BUF_SIZE bytes with nothing buffered goes straight to the stream.
    // The count read, 0 at the stream's end, -1 on error.
    long long     read([]unsigned char out);

    // Appends bytes to 'out' up to and including the first occurrence of
    // 'term' (a non-empty string; "\n" reads a line, "\r\n\r\n" an HTTP
    // header block). 1 when 'term' arrived; 0 when the stream ended or
    // failed first ('out' then holds what did arrive). Bytes after 'term'
    // stay buffered for the next read.
    int           read_until(&String out, const &char term);

    // Bytes already read from the stream and not yet returned: a read of
    // at most this many doesn't touch the stream.
    unsigned long buffered() const;
};

generic<R: Reader>
struct BufReader<R> BufReader::over(&R inner) {
    struct BufReader<R> b;
    b.inner = inner;
    b.pos = 0UL;
    b.len = 0UL;
    return b;
}

generic<R: Reader>
long long BufReader::read([]unsigned char out_s) {
    unsigned long n = out_s.len;
    if (n == 0UL) { return 0LL; }
    &unsigned char out;
    unsafe { out = (&unsigned char)out_s.ptr; }
    if (self.pos == self.len) {
        if (n >= (unsigned long)IO_BUF_SIZE) { return self.inner.read(out_s); }
        long long k = self.inner.read(self.buf[0..(unsigned long)IO_BUF_SIZE]);
        if (k <= 0LL) { return k; }
        self.pos = 0UL;
        self.len = (unsigned long)k;
    }
    unsigned long m = self.len - self.pos;
    if (m > n) { m = n; }
    unsafe { safe_memcpy_raw_((unsigned char*)out, (const unsigned char*)&self.buf[self.pos], m); }
    self.pos = self.pos + m;
    return (long long)m;
}

// After 'matched' bytes of 'term' and then byte c: how many bytes of 'term'
// the input now ends with (the longest prefix of 'term' that is a suffix of
// term[0..matched) + c).
static unsigned long io_term_step_(const char* term, unsigned long matched, unsigned char c) {
    unsafe {
        if ((unsigned char)term[matched] == c) { return matched + 1UL; }
        unsigned long k = matched;
        while (k > 0UL) {
            // candidate: term[0..k) == term[matched-k+1..matched) + c
            if ((unsigned char)term[k - 1UL] == c) {
                unsigned long j = 0UL;
                while (j + 1UL < k && term[j] == term[matched - k + 1UL + j]) { j = j + 1UL; }
                if (j + 1UL == k) { return k; }
            }
            k = k - 1UL;
        }
    }
    return 0UL;
}

generic<R: Reader>
int BufReader::read_until(&String out, const &char term) {
    const char* t = term;
    unsigned long tl = 0UL;
    unsafe { while (t[tl] != (char)0) { tl = tl + 1UL; } }
    if (tl == 0UL) { return 1; }
    unsigned long matched = 0UL;
    while (1) {
        if (self.pos == self.len) {
            long long k = self.inner.read(self.buf[0..(unsigned long)IO_BUF_SIZE]);
            if (k <= 0LL) { return 0; }
            self.pos = 0UL;
            self.len = (unsigned long)k;
        }
        unsigned long i = self.pos;
        while (i < self.len) {
            matched = io_term_step_(t, matched, self.buf[i]);
            i = i + 1UL;
            if (matched == tl) {
                unsafe { out.push_n_((const &char)(const char*)&self.buf[self.pos], i - self.pos); }
                self.pos = i;
                return 1;
            }
        }
        unsafe { out.push_n_((const &char)(const char*)&self.buf[self.pos], self.len - self.pos); }
        self.pos = self.len;
    }
    return 0;
}

generic<R: Reader>
unsigned long BufReader::buffered() const { return self.len - self.pos; }

// ── BufWriter ───────────────────────────────────────────────────────────────

generic<W: Writer>
struct BufWriter {
    &W            inner;    // the stream written to (borrowed)
    unsigned long len;      // bytes waiting in buf
    unsigned char buf[IO_BUF_SIZE];

    static struct BufWriter<W> over(&W inner);

    // Writer: takes all of data (into the buffer, or, for a write that
    // doesn't fit, after flushing it, straight to the stream when it is at
    // least IO_BUF_SIZE). data.len, or -1 if the stream failed.
    long long     write([]const unsigned char data);

    // Flusher: writes out everything buffered. 0 on success, -1 if the
    // stream failed (the buffered bytes are dropped either way).
    int           flush();
};

generic<W: Writer>
struct BufWriter<W> BufWriter::over(&W inner) {
    struct BufWriter<W> b;
    b.inner = inner;
    b.len = 0UL;
    return b;
}

generic<W: Writer>
long long BufWriter::write([]const unsigned char data_s) {
    unsigned long n = data_s.len;
    if (n == 0UL) { return 0LL; }
    const &unsigned char data;
    unsafe { data = (const &unsigned char)data_s.ptr; }
    if (n > (unsigned long)IO_BUF_SIZE - self.len) {
        if (self.flush() != 0) { return -1LL; }
        if (n >= (unsigned long)IO_BUF_SIZE) {
            return write_all(self.inner, data_s) ? (long long)n : -1LL;
        }
    }
    unsafe { safe_memcpy_raw_((unsigned char*)&self.buf[self.len], (const unsigned char*)data, n); }
    self.len = self.len + n;
    return (long long)n;
}

generic<W: Writer>
int BufWriter::flush() {
    if (self.len == 0UL) { return 0; }
    int ok = write_all(self.inner, self.buf[0..self.len]);
    self.len = 0UL;
    return ok ? 0 : -1;
}

} // namespace std
