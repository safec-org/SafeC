#pragma once
// SafeC Standard Library — byte-stream traits.
//
// std's blocking byte streams share one read/write shape, so the loops
// every caller needs around them (read until full, write until done, copy
// one stream into another) are written once here. One trait per
// capability, composed into the roles generic code asks for:
//
//   Reader          read(buf): File, Pipe, TcpStream, TlsStream, BufReader<R>
//   Writer          write(buf): File, Pipe, TcpStream, TlsStream, BufWriter<W>
//   Seekable        seek(off, whence) / tell(): File
//   Flusher         flush(): File, BufWriter<W>
//   Stream          Reader + Writer: File, Pipe, TcpStream
//   SeekableStream  Stream + Seekable: File
//
//   File      (io_file.h)    f.read / f.write on an open file
//   Pipe      (ipc/pipe.h)   read end / write end of an anonymous pipe
//   TcpStream (http/http.h)  a connected, blocking TCP socket
//   BufReader / BufWriter (io_buf.h)  buffering over any Reader / Writer
//
// Traits are structural: a type conforms by having the methods, so your own
// stream works with read_exact / write_all / copy_stream (and with any
// generic<R: Reader> code) as soon as it defines read or write with these
// signatures. Each call is resolved at compile time (no dispatch at run
// time).
//
//   ?&File in = File::open("in.bin", "rb");
//   struct Pipe p; p.init();
//   ...
//   long long n = copy_stream(f, &p);      // R = File, W = Pipe
//   if (!write_all(&p, hdr[0..16])) { ... }
//
// Not conforming, on purpose:
//   WsConn  (http/websocket.h)  message-based: recv() yields whole frames.
//   TcpConn (net/tcp.h)         a freestanding, non-blocking segment buffer:
//                               read() returning 0 means "nothing yet", not
//                               EOF, so the loops below would misread it.
#include <std/mem.h>

namespace std {

trait Reader {
    // Reads at most buf.len bytes into buf. Returns the count read, 0 at
    // end of stream, -1 on error. May return fewer than buf.len before
    // the end.
    long long read([]unsigned char buf);
}

trait Writer {
    // Writes at most buf.len bytes from buf. Returns the count written
    // (may be fewer), -1 on error.
    long long write([]const unsigned char buf);
}

trait Seekable {
    // Moves to 'offset' from 'whence' (0 = start, 1 = current, 2 = end).
    // 0 on success.
    int seek(long long offset, int whence);
    // The current offset, -1 on error.
    long long tell();
}

trait Flusher {
    // Pushes buffered writes on to where they go. 0 on success.
    int flush();
}

trait Stream : Reader + Writer {}

trait SeekableStream : Stream + Seekable {}

// Fills buf completely, looping over short reads. 1 when all buf.len bytes
// arrived; 0 when the stream ended or failed first (buf then holds what did
// arrive).
generic<R: Reader>
int read_exact(&R r, []unsigned char buf) {
    unsigned long got = 0UL;
    while (got < buf.len) {
        long long k = r.read(buf[got..buf.len]);
        if (k <= 0LL) { return 0; }
        got = got + (unsigned long)k;
    }
    return 1;
}

// Writes all of buf, looping over short writes. 1 on success, 0 if the
// stream failed (or accepted nothing) first.
generic<W: Writer>
int write_all(&W w, []const unsigned char buf) {
    unsigned long sent = 0UL;
    while (sent < buf.len) {
        long long k = w.write(buf[sent..buf.len]);
        if (k <= 0LL) { return 0; }
        sent = sent + (unsigned long)k;
    }
    return 1;
}

// Copies r into w until r's end. Returns the byte count copied, or -1 if a
// read or write failed (what was copied before stays written).
generic<R: Reader, W: Writer>
long long copy_stream(&R r, &W w) {
    unsigned char chunk[4096];
    long long total = 0LL;
    while (1) {
        long long k = r.read(chunk[..]);
        if (k == 0LL) { return total; }
        if (k < 0LL) { return -1LL; }
        if (!write_all(w, chunk[0..(unsigned long)k])) { return -1LL; }
        total = total + k;
    }
    return total;
}

// The stream's length in bytes (its end offset), leaving the position
// where it was; -1 on error.
generic<S: Seekable>
long long stream_size(&S s) {
    long long cur = s.tell();
    if (cur < 0LL) { return -1LL; }
    if (s.seek(0LL, 2) != 0) { return -1LL; }
    long long end = s.tell();
    if (s.seek(cur, 0) != 0) { return -1LL; }
    return end;
}

} // namespace std
