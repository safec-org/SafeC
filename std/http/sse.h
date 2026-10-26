// SafeC Standard Library — Server-Sent Events (the WHATWG
// text/event-stream format, what the browser's EventSource reads).
//
// Server: a handler takes its connection over with SseStream::start and
// writes events until it is done or the client goes away (send* returns 0):
//
//   struct HttpResponse events(&HttpRequest req, &HttpCtx ctx) {
//       struct SseStream s = std::SseStream::start(ctx.conn, null);
//       while (s.send_event("tick", "{\"n\":1}", null)) { std::thread_sleep_ms(1000UL); }
//       s.close();
//       return std::HttpResponse::new(200);     // ignored: the connection was detached
//   }
//
// The stream blocks the worker thread that runs it; give long-lived
// streams their own thread (or a server with enough workers).
//
// Client: SseClient connects to an http:// URL and yields parsed events;
// SseParser is the incremental parser underneath, usable on any byte
// source.
#pragma once
#include <std/http/http.h>
#include <std/collections/string.h>

namespace std {

struct SseStream {
    struct TcpStream s;
    int              ok;   // 0 once a write failed (the client left)

    // Writes "200 OK" with Content-Type: text/event-stream, Cache-Control:
    // no-cache and 'extra_headers' (null or "Name: value" lines), and
    // detaches the connection.
    static struct SseStream start(&HttpConn conn, const ?&char extra_headers);
    // "data: ..." (one data: line per line of 'data') + blank line.
    int send(const &char data);
    // With an optional event name and id (null to omit).
    int send_event(const ?&char event, const &char data, const ?&char id);
    // ": text" — a comment, the usual keep-alive ping.
    int comment(const &char text);
    // "retry: ms" — the client's reconnection delay.
    int retry(int ms);
    void drop();
    void close() consume;   // = drop()
};

struct SseEvent {
    struct String event;   // "message" when the stream gave none
    struct String data;    // data: lines joined with '\n'
    struct String id;      // the last event id seen (sticky, as EventSource)
    int           retry;   // retry: value from this event, -1 if none
    void drop();

    void free() consume;   // = drop()
};

struct SseParser {
    struct String buf;         // unconsumed bytes
    struct String data;
    struct String event;
    struct String last_id;
    int           has_data;
    int           retry;

    static struct SseParser new();
    // Adds received bytes.
    void feed(const &unsigned char bytes, unsigned long len);
    // The next complete event (1, 'out' filled — free it) or 0 if more
    // input is needed. Comment lines and events without data are skipped.
    int  next(&SseEvent out);
    void drop();
    void free() consume;   // = drop()
};

struct SseClient {
    int              fd;
    int              chunked;   // the response used chunked transfer coding
    unsigned long    chunk_left;
    int              eof;
    struct SseParser parser;
    struct String    raw;       // undecoded bytes when chunked

    // GETs 'url' (http:// only) with Accept: text/event-stream and checks
    // for a 200 text/event-stream response. 'last_event_id' (or null) is
    // sent as Last-Event-ID. *ok = 0 on any failure.
    static struct SseClient connect(const &char url, const ?&char last_event_id, int timeout_ms, ?&int ok);
    // Blocks for the next event: 1 with 'out' filled, 0 once the stream
    // ended or failed.
    int  next(&SseEvent out);
    void drop();
    void close() consume;   // = drop()
};

} // namespace std
