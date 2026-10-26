// SafeC Standard Library — RPC implementation (see rpc.h)
#pragma once
#include <std/rpc/rpc.h>
#include <std/http/http.h>
#include <std/collections/string.h>
#include <std/collections/map.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>
#include <std/panic.h>
// See http.h's top comment: a std/sched/io_nb_*.sc backend and
// std/http/http.sc must both already be included by the caller before this
// file.

namespace std {

[]const unsigned char RpcResult::bytes() const {
    match (self.data) {
        case some(d): { unsafe { return ((const unsigned char*)d)[0..self.len]; } }
        case null: { }
    }
    unsafe { return ((const unsigned char*)"")[0..0]; }
}

void RpcResult::free() { self.drop(); }

void RpcResult::drop() {
    if (self.data != (unsigned char*)0) {
        unsafe { dealloc(self.data); }
    }
    self.error_message.free();
}

// ── gRPC-style message framing: 1 compression-flag byte (always 0 here —
// no compression support) + 4-byte big-endian length + message bytes. ──

static struct String rpc_frame_(const unsigned char* data, unsigned long len) {
    struct String out = String::new();
    unsafe {
        out.push_char((char)0); // compression flag: uncompressed
        out.push_char((char)((len >> 24) & 0xFFUL));
        out.push_char((char)((len >> 16) & 0xFFUL));
        out.push_char((char)((len >> 8) & 0xFFUL));
        out.push_char((char)(len & 0xFFUL));
        if (len > 0UL) { out.push_n_((const char*)data, len); }
    }
    return out;
}

// Returns 1 on success (filling *outData/*outLen with a heap copy the
// caller owns), 0 if 'raw' is too short to contain a valid frame header +
// declared-length payload.
static int rpc_unframe_(const char* raw, unsigned long rawLen,
                         unsigned char** outData, unsigned long* outLen) {
    if (rawLen < 5UL) { return 0; }
    unsigned long len;
    unsafe {
        unsigned char b0 = (unsigned char)raw[1];
        unsigned char b1 = (unsigned char)raw[2];
        unsigned char b2 = (unsigned char)raw[3];
        unsigned char b3 = (unsigned char)raw[4];
        len = ((unsigned long)b0 << 24) | ((unsigned long)b1 << 16) |
              ((unsigned long)b2 << 8) | (unsigned long)b3;
    }
    if (rawLen < 5UL + len) { return 0; }
    unsigned char* copy = (unsigned char*)0;
    if (len > 0UL) {
        unsafe { copy = (unsigned char*)alloc_raw_(len); }
        if (copy == (unsigned char*)0) { return 0; }
        unsafe { safe_memcpy_raw_((void*)copy, (const void*)(raw + 5), len); }
    }
    unsafe { *outData = copy; *outLen = len; }
    return 1;
}

// ── Client ────────────────────────────────────────────────────────────────

struct RpcResult rpc_call(const &char host, unsigned short port, const &char path,
                           []const unsigned char request_data_s) {
    unsigned long request_len = request_data_s.len;
    const &unsigned char request_data;
    unsafe { request_data = (const &unsigned char)request_data_s.ptr; }
    struct RpcResult result;
    result.ok            = 0;
    result.data           = (unsigned char*)0;
    result.len            = 0UL;
    result.error_message  = String::new();

    struct String frame = rpc_frame_(request_data, request_len);
    int httpOk = 0;
    struct HttpResponse resp;
    unsafe {
        resp = http_post_n_(host, port, path, "application/grpc+safec",
                          (const unsigned char*)frame.as_ptr(), frame.length(), &httpOk);
    }
    frame.free();

    if (!httpOk) {
        unsafe { result.error_message.push("transport error (connect/send/recv failed)"); }
        resp.free();
        return result;
    }
    if (resp.status != 200) {
        // A handler-reported failure (rpc_dispatch_'s ok==0 path) puts the
        // real error text in the response body — prefer that over the
        // generic status-code message when present.
        if (!resp.body.is_empty()) {
            unsafe { result.error_message.push(resp.body.as_ptr()); }
        } else {
            unsafe {
                result.error_message.push("server returned HTTP ");
                result.error_message.push_int((long long)resp.status);
            }
        }
        resp.free();
        return result;
    }

    unsigned char* data;
    unsigned long len;
    if (!rpc_unframe_(resp.body.as_ptr(), resp.body.length(), &data, &len)) {
        unsafe { result.error_message.push("malformed response framing"); }
        resp.free();
        return result;
    }
    resp.free();

    result.ok   = 1;
    result.data = data;
    result.len  = len;
    return result;
}

// ── Server ────────────────────────────────────────────────────────────────

static struct StrMap<RpcMethodHandler> rpc_handlers_;
static int rpc_handlers_init_ = 0;

void rpc_register(const &char path, RpcMethodHandler handler) {
    if (rpc_handlers_init_ == 0) {
        rpc_handlers_ = StrMap::new();
        rpc_handlers_init_ = 1;
    }
    rpc_handlers_.insert(path, handler);
}

static struct HttpResponse rpc_dispatch_(&HttpRequest req) {
    const char* path;
    unsafe { path = req->path.as_ptr(); }

    ?&RpcMethodHandler found = null;
    unsafe { if (rpc_handlers_init_ != 0) found = rpc_handlers_.get(path); }

    struct HttpResponse resp;
    resp.headers = String::new();
    unsafe { resp.headers.push("Content-Type: application/grpc+safec\r\n"); }

    if (found.is_null()) {
        resp.status = 404;
        unsafe { resp.body = String::from("no RPC method registered for this path"); }
        return resp;
    }

    unsigned char* reqData;
    unsigned long reqLen;
    const char* bodyPtr;
    unsigned long bodyLen;
    unsafe {
        bodyPtr = req->body.as_ptr();
        bodyLen = req->body.length();
    }
    if (!rpc_unframe_(bodyPtr, bodyLen, &reqData, &reqLen)) {
        resp.status = 400;
        unsafe { resp.body = String::from("malformed request framing"); }
        return resp;
    }

    RpcMethodHandler handlerFn = *expect_ref(found, "rpc handler");
    // An empty payload has no buffer; hand the handler a zero-length view.
    unsigned char emptyPayload[1];
    struct RpcResult result;
    if (reqData == (unsigned char*)0) {
        result = handlerFn(emptyPayload[0..0]);
    } else {
        unsafe { result = handlerFn(((const unsigned char*)reqData)[0..reqLen]); }
    }
    if (reqData != (unsigned char*)0) { unsafe { dealloc(reqData); } }

    if (!result.ok) {
        resp.status = 500;
        unsafe { resp.body = result.error_message.clone(); }
        result.free();
        return resp;
    }

    struct String framed = rpc_frame_(result.data, result.len);
    result.free();
    resp.status = 200;
    resp.body = framed;
    return resp;
}

int rpc_serve(unsigned short port) {
    return http_serve(port, rpc_dispatch_);
}

} // namespace std
