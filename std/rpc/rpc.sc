// SafeC Standard Library — RPC implementation (see rpc.h)
#pragma once
#include <std/rpc/rpc.h>
#include <std/http/http.h>
#include <std/collections/string.sc>
#include <std/collections/map.sc>
#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>
#include <std/str.sc>
// See http.h's top comment: a std/sched/io_nb_*.sc backend and
// std/http/http.sc must both already be included by the caller before this
// file.

namespace std {

void RpcResult::free() {
    if (self.data != (unsigned char*)0) {
        unsafe { dealloc((void*)self.data); }
    }
    self.error_message.free();
}

// ── gRPC-style message framing: 1 compression-flag byte (always 0 here —
// no compression support) + 4-byte big-endian length + message bytes. ──

static struct String rpc_frame_(const unsigned char* data, unsigned long len) {
    struct String out = string_new();
    unsafe {
        out.push_char((char)0); // compression flag: uncompressed
        out.push_char((char)((len >> 24) & 0xFFUL));
        out.push_char((char)((len >> 16) & 0xFFUL));
        out.push_char((char)((len >> 8) & 0xFFUL));
        out.push_char((char)(len & 0xFFUL));
        if (len > 0UL) { out.push_n((const char*)data, len); }
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
        unsafe { copy = (unsigned char*)alloc(len); }
        if (copy == (unsigned char*)0) { return 0; }
        unsafe { safe_memcpy((void*)copy, (const void*)(raw + 5), len); }
    }
    unsafe { *outData = copy; *outLen = len; }
    return 1;
}

// ── Client ────────────────────────────────────────────────────────────────

struct RpcResult rpc_call(const char* host, unsigned short port, const char* path,
                           const unsigned char* request_data, unsigned long request_len) {
    struct RpcResult result;
    result.ok            = 0;
    result.data           = (unsigned char*)0;
    result.len            = 0UL;
    result.error_message  = string_new();

    struct String frame = rpc_frame_(request_data, request_len);
    int httpOk = 0;
    struct HttpResponse resp;
    unsafe {
        resp = http_post(host, port, path, "application/grpc+safec",
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

static struct HashMap rpc_handlers_;
static int rpc_handlers_init_ = 0;

void rpc_register(const char* path, RpcMethodHandler handler) {
    if (rpc_handlers_init_ == 0) {
        rpc_handlers_ = str_map_new(sizeof(void*));
        rpc_handlers_init_ = 1;
    }
    void* h;
    unsafe { h = (void*)handler; }
    unsafe { str_map_insert(&rpc_handlers_, path, (const void*)&h); }
}

static struct HttpResponse rpc_dispatch_(struct HttpRequest* req) {
    const char* path;
    unsafe { path = req->path.as_ptr(); }

    void* found = (void*)0;
    if (rpc_handlers_init_ != 0) {
        void* slot;
        unsafe { slot = str_map_get(&rpc_handlers_, path); }
        if (slot != (void*)0) {
            unsafe { found = *(void**)slot; }
        }
    }

    struct HttpResponse resp;
    resp.headers = string_new();
    unsafe { resp.headers.push("Content-Type: application/grpc+safec\r\n"); }

    if (found == (void*)0) {
        resp.status = 404;
        unsafe { resp.body = string_from("no RPC method registered for this path"); }
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
        unsafe { resp.body = string_from("malformed request framing"); }
        return resp;
    }

    RpcMethodHandler handlerFn;
    unsafe { handlerFn = (RpcMethodHandler)found; }
    struct RpcResult result = handlerFn(reqData, reqLen);
    if (reqData != (unsigned char*)0) { unsafe { dealloc((void*)reqData); } }

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
