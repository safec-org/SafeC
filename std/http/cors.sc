// SafeC Standard Library — CORS implementation (see cors.h).
#pragma once
#include <std/http/cors.h>
#include <std/http/http.h>
#include <std/collections/string.h>
#include <std/collections/string.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/str.h>

namespace std {

int cors_is_preflight(const &HttpRequest req) {
    int isOptions;
    int hasReqMethod;
    unsafe {
        isOptions = req.method.eq_cstr("OPTIONS");
        hasReqMethod = req.headers.contains("Access-Control-Request-Method");
    }
    return isOptions && hasReqMethod;
}

void cors_apply_headers(&HttpResponse resp, const &char allowedOrigin) {
    int isWildcard;
    unsafe { isWildcard = str_eq(allowedOrigin, "*"); }
    // http_headers_add, not a raw push: the block may or may not end in
    // a CRLF (HttpResponse's constructors leave none).
    http_headers_set(&resp.headers, "Access-Control-Allow-Origin", allowedOrigin);
    if (!isWildcard) {
        http_headers_set(&resp.headers, "Access-Control-Allow-Credentials", "true");
    }
    http_headers_add(&resp.headers, "Vary", "Origin");
}

struct HttpResponse cors_preflight_response(const &char allowedOrigin,
                                             const &char allowedMethods,
                                             const &char allowedHeaders) {
    struct HttpResponse resp;
    resp.status = 204;
    resp.headers = String::new();
    http_headers_add(&resp.headers, "Access-Control-Allow-Methods", allowedMethods);
    http_headers_add(&resp.headers, "Access-Control-Allow-Headers", allowedHeaders);
    http_headers_add(&resp.headers, "Access-Control-Max-Age", "86400");
    resp.body = String::new();
    cors_apply_headers(&resp, allowedOrigin);
    return resp;
}

} // namespace std
