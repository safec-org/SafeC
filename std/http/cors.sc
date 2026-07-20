// SafeC Standard Library — CORS implementation (see cors.h).
#pragma once
#include <std/http/cors.h>
#include <std/http/http.h>
#include <std/collections/string.h>
#include <std/collections/string.sc>
#include <std/mem.sc>
#include <std/str.h>
#include <std/str.sc>

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

void cors_apply_headers(&HttpResponse resp, const char* allowedOrigin) {
    int isWildcard;
    unsafe { isWildcard = str_eq(allowedOrigin, "*"); }
    unsafe {
        resp.headers.push("Access-Control-Allow-Origin: ");
        resp.headers.push(allowedOrigin);
        resp.headers.push("\r\n");
        if (!isWildcard) {
            resp.headers.push("Access-Control-Allow-Credentials: true\r\n");
        }
        resp.headers.push("Vary: Origin\r\n");
    }
}

struct HttpResponse cors_preflight_response(const char* allowedOrigin,
                                             const char* allowedMethods,
                                             const char* allowedHeaders) {
    struct HttpResponse resp;
    resp.status = 204;
    resp.headers = string_new();
    unsafe {
        resp.headers.push("Access-Control-Allow-Methods: ");
        resp.headers.push(allowedMethods);
        resp.headers.push("\r\n");
        resp.headers.push("Access-Control-Allow-Headers: ");
        resp.headers.push(allowedHeaders);
        resp.headers.push("\r\n");
        resp.headers.push("Access-Control-Max-Age: 86400\r\n");
    }
    resp.body = string_new();
    cors_apply_headers(&resp, allowedOrigin);
    return resp;
}

} // namespace std
