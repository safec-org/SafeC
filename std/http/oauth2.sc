// SafeC Standard Library — OAuth2 client implementation (see oauth2.h).
#pragma once
#include <std/http/oauth2.h>
#include <std/http/http.h>
#include <std/http/http.sc>
#include <std/serial/value.h>
#include <std/serial/value.sc>
#include <std/serial/json.h>
#include <std/serial/json.sc>
#include <std/encoding/url.h>
#include <std/encoding/url.sc>
#include <std/collections/string.h>
#include <std/collections/string.sc>
#include <std/mem.sc>

namespace std {

static void __form_add(struct String* body, const char* key, const char* val) {
    unsafe {
        if (body->length() > 0UL) { body->push("&"); }
        body->push(key);
        body->push("=");
        struct String enc = url_encode(val);
        body->push_str(&enc);
        enc.free();
    }
}

static struct OAuth2Token __token_from_json(const struct Value* root) {
    struct OAuth2Token t;
    t.accessToken = string_new();
    t.refreshToken = string_new();
    t.tokenType = string_new();
    t.expiresIn = 0LL;
    unsafe {
        struct Value* at = root->object_get("access_token");
        if (at != (struct Value*)0) { t.accessToken = string_from(at->as_string()); }
        struct Value* rt = root->object_get("refresh_token");
        if (rt != (struct Value*)0) { t.refreshToken = string_from(rt->as_string()); }
        struct Value* tt = root->object_get("token_type");
        if (tt != (struct Value*)0) { t.tokenType = string_from(tt->as_string()); }
        struct Value* ei = root->object_get("expires_in");
        if (ei != (struct Value*)0) { t.expiresIn = ei->as_int(); }
    }
    return t;
}

static struct OAuth2Token __empty_token() {
    struct OAuth2Token t;
    t.accessToken = string_new();
    t.refreshToken = string_new();
    t.tokenType = string_new();
    t.expiresIn = 0LL;
    return t;
}

static struct OAuth2Token __token_request(const char* tokenHost, unsigned short tokenPort,
                                          const char* tokenPath, struct String* body, int* ok) {
    int httpOk = 0;
    struct HttpResponse resp;
    unsafe {
        resp = http_request(tokenHost, tokenPort, "POST", tokenPath,
                             "Content-Type: application/x-www-form-urlencoded\r\n",
                             (const unsigned char*)body->data, body->len, &httpOk);
    }
    if (!httpOk || resp.status < 200 || resp.status >= 300) {
        unsafe { *ok = 0; resp.free(); }
        return __empty_token();
    }
    int parseOk;
    struct Value root;
    unsafe { root = json_parse(resp.body.as_ptr(), &parseOk); resp.free(); }
    if (!parseOk) {
        unsafe { *ok = 0; value_free(&root); }
        return __empty_token();
    }
    struct OAuth2Token t = __token_from_json(&root);
    unsafe { value_free(&root); }
    int hasAccess;
    unsafe { hasAccess = t.accessToken.length() > 0UL; }
    unsafe { *ok = hasAccess; }
    return t;
}

struct OAuth2Token oauth2_exchange_code(const char* tokenHost, unsigned short tokenPort,
                                        const char* tokenPath, const char* clientId,
                                        const char* clientSecret, const char* code,
                                        const char* redirectUri, int* ok) {
    struct String body = string_new();
    __form_add(&body, "grant_type", "authorization_code");
    __form_add(&body, "code", code);
    __form_add(&body, "redirect_uri", redirectUri);
    __form_add(&body, "client_id", clientId);
    __form_add(&body, "client_secret", clientSecret);
    struct OAuth2Token t = __token_request(tokenHost, tokenPort, tokenPath, &body, ok);
    unsafe { body.free(); }
    return t;
}

struct OAuth2Token oauth2_refresh(const char* tokenHost, unsigned short tokenPort,
                                  const char* tokenPath, const char* clientId,
                                  const char* clientSecret, const char* refreshToken, int* ok) {
    struct String body = string_new();
    __form_add(&body, "grant_type", "refresh_token");
    __form_add(&body, "refresh_token", refreshToken);
    __form_add(&body, "client_id", clientId);
    __form_add(&body, "client_secret", clientSecret);
    struct OAuth2Token t = __token_request(tokenHost, tokenPort, tokenPath, &body, ok);
    unsafe { body.free(); }
    return t;
}

void oauth2_token_free(&OAuth2Token t) {
    unsafe {
        t.accessToken.free();
        t.refreshToken.free();
        t.tokenType.free();
    }
}

} // namespace std
