// SafeC Standard Library — HTTP cookies (RFC 6265).
//
// Reading: a request's "Cookie: a=1; b=2" header parsed into name/value
// pairs (CookieList, or HttpRequest::cookie() in std/http/http.h for one
// value). Writing: SetCookie builds a "Set-Cookie" header value with its
// attributes; http_set_cookie adds one to a response. SetCookie::parse
// reads one back (what a client stores).
//
// Values are taken and written verbatim (RFC 6265 allows no escaping):
// a value with spaces, ';', ',' or '"' should be encoded by the caller —
// url_encode/form_encode (std/encoding/url.h) are the usual choice.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/http/http.h>

namespace std {

#define COOKIE_SAMESITE_UNSET  0
#define COOKIE_SAMESITE_LAX    1
#define COOKIE_SAMESITE_STRICT 2
#define COOKIE_SAMESITE_NONE   3

struct CookiePair {
    struct String name;
    struct String value;
};

// The cookies of one Cookie header, in order.
struct CookieList {
    struct Vec<struct CookiePair> items;

    // Parses "a=1; b=2" (surrounding double quotes on a value are
    // removed; a part without '=' is skipped).
    static struct CookieList parse(const &char cookie_header);
    // Every Cookie header of a request (HTTP/2 proxies may split them).
    static struct CookieList from_request(const &HttpRequest req);

    unsigned long length() const;
    // The first cookie with this name (case-sensitive), or null; aliases
    // this list's storage.
    const ?&char  get(const &char name) const;
    const ?&char  name_at(unsigned long i) const;
    const ?&char  value_at(unsigned long i) const;
    void          drop();
    void          free() consume;   // = drop()
};

struct SetCookie {
    struct String name;
    struct String value;
    struct String path;        // empty: not sent
    struct String domain;      // empty: not sent (host-only cookie)
    long long     max_age;     // seconds; -1 = not sent; 0 deletes the cookie
    long long     expires;     // Unix seconds; 0 = not sent (session cookie)
    int           secure;
    int           http_only;
    int           same_site;   // COOKIE_SAMESITE_*
    int           partitioned; // CHIPS "Partitioned" attribute

    // name=value with no attributes (path "/" is a common next step).
    static struct SetCookie new(const &char name, const &char value);
    // A cookie that tells the browser to delete 'name' (empty value,
    // Max-Age=0, Expires in 1970) for the given path.
    static struct SetCookie removal(const &char name, const &char path);
    // Parses one Set-Cookie header value. *ok = 0 if there is no
    // "name=value" first part. Unknown attributes are ignored, as
    // browsers do.
    static struct SetCookie parse(const &char header_value, ?&int ok);

    // "name=value; Path=/; Max-Age=3600; Expires=...; Secure; HttpOnly;
    // SameSite=Lax" — the header value.
    struct String to_string() const;
    void          drop();
    void          free() consume;   // = drop()
};

// Appends "Set-Cookie: <c.to_string()>" to the response's headers (one per
// call; several cookies mean several calls).
void http_set_cookie(&HttpResponse resp, const &SetCookie c);

} // namespace std
