// SafeC Standard Library — URLs: percent-encoding (RFC 3986), the
// application/x-www-form-urlencoded format (query strings and HTML form
// bodies, as the WHATWG URL standard's URLSearchParams defines it), and a
// structured URL type with parsing, serialization and relative-reference
// resolution (RFC 3986 §5).
//
// Pure string code, no I/O: usable on hosted, bare-metal and wasm32
// targets alike (std/http uses it on the server, std/wasm-side code can use
// it for client-side routing).
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// ── Percent-encoding ───────────────────────────────────────────────────────

// Percent-encodes every byte except the unreserved set (A-Z a-z 0-9 - _ . ~)
// — this matches the "component" flavor used for query-string keys/values
// and path segments (JavaScript's encodeURIComponent, not encodeURI).
// Space becomes "%20" (not "+" — that's an application/x-www-form-urlencoded
// convention some callers layer on top, not part of RFC 3986 itself; see
// form_encode below).
struct String url_encode(const &char s);

// Reverses url_encode: "%XX" becomes the raw byte 0xXX, "+" is left as a
// literal '+' (not decoded to space — see url_encode's note; use
// form_decode for form data). On a malformed "%" escape (not followed by
// two hex digits), '*ok' (if non-NULL) is set to 0 and decoding stops at
// that point; otherwise '*ok' is set to 1.
struct String url_decode(const &char s, ?&int ok);

// Like url_encode, but keeps the characters a URL *path* may hold as-is:
// '/', and RFC 3986's sub-delims and ':' '@' (so "/a b/c:d" becomes
// "/a%20b/c:d"). '%' itself is always escaped, so this is not idempotent —
// encode raw paths, not already-encoded ones.
struct String url_encode_path(const &char s);

// application/x-www-form-urlencoded: url_encode, except that space becomes
// '+' and '*' stays as-is (the WHATWG urlencoded serializer's byte set).
struct String form_encode(const &char s);

// Reverses form_encode: '+' becomes a space, "%XX" a byte. Unlike
// url_decode, a malformed '%' escape is kept literally (browsers do the
// same) and never fails.
struct String form_decode(const &char s);

// ── Query strings / form bodies ────────────────────────────────────────────

struct UrlQueryPair {
    struct String key;     // decoded
    struct String value;   // decoded
};

// An ordered multi-map of decoded key/value pairs — the parsed form of
// "a=1&b=two%20words&a=3" (a query string without its '?', or an
// application/x-www-form-urlencoded request body). Keys may repeat; get()
// returns the first. Lookups are case-sensitive, like URLSearchParams.
struct QueryParams {
    struct Vec<struct UrlQueryPair> pairs;

    static struct QueryParams new();
    // Parses 'qs' (a leading '?' is skipped). Empty segments ("a=1&&b=2")
    // are dropped; a segment without '=' is a key with an empty value.
    static struct QueryParams parse(const &char qs);
    // parse() over 'n' raw bytes (e.g. a request body, which is not
    // NUL-terminated in general).
    static struct QueryParams parse_n(const &char qs, unsigned long n);

    unsigned long length() const;
    // The first value for 'key', or null if absent. The reference aliases
    // this object's storage: valid until the next mutation or free().
    const ?&char  get(const &char key) const;
    // get() with a fallback for an absent key.
    const &char   get_or(const &char key, const &char fallback) const;
    // How many pairs have this key, and the n-th of them (0-based; null
    // past the end).
    unsigned long count(const &char key) const;
    const ?&char  get_nth(const &char key, unsigned long n) const;
    int           has(const &char key) const;
    // i-th pair in order (null past the end).
    const ?&char  key_at(unsigned long i) const;
    const ?&char  value_at(unsigned long i) const;

    // Adds a pair at the end (keeps existing ones with the same key).
    void          append(const &char key, const &char value);
    // Replaces the first pair with this key and removes the others, or
    // appends one if there was none.
    void          set(const &char key, const &char value);
    // Removes every pair with this key; returns how many there were.
    unsigned long remove(const &char key);
    // Stable sort by key (byte order), as URLSearchParams.sort().
    void          sort();

    // Serializes back to "k=v&k2=v2" with form_encode on both halves
    // (no leading '?').
    struct String to_string() const;
    struct QueryParams clone() const;
    void          drop();
    void          free() consume;   // = drop()
};

// ── URL ────────────────────────────────────────────────────────────────────

// A URL split into its RFC 3986 components:
//   scheme://username:password@host:port/path?query#fragment
// Components are stored as they appear in the text (still percent-encoded,
// so they can be serialized back unchanged), except that the scheme and
// a registered-name host are lowercased and an IPv6 literal's brackets are
// dropped from 'host' ("[::1]" is stored as "::1"). 'port' is -1 when the
// URL has none. has_authority / has_query / has_fragment tell "no query"
// apart from an empty one ("/p?" vs "/p").
struct Url {
    struct String scheme;     // "https" (no ':'), empty for a relative reference
    struct String username;
    struct String password;
    struct String host;
    int           port;       // -1: none given
    struct String path;       // "/a/b" (may be empty, e.g. "mailto:x" has path "x")
    struct String query;      // without the '?'
    struct String fragment;   // without the '#'
    int           has_authority;   // "//" was present
    int           has_query;
    int           has_fragment;

    // An empty relative reference (every component empty, port -1).
    static struct Url new();
    // Parses an absolute URL or a relative reference. Fails (*ok = 0, and
    // an empty Url) on a bad scheme, a non-numeric or > 65535 port, an
    // unterminated '[' IPv6 literal, or a control character/space.
    static struct Url parse(const &char s, ?&int ok);

    // Back to text (the RFC 3986 §5.3 recomposition).
    struct String to_string() const;
    // Resolves the reference 'ref' against this (absolute) base URL —
    // RFC 3986 §5.2, including dot-segment removal: base
    // "http://a/b/c/d;p?q" + "../g" = "http://a/b/g". Fails if 'ref'
    // doesn't parse or this base has no scheme.
    struct Url    resolve(const &char ref, ?&int ok) const;

    // 'port', or the scheme's default (http/ws 80, https/wss 443, ftp 21)
    // when none was given; -1 for an unknown scheme without a port.
    int           port_or_default() const;
    // "host" or "host:port" (only an explicit port), IPv6 re-bracketed —
    // what an HTTP Host header carries.
    struct String host_port() const;
    // "scheme://host[:port]" — the web origin (for CORS, cookies, ...).
    struct String origin() const;
    // path (or "/" when empty) plus "?query" — an HTTP request-target.
    struct String request_target() const;
    // The decoded query parameters.
    struct QueryParams query_params() const;
    // Replaces the query with params.to_string() (has_query = 1, or 0 for
    // an empty params).
    void          set_query_params(const &QueryParams params);
    // The path with each %XX decoded (for file lookups, routing, ...).
    struct String path_decoded() const;
    int           is_absolute() const;     // has a scheme

    struct Url    clone() const;
    void          drop();
    void          free() consume;   // = drop()
};

// RFC 3986 §5.2.4: "/a/b/../c/./d" -> "/a/c/d".
struct String url_remove_dot_segments(const &char path);

} // namespace std
