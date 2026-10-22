// SafeC Standard Library — HttpHeaders: a structured, ordered,
// case-insensitive HTTP header map.
//
// HttpRequest/HttpResponse keep their headers as one raw "Key: value"
// block (std/http/http.h) — cheap to build, and what the wire carries.
// HttpHeaders is the structured view for code that wants to iterate,
// edit or look headers up repeatedly: parse() a raw block into one,
// to_raw() it back. Names keep the case they were given; every lookup
// ignores ASCII case (RFC 9110 §5.1). Order is preserved, and a name may
// appear more than once (Set-Cookie, Vary, ...).
//
// For a single lookup or edit on a raw block, std/http/http.h's
// http_header_get / http_headers_set / http_headers_add /
// http_headers_remove avoid building one of these at all.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

struct HttpHeaderField {
    struct String name;
    struct String value;
};

struct HttpHeaders {
    struct Vec<struct HttpHeaderField> fields;

    static struct HttpHeaders new();
    // Parses a raw "Key: value" block (lines end in "\r\n" or "\n"; a
    // line without ':' is skipped; leading/trailing blanks are trimmed;
    // an obsolete folded continuation line — leading space/tab — is joined
    // onto the previous value with one space).
    static struct HttpHeaders parse(const &char raw);

    unsigned long length() const;
    // The first value for 'name' (any case), or null. The reference
    // aliases this object's storage: valid until the next mutation/free().
    const ?&char  get(const &char name) const;
    const &char   get_or(const &char name, const &char fallback) const;
    // How many fields have this name, and the n-th one's value.
    unsigned long count(const &char name) const;
    const ?&char  get_nth(const &char name, unsigned long n) const;
    // Every value for 'name' joined with ", " (the RFC 9110 §5.3
    // combination; empty String if absent). Not meaningful for Set-Cookie.
    struct String get_joined(const &char name) const;
    int           has(const &char name) const;
    // 1 if 'name' holds a comma-separated list containing 'token' (case-
    // insensitive), e.g. contains_token("Connection", "upgrade").
    int           contains_token(const &char name, const &char token) const;
    // i-th field in order (null past the end).
    const ?&char  name_at(unsigned long i) const;
    const ?&char  value_at(unsigned long i) const;

    // Adds a field at the end, keeping any existing ones with this name.
    void          append(const &char name, const &char value);
    // Replaces the first field with this name (keeping its position) and
    // drops the others; appends one if there was none.
    void          set(const &char name, const &char value);
    // Removes every field with this name; returns how many there were.
    unsigned long remove(const &char name);
    void          clear();

    // "Name: value\r\n" per field (CRLF after every line, including the
    // last), appended to 'out' — the shape a request/response header
    // block is sent in.
    void          write_to(&String out) const;
    // The raw block form HttpRequest/HttpResponse store: lines CRLF-
    // separated, no trailing CRLF.
    struct String to_raw() const;
    struct HttpHeaders clone() const;
    void          free();
};

// A parameter of a structured header value — the "boundary" of
// 'multipart/form-data; boundary="xyz"', the "filename" of a
// Content-Disposition. Quoted values are unquoted (backslash escapes
// resolved); names compare case-insensitively. Empty String if absent.
struct String http_header_param(const &char value, const &char name);
// The part of such a value before the first ';', trimmed and lowercased
// ("multipart/form-data").
struct String http_header_main_value(const &char value);

// Case-insensitive ASCII comparison of two NUL-terminated strings (1 if
// equal) — the comparison header names use.
int http_name_eq(const &char a, const &char b);

} // namespace std
