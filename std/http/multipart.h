// SafeC Standard Library — multipart/form-data (RFC 7578): parsing an
// HTML form upload on the server, building one for a client request.
//
//   int ok = 0;
//   struct MultipartForm form = std::MultipartForm::from_request(req, &ok);
//   match (form.get("avatar")) {
//       case some(p): { /* p.filename, p.content_type, p.data (binary-safe String) */ }
//       case null: { }
//   }
//   form.free();
#pragma once
#include <std/http/http.h>
#include <std/http/headers.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

struct MultipartPart {
    struct String      name;          // Content-Disposition name=
    struct String      filename;      // filename= (empty for a plain field)
    int                has_filename;  // 1 if filename= was present (even empty)
    struct String      content_type;  // the part's Content-Type ("text/plain" if absent)
    struct HttpHeaders headers;       // all of the part's headers
    struct String      data;          // the body, binary-safe (use .length())
};

struct MultipartForm {
    struct Vec<struct MultipartPart> parts;

    // Parses a body given its Content-Type header value (which carries the
    // boundary). *ok = 0 on a missing boundary, a missing closing
    // delimiter or a part without headers.
    static struct MultipartForm parse(const &char content_type, const &unsigned char body,
                                      unsigned long len, ?&int ok);
    // parse() on a request's Content-Type and body.
    static struct MultipartForm from_request(const &HttpRequest req, ?&int ok);

    unsigned long length() const;
    // The first part named 'name', or null; aliases this form's storage.
    ?&MultipartPart get(const &char name);
    ?&MultipartPart part_at(unsigned long i);
    // A plain field's text (its data), or null.
    const ?&char    value(const &char name) const;
    void            free();
};

// Builds a multipart/form-data body.
struct MultipartBuilder {
    struct String boundary;
    struct String body;

    // A random boundary.
    static struct MultipartBuilder new();
    void add_field(const &char name, const &char value);
    void add_file(const &char name, const &char filename, const &char content_type,
                  const &unsigned char data, unsigned long len);
    // "multipart/form-data; boundary=..." for the Content-Type header.
    struct String content_type() const;
    // Appends the closing delimiter; 'body' is then complete.
    void finish();
    void free();
};

} // namespace std
