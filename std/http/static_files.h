// SafeC Standard Library — static file serving for std/http servers.
//
// http_serve_file answers a request with one file; http_serve_dir maps a
// URL path under a directory root (rejecting anything that would escape
// it). Both handle what browsers and CDNs expect of a static server:
//   - Content-Type from the extension (http_mime_type),
//   - ETag (size + mtime) and Last-Modified validators, answering
//     If-None-Match / If-Modified-Since with 304,
//   - single byte ranges (Range: bytes=a-b / a- / -n, with If-Range) as
//     206 with Content-Range, 416 when unsatisfiable,
//   - HEAD (the server drops the body), Cache-Control: max-age,
//   - index.html for directories, an optional SPA fallback.
// Only GET and HEAD are served; anything else is 405.
//
// Files are read whole into the response body (a range reads only its
// part); this is a development/small-site server, not a sendfile() one.
#pragma once
#include <std/http/http.h>
#include <std/collections/string.h>

namespace std {

struct StaticOptions {
    int       etag;            // send ETag and honor If-None-Match (default 1)
    int       last_modified;   // send Last-Modified, honor If-Modified-Since (default 1)
    int       ranges;          // honor Range (default 1)
    long long max_age;         // Cache-Control: max-age=N; -1 = no header (default -1)
    int       index_html;      // a directory serves its index.html (default 1)
    int       spa_fallback;    // a missing path serves the root's index.html (default 0)
    int       dotfiles;        // serve names starting with '.' (default 0: 404)

    static struct StaticOptions defaults();
};

// The media type for a file name's extension (case-insensitive):
// text/html; charset=utf-8, text/css, text/javascript, application/json,
// application/wasm, image/png, image/svg+xml, font/woff2, ... and
// application/octet-stream for anything unknown.
const &char http_mime_type(const &char path);
// 1 for text-like types worth compressing (text/*, json, javascript, xml,
// svg, wasm).
int http_mime_is_compressible(const &char mime);

// Serves the file at 'file_path' for 'req' (see the file comment). 404 if
// it doesn't exist or is a directory. 'opts' may be null (defaults()).
struct HttpResponse http_serve_file(const &HttpRequest req, const &char file_path,
                                    const ?&StaticOptions opts);

// Serves 'url_path' (percent-encoded, as in the request; a query string is
// ignored) from under the directory 'root'. 403 for a path that tries to
// leave the root ("..", a backslash, a NUL byte); a directory without a
// trailing '/' is redirected (301) to one so relative links resolve.
struct HttpResponse http_serve_dir(const &HttpRequest req, const &char root, const &char url_path,
                                   const ?&StaticOptions opts);

// Joins a URL path under 'root' after percent-decoding it and removing
// empty and "." segments; 0 (and 'out' untouched) if it contains a ".."
// segment, a backslash, a NUL byte or (unless allow_dotfiles) a segment
// starting with '.'.
int http_safe_join(const &char root, const &char url_path, int allow_dotfiles, &String out);

// Size (bytes), modification time (Unix seconds) and kind of a file:
// 1 for a regular file, 2 for a directory, 0 if it doesn't exist.
int http_file_info(const &char path, &long long size, &long long mtime);

} // namespace std
