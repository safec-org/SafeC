// SafeC Standard Library — static file serving (see static_files.h).
#pragma once
#include <std/http/static_files.h>
#include <std/http/http.h>
#include <std/http/date.h>
#include <std/encoding/url.h>
#include <std/collections/string.h>
#include <std/io_file.h>
#include <std/mem.h>
#include <std/str.h>

namespace std {

// ── File metadata ─────────────────────────────────────────────────────────
// struct stat differs by OS and architecture, and glibc before 2.33 has no
// exported 'stat' at all (only __xstat), so Linux goes through the raw
// newfstatat system call. The offsets below are each ABI's documented
// struct layout (only mode, size and mtime are read).
#if defined(_WIN32)
extern int _stat64(const char* path, void* buf);
#define HTTP_ST_MODE_U16_ 6
#define HTTP_ST_SIZE_     24
#define HTTP_ST_MTIME_    40
#elif defined(__APPLE__)
extern int stat(const char* path, void* buf);
#define HTTP_ST_MODE_U16_ 4
#define HTTP_ST_SIZE_     96
#define HTTP_ST_MTIME_    48
#elif defined(__linux__)
extern long syscall(long number, ...);
#if defined(__aarch64__) || defined(__riscv)
#define HTTP_SYS_NEWFSTATAT_ 79
#define HTTP_ST_MODE_U32_    16
#else
#define HTTP_SYS_NEWFSTATAT_ 262
#define HTTP_ST_MODE_U32_    24
#endif
#define HTTP_ST_SIZE_     48
#define HTTP_ST_MTIME_    88
#else
// FreeBSD 12+ (ino64 layout).
extern int stat(const char* path, void* buf);
#define HTTP_ST_MODE_U16_ 24
#define HTTP_ST_SIZE_     112
#define HTTP_ST_MTIME_    64
#endif

int http_file_info(const &char path, &long long size, &long long mtime) {
    unsigned char st[256];
    int i = 0;
    while (i < 256) { st[i] = (unsigned char)0; i = i + 1; }
    int rc;
    unsigned int mode = 0U;
    unsafe {
#if defined(_WIN32)
        rc = _stat64(path, (void*)&st[0]);
#elif defined(__APPLE__)
        rc = stat(path, (void*)&st[0]);
#elif defined(__linux__)
        rc = (int)syscall((long)HTTP_SYS_NEWFSTATAT_, -100L, path, (void*)&st[0], 0L);
#else
        rc = stat(path, (void*)&st[0]);
#endif
        if (rc != 0) { return 0; }
#ifdef HTTP_ST_MODE_U32_
        mode = *(unsigned int*)&st[HTTP_ST_MODE_U32_];
#else
        mode = (unsigned int)*(unsigned short*)&st[HTTP_ST_MODE_U16_];
#endif
        *size = *(long long*)&st[HTTP_ST_SIZE_];
        *mtime = *(long long*)&st[HTTP_ST_MTIME_];
    }
    unsigned int fmt = mode & 0xF000U;
    if (fmt == 0x4000U) { return 2; }
    if (fmt == 0x8000U) { return 1; }
    return 3;   // something else (device, fifo): not served
}

// ── MIME types ────────────────────────────────────────────────────────────

static int sf_ext_is_(const char* ext, unsigned long n, const char* want) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            char c = ext[i];
            if (c >= 'A' && c <= 'Z') { c = (char)(c + 32); }
            if (want[i] == (char)0 || want[i] != c) { return 0; }
            i = i + 1UL;
        }
        return want[i] == (char)0;
    }
}

const &char http_mime_type(const &char path) {
    unsigned long n = str_len(path);
    unsigned long dot = n;
    unsafe {
        unsigned long i = n;
        while (i > 0UL) {
            i = i - 1UL;
            if (path[i] == '.') { dot = i; break; }
            if (path[i] == '/' || path[i] == '\\') { break; }
        }
    }
    if (dot == n) { return "application/octet-stream"; }
    const char* e;
    unsafe { e = &path[dot + 1UL]; }
    unsigned long el = n - dot - 1UL;
    if (sf_ext_is_(e, el, "html") || sf_ext_is_(e, el, "htm")) { return "text/html; charset=utf-8"; }
    if (sf_ext_is_(e, el, "css")) { return "text/css; charset=utf-8"; }
    if (sf_ext_is_(e, el, "js") || sf_ext_is_(e, el, "mjs")) { return "text/javascript; charset=utf-8"; }
    if (sf_ext_is_(e, el, "json") || sf_ext_is_(e, el, "map")) { return "application/json"; }
    if (sf_ext_is_(e, el, "wasm")) { return "application/wasm"; }
    if (sf_ext_is_(e, el, "txt") || sf_ext_is_(e, el, "text")) { return "text/plain; charset=utf-8"; }
    if (sf_ext_is_(e, el, "md") || sf_ext_is_(e, el, "markdown")) { return "text/markdown; charset=utf-8"; }
    if (sf_ext_is_(e, el, "csv")) { return "text/csv; charset=utf-8"; }
    if (sf_ext_is_(e, el, "xml")) { return "application/xml"; }
    if (sf_ext_is_(e, el, "svg")) { return "image/svg+xml"; }
    if (sf_ext_is_(e, el, "png")) { return "image/png"; }
    if (sf_ext_is_(e, el, "jpg") || sf_ext_is_(e, el, "jpeg")) { return "image/jpeg"; }
    if (sf_ext_is_(e, el, "gif")) { return "image/gif"; }
    if (sf_ext_is_(e, el, "webp")) { return "image/webp"; }
    if (sf_ext_is_(e, el, "avif")) { return "image/avif"; }
    if (sf_ext_is_(e, el, "ico")) { return "image/x-icon"; }
    if (sf_ext_is_(e, el, "bmp")) { return "image/bmp"; }
    if (sf_ext_is_(e, el, "woff")) { return "font/woff"; }
    if (sf_ext_is_(e, el, "woff2")) { return "font/woff2"; }
    if (sf_ext_is_(e, el, "ttf")) { return "font/ttf"; }
    if (sf_ext_is_(e, el, "otf")) { return "font/otf"; }
    if (sf_ext_is_(e, el, "pdf")) { return "application/pdf"; }
    if (sf_ext_is_(e, el, "zip")) { return "application/zip"; }
    if (sf_ext_is_(e, el, "gz")) { return "application/gzip"; }
    if (sf_ext_is_(e, el, "tar")) { return "application/x-tar"; }
    if (sf_ext_is_(e, el, "mp3")) { return "audio/mpeg"; }
    if (sf_ext_is_(e, el, "wav")) { return "audio/wav"; }
    if (sf_ext_is_(e, el, "ogg")) { return "audio/ogg"; }
    if (sf_ext_is_(e, el, "mp4")) { return "video/mp4"; }
    if (sf_ext_is_(e, el, "webm")) { return "video/webm"; }
    if (sf_ext_is_(e, el, "webmanifest")) { return "application/manifest+json"; }
    if (sf_ext_is_(e, el, "sc") || sf_ext_is_(e, el, "h") || sf_ext_is_(e, el, "c")) { return "text/plain; charset=utf-8"; }
    return "application/octet-stream";
}

static int sf_starts_(const &char s, const &char prefix) {
    unsafe {
        unsigned long i = 0UL;
        while (prefix[i] != (char)0) {
            if (s[i] != prefix[i]) { return 0; }
            i = i + 1UL;
        }
    }
    return 1;
}

int http_mime_is_compressible(const &char mime) {
    unsafe {
        if (sf_starts_(mime, "text/")) { return 1; }
        if (sf_starts_(mime, "application/json") || sf_starts_(mime, "application/javascript") ||
            sf_starts_(mime, "application/xml") || sf_starts_(mime, "application/wasm") ||
            sf_starts_(mime, "application/manifest+json") || sf_starts_(mime, "image/svg+xml")) {
            return 1;
        }
    }
    return 0;
}

// ── Options, paths ────────────────────────────────────────────────────────

struct StaticOptions StaticOptions::defaults() {
    struct StaticOptions o;
    o.etag = 1;
    o.last_modified = 1;
    o.ranges = 1;
    o.max_age = -1LL;
    o.index_html = 1;
    o.spa_fallback = 0;
    o.dotfiles = 0;
    return o;
}

int http_safe_join(const &char root, const &char url_path, int allow_dotfiles, &String out) {
    struct String path = String::from(url_path);
    long long q = path.find_char('?');
    if (q >= 0LL) { path.truncate((unsigned long)q); }
    long long h = path.find_char('#');
    if (h >= 0LL) { path.truncate((unsigned long)h); }
    int ok = 0;
    struct String dec = url_decode(path.as_ptr(), &ok);
    path.free();
    if (!ok || dec.find_char('\\') >= 0LL) { dec.free(); return 0; }
    // A NUL inside the decoded path ("%00") would truncate it.
    if (str_len(dec.as_ptr()) != dec.length()) { dec.free(); return 0; }
    struct String joined = String::from(root);
    while (joined.length() > 1UL && joined.ends_with("/")) { joined.truncate(joined.length() - 1UL); }
    unsigned long i = 0UL;
    int good = 1;
    unsafe {
        const char* d = dec.as_ptr();
        unsigned long n = dec.length();
        while (i < n) {
            while (i < n && d[i] == '/') { i = i + 1UL; }
            unsigned long s = i;
            while (i < n && d[i] != '/') { i = i + 1UL; }
            unsigned long len = i - s;
            if (len == 0UL) { continue; }
            if (len == 1UL && d[s] == '.') { continue; }
            if (len == 2UL && d[s] == '.' && d[s + 1UL] == '.') { good = 0; break; }
            if (d[s] == '.' && !allow_dotfiles) { good = 0; break; }
            if (!joined.ends_with("/")) { joined.push_char('/'); }
            joined.push_n(&d[s], len);
        }
    }
    dec.free();
    if (!good) { joined.free(); return 0; }
    out.clear();
    out.push_str(&joined);
    joined.free();
    return 1;
}

// ── Serving ───────────────────────────────────────────────────────────────

static struct String sf_etag_(long long size, long long mtime) {
    struct String e = String::from("\"");
    char hex[32];
    unsigned long long v = (unsigned long long)size;
    int n = 0;
    if (v == 0ULL) { hex[0] = '0'; n = 1; }
    while (v > 0ULL && n < 32) {
        unsigned int d = (unsigned int)(v & 15ULL);
        unsafe { hex[n] = (char)(d < 10U ? (int)'0' + (int)d : (int)'a' + (int)d - 10); }
        v = v >> 4;
        n = n + 1;
    }
    while (n > 0) { n = n - 1; unsafe { e.push_char(hex[n]); } }
    e.push_char('-');
    v = (unsigned long long)mtime;
    n = 0;
    if (v == 0ULL) { hex[0] = '0'; n = 1; }
    while (v > 0ULL && n < 32) {
        unsigned int d = (unsigned int)(v & 15ULL);
        unsafe { hex[n] = (char)(d < 10U ? (int)'0' + (int)d : (int)'a' + (int)d - 10); }
        v = v >> 4;
        n = n + 1;
    }
    while (n > 0) { n = n - 1; unsafe { e.push_char(hex[n]); } }
    e.push_char('"');
    return e;
}

// Does the If-None-Match list contain 'etag' (weak comparison) or "*"?
static int sf_etag_listed_(const &String list, const &String etag) {
    if (list.trim().eq_cstr("*")) { return 1; }
    struct String items[64];
    unsigned long n = list.split(",", items[0..64]);
    int found = 0;
    unsigned long i = 0UL;
    while (i < n) {
        struct String t = items[i].trim();
        struct String s = t.strip_prefix("W/");
        if (s.eq(etag)) { found = 1; }
        s.free();
        t.free();
        items[i].free();
        i = i + 1UL;
    }
    return found;
}

// Parses a single "bytes=" range against 'size'. 1 with [*start, *end]
// (inclusive) set, 0 to ignore the header (serve 200), -1 unsatisfiable.
static int sf_parse_range_(const &String hdr, long long size, &long long start, &long long end) {
    if (!hdr.starts_with("bytes=")) { return 0; }
    struct String spec = hdr.substr(6UL, hdr.length());
    if (spec.find_char(',') >= 0LL) { spec.free(); return 0; }   // multiple ranges: send it all
    struct String t = spec.trim();
    spec.free();
    long long dash = t.find_char('-');
    if (dash < 0LL) { t.free(); return 0; }
    struct String a = t.substr(0UL, (unsigned long)dash);
    struct String b = t.substr((unsigned long)dash + 1UL, t.length());
    t.free();
    int aok = 0;
    int bok = 0;
    long long av = a.is_empty() ? -1LL : a.parse_int(&aok);
    long long bv = b.is_empty() ? -1LL : b.parse_int(&bok);
    int aempty = a.is_empty();
    int bempty = b.is_empty();
    a.free();
    b.free();
    if ((!aempty && !aok) || (!bempty && !bok) || (aempty && bempty)) { return 0; }
    if (aempty) {
        if (bv <= 0LL) { return -1; }
        if (bv > size) { bv = size; }
        *start = size - bv;
        *end = size - 1LL;
        return size > 0LL ? 1 : -1;
    }
    if (av >= size) { return -1; }
    *start = av;
    *end = (bempty || bv >= size) ? size - 1LL : bv;
    if (*end < *start) { return 0; }
    return 1;
}

static struct StaticOptions sf_opts_(const ?&StaticOptions opts) {
    match (opts) {
        case some(o): { return *o; }
        case null: { }
    }
    return StaticOptions::defaults();
}

struct HttpResponse http_serve_file(const &HttpRequest req, const &char file_path,
                                    const ?&StaticOptions optsIn) {
    struct StaticOptions o = sf_opts_(optsIn);
    if (!req.method.eq_cstr("GET") && !req.method.eq_cstr("HEAD")) {
        struct HttpResponse r = HttpResponse::error(405);
        r.set_header("Allow", "GET, HEAD");
        return r;
    }
    long long size = 0LL;
    long long mtime = 0LL;
    if (http_file_info(file_path, &size, &mtime) != 1) { return HttpResponse::error(404); }
    struct String etag = sf_etag_(size, mtime);
    struct String lastMod = http_date_format(mtime);
    struct HttpResponse r = HttpResponse::new(200);
    r.add_header("Content-Type", http_mime_type(file_path));
    if (o.etag) { r.add_header("ETag", etag.as_ptr()); }
    if (o.last_modified) { r.add_header("Last-Modified", lastMod.as_ptr()); }
    if (o.ranges) { r.add_header("Accept-Ranges", "bytes"); }
    if (o.max_age >= 0LL) {
        struct String cc = String::from("max-age=");
        cc.push_int(o.max_age);
        r.add_header("Cache-Control", cc.as_ptr());
        cc.free();
    }
    // Conditional requests: If-None-Match wins over If-Modified-Since.
    int notModified = 0;
    struct String inm = req.header("If-None-Match");
    if (o.etag && !inm.is_empty()) {
        notModified = sf_etag_listed_(&inm, &etag);
    } else if (o.last_modified) {
        struct String ims = req.header("If-Modified-Since");
        if (!ims.is_empty()) {
            int dok = 0;
            long long t = http_date_parse(ims.as_ptr(), &dok);
            if (dok && mtime <= t) { notModified = 1; }
        }
        ims.free();
    }
    inm.free();
    if (notModified) {
        r.status = 304;
        r.remove_header("Content-Type");
        etag.free();
        lastMod.free();
        return r;
    }
    long long start = 0LL;
    long long end = size - 1LL;
    int partial = 0;
    struct String range = req.header("Range");
    if (o.ranges && !range.is_empty()) {
        int useRange = 1;
        struct String ifRange = req.header("If-Range");
        if (!ifRange.is_empty()) {
            if (ifRange.starts_with("\"") || ifRange.starts_with("W/")) {
                useRange = ifRange.eq(&etag);
            } else {
                int dok = 0;
                long long t = http_date_parse(ifRange.as_ptr(), &dok);
                useRange = dok && t >= mtime;
            }
        }
        ifRange.free();
        if (useRange) {
            int pr = sf_parse_range_(&range, size, &start, &end);
            if (pr == -1) {
                range.free();
                etag.free();
                lastMod.free();
                r.free();
                struct HttpResponse u = HttpResponse::error(416);
                struct String cr = String::from("bytes */");
                cr.push_int(size);
                u.set_header("Content-Range", cr.as_ptr());
                cr.free();
                return u;
            }
            if (pr == 1) { partial = 1; }
        }
    }
    range.free();
    etag.free();
    lastMod.free();
    if (partial) {
        r.status = 206;
        struct String cr = String::from("bytes ");
        cr.push_int(start);
        cr.push_char('-');
        cr.push_int(end);
        cr.push_char('/');
        cr.push_int(size);
        r.add_header("Content-Range", cr.as_ptr());
        cr.free();
    }
    unsigned long want = size > 0LL ? (unsigned long)(end - start + 1LL) : 0UL;
    if (req.method.eq_cstr("HEAD")) {
        // No body: an explicit Content-Length tells the server (which
        // honors it for HEAD) what a GET would have sent.
        struct String cl = String::new();
        cl.push_uint((unsigned long long)want);
        r.set_header("Content-Length", cl.as_ptr());
        cl.free();
        return r;
    }
    ?&File f = File::open(file_path, "rb");
    match (f) {
        case null: {
            r.free();
            return HttpResponse::error(404);
        }
        case some(file) : {
            if (start > 0LL) { file.seek(start, 0); }
            if (!r.body.reserve(want)) { file.close(); r.free(); return HttpResponse::error(500); }
            unsigned char buf[16384];
            unsigned long left = want;
            while (left > 0UL) {
                unsigned long ask = left < 16384UL ? left : 16384UL;
                long long got = file.read(buf[0..ask]);
                if (got <= 0LL) { break; }
                unsafe { r.body.push_n((const char*)&buf[0], (unsigned long)got); }
                left = left - (unsigned long)got;
            }
            file.close();
            if (left > 0UL) { r.free(); return HttpResponse::error(500); }
        }
    }
    return r;
}

struct HttpResponse http_serve_dir(const &HttpRequest req, const &char root, const &char url_path,
                                   const ?&StaticOptions optsIn) {
    struct StaticOptions o = sf_opts_(optsIn);
    struct String full = String::new();
    if (!http_safe_join(root, url_path, o.dotfiles, &full)) {
        full.free();
        return HttpResponse::error(403);
    }
    long long size = 0LL;
    long long mtime = 0LL;
    int kind = http_file_info(full.as_ptr(), &size, &mtime);
    if (kind == 2) {
        // Directory: redirect to the slash form, then serve its index.
        struct String up = String::from(url_path);
        long long q = up.find_char('?');
        struct String query = String::new();
        if (q >= 0LL) {
            unsafe { query.push(&up.as_ptr()[q]); }
            up.truncate((unsigned long)q);
        }
        if (!up.ends_with("/")) {
            up.push_char('/');
            up.push_str(&query);
            struct HttpResponse r = HttpResponse::redirect(301, up.as_ptr());
            up.free();
            query.free();
            full.free();
            return r;
        }
        up.free();
        query.free();
        if (!o.index_html) { full.free(); return HttpResponse::error(404); }
        full.push("/index.html");
        kind = http_file_info(full.as_ptr(), &size, &mtime);
    }
    if (kind != 1 && o.spa_fallback) {
        full.clear();
        full.push(root);
        full.push("/index.html");
        kind = http_file_info(full.as_ptr(), &size, &mtime);
    }
    if (kind != 1) { full.free(); return HttpResponse::error(404); }
    struct HttpResponse r = http_serve_file(req, full.as_ptr(), optsIn);
    full.free();
    return r;
}

} // namespace std
