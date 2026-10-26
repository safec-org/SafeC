// safec-doc — the SafeC API reference generator ('safeguard doc' runs it).
//
//   safec-doc build <config.json>        doc.json -> HTML (config: see below)
//   safec-doc check <dir>                link check of a generated reference
//   safec-doc serve <dir> [--port 8000]  serve it over std/http
//   safec-doc --version                  the doc.json format it reads
//
// config.json (written by safeguard):
//   input        the 'safec --emit-doc-json' output
//   out          output directory (emptied first)
//   assets       directory copied to <out>/assets (style.css, doc.js,
//                doc.wasm, safec_dom.js)
//   root         the directory doc.json's file paths are relative to
//   name, version, description, repository, readme (a path)
//   source_url   "https://host/o/r/blob/<ref>/{file}" for source pages,
//                with 'source_prefix' put before {file}
//   externs      { "std": "https://…/api/" }: other crates' references
//   private      also document hidden items
#include "doctool.h"
#include <std/io.h>
#include <std/io_file.h>
#include <std/str.h>
#include <std/sys.h>
#include <std/path.h>
#include <std/http/http.h>

// ── serve ────────────────────────────────────────────────────────────────────

static struct String gServeRoot;

static const char* mime_for(const &String path) {
    if (path.ends_with(".html")) { return "text/html; charset=utf-8"; }
    if (path.ends_with(".css")) { return "text/css; charset=utf-8"; }
    if (path.ends_with(".js") || path.ends_with(".mjs")) { return "text/javascript; charset=utf-8"; }
    if (path.ends_with(".json")) { return "application/json"; }
    if (path.ends_with(".wasm")) { return "application/wasm"; }
    if (path.ends_with(".svg")) { return "image/svg+xml"; }
    if (path.ends_with(".png")) { return "image/png"; }
    return "text/plain; charset=utf-8";
}

static struct HttpResponse serve_handler(&HttpRequest req) {
    struct String path = req.path.clone();
    long long q = path.find_char('?');
    if (q >= 0LL) { path.truncate((unsigned long)q); }
    long long h = path.find_char('#');
    if (h >= 0LL) { path.truncate((unsigned long)h); }
    if (path.ends_with("/")) { path.push("index.html"); }
    if (path.contains("..")) { path.clear(); path.push("/index.html"); }
    struct HttpResponse resp;
    resp.status = 200;
    resp.body = String::new();
    struct String rel = path.strip_prefix("/");
    struct String file = path_join(gServeRoot.as_ptr(), rel.as_ptr());
    rel.free();
    if (!File::read_to_string(file.as_ptr(), &resp.body)) {
        resp.status = 404;
        resp.body.push("<!doctype html><title>Not found</title><p>Not found. <a href=\"/index.html\">API index</a></p>");
        path.clear();
        path.push("404.html");
    }
    resp.headers = String::from("Content-Type: ");
    unsafe { resp.headers.push(mime_for(&path)); }
    resp.headers.push("\r\nCache-Control: no-store");
    file.free();
    path.free();
    return resp;
}

// ── main ─────────────────────────────────────────────────────────────────────

static void usage() {
    eprintln("usage: safec-doc build <config.json>");
    eprintln("       safec-doc check <dir>");
    eprintln("       safec-doc serve <dir> [--port N]");
    eprintln("       safec-doc --version");
}

int main(int argc, char** argv) {
    // argv is the one raw-pointer boundary: copy it into Strings.
    struct Vec<struct String> args = Vec::new();
    for (int i = 0; i < argc; i = i + 1) {
        unsafe { args.push(String::from(argv[i])); }
    }
    if (args.len < 2UL) { usage(); return 2; }
    &String cmd = args.at(1UL);
    if (cmd.eq_cstr("--version")) {
        println("safec-doc 0.1.0 (doc.json format 1)");
        return 0;
    }
    if (args.len < 3UL) { usage(); return 2; }
    &String arg = args.at(2UL);
    if (cmd.eq_cstr("build")) {
        struct Crate c;
        if (!crate_load(&c, arg.as_ptr())) { return 1; }
        int ok = pages_build(&c);
        print("safec-doc: ");
        print_int((long long)c.items.len);
        print(" items in ");
        print_int((long long)c.modules.len);
        print(" modules -> ");
        println(c.out.as_ptr());
        return ok ? 0 : 1;
    }
    if (cmd.eq_cstr("check")) {
        return check_links(arg.as_ptr()) == 0 ? 0 : 1;
    }
    if (cmd.eq_cstr("serve")) {
        int port = 8000;
        for (unsigned long i = 3UL; i + 1UL < args.len; i = i + 1UL) {
            if (args.at(i).eq_cstr("--port")) { port = sys_atoi(args.at(i + 1UL).as_ptr()); }
        }
        gServeRoot = arg.clone();
        print("safec-doc: serving ");
        print(arg.as_ptr());
        print(" at http://localhost:");
        print_int((long long)port);
        println("/");
        return http_serve((unsigned short)port, serve_handler) == 0 ? 0 : 1;
    }
    usage();
    return 2;
}
