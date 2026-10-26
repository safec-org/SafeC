// safec-doc — the API reference generator behind 'safeguard doc', written in
// SafeC on std's web libraries.
//
// Pipeline:
//
//   safec --emit-doc-json ──► doc.json (declarations, doc comments,
//                              signatures, locations; see the compiler's
//                              DocJson.cpp)
//   config.json + doc.json ──► Crate: modules (one per header, nested by
//                              directory), items, members      (model.sc)
//   doc comment Markdown ──► HTML through a std/serial/markdown render hook:
//                              intra-doc links ([Vec::push]), highlighted
//                              code, heading anchors            (render.sc)
//   Crate ──► one page per module and item, all.html, source pages, the
//             search index; scx templates for the page shell   (pages.sc,
//                                                               templates/)
//   build/doc/ ──► link check (every href / #fragment resolves) (check.sc)
//              ──► 'serve': std/http on the output directory   (main.sc)
//
// Every link in the output is relative, so the result works from any URL
// prefix (a Pages project site, /api/ under another site) and from file://.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/collections/map.h>
#include <std/serial/value.h>

// ── util.sc ──────────────────────────────────────────────────────────────────

int  util_write_file(const &char path, const &String data);
void util_push_esc(&String out, const &char s);
void util_push_esc_n(&String out, const &String s, unsigned long from, unsigned long to);
const &char util_vstr(const &Value obj, const &char key, const &char fallback);
// obj[key], or 'fallback' when missing.
&Value util_vget(const &Value obj, const &char key, const &Value fallback);
long long util_vint(const &Value obj, const &char key, long long fallback);
int  util_vbool(const &Value obj, const &char key);
// "../" repeated once per '/' in 'url' (a page path relative to the root).
struct String util_root_prefix(const &String url);

// ── model.sc ─────────────────────────────────────────────────────────────────

struct Member {
    struct String kind;      // "field" | "method" | "variant"
    struct String name;
    struct String sig;
    struct String doc;
    int md;                  // doc is '///' Markdown (else plain '//' comments)
    long long line;
    struct String def_file;  // out-of-line definition, if any
    long long def_line;
    long long value;         // variant value
    int has_value;
    int is_static;
};

struct Item {
    struct String kind;      // struct union enum trait function macro typedef newtype const global region
    struct String name;
    struct String ns;        // "std", "geo::shapes", ""
    struct String file;      // relative source file
    long long line;
    struct String sig;
    struct String doc;
    int md;
    struct String def_file;
    long long def_line;
    long long module;        // index into Crate.modules
    struct String url;       // page path relative to the doc root
    struct String path;      // display path: ns::Name
    struct Vec<struct Member> members;
};

struct Module {
    struct String path;      // "collections/vec" ("" for the crate root)
    struct String name;      // "vec"
    struct String file;      // its header ("" for a directory-only module)
    struct String doc;
    int md;
    long long parent;        // -1 for the root
    struct Vec<long long> children;
    struct Vec<long long> items;
    struct String url;       // "collections/vec/index.html"
};

// A file with a source page (src/<file>.html).
struct SourceFile {
    struct String rel;       // as in doc.json
    struct String url;       // "src/collections/vec.h.html"
};

struct Crate {
    struct Value cfg;        // config.json
    struct String name;
    struct String version;
    struct String description;
    struct String out;       // output directory
    struct String root;      // the sources' root directory (for source pages)
    struct Vec<struct Module> modules;
    struct Vec<struct Item> items;
    struct Vec<struct SourceFile> sources;
    // Intra-doc link targets ("Vec", "std::Vec", "Vec::push", ...; 'links'
    // maps each name to its index here, see crate_resolve): absolute
    // https:// for another crate's items, else relative to the doc root.
    struct Vec<struct String> link_urls;
    struct StrMap<long long> links;
    struct String readme;    // README.md text ("" if none)
};

// Loads config.json and the doc.json it names. 1 on success.
int crate_load(&Crate c, const &char config_path);

// Index of the item/member link for 'name' (as written in a doc comment or
// signature), or -1.
long long crate_resolve(const &Crate c, const &char name);

// The page path (relative to the root) of a module path's index page.
struct String module_url(const &String module_path);
// The page file name prefix for an item kind ("struct", "fn", ...).
const &char kind_page(const &char kind);
// "Struct", "Function", ... for headings.
const &char kind_title(const &char kind);

// ── render.sc ────────────────────────────────────────────────────────────────

// Doc comment -> HTML. 'from' is the page path (relative to the root) the
// HTML goes on, so links come out relative to it.
struct String render_doc(const &Crate c, const &String text, int md, const &String from);
// The first paragraph as inline HTML (for item tables), or "".
struct String render_summary(const &Crate c, const &String text, int md, const &String from);
// The first sentence as plain text (search index), at most 'max' bytes.
struct String doc_summary_text(const &String text, int md, unsigned long max);
// A signature, highlighted, with known names linked; long parameter lists
// broken one per line. 'self_name' is not linked.
struct String render_sig(const &Crate c, const &String sig, const &String from, const &char self_name);
// A link target ('href' as in the doc.json world: "@@doc/<rel>" for items,
// or a plain URL) relative to page 'from'.
struct String rel_href(const &String from, const &String target);

// ── highlight.sc ─────────────────────────────────────────────────────────────

void highlight_code(const &char lang, const &char code, unsigned long len, &String out);

// ── pages.sc ─────────────────────────────────────────────────────────────────

// Writes every page, the search index and the assets. 1 on success.
int pages_build(&Crate c);

// The pieces of a page the scx template interpolates.
struct PageView {
    struct String title;
    struct String description;
    struct String root;       // "../../" — the doc root, relative to this page
    struct String crate_name;
    struct String version;
    struct String home_href;
    struct String repo;       // repository URL ("" for none)
    struct String sidebar;    // raw
    struct String content;    // raw
    struct String toc;        // raw
    struct String page_class; // "api-item", "api-module", "api-source"
    struct String home;       // raw: the link back to the hosting site ("" for none)
    struct String css;        // <root>assets/style.css
    struct String js;         // <root>assets/doc.js
};

struct String tpl_page(const &PageView v);

// ── check.sc ─────────────────────────────────────────────────────────────────

// Every href/src in every .html under 'dir' that points into it must name
// an existing file, and its #fragment an id on that page. Returns the
// number of broken links (printed to stderr).
int check_links(const &char dir);
