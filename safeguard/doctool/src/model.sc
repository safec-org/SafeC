// safec-doc — config.json + doc.json -> Crate (see doctool.h).
//
// Modules: one per documented header, at its path without the extension
// and without a leading include/ or src/ ("include/geo/shapes.h" ->
// "geo/shapes"); directories are modules too, so the tree nests. A package
// without headers documents its .sc files instead. Pages:
//   <module>/index.html             the module
//   <module>/<kind>.<Name>.html     an item (struct.Vec.html, fn.vec_new.html)
//   ...#method.push / #field.len / #variant.RED   its members
// The same scheme names the pages of an external crate (std), so links into
// its published reference are computed, not looked up.
#include "doctool.h"
#include <std/io.h>
#include <std/io_file.h>
#include <std/path.h>
#include <std/serial/json.h>

const &char kind_page(const &char kind) {
    struct String k = String::from(kind);
    const &char r = "item";
    if (k.eq_cstr("struct")) { r = "struct"; }
    else if (k.eq_cstr("union")) { r = "union"; }
    else if (k.eq_cstr("enum")) { r = "enum"; }
    else if (k.eq_cstr("trait")) { r = "trait"; }
    else if (k.eq_cstr("function")) { r = "fn"; }
    else if (k.eq_cstr("macro")) { r = "macro"; }
    else if (k.eq_cstr("typedef") || k.eq_cstr("newtype")) { r = "type"; }
    else if (k.eq_cstr("const")) { r = "constant"; }
    else if (k.eq_cstr("global")) { r = "static"; }
    else if (k.eq_cstr("region")) { r = "region"; }
    k.free();
    return r;
}

const &char kind_title(const &char kind) {
    struct String k = String::from(kind);
    const &char r = "Item";
    if (k.eq_cstr("struct")) { r = "Struct"; }
    else if (k.eq_cstr("union")) { r = "Union"; }
    else if (k.eq_cstr("enum")) { r = "Enum"; }
    else if (k.eq_cstr("trait")) { r = "Trait"; }
    else if (k.eq_cstr("function")) { r = "Function"; }
    else if (k.eq_cstr("macro")) { r = "Macro"; }
    else if (k.eq_cstr("typedef")) { r = "Type alias"; }
    else if (k.eq_cstr("newtype")) { r = "Newtype"; }
    else if (k.eq_cstr("const")) { r = "Constant"; }
    else if (k.eq_cstr("global")) { r = "Global"; }
    else if (k.eq_cstr("region")) { r = "Region"; }
    k.free();
    return r;
}

// "include/geo/shapes.h" -> "geo/shapes".
static struct String module_path_of(const &String rel) {
    struct String p = rel.clone();
    if (p.starts_with("include/")) { p.remove_range(0UL, 8UL); }
    else if (p.starts_with("src/")) { p.remove_range(0UL, 4UL); }
    long long dot = p.rfind_char('.');
    long long slash = p.rfind_char('/');
    if (dot > slash) { p.truncate((unsigned long)dot); }
    return p;
}

struct String module_url(const &String module_path) {
    struct String u = module_path.clone();
    if (u.len > 0UL) { u.push_char('/'); }
    u.push("index.html");
    return u;
}

static struct String module_dir(const &String module_path) {
    struct String u = module_path.clone();
    if (u.len > 0UL) { u.push_char('/'); }
    return u;
}

static int is_header(const &String rel) { return rel.ends_with(".h"); }

// ── links ────────────────────────────────────────────────────────────────────

// Makes 'url' (an item page, ".../kind.Name.html") unique ignoring case,
// and claims it: the first page keeps its path, a later one whose path
// matches it but for case (macro.PRIx32 / macro.PRIX32), or exactly (one
// declaration under two #if branches), gets "-2", "-3", ... On a
// case-insensitive file system (macOS, Windows) the two would otherwise be
// one file, the second overwriting the first. Items are claimed in doc.json
// order, so another crate's reference linking into this one (its items in
// the same order) picks the same paths.
static void claim_page(&Crate c, &String url) {
    struct String base = url.strip_suffix(".html");
    struct String cand = url.clone();
    long long n = 2LL;
    for (;;) {
        struct String lc = cand.to_lower();
        if (!c.pages.contains(lc.as_ptr())) {
            c.pages.insert(lc.as_ptr(), 1LL);
            lc.free();
            break;
        }
        lc.free();
        cand.clear();
        cand.push_str(&base);
        cand.push("-");
        cand.push_int(n);
        cand.push(".html");
        n = n + 1LL;
    }
    url.clear();
    url.push_str(&cand);
    cand.free();
    base.free();
}

// The first link registered for a name wins.
static void add_link(&Crate c, const &String key, const &String url) {
    if (key.len == 0UL) { return; }
    if (c.links.contains(key.as_ptr())) { return; }
    c.link_urls.push(url.clone());
    c.links.insert(key.as_ptr(), (long long)c.link_urls.len - 1LL);
}

static void add_link_cstr(&Crate c, const &char key, const &String url) {
    struct String k = String::from(key);
    add_link(c, &k, url);
    k.free();
}

long long crate_resolve(const &Crate c, const &char name) {
    ?&long long e = c.links.get(name);
    match (e) { case some(v): { return *v; } case null: { } }
    return -1LL;
}

// Registers 'Name', 'ns::Name' and each member 'Name::m' / 'ns::Name::m'.
static void link_item(&Crate c, const &Value it, const &String url) {
    struct String name = String::from(util_vstr(it, "name", ""));
    struct String ns = String::from(util_vstr(it, "ns", ""));
    struct String full = ns.clone();
    if (full.len > 0UL) { full.push("::"); }
    full.push_str(&name);
    add_link(c, &full, url);
    add_link(c, &name, url);
    ?&Value members = it.object_get("members");
    match (members) {
        case some(ms): {
            for (unsigned long i = 0UL; i < ms.array_len(); i = i + 1UL) {
                &Value m = ms.array_at(i);
                if (util_vbool(m, "hidden")) { continue; }
                struct String murl = url.clone();
                murl.push("#");
                murl.push(util_vstr(m, "kind", "method"));
                murl.push(".");
                murl.push(util_vstr(m, "name", ""));
                struct String k1 = name.clone();
                k1.push("::");
                k1.push(util_vstr(m, "name", ""));
                struct String k2 = full.clone();
                k2.push("::");
                k2.push(util_vstr(m, "name", ""));
                add_link(c, &k2, &murl);
                add_link(c, &k1, &murl);
                // Enumerators are names of their own in C: [SHAPE_CIRCLE].
                struct String mk = String::from(util_vstr(m, "kind", ""));
                if (mk.eq_cstr("variant")) { add_link_cstr(c, util_vstr(m, "name", ""), &murl); }
                mk.free();
                k1.free(); k2.free(); murl.free();
            }
        }
        case null: { }
    }
    name.free(); ns.free(); full.free();
}

// ── modules ──────────────────────────────────────────────────────────────────

static long long find_module(&Crate c, const &String path) {
    for (unsigned long i = 0UL; i < c.modules.len; i = i + 1UL) {
        if (c.modules.at(i).path.eq(path)) { return (long long)i; }
    }
    return -1LL;
}

// The module at 'path', created (with its parents) when missing.
static long long ensure_module(&Crate c, const &String path) {
    long long found = find_module(c, path);
    if (found >= 0LL) { return found; }
    long long parent = 0LL;
    long long slash = path.rfind_char('/');
    if (path.len > 0UL) {
        struct String pp = String::new();
        if (slash >= 0LL) { pp = path.substr(0UL, (unsigned long)slash); }
        parent = ensure_module(c, &pp);
        pp.free();
    }
    struct Module m;
    m.path = path.clone();
    m.name = slash >= 0LL ? path.substr((unsigned long)slash + 1UL, path.len) : path.clone();
    if (path.len == 0UL) { m.name.push_str(&c.name); }
    m.file = String::new();
    m.doc = String::new();
    m.md = 0;
    m.parent = path.len == 0UL ? -1LL : parent;
    m.children = Vec::new();
    m.items = Vec::new();
    m.url = module_url(path);
    c.modules.push(m);
    long long idx = (long long)c.modules.len - 1LL;
    if (path.len > 0UL) {
        &Module pm = c.modules.at((unsigned long)parent);
        pm.children.push(idx);
    }
    struct String key = path.clone();
    key.replace_all("/", "::");
    struct String url = module_url(path);   // a copy: add_link grows c's tables
    add_link(c, &key, &url);
    key.free(); url.free();
    return idx;
}

// ── items ────────────────────────────────────────────────────────────────────

// Shown in the reference: not hidden, an extern C declaration only when
// documented, and (when the package has headers) declared in a header.
static int item_public(const &Value it, int has_headers, int show_private) {
    if (util_vbool(it, "hidden") && !show_private) { return 0; }
    struct String file = String::from(util_vstr(it, "file", ""));
    int ok = !has_headers || is_header(&file);
    file.free();
    if (util_vbool(it, "extern") && it.object_get("doc").is_null() && !show_private) { ok = 0; }
    return ok;
}

static struct String item_page_name(const &Value it) {
    struct String p = String::from(kind_page(util_vstr(it, "kind", "")));
    p.push(".");
    p.push(util_vstr(it, "name", ""));
    p.push(".html");
    return p;
}

static void load_members(&Item item, const &Value it, int show_private) {
    item.members = Vec::new();
    ?&Value members = it.object_get("members");
    match (members) {
        case some(ms): {
            for (unsigned long i = 0UL; i < ms.array_len(); i = i + 1UL) {
                &Value m = ms.array_at(i);
                if (util_vbool(m, "hidden") && !show_private) { continue; }
                struct Member mb;
                mb.kind = String::from(util_vstr(m, "kind", "method"));
                mb.name = String::from(util_vstr(m, "name", ""));
                mb.sig = String::from(util_vstr(m, "sig", ""));
                mb.doc = String::from(util_vstr(m, "doc", ""));
                mb.md = util_vbool(m, "md");
                mb.line = util_vint(m, "line", 0LL);
                mb.def_file = String::from(util_vstr(m, "def_file", ""));
                mb.def_line = util_vint(m, "def_line", 0LL);
                mb.has_value = !m.object_get("value").is_null();
                mb.value = util_vint(m, "value", 0LL);
                mb.is_static = util_vbool(m, "static");
                item.members.push(mb);
            }
        }
        case null: { }
    }
}

static int read_json(const &char path, &Value out) {
    struct String text = String::new();
    if (!File::read_to_string(path, &text)) {
        eprint("safec-doc: cannot read ");
        eprintln(path);
        text.free();
        return 0;
    }
    int ok = 0;
    *out = json_parse(text.as_ptr(), &ok);
    text.free();
    if (!ok) {
        eprint("safec-doc: malformed JSON in ");
        eprintln(path);
        return 0;
    }
    return 1;
}

int crate_load(&Crate c, const &char config_path) {
    if (!read_json(config_path, &c.cfg)) { return 0; }
    struct Value doc;
    if (!read_json(util_vstr(&c.cfg, "input", ""), &doc)) { return 0; }
    c.name = String::from(util_vstr(&c.cfg, "name", "crate"));
    c.version = String::from(util_vstr(&c.cfg, "version", ""));
    c.description = String::from(util_vstr(&c.cfg, "description", ""));
    c.out = String::from(util_vstr(&c.cfg, "out", "build/doc"));
    c.root = String::from(util_vstr(&c.cfg, "root", "."));
    c.modules = Vec::new();
    c.items = Vec::new();
    c.sources = Vec::new();
    c.link_urls = Vec::new();
    c.readme = String::new();
    struct String readme = String::from(util_vstr(&c.cfg, "readme", ""));
    if (readme.len > 0UL) { File::read_to_string(readme.as_ptr(), &c.readme); }
    readme.free();
    int show_private = util_vbool(&c.cfg, "private");

    c.links = StrMap::new();
    c.pages = StrMap::new();

    struct String empty = String::new();
    ensure_module(c, &empty);

    // Files: modules (headers, or every file without any) and source pages.
    &Value files = util_vget(&doc, "files", &doc);
    int has_headers = 0;
    for (unsigned long i = 0UL; i < files.array_len(); i = i + 1UL) {
        if (util_vbool(files.array_at(i), "header")) { has_headers = 1; }
    }
    for (unsigned long i = 0UL; i < files.array_len(); i = i + 1UL) {
        &Value f = files.array_at(i);
        struct String rel = String::from(util_vstr(f, "path", ""));
        struct SourceFile sf;
        sf.rel = rel.clone();
        sf.url = String::from("src/");
        sf.url.push_str(&rel);
        sf.url.push(".html");
        c.sources.push(sf);
        if ((!has_headers || is_header(&rel)) && (!util_vbool(f, "hidden") || show_private)) {
            struct String mp = module_path_of(&rel);
            long long mi = ensure_module(c, &mp);
            &Module m = c.modules.at((unsigned long)mi);
            if (m.file.len == 0UL) { m.file.push_str(&rel); }
            if (m.doc.len == 0UL) {
                m.doc.push(util_vstr(f, "doc", ""));
                m.md = util_vbool(f, "md");
            }
            mp.free();
        }
        rel.free();
    }

    // Local items first (their names win), then other crates' for links.
    struct Value no_externs = Value::object();
    &Value externs = util_vget(&c.cfg, "externs", &no_externs);
    &Value items = util_vget(&doc, "items", &doc);
    for (int pass = 0; pass < 2; pass = pass + 1) {
        for (unsigned long i = 0UL; i < items.array_len(); i = i + 1UL) {
            &Value it = items.array_at(i);
            struct String crate_name = String::from(util_vstr(it, "crate", ""));
            int local = crate_name.len == 0UL;
            if ((pass == 0) != local) { crate_name.free(); continue; }
            struct String file = String::from(util_vstr(it, "file", ""));
            struct String mp = module_path_of(&file);
            if (!local) {
                // Another crate's item: a link into its reference, when it
                // is published (always a header item there).
                struct String url = String::from(util_vstr(externs, crate_name.as_ptr(), ""));
                if (url.len > 0UL && is_header(&file) && item_public(it, 1, 0)) {
                    if (!url.ends_with("/")) { url.push_char('/'); }
                    struct String dir = module_dir(&mp);
                    url.push_str(&dir);
                    struct String page = item_page_name(it);
                    url.push_str(&page);
                    claim_page(c, &url);
                    link_item(c, it, &url);
                    dir.free(); page.free();
                }
                url.free();
                file.free(); mp.free(); crate_name.free();
                continue;
            }
            if (!item_public(it, has_headers, show_private)) { file.free(); mp.free(); crate_name.free(); continue; }
            long long mi = ensure_module(c, &mp);
            struct Item item;
            item.kind = String::from(util_vstr(it, "kind", ""));
            item.name = String::from(util_vstr(it, "name", ""));
            item.ns = String::from(util_vstr(it, "ns", ""));
            item.file = file.clone();
            item.line = util_vint(it, "line", 0LL);
            item.sig = String::from(util_vstr(it, "sig", ""));
            item.doc = String::from(util_vstr(it, "doc", ""));
            item.md = util_vbool(it, "md");
            item.def_file = String::from(util_vstr(it, "def_file", ""));
            item.def_line = util_vint(it, "def_line", 0LL);
            item.module = mi;
            item.path = item.ns.clone();
            if (item.path.len > 0UL) { item.path.push("::"); }
            item.path.push_str(&item.name);
            item.url = module_dir(&c.modules.at((unsigned long)mi).path);
            struct String page = item_page_name(it);
            item.url.push_str(&page);
            page.free();
            // Its own page, even beside one of the same name but for case.
            claim_page(c, &item.url);
            crate_name.free();
            load_members(&item, it, show_private);
            link_item(c, it, &item.url);
            c.items.push(item);
            &Module owner = c.modules.at((unsigned long)mi);
            owner.items.push((long long)c.items.len - 1LL);
            file.free(); mp.free();
        }
    }
    no_externs.free();
    doc.free();
    empty.free();
    return 1;
}
