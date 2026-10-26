// safec-doc — every page of the reference (see doctool.h):
//   index.html                 the crate: description, README, module tree
//   all.html                   every item, by kind
//   <module>/index.html        a module: its doc, submodules, items by kind
//   <module>/<kind>.<Name>.html  an item: declaration, doc, fields,
//                              methods / variants with anchors
//   src/<file>.html            highlighted source with #L<n> line anchors
//   search-index.json          what the wasm client searches
//   assets/                    style.css, doc.js, doc.wasm, safec_dom.js
#include "doctool.h"
#include <std/io.h>
#include <std/io_file.h>
#include <std/dir.h>
#include <std/path.h>
#include <std/serial/json.h>

// Item kinds, in the order module pages list them.
static const &char KIND_ORDER(int i) {
    if (i == 0) { return "struct"; }
    if (i == 1) { return "union"; }
    if (i == 2) { return "enum"; }
    if (i == 3) { return "trait"; }
    if (i == 4) { return "typedef"; }
    if (i == 5) { return "newtype"; }
    if (i == 6) { return "function"; }
    if (i == 7) { return "macro"; }
    if (i == 8) { return "const"; }
    if (i == 9) { return "global"; }
    return "region";
}
#define KIND_COUNT 11

static const &char KIND_PLURAL(int i) {
    if (i == 0) { return "Structs"; }
    if (i == 1) { return "Unions"; }
    if (i == 2) { return "Enums"; }
    if (i == 3) { return "Traits"; }
    if (i == 4) { return "Type aliases"; }
    if (i == 5) { return "Newtypes"; }
    if (i == 6) { return "Functions"; }
    if (i == 7) { return "Macros"; }
    if (i == 8) { return "Constants"; }
    if (i == 9) { return "Globals"; }
    return "Regions";
}

static const &char KIND_ANCHOR(int i) {
    if (i == 0) { return "structs"; }
    if (i == 1) { return "unions"; }
    if (i == 2) { return "enums"; }
    if (i == 3) { return "traits"; }
    if (i == 4) { return "types"; }
    if (i == 5) { return "newtypes"; }
    if (i == 6) { return "functions"; }
    if (i == 7) { return "macros"; }
    if (i == 8) { return "constants"; }
    if (i == 9) { return "globals"; }
    return "regions";
}

// ── small builders ───────────────────────────────────────────────────────────

static void push_href(&String out, const &String from, const &String target) {
    struct String h = rel_href(from, target);
    out.push(" href=\"");
    util_push_esc(out, h.as_ptr());
    out.push("\"");
    h.free();
}

static void push_href_cstr(&String out, const &String from, const &char target) {
    struct String t = String::from(target);
    push_href(out, from, &t);
    t.free();
}

// "collections::vec" for module path "collections/vec" (the crate name for "").
static struct String module_display(const &Crate c, const &Module m) {
    if (m.path.len == 0UL) { return c.name.clone(); }
    struct String d = m.path.clone();
    d.replace_all("/", "::");
    return d;
}

static struct String source_url(const &String rel, long long line) {
    struct String u = String::from("src/");
    u.push_str(rel);
    u.push(".html");
    if (line > 0LL) { u.push("#L"); u.push_int(line); }
    return u;
}

static void push_src_link(&String out, const &String from, const &String rel, long long line, const &char label) {
    if (rel.len == 0UL) { return; }
    struct String u = source_url(rel, line);
    out.push("<a class=\"src-link\"");
    push_href(out, from, &u);
    out.push(">");
    util_push_esc(out, label);
    out.push("</a>");
    u.free();
}

// ── sidebar ──────────────────────────────────────────────────────────────────

// Crate links, the module's ancestry, its submodules and its items.
static void build_sidebar(const &Crate c, long long mod_idx, const &String from, const &String active, &String out) {
    out.push("<section class=\"sb-group\"><p class=\"sb-title\"><a class=\"sb-crate\"");
    push_href_cstr(out, from, "index.html");
    out.push(">");
    util_push_esc(out, c.name.as_ptr());
    out.push("</a></p><ul><li><a class=\"sb-link\"");
    push_href_cstr(out, from, "all.html");
    out.push(">All items</a></li></ul></section>");
    if (mod_idx < 0LL) { mod_idx = 0LL; }
    &Module m = c.modules.at((unsigned long)mod_idx);
    // Breadcrumbs: root → … → this module.
    struct Vec<long long> chain = Vec::new();
    long long cur = mod_idx;
    while (cur > 0LL) { chain.push(cur); cur = c.modules.at((unsigned long)cur).parent; }
    if (chain.len > 0UL) {
        out.push("<section class=\"sb-group\"><p class=\"sb-title\">Module</p><ul class=\"sb-crumbs\">");
        unsigned long n = chain.len;
        while (n > 0UL) {
            n = n - 1UL;
            &Module a = c.modules.at((unsigned long)*chain.at(n));
            out.push("<li><a class=\"sb-link");
            if (a.url.eq(active)) { out.push(" active"); }
            out.push("\"");
            push_href(out, from, &a.url);
            out.push(">");
            struct String d = module_display(c, a);
            util_push_esc(out, d.as_ptr());
            d.free();
            out.push("</a></li>");
        }
        out.push("</ul></section>");
    }
    chain.free();
    if (m.children.len > 0UL) {
        out.push("<section class=\"sb-group\"><p class=\"sb-title\">Modules</p><ul>");
        for (unsigned long i = 0UL; i < m.children.len; i = i + 1UL) {
            &Module ch = c.modules.at((unsigned long)*m.children.at(i));
            out.push("<li><a class=\"sb-link\"");
            push_href(out, from, &ch.url);
            out.push(">");
            util_push_esc(out, ch.name.as_ptr());
            out.push("</a></li>");
        }
        out.push("</ul></section>");
    }
    for (int k = 0; k < KIND_COUNT; k = k + 1) {
        int any = 0;
        struct String kname = String::from(KIND_ORDER(k));
        for (unsigned long i = 0UL; i < m.items.len; i = i + 1UL) {
            &Item it = c.items.at((unsigned long)*m.items.at(i));
            if (!it.kind.eq(&kname)) { continue; }
            if (!any) {
                out.push("<section class=\"sb-group\"><p class=\"sb-title\">");
                out.push(KIND_PLURAL(k));
                out.push("</p><ul>");
                any = 1;
            }
            out.push("<li><a class=\"sb-link kind-");
            util_push_esc(out, it.kind.as_ptr());
            if (it.url.eq(active)) { out.push(" active"); }
            out.push("\"");
            push_href(out, from, &it.url);
            out.push(">");
            util_push_esc(out, it.name.as_ptr());
            out.push("</a></li>");
        }
        if (any) { out.push("</ul></section>"); }
        kname.free();
    }
}

// ── page assembly ────────────────────────────────────────────────────────────

static int write_page(const &Crate c, const &String url, const &String title, const &String description,
                      const &String sidebar, const &String content, const &String toc, const &char cls) {
    struct PageView v;
    v.title = title.clone();
    v.description = description.clone();
    v.root = util_root_prefix(url);
    v.crate_name = c.name.clone();
    v.version = c.version.clone();
    v.home_href = v.root.clone();
    v.home_href.push("index.html");
    v.repo = String::from(util_vstr(&c.cfg, "repository", ""));
    v.sidebar = sidebar.clone();
    v.content = content.clone();
    v.toc = toc.clone();
    v.page_class = String::from(cls);
    v.css = v.root.clone();
    v.css.push("assets/style.css");
    v.js = v.root.clone();
    v.js.push("assets/doc.js");
    struct String html = tpl_page(&v);
    struct String path = path_join(c.out.as_ptr(), url.as_ptr());
    int ok = util_write_file(path.as_ptr(), &html);
    if (!ok) { eprint("safec-doc: cannot write "); eprintln(path.as_ptr()); }
    html.free(); path.free();
    v.title.free(); v.description.free(); v.root.free(); v.crate_name.free(); v.version.free();
    v.home_href.free(); v.repo.free(); v.sidebar.free(); v.content.free(); v.toc.free();
    v.page_class.free(); v.css.free(); v.js.free();
    return ok;
}

static void toc_entry(&String toc, const &char id, const &char text, int level) {
    toc.push("<li><a class=\"toc-l");
    toc.push_int((long long)level);
    toc.push("\" href=\"#");
    util_push_esc(toc, id);
    toc.push("\">");
    util_push_esc(toc, text);
    toc.push("</a></li>");
}

// A table of items: name (linked) and summary.
static void item_rows(const &Crate c, const &Vec<long long> items, const &char kind, const &String from, &String out) {
    for (unsigned long i = 0UL; i < items.len; i = i + 1UL) {
        &Item it = c.items.at((unsigned long)*items.at(i));
        if (!it.kind.eq_cstr(kind)) { continue; }
        out.push("<tr><td class=\"item-name\"><a class=\"kind-");
        util_push_esc(out, it.kind.as_ptr());
        out.push("\"");
        push_href(out, from, &it.url);
        out.push(">");
        util_push_esc(out, it.name.as_ptr());
        out.push("</a></td><td class=\"item-summary\">");
        struct String s = render_summary(c, &it.doc, it.md, from);
        out.push_str(&s);
        s.free();
        out.push("</td></tr>");
    }
}

static int count_kind(const &Crate c, const &Vec<long long> items, const &char kind) {
    int n = 0;
    for (unsigned long i = 0UL; i < items.len; i = i + 1UL) {
        if (c.items.at((unsigned long)*items.at(i)).kind.eq_cstr(kind)) { n = n + 1; }
    }
    return n;
}

// Module tree (index page): nested lists with summaries.
static void module_tree(const &Crate c, long long idx, const &String from, &String out) {
    &Module m = c.modules.at((unsigned long)idx);
    if (m.children.len == 0UL) { return; }
    out.push("<ul class=\"module-tree\">");
    for (unsigned long i = 0UL; i < m.children.len; i = i + 1UL) {
        long long ci = *m.children.at(i);
        &Module ch = c.modules.at((unsigned long)ci);
        out.push("<li><a class=\"mod\"");
        push_href(out, from, &ch.url);
        out.push(">");
        util_push_esc(out, ch.name.as_ptr());
        out.push("</a>");
        struct String s = render_summary(c, &ch.doc, ch.md, from);
        if (s.len > 0UL) { out.push(" <span class=\"mod-summary\">— "); out.push_str(&s); out.push("</span>"); }
        s.free();
        module_tree(c, ci, from, out);
        out.push("</li>");
    }
    out.push("</ul>");
}

// The module's own doc minus std's boilerplate "SafeC Standard Library — "
// title prefix.
static struct String module_doc_text(const &Module m) {
    struct String d = m.doc.clone();
    if (d.starts_with("SafeC Standard Library — ")) { d.remove_range(0UL, 27UL); }
    else if (d.starts_with("SafeC Standard Library - ")) { d.remove_range(0UL, 25UL); }
    return d;
}

// What a user writes to include 'file': "<std/collections/vec.h>",
// "<geo/shapes.h>" (include/ and src/ are on the include path).
static void push_include(const &Crate c, const &String file, &String out) {
    struct String f = file.clone();
    if (f.starts_with("include/")) { f.remove_range(0UL, 8UL); }
    else if (f.starts_with("src/")) { f.remove_range(0UL, 4UL); }
    out.push("<code>#include &lt;");
    util_push_esc(out, util_vstr(&c.cfg, "include_prefix", ""));
    util_push_esc(out, f.as_ptr());
    out.push("&gt;</code> ");
    f.free();
}

static int build_module_page(const &Crate c, long long idx) {
    &Module m = c.modules.at((unsigned long)idx);
    struct String from = m.url.clone();
    struct String content = String::new();
    struct String toc = String::new();
    struct String disp = module_display(c, m);
    content.push("<div class=\"item-head\"><h1>");
    content.push(idx == 0LL ? "Crate " : "Module ");
    content.push("<span class=\"kind-module\">");
    util_push_esc(&content, disp.as_ptr());
    content.push("</span>");
    if (idx == 0LL && c.version.len > 0UL) {
        content.push(" <span class=\"crate-ver\">"); util_push_esc(&content, c.version.as_ptr()); content.push("</span>");
    }
    content.push("</h1>");
    if (m.file.len > 0UL) {
        content.push("<p class=\"item-meta\">");
        if (m.file.ends_with(".h")) { push_include(c, &m.file, &content); }
        push_src_link(&content, &from, &m.file, 0LL, "source");
        content.push("</p>");
    }
    content.push("</div>");
    struct String mdoc = module_doc_text(m);
    struct String doc_html = render_doc(c, &mdoc, m.md, &from);
    content.push("<div class=\"docblock\">"); content.push_str(&doc_html); content.push("</div>");
    doc_html.free(); mdoc.free();
    if (idx == 0LL) {
        if (c.description.len > 0UL && m.doc.len == 0UL) {
            content.push("<p class=\"crate-desc\">"); util_push_esc(&content, c.description.as_ptr()); content.push("</p>");
        }
        if (c.readme.len > 0UL) {
            struct String rd = render_doc(c, &c.readme, 1, &from);
            content.push("<section class=\"readme docblock\">"); content.push_str(&rd); content.push("</section>");
            rd.free();
        }
    }
    if (m.children.len > 0UL) {
        content.push("<h2 id=\"modules\">Modules</h2>");
        toc_entry(&toc, "modules", "Modules", 1);
        if (idx == 0LL) {
            module_tree(c, idx, &from, &content);
        } else {
            content.push("<table class=\"item-table\">");
            for (unsigned long i = 0UL; i < m.children.len; i = i + 1UL) {
                &Module ch = c.modules.at((unsigned long)*m.children.at(i));
                content.push("<tr><td class=\"item-name\"><a class=\"kind-module\"");
                push_href(&content, &from, &ch.url);
                content.push(">");
                util_push_esc(&content, ch.name.as_ptr());
                content.push("</a></td><td class=\"item-summary\">");
                struct String s = render_summary(c, &ch.doc, ch.md, &from);
                content.push_str(&s);
                s.free();
                content.push("</td></tr>");
            }
            content.push("</table>");
        }
    }
    for (int k = 0; k < KIND_COUNT; k = k + 1) {
        const &char kind = KIND_ORDER(k);
        int n = 0;
        n = count_kind(c, &m.items, kind);
        if (n == 0) { continue; }
        content.push("<h2 id=\"");
        content.push(KIND_ANCHOR(k));
        content.push("\">");
        content.push(KIND_PLURAL(k));
        content.push("</h2><table class=\"item-table\">");
        item_rows(c, &m.items, kind, &from, &content);
        toc_entry(&toc, KIND_ANCHOR(k), KIND_PLURAL(k), 1);
        content.push("</table>");
    }
    struct String sidebar = String::new();
    build_sidebar(c, idx, &from, &m.url, &sidebar);
    struct String title = disp.clone();
    title.push(" - ");
    title.push_str(&c.name);
    title.push(" API");
    struct String desc = doc_summary_text(&m.doc, m.md, 160UL);
    if (desc.len == 0UL) { desc.push("API reference for "); desc.push_str(&disp); }
    int ok = write_page(c, &m.url, &title, &desc, &sidebar, &content, &toc, "api-module");
    from.free(); content.free(); toc.free(); disp.free();
    sidebar.free(); title.free(); desc.free();
    return ok;
}

// ── item pages ───────────────────────────────────────────────────────────────

// The declaration block: a struct with its fields and method signatures, an
// enum with its variants, else the signature.
static void declaration(const &Crate c, const &Item it, const &String from, &String out) {
    out.push("<div class=\"code-block decl\"><pre><code>");
    struct String head = render_sig(c, &it.sig, from, it.name.as_ptr());
    out.push_str(&head);
    head.free();
    int body = it.kind.eq_cstr("struct") || it.kind.eq_cstr("union") || it.kind.eq_cstr("enum") || it.kind.eq_cstr("trait");
    if (body) {
        out.push(" {");
        int any = 0;
        for (unsigned long i = 0UL; i < it.members.len; i = i + 1UL) {
            &Member m = it.members.at(i);
            if (m.kind.eq_cstr("method") && !it.kind.eq_cstr("trait")) { continue; }
            out.push("\n    ");
            if (m.kind.eq_cstr("variant")) {
                out.push("<a class=\"tok-member\" href=\"#variant.");
                util_push_esc(out, m.name.as_ptr());
                out.push("\">");
                util_push_esc(out, m.name.as_ptr());
                out.push("</a>");
                if (m.has_value) { out.push(" = <span class=\"tok-num\">"); out.push_int(m.value); out.push("</span>"); }
                out.push(",");
            } else {
                struct String s = render_sig(c, &m.sig, from, "");
                out.push_str(&s);
                s.free();
                out.push(";");
            }
            any = 1;
        }
        out.push(any ? "\n}" : " }");
    }
    out.push("</code></pre></div>");
}

static void member_section(const &Crate c, const &Item it, const &char kind, const &char title, const &char anchor,
                           const &String from, &String out, &String toc) {
    int n = 0;
    for (unsigned long i = 0UL; i < it.members.len; i = i + 1UL) {
        if (it.members.at(i).kind.eq_cstr(kind)) { n = n + 1; }
    }
    if (n == 0) { return; }
    out.push("<h2 id=\"");
    util_push_esc(out, anchor);
    out.push("\">");
    util_push_esc(out, title);
    out.push("</h2>");
    toc_entry(toc, anchor, title, 1);
    for (unsigned long i = 0UL; i < it.members.len; i = i + 1UL) {
        &Member m = it.members.at(i);
        if (!m.kind.eq_cstr(kind)) { continue; }
        struct String id = m.kind.clone();
        id.push(".");
        id.push_str(&m.name);
        out.push("<section class=\"member\" id=\"");
        util_push_esc(out, id.as_ptr());
        out.push("\"><div class=\"member-head\"><a class=\"anchor\" href=\"#");
        util_push_esc(out, id.as_ptr());
        out.push("\" aria-label=\"Link to this member\">§</a><code class=\"member-sig\">");
        if (m.kind.eq_cstr("variant")) {
            out.push("<span class=\"tok-fn\">"); util_push_esc(out, m.name.as_ptr()); out.push("</span>");
            if (m.has_value) { out.push(" = <span class=\"tok-num\">"); out.push_int(m.value); out.push("</span>"); }
        } else {
            struct String s = render_sig(c, &m.sig, from, m.name.as_ptr());
            out.push_str(&s);
            s.free();
        }
        out.push("</code><span class=\"member-src\">");
        push_src_link(out, from, &it.file, m.line, "source");
        if (m.def_line > 0LL && (!m.def_file.eq(&it.file) || m.def_line != m.line)) {
            out.push(" · ");
            push_src_link(out, from, &m.def_file, m.def_line, "definition");
        }
        out.push("</span></div>");
        if (m.doc.len > 0UL) {
            struct String d = render_doc(c, &m.doc, m.md, from);
            out.push("<div class=\"docblock\">"); out.push_str(&d); out.push("</div>");
            d.free();
        }
        out.push("</section>");
        toc_entry(toc, id.as_ptr(), m.name.as_ptr(), 2);
        id.free();
    }
}

static int build_item_page(const &Crate c, unsigned long idx) {
    &Item it = c.items.at(idx);
    struct String from = it.url.clone();
    struct String content = String::new();
    struct String toc = String::new();
    &Module m = c.modules.at((unsigned long)it.module);
    // Breadcrumb: crate::module::Item
    content.push("<div class=\"item-head\"><p class=\"crumbs\"><a");
    push_href_cstr(&content, &from, "index.html");
    content.push(">");
    util_push_esc(&content, c.name.as_ptr());
    content.push("</a>");
    if (it.module > 0LL) {
        content.push(" :: <a");
        push_href(&content, &from, &m.url);
        content.push(">");
        struct String d = module_display(c, m);
        util_push_esc(&content, d.as_ptr());
        d.free();
        content.push("</a>");
    }
    content.push("</p><h1>");
    util_push_esc(&content, kind_title(it.kind.as_ptr()));
    content.push(" <span class=\"kind-");
    util_push_esc(&content, it.kind.as_ptr());
    content.push("\">");
    util_push_esc(&content, it.path.as_ptr());
    content.push("</span></h1><p class=\"item-meta\">");
    if (m.file.ends_with(".h")) { push_include(c, &m.file, &content); }
    push_src_link(&content, &from, &it.file, it.line, "source");
    if (it.def_line > 0LL && (!it.def_file.eq(&it.file) || it.def_line != it.line)) {
        content.push(" · ");
        push_src_link(&content, &from, &it.def_file, it.def_line, "definition");
    }
    content.push("</p></div>");
    declaration(c, it, &from, &content);
    if (it.doc.len > 0UL) {
        struct String d = render_doc(c, &it.doc, it.md, &from);
        content.push("<div class=\"docblock\">"); content.push_str(&d); content.push("</div>");
        d.free();
    }
    member_section(c, it, "field", "Fields", "fields", &from, &content, &toc);
    member_section(c, it, "variant", "Variants", "variants", &from, &content, &toc);
    member_section(c, it, "method", it.kind.eq_cstr("trait") ? "Required methods" : "Methods", "methods", &from, &content, &toc);
    struct String sidebar = String::new();
    build_sidebar(c, it.module, &from, &it.url, &sidebar);
    struct String title = it.path.clone();
    title.push(" - ");
    title.push_str(&c.name);
    title.push(" API");
    struct String desc = doc_summary_text(&it.doc, it.md, 160UL);
    if (desc.len == 0UL) { desc.push(kind_title(it.kind.as_ptr())); desc.push(" "); desc.push_str(&it.path); }
    int ok = write_page(c, &it.url, &title, &desc, &sidebar, &content, &toc, "api-item");
    from.free(); content.free(); toc.free(); sidebar.free(); title.free(); desc.free();
    return ok;
}

// ── all.html ─────────────────────────────────────────────────────────────────

static int build_all_page(const &Crate c) {
    struct String from = String::from("all.html");
    struct String content = String::from("<div class=\"item-head\"><h1>All items</h1></div>");
    struct String toc = String::new();
    struct Vec<long long> all = Vec::new();
    for (unsigned long i = 0UL; i < c.items.len; i = i + 1UL) { all.push((long long)i); }
    for (int k = 0; k < KIND_COUNT; k = k + 1) {
        const &char kind = KIND_ORDER(k);
        int n = 0;
        n = count_kind(c, &all, kind);
        if (n == 0) { continue; }
        content.push("<h2 id=\"");
        content.push(KIND_ANCHOR(k));
        content.push("\">");
        content.push(KIND_PLURAL(k)); toc_entry(&toc, KIND_ANCHOR(k), KIND_PLURAL(k), 1);
        content.push("</h2><ul class=\"all-items\">");
        for (unsigned long i = 0UL; i < c.items.len; i = i + 1UL) {
            &Item it = c.items.at(i);
            if (!it.kind.eq_cstr(kind)) { continue; }
            content.push("<li><a class=\"kind-");
            util_push_esc(&content, it.kind.as_ptr());
            content.push("\"");
            push_href(&content, &from, &it.url);
            content.push(">");
            util_push_esc(&content, it.path.as_ptr());
            content.push("</a> <span class=\"all-mod\">");
            struct String d = module_display(c, c.modules.at((unsigned long)it.module));
            util_push_esc(&content, d.as_ptr());
            d.free();
            content.push("</span></li>");
        }
        content.push("</ul>");
    }
    struct String sidebar = String::new();
    build_sidebar(c, 0LL, &from, &from, &sidebar);
    struct String title = String::from("All items - ");
    title.push_str(&c.name);
    title.push(" API");
    struct String desc = String::from("Every item in the ");
    desc.push_str(&c.name);
    desc.push(" API reference");
    int ok = write_page(c, &from, &title, &desc, &sidebar, &content, &toc, "api-module");
    from.free(); content.free(); toc.free(); all.free(); sidebar.free(); title.free(); desc.free();
    return ok;
}

// ── source pages ─────────────────────────────────────────────────────────────

static int build_source_page(const &Crate c, const &SourceFile sf) {
    struct String full = path_join(c.root.as_ptr(), sf.rel.as_ptr());
    struct String text = String::new();
    if (!File::read_to_string(full.as_ptr(), &text)) {
        eprint("safec-doc: cannot read ");
        eprintln(full.as_ptr());
        full.free(); text.free();
        return 0;
    }
    struct String content = String::from("<div class=\"item-head\"><h1>Source <span class=\"kind-module\">");
    util_push_esc(&content, sf.rel.as_ptr());
    content.push("</span></h1>");
    struct String u = String::from(util_vstr(&c.cfg, "source_url", ""));
    if (u.len > 0UL) {
        // "https://host/owner/repo/blob/<ref>/{file}"
        struct String prefix = String::from(util_vstr(&c.cfg, "source_prefix", ""));
        prefix.push_str(&sf.rel);
        u.replace_all("{file}", prefix.as_ptr());
        content.push("<p class=\"item-meta\"><a href=\"");
        util_push_esc(&content, u.as_ptr());
        content.push("\" rel=\"noreferrer\">View in the repository</a></p>");
        prefix.free();
    }
    u.free();
    unsigned long lines = 1UL;
    for (unsigned long i = 0UL; i < text.len; i = i + 1UL) {
        if (text.char_at(i) == (int)'\n' && i + 1UL < text.len) { lines = lines + 1UL; }
    }
    // Generated files (embedded kernels, tables) are not worth a page of
    // their own: just the line anchors links into them land on.
    if (text.len > 1048576UL) {
        content.push("</div><p class=\"too-large\">This file is ");
        content.push_uint(text.len / 1024UL);
        content.push(" KiB (generated); its source is not shown.</p><div class=\"line-anchors\">");
        for (unsigned long l = 1UL; l <= lines; l = l + 1UL) {
            content.push("<span id=\"L"); content.push_uint(l); content.push("\"></span>");
        }
        content.push("</div>");
        text.clear();
    } else {
    content.push("</div><div class=\"source-view\"><pre class=\"line-numbers\" aria-hidden=\"true\">");
    for (unsigned long l = 1UL; l <= lines; l = l + 1UL) {
        content.push("<a id=\"L"); content.push_uint(l); content.push("\" href=\"#L"); content.push_uint(l);
        content.push("\">"); content.push_uint(l); content.push("</a>\n");
    }
    content.push("</pre><div class=\"code-block\"><pre><code>");
    struct String lang = path_ext(sf.rel.as_ptr());
    if (lang.starts_with(".")) { lang.remove_range(0UL, 1UL); }
    highlight_code(lang.as_ptr(), text.as_ptr(), text.len, &content);
    content.push("</code></pre></div></div>");
    lang.free();
    }
    struct String title = sf.rel.clone();
    title.push(" - source - ");
    title.push_str(&c.name);
    struct String desc = String::from("Source of ");
    desc.push_str(&sf.rel);
    struct String sidebar = String::new();
    build_sidebar(c, 0LL, &sf.url, &sf.url, &sidebar);
    struct String toc = String::new();
    int ok = write_page(c, &sf.url, &title, &desc, &sidebar, &content, &toc, "api-source");
    full.free(); text.free(); content.free(); title.free(); desc.free(); sidebar.free(); toc.free();
    return ok;
}

// ── search index ─────────────────────────────────────────────────────────────

// [{ "n": name, "p": path, "k": kind, "u": url, "d": summary }, ...] —
// items, then their members ("Vec::push").
static int build_search_index(const &Crate c) {
    struct Value idx = Value::array();
    for (unsigned long i = 0UL; i < c.modules.len; i = i + 1UL) {
        &Module m = c.modules.at(i);
        if (i == 0UL) { continue; }
        struct Value r = Value::object();
        struct String d = module_display(c, m);
        r.object_set("n", Value::from_str(m.name.as_ptr()));
        r.object_set("p", Value::from_str(d.as_ptr()));
        r.object_set("k", Value::from_str("module"));
        r.object_set("u", Value::from_str(m.url.as_ptr()));
        struct String s = doc_summary_text(&m.doc, m.md, 120UL);
        r.object_set("d", Value::from_str(s.as_ptr()));
        idx.array_push(r);
        d.free(); s.free();
    }
    for (unsigned long i = 0UL; i < c.items.len; i = i + 1UL) {
        &Item it = c.items.at(i);
        struct Value r = Value::object();
        r.object_set("n", Value::from_str(it.name.as_ptr()));
        r.object_set("p", Value::from_str(it.path.as_ptr()));
        r.object_set("k", Value::from_str(it.kind.as_ptr()));
        r.object_set("u", Value::from_str(it.url.as_ptr()));
        struct String s = doc_summary_text(&it.doc, it.md, 120UL);
        r.object_set("d", Value::from_str(s.as_ptr()));
        idx.array_push(r);
        s.free();
        for (unsigned long k = 0UL; k < it.members.len; k = k + 1UL) {
            &Member m = it.members.at(k);
            struct Value mr = Value::object();
            struct String p = it.path.clone();
            p.push("::");
            p.push_str(&m.name);
            struct String u = it.url.clone();
            u.push("#");
            u.push_str(&m.kind);
            u.push(".");
            u.push_str(&m.name);
            mr.object_set("n", Value::from_str(m.name.as_ptr()));
            mr.object_set("p", Value::from_str(p.as_ptr()));
            mr.object_set("k", Value::from_str(m.kind.as_ptr()));
            mr.object_set("u", Value::from_str(u.as_ptr()));
            struct String ms = doc_summary_text(&m.doc, m.md, 100UL);
            mr.object_set("d", Value::from_str(ms.as_ptr()));
            idx.array_push(mr);
            p.free(); u.free(); ms.free();
        }
    }
    struct String json = value_to_json(&idx);
    struct String path = path_join(c.out.as_ptr(), "search-index.json");
    int ok = util_write_file(path.as_ptr(), &json);
    json.free(); path.free(); idx.free();
    return ok;
}

// ── everything ───────────────────────────────────────────────────────────────

int pages_build(&Crate c) {
    int ok = 1;
    dir_remove_all(c.out.as_ptr());
    dir_create_all(c.out.as_ptr());
    struct String assets = path_join(c.out.as_ptr(), "assets");
    if (!dir_copy_tree(util_vstr(&c.cfg, "assets", "theme"), assets.as_ptr())) {
        eprint("safec-doc: cannot copy the assets from ");
        eprintln(util_vstr(&c.cfg, "assets", "theme"));
        ok = 0;
    }
    assets.free();
    for (unsigned long i = 0UL; i < c.modules.len; i = i + 1UL) {
        if (!build_module_page(c, (long long)i)) { ok = 0; }
    }
    for (unsigned long i = 0UL; i < c.items.len; i = i + 1UL) {
        if (!build_item_page(c, i)) { ok = 0; }
    }
    if (!build_all_page(c)) { ok = 0; }
    for (unsigned long i = 0UL; i < c.sources.len; i = i + 1UL) {
        if (!build_source_page(c, c.sources.at(i))) { ok = 0; }
    }
    if (!build_search_index(c)) { ok = 0; }
    // .nojekyll: GitHub Pages must not drop files starting with '_'.
    struct String nj = path_join(c.out.as_ptr(), ".nojekyll");
    struct String empty = String::new();
    util_write_file(nj.as_ptr(), &empty);
    nj.free(); empty.free();
    return ok;
}
