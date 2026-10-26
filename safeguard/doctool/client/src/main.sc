// safec-doc client — the API reference's in-browser behavior, a wasm32
// SafeC module on std/wasm/dom (loaded by theme/doc.js through std's JS
// glue, std/wasm/safec_dom.js). Adapted from the SafeC docs site's client
// (safec-docs/site/client). Build:
//
//   cd safeguard/doctool/client && safeguard build --release --target wasm32
//
// Every page is complete server-rendered HTML that works without it; the
// client adds:
//   - theme toggle (html.dark), remembered in localStorage;
//   - the mobile sidebar (body.menu-open);
//   - copy buttons on code blocks;
//   - outline scroll-spy;
//   - search: '/' or Ctrl/Cmd+K or the header button opens a dialog; the
//     reference's search-index.json (one record per module, item and
//     member) is fetched once, then every keystroke ranks it here, in
//     SafeC — exact names first, then name prefixes, then paths and
//     summaries; arrows + Enter navigate.
#include <std/wasm/wasm_rt.h>
#include <std/wasm/dom.h>
#include <std/serial/json.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

// Provided by theme/site.js (std/wasm has no clipboard binding).
extern void doc_clipboard_write(const char* text, int len);

// ── small helpers ────────────────────────────────────────────────────────────

static void esc(&String out, const &char s, unsigned long n) {
    for (unsigned long i = 0UL; i < n; i = i + 1UL) {
        char c;
        unsafe { c = s[i]; }
        if (c == '&') { out.push("&amp;"); }
        else if (c == '<') { out.push("&lt;"); }
        else if (c == '>') { out.push("&gt;"); }
        else if (c == '"') { out.push("&quot;"); }
        else { out.push_char(c); }
    }
}

static char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

// Byte index of 'needle' in 'hay' ignoring ASCII case, from 'from'; -1 if none.
static long long find_ci(const &String hay, const &String needle, unsigned long from) {
    if (needle.len == 0UL || needle.len > hay.len) { return -1LL; }
    for (unsigned long i = from; i + needle.len <= hay.len; i = i + 1UL) {
        unsigned long k = 0UL;
        while (k < needle.len && lower((char)hay.char_at(i + k)) == (char)needle.char_at(k)) { k = k + 1UL; }
        if (k == needle.len) { return (long long)i; }
    }
    return -1LL;
}

// Walks up from 'h' (up to 5 levels) to the first element with class
// 'cls'; its handle, or DOM_NONE. Releases the handles it passes.
static int up_to_class(int h, const &char cls) {
    int cur = h;
    for (int depth = 0; depth < 6 && cur >= 0; depth = depth + 1) {
        if (dom_class_contains(cur, cls)) { return cur; }
        int p = dom_parent(cur);
        if (cur != h) { dom_release(cur); }
        cur = p;
    }
    if (cur >= 0 && cur != h) { dom_release(cur); }
    return DOM_NONE;
}

// The data-action of 'h' or its nearest ancestor that has one ("" if none).
static struct String action_of(int h) {
    int cur = h;
    for (int depth = 0; depth < 6 && cur >= 0; depth = depth + 1) {
        if (dom_has_attr(cur, "data-action")) {
            struct String a = dom_get_attr(cur, "data-action");
            if (cur != h) { dom_release(cur); }
            return a;
        }
        int p = dom_parent(cur);
        if (cur != h) { dom_release(cur); }
        cur = p;
    }
    if (cur >= 0 && cur != h) { dom_release(cur); }
    return String::new();
}

// ── state ────────────────────────────────────────────────────────────────────

struct SearchRec {
    struct String url;     // relative to the doc root
    struct String name;    // "push"
    struct String path;    // "std::Vec::push"
    struct String kind;    // "method"
    struct String text;    // summary
};

struct Client {
    int html;
    int body;
    int dialog;
    int input;
    int results;
    int empty;
    int search_open;
    int index_state;          // 0 not loaded, 1 loading, 2 ready, 3 failed
    int selected;
    int result_count;
    struct Vec<struct SearchRec> recs;
    struct String base;       // the doc root, relative to this page ("../../")
    struct String empty_label;
    struct String copied_label;
    // scroll-spy
    struct Vec<int> toc_links;
    struct Vec<int> toc_targets;
    int active_toc;
    int frame_pending;
};

static struct Client gC;

// ── theme, menu, nav menus ───────────────────────────────────────────────────

static void toggle_theme(&Client c) {
    int dark = dom_class_toggle(c.html, "dark");
    dom_storage_set(DOM_STORAGE_LOCAL, "safec-theme", dark ? "dark" : "light");
}

static void set_menu(&Client c, int open) {
    if (open) { dom_class_add(c.body, "menu-open"); } else { dom_class_remove(c.body, "menu-open"); }
    int btn = dom_query(".menu-toggle");
    if (btn >= 0) { dom_set_attr(btn, "aria-expanded", open ? "true" : "false"); dom_release(btn); }
}

static void close_nav_menus(int except) {
    struct Vec<int> groups = Vec::new();
    dom_query_all(".nav-group.open", &groups);
    for (unsigned long i = 0UL; i < groups.len; i = i + 1UL) {
        int g = *groups.at(i);
        if (g != except) {
            dom_class_remove(g, "open");
            int b = dom_query_in(g, ".nav-group-btn");
            if (b >= 0) { dom_set_attr(b, "aria-expanded", "false"); dom_release(b); }
        }
        dom_release(g);
    }
    groups.free();
}

// ── copy buttons ─────────────────────────────────────────────────────────────

static int gCopiedBtn = -1;

static void uncopy(?&Client c) {
    if (gCopiedBtn >= 0) {
        dom_class_remove(gCopiedBtn, "copied");
        dom_release(gCopiedBtn);
        gCopiedBtn = -1;
    }
}

static void copy_code(&Client c, int btn) {
    int block = dom_parent(btn);
    if (block < 0) { return; }
    int code = dom_query_in(block, "code");
    if (code >= 0) {
        struct String text = dom_get_text(code);
        unsafe { doc_clipboard_write(text.as_ptr(), (int)text.len); }
        text.free();
        dom_release(code);
        uncopy((?&Client)null);
        dom_set_attr(btn, "data-label", c.copied_label.as_ptr());
        dom_class_add(btn, "copied");
        gCopiedBtn = btn;
        unsafe { dom_set_timeout(uncopy, (?&Client)&gC, 2000); }
    }
    dom_release(block);
}

// ── scroll-spy ───────────────────────────────────────────────────────────────

static void spy_frame(?&Client cp, double ts) {
    match (cp) {
        case some(c): {
            c.frame_pending = 0;
            double y = dom_get_prop_num(DOM_WINDOW, "scrollY") + 96.0;
            double bottom = dom_get_prop_num(DOM_WINDOW, "innerHeight") + y - 96.0;
            double height = dom_get_prop_num(c.body, "scrollHeight");
            int active = -1;
            for (unsigned long i = 0UL; i < c.toc_targets.len; i = i + 1UL) {
                int t = *c.toc_targets.at(i);
                if (t >= 0 && dom_get_prop_num(t, "offsetTop") <= y) { active = (int)i; }
            }
            // At the very bottom, the last section is the current one.
            if (bottom >= height - 2.0 && c.toc_targets.len > 0UL) { active = (int)c.toc_targets.len - 1; }
            if (active != c.active_toc) {
                if (c.active_toc >= 0) { dom_class_remove(*c.toc_links.at((unsigned long)c.active_toc), "active"); }
                if (active >= 0) { dom_class_add(*c.toc_links.at((unsigned long)active), "active"); }
                c.active_toc = active;
            }
        }
        case null: { }
    }
}

static void on_scroll(?&Client cp, &DomEvent ev) {
    match (cp) {
        case some(c): {
            if (!c.frame_pending) {
                c.frame_pending = 1;
                unsafe { dom_request_animation_frame(spy_frame, cp); }
            }
        }
        case null: { }
    }
}

static void setup_spy(&Client c) {
    dom_query_all(".toc a", &c.toc_links);
    for (unsigned long i = 0UL; i < c.toc_links.len; i = i + 1UL) {
        struct String href = dom_get_attr(*c.toc_links.at(i), "href");
        struct String id = href.strip_prefix("#");
        c.toc_targets.push(dom_get_by_id(id.as_ptr()));
        href.free(); id.free();
    }
    if (c.toc_links.len > 0UL) {
        unsafe { dom_on(DOM_WINDOW, "scroll", on_scroll, (?&Client)&gC); }
        spy_frame((?&Client)&gC, 0.0);
    }
}

// ── search ───────────────────────────────────────────────────────────────────

static void search_run(&Client c);

static void on_index(?&Client cp, &DomFetchResponse r) {
    match (cp) {
        case some(c): {
            if (!r.ok) { c.index_state = 3; return; }
            struct String body = r.text();
            int ok = 0;
            struct Value v = json_parse(body.as_ptr(), &ok);
            body.free();
            if (!ok || v.kind_of() != VAL_ARRAY) { v.free(); c.index_state = 3; return; }
            for (unsigned long i = 0UL; i < v.array_len(); i = i + 1UL) {
                &Value e = v.array_at(i);
                struct SearchRec rec;
                rec.url = c.base.clone(); rec.name = String::new(); rec.path = String::new();
                rec.kind = String::new(); rec.text = String::new();
                match (e.object_get("u")) { case some(x): { rec.url.push(x.as_string()); } case null: { } }
                match (e.object_get("n")) { case some(x): { rec.name.push(x.as_string()); } case null: { } }
                match (e.object_get("p")) { case some(x): { rec.path.push(x.as_string()); } case null: { } }
                match (e.object_get("k")) { case some(x): { rec.kind.push(x.as_string()); } case null: { } }
                match (e.object_get("d")) { case some(x): { rec.text.push(x.as_string()); } case null: { } }
                c.recs.push(rec);
            }
            v.free();
            c.index_state = 2;
            search_run(c);
        }
        case null: { }
    }
}

static void load_index(&Client c) {
    if (c.index_state != 0) { return; }
    c.index_state = 1;
    struct String url = c.base.clone();
    url.push("search-index.json");
    unsafe { dom_fetch_get(url.as_ptr(), on_index, (?&Client)&gC); }
    url.free();
}

// Appends s[0..n) escaped, with every occurrence of a query token wrapped in <mark>.
static void push_marked(&String out, const &String s, const &Vec<struct String> toks) {
    unsigned long i = 0UL;
    while (i < s.len) {
        unsigned long best = 0UL;
        for (unsigned long t = 0UL; t < toks.len; t = t + 1UL) {
            &String tok = toks.at(t);
            if (tok.len > best && i + tok.len <= s.len) {
                unsigned long k = 0UL;
                while (k < tok.len && lower((char)s.char_at(i + k)) == (char)tok.char_at(k)) { k = k + 1UL; }
                if (k == tok.len) { best = tok.len; }
            }
        }
        if (best > 0UL) {
            out.push("<mark>");
            unsafe { esc(out, (const char*)s.as_ptr() + i, best); }
            out.push("</mark>");
            i = i + best;
        } else {
            unsafe { esc(out, (const char*)s.as_ptr() + i, 1UL); }
            i = i + 1UL;
        }
    }
}

// A ~180-byte excerpt of 'text' around byte 'at', on UTF-8 boundaries.
static struct String excerpt(const &String text, long long at) {
    unsigned long start = at > 60LL ? (unsigned long)at - 60UL : 0UL;
    unsigned long end = start + 180UL;
    if (end > text.len) { end = text.len; }
    while (start > 0UL && (text.char_at(start) & 0xC0) == 0x80) { start = start - 1UL; }
    while (end < text.len && (text.char_at(end) & 0xC0) == 0x80) { end = end + 1UL; }
    struct String s = String::new();
    if (start > 0UL) { s.push("… "); }
    struct String mid = text.substr(start, end);
    s.push_str(&mid);
    mid.free();
    if (end < text.len) { s.push(" …"); }
    return s;
}

#define SEARCH_MAX 30

static void search_render(&Client c, const &String query) {
    // Lower-cased tokens.
    struct Vec<struct String> toks = Vec::new();
    struct String cur = String::new();
    for (unsigned long i = 0UL; i <= query.len; i = i + 1UL) {
        char ch = i < query.len ? (char)query.char_at(i) : ' ';
        if (ch == ' ' || ch == '\t') {
            if (cur.len > 0UL && toks.len < 8UL) { toks.push(cur.clone()); }
            cur.clear();
        } else { cur.push_char(lower(ch)); }
    }
    cur.free();
    // Rank: every token must occur in the path or the summary; the name
    // matching the query counts most (exact, then prefix, then substring).
    int best_idx[SEARCH_MAX];
    int best_score[SEARCH_MAX];
    int nbest = 0;
    struct String q = query.to_lower();
    struct String q0 = q.trim();
    q.free();
    q = q0;
    for (unsigned long r = 0UL; r < c.recs.len && toks.len > 0UL; r = r + 1UL) {
        &SearchRec rec = c.recs.at(r);
        int score = 0;
        int all = 1;
        for (unsigned long t = 0UL; t < toks.len; t = t + 1UL) {
            &String tok = toks.at(t);
            long long ip = find_ci(&rec.path, tok, 0UL);
            long long ix = find_ci(&rec.text, tok, 0UL);
            if (ip < 0LL && ix < 0LL) { all = 0; break; }
            if (ip >= 0LL) { score = score + 10; }
            if (ix >= 0LL) { score = score + 1; }
        }
        if (!all) { continue; }
        long long in_name = find_ci(&rec.name, &q, 0UL);
        if (in_name == 0LL && rec.name.len == q.len) { score = score + 100; }
        else if (in_name == 0LL) { score = score + 50; }
        else if (in_name > 0LL) { score = score + 20; }
        if (find_ci(&rec.path, &q, 0UL) >= 0LL) { score = score + 15; }
        // Items before members, shorter paths first among equals.
        if (!rec.kind.eq_cstr("method") && !rec.kind.eq_cstr("field") && !rec.kind.eq_cstr("variant")) { score = score + 5; }
        score = score * 64 - (int)(rec.path.len > 63UL ? 63UL : rec.path.len);
        // Insert into the top list (sorted by score, stable).
        int pos = nbest;
        while (pos > 0 && best_score[pos - 1] < score) { pos = pos - 1; }
        if (pos >= SEARCH_MAX) { continue; }
        int last = nbest < SEARCH_MAX ? nbest : SEARCH_MAX - 1;
        for (int k = last; k > pos; k = k - 1) { best_idx[k] = best_idx[k - 1]; best_score[k] = best_score[k - 1]; }
        best_idx[pos] = (int)r;
        best_score[pos] = score;
        if (nbest < SEARCH_MAX) { nbest = nbest + 1; }
    }
    struct String html = String::new();
    for (int k = 0; k < nbest; k = k + 1) {
        &SearchRec rec = c.recs.at((unsigned long)best_idx[k]);
        html.push("<li><a href=\"");
        unsafe { esc(&html, rec.url.as_ptr(), rec.url.len); }
        html.push(k == 0 ? "\" class=\"selected\">" : "\">");
        html.push("<span class=\"crumb kind-");
        unsafe { esc(&html, rec.kind.as_ptr(), rec.kind.len); }
        html.push("\">");
        unsafe { esc(&html, rec.kind.as_ptr(), rec.kind.len); }
        html.push("</span><span class=\"title\">");
        push_marked(&html, &rec.path, &toks);
        html.push("</span>");
        if (rec.text.len > 0UL) {
            long long at = find_ci(&rec.text, toks.at(0UL), 0UL);
            struct String ex = excerpt(&rec.text, at < 0LL ? 0LL : at);
            html.push("<span class=\"excerpt\">");
            push_marked(&html, &ex, &toks);
            html.push("</span>");
            ex.free();
        }
        html.push("</a></li>");
    }
    dom_set_html(c.results, html.as_ptr());
    c.selected = 0;
    c.result_count = nbest;
    if (nbest == 0 && toks.len > 0UL && c.index_state == 2) {
        struct String msg = c.empty_label.clone();
        msg.push(" \"");
        msg.push_str(query);
        msg.push("\"");
        dom_set_text(c.empty, msg.as_ptr());
        dom_set_visible(c.empty, 1);
        msg.free();
    } else {
        dom_set_visible(c.empty, 0);
    }
    html.free(); q.free();
    for (unsigned long t = 0UL; t < toks.len; t = t + 1UL) { toks.at(t).free(); }
    toks.free();
}

static void search_run(&Client c) {
    if (!c.search_open) { return; }
    struct String q = dom_get_value(c.input);
    search_render(c, &q);
    q.free();
}

static void open_search(&Client c) {
    if (c.search_open) { return; }
    c.search_open = 1;
    dom_remove_attr(c.dialog, "hidden");
    dom_focus(c.input);
    load_index(c);
    search_run(c);
}

static void close_search(&Client c) {
    if (!c.search_open) { return; }
    c.search_open = 0;
    dom_set_attr(c.dialog, "hidden", "");
}

static void move_selection(&Client c, int delta) {
    if (c.result_count == 0) { return; }
    struct Vec<int> links = Vec::new();
    dom_query_all_in(c.results, "a", &links);
    if (links.len == 0UL) { links.free(); return; }
    int n = (int)links.len;
    dom_class_remove(*links.at((unsigned long)c.selected), "selected");
    c.selected = (c.selected + delta + n) % n;
    int sel = *links.at((unsigned long)c.selected);
    dom_class_add(sel, "selected");
    dom_call(sel, "scrollIntoView");
    for (unsigned long i = 0UL; i < links.len; i = i + 1UL) { dom_release(*links.at(i)); }
    links.free();
}

static void open_selected(&Client c) {
    int sel = dom_query_in(c.results, "a.selected");
    if (sel < 0) { return; }
    struct String href = dom_get_attr(sel, "href");
    dom_release(sel);
    close_search(c);
    dom_navigate(href.as_ptr());
    href.free();
}

static void on_input(?&Client cp, &DomEvent ev) {
    match (cp) { case some(c): { search_run(c); } case null: { } }
}

// ── global listeners ─────────────────────────────────────────────────────────

static void on_click(?&Client cp, &DomEvent ev) {
    match (cp) {
        case some(c): {
            int t = ev.target();
            if (t < 0) { return; }
            if (c.search_open && t == c.dialog) { close_search(c); dom_release(t); return; }
            if (dom_class_contains(t, "copy")) { copy_code(c, t); return; }   // keeps 't' (uncopy releases it)
            struct String action = action_of(t);
            if (action.eq_cstr("theme")) { toggle_theme(c); }
            else if (action.eq_cstr("menu")) { set_menu(c, !dom_class_contains(c.body, "menu-open")); }
            else if (action.eq_cstr("search")) { open_search(c); }
            action.free();
            int btn = up_to_class(t, "nav-group-btn");
            if (btn >= 0) {
                int group = dom_parent(btn);
                close_nav_menus(group);
                int open = dom_class_toggle(group, "open");
                dom_set_attr(btn, "aria-expanded", open ? "true" : "false");
                dom_release(group);
                if (btn != t) { dom_release(btn); }
            } else {
                int group = up_to_class(t, "nav-group");
                if (group < 0) { close_nav_menus(DOM_NONE); }
                else if (group != t) { dom_release(group); }
            }
            if (c.search_open) {
                int list = up_to_class(t, "search-results");
                if (list >= 0) {
                    close_search(c);   // a result link is being followed
                    if (list != t) { dom_release(list); }
                }
            }
            dom_release(t);
        }
        case null: { }
    }
}

static void on_keydown(?&Client cp, &DomEvent ev) {
    match (cp) {
        case some(c): {
            struct String key = ev.key();
            if (c.search_open) {
                if (key.eq_cstr("Escape")) { close_search(c); }
                else if (key.eq_cstr("ArrowDown")) { ev.prevent_default(); move_selection(c, 1); }
                else if (key.eq_cstr("ArrowUp")) { ev.prevent_default(); move_selection(c, -1); }
                else if (key.eq_cstr("Enter")) { ev.prevent_default(); open_selected(c); }
            } else {
                int typing = 0;
                int t = ev.target();
                if (t >= 0) {
                    struct String tag = dom_get_prop(t, "tagName");
                    typing = tag.eq_cstr("INPUT") || tag.eq_cstr("TEXTAREA");
                    tag.free();
                    dom_release(t);
                }
                int k_combo = (ev.ctrl_key() || ev.meta_key()) && (key.eq_cstr("k") || key.eq_cstr("K"));
                if (k_combo || (key.eq_cstr("/") && !typing)) {
                    ev.prevent_default();
                    open_search(c);
                } else if (key.eq_cstr("Escape")) {
                    close_nav_menus(DOM_NONE);
                    set_menu(c, 0);
                }
            }
            key.free();
        }
        case null: { }
    }
}

// ── entry point ──────────────────────────────────────────────────────────────

__attribute__((export_name("doc_init"))) int doc_init() {
    gC.html = dom_query("html");
    gC.body = dom_body();
    gC.dialog = dom_get_by_id("search");
    gC.input = dom_query(".search-input");
    gC.results = dom_query(".search-results");
    gC.empty = dom_query(".search-empty");
    gC.search_open = 0;
    gC.index_state = 0;
    gC.selected = 0;
    gC.result_count = 0;
    gC.recs = Vec::new();
    gC.base = dom_get_attr(gC.body, "data-root");
    gC.empty_label = dom_get_attr(gC.dialog, "data-empty");
    gC.copied_label = dom_get_attr(gC.body, "data-copied");
    gC.toc_links = Vec::new();
    gC.toc_targets = Vec::new();
    gC.active_toc = -1;
    gC.frame_pending = 0;
    unsafe {
        dom_on(DOM_DOCUMENT, "click", on_click, (?&Client)&gC);
        dom_on(DOM_DOCUMENT, "keydown", on_keydown, (?&Client)&gC);
        if (gC.input >= 0) { dom_on(gC.input, "input", on_input, (?&Client)&gC); }
    }
    setup_spy(&gC);
    return 0;
}
