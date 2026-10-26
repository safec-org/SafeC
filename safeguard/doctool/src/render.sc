// safec-doc — doc comments and signatures -> HTML (see doctool.h), on
// std/serial/markdown (CommonMark + GFM tables, strikethrough, autolinks;
// raw HTML stays escaped) through a render hook that
//   - resolves intra-doc links: '[Vec::push]', '[`String`]', '[vec_new()]'
//     (shorthand, rewritten before parsing) and '[text](Vec::push)';
//   - highlights code blocks (SafeC when the fence names no language);
//   - shifts headings below the page's own (# -> <h3>).
// Plain '//' comments (std's convention) are read as Markdown too, with
// their indented runs (after a blank line or a line ending in ':') taken
// as code blocks.
#include "doctool.h"
#include <std/serial/markdown.h>
#include <std/serial/html.h>

// ── links ────────────────────────────────────────────────────────────────────

struct String rel_href(const &String from, const &String target) {
    if (target.contains("://") || target.starts_with("#") || target.starts_with("mailto:")) {
        return target.clone();
    }
    // The shortest relative path: drop the directories 'from' and the
    // target share, then one "../" per remaining directory of 'from'.
    unsigned long common = 0UL;
    unsigned long i = 0UL;
    while (i < from.len && i < target.len && from.char_at(i) == target.char_at(i)) {
        if (from.char_at(i) == (int)'/') { common = i + 1UL; }
        i = i + 1UL;
    }
    struct String rest_from = from.substr(common, from.len);
    struct String r = util_root_prefix(&rest_from);
    rest_from.free();
    struct String rest = target.substr(common, target.len);
    r.push_str(&rest);
    rest.free();
    // Same page: just the fragment.
    long long hash = target.find_char('#');
    if (hash >= 0LL) {
        struct String page = target.substr(0UL, (unsigned long)hash);
        if (page.eq(from)) {
            r.free();
            r = target.substr((unsigned long)hash, target.len);
        }
        page.free();
    }
    return r;
}

// "`Vec::push()`" -> "Vec::push".
static struct String link_key(const &String label) {
    struct String k = String::new();
    for (unsigned long i = 0UL; i < label.len; i = i + 1UL) {
        int ch = label.char_at(i);
        if (ch != (int)'`') { k.push_char((char)ch); }
    }
    struct String t = k.trim();
    k.free();
    if (t.ends_with("()")) { t.truncate(t.len - 2UL); }
    struct String r = t.strip_prefix("struct ");
    t.free();
    return r;
}

// The link for a key, relative to page 'from' (empty if it doesn't resolve).
static struct String resolve_href(const &Crate c, const &String key, const &String from) {
    long long idx = crate_resolve(c, key.as_ptr());
    if (idx < 0LL) { return String::new(); }
    return rel_href(from, c.link_urls.at((unsigned long)idx));
}

// ── Markdown preparation ─────────────────────────────────────────────────────

static unsigned long indent_of(const &String line) {
    unsigned long n = 0UL;
    while (n < line.len && line.char_at(n) == (int)' ') { n = n + 1UL; }
    return n;
}

static int is_blank(const &String line) { return indent_of(line) == line.len; }

static int is_fence(const &String line) {
    struct String t = line.trim();
    int f = t.starts_with("```") || t.starts_with("~~~");
    t.free();
    return f;
}

// Plain comments -> Markdown: indented runs become ```c blocks.
static struct String plain_to_md(const &String text) {
    struct Vec<struct String> lines = Vec::new();
    unsigned long start = 0UL;
    for (unsigned long i = 0UL; i <= text.len; i = i + 1UL) {
        if (i == text.len || text.char_at(i) == (int)'\n') {
            lines.push(text.substr(start, i));
            start = i + 1UL;
        }
    }
    struct String out = String::new();
    int in_code = 0;
    int in_fence = 0;
    unsigned long base = 0UL;
    int prev_blank = 1;
    int prev_colon = 0;
    for (unsigned long i = 0UL; i < lines.len; i = i + 1UL) {
        &String line = lines.at(i);
        if (is_fence(line)) { in_fence = !in_fence; }
        if (in_fence || is_fence(line)) {
            out.push_str(line); out.push_char('\n');
            prev_blank = 0; prev_colon = 0;
            continue;
        }
        int blank = is_blank(line);
        unsigned long ind = indent_of(line);
        if (in_code) {
            if (blank) {
                // A blank line inside the run, if more indented lines follow.
                unsigned long j = i + 1UL;
                while (j < lines.len && is_blank(lines.at(j))) { j = j + 1UL; }
                if (j < lines.len && indent_of(lines.at(j)) >= 2UL) { out.push_char('\n'); continue; }
                out.push("```\n\n");
                in_code = 0;
                prev_blank = 1;
                continue;
            }
            if (ind >= 2UL) {
                struct String rest = line.substr(ind < base ? ind : base, line.len);
                out.push_str(&rest); out.push_char('\n');
                rest.free();
                continue;
            }
            out.push("```\n");
            in_code = 0;
        }
        if (!blank && ind >= 2UL && (prev_blank || prev_colon)) {
            if (!prev_blank) { out.push_char('\n'); }
            out.push("```c\n");
            in_code = 1;
            base = ind;
            struct String rest = line.substr(ind, line.len);
            out.push_str(&rest); out.push_char('\n');
            rest.free();
            continue;
        }
        out.push_str(line); out.push_char('\n');
        prev_blank = blank;
        prev_colon = line.ends_with(":");
    }
    if (in_code) { out.push("```\n"); }
    lines.free();   // and each line
    return out;
}

// '[Name]' shorthand links -> '[Name](href)' when Name resolves; code
// spans and fenced blocks are left alone.
static struct String rewrite_shorthand(const &Crate c, const &String md, const &String from) {
    struct String out = String::new();
    unsigned long i = 0UL;
    int line_start = 1;
    int in_fence = 0;
    while (i < md.len) {
        int ch = md.char_at(i);
        if (line_start) {
            unsigned long j = i;
            while (j < md.len && md.char_at(j) == (int)' ') { j = j + 1UL; }
            if (j + 2UL < md.len && ((md.char_at(j) == (int)'`' && md.char_at(j + 1UL) == (int)'`' && md.char_at(j + 2UL) == (int)'`') ||
                                     (md.char_at(j) == (int)'~' && md.char_at(j + 1UL) == (int)'~' && md.char_at(j + 2UL) == (int)'~'))) {
                in_fence = !in_fence;
            }
        }
        line_start = ch == (int)'\n';
        if (in_fence) { out.push_char((char)ch); i = i + 1UL; continue; }
        if (ch == (int)'`') {
            // A code span: copy through its closing run of the same length.
            unsigned long n = 0UL;
            while (i + n < md.len && md.char_at(i + n) == (int)'`') { n = n + 1UL; }
            unsigned long k = i + n;
            unsigned long close = md.len;
            while (k < md.len) {
                if (md.char_at(k) == (int)'`') {
                    unsigned long m = 0UL;
                    while (k + m < md.len && md.char_at(k + m) == (int)'`') { m = m + 1UL; }
                    if (m == n) { close = k + m; break; }
                    k = k + m;
                } else { k = k + 1UL; }
            }
            if (close == md.len) { close = i + n; }   // unmatched: literal backticks
            struct String span = md.substr(i, close);
            out.push_str(&span);
            span.free();
            i = close;
            continue;
        }
        if (ch == (int)'[' && (i == 0UL || (md.char_at(i - 1UL) != (int)'!' && md.char_at(i - 1UL) != (int)']' && md.char_at(i - 1UL) != (int)'\\'))) {
            unsigned long k = i + 1UL;
            while (k < md.len && md.char_at(k) != (int)']' && md.char_at(k) != (int)'[' && md.char_at(k) != (int)'\n') { k = k + 1UL; }
            int follows = k + 1UL < md.len ? md.char_at(k + 1UL) : 0;
            if (k < md.len && md.char_at(k) == (int)']' && k > i + 1UL &&
                follows != (int)'(' && follows != (int)'[' && follows != (int)':') {
                struct String label = md.substr(i + 1UL, k);
                struct String key = link_key(&label);
                struct String href = resolve_href(c, &key, from);
                if (href.len > 0UL) {
                    out.push("["); out.push_str(&label); out.push("](");
                    out.push_str(&href); out.push(")");
                    i = k + 1UL;
                    label.free(); key.free(); href.free();
                    continue;
                }
                label.free(); key.free(); href.free();
            }
        }
        out.push_char((char)ch);
        i = i + 1UL;
    }
    return out;
}

static struct String prepare(const &Crate c, const &String text, int md, const &String from) {
    struct String base = md ? text.clone() : plain_to_md(text);
    struct String r = rewrite_shorthand(c, &base, from);
    base.free();
    return r;
}

// ── the render hook ──────────────────────────────────────────────────────────

struct RenderCtx {
    const struct Crate* crate;
    const struct String* from;
};

static int render_hook(?&RenderCtx ctx, const &MdNode n, int entering, &String out) {
    const struct Crate* cp;
    const struct String* fp;
    match (ctx) {
        case some(x): { unsafe { cp = x.crate; fp = x.from; } }
        case null: { return 0; }
    }
    if (n.kind == MD_HEADING) {
        int level = n.level + 2;
        if (level > 6) { level = 6; }
        out.push(entering ? "<h" : "</h");
        out.push_int((long long)level);
        out.push(entering ? " class=\"doc-heading\">" : ">\n");
        return entering ? 2 : 1;
    }
    if (n.kind == MD_LINK && entering) {
        struct String href = String::new();
        // '[text](Vec::push)': an item path as the target.
        if (!n.text.contains("/") && !n.text.contains("://") && !n.text.starts_with("#") && !n.text.contains(".html")) {
            struct String key = link_key(&n.text);
            unsafe { href = resolve_href((const &Crate)cp, &key, (const &String)fp); }
            key.free();
        }
        if (href.len == 0UL) { href.push_str(&n.text); }
        out.push("<a href=\"");
        util_push_esc(out, href.as_ptr());
        out.push("\"");
        if (n.title.len > 0UL) { out.push(" title=\""); util_push_esc(out, n.title.as_ptr()); out.push("\""); }
        if (href.starts_with("http://") || href.starts_with("https://")) {
            out.push(" rel=\"noreferrer\"");
        }
        out.push(">");
        href.free();
        return 2;
    }
    if (n.kind == MD_CODE_BLOCK && entering) {
        long long sp = n.info.find_char(' ');
        struct String lang = sp >= 0LL ? n.info.substr(0UL, (unsigned long)sp) : n.info.clone();
        if (lang.len == 0UL) { lang.push("c"); }
        out.push("<div class=\"language-");
        util_push_esc(out, lang.as_ptr());
        out.push(" code-block\"><button class=\"copy\" type=\"button\" aria-label=\"Copy code\"></button><pre><code>");
        highlight_code(lang.as_ptr(), n.text.as_ptr(), n.text.len, out);
        out.push("</code></pre></div>\n");
        lang.free();
        return 1;
    }
    return 0;
}

static struct String render_md(const &Crate c, const &String md_text, const &String from) {
    struct MdOptions o = MdOptions::gfm();
    o.html = 0;
    &heap MdNode doc = MdNode::parse_opts(md_text.as_ptr(), &o);
    struct RenderCtx ctx;
    unsafe {
        ctx.crate = (const struct Crate*)c;
        ctx.from = (const struct String*)from;
    }
    struct String html;
    unsafe { html = md_render_html_with(doc, render_hook, (?&RenderCtx)&ctx); }
    doc.free();
    return html;
}

struct String render_doc(const &Crate c, const &String text, int md, const &String from) {
    if (text.len == 0UL) { return String::new(); }
    struct String prepared = prepare(c, text, md, from);
    struct String html = render_md(c, &prepared, from);
    prepared.free();
    return html;
}

// The text up to the first blank line.
static struct String first_paragraph(const &String text) {
    long long cut = text.index_of("\n\n");
    struct String p = cut >= 0LL ? text.substr(0UL, (unsigned long)cut) : text.clone();
    // A paragraph that opens a code example ends before it.
    long long colon = p.index_of(":\n ");
    if (colon >= 0LL) { p.truncate((unsigned long)colon + 1UL); }
    return p;
}

struct String render_summary(const &Crate c, const &String text, int md, const &String from) {
    if (text.len == 0UL) { return String::new(); }
    struct String para = first_paragraph(text);
    struct String prepared = rewrite_shorthand(c, &para, from);
    struct String html = render_md(c, &prepared, from);
    para.free(); prepared.free();
    // '<p>…</p>\n' -> '…'
    struct String inner = html.trim();
    if (inner.starts_with("<p>") && inner.ends_with("</p>")) {
        struct String s = inner.substr(3UL, inner.len - 4UL);
        inner.free();
        inner = s;
    }
    html.free();
    return inner;
}

struct String doc_summary_text(const &String text, int md, unsigned long max) {
    struct String para = first_paragraph(text);
    struct String out = String::new();
    int space = 0;
    for (unsigned long i = 0UL; i < para.len; i = i + 1UL) {
        int ch = para.char_at(i);
        if (ch == (int)'\n' || ch == (int)' ' || ch == (int)'\t') { space = 1; continue; }
        if (ch == (int)'`') { continue; }
        if (space && out.len > 0UL) { out.push_char(' '); }
        space = 0;
        out.push_char((char)ch);
        // End of the first sentence.
        if (ch == (int)'.' && i + 1UL < para.len && (para.char_at(i + 1UL) == (int)' ' || para.char_at(i + 1UL) == (int)'\n') && out.len > 12UL) {
            break;
        }
    }
    if (out.len > max) {
        unsigned long cut = max;
        while (cut > 0UL && (out.char_at(cut) & 0xC0) == 0x80) { cut = cut - 1UL; }
        out.truncate(cut);
        out.push("…");
    }
    para.free();
    return out;
}

// ── signatures ───────────────────────────────────────────────────────────────

static const char* SIG_KEYWORDS() {
    return " struct union enum trait generic static const volatile inline extern typedef newtype"
           " region unsafe consteval must_use noreturn pure packed fn stack heap arena atomic"
           " restrict register thread_local define __attribute__ ";
}

static const char* SIG_TYPES() {
    return " void char short int long float double signed unsigned bool int8 int16 int32 int64"
           " uint8 uint16 uint32 uint64 size_t float16 bfloat16 vec ";
}

static int in_word_list(const char* list, const &String w) {
    struct String needle = String::from(" ");
    needle.push_str(w);
    needle.push(" ");
    struct String l;
    unsafe { l = String::from(list); }
    int hit = l.contains(needle.as_ptr());
    needle.free(); l.free();
    return hit;
}

static int ident_char(int ch) {
    return (ch >= (int)'a' && ch <= (int)'z') || (ch >= (int)'A' && ch <= (int)'Z') ||
           (ch >= (int)'0' && ch <= (int)'9') || ch == (int)'_';
}

// Long parameter lists one per line, rustdoc-style.
static struct String format_sig(const &String sig) {
    if (sig.len <= 90UL || sig.starts_with("#define")) { return sig.clone(); }
    long long open = -1LL;
    int depth = 0;
    for (unsigned long i = 0UL; i < sig.len; i = i + 1UL) {
        int ch = sig.char_at(i);
        if (ch == (int)'(') { if (depth == 0 && open < 0LL) { open = (long long)i; } depth = depth + 1; }
        if (ch == (int)')') { depth = depth - 1; }
    }
    if (open < 0LL) { return sig.clone(); }
    struct String out = sig.substr(0UL, (unsigned long)open + 1UL);
    depth = 0;
    unsigned long i = (unsigned long)open + 1UL;
    out.push("\n    ");
    while (i < sig.len) {
        int ch = sig.char_at(i);
        if (ch == (int)'(' || ch == (int)'<' || ch == (int)'[') { depth = depth + 1; }
        if (ch == (int)')' || ch == (int)'>' || ch == (int)']') {
            if (depth == 0 && ch == (int)')') {
                out.push(",\n");
                struct String rest = sig.substr(i, sig.len);
                out.push_str(&rest);
                rest.free();
                return out;
            }
            depth = depth - 1;
        }
        if (ch == (int)',' && depth == 0) {
            out.push(",\n    ");
            i = i + 1UL;
            while (i < sig.len && sig.char_at(i) == (int)' ') { i = i + 1UL; }
            continue;
        }
        out.push_char((char)ch);
        i = i + 1UL;
    }
    return out;
}

struct String render_sig(const &Crate c, const &String sig_in, const &String from, const &char self_name) {
    struct String sig = format_sig(sig_in);
    struct String self_s = String::from(self_name);
    struct String out = String::new();
    unsigned long i = 0UL;
    while (i < sig.len) {
        int ch = sig.char_at(i);
        if (ch == (int)'"' || ch == (int)'\'') {
            unsigned long k = i + 1UL;
            while (k < sig.len && sig.char_at(k) != ch) { if (sig.char_at(k) == (int)'\\') { k = k + 1UL; } k = k + 1UL; }
            if (k < sig.len) { k = k + 1UL; }
            out.push("<span class=\"tok-str\">");
            util_push_esc_n(&out, &sig, i, k);
            out.push("</span>");
            i = k;
            continue;
        }
        if (ident_char(ch) && !(ch >= (int)'0' && ch <= (int)'9')) {
            // An identifier, with '::' path segments.
            unsigned long k = i;
            for (;;) {
                while (k < sig.len && ident_char(sig.char_at(k))) { k = k + 1UL; }
                if (k + 2UL < sig.len && sig.char_at(k) == (int)':' && sig.char_at(k + 1UL) == (int)':' && ident_char(sig.char_at(k + 2UL))) {
                    k = k + 2UL;
                    continue;
                }
                break;
            }
            struct String word = sig.substr(i, k);
            if (in_word_list(SIG_KEYWORDS(), &word)) {
                out.push("<span class=\"tok-kw\">"); util_push_esc(&out, word.as_ptr()); out.push("</span>");
            } else if (in_word_list(SIG_TYPES(), &word)) {
                out.push("<span class=\"tok-ty\">"); util_push_esc(&out, word.as_ptr()); out.push("</span>");
            } else if (word.eq(&self_s)) {
                out.push("<span class=\"tok-fn\">"); util_push_esc(&out, word.as_ptr()); out.push("</span>");
            } else {
                struct String href = resolve_href(c, &word, from);
                if (href.len > 0UL) {
                    out.push("<a class=\"tok-link\" href=\""); util_push_esc(&out, href.as_ptr()); out.push("\">");
                    util_push_esc(&out, word.as_ptr());
                    out.push("</a>");
                } else {
                    util_push_esc(&out, word.as_ptr());
                }
                href.free();
            }
            word.free();
            i = k;
            continue;
        }
        if (ch >= (int)'0' && ch <= (int)'9') {
            unsigned long k = i;
            while (k < sig.len && (ident_char(sig.char_at(k)) || sig.char_at(k) == (int)'.')) { k = k + 1UL; }
            out.push("<span class=\"tok-num\">");
            util_push_esc_n(&out, &sig, i, k);
            out.push("</span>");
            i = k;
            continue;
        }
        util_push_esc_n(&out, &sig, i, i + 1UL);
        i = i + 1UL;
    }
    sig.free(); self_s.free();
    return out;
}
