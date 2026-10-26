// SafeC Standard Library — HTML serialization implementation (see html.h)
#pragma once
#include <std/serial/html.h>
#include <std/str.h>
#include <std/serial/value.h>
#include <std/collections/string.h>
#include <std/fmt.h>
#include <std/convert.h>

namespace std {

// Batches runs of bytes needing no escaping into one push_n_() call rather
// than one push_char() per byte — most string content has no special
// chars, so this is the hot path.
static void html_flush_run_(&String out, const char* s, unsigned long run_start, unsigned long i) {
    if (i > run_start) {
        unsafe { out.push_n_(s + run_start, i - run_start); }
    }
}

void html_escape(&String out, const &char s) {
    unsigned long i = 0UL;
    unsigned long run_start = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '&' || c == '<' || c == '>' || c == '"') {
            html_flush_run_(out, s, run_start, i);
            if (c == '&') { unsafe { out.push("&amp;"); } }
            else if (c == '<') { unsafe { out.push("&lt;"); } }
            else if (c == '>') { unsafe { out.push("&gt;"); } }
            else { unsafe { out.push("&quot;"); } }
            i = i + 1UL;
            run_start = i;
        } else {
            i = i + 1UL;
        }
    }
    html_flush_run_(out, s, run_start, i);
}

// Trims push_float's fixed trailing zeros — see the identical helper's
// comment in json.sc/xml.sc for why this is duplicated per format module.
static void html_append_float_(&String out, double v) {
    unsafe {
        unsigned long before = out.length();
        out.push_float(v, 6);
        unsigned long end = out.length();
        while (end > before) {
            int c = out.char_at(end - 1UL);
            if (c != (int)'0') { break; }
            end = end - 1UL;
        }
        if (end > before) {
            int c = out.char_at(end - 1UL);
            if (c == (int)'.') { end = end - 1UL; }
        }
        out.truncate(end);
    }
}

void html_write(const &Value v, &String out) {
    int kind;
    unsafe { kind = v.kind; }

    if (kind == VAL_NULL) {
        unsafe { out.push("<em>null</em>"); }
    } else if (kind == VAL_BOOL) {
        int b;
        unsafe { b = v.bool_val; }
        unsafe { out.push(b != 0 ? "true" : "false"); }
    } else if (kind == VAL_INT) {
        long long n;
        unsafe { n = v.int_val; }
        unsafe { out.push_int(n); }
    } else if (kind == VAL_FLOAT) {
        double f;
        unsafe { f = v.float_val; }
        html_append_float_(out, f);
    } else if (kind == VAL_STRING) {
        const char* s;
        unsafe {
            if (v.str_val != (char*)0) { s = (const char*)v.str_val; }
            else { s = (const char*)""; }
        }
        unsafe { html_escape(out, s); }
    } else if (kind == VAL_ARRAY) {
        unsafe { out.push("<ul>"); }
        unsigned long n;
        unsafe { n = v.arr_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            &Value elem;
            unsafe { elem = v.arr_val.at(i); }
            unsafe { out.push("<li>"); }
            html_write(elem, out);
            unsafe { out.push("</li>"); }
            i = i + 1UL;
        }
        unsafe { out.push("</ul>"); }
    } else if (kind == VAL_OBJECT) {
        unsafe { out.push("<dl>"); }
        unsigned long n;
        unsafe { n = v.obj_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct ObjectEntry* e;
            unsafe { e = v.obj_val.at(i); }
            const char* key;
            unsafe { key = (const char*)e->key; }
            &Value val;
            unsafe { val = e->val; }
            unsafe { out.push("<dt>"); }
            unsafe { html_escape(out, key); }
            unsafe { out.push("</dt><dd>"); }
            html_write(val, out);
            unsafe { out.push("</dd>"); }
            i = i + 1UL;
        }
        unsafe { out.push("</dl>"); }
    }
}

inline struct String value_to_html(const &Value v) {
    struct String out = String::new();
    html_write(v, &out);
    return out;
}

// UTF-8 for one code point (no std/encoding dependency for this one use).
static void html_push_utf8_(&String out, unsigned int cp) {
    if (cp < 0x80U) { out.push_char((char)cp); }
    else if (cp < 0x800U) {
        out.push_char((char)(0xC0U | (cp >> 6)));
        out.push_char((char)(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
        out.push_char((char)(0xE0U | (cp >> 12)));
        out.push_char((char)(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_char((char)(0x80U | (cp & 0x3FU)));
    } else {
        out.push_char((char)(0xF0U | (cp >> 18)));
        out.push_char((char)(0x80U | ((cp >> 12) & 0x3FU)));
        out.push_char((char)(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_char((char)(0x80U | (cp & 0x3FU)));
    }
}

// Code point for a named reference (the name without '&' / ';'), or 0.
static unsigned int html_named_entity_(const &String name) {
    if (name.eq_cstr("amp")) { return 0x26U; }
    if (name.eq_cstr("lt")) { return 0x3CU; }
    if (name.eq_cstr("gt")) { return 0x3EU; }
    if (name.eq_cstr("quot")) { return 0x22U; }
    if (name.eq_cstr("apos")) { return 0x27U; }
    if (name.eq_cstr("nbsp")) { return 0xA0U; }
    if (name.eq_cstr("copy")) { return 0xA9U; }
    if (name.eq_cstr("reg")) { return 0xAEU; }
    if (name.eq_cstr("trade")) { return 0x2122U; }
    if (name.eq_cstr("hellip")) { return 0x2026U; }
    if (name.eq_cstr("mdash")) { return 0x2014U; }
    if (name.eq_cstr("ndash")) { return 0x2013U; }
    if (name.eq_cstr("lsquo")) { return 0x2018U; }
    if (name.eq_cstr("rsquo")) { return 0x2019U; }
    if (name.eq_cstr("ldquo")) { return 0x201CU; }
    if (name.eq_cstr("rdquo")) { return 0x201DU; }
    if (name.eq_cstr("laquo")) { return 0xABU; }
    if (name.eq_cstr("raquo")) { return 0xBBU; }
    if (name.eq_cstr("middot")) { return 0xB7U; }
    if (name.eq_cstr("bull")) { return 0x2022U; }
    if (name.eq_cstr("times")) { return 0xD7U; }
    if (name.eq_cstr("divide")) { return 0xF7U; }
    if (name.eq_cstr("deg")) { return 0xB0U; }
    if (name.eq_cstr("larr")) { return 0x2190U; }
    if (name.eq_cstr("uarr")) { return 0x2191U; }
    if (name.eq_cstr("rarr")) { return 0x2192U; }
    if (name.eq_cstr("darr")) { return 0x2193U; }
    if (name.eq_cstr("harr")) { return 0x2194U; }
    if (name.eq_cstr("ne")) { return 0x2260U; }
    if (name.eq_cstr("le")) { return 0x2264U; }
    if (name.eq_cstr("ge")) { return 0x2265U; }
    return 0U;
}

void html_unescape(&String out, const &char s) {
    unsigned long i = 0UL;
    for (;;) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c != '&') { out.push_char(c); i = i + 1UL; continue; }
        // Find the ';' (references are short).
        unsigned long j = i + 1UL;
        unsafe { while (j - i <= 32UL && s[j] != (char)0 && s[j] != ';' && s[j] != '&' && s[j] != ' ') { j = j + 1UL; } }
        char end;
        unsafe { end = s[j]; }
        unsigned int cp = 0U;
        if (end == ';' && j > i + 1UL) {
            char h;
            unsafe { h = s[i + 1UL]; }
            if (h == '#') {
                unsigned long k = i + 2UL;
                int hex = 0;
                unsafe { if (s[k] == 'x' || s[k] == 'X') { hex = 1; k = k + 1UL; } }
                unsigned int v = 0U;
                int digits = 0;
                while (k < j) {
                    char d;
                    unsafe { d = s[k]; }
                    unsigned int dv = 99U;
                    if (d >= '0' && d <= '9') { dv = (unsigned int)(d - '0'); }
                    else if (hex && d >= 'a' && d <= 'f') { dv = (unsigned int)(d - 'a') + 10U; }
                    else if (hex && d >= 'A' && d <= 'F') { dv = (unsigned int)(d - 'A') + 10U; }
                    if (dv >= (hex ? 16U : 10U)) { digits = -1; break; }
                    if (v < 0x110000U) { v = v * (hex ? 16U : 10U) + dv; }
                    digits = digits + 1;
                    k = k + 1UL;
                }
                if (digits > 0 && v > 0U && v < 0x110000U) { cp = v; }
            } else {
                struct String name;
                unsafe { name = String::from_n_((const char*)s + i + 1UL, j - i - 1UL); }
                cp = html_named_entity_(&name);
                name.free();
            }
        }
        if (cp == 0U) { out.push_char('&'); i = i + 1UL; continue; }
        html_push_utf8_(out, cp);
        i = j + 1UL;
    }
}

// Is the tag name s[a..b) (lowercase compare) a block-level boundary?
static int html_is_block_tag_(const &char s, unsigned long a, unsigned long b) {
    struct String name;
    unsafe { name = String::from_n_((const char*)s + a, b - a); }
    struct String n = name.to_lower();
    name.free();
    int block = n.eq_cstr("p") || n.eq_cstr("br") || n.eq_cstr("div") || n.eq_cstr("li") ||
                n.eq_cstr("ul") || n.eq_cstr("ol") || n.eq_cstr("dl") || n.eq_cstr("dt") ||
                n.eq_cstr("dd") || n.eq_cstr("tr") || n.eq_cstr("td") || n.eq_cstr("th") ||
                n.eq_cstr("table") || n.eq_cstr("thead") || n.eq_cstr("tbody") || n.eq_cstr("pre") ||
                n.eq_cstr("blockquote") || n.eq_cstr("section") || n.eq_cstr("article") ||
                n.eq_cstr("header") || n.eq_cstr("footer") || n.eq_cstr("nav") || n.eq_cstr("aside") ||
                n.eq_cstr("main") || n.eq_cstr("hr") || n.eq_cstr("details") || n.eq_cstr("summary") ||
                n.eq_cstr("figure") || n.eq_cstr("figcaption") ||
                (n.len == 2UL && n.char_at(0UL) == (int)'h' && n.char_at(1UL) >= (int)'1' && n.char_at(1UL) <= (int)'6');
    n.free();
    return block;
}

void html_strip_tags(&String out, const &char s) {
    unsigned long i = 0UL;
    for (;;) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c != '<') { out.push_char(c); i = i + 1UL; continue; }
        unsafe {
            // Comment.
            if (s[i + 1UL] == '!' && s[i + 2UL] == '-' && s[i + 3UL] == '-') {
                unsigned long k = i + 4UL;
                while (s[k] != (char)0 && !(s[k] == '-' && s[k + 1UL] == '-' && s[k + 2UL] == '>')) { k = k + 1UL; }
                i = s[k] == (char)0 ? k : k + 3UL;
                continue;
            }
            unsigned long a = i + 1UL;
            if (s[a] == '/') { a = a + 1UL; }
            unsigned long b = a;
            while ((s[b] >= 'a' && s[b] <= 'z') || (s[b] >= 'A' && s[b] <= 'Z') || (s[b] >= '0' && s[b] <= '9')) { b = b + 1UL; }
            if (b == a && s[i + 1UL] != '!' && s[i + 1UL] != '?') {
                out.push_char('<');   // a lone '<', not a tag
                i = i + 1UL;
                continue;
            }
            // Skip to the tag's '>' (quoted attribute values may hold '>').
            unsigned long k = b;
            char q = (char)0;
            while (s[k] != (char)0) {
                if (q != (char)0) { if (s[k] == q) { q = (char)0; } }
                else if (s[k] == '"' || s[k] == '\'') { q = s[k]; }
                else if (s[k] == '>') { break; }
                k = k + 1UL;
            }
            int closing = s[i + 1UL] == '/';
            if (html_is_block_tag_(s, a, b) && out.len > 0UL) {
                int last = out.char_at(out.len - 1UL);
                if (last != (int)' ' && last != (int)'\n') { out.push_char(' '); }
            }
            i = s[k] == (char)0 ? k : k + 1UL;
            // <script>/<style>: drop everything up to the close tag.
            if (!closing && b - a == 6UL) {
                struct String nm = String::from_n_((const char*)s + a, 6UL);
                struct String lo = nm.to_lower();
                int raw = lo.eq_cstr("script");
                nm.free(); lo.free();
                if (raw) {
                    const ?&char close = str_find((const char*)s + i, "</script");
                    match (close) { case some(p): { i = (unsigned long)(p - (const char*)s); } case null: { } }
                }
            } else if (!closing && b - a == 5UL) {
                struct String nm = String::from_n_((const char*)s + a, 5UL);
                struct String lo = nm.to_lower();
                int raw = lo.eq_cstr("style");
                nm.free(); lo.free();
                if (raw) {
                    const ?&char close = str_find((const char*)s + i, "</style");
                    match (close) { case some(p): { i = (unsigned long)(p - (const char*)s); } case null: { } }
                }
            }
        }
    }
}

// ── Parser ────────────────────────────────────────────────────────────────
// Recursive descent over this module's own output shape (see html_write
// above): <em>null</em>, <ul><li>...</li>...</ul>, <dl><dt>k</dt><dd>v</dd>
// ...</dl>, or bare (possibly escaped) text for a scalar — not a general
// HTML parser. Same two structural ambiguities as xml_parse (see xml.sc):
// an empty <dd></dd>/<li></li>/top-level input reads back as VAL_STRING
// "", and numeric-/bool-looking text reads back as VAL_INT/VAL_FLOAT/
// VAL_BOOL rather than VAL_STRING. Prefer JSON when exact round-tripping
// of arbitrary string content matters.
struct HtmlCursor_ {
    const char* text;
    unsigned long pos;
    int ok;
};

static char html_peek_(struct HtmlCursor_* c) {
    unsafe { return c->text[c->pos]; }
}

static int html_eat_(struct HtmlCursor_* c, char expected) {
    if (html_peek_(c) != expected) { unsafe { c->ok = 0; } return 0; }
    unsafe { c->pos = c->pos + 1UL; }
    return 1;
}

static int html_eat_literal_(struct HtmlCursor_* c, const char* word) {
    unsigned long start;
    unsafe { start = c->pos; }
    unsigned long i = 0UL;
    while (1) {
        char want;
        unsafe { want = word[i]; }
        if (want == (char)0) { break; }
        if (html_peek_(c) != want) { unsafe { c->pos = start; } return 0; }
        unsafe { c->pos = c->pos + 1UL; }
        i = i + 1UL;
    }
    return 1;
}

// True if the cursor sits at '</' — i.e. no more siblings / no text.
static int html_at_close_(struct HtmlCursor_* c) {
    if (html_peek_(c) != '<') { return 0; }
    unsigned long p;
    char n;
    unsafe { p = c->pos; n = c->text[p + 1UL]; }
    return n == '/';
}

static struct String html_read_name_(struct HtmlCursor_* c) {
    unsigned long start;
    unsafe { start = c->pos; }
    while (1) {
        char ch = html_peek_(c);
        if (ch == (char)0 || ch == '>' || ch == '/' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    unsigned long end;
    unsafe { end = c->pos; }
    struct String out = String::new();
    if (end > start) { unsafe { out.push_n_(c->text + start, end - start); } }
    return out;
}

// Consumes '<name...>', flagging a parse error if the tag isn't 'name'.
static void html_expect_open_(struct HtmlCursor_* c, const char* name) {
    html_eat_(c, '<');
    struct String tag = html_read_name_(c);
    unsafe { if (tag.eq_cstr(name) == 0) { unsafe { c->ok = 0; } } }
    tag.free();
    while (1) {
        char cc = html_peek_(c);
        if (cc == '>' || cc == (char)0) { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    html_eat_(c, '>');
}

// Consumes '</name>', flagging a parse error if the tag isn't 'name'.
static void html_expect_close_(struct HtmlCursor_* c, const char* name) {
    html_eat_(c, '<');
    html_eat_(c, '/');
    struct String tag = html_read_name_(c);
    unsafe { if (tag.eq_cstr(name) == 0) { unsafe { c->ok = 0; } } }
    tag.free();
    while (1) {
        char cc = html_peek_(c);
        if (cc == '>' || cc == (char)0) { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    html_eat_(c, '>');
}

// Reads raw text up to the next '<' or end of input, decoding the 4
// entities html_escape emits.
static struct String html_decode_text_(struct HtmlCursor_* c) {
    struct String out = String::new();
    unsigned long run_start;
    unsafe { run_start = c->pos; }
    while (1) {
        char ch = html_peek_(c);
        if (ch == (char)0 || ch == '<') { break; }
        if (ch == '&') {
            unsigned long cur;
            unsafe { cur = c->pos; }
            if (cur > run_start) { unsafe { out.push_n_(c->text + run_start, cur - run_start); } }
            unsafe { c->pos = c->pos + 1UL; }
            if (html_eat_literal_(c, "amp;")) { out.push_char('&'); }
            else if (html_eat_literal_(c, "lt;")) { out.push_char('<'); }
            else if (html_eat_literal_(c, "gt;")) { out.push_char('>'); }
            else if (html_eat_literal_(c, "quot;")) { out.push_char('"'); }
            else { unsafe { c->ok = 0; } break; }
            unsafe { run_start = c->pos; }
        } else {
            unsafe { c->pos = c->pos + 1UL; }
        }
    }
    unsigned long cur2;
    unsafe { cur2 = c->pos; }
    if (cur2 > run_start) { unsafe { out.push_n_(c->text + run_start, cur2 - run_start); } }
    return out;
}

// Same classification rule as xml_infer_scalar_ (see its comment).
static struct Value html_infer_scalar_(struct String* text) {
    if (text->eq_cstr("true")) { return Value::from_bool(1); }
    if (text->eq_cstr("false")) { return Value::from_bool(0); }
    unsigned long len = text->length();
    if (len > 0UL) {
        int allInt = 1;
        int allFloatChars = 1;
        int hasDigit = 0;
        int hasDotOrExp = 0;
        unsigned long i = 0UL;
        while (i < len) {
            int ch = text->char_at(i);
            int isDigit = (ch >= (int)'0' && ch <= (int)'9');
            int isSign = (ch == (int)'-' || ch == (int)'+');
            int isDotExp = (ch == (int)'.' || ch == (int)'e' || ch == (int)'E');
            if (isDigit) { hasDigit = 1; }
            if (isDotExp) { hasDotOrExp = 1; }
            if (!(isDigit || (isSign && i == 0UL))) { allInt = 0; }
            if (!(isDigit || isSign || isDotExp)) { allFloatChars = 0; }
            i = i + 1UL;
        }
        if (allInt && hasDigit) {
            int ok = 0;
            long long n = text->parse_int(&ok);
            if (ok) { return Value::from_int(n); }
        }
        if (allFloatChars && hasDigit && hasDotOrExp) {
            int ok = 0;
            double f = text->parse_float(&ok);
            if (ok) { return Value::from_float(f); }
        }
    }
    return Value::from_str(text->as_ptr());
}

static struct Value html_parse_value_(struct HtmlCursor_* c) {
    int okNow;
    unsafe { okNow = c->ok; }
    if (!okNow) { return Value::null(); }

    if (html_at_close_(c)) {
        // No more content before the enclosing tag closes (or end of
        // input): could be an empty string or an empty object/array — see
        // the file-level ambiguity note above; resolved as "".
        return Value::from_str("");
    }
    if (html_peek_(c) != '<') {
        struct String text = html_decode_text_(c);
        struct Value v = html_infer_scalar_(&text);
        text.free();
        return v;
    }

    html_eat_(c, '<');
    struct String tag = html_read_name_(c);
    while (1) {
        char cc = html_peek_(c);
        if (cc == '>' || cc == (char)0) { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    html_eat_(c, '>');

    struct Value result;
    if (tag.eq_cstr("em")) {
        html_eat_literal_(c, "null");
        result = Value::null();
    } else if (tag.eq_cstr("ul")) {
        struct Value arr = Value::array();
        while (html_at_close_(c) == 0) {
            int stillOk;
            unsafe { stillOk = c->ok; }
            if (!stillOk) { break; }
            if (html_peek_(c) != '<') { unsafe { c->ok = 0; } break; }
            html_expect_open_(c, "li");
            struct Value elem = html_parse_value_(c);
            arr.array_push(elem);
            html_expect_close_(c, "li");
        }
        result = arr;
    } else if (tag.eq_cstr("dl")) {
        struct Value obj = Value::object();
        while (html_at_close_(c) == 0) {
            int stillOk2;
            unsafe { stillOk2 = c->ok; }
            if (!stillOk2) { break; }
            if (html_peek_(c) != '<') { unsafe { c->ok = 0; } break; }
            html_expect_open_(c, "dt");
            struct String key = html_decode_text_(c);
            html_expect_close_(c, "dt");
            html_expect_open_(c, "dd");
            struct Value val = html_parse_value_(c);
            html_expect_close_(c, "dd");
            unsafe { obj.object_set(key.as_ptr(), val); }
            key.free();
        }
        result = obj;
    } else {
        unsafe { c->ok = 0; }
        result = Value::null();
    }

    html_expect_close_(c, tag.as_ptr());
    tag.free();
    return result;
}

struct Value html_parse(const &char text, ?&int ok) {
    struct HtmlCursor_ c;
    c.text = text;
    c.pos = 0UL;
    c.ok = 1;
    struct Value v = html_parse_value_(&c);
    if (ok != (int*)0) { unsafe { *ok = c.ok; } }
    if (c.ok == 0) {
        v.free();
        return Value::null();
    }
    return v;
}

struct Html Html::new() {
    struct Html f;
    f.unused_ = 0;
    return f;
}

void Html::write(const &Value v, &String out) const { html_write(v, out); }

struct Value Html::parse(const &char text, ?&int ok) const { return html_parse(text, ok); }

} // namespace std
