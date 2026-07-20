// SafeC Standard Library — HTML serialization implementation (see html.h)
#pragma once
#include <std/serial/html.h>
#include <std/serial/value.sc>
#include <std/collections/string.sc>
#include <std/fmt.sc>
#include <std/convert.sc>

namespace std {

// Batches runs of bytes needing no escaping into one push_n() call rather
// than one push_char() per byte — most string content has no special
// chars, so this is the hot path.
static void html_flush_run_(&String out, const char* s, unsigned long run_start, unsigned long i) {
    if (i > run_start) {
        unsafe { out.push_n(s + run_start, i - run_start); }
    }
}

void html_escape(&String out, const char* s) {
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
        html_escape(out, s);
    } else if (kind == VAL_ARRAY) {
        unsafe { out.push("<ul>"); }
        unsigned long n;
        unsafe { n = v.arr_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct Value* elem;
            unsafe { elem = (struct Value*)v.arr_val.get_raw(i); }
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
            unsafe { e = (struct ObjectEntry*)v.obj_val.get_raw(i); }
            const char* key;
            unsafe { key = (const char*)e->key; }
            struct Value* val;
            unsafe { val = e->val; }
            unsafe { out.push("<dt>"); }
            html_escape(out, key);
            unsafe { out.push("</dt><dd>"); }
            html_write(val, out);
            unsafe { out.push("</dd>"); }
            i = i + 1UL;
        }
        unsafe { out.push("</dl>"); }
    }
}

inline struct String value_to_html(const &Value v) {
    struct String out = string_new();
    html_write(v, &out);
    return out;
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
    struct String out = string_new();
    if (end > start) { unsafe { out.push_n(c->text + start, end - start); } }
    return out;
}

// Consumes '<name...>', flagging a parse error if the tag isn't 'name'.
static void html_expect_open_(struct HtmlCursor_* c, const char* name) {
    html_eat_(c, '<');
    struct String tag = html_read_name_(c);
    if (tag.eq_cstr(name) == 0) { unsafe { c->ok = 0; } }
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
    if (tag.eq_cstr(name) == 0) { unsafe { c->ok = 0; } }
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
    struct String out = string_new();
    unsigned long run_start;
    unsafe { run_start = c->pos; }
    while (1) {
        char ch = html_peek_(c);
        if (ch == (char)0 || ch == '<') { break; }
        if (ch == '&') {
            unsigned long cur;
            unsafe { cur = c->pos; }
            if (cur > run_start) { unsafe { out.push_n(c->text + run_start, cur - run_start); } }
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
    if (cur2 > run_start) { unsafe { out.push_n(c->text + run_start, cur2 - run_start); } }
    return out;
}

// Same classification rule as xml_infer_scalar_ (see its comment).
static struct Value html_infer_scalar_(struct String* text) {
    if (text->eq_cstr("true")) { return value_bool(1); }
    if (text->eq_cstr("false")) { return value_bool(0); }
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
            if (ok) { return value_int(n); }
        }
        if (allFloatChars && hasDigit && hasDotOrExp) {
            int ok = 0;
            double f = text->parse_float(&ok);
            if (ok) { return value_float(f); }
        }
    }
    return value_string(text->as_ptr());
}

static struct Value html_parse_value_(struct HtmlCursor_* c) {
    int okNow;
    unsafe { okNow = c->ok; }
    if (!okNow) { return value_null(); }

    if (html_at_close_(c)) {
        // No more content before the enclosing tag closes (or end of
        // input): could be an empty string or an empty object/array — see
        // the file-level ambiguity note above; resolved as "".
        return value_string("");
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
        result = value_null();
    } else if (tag.eq_cstr("ul")) {
        struct Value arr = value_array();
        while (html_at_close_(c) == 0) {
            int stillOk;
            unsafe { stillOk = c->ok; }
            if (!stillOk) { break; }
            if (html_peek_(c) != '<') { unsafe { c->ok = 0; } break; }
            html_expect_open_(c, "li");
            struct Value elem = html_parse_value_(c);
            value_array_push(&arr, elem);
            html_expect_close_(c, "li");
        }
        result = arr;
    } else if (tag.eq_cstr("dl")) {
        struct Value obj = value_object();
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
            unsafe { value_object_set(&obj, key.as_ptr(), val); }
            key.free();
        }
        result = obj;
    } else {
        unsafe { c->ok = 0; }
        result = value_null();
    }

    html_expect_close_(c, tag.as_ptr());
    tag.free();
    return result;
}

struct Value html_parse(const char* text, int* ok) {
    struct HtmlCursor_ c;
    c.text = text;
    c.pos = 0UL;
    c.ok = 1;
    struct Value v = html_parse_value_(&c);
    if (ok != (int*)0) { unsafe { *ok = c.ok; } }
    if (c.ok == 0) {
        value_free(&v);
        return value_null();
    }
    return v;
}

} // namespace std
