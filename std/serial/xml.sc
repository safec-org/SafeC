// SafeC Standard Library — XML serialization implementation (see xml.h)
#pragma once
#include <std/serial/xml.h>
#include <std/serial/value.sc>
#include <std/collections/string.sc>
#include <std/fmt.sc>
#include <std/convert.sc>

namespace std {

// Batches runs of bytes needing no escaping into one push_n() call rather
// than one push_char() per byte — most string content has no special
// chars, so this is the hot path.
static void xml_flush_run_(struct String* out, const char* s, unsigned long run_start, unsigned long i) {
    if (i > run_start) {
        unsafe { out->push_n(s + run_start, i - run_start); }
    }
}

static void xml_append_escaped_(struct String* out, const char* s) {
    unsigned long i = 0UL;
    unsigned long run_start = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') {
            xml_flush_run_(out, s, run_start, i);
            if (c == '&') { unsafe { out->push("&amp;"); } }
            else if (c == '<') { unsafe { out->push("&lt;"); } }
            else if (c == '>') { unsafe { out->push("&gt;"); } }
            else if (c == '"') { unsafe { out->push("&quot;"); } }
            else { unsafe { out->push("&apos;"); } }
            i = i + 1UL;
            run_start = i;
        } else {
            i = i + 1UL;
        }
    }
    xml_flush_run_(out, s, run_start, i);
}

// Shared by json.sc's json_append_float_ in spirit, duplicated rather than
// shared across modules since the two are otherwise independent leaf
// files (see the file-level comments in each) — trims push_float's fixed
// trailing zeros.
static void xml_append_float_(struct String* out, double v) {
    unsafe {
        unsigned long before = out->length();
        out->push_float(v, 6);
        unsigned long end = out->length();
        while (end > before) {
            int c = out->char_at(end - 1UL);
            if (c != (int)'0') { break; }
            end = end - 1UL;
        }
        if (end > before) {
            int c = out->char_at(end - 1UL);
            if (c == (int)'.') { end = end - 1UL; }
        }
        out->truncate(end);
    }
}

void xml_write(const struct Value* v, const char* tag, struct String* out) {
    int kind;
    unsafe { kind = v->kind; }

    if (kind == VAL_NULL) {
        unsafe { out->push_char('<'); out->push(tag); out->push("/>"); }
        return;
    }

    unsafe { out->push_char('<'); out->push(tag); out->push_char('>'); }

    if (kind == VAL_BOOL) {
        int b;
        unsafe { b = v->bool_val; }
        unsafe { out->push(b != 0 ? "true" : "false"); }
    } else if (kind == VAL_INT) {
        long long n;
        unsafe { n = v->int_val; }
        unsafe { out->push_int(n); }
    } else if (kind == VAL_FLOAT) {
        double f;
        unsafe { f = v->float_val; }
        xml_append_float_(out, f);
    } else if (kind == VAL_STRING) {
        const char* s;
        unsafe {
            if (v->str_val != (char*)0) { s = (const char*)v->str_val; }
            else { s = (const char*)""; }
        }
        xml_append_escaped_(out, s);
    } else if (kind == VAL_ARRAY) {
        unsigned long n;
        unsafe { n = v->arr_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct Value* elem;
            unsafe { elem = (struct Value*)v->arr_val.get_raw(i); }
            xml_write(elem, "item", out);
            i = i + 1UL;
        }
    } else if (kind == VAL_OBJECT) {
        unsigned long n;
        unsafe { n = v->obj_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            struct ObjectEntry* e;
            unsafe { e = (struct ObjectEntry*)v->obj_val.get_raw(i); }
            const char* key;
            unsafe { key = (const char*)e->key; }
            struct Value* val;
            unsafe { val = e->val; }
            xml_write(val, key, out);
            i = i + 1UL;
        }
    }

    unsafe { out->push("</"); out->push(tag); out->push_char('>'); }
}

inline struct String value_to_xml(const struct Value* v, const char* root_tag) {
    struct String out = string_new();
    xml_write(v, root_tag, &out);
    return out;
}

// ── Parser ────────────────────────────────────────────────────────────────
// Recursive descent over this module's own output shape (see xml_write
// above) — not a general XML parser (no attributes, no CDATA, no entity
// refs beyond the 5 xml_append_escaped_ emits, no processing instructions).
// Two mapping ambiguities are inherent to the tag-per-value grammar itself
// (not parser bugs) and are resolved the same way on every parse:
//   - <t></t> (no children, no text) could be an empty string or an empty
//     object; this parser always resolves it to VAL_STRING "".
//   - scalar text content that happens to look like a number/bool ("42",
//     "true") is read back as VAL_INT/VAL_BOOL rather than VAL_STRING, same
//     as it would if it had actually been serialized from one. JSON has no
//     such ambiguity (its syntax marks type explicitly) — prefer it when
//     lossless round-tripping through arbitrary string content matters.
struct XmlCursor_ {
    const char* text;
    unsigned long pos;
    int ok;
};

static char xml_peek_(struct XmlCursor_* c) {
    unsafe { return c->text[c->pos]; }
}

static int xml_eat_(struct XmlCursor_* c, char expected) {
    if (xml_peek_(c) != expected) { unsafe { c->ok = 0; } return 0; }
    unsafe { c->pos = c->pos + 1UL; }
    return 1;
}

// Matches a literal keyword starting at the cursor, consuming it on success
// and leaving the cursor untouched on failure.
static int xml_eat_literal_(struct XmlCursor_* c, const char* word) {
    unsigned long start;
    unsafe { start = c->pos; }
    unsigned long i = 0UL;
    while (1) {
        char want;
        unsafe { want = word[i]; }
        if (want == (char)0) { break; }
        if (xml_peek_(c) != want) { unsafe { c->pos = start; } return 0; }
        unsafe { c->pos = c->pos + 1UL; }
        i = i + 1UL;
    }
    return 1;
}

// True if the cursor sits at '</' — the closing tag of the element whose
// content is currently being parsed (i.e. no more children / no text).
static int xml_at_close_(struct XmlCursor_* c) {
    if (xml_peek_(c) != '<') { return 0; }
    unsigned long p;
    char n;
    unsafe { p = c->pos; n = c->text[p + 1UL]; }
    return n == '/';
}

static struct String xml_read_name_(struct XmlCursor_* c) {
    unsigned long start;
    unsafe { start = c->pos; }
    while (1) {
        char ch = xml_peek_(c);
        if (ch == (char)0 || ch == '>' || ch == '/' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    unsigned long end;
    unsafe { end = c->pos; }
    struct String out = string_new();
    if (end > start) { unsafe { out.push_n(c->text + start, end - start); } }
    return out;
}

// Reads the tag name of the element starting at the cursor without moving
// it — used by the children loop to decide array-vs-object (and to get an
// object key) before recursing into xml_parse_element_ for the value.
static struct String xml_peek_tag_name_(struct XmlCursor_* c) {
    unsigned long saved;
    unsafe { saved = c->pos; }
    unsafe { c->pos = c->pos + 1UL; } // skip '<'
    struct String name = xml_read_name_(c);
    unsafe { c->pos = saved; }
    return name;
}

// Reads raw text up to (not including) the next '<' or end of input,
// decoding the 5 entities xml_append_escaped_ emits.
static struct String xml_decode_text_(struct XmlCursor_* c) {
    struct String out = string_new();
    unsigned long run_start;
    unsafe { run_start = c->pos; }
    while (1) {
        char ch = xml_peek_(c);
        if (ch == (char)0 || ch == '<') { break; }
        if (ch == '&') {
            unsigned long cur;
            unsafe { cur = c->pos; }
            if (cur > run_start) { unsafe { out.push_n(c->text + run_start, cur - run_start); } }
            unsafe { c->pos = c->pos + 1UL; }
            if (xml_eat_literal_(c, "amp;")) { out.push_char('&'); }
            else if (xml_eat_literal_(c, "lt;")) { out.push_char('<'); }
            else if (xml_eat_literal_(c, "gt;")) { out.push_char('>'); }
            else if (xml_eat_literal_(c, "quot;")) { out.push_char('"'); }
            else if (xml_eat_literal_(c, "apos;")) { out.push_char('\''); }
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

// Classifies decoded scalar text the same way json_parse_number_ would —
// "true"/"false" -> bool, an all-digit (optional leading '-') span -> int,
// a digit span containing '.'/'e'/'E' -> float, everything else -> string.
static struct Value xml_infer_scalar_(struct String* text) {
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

static struct Value xml_parse_element_(struct XmlCursor_* c) {
    xml_eat_(c, '<');
    struct String tag = xml_read_name_(c);
    while (1) {
        char ch = xml_peek_(c);
        if (ch == (char)0) { unsafe { c->ok = 0; } break; }
        if (ch == '>' || ch == '/') { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
    int selfClose = 0;
    if (xml_peek_(c) == '/') { selfClose = 1; unsafe { c->pos = c->pos + 1UL; } }
    xml_eat_(c, '>');

    struct Value result;
    int okNow;
    unsafe { okNow = c->ok; }
    if (selfClose || !okNow) {
        result = value_null();
    } else if (xml_at_close_(c)) {
        // No children, no text: could be an empty string or empty object —
        // see the file-level ambiguity note above.
        result = value_string("");
    } else if (xml_peek_(c) == '<') {
        struct Value arr = value_array();
        struct Value obj = value_object();
        int isArray = 1;
        int first = 1;
        while (xml_at_close_(c) == 0) {
            int stillOk;
            unsafe { stillOk = c->ok; }
            if (!stillOk) { break; }
            if (xml_peek_(c) != '<') { unsafe { c->ok = 0; } break; }
            struct String childTag = xml_peek_tag_name_(c);
            struct Value childVal = xml_parse_element_(c);
            int childIsItem = childTag.eq_cstr("item");
            if (first) { isArray = childIsItem; first = 0; }
            if (isArray) {
                value_array_push(&arr, childVal);
            } else {
                unsafe { value_object_set(&obj, childTag.as_ptr(), childVal); }
            }
            childTag.free();
        }
        if (isArray) { value_free(&obj); result = arr; }
        else { value_free(&arr); result = obj; }
    } else {
        struct String text = xml_decode_text_(c);
        result = xml_infer_scalar_(&text);
        text.free();
    }

    // Self-closing elements ('<t/>') have no separate '</t>' to consume.
    if (!selfClose) {
        xml_eat_(c, '<');
        xml_eat_(c, '/');
        struct String closeName = xml_read_name_(c);
        if (closeName.eq(&tag) == 0) { unsafe { c->ok = 0; } }
        closeName.free();
        while (1) {
            char cc = xml_peek_(c);
            if (cc == '>' || cc == (char)0) { break; }
            unsafe { c->pos = c->pos + 1UL; }
        }
        xml_eat_(c, '>');
    }
    tag.free();
    return result;
}

struct Value xml_parse(const char* text, int* ok) {
    struct XmlCursor_ c;
    c.text = text;
    c.pos = 0UL;
    c.ok = 1;
    struct Value v;
    if (xml_peek_(&c) != '<') {
        c.ok = 0;
        v = value_null();
    } else {
        v = xml_parse_element_(&c);
    }
    if (ok != (int*)0) { unsafe { *ok = c.ok; } }
    if (c.ok == 0) {
        value_free(&v);
        return value_null();
    }
    return v;
}

} // namespace std
