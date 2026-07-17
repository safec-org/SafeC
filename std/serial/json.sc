// SafeC Standard Library — JSON serialization implementation (see json.h)
#pragma once
#include <std/serial/json.h>
#include <std/serial/value.sc>
#include <std/collections/string.sc>
#include <std/mem.sc>
// String::push_int/push_float/parse_int/parse_float (both used heavily
// below) are declared in string.h but only *defined* in these two files —
// string.sc itself doesn't pull them in (see std/collections/string.sc's
// own includes), so anything that actually calls them, like this file,
// needs to.
#include <std/fmt.sc>
#include <std/convert.sc>

namespace std {

// ── Writer ──────────────────────────────────────────────────────────────────

static char json_hex_digit_(int v) {
    if (v < 10) { return (char)((int)'0' + v); }
    return (char)((int)'a' + (v - 10));
}

// Appends s[run_start, i) as one push_n() call — batching runs of bytes
// that need no escaping avoids a push_char() (and reserve_() check) per
// byte, which matters since most string content has no special chars.
static void json_flush_run_(struct String* out, const char* s, unsigned long run_start, unsigned long i) {
    if (i > run_start) {
        unsafe { out->push_n(s + run_start, i - run_start); }
    }
}

static void json_append_escaped_(struct String* out, const char* s) {
    unsafe { out->push_char('"'); }
    unsigned long i = 0UL;
    unsigned long run_start = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t' || (int)c < 32) {
            json_flush_run_(out, s, run_start, i);
            if (c == '"') {
                unsafe { out->push("\\\""); }
            } else if (c == '\\') {
                unsafe { out->push("\\\\"); }
            } else if (c == '\n') {
                unsafe { out->push("\\n"); }
            } else if (c == '\r') {
                unsafe { out->push("\\r"); }
            } else if (c == '\t') {
                unsafe { out->push("\\t"); }
            } else {
                unsafe {
                    out->push("\\u00");
                    out->push_char(json_hex_digit_(((int)c >> 4) & 0xF));
                    out->push_char(json_hex_digit_((int)c & 0xF));
                }
            }
            i = i + 1UL;
            run_start = i;
        } else {
            i = i + 1UL;
        }
    }
    json_flush_run_(out, s, run_start, i);
    unsafe { out->push_char('"'); }
}

// push_float always emits a fixed number of decimals ('3.140000') — trim
// the trailing zeros (and a now-bare trailing '.') so ordinary values read
// the way a human (or another JSON parser's float formatter) would write
// them, without losing precision push_float actually computed.
static void json_append_float_(struct String* out, double v) {
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

void json_write(const struct Value* v, struct String* out) {
    int kind;
    unsafe { kind = v->kind; }
    if (kind == VAL_NULL) {
        unsafe { out->push("null"); }
    } else if (kind == VAL_BOOL) {
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
        json_append_float_(out, f);
    } else if (kind == VAL_STRING) {
        const char* s;
        unsafe {
            if (v->str_val != (char*)0) { s = (const char*)v->str_val; }
            else { s = (const char*)""; }
        }
        json_append_escaped_(out, s);
    } else if (kind == VAL_ARRAY) {
        unsafe { out->push_char('['); }
        unsigned long n;
        unsafe { n = v->arr_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            if (i > 0UL) { unsafe { out->push_char(','); } }
            struct Value* elem;
            unsafe { elem = (struct Value*)v->arr_val.get_raw(i); }
            json_write(elem, out);
            i = i + 1UL;
        }
        unsafe { out->push_char(']'); }
    } else if (kind == VAL_OBJECT) {
        unsafe { out->push_char('{'); }
        unsigned long n;
        unsafe { n = v->obj_val.length(); }
        unsigned long i = 0UL;
        while (i < n) {
            if (i > 0UL) { unsafe { out->push_char(','); } }
            struct ObjectEntry* e;
            unsafe { e = (struct ObjectEntry*)v->obj_val.get_raw(i); }
            const char* key;
            unsafe { key = (const char*)e->key; }
            json_append_escaped_(out, key);
            unsafe { out->push_char(':'); }
            struct Value* val;
            unsafe { val = e->val; }
            json_write(val, out);
            i = i + 1UL;
        }
        unsafe { out->push_char('}'); }
    }
}

inline struct String value_to_json(const struct Value* v) {
    struct String out = string_new();
    json_write(v, &out);
    return out;
}

// ── Parser ────────────────────────────────────────────────────────────────
// Plain recursive descent over a NUL-terminated C string. No exceptions in
// this language, so failure propagates via a shared 'ok' flag every
// sub-parser checks before doing any work — once it's 0, every remaining
// call in the current parse becomes a cheap no-op that returns value_null().

struct JsonCursor_ {
    const char* text;
    unsigned long pos;
    int ok;
};

static char json_peek_(struct JsonCursor_* c) {
    unsafe { return c->text[c->pos]; }
}

static void json_skip_ws_(struct JsonCursor_* c) {
    while (1) {
        char ch = json_peek_(c);
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') { break; }
        unsafe { c->pos = c->pos + 1UL; }
    }
}

static int json_eat_(struct JsonCursor_* c, char expected) {
    if (json_peek_(c) != expected) { unsafe { c->ok = 0; } return 0; }
    unsafe { c->pos = c->pos + 1UL; }
    return 1;
}

// Matches a literal keyword ("true"/"false"/"null") starting at the cursor.
static int json_eat_literal_(struct JsonCursor_* c, const char* word) {
    unsigned long start;
    unsafe { start = c->pos; }
    unsigned long i = 0UL;
    while (1) {
        char want;
        unsafe { want = word[i]; }
        if (want == (char)0) { break; }
        if (json_peek_(c) != want) { unsafe { c->pos = start; } return 0; }
        unsafe { c->pos = c->pos + 1UL; }
        i = i + 1UL;
    }
    return 1;
}

static struct Value json_parse_value_(struct JsonCursor_* c);

static struct Value json_parse_string_raw_(struct JsonCursor_* c) {
    // Caller already confirmed the opening '"'.
    struct String out = string_new();
    json_eat_(c, '"');
    while (1) {
        int stillOk;
        unsafe { stillOk = c->ok; }
        if (stillOk == 0) { break; }
        char ch = json_peek_(c);
        if (ch == (char)0) { unsafe { c->ok = 0; } break; }
        if (ch == '"') { unsafe { c->pos = c->pos + 1UL; } break; }
        if (ch == '\\') {
            unsafe { c->pos = c->pos + 1UL; }
            char esc = json_peek_(c);
            unsafe { c->pos = c->pos + 1UL; }
            if (esc == 'n') { unsafe { out.push_char('\n'); } }
            else if (esc == 't') { unsafe { out.push_char('\t'); } }
            else if (esc == 'r') { unsafe { out.push_char('\r'); } }
            else if (esc == '"') { unsafe { out.push_char('"'); } }
            else if (esc == '\\') { unsafe { out.push_char('\\'); } }
            else if (esc == '/') { unsafe { out.push_char('/'); } }
            else if (esc == 'u') {
                // \uXXXX — only the common BMP/ASCII-range case is decoded
                // (surrogate pairs for astral characters are not); a
                // best-effort single-byte fallback for anything outside
                // that, rather than failing the whole parse over it.
                int val = 0;
                int k = 0;
                while (k < 4) {
                    char hc = json_peek_(c);
                    unsafe { c->pos = c->pos + 1UL; }
                    int digit = 0;
                    if (hc >= '0' && hc <= '9') { digit = (int)hc - (int)'0'; }
                    else if (hc >= 'a' && hc <= 'f') { digit = (int)hc - (int)'a' + 10; }
                    else if (hc >= 'A' && hc <= 'F') { digit = (int)hc - (int)'A' + 10; }
                    else { unsafe { c->ok = 0; } }
                    val = val * 16 + digit;
                    k = k + 1;
                }
                unsafe { out.push_char((char)(val & 0xFF)); }
            } else {
                unsafe { c->ok = 0; }
            }
        } else {
            unsafe { out.push_char(ch); c->pos = c->pos + 1UL; }
        }
    }
    struct Value v = value_string(out.as_ptr());
    out.free();
    return v;
}

static struct Value json_parse_number_(struct JsonCursor_* c) {
    unsigned long start;
    unsafe { start = c->pos; }
    int isFloat = 0;
    if (json_peek_(c) == '-') { unsafe { c->pos = c->pos + 1UL; } }
    while (1) {
        char ch = json_peek_(c);
        if (ch >= '0' && ch <= '9') { unsafe { c->pos = c->pos + 1UL; } continue; }
        if (ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-') {
            isFloat = 1;
            unsafe { c->pos = c->pos + 1UL; }
            continue;
        }
        break;
    }
    unsigned long len;
    unsafe { len = c->pos - start; }
    if (len == 0UL) { unsafe { c->ok = 0; } return value_null(); }

    struct String tok = string_new();
    struct String numStr;
    unsafe { numStr = string_from_n(c->text + start, len); }
    tok.push_str(&numStr);
    numStr.free();
    int ok = 0;
    struct Value v;
    if (isFloat) {
        double f = tok.parse_float(&ok);
        v = value_float(f);
    } else {
        long long n = tok.parse_int(&ok);
        v = value_int(n);
    }
    tok.free();
    if (ok == 0) { unsafe { c->ok = 0; } }
    return v;
}

static struct Value json_parse_array_(struct JsonCursor_* c) {
    struct Value arr = value_array();
    json_eat_(c, '[');
    json_skip_ws_(c);
    if (json_peek_(c) == ']') { unsafe { c->pos = c->pos + 1UL; } return arr; }
    while (1) {
        int stillOk;
        unsafe { stillOk = c->ok; }
        if (stillOk == 0) { break; }
        json_skip_ws_(c);
        struct Value elem = json_parse_value_(c);
        value_array_push(&arr, elem);
        json_skip_ws_(c);
        char ch = json_peek_(c);
        if (ch == ',') { unsafe { c->pos = c->pos + 1UL; continue; } }
        if (ch == ']') { unsafe { c->pos = c->pos + 1UL; break; } }
        unsafe { c->ok = 0; }
        break;
    }
    return arr;
}

static struct Value json_parse_object_(struct JsonCursor_* c) {
    struct Value obj = value_object();
    json_eat_(c, '{');
    json_skip_ws_(c);
    if (json_peek_(c) == '}') { unsafe { c->pos = c->pos + 1UL; } return obj; }
    while (1) {
        int stillOk;
        unsafe { stillOk = c->ok; }
        if (stillOk == 0) { break; }
        json_skip_ws_(c);
        if (json_peek_(c) != '"') { unsafe { c->ok = 0; } break; }
        struct Value keyVal = json_parse_string_raw_(c);
        json_skip_ws_(c);
        json_eat_(c, ':');
        json_skip_ws_(c);
        struct Value val = json_parse_value_(c);
        unsafe { value_object_set(&obj, keyVal.as_string(), val); }
        value_free(&keyVal);
        json_skip_ws_(c);
        char ch = json_peek_(c);
        if (ch == ',') { unsafe { c->pos = c->pos + 1UL; continue; } }
        if (ch == '}') { unsafe { c->pos = c->pos + 1UL; break; } }
        unsafe { c->ok = 0; }
        break;
    }
    return obj;
}

static struct Value json_parse_value_(struct JsonCursor_* c) {
    json_skip_ws_(c);
    int stillOk2;
    unsafe { stillOk2 = c->ok; }
    if (stillOk2 == 0) { return value_null(); }
    char ch = json_peek_(c);
    if (ch == '{') { return json_parse_object_(c); }
    if (ch == '[') { return json_parse_array_(c); }
    if (ch == '"') { return json_parse_string_raw_(c); }
    if (ch == 't') {
        if (json_eat_literal_(c, "true")) { return value_bool(1); }
        unsafe { c->ok = 0; } return value_null();
    }
    if (ch == 'f') {
        if (json_eat_literal_(c, "false")) { return value_bool(0); }
        unsafe { c->ok = 0; } return value_null();
    }
    if (ch == 'n') {
        if (json_eat_literal_(c, "null")) { return value_null(); }
        unsafe { c->ok = 0; } return value_null();
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) { return json_parse_number_(c); }
    unsafe { c->ok = 0; }
    return value_null();
}

struct Value json_parse(const char* text, int* ok) {
    struct JsonCursor_ c;
    c.text = text;
    c.pos = 0UL;
    c.ok = 1;
    struct Value v = json_parse_value_(&c);
    if (ok != (int*)0) { unsafe { *ok = c.ok; } }
    if (c.ok == 0) {
        value_free(&v);
        return value_null();
    }
    return v;
}

} // namespace std
