// SafeC Standard Library — CSV serialization implementation (see csv.h)
#pragma once
#include <std/serial/csv.h>
#include <std/serial/value.sc>
#include <std/collections/string.sc>
#include <std/mem.sc>
// String::push_int/push_float (used by csv_append_field_ below) are
// declared in string.h but only *defined* in these two files — see
// json.sc's identical comment for why both are needed here too.
#include <std/fmt.sc>
#include <std/convert.sc>

namespace std {

// ── Writer ──────────────────────────────────────────────────────────────────

static int csv_needs_quoting_(const char* s) {
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { return 0; }
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { return 1; }
        i = i + 1UL;
    }
}

static void csv_append_quoted_(&String out, const char* s) {
    unsafe { out.push_char('"'); }
    unsigned long i = 0UL;
    unsigned long run_start = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '"') {
            if (i > run_start) { unsafe { out.push_n(s + run_start, i - run_start); } }
            unsafe { out.push("\"\""); } // escape: one quote becomes two
            i = i + 1UL;
            run_start = i;
        } else {
            i = i + 1UL;
        }
    }
    if (i > run_start) { unsafe { out.push_n(s + run_start, i - run_start); } }
    unsafe { out.push_char('"'); }
}

// Same trailing-zero-trim behavior as json.sc's json_append_float_ (kept
// as a separate copy rather than shared, matching html.sc/xml.sc each
// having their own small helpers instead of a cross-format-backend
// dependency between the format modules themselves).
static void csv_append_float_(&String out, double v) {
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

static void csv_append_field_(&String out, const struct Value* field) {
    int kind;
    unsafe { kind = field->kind; }
    if (kind == VAL_NULL) {
        return; // empty field
    } else if (kind == VAL_STRING) {
        const char* s;
        unsafe {
            if (field->str_val != (char*)0) { s = (const char*)field->str_val; }
            else { s = (const char*)""; }
        }
        if (csv_needs_quoting_(s)) { csv_append_quoted_(out, s); }
        else { unsafe { out.push(s); } }
    } else if (kind == VAL_INT) {
        long long n;
        unsafe { n = field->int_val; }
        unsafe { out.push_int(n); }
    } else if (kind == VAL_FLOAT) {
        double f;
        unsafe { f = field->float_val; }
        csv_append_float_(out, f);
    } else if (kind == VAL_BOOL) {
        int b;
        unsafe { b = field->bool_val; }
        unsafe { out.push(b != 0 ? "true" : "false"); }
    }
    // VAL_ARRAY/VAL_OBJECT fields aren't meaningful in a CSV cell — silently
    // written as nothing, same as VAL_NULL, rather than erroring: a caller
    // building rows programmatically shouldn't have a malformed nested
    // value abort the whole document.
}

void csv_write(const &Value v, &String out) {
    unsigned long rowCount;
    unsafe { rowCount = v.arr_val.length(); }
    unsigned long r = 0UL;
    while (r < rowCount) {
        if (r > 0UL) { unsafe { out.push_char('\n'); } }
        struct Value* row;
        unsafe { row = (struct Value*)v.arr_val.get_raw(r); }
        unsigned long fieldCount;
        unsafe { fieldCount = row->arr_val.length(); }
        unsigned long f = 0UL;
        while (f < fieldCount) {
            if (f > 0UL) { unsafe { out.push_char(','); } }
            struct Value* field;
            unsafe { field = (struct Value*)row->arr_val.get_raw(f); }
            csv_append_field_(out, field);
            f = f + 1UL;
        }
        r = r + 1UL;
    }
}

inline struct String value_to_csv(const &Value v) {
    struct String out = string_new();
    csv_write(v, &out);
    return out;
}

// ── Parser ────────────────────────────────────────────────────────────────

struct CsvCursor_ {
    const char* text;
    unsigned long pos;
    int ok;
};

static char csv_peek_(struct CsvCursor_* c) {
    unsafe { return c->text[c->pos]; }
}

// Parses one field starting at the cursor, stopping at (but not consuming)
// the following ',' / '\n' / '\r' / end-of-text.
static struct Value csv_parse_field_(struct CsvCursor_* c) {
    struct String out = string_new();
    if (csv_peek_(c) == '"') {
        unsafe { c->pos = c->pos + 1UL; } // consume opening quote
        while (1) {
            char ch = csv_peek_(c);
            if (ch == (char)0) {
                unsafe { c->ok = 0; } // unterminated quoted field
                break;
            }
            if (ch == '"') {
                char next;
                unsafe { next = c->text[c->pos + 1UL]; }
                if (next == '"') {
                    unsafe { out.push_char('"'); c->pos = c->pos + 2UL; }
                    continue;
                }
                unsafe { c->pos = c->pos + 1UL; } // consume closing quote
                break;
            }
            unsafe { out.push_char(ch); c->pos = c->pos + 1UL; }
        }
    } else {
        while (1) {
            char ch = csv_peek_(c);
            if (ch == (char)0 || ch == ',' || ch == '\n' || ch == '\r') { break; }
            unsafe { out.push_char(ch); c->pos = c->pos + 1UL; }
        }
    }
    struct Value v = value_string(out.as_ptr());
    out.free();
    return v;
}

static struct Value csv_parse_row_(struct CsvCursor_* c) {
    struct Value row = value_array();
    while (1) {
        struct Value field = csv_parse_field_(c);
        value_array_push(&row, field);
        int stillOk;
        unsafe { stillOk = c->ok; }
        if (stillOk == 0) { break; }
        char ch = csv_peek_(c);
        if (ch == ',') { unsafe { c->pos = c->pos + 1UL; } continue; }
        break; // '\n', '\r', or end of text ends the row
    }
    return row;
}

struct Value csv_parse(const char* text, int* ok) {
    struct Value rows = value_array();
    struct CsvCursor_ c;
    c.text = text;
    c.pos = 0UL;
    c.ok = 1;
    while (1) {
        char first = csv_peek_(&c);
        if (first == (char)0) { break; } // no trailing phantom row after EOF
        struct Value row = csv_parse_row_(&c);
        value_array_push(&rows, row);
        if (c.ok == 0) { break; }
        char ch = csv_peek_(&c);
        if (ch == '\r') { unsafe { c.pos = c.pos + 1UL; } ch = csv_peek_(&c); }
        if (ch == '\n') { unsafe { c.pos = c.pos + 1UL; } }
    }
    if (ok != (int*)0) { unsafe { *ok = c.ok; } }
    return rows;
}

} // namespace std
