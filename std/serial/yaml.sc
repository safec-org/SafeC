// SafeC Standard Library — YAML implementation (see yaml.h for the
// supported subset).
#pragma once
#include <std/serial/yaml.h>
#include <std/serial/value.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>

namespace std {

// ── Writer ──────────────────────────────────────────────────────────────────

static void yaml_indent_(&String out, int depth) {
    int i = 0;
    while (i < depth) {
        unsafe { out.push("  "); }
        i = i + 1;
    }
}

static int yaml_scalar_needs_quoting_(const char* s) {
    char c0;
    unsafe { c0 = s[0]; }
    if (c0 == (char)0) { return 1; } // empty string must be quoted ('' ), else it reads as null
    if (c0 == ' ' || c0 == '-' || c0 == '?' || c0 == ':' || c0 == '#' ||
        c0 == '&' || c0 == '*' || c0 == '!' || c0 == '|' || c0 == '>' ||
        c0 == '\'' || c0 == '"' || c0 == '%' || c0 == '@' || c0 == '`' ||
        c0 == '[' || c0 == ']' || c0 == '{' || c0 == '}' || c0 == ',') {
        return 1;
    }
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == ':' || c == '#' || c == '\n') { return 1; }
        i = i + 1UL;
    }
    struct String trimmed; unsafe { trimmed = String::from(s); }
    struct String trimmedCopy = trimmed.trim();
    trimmed.free();
    int looksLikeOther = trimmedCopy.eq_cstr("true") || trimmedCopy.eq_cstr("false") ||
                          trimmedCopy.eq_cstr("null") || trimmedCopy.eq_cstr("~") ||
                          trimmedCopy.eq_cstr("True") || trimmedCopy.eq_cstr("False") ||
                          trimmedCopy.eq_cstr("Null");
    trimmedCopy.free();
    return looksLikeOther;
}

static void yaml_append_scalar_string_(&String out, const char* s) {
    if (!yaml_scalar_needs_quoting_(s)) {
        unsafe { out.push(s); }
        return;
    }
    // Single-quoted form: only escaping rule is '' for a literal quote —
    // simpler than double-quoted, and sufficient since we're only quoting
    // for YAML's syntactic reasons above, not to represent control chars
    // (a control-char-bearing string is a corner case this writer doesn't
    // specially handle, matching the writer's overall "common case" scope).
    unsafe { out.push_char('\''); }
    unsigned long i = 0UL;
    unsigned long run_start = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '\'') {
            if (i > run_start) { unsafe { out.push_n_(s + run_start, i - run_start); } }
            unsafe { out.push("''"); }
            i = i + 1UL;
            run_start = i;
        } else {
            i = i + 1UL;
        }
    }
    if (i > run_start) { unsafe { out.push_n_(s + run_start, i - run_start); } }
    unsafe { out.push_char('\''); }
}

static void yaml_append_float_(&String out, double v) {
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

static void yaml_append_scalar_(&String out, const &Value v) {
    int kind;
    unsafe { kind = v.kind; }
    if (kind == VAL_NULL) { unsafe { out.push("null"); } }
    else if (kind == VAL_BOOL) {
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
        yaml_append_float_(out, f);
    } else if (kind == VAL_STRING) {
        const char* s;
        unsafe {
            if (v.str_val != (char*)0) { s = (const char*)v.str_val; }
            else { s = (const char*)""; }
        }
        yaml_append_scalar_string_(out, s);
    }
}

static void yaml_write_block_(const &Value v, &String out, int depth);

static void yaml_write_array_(const &Value v, &String out, int depth) {
    unsigned long n;
    unsafe { n = v.arr_val.length(); }
    if (n == 0UL) { unsafe { out.push("[]\n"); } return; }
    unsigned long i = 0UL;
    while (i < n) {
        yaml_indent_(out, depth);
        unsafe { out.push_char('-'); }
        &Value elem;
        unsafe { elem = v.arr_val.at(i); }
        int ekind;
        unsafe { ekind = elem->kind; }
        if (ekind == VAL_ARRAY || ekind == VAL_OBJECT) {
            unsafe { out.push_char('\n'); }
            yaml_write_block_(elem, out, depth + 1);
        } else {
            unsafe { out.push_char(' '); }
            yaml_append_scalar_(out, elem);
            unsafe { out.push_char('\n'); }
        }
        i = i + 1UL;
    }
}

static void yaml_write_object_(const &Value v, &String out, int depth) {
    unsigned long n;
    unsafe { n = v.obj_val.length(); }
    if (n == 0UL) { unsafe { out.push("{}\n"); } return; }
    unsigned long i = 0UL;
    while (i < n) {
        yaml_indent_(out, depth);
        struct ObjectEntry* e;
        unsafe { e = v.obj_val.at(i); }
        const char* key;
        unsafe { key = (const char*)e->key; }
        yaml_append_scalar_string_(out, key);
        unsafe { out.push_char(':'); }
        &Value val;
        unsafe { val = e->val; }
        int vkind;
        unsafe { vkind = val->kind; }
        if (vkind == VAL_ARRAY || vkind == VAL_OBJECT) {
            unsafe { out.push_char('\n'); }
            yaml_write_block_(val, out, depth + 1);
        } else {
            unsafe { out.push_char(' '); }
            yaml_append_scalar_(out, val);
            unsafe { out.push_char('\n'); }
        }
        i = i + 1UL;
    }
}

static void yaml_write_block_(const &Value v, &String out, int depth) {
    int kind;
    unsafe { kind = v.kind; }
    if (kind == VAL_ARRAY) { yaml_write_array_(v, out, depth); }
    else if (kind == VAL_OBJECT) { yaml_write_object_(v, out, depth); }
    else { yaml_append_scalar_(out, v); unsafe { out.push_char('\n'); } }
}

void yaml_write(const &Value v, &String out) {
    yaml_write_block_(v, out, 0);
}

inline struct String value_to_yaml(const &Value v) {
    struct String out = String::new();
    yaml_write(v, &out);
    return out;
}

// ── Parser ────────────────────────────────────────────────────────────────

struct YamlLine_ {
    unsigned long indent;
    struct String content; // comment-stripped, trailing-whitespace-trimmed
    const char*   doc;     // the whole document (block scalars re-read raw lines)
    unsigned long end;     // offset just past this line's '\n' in 'doc'
};

// Strips a trailing '#...' comment (only when the '#' is outside any
// quoted scalar on the line) and trailing whitespace.
static struct String yaml_strip_comment_(const char* line) {
    struct String out = String::new();
    unsigned long i = 0UL;
    char quote = (char)0; // 0 = not in a quote, else the active quote char
    while (1) {
        char c;
        unsafe { c = line[i]; }
        if (c == (char)0) { break; }
        if (quote != (char)0) {
            unsafe { out.push_char(c); }
            if (c == quote) { quote = (char)0; }
            i = i + 1UL;
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            unsafe { out.push_char(c); }
            i = i + 1UL;
            continue;
        }
        if (c == '#') {
            // A comment starts at '#' only at the line start or after
            // whitespace ('a#b' is a plain scalar).
            char prev = (char)0;
            unsafe { if (i > 0UL) { prev = line[i - 1UL]; } }
            if (i == 0UL || prev == ' ' || prev == '\t') { break; }
        }
        unsafe { out.push_char(c); }
        i = i + 1UL;
    }
    struct String trimmed = out.trim_right();
    out.free();
    return trimmed;
}

// Splits 'text' into non-blank, comment-stripped lines with their leading-
// space indent measured. Tabs are not accepted as indentation (matching
// the YAML spec, which forbids tabs for indent) — a tab in leading
// whitespace ends the indent count at that point, generally producing an
// indent that won't line up with anything and surfacing as a structural
// mismatch rather than being silently misinterpreted.
static struct Vec<struct YamlLine_> yaml_split_lines_(const char* text) {
    struct Vec<struct YamlLine_> lines = Vec::new();
    unsigned long i = 0UL;
    while (1) {
        char c0;
        unsafe { c0 = text[i]; }
        if (c0 == (char)0) { break; }
        unsigned long lineStart = i;
        while (1) {
            char c;
            unsafe { c = text[i]; }
            if (c == (char)0 || c == '\n') { break; }
            i = i + 1UL;
        }
        unsigned long lineEnd = i;
        char c;
        unsafe { c = text[i]; }
        if (c == '\n') { i = i + 1UL; }

        unsigned long indent = 0UL;
        unsigned long p = lineStart;
        while (p < lineEnd) {
            char sc;
            unsafe { sc = text[p]; }
            if (sc != ' ') { break; }
            indent = indent + 1UL;
            p = p + 1UL;
        }
        struct String rawLine;
        unsafe { rawLine = String::from_n_(text + p, lineEnd - p); }
        struct String content = yaml_strip_comment_(rawLine.as_ptr());
        rawLine.free();
        if (!content.is_empty()) {
            struct YamlLine_ yl;
            yl.indent = indent;
            yl.content = content;
            yl.doc = text;
            yl.end = i;
            lines.push(yl);
        } else {
            content.free();
        }
    }
    return lines;
}

static void yaml_free_lines_(&Vec<struct YamlLine_> lines) {
    unsigned long n;
    unsafe { n = lines->length(); }
    unsigned long i = 0UL;
    while (i < n) {
        struct YamlLine_* yl;
        unsafe { yl = lines->at(i); }
        unsafe { yl->content.free(); }
        i = i + 1UL;
    }
    unsafe { lines->free(); }
}

static struct Value yaml_parse_flow_value_(const char* t);

static struct Value yaml_parse_scalar_(const char* text) {
    struct String s; unsafe { s = String::from(text); }
    struct String trimmed = s.trim();
    s.free();
    const char* t = trimmed.as_ptr();
    char c0;
    unsafe { c0 = t[0]; }

    struct Value result;
    int handled = 1;
    if (trimmed.is_empty() || trimmed.eq_cstr("~") || trimmed.eq_cstr_ignore_case("null")) {
        result = Value::null();
    } else if (trimmed.eq_cstr_ignore_case("true")) {
        result = Value::from_bool(1);
    } else if (trimmed.eq_cstr_ignore_case("false")) {
        result = Value::from_bool(0);
    } else if (c0 == '[' || c0 == '{') {
        result = yaml_parse_flow_value_(t);
    } else if (c0 == '"') {
        struct String out = String::new();
        unsigned long i = 1UL;
        while (1) {
            char c;
            unsafe { c = t[i]; }
            if (c == (char)0 || c == '"') { break; }
            if (c == '\\') {
                char esc;
                unsafe { esc = t[i + 1UL]; }
                if (esc == 'n') { unsafe { out.push_char('\n'); } }
                else if (esc == 't') { unsafe { out.push_char('\t'); } }
                else if (esc == 'r') { unsafe { out.push_char('\r'); } }
                else if (esc == '"') { unsafe { out.push_char('"'); } }
                else if (esc == '\\') { unsafe { out.push_char('\\'); } }
                else { unsafe { out.push_char(esc); } }
                i = i + 2UL;
            } else {
                unsafe { out.push_char(c); }
                i = i + 1UL;
            }
        }
        result = Value::from_str(out.as_ptr());
        out.free();
    } else if (c0 == '\'') {
        struct String out = String::new();
        unsigned long i = 1UL;
        while (1) {
            char c;
            unsafe { c = t[i]; }
            if (c == (char)0) { break; }
            if (c == '\'') {
                char next;
                unsafe { next = t[i + 1UL]; }
                if (next == '\'') { unsafe { out.push_char('\''); } i = i + 2UL; continue; }
                break;
            }
            unsafe { out.push_char(c); }
            i = i + 1UL;
        }
        result = Value::from_str(out.as_ptr());
        out.free();
    } else {
        // Plain scalar: number if it looks like one, else a bare string.
        int isNumeric = (c0 == '-' || (c0 >= '0' && c0 <= '9'));
        int isFloat = 0;
        if (isNumeric) {
            unsigned long i = (c0 == '-') ? 1UL : 0UL;
            unsigned long len = trimmed.length();
            if (i >= len) { isNumeric = 0; }
            while (i < len) {
                char c;
                unsafe { c = t[i]; }
                if (c >= '0' && c <= '9') { i = i + 1UL; continue; }
                if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    isFloat = 1; i = i + 1UL; continue;
                }
                isNumeric = 0;
                break;
            }
        }
        if (isNumeric) {
            int pok = 0;
            if (isFloat) { result = Value::from_float(trimmed.parse_float(&pok)); }
            else { result = Value::from_int(trimmed.parse_int(&pok)); }
            unsafe { if (pok == 0) { result = Value::from_str(t); } }
        } else {
            unsafe { result = Value::from_str(t); }
        }
    }
    (void)handled;
    trimmed.free();
    return result;
}

// ── Flow collections ('[a, b]', '{k: v}') ───────────────────────────────────
// One line only (a flow collection spanning lines isn't supported). Items
// are flow scalars — quoted, or plain text up to the next ',' ']' '}' —
// or nested flow collections.

static struct Value yaml_parse_flow_(const char* s, unsigned long* i, int* ok);

static void yaml_flow_ws_(const char* s, unsigned long* i) {
    unsafe { while (s[*i] == ' ' || s[*i] == '\t') { *i = *i + 1UL; } }
}

// A flow scalar starting at s[*i]; 'is_key' also stops a plain scalar at
// ': ' (or ':' before ',', '}' or the end).
static struct Value yaml_parse_flow_scalar_(const char* s, unsigned long* i, int is_key, int* ok) {
    unsigned long start;
    unsigned long end;
    unsafe {
        start = *i;
        char q = s[start];
        if (q == '"' || q == '\'') {
            unsigned long k = start + 1UL;
            while (s[k] != (char)0) {
                if (q == '"' && s[k] == '\\' && s[k + 1UL] != (char)0) { k = k + 2UL; continue; }
                if (s[k] == q) {
                    if (q == '\'' && s[k + 1UL] == '\'') { k = k + 2UL; continue; }
                    break;
                }
                k = k + 1UL;
            }
            if (s[k] != q) { *ok = 0; return Value::null(); }
            end = k + 1UL;
        } else {
            unsigned long k = start;
            while (s[k] != (char)0 && s[k] != ',' && s[k] != ']' && s[k] != '}') {
                if (is_key && s[k] == ':' &&
                    (s[k + 1UL] == ' ' || s[k + 1UL] == ',' || s[k + 1UL] == '}' || s[k + 1UL] == (char)0)) { break; }
                k = k + 1UL;
            }
            end = k;
        }
        *i = end;
        struct String piece = String::from_n_(s + start, end - start);
        struct Value v = yaml_parse_scalar_(piece.as_ptr());
        piece.free();
        return v;
    }
}

static struct Value yaml_parse_flow_(const char* s, unsigned long* i, int* ok) {
    yaml_flow_ws_(s, i);
    char c;
    unsafe { c = s[*i]; }
    if (c == '[') {
        struct Value arr = Value::array();
        unsafe { *i = *i + 1UL; }
        for (;;) {
            yaml_flow_ws_(s, i);
            unsafe { c = s[*i]; }
            if (c == ']') { unsafe { *i = *i + 1UL; } return arr; }
            if (c == (char)0) { unsafe { *ok = 0; } return arr; }
            arr.array_push(yaml_parse_flow_(s, i, ok));
            int good; unsafe { good = *ok; }
            if (!good) { return arr; }
            yaml_flow_ws_(s, i);
            unsafe { c = s[*i]; }
            if (c == ',') { unsafe { *i = *i + 1UL; } }
            else if (c != ']') { unsafe { *ok = 0; } return arr; }
        }
    }
    if (c == '{') {
        struct Value obj = Value::object();
        unsafe { *i = *i + 1UL; }
        for (;;) {
            yaml_flow_ws_(s, i);
            unsafe { c = s[*i]; }
            if (c == '}') { unsafe { *i = *i + 1UL; } return obj; }
            if (c == (char)0) { unsafe { *ok = 0; } return obj; }
            struct Value key = yaml_parse_flow_scalar_(s, i, 1, ok);
            struct String keyStr = String::new();
            if (key.kind_of() == VAL_STRING) { keyStr.push(key.as_string()); }
            else if (key.kind_of() == VAL_INT) { keyStr.push_int(key.as_int()); }
            else if (key.kind_of() == VAL_BOOL) { keyStr.push(key.as_bool() ? "true" : "false"); }
            key.free();
            yaml_flow_ws_(s, i);
            unsafe { c = s[*i]; }
            struct Value val;
            if (c == ':') {
                unsafe { *i = *i + 1UL; }
                yaml_flow_ws_(s, i);
                unsafe { c = s[*i]; }
                if (c == ',' || c == '}') { val = Value::null(); }
                else { val = yaml_parse_flow_(s, i, ok); }
            } else {
                val = Value::null();   // '{a, b}': keys with null values
            }
            obj.object_set(keyStr.as_ptr(), val);
            keyStr.free();
            int good; unsafe { good = *ok; }
            if (!good) { return obj; }
            yaml_flow_ws_(s, i);
            unsafe { c = s[*i]; }
            if (c == ',') { unsafe { *i = *i + 1UL; } }
            else if (c != '}') { unsafe { *ok = 0; } return obj; }
        }
    }
    return yaml_parse_flow_scalar_(s, i, 0, ok);
}

// A whole scalar position holding '[...]' / '{...}'. Text that doesn't
// parse as one flow collection stays a plain string (least surprise).
static struct Value yaml_parse_flow_value_(const char* t) {
    unsigned long i = 0UL;
    int ok = 1;
    struct Value v = yaml_parse_flow_(t, &i, &ok);
    yaml_flow_ws_(t, &i);
    char rest;
    unsafe { rest = t[i]; }
    if (!ok || rest != (char)0) {
        v.free();
        unsafe { return Value::from_str(t); }
    }
    return v;
}

// ── Block scalars ('|' literal, '>' folded) ─────────────────────────────────

// 1 if 'h' is a block scalar header: '|' or '>' plus an optional
// chomping indicator ('-' strip, '+' keep) and/or indentation digit.
static int yaml_is_block_header_(const &String h) {
    int c0 = h.char_at(0UL);
    if (c0 != (int)'|' && c0 != (int)'>') { return 0; }
    for (unsigned long k = 1UL; k < h.len; k = k + 1UL) {
        int c = h.char_at(k);
        if (c != (int)'-' && c != (int)'+' && !(c >= (int)'1' && c <= (int)'9')) { return 0; }
    }
    return 1;
}

// Reads the block scalar whose header ends the line lines[*idx - 1]; its
// content is every following raw line indented more than 'parentIndent'
// (blank lines included). Advances *idx past the lines it consumed.
static struct Value yaml_block_scalar_(&Vec<struct YamlLine_> lines, unsigned long* idx,
                                       const &String header, unsigned long parentIndent) {
    struct YamlLine_* hl;
    unsafe { hl = lines->at(*idx - 1UL); }
    const char* doc;
    unsigned long pos;
    unsafe { doc = hl->doc; pos = hl->end; }
    int folded = header.char_at(0UL) == (int)'>';
    int chomp = header.contains("-") ? -1 : (header.contains("+") ? 1 : 0);
    unsigned long blockIndent = 0UL;
    struct String out = String::new();
    struct String pendingBreaks = String::new();   // newlines held until more content (or chomping)
    int any = 0;
    unsigned long endPos = pos;
    unsafe {
        while (doc[pos] != (char)0) {
            unsigned long ls = pos;
            unsigned long ind = 0UL;
            while (doc[pos] == ' ') { ind = ind + 1UL; pos = pos + 1UL; }
            unsigned long cs = pos;
            while (doc[pos] != (char)0 && doc[pos] != '\n') { pos = pos + 1UL; }
            unsigned long ce = pos;
            if (ce > cs && doc[ce - 1UL] == '\r') { ce = ce - 1UL; }
            if (doc[pos] == '\n') { pos = pos + 1UL; }
            int blank = ce == cs;
            if (!blank && ind <= parentIndent) { pos = ls; break; }
            if (!blank && blockIndent == 0UL) { blockIndent = ind; }
            endPos = pos;
            if (blank) { pendingBreaks.push_char('\n'); continue; }
            unsigned long from = ls + (blockIndent < ind ? blockIndent : ind);
            if (any) {
                if (folded && pendingBreaks.len == 0UL && ind == blockIndent) { out.push_char(' '); }
                else if (folded) { out.push_str(&pendingBreaks); if (ind != blockIndent) { out.push_char('\n'); } }
                else { out.push_char('\n'); out.push_str(&pendingBreaks); }
            } else {
                out.push_str(&pendingBreaks);
            }
            pendingBreaks.clear();
            out.push_n_(doc + from, ce - from);
            any = 1;
        }
    }
    if (any) {
        if (chomp == 0) { out.push_char('\n'); }
        else if (chomp == 1) { out.push_char('\n'); out.push_str(&pendingBreaks); }
    }
    pendingBreaks.free();
    // Skip the parsed lines that belong to the block.
    unsigned long n;
    unsafe { n = lines->length(); }
    while (1) {
        unsigned long cur;
        unsafe { cur = *idx; }
        if (cur >= n) { break; }
        struct YamlLine_* yl;
        unsafe { yl = lines->at(cur); }
        unsigned long lineEnd;
        unsafe { lineEnd = yl->end; }
        if (lineEnd > endPos) { break; }
        unsafe { *idx = cur + 1UL; }
    }
    struct Value v = Value::from_str(out.as_ptr());
    out.free();
    return v;
}

static struct Value yaml_parse_block_(&Vec<struct YamlLine_> lines, unsigned long* idx,
                                       unsigned long minIndent, int* ok);

// Splits "key: value" / "key:" at the first unquoted ':' followed by a
// space or end-of-line (a ':' inside a scalar value like a URL or time-of-
// day is not a key separator unless followed by whitespace/EOL, per the
// YAML spec's own rule for plain scalars).
static long long yaml_find_colon_(const char* s) {
    unsigned long i = 0UL;
    char quote = (char)0;
    int depth = 0;   // inside a flow collection: its ':'s aren't ours
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { return -1; }
        if (quote != (char)0) {
            if (c == quote) { quote = (char)0; }
            i = i + 1UL;
            continue;
        }
        if (c == '\'' || c == '"') { quote = c; i = i + 1UL; continue; }
        if (c == '[' || c == '{') { depth = depth + 1; }
        else if ((c == ']' || c == '}') && depth > 0) { depth = depth - 1; }
        if (c == ':' && depth == 0) {
            char next;
            unsafe { next = s[i + 1UL]; }
            if (next == (char)0 || next == ' ') { return (long long)i; }
        }
        i = i + 1UL;
    }
}

static struct Value yaml_parse_mapping_(&Vec<struct YamlLine_> lines, unsigned long* idx,
                                         unsigned long indent, int* ok) {
    struct Value obj = Value::object();
    unsigned long n;
    unsafe { n = lines->length(); }
    while (1) {
        unsigned long curIdx;
        unsafe { curIdx = *idx; }
        if (curIdx >= n) { break; }
        struct YamlLine_* yl;
        unsafe { yl = lines->at(curIdx); }
        unsigned long ylIndent;
        unsafe { ylIndent = yl->indent; }
        if (ylIndent != indent) { break; }
        const char* content;
        unsafe { content = yl->content.as_ptr(); }
        long long colonPos = yaml_find_colon_(content);
        if (colonPos < 0) { unsafe { *ok = 0; } break; }
        struct String keyRaw;
        unsafe { keyRaw = String::from_n_(content, (unsigned long)colonPos); }
        struct Value keyVal = yaml_parse_scalar_(keyRaw.as_ptr());
        keyRaw.free();
        // Only string-shaped keys make sense as object keys here (a
        // numeric-looking key stays a string key — same 'stringly-keyed'
        // convention std::Value's VAL_OBJECT already uses for JSON).
        const char* key = keyVal.as_string();
        struct String keyStr; unsafe { keyStr = key ? String::from(key) : String::new(); }
        keyVal.free();

        unsigned long restStart = (unsigned long)colonPos + 1UL;
        unsigned long contentLen;
        unsafe { contentLen = yl->content.length(); }
        struct String restRaw;
        unsafe {
            restRaw = (restStart <= contentLen)
                ? yl->content.substr(restStart, contentLen)
                : String::new();
        }
        struct String restTrimmed = restRaw.trim();
        restRaw.free();

        unsafe { *idx = *idx + 1UL; }
        if (restTrimmed.is_empty()) {
            struct Value child = yaml_parse_block_(lines, idx, indent + 1UL, ok);
            unsafe { obj.object_set(keyStr.as_ptr(), child); }
        } else if (yaml_is_block_header_(&restTrimmed)) {
            struct Value child = yaml_block_scalar_(lines, idx, &restTrimmed, indent);
            unsafe { obj.object_set(keyStr.as_ptr(), child); }
        } else {
            struct Value scalar = yaml_parse_scalar_(restTrimmed.as_ptr());
            unsafe { obj.object_set(keyStr.as_ptr(), scalar); }
        }
        restTrimmed.free();
        keyStr.free();
        int stillOk;
        unsafe { stillOk = *ok; }
        if (stillOk == 0) { break; }
    }
    return obj;
}

static struct Value yaml_parse_sequence_(&Vec<struct YamlLine_> lines, unsigned long* idx,
                                          unsigned long indent, int* ok) {
    struct Value arr = Value::array();
    unsigned long n;
    unsafe { n = lines->length(); }
    while (1) {
        unsigned long curIdx;
        unsafe { curIdx = *idx; }
        if (curIdx >= n) { break; }
        struct YamlLine_* yl;
        unsafe { yl = lines->at(curIdx); }
        unsigned long ylIndent;
        unsafe { ylIndent = yl->indent; }
        if (ylIndent != indent) { break; }
        const char* content;
        unsafe { content = yl->content.as_ptr(); }
        char c0;
        unsafe { c0 = content[0]; }
        if (c0 != '-') { break; }
        char c1;
        unsafe { c1 = content[1]; }
        if (c1 != (char)0 && c1 != ' ') { break; } // e.g. "-5" is a scalar, not a list item

        struct String restRaw;
        unsafe { restRaw = yl->content.substr(1UL, yl->content.length()); }
        struct String rest = restRaw.trim_left();
        restRaw.free();

        if (rest.is_empty()) {
            unsafe { *idx = *idx + 1UL; }
            struct Value child = yaml_parse_block_(lines, idx, indent + 1UL, ok);
            arr.array_push(child);
        } else if (yaml_is_block_header_(&rest)) {
            unsafe { *idx = *idx + 1UL; }
            arr.array_push(yaml_block_scalar_(lines, idx, &rest, indent));
        } else if (yaml_find_colon_(rest.as_ptr()) >= 0) {
            // "- key: value" compact list-of-mappings shorthand: rewrite
            // this line in place as a mapping key at the column right
            // after "- ", then let yaml_parse_mapping_ consume it plus any
            // further real lines indented to match (see yaml.h's comment).
            unsigned long dashIndent;
            unsafe { dashIndent = yl->indent; }
            unsafe {
                yl->content.free();
                yl->content = rest.clone();
                yl->indent  = dashIndent + 2UL;
            }
            struct Value child = yaml_parse_mapping_(lines, idx, dashIndent + 2UL, ok);
            arr.array_push(child);
        } else {
            struct Value scalar = yaml_parse_scalar_(rest.as_ptr());
            arr.array_push(scalar);
            unsafe { *idx = *idx + 1UL; }
        }
        rest.free();
        int stillOk;
        unsafe { stillOk = *ok; }
        if (stillOk == 0) { break; }
    }
    return arr;
}

static struct Value yaml_parse_block_(&Vec<struct YamlLine_> lines, unsigned long* idx,
                                       unsigned long minIndent, int* ok) {
    unsigned long n;
    unsafe { n = lines->length(); }
    unsigned long curIdx;
    unsafe { curIdx = *idx; }
    if (curIdx >= n) { return Value::null(); }
    struct YamlLine_* yl;
    unsafe { yl = lines->at(curIdx); }
    unsigned long ylIndent;
    unsafe { ylIndent = yl->indent; }
    if (ylIndent < minIndent) { return Value::null(); }
    unsigned long indent = ylIndent;
    const char* content;
    unsafe { content = yl->content.as_ptr(); }
    char c0;
    unsafe { c0 = content[0]; }
    char c1;
    unsafe { c1 = content[1]; }
    if (c0 == '-' && (c1 == (char)0 || c1 == ' ')) {
        return yaml_parse_sequence_(lines, idx, indent, ok);
    }
    if (yaml_find_colon_(content) >= 0) {
        return yaml_parse_mapping_(lines, idx, indent, ok);
    }
    // A bare scalar as the whole document/block value.
    struct Value v = yaml_parse_scalar_(content);
    unsafe { *idx = *idx + 1UL; }
    return v;
}

struct Value yaml_parse(const &char text, ?&int ok) {
    struct Vec<struct YamlLine_> lines = yaml_split_lines_(text);
    unsigned long idx = 0UL;
    int good = 1;
    struct Value result = yaml_parse_block_(&lines, &idx, 0UL, &good);
    if (good != 0 && idx != lines.length()) {
        // Leftover lines the block parser couldn't attach anywhere (e.g. a
        // dedent that doesn't match any enclosing indent level).
        good = 0;
    }
    yaml_free_lines_(&lines);
    if (ok != (int*)0) { unsafe { *ok = good; } }
    if (good == 0) {
        result.free();
        return Value::null();
    }
    return result;
}

struct Yaml Yaml::new() {
    struct Yaml f;
    f.unused_ = 0;
    return f;
}

void Yaml::write(const &Value v, &String out) const { yaml_write(v, out); }

struct Value Yaml::parse(const &char text, ?&int ok) const { return yaml_parse(text, ok); }

} // namespace std
