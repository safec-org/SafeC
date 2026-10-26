// safec-doc — small helpers over std (see doctool.h).
#include "doctool.h"
#include <std/io_file.h>
#include <std/dir.h>
#include <std/path.h>

int util_write_file(const &char path, const &String data) {
    struct String dir = path_dirname(path);
    if (dir.len > 0UL) { dir_create_all(dir.as_ptr()); }
    dir.free();
    return File::write_bytes(path, data.as_bytes());
}

static void push_esc_char(&String out, char c) {
    if (c == '&') { out.push("&amp;"); }
    else if (c == '<') { out.push("&lt;"); }
    else if (c == '>') { out.push("&gt;"); }
    else if (c == '"') { out.push("&quot;"); }
    else if (c == '\'') { out.push("&#39;"); }
    else { out.push_char(c); }
}

void util_push_esc(&String out, const &char s) {
    unsigned long i = 0UL;
    unsafe {
        while (s[i] != '\0') {
            push_esc_char(out, s[i]);
            i = i + 1UL;
        }
    }
}

void util_push_esc_n(&String out, const &String s, unsigned long from, unsigned long to) {
    for (unsigned long i = from; i < to && i < s.len; i = i + 1UL) {
        push_esc_char(out, (char)s.char_at(i));
    }
}

const &char util_vstr(const &Value obj, const &char key, const &char fallback) {
    if (obj.kind_of() != VAL_OBJECT) { return fallback; }
    ?&Value v = obj.object_get(key);
    match (v) {
        case some(x): {
            if (x.kind_of() == VAL_STRING) { return x.as_string(); }
        }
        case null: { }
    }
    return fallback;
}

&Value util_vget(const &Value obj, const &char key, const &Value fallback) {
    if (obj.kind_of() == VAL_OBJECT) {
        ?&Value v = obj.object_get(key);
        match (v) {
            case some(x): { return x; }
            case null: { }
        }
    }
    return fallback;
}

long long util_vint(const &Value obj, const &char key, long long fallback) {
    if (obj.kind_of() != VAL_OBJECT) { return fallback; }
    ?&Value v = obj.object_get(key);
    match (v) {
        case some(x): {
            if (x.kind_of() == VAL_INT) { return x.as_int(); }
        }
        case null: { }
    }
    return fallback;
}

int util_vbool(const &Value obj, const &char key) {
    if (obj.kind_of() != VAL_OBJECT) { return 0; }
    ?&Value v = obj.object_get(key);
    match (v) {
        case some(x): {
            if (x.kind_of() == VAL_BOOL) { return x.as_bool(); }
        }
        case null: { }
    }
    return 0;
}

struct String util_root_prefix(const &String url) {
    struct String p = String::new();
    for (unsigned long i = 0UL; i < url.len; i = i + 1UL) {
        if (url.char_at(i) == (int)'/') { p.push("../"); }
    }
    return p;
}
