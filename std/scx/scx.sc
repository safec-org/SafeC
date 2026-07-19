// SafeC Standard Library — scx runtime support (see scx.h)
//
// Deliberately does NOT '#include <std/collections/string.sc>' (the String
// implementation) — scx_append_esc only needs String's declared shape
// (from scx.h -> string.h) to call .push()/.push_char() on a pointer;
// pulling in the implementation here too would make this file's own
// archive member (built as part of libsafec_std.a) redefine every String
// method a second time, which conflicts at link time with any project that
// also links in a String implementation directly (e.g. via std/http/http.sc,
// which does include string.sc) — the same duplicate-symbol trap documented
// in safeguard's ScxTranspiler.cpp.
#include <std/scx/scx.h>

namespace std {

void scx_append_esc(struct String* buf, const char* s) {
    unsigned long i = 0UL;
    unsafe {
        while (s[i] != '\0') {
            char c = s[i];
            if (c == '&') { buf->push("&amp;"); }
            else if (c == '<') { buf->push("&lt;"); }
            else if (c == '>') { buf->push("&gt;"); }
            else if (c == '"') { buf->push("&quot;"); }
            else if (c == '\'') { buf->push("&#39;"); }
            else { buf->push_char(c); }
            i = i + 1UL;
        }
    }
}

} // namespace std
