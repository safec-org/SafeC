// safec-doc — build-time syntax highlighting (shared with the SafeC docs
// site generator, safec-docs/site/src/highlight.sc).
//
// A small table-driven lexer: per language, a keyword list, a type-ish
// word list, and which comment / string syntaxes apply. Output is
// HTML-escaped code with <span class="tok-…"> runs styled by
// theme/style.css (tok-kw, tok-ty, tok-str, tok-num, tok-com, tok-pp,
// tok-fn). Unknown languages are emitted escaped, unhighlighted.
//
// Fences in the docs today: c (SafeC — ~1800 blocks), bash, toml, cpp,
// rust, zig, js, powershell, text.
#include "doctool.h"
#include <std/ctype.h>
#include <std/str.h>

struct LangSpec {
    const char* name;
    const char* keywords;      // space-separated, with leading/trailing space
    const char* types;
    int line_comment_slash;    // //
    int line_comment_hash;     // #
    int block_comment;         // /* */
    int preprocessor;          // # at line start is a directive (C family)
    int single_quote_str;      // '…' is a string (not a char literal)
    int plain;                 // no highlighting (unknown language)
};

// SafeC = C keywords + SafeC's own (region qualifiers, generics, match, …).
static const char* KW_SAFEC() {
    return    " if else for while do switch case default break continue return goto sizeof typedef"
    " struct union enum extern static const volatile inline register restrict auto"
    " region arena new unsafe match generic consteval defer errdefer spawn pure must_use"
    " noreturn namespace using trait impl fn some null none self Self true false"
    " static_assert align packed naked interrupt asm try catch where operator stack heap"
    " _Alignas _Alignof _Atomic _Bool _Generic _Noreturn _Static_assert _Thread_local ";
}
static const char* TY_C() {
    return    " void char short int long float double signed unsigned bool size_t ssize_t"
    " int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t uintptr_t intptr_t"
    " float16 bfloat16 String Vec HashMap StrMap Value Result Option ";
}
static const char* KW_CPP() {
    return    " if else for while do switch case default break continue return goto sizeof typedef"
    " struct union enum extern static const volatile inline class public private protected"
    " virtual override template typename namespace using new delete this nullptr true false"
    " constexpr auto noexcept try catch throw operator friend explicit mutable decltype ";
}
static const char* KW_RUST() {
    return    " as break const continue crate else enum extern false fn for if impl in let loop match"
    " mod move mut pub ref return self Self static struct super trait true type unsafe use"
    " where while async await dyn ";
}
static const char* TY_RUST() {
    return    " i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str String Vec"
    " Option Result Box ";
}
static const char* KW_ZIG() {
    return    " const var fn pub return if else while for switch break continue defer errdefer try"
    " catch comptime struct enum union error test inline extern export unreachable null"
    " undefined true false orelse and or ";
}
static const char* TY_ZIG() {
    return    " u8 u16 u32 u64 usize i8 i16 i32 i64 isize f32 f64 bool void anyerror type ";
}
static const char* KW_JS() {
    return    " const let var function return if else for while do switch case default break"
    " continue new delete typeof instanceof in of class extends super this null undefined"
    " true false async await import export from try catch finally throw yield ";
}
static const char* KW_SH() {
    return    " if then else elif fi for while until do done case esac in function return export"
    " local set unset source echo cd exit sudo ";
}
static const char* KW_PS() {
    return    " if else elseif foreach for while function return param try catch finally throw ";
}
static const char* KW_TOML() {
    return " true false ";
}
static const char* NONE() {
    return " ";
}

static struct LangSpec spec_for(const &char lang) {
    struct LangSpec s;
    s.name = "text"; s.keywords = NONE(); s.types = NONE();
    s.line_comment_slash = 0; s.line_comment_hash = 0; s.block_comment = 0;
    s.preprocessor = 0; s.single_quote_str = 0; s.plain = 1;
    if (str_eq(lang, "c") || str_eq(lang, "safec") || str_eq(lang, "sc") || str_eq(lang, "h") || str_eq(lang, "scx")) {
        s.name = "c"; s.keywords = KW_SAFEC(); s.types = TY_C();
        s.line_comment_slash = 1; s.block_comment = 1; s.preprocessor = 1;
    } else if (str_eq(lang, "cpp") || str_eq(lang, "c++") || str_eq(lang, "hpp")) {
        s.name = "cpp"; s.keywords = KW_CPP(); s.types = TY_C();
        s.line_comment_slash = 1; s.block_comment = 1; s.preprocessor = 1;
    } else if (str_eq(lang, "rust") || str_eq(lang, "rs")) {
        s.name = "rust"; s.keywords = KW_RUST(); s.types = TY_RUST();
        s.line_comment_slash = 1; s.block_comment = 1;
    } else if (str_eq(lang, "zig")) {
        s.name = "zig"; s.keywords = KW_ZIG(); s.types = TY_ZIG(); s.line_comment_slash = 1;
    } else if (str_eq(lang, "js") || str_eq(lang, "javascript") || str_eq(lang, "ts")) {
        s.name = "js"; s.keywords = KW_JS(); s.line_comment_slash = 1; s.block_comment = 1;
        s.single_quote_str = 1;
    } else if (str_eq(lang, "bash") || str_eq(lang, "sh") || str_eq(lang, "shell")) {
        s.name = "bash"; s.keywords = KW_SH(); s.line_comment_hash = 1; s.single_quote_str = 1;
    } else if (str_eq(lang, "powershell") || str_eq(lang, "ps1")) {
        s.name = "powershell"; s.keywords = KW_PS(); s.line_comment_hash = 1; s.single_quote_str = 1;
    } else if (str_eq(lang, "toml")) {
        s.name = "toml"; s.keywords = KW_TOML(); s.line_comment_hash = 1; s.single_quote_str = 1;
    }
    unsafe { s.plain = str_eq(s.name, "text"); }
    return s;
}

static void esc_n(&String out, const char* s, unsigned long n) {
    for (unsigned long k = 0UL; k < n; k = k + 1UL) {
        char c;
        unsafe { c = s[k]; }
        if (c == '&') { out.push("&amp;"); }
        else if (c == '<') { out.push("&lt;"); }
        else if (c == '>') { out.push("&gt;"); }
        else { out.push_char(c); }
    }
}

static void span(&String out, const char* cls, const char* s, unsigned long n) {
    out.push("<span class=\"");
    unsafe { out.push(cls); }
    out.push("\">");
    esc_n(out, s, n);
    out.push("</span>");
}

// Is s[0..n) one of the space-separated words in 'list'?
static int in_list(const char* list, const char* s, unsigned long n) {
    char word[64];
    if (n == 0UL || n + 3UL > 64UL) { return 0; }
    word[0] = ' ';
    for (unsigned long k = 0UL; k < n; k = k + 1UL) { unsafe { word[k + 1UL] = s[k]; } }
    word[n + 1UL] = ' ';
    word[n + 2UL] = '\0';
    const ?&char hit;
    unsafe { hit = str_find(list, &word[0]); }
    match (hit) { case some(h): { return 1; } case null: { return 0; } }
    return 0;
}

static int ident_start(char c) { return char_is_alpha((int)c) || c == '_' || ((unsigned char)c) >= 0x80; }
static int ident_char(char c) { return char_is_alnum((int)c) || c == '_' || ((unsigned char)c) >= 0x80; }

void highlight_code(const &char lang, const &char code, unsigned long len, &String out) {
    struct LangSpec L = spec_for(lang);
    const char* s;
    unsafe { s = code; }
    int line_start = 1;
    unsigned long i = 0UL;
    // Raw-pointer scanning over the fenced block's bytes ('code' is a
    // borrowed view into the page source, valid for this whole call).
    unsafe {
    while (i < len) {
        char c;
        unsafe { c = s[i]; }
        char d = '\0';
        unsafe { if (i + 1UL < len) { d = s[i + 1UL]; } }
        unsigned long j = i;

        if (c == '\n') { out.push_char('\n'); line_start = 1; i = i + 1UL; continue; }
        if (c == ' ' || c == '\t') { out.push_char(c); i = i + 1UL; continue; }

        // Comments.
        if ((L.line_comment_slash && c == '/' && d == '/') || (L.line_comment_hash && c == '#')) {
            unsafe { while (j < len && s[j] != '\n') { j = j + 1UL; } }
            span(out, "tok-com", s + i, j - i);
            i = j; line_start = 0; continue;
        }
        if (L.block_comment && c == '/' && d == '*') {
            j = i + 2UL;
            unsafe { while (j + 1UL < len && !(s[j] == '*' && s[j + 1UL] == '/')) { j = j + 1UL; } }
            j = j + 2UL;
            if (j > len) { j = len; }
            span(out, "tok-com", s + i, j - i);
            i = j; line_start = 0; continue;
        }
        // Preprocessor directive: '#word' at line start (C family).
        if (L.preprocessor && c == '#' && line_start) {
            j = i + 1UL;
            unsafe {
                while (j < len && s[j] == ' ') { j = j + 1UL; }
                while (j < len && char_is_alpha((int)s[j])) { j = j + 1UL; }
            }
            span(out, "tok-pp", s + i, j - i);
            i = j; line_start = 0; continue;
        }
        // Strings / char literals.
        if (c == '"' || (c == '\'' && (L.single_quote_str || L.preprocessor))) {
            j = i + 1UL;
            unsafe {
                while (j < len && s[j] != c && s[j] != '\n') {
                    if (s[j] == '\\' && j + 1UL < len) { j = j + 1UL; }
                    j = j + 1UL;
                }
            }
            if (j < len) { j = j + 1UL; }
            span(out, "tok-str", s + i, j - i);
            i = j; line_start = 0; continue;
        }
        // Numbers.
        if (char_is_digit((int)c)) {
            unsafe { while (j < len && (char_is_alnum((int)s[j]) || s[j] == '.' || s[j] == '_')) { j = j + 1UL; } }
            span(out, "tok-num", s + i, j - i);
            i = j; line_start = 0; continue;
        }
        // Words.
        if (ident_start(c)) {
            unsafe { while (j < len && ident_char(s[j])) { j = j + 1UL; } }
            unsigned long n = j - i;
            char next = '\0';
            unsafe { if (j < len) { next = s[j]; } }
            if (in_list(L.keywords, s + i, n)) { span(out, "tok-kw", s + i, n); }
            else if (in_list(L.types, s + i, n)) { span(out, "tok-ty", s + i, n); }
            else if (next == '(' && !L.plain) { span(out, "tok-fn", s + i, n); }
            else { esc_n(out, s + i, n); }
            i = j; line_start = 0; continue;
        }
        esc_n(out, s + i, 1UL);
        i = i + 1UL;
        line_start = 0;
    }
    }
}
