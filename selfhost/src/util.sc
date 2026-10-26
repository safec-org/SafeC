// util.sc — string interning, message building, diagnostics.
#include "scc.h"

unsigned char g_src[SRC_MAX];
int g_src_len = 0;
int g_file_name[NFILE_MAX];
int g_file_start[NFILE_MAX];
int g_file_len[NFILE_MAX];
int g_nfile = 0;

char g_sp[SP_MAX];
int g_sp_len = 0;
int g_str_off[NSTR_MAX];
int g_str_len[NSTR_MAX];
int g_str_next[NSTR_MAX];
int g_nstr = 0;
int g_strhash[STRHASH_SIZE];
int g_kw[NSTR_MAX];

unsigned char g_out[OUT_MAX];
int g_out_len = 0;
char g_diag[DIAG_MAX];
int g_diag_len = 0;
int g_nerrors = 0;
int g_nwarnings = 0;
char g_msg[MSG_MAX];
int g_msg_len = 0;
unsigned char g_tmp[TMP_MAX];
int g_tmp_len = 0;

int S_main = 0;
int S_self = 0;
int S_len = 0;
int S_ptr = 0;
int S_stack = 0;
int S_heap = 0;
int S_arena = 0;
int S_static = 0;
int S_some = 0;
int S_none = 0;
int S_is_null = 0;
int S_is_none = 0;
int S_default = 0;
int S_attribute = 0;
int S_export_name = 0;
int S_defined = 0;
int S_va_ptr = 0;
int S_capacity = 0;
int S_once = 0;
int S_include = 0;
int S_define = 0;
int S_undef = 0;
int S_if = 0;
int S_ifdef = 0;
int S_ifndef = 0;
int S_elif = 0;
int S_else = 0;
int S_endif = 0;
int S_pragma = 0;
int S_error = 0;
int S_warning = 0;
int S_line = 0;
int S_VA_ARGS = 0;
int S_FILE = 0;
int S_LINE = 0;
int S_runtime = 0;
int S_wild = 0;

int lit_len(const char* s) {
    int n = 0;
    unsafe { while (s[n] != (char)0) { n = n + 1; } }
    return n;
}

int lit_char(const char* s, int i) {
    int c = 0;
    unsafe { c = (int)(unsigned char)s[i]; }
    return c;
}

int g_fatal = 0;

void fatal(const char* s) {
    if (g_fatal == 1) { g_nerrors = g_nerrors + 1; return; }
    g_fatal = 1;
    msg_reset();
    msg_lit("internal limit exceeded: ");
    msg_lit(s);
    diag_at(0, 0, 0, 1);
}

int hash_bytes_tmp(int from_src, int off, int len) {
    unsigned int h = 2166136261U;
    int i = 0;
    while (i < len) {
        unsigned int c = 0U;
        if (from_src == 1) { c = (unsigned int)g_src[off + i]; }
        else if (from_src == 2) { c = (unsigned int)(unsigned char)g_sp[off + i]; }
        else { c = (unsigned int)g_tmp[off + i]; }
        h = (h ^ c) *| 16777619U;
        i = i + 1;
    }
    return (int)(h & (unsigned int)(STRHASH_SIZE - 1));
}

// from_src: 1 = g_src, 0 = g_tmp, 2 = g_sp
int intern_bytes(int from_src, int off, int len) {
    int h = hash_bytes_tmp(from_src, off, len);
    int s = g_strhash[h];
    while (s != 0) {
        if (g_str_len[s] == len) {
            int o = g_str_off[s];
            int i = 0;
            int same = 1;
            while (i < len) {
                int c = 0;
                if (from_src == 1) { c = (int)g_src[off + i]; }
                else if (from_src == 2) { c = (int)(unsigned char)g_sp[off + i]; }
                else { c = (int)g_tmp[off + i]; }
                if ((int)(unsigned char)g_sp[o + i] != c) { same = 0; break; }
                i = i + 1;
            }
            if (same == 1) { return s; }
        }
        s = g_str_next[s];
    }
    if (g_nstr + 1 >= NSTR_MAX || g_sp_len + len + 1 >= SP_MAX) {
        fatal("string table");
        return 0;
    }
    g_nstr = g_nstr + 1;
    s = g_nstr;
    int base = g_sp_len;
    int i = 0;
    while (i < len) {
        int c = 0;
        if (from_src == 1) { c = (int)g_src[off + i]; }
        else if (from_src == 2) { c = (int)(unsigned char)g_sp[off + i]; }
        else { c = (int)g_tmp[off + i]; }
        g_sp[base + i] = (char)c;
        i = i + 1;
    }
    g_sp[base + len] = (char)0;
    g_sp_len = base + len + 1;
    g_str_off[s] = base;
    g_str_len[s] = len;
    g_str_next[s] = g_strhash[h];
    g_strhash[h] = s;
    return s;
}

void tmp_reset() { g_tmp_len = 0; }

void tmp_char(int c) {
    if (g_tmp_len < TMP_MAX) {
        g_tmp[g_tmp_len] = (unsigned char)c;
        g_tmp_len = g_tmp_len + 1;
    }
}

void tmp_lit(const char* s) {
    int n = lit_len(s);
    int i = 0;
    while (i < n) { tmp_char(lit_char(s, i)); i = i + 1; }
}

void tmp_str(int sid) {
    int n = g_str_len[sid];
    int o = g_str_off[sid];
    int i = 0;
    while (i < n) { tmp_char((int)(unsigned char)g_sp[o + i]); i = i + 1; }
}

void tmp_int(long v) {
    char digits[24];
    int n = 0;
    if (v < 0L) { tmp_char('-'); v = 0L - v; }
    if (v == 0L) { tmp_char('0'); return; }
    while (v > 0L && n < 22) {
        digits[n] = (char)(48L + v % 10L);
        v = v / 10L;
        n = n + 1;
    }
    while (n > 0) { n = n - 1; tmp_char((int)digits[n]); }
}

int intern_tmp() { return intern_bytes(0, 0, g_tmp_len); }

int intern_lit(const char* s) {
    tmp_reset();
    tmp_lit(s);
    return intern_tmp();
}

int str_char(int sid, int i) {
    return (int)(unsigned char)g_sp[g_str_off[sid] + i];
}

int str_eq_lit(int sid, const char* s) {
    int n = lit_len(s);
    if (g_str_len[sid] != n) { return 0; }
    int i = 0;
    while (i < n) {
        if (str_char(sid, i) != lit_char(s, i)) { return 0; }
        i = i + 1;
    }
    return 1;
}

int str_concat3(int a, const char* mid, int b) {
    tmp_reset();
    tmp_str(a);
    tmp_lit(mid);
    tmp_str(b);
    return intern_tmp();
}

// ── message buffer ───────────────────────────────────────────────────────────
void msg_reset() { g_msg_len = 0; }

void msg_char(int c) {
    if (g_msg_len < MSG_MAX - 1) {
        g_msg[g_msg_len] = (char)c;
        g_msg_len = g_msg_len + 1;
    }
}

void msg_lit(const char* s) {
    int n = lit_len(s);
    int i = 0;
    while (i < n) { msg_char(lit_char(s, i)); i = i + 1; }
}

void msg_str(int sid) {
    int n = g_str_len[sid];
    int i = 0;
    while (i < n) { msg_char(str_char(sid, i)); i = i + 1; }
}

void msg_int(long v) {
    char digits[24];
    int n = 0;
    if (v < 0L) { msg_char('-'); v = 0L - v; }
    if (v == 0L) { msg_char('0'); return; }
    while (v > 0L && n < 22) {
        digits[n] = (char)(48L + v % 10L);
        v = v / 10L;
        n = n + 1;
    }
    while (n > 0) { n = n - 1; msg_char((int)digits[n]); }
}

// ── diagnostics ──────────────────────────────────────────────────────────────
void diag_char(int c) {
    if (g_diag_len < DIAG_MAX - 1) {
        g_diag[g_diag_len] = (char)c;
        g_diag_len = g_diag_len + 1;
    }
}

void diag_sid(int sid) {
    int n = g_str_len[sid];
    int i = 0;
    while (i < n) { diag_char(str_char(sid, i)); i = i + 1; }
}

void diag_lit(const char* s) {
    int n = lit_len(s);
    int i = 0;
    while (i < n) { diag_char(lit_char(s, i)); i = i + 1; }
}

void diag_num(long v) {
    char digits[24];
    int n = 0;
    if (v == 0L) { diag_char('0'); return; }
    while (v > 0L && n < 22) {
        digits[n] = (char)(48L + v % 10L);
        v = v / 10L;
        n = n + 1;
    }
    while (n > 0) { n = n - 1; diag_char((int)digits[n]); }
}

int g_diag_limit_hit = 0;

void diag_at(int file, int line, int col, int is_error) {
    if (is_error == 1) { g_nerrors = g_nerrors + 1; }
    else { g_nwarnings = g_nwarnings + 1; }
    if (g_nerrors + g_nwarnings > 200) {
        if (g_diag_limit_hit == 0) {
            g_diag_limit_hit = 1;
            diag_lit("error: too many diagnostics\n");
        }
        return;
    }
    if (file > 0 && file <= g_nfile) {
        diag_sid(g_file_name[file]);
        diag_char(':');
        diag_num((long)line);
        diag_char(':');
        diag_num((long)col);
        diag_lit(": ");
    }
    if (is_error == 1) { diag_lit("error: "); } else { diag_lit("warning: "); }
    int i = 0;
    while (i < g_msg_len) { diag_char((int)g_msg[i]); i = i + 1; }
    diag_char('\n');
}

void error_tok(int tok) {
    diag_at(g_tok[tok].file, g_tok[tok].line, g_tok[tok].col, 1);
}

void warn_tok(int tok) {
    diag_at(g_tok[tok].file, g_tok[tok].line, g_tok[tok].col, 0);
}

void error_node(int node) { error_tok(g_node[node].tok); }
void warn_node(int node) { warn_tok(g_node[node].tok); }

void out_byte(int b) {
    if (g_out_len < OUT_MAX) {
        g_out[g_out_len] = (unsigned char)b;
        g_out_len = g_out_len + 1;
    }
}

int ilog_hash(int a, int b) {
    unsigned int h = (unsigned int)a *| 2654435761U;
    h = (h ^ (unsigned int)b) *| 2246822519U;
    return (int)(h >> 7U);
}

void util_init() {
    S_main = intern_lit("main");
    S_self = intern_lit("self");
    S_len = intern_lit("len");
    S_ptr = intern_lit("ptr");
    S_stack = intern_lit("stack");
    S_heap = intern_lit("heap");
    S_arena = intern_lit("arena");
    S_static = intern_lit("static");
    S_some = intern_lit("some");
    S_none = intern_lit("none");
    S_is_null = intern_lit("is_null");
    S_is_none = intern_lit("is_none");
    S_default = intern_lit("default");
    S_attribute = intern_lit("__attribute__");
    S_export_name = intern_lit("export_name");
    S_defined = intern_lit("defined");
    S_va_ptr = intern_lit("__va_ptr");
    S_capacity = intern_lit("capacity");
    S_once = intern_lit("once");
    S_include = intern_lit("include");
    S_define = intern_lit("define");
    S_undef = intern_lit("undef");
    S_if = intern_lit("if");
    S_ifdef = intern_lit("ifdef");
    S_ifndef = intern_lit("ifndef");
    S_elif = intern_lit("elif");
    S_else = intern_lit("else");
    S_endif = intern_lit("endif");
    S_pragma = intern_lit("pragma");
    S_error = intern_lit("error");
    S_warning = intern_lit("warning");
    S_line = intern_lit("line");
    S_VA_ARGS = intern_lit("__VA_ARGS__");
    S_FILE = intern_lit("__FILE__");
    S_LINE = intern_lit("__LINE__");
    S_runtime = intern_lit("<runtime>");
    S_wild = intern_lit("_");
}

// debug tracing for the bootstrapped compiler (stage 2 under Node)
#ifdef SCC_TRACE
extern int __host_write(int fd, const char* buf, int len);
void dbg(const char* s) { unsafe { __host_write(2, s, lit_len(s)); } }
void dbg_int(long v) { tmp_reset(); tmp_int(v); tmp_char(10); unsafe { __host_write(2, (const char*)&g_tmp[0], g_tmp_len); } }
#else
void dbg(const char* s) { }
void dbg_int(long v) { }
#endif
