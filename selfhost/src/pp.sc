// pp.sc — the preprocessor: #include (bundled headers + host files),
// object-like and function-like #define (with # and ##), #undef,
// #if/#ifdef/#ifndef/#elif/#else/#endif, #pragma once, #error/#warning.
// Works on tokens: each file is lexed onto the top of g_ptok, directives
// are executed, and everything else is macro-expanded into g_tok.
#include "scc.h"
#include "lex.h"
#include "driver.h"

struct Tok g_tok[TOK_MAX];
int g_ntok = 0;

#define MAXPARAM 64

struct Macro {
    int sid;
    int body;
    int nbody;
    int params;
    int nparams;
    int funclike;
    int variadic;
    int active;
    int file;       // where it was #defined
};

struct Macro g_macro[NMACRO_MAX];
int g_nmacro = 0;
int g_macro_of[NSTR_MAX];
struct Tok g_mtok[MTOK_MAX];
int g_nmtok = 0;
int g_mparam[NMACRO_MAX * 4];
int g_nmparam = 0;
int g_once[NSTR_MAX];
int g_include_depth = 0;

// conditional stack
int g_cond_active[256];   // this group is being emitted
int g_cond_taken[256];    // some group of this #if chain was taken
int g_cond_parent[256];   // enclosing context was active
int g_cond_depth = 0;

int pp_skipping() {
    if (g_cond_depth == 0) { return 0; }
    if (g_cond_active[g_cond_depth - 1] == 1) { return 0; }
    return 1;
}

void emit_tok(int i, int flags_or) {
    if (g_ntok >= TOK_MAX - 2) { fatal("tokens"); return; }
    g_tok[g_ntok].kind = g_ptok[i].kind;
    g_tok[g_ntok].sid = g_ptok[i].sid;
    g_tok[g_ntok].file = g_ptok[i].file;
    g_tok[g_ntok].line = g_ptok[i].line;
    g_tok[g_ntok].col = g_ptok[i].col;
    g_tok[g_ntok].flags = g_ptok[i].flags | flags_or;
    g_tok[g_ntok].ival = g_ptok[i].ival;
    if (g_tok[g_ntok].kind == T_IDENT && g_kw[g_tok[g_ntok].sid] != 0) {
        g_tok[g_ntok].kind = g_kw[g_tok[g_ntok].sid];
    }
    g_ntok = g_ntok + 1;
}

// copy a token onto the top of g_ptok, relocated to (file, line, col)
void push_copy(struct Tok t, int file, int line, int col) {
    if (g_nptok >= RAW_MAX - 1) { fatal("expansion tokens"); return; }
    g_ptok[g_nptok] = t;
    if (file != 0) {
        g_ptok[g_nptok].file = file;
        g_ptok[g_nptok].line = line;
        g_ptok[g_nptok].col = col;
    }
    g_nptok = g_nptok + 1;
}

void pp_error(int i, const char* s) {
    msg_reset();
    msg_lit(s);
    diag_at(g_ptok[i].file, g_ptok[i].line, 1, 1);
}

int find_param(int m, int sid) {
    int k = 0;
    while (k < g_macro[m].nparams) {
        if (g_mparam[g_macro[m].params + k] == sid) { return k; }
        k = k + 1;
    }
    return -1;
}

// Mini-lexer for the result of a ## paste (held in g_tmp).
void push_pasted(int file, int line, int col, int flags) {
    int sid = intern_tmp();
    int n = g_tmp_len;
    int c0 = 0;
    if (n > 0) { c0 = (int)g_tmp[0]; }
    if (n > 0 && is_ident_start(c0)) {
        push_ptok(T_IDENT, sid, line, col, flags, 0L);
    } else if (n > 0 && is_digit(c0)) {
        long v = 0L;
        int i = 0;
        while (i < n && is_digit((int)g_tmp[i])) { v = v * 10L + (long)((int)g_tmp[i] - '0'); i = i + 1; }
        push_ptok(T_INT, sid, line, col, flags, v);
    } else {
        int k = 0;
        if (n == 1) { k = punct1(c0); }
        else if (n == 2) { k = punct2(c0, (int)g_tmp[1]); }
        else if (n == 3) { k = punct3(c0, (int)g_tmp[1], (int)g_tmp[2]); }
        if (k == 0) { k = T_INVALID; }
        push_ptok(k, 0, line, col, flags, 0L);
    }
    g_ptok[g_nptok - 1].file = file;
}

int expand_one(int i, int end);

void expand_range(int start, int end) {
    int i = start;
    while (i < end) { i = expand_one(i, end); }
}

int g_lastpaste = 0;

int expand_one(int i, int end) {
    int kind = g_ptok[i].kind;
    if (kind != T_IDENT || (g_ptok[i].flags & TF_NOEXP) != 0) { emit_tok(i, 0); return i + 1; }
    int sid = g_ptok[i].sid;
    if (sid == S_LINE) {
        tmp_reset();
        tmp_int((long)g_ptok[i].line);
        int save = g_nptok;
        push_ptok(T_INT, intern_tmp(), g_ptok[i].line, g_ptok[i].col, g_ptok[i].flags, (long)g_ptok[i].line);
        g_ptok[save].file = g_ptok[i].file;
        emit_tok(save, 0);
        g_nptok = save;
        return i + 1;
    }
    if (sid == S_FILE) {
        int save = g_nptok;
        push_ptok(T_STRING, g_file_name[g_ptok[i].file], g_ptok[i].line, g_ptok[i].col, g_ptok[i].flags, 0L);
        g_ptok[save].file = g_ptok[i].file;
        emit_tok(save, 0);
        g_nptok = save;
        return i + 1;
    }
    int m = g_macro_of[sid];
    if (m == 0) { emit_tok(i, 0); return i + 1; }
    if (g_macro[m].active == 1) { emit_tok(i, TF_NOEXP); return i + 1; }
    int file = g_ptok[i].file;
    int line = g_ptok[i].line;
    int col = g_ptok[i].col;
    int lead = g_ptok[i].flags & (TF_BOL | TF_SPACE);
    if (g_macro[m].funclike == 0) {
        int save = g_nptok;
        int j = 0;
        while (j < g_macro[m].nbody) {
            push_copy(g_mtok[g_macro[m].body + j], file, line, col);
            if (j == 0) { g_ptok[g_nptok - 1].flags = (g_ptok[g_nptok - 1].flags & ~(TF_BOL | TF_SPACE)) | lead; }
            j = j + 1;
        }
        g_macro[m].active = 1;
        expand_range(save, g_nptok);
        g_macro[m].active = 0;
        g_nptok = save;
        return i + 1;
    }
    // function-like: needs '(' next
    if (i + 1 >= end || g_ptok[i + 1].kind != T_LPAREN) { emit_tok(i, 0); return i + 1; }
    int argstart[MAXPARAM];
    int argend[MAXPARAM];
    int nargs = 0;
    int np = g_macro[m].nparams;
    int depth = 0;
    int j = i + 2;
    int cur = j;
    int close = -1;
    while (j < end) {
        int k = g_ptok[j].kind;
        if (k == T_EOF) { break; }
        if (k == T_LPAREN || k == T_LBRACKET || k == T_LBRACE) { depth = depth + 1; }
        else if ((k == T_RPAREN || k == T_RBRACKET || k == T_RBRACE) && depth > 0) { depth = depth - 1; }
        else if (k == T_RPAREN && depth == 0) {
            if (nargs < MAXPARAM) { argstart[nargs] = cur; argend[nargs] = j; nargs = nargs + 1; }
            close = j;
            break;
        } else if (k == T_COMMA && depth == 0) {
            int is_va = 0;
            if (g_macro[m].variadic == 1 && nargs >= np - 1) { is_va = 1; }
            if (is_va == 0) {
                if (nargs < MAXPARAM) { argstart[nargs] = cur; argend[nargs] = j; nargs = nargs + 1; }
                cur = j + 1;
            }
        }
        j = j + 1;
    }
    if (close < 0) {
        pp_error(i, "unterminated macro invocation");
        return end;
    }
    // 'F()' with zero parameters gives one empty argument
    if (np == 0 && nargs == 1 && argstart[0] == argend[0]) { nargs = 0; }
    if (g_macro[m].variadic == 1 && nargs == np - 1) {
        argstart[nargs] = close; argend[nargs] = close; nargs = nargs + 1;
    }
    if (nargs != np) {
        msg_reset();
        msg_lit("macro '");
        msg_str(sid);
        msg_lit("' expects ");
        msg_int((long)np);
        msg_lit(" argument(s), got ");
        msg_int((long)nargs);
        diag_at(file, line, col, 1);
        return close + 1;
    }
    int save = g_nptok;
    // pre-expand each argument
    int es[MAXPARAM];
    int ee[MAXPARAM];
    int k = 0;
    while (k < nargs) {
        int mark = g_ntok;
        expand_range(argstart[k], argend[k]);
        es[k] = g_nptok;
        int q = mark;
        while (q < g_ntok) {
            if (g_nptok >= RAW_MAX - 1) { fatal("expansion tokens"); break; }
            g_ptok[g_nptok] = g_tok[q];
            g_nptok = g_nptok + 1;
            q = q + 1;
        }
        // keywords were converted on emission: keep them identifiers here
        q = es[k];
        while (q < g_nptok) {
            if (g_ptok[q].kind >= K_FIRST) { g_ptok[q].kind = T_IDENT; }
            q = q + 1;
        }
        ee[k] = g_nptok;
        g_ntok = mark;
        k = k + 1;
    }
    // substitute
    int bstart = g_nptok;
    int nb = g_macro[m].nbody;
    int b0 = g_macro[m].body;
    int bj = 0;
    while (bj < nb) {
        struct Tok bt = g_mtok[b0 + bj];
        int first_flags = 0;
        if (bj == 0) { first_flags = lead; }
        if (bt.kind == T_HASH && bj + 1 < nb && g_mtok[b0 + bj + 1].kind == T_IDENT &&
            find_param(m, g_mtok[b0 + bj + 1].sid) >= 0) {
            int p = find_param(m, g_mtok[b0 + bj + 1].sid);
            tmp_reset();
            int q = argstart[p];
            while (q < argend[p]) {
                if (q > argstart[p] && (g_ptok[q].flags & (TF_SPACE | TF_BOL)) != 0) { tmp_char(' '); }
                tmp_tok_spelling(g_ptok[q].kind, g_ptok[q].sid);
                q = q + 1;
            }
            int s = intern_tmp();
            push_ptok(T_STRING, s, line, col, first_flags | (bt.flags & TF_SPACE), 0L);
            g_ptok[g_nptok - 1].file = file;
            bj = bj + 2;
            continue;
        }
        if (bt.kind == T_HASHHASH && g_nptok > bstart && bj + 1 < nb) {
            struct Tok rt = g_mtok[b0 + bj + 1];
            int rp = -1;
            if (rt.kind == T_IDENT) { rp = find_param(m, rt.sid); }
            struct Tok lt = g_ptok[g_nptok - 1];
            g_nptok = g_nptok - 1;
            int rest_from = 0;
            int rest_to = 0;
            tmp_reset();
            tmp_tok_spelling(lt.kind, lt.sid);
            if (rp >= 0) {
                if (argstart[rp] < argend[rp]) {
                    tmp_tok_spelling(g_ptok[argstart[rp]].kind, g_ptok[argstart[rp]].sid);
                    rest_from = argstart[rp] + 1;
                    rest_to = argend[rp];
                }
            } else {
                tmp_tok_spelling(rt.kind, rt.sid);
            }
            if (g_tmp_len == 0) { bj = bj + 2; continue; }
            push_pasted(file, line, col, lt.flags);
            int q = rest_from;
            while (q < rest_to) { push_copy(g_ptok[q], file, line, col); q = q + 1; }
            bj = bj + 2;
            continue;
        }
        if (bt.kind == T_IDENT) {
            int p = find_param(m, bt.sid);
            if (p >= 0) {
                int raw = 0;
                if (bj + 1 < nb && g_mtok[b0 + bj + 1].kind == T_HASHHASH) { raw = 1; }
                if (bj > 0 && g_mtok[b0 + bj - 1].kind == T_HASHHASH) { raw = 1; }
                int fs = es[p];
                int fe = ee[p];
                if (raw == 1) { fs = argstart[p]; fe = argend[p]; }
                int q = fs;
                int firstq = 1;
                while (q < fe) {
                    push_copy(g_ptok[q], file, line, col);
                    if (firstq == 1) {
                        g_ptok[g_nptok - 1].flags = (g_ptok[g_nptok - 1].flags & ~(TF_BOL | TF_SPACE)) | first_flags | (bt.flags & TF_SPACE);
                        firstq = 0;
                    }
                    q = q + 1;
                }
                bj = bj + 1;
                continue;
            }
        }
        push_copy(bt, file, line, col);
        if (bj == 0) { g_ptok[g_nptok - 1].flags = (g_ptok[g_nptok - 1].flags & ~(TF_BOL | TF_SPACE)) | lead; }
        bj = bj + 1;
    }
    int bend = g_nptok;
    g_macro[m].active = 1;
    expand_range(bstart, bend);
    g_macro[m].active = 0;
    g_nptok = save;
    return close + 1;
}

// ── #if expression evaluation (over g_tok[g_ev_pos .. g_ev_end)) ─────────────
int g_ev_pos = 0;
int g_ev_end = 0;
int g_ev_err = 0;

int ev_kind() {
    if (g_ev_pos >= g_ev_end) { return T_EOF; }
    return g_tok[g_ev_pos].kind;
}

long ev_cond();

long ev_primary() {
    int k = ev_kind();
    if (k == T_INT || k == T_CHAR) { long v = g_tok[g_ev_pos].ival; g_ev_pos = g_ev_pos + 1; return v; }
    if (k == T_LPAREN) {
        g_ev_pos = g_ev_pos + 1;
        long v = ev_cond();
        if (ev_kind() == T_RPAREN) { g_ev_pos = g_ev_pos + 1; } else { g_ev_err = 1; }
        return v;
    }
    if (k == T_BANG) { g_ev_pos = g_ev_pos + 1; if (ev_primary() == 0L) { return 1L; } return 0L; }
    if (k == T_MINUS) { g_ev_pos = g_ev_pos + 1; return 0L - ev_primary(); }
    if (k == T_PLUS) { g_ev_pos = g_ev_pos + 1; return ev_primary(); }
    if (k == T_TILDE) { g_ev_pos = g_ev_pos + 1; return ~ev_primary(); }
    if (k == T_IDENT || k >= K_FIRST) {
        g_ev_pos = g_ev_pos + 1;
        if (k == K_TRUE) { return 1L; }
        // a function-like macro use left unexpanded, or an unknown name: 0
        if (ev_kind() == T_LPAREN) {
            int depth = 0;
            while (g_ev_pos < g_ev_end) {
                int kk = ev_kind();
                g_ev_pos = g_ev_pos + 1;
                if (kk == T_LPAREN) { depth = depth + 1; }
                else if (kk == T_RPAREN) { depth = depth - 1; if (depth == 0) { break; } }
            }
        }
        return 0L;
    }
    g_ev_err = 1;
    g_ev_pos = g_ev_pos + 1;
    return 0L;
}

int ev_prec(int k) {
    if (k == T_STAR || k == T_SLASH || k == T_PERCENT) { return 10; }
    if (k == T_PLUS || k == T_MINUS) { return 9; }
    if (k == T_SHL || k == T_SHR) { return 8; }
    if (k == T_LT || k == T_GT || k == T_LE || k == T_GE) { return 7; }
    if (k == T_EQ || k == T_NE) { return 6; }
    if (k == T_AMP) { return 5; }
    if (k == T_CARET) { return 4; }
    if (k == T_PIPE) { return 3; }
    if (k == T_ANDAND) { return 2; }
    if (k == T_OROR) { return 1; }
    return 0;
}

long ev_binary(int minp) {
    long l = ev_primary();
    while (1) {
        int k = ev_kind();
        int p = ev_prec(k);
        if (p == 0 || p < minp) { break; }
        g_ev_pos = g_ev_pos + 1;
        long r = ev_binary(p + 1);
        if (k == T_STAR) { l = l * r; }
        else if (k == T_SLASH) { if (r != 0L) { l = l / r; } else { l = 0L; } }
        else if (k == T_PERCENT) { if (r != 0L) { l = l % r; } else { l = 0L; } }
        else if (k == T_PLUS) { l = l + r; }
        else if (k == T_MINUS) { l = l - r; }
        else if (k == T_SHL) { l = l << r; }
        else if (k == T_SHR) { l = l >> r; }
        else if (k == T_LT) { l = (long)(l < r); }
        else if (k == T_GT) { l = (long)(l > r); }
        else if (k == T_LE) { l = (long)(l <= r); }
        else if (k == T_GE) { l = (long)(l >= r); }
        else if (k == T_EQ) { l = (long)(l == r); }
        else if (k == T_NE) { l = (long)(l != r); }
        else if (k == T_AMP) { l = l & r; }
        else if (k == T_CARET) { l = l ^ r; }
        else if (k == T_PIPE) { l = l | r; }
        else if (k == T_ANDAND) { l = (long)(l != 0L && r != 0L); }
        else if (k == T_OROR) { l = (long)(l != 0L || r != 0L); }
    }
    return l;
}

long ev_cond() {
    long c = ev_binary(1);
    if (ev_kind() == T_QUESTION) {
        g_ev_pos = g_ev_pos + 1;
        long a = ev_cond();
        if (ev_kind() == T_COLON) { g_ev_pos = g_ev_pos + 1; } else { g_ev_err = 1; }
        long b = ev_cond();
        if (c != 0L) { return a; }
        return b;
    }
    return c;
}

// Evaluates directive tokens g_ptok[s..e)
long pp_eval(int s, int e) {
    int save = g_nptok;
    int i = s;
    while (i < e) {
        if (g_ptok[i].kind == T_IDENT && g_ptok[i].sid == S_defined) {
            int j = i + 1;
            int paren = 0;
            if (j < e && g_ptok[j].kind == T_LPAREN) { paren = 1; j = j + 1; }
            long v = 0L;
            if (j < e && g_ptok[j].kind == T_IDENT) {
                if (g_macro_of[g_ptok[j].sid] != 0) { v = 1L; }
                j = j + 1;
            }
            if (paren == 1 && j < e && g_ptok[j].kind == T_RPAREN) { j = j + 1; }
            push_ptok(T_INT, 0, g_ptok[i].line, g_ptok[i].col, 0, v);
            i = j;
            continue;
        }
        push_copy(g_ptok[i], 0, 0, 0);
        i = i + 1;
    }
    int mark = g_ntok;
    expand_range(save, g_nptok);
    g_ev_pos = mark;
    g_ev_end = g_ntok;
    g_ev_err = 0;
    long v = ev_cond();
    if (g_ev_pos < g_ev_end) { g_ev_err = 1; }
    if (g_ev_err == 1) { pp_error(s - 1, "invalid preprocessor expression"); }
    g_ntok = mark;
    g_nptok = save;
    return v;
}

// ── directives ───────────────────────────────────────────────────────────────
void do_define(int s, int e) {
    if (s >= e || (g_ptok[s].kind != T_IDENT && g_ptok[s].kind < K_FIRST)) {
        pp_error(s - 1, "invalid #define");
        return;
    }
    int name = g_ptok[s].sid;
    if (g_nmacro + 1 >= NMACRO_MAX) { fatal("macros"); return; }
    g_nmacro = g_nmacro + 1;
    int m = g_nmacro;
    g_macro[m].sid = name;
    g_macro[m].funclike = 0;
    g_macro[m].variadic = 0;
    g_macro[m].nparams = 0;
    g_macro[m].params = g_nmparam;
    g_macro[m].active = 0;
    g_macro[m].file = g_ptok[s].file;
    int i = s + 1;
    if (i < e && g_ptok[i].kind == T_LPAREN && (g_ptok[i].flags & TF_SPACE) == 0) {
        g_macro[m].funclike = 1;
        i = i + 1;
        while (i < e && g_ptok[i].kind != T_RPAREN) {
            if (g_ptok[i].kind == T_ELLIPSIS) {
                g_macro[m].variadic = 1;
                if (g_nmparam < NMACRO_MAX * 4) { g_mparam[g_nmparam] = S_VA_ARGS; g_nmparam = g_nmparam + 1; }
                g_macro[m].nparams = g_macro[m].nparams + 1;
                i = i + 1;
                continue;
            }
            if (g_ptok[i].kind == T_IDENT) {
                if (g_nmparam < NMACRO_MAX * 4) { g_mparam[g_nmparam] = g_ptok[i].sid; g_nmparam = g_nmparam + 1; }
                g_macro[m].nparams = g_macro[m].nparams + 1;
            }
            i = i + 1;
        }
        i = i + 1;
    }
    g_macro[m].body = g_nmtok;
    g_macro[m].nbody = 0;
    while (i < e) {
        if (g_nmtok >= MTOK_MAX) { fatal("macro tokens"); return; }
        g_mtok[g_nmtok] = g_ptok[i];
        g_nmtok = g_nmtok + 1;
        g_macro[m].nbody = g_macro[m].nbody + 1;
        i = i + 1;
    }
    g_macro_of[name] = m;
}

void pp_define_lit(const char* name, const char* value) {
    // lex "name value" from a scratch file
    tmp_reset();
    tmp_lit(name);
    tmp_char(' ');
    tmp_lit(value);
    int start = g_src_len;
    int i = 0;
    while (i < g_tmp_len && g_src_len < SRC_MAX) { g_src[g_src_len] = g_tmp[i]; g_src_len = g_src_len + 1; i = i + 1; }
    int len = g_tmp_len;
    int f = add_file(intern_lit("<command line>"), start, len);
    int first = lex_file(f);
    do_define(first, g_nptok - 1);
    g_nptok = first;
}

int dir_of_file(int file) {
    int name = g_file_name[file];
    int n = g_str_len[name];
    int last = -1;
    int i = 0;
    while (i < n) { if (str_char(name, i) == '/') { last = i; } i = i + 1; }
    if (last < 0) { return 0; }
    tmp_reset();
    i = 0;
    while (i <= last) { tmp_char(str_char(name, i)); i = i + 1; }
    return intern_tmp();
}

void pp_run(int file);

void do_include(int hash_idx, int s, int e) {
    int quoted = 0;
    int name = 0;
    if (s < e && g_ptok[s].kind == T_STRING) {
        quoted = 1;
        name = g_ptok[s].sid;
    } else if (s < e && g_ptok[s].kind == T_LT) {
        tmp_reset();
        int i = s + 1;
        while (i < e && g_ptok[i].kind != T_GT) { tmp_tok_spelling(g_ptok[i].kind, g_ptok[i].sid); i = i + 1; }
        name = intern_tmp();
    } else {
        pp_error(hash_idx, "invalid #include");
        return;
    }
    int cur = g_ptok[hash_idx].file;
    int f = 0;
    if (quoted == 1) {
        int d = dir_of_file(cur);
        int path = name;
        if (d != 0) { path = str_concat3(d, "", name); }
        f = host_load_file(path);
    }
    if (f == 0) { f = vfs_load(name); }
    if (f == 0) { f = host_find_include(name); }
    if (f == 0) {
        msg_reset();
        msg_lit("cannot find include file '");
        msg_str(name);
        msg_lit("'");
        diag_at(g_ptok[hash_idx].file, g_ptok[hash_idx].line, 1, 1);
        return;
    }
    if (g_once[g_file_name[f]] == 1) { return; }
    if (g_include_depth > 60) { pp_error(hash_idx, "#include nested too deeply"); return; }
    g_include_depth = g_include_depth + 1;
    pp_run(f);
    g_include_depth = g_include_depth - 1;
}

// '#pragma scc_impl <std/x.sc>' (written by tools/genvfs.py at the end of a
// bundled std header): compiles the header's implementation into this
// translation unit as if it were its own, as natively it is. The
// function-like macros the header itself defined are hidden meanwhile
// (an implementation defines e.g. SAFEC_ASSERT_IMPL_ to get the functions
// those macros would otherwise rewrite).
int g_hide_sid[256];
int g_hide_m[256];
int g_hide_top = 0;

void do_impl_include(int hash_idx, int s, int e) {
    int file = g_ptok[hash_idx].file;
    int base = g_hide_top;
    int n = base;
    int m = 1;
    while (m <= g_nmacro && n < 256) {
        int sid = g_macro[m].sid;
        if (g_macro[m].file == file && g_macro[m].funclike == 1 && g_macro_of[sid] == m) {
            g_hide_sid[n] = sid;
            g_hide_m[n] = m;
            g_macro_of[sid] = 0;
            n = n + 1;
        }
        m = m + 1;
    }
    g_hide_top = n;
    do_include(hash_idx, s, e);
    int i = base;
    while (i < n) {
        if (g_macro_of[g_hide_sid[i]] == 0) { g_macro_of[g_hide_sid[i]] = g_hide_m[i]; }
        i = i + 1;
    }
    g_hide_top = base;
}

void cond_push(int active) {
    if (g_cond_depth >= 255) { fatal("#if nesting"); return; }
    int parent = 1;
    if (pp_skipping() == 1) { parent = 0; }
    g_cond_parent[g_cond_depth] = parent;
    g_cond_active[g_cond_depth] = 0;
    if (parent == 1 && active == 1) { g_cond_active[g_cond_depth] = 1; }
    g_cond_taken[g_cond_depth] = g_cond_active[g_cond_depth];
    g_cond_depth = g_cond_depth + 1;
}

// '#line N ["file"]': the rest of this file's tokens (already lexed, up to
// its end-of-file token at the top of g_ptok) take the presumed line and
// file, for __LINE__/__FILE__ and diagnostics.
void do_line(int hash_idx, int s, int e) {
    if (s >= e || g_ptok[s].kind != T_INT) { pp_error(hash_idx, "#line expects a line number"); return; }
    int delta = (int)g_ptok[s].ival - (g_ptok[hash_idx].line + 1);
    int file = g_ptok[hash_idx].file;
    int nfile = file;
    if (s + 1 < e && g_ptok[s + 1].kind == T_STRING) {
        nfile = add_file(g_ptok[s + 1].sid, g_file_start[file], g_file_len[file]);
    }
    int j = e;
    while (j < g_nptok) {
        if (g_ptok[j].file == file) {
            g_ptok[j].line = g_ptok[j].line + delta;
            g_ptok[j].file = nfile;
        }
        j = j + 1;
    }
}

void do_directive(int hash_idx, int s, int e) {
    if (s >= e) { return; }                         // null directive
    int k = g_ptok[s].kind;
    int d = g_ptok[s].sid;
    if (k != T_IDENT) {
        if (pp_skipping() == 0) { pp_error(hash_idx, "invalid preprocessor directive"); }
        return;
    }
    // conditionals are processed even while skipping
    if (d == S_ifdef || d == S_ifndef) {
        int def = 0;
        if (s + 1 < e && g_macro_of[g_ptok[s + 1].sid] != 0) { def = 1; }
        if (d == S_ifndef) { def = 1 - def; }
        cond_push(def);
        return;
    }
    if (d == S_if) {
        int v = 0;
        if (pp_skipping() == 0) { if (pp_eval(s + 1, e) != 0L) { v = 1; } }
        cond_push(v);
        return;
    }
    if (d == S_elif) {
        if (g_cond_depth == 0) { pp_error(hash_idx, "#elif without #if"); return; }
        int t = g_cond_depth - 1;
        if (g_cond_taken[t] == 1 || g_cond_parent[t] == 0) { g_cond_active[t] = 0; return; }
        g_cond_active[t] = 1;   // so pp_eval sees an active context
        if (pp_eval(s + 1, e) != 0L) { g_cond_taken[t] = 1; } else { g_cond_active[t] = 0; }
        return;
    }
    if (d == S_else) {
        if (g_cond_depth == 0) { pp_error(hash_idx, "#else without #if"); return; }
        int t = g_cond_depth - 1;
        if (g_cond_taken[t] == 1 || g_cond_parent[t] == 0) { g_cond_active[t] = 0; }
        else { g_cond_active[t] = 1; g_cond_taken[t] = 1; }
        return;
    }
    if (d == S_endif) {
        if (g_cond_depth == 0) { pp_error(hash_idx, "#endif without #if"); return; }
        g_cond_depth = g_cond_depth - 1;
        return;
    }
    if (pp_skipping() == 1) { return; }
    if (d == S_define) { do_define(s + 1, e); return; }
    if (d == S_undef) {
        if (s + 1 < e) { g_macro_of[g_ptok[s + 1].sid] = 0; }
        return;
    }
    if (d == S_include) { do_include(hash_idx, s + 1, e); return; }
    if (d == S_pragma) {
        if (s + 1 < e && g_ptok[s + 1].sid == S_once) { g_once[g_file_name[g_ptok[hash_idx].file]] = 1; }
        if (s + 1 < e && str_eq_lit(g_ptok[s + 1].sid, "scc_impl") == 1) { do_impl_include(hash_idx, s + 2, e); }
        return;
    }
    if (d == S_error || d == S_warning) {
        msg_reset();
        if (d == S_error) { msg_lit("#error"); } else { msg_lit("#warning"); }
        int i = s + 1;
        tmp_reset();
        while (i < e) {
            if (g_tmp_len > 0) { tmp_char(' '); }
            tmp_tok_spelling(g_ptok[i].kind, g_ptok[i].sid);
            i = i + 1;
        }
        if (g_tmp_len > 0) { msg_char(' '); }
        i = 0;
        while (i < g_tmp_len) { msg_char((int)g_tmp[i]); i = i + 1; }
        int is_err = 0;
        if (d == S_error) { is_err = 1; }
        diag_at(g_ptok[hash_idx].file, g_ptok[hash_idx].line, 1, is_err);
        return;
    }
    if (d == S_line) { do_line(hash_idx, s + 1, e); return; }
    msg_reset();
    msg_lit("unknown preprocessor directive '#");
    msg_str(d);
    msg_lit("'");
    diag_at(g_ptok[hash_idx].file, g_ptok[hash_idx].line, 1, 0);
}

void pp_run(int file) {
    dbg("pp_run file "); dbg_int((long)file);
    int first = lex_file(file);
    dbg("lexed "); dbg_int((long)(g_nptok - first));
    int eof = g_nptok - 1;
    int cond_base = g_cond_depth;
    int i = first;
    while (i < eof) {
        if (g_ptok[i].kind == T_HASH && (g_ptok[i].flags & TF_BOL) != 0) {
            int j = i + 1;
            while (j < eof && (g_ptok[j].flags & TF_BOL) == 0) { j = j + 1; }
            do_directive(i, i + 1, j);
            i = j;
            continue;
        }
        if (pp_skipping() == 1) { i = i + 1; continue; }
        // a run of non-directive tokens up to the next directive line
        int j = i;
        while (j < eof && !(g_ptok[j].kind == T_HASH && (g_ptok[j].flags & TF_BOL) != 0)) { j = j + 1; }
        while (i < j) { i = expand_one(i, j); }
    }
    if (g_cond_depth > cond_base) {
        msg_reset();
        msg_lit("unterminated #if");
        diag_at(file, g_ptok[eof].line, 1, 1);
        g_cond_depth = cond_base;
    }
    g_nptok = first;
}

void pp_init() {
    g_ntok = 0;
    // token 0 is a dummy so token index 0 can mean "no location"
    g_tok[0].kind = T_EOF;
    g_ntok = 1;
}

void pp_finish() {
    g_tok[g_ntok].kind = T_EOF;
    g_tok[g_ntok].file = 0;
    if (g_ntok > 1) {
        g_tok[g_ntok].file = g_tok[g_ntok - 1].file;
        g_tok[g_ntok].line = g_tok[g_ntok - 1].line;
        g_tok[g_ntok].col = g_tok[g_ntok - 1].col + 1;
    }
    g_ntok = g_ntok + 1;
}
