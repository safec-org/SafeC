// memflow.sc — compile-time use-after-free / use-after-reset checking for
// '&heap' and '&arena<R>' references held in local variables: a forward
// dataflow over each function body (branches joined, loops iterated to a
// fixpoint, a reset or dealloc inside a callee seen through a per-function
// summary). A simplified version of the C++ compiler's MemoryFlow pass.
#include "scc.h"
#include "sema.h"
#include "sema_int.h"

#define MF_V 256        // tracked locals per function
#define MF_T 64         // allocation tokens per function
#define MF_D 64         // saved-state stack depth

int g_mf_loc0 = 0;
int g_mf_nv = 0;
long g_mf_var[MF_V];        // tokens a variable may hold
int g_mf_mark[MF_V];        // a mark (token seq) held by a variable, or -1
long g_mf_inv = 0L;         // tokens that may be invalid (reset/freed)
int g_mf_live = 1;
int g_mf_unsafe = 0;

int g_mf_ntok = 0;
int g_mf_tregion[MF_T];     // arena rgn sid (0: heap)
int g_mf_tseq[MF_T];
int g_mf_seq = 0;

// saved states
long g_mf_svar[MF_D * MF_V];
int g_mf_smark[MF_D * MF_V];
long g_mf_sinv[MF_D];
int g_mf_slive[MF_D];
int g_mf_nsave = 0;

// per-function summaries: regions a function resets (a few), frees params
int g_mf_resets[NFUNC_MAX * 4];
int g_mf_nresets[NFUNC_MAX];
int g_mf_summ_done[NFUNC_MAX];

// reported (dedup)
int g_mf_rep_tok[512];
int g_mf_nrep = 0;

int mf_var_of(int l) {
    int v = l - g_mf_loc0;
    if (v < 0 || v >= g_mf_nv) { return -1; }
    return v;
}

int mf_save() {
    if (g_mf_nsave >= MF_D) { return -1; }
    int s = g_mf_nsave;
    int i = 0;
    while (i < g_mf_nv) {
        g_mf_svar[s * MF_V + i] = g_mf_var[i];
        g_mf_smark[s * MF_V + i] = g_mf_mark[i];
        i = i + 1;
    }
    g_mf_sinv[s] = g_mf_inv;
    g_mf_slive[s] = g_mf_live;
    g_mf_nsave = s + 1;
    return s;
}

void mf_restore(int s) {
    if (s < 0) { return; }
    int i = 0;
    while (i < g_mf_nv) {
        g_mf_var[i] = g_mf_svar[s * MF_V + i];
        g_mf_mark[i] = g_mf_smark[s * MF_V + i];
        i = i + 1;
    }
    g_mf_inv = g_mf_sinv[s];
    g_mf_live = g_mf_slive[s];
}

// current state ∪= saved state s
void mf_join(int s) {
    if (s < 0) { return; }
    if (g_mf_slive[s] == 0) { return; }
    if (g_mf_live == 0) { mf_restore(s); return; }
    int i = 0;
    while (i < g_mf_nv) {
        g_mf_var[i] = g_mf_var[i] | g_mf_svar[s * MF_V + i];
        if (g_mf_mark[i] != g_mf_smark[s * MF_V + i]) { g_mf_mark[i] = -1; }
        i = i + 1;
    }
    g_mf_inv = g_mf_inv | g_mf_sinv[s];
}

void mf_pop(int s) { if (s >= 0) { g_mf_nsave = s; } }

int mf_new_tok(int rgn) {
    int t = g_mf_ntok;
    if (t >= MF_T) { t = MF_T - 1; } else { g_mf_ntok = t + 1; }
    g_mf_tregion[t] = rgn;
    g_mf_seq = g_mf_seq + 1;
    g_mf_tseq[t] = g_mf_seq;
    g_mf_inv = g_mf_inv & ~(1L << (long)t);
    return t;
}

void mf_invalidate_region(int rgn, int after_seq) {
    int t = 0;
    while (t < g_mf_ntok) {
        if (g_mf_tregion[t] == rgn && g_mf_tseq[t] > after_seq) { g_mf_inv = g_mf_inv | (1L << (long)t); }
        t = t + 1;
    }
}

void mf_report(int e, int v, long bad) {
    int i = 0;
    while (i < g_mf_nrep) { if (g_mf_rep_tok[i] == g_node[e].tok) { return; } i = i + 1; }
    if (g_mf_nrep < 512) { g_mf_rep_tok[g_mf_nrep] = g_node[e].tok; g_mf_nrep = g_mf_nrep + 1; }
    int rgn = 0;
    int t = 0;
    while (t < g_mf_ntok) {
        if ((bad & (1L << (long)t)) != 0L) { rgn = g_mf_tregion[t]; break; }
        t = t + 1;
    }
    int lt = g_local[g_mf_loc0 + v].ty;
    msg_reset();
    msg_lit("use of '");
    msg_str(g_node[e].sid);
    msg_lit("' ");
    if (rgn == 0) {
        if (g_ty[lt].kind == TY_REF && g_ty[lt].n == R_HEAP) { msg_lit("(&heap reference)"); }
        else { msg_lit("(a reference into a &heap allocation)"); }
        msg_lit(" after std::dealloc() freed it");
    } else {
        if (g_ty[lt].kind == TY_REF && g_ty[lt].n == R_ARENA) { msg_lit("(&arena<"); msg_str(rgn); msg_lit("> reference)"); }
        else { msg_lit("(a reference into arena<"); msg_str(rgn); msg_lit(">)"); }
        msg_lit(" after arena_reset<"); msg_str(rgn);
        msg_lit(">(), arena_destroy<"); msg_str(rgn);
        msg_lit(">(), or arena_free_to<"); msg_str(rgn);
        msg_lit(">() invalidated it");
    }
    error_node(e);
}

int is_dealloc_name(int sid) {
    int n = g_str_len[sid];
    if (str_eq_lit(sid, "dealloc") == 1 || str_eq_lit(sid, "dealloc_raw_") == 1) { return 1; }
    if (n > 9) {
        int p = n - 9;
        if (str_char(sid, p) == ':' && str_char(sid, p + 1) == ':' && str_char(sid, p + 2) == 'd' &&
            str_char(sid, p + 3) == 'e' && str_char(sid, p + 8) == 'c') { return 1; }
    }
    return 0;
}

// summary: arena regions a function body resets (directly or via calls)
void mf_summary(int f);

void mf_scan_resets(int f, int n) {
    if (n == 0) { return; }
    int k = g_node[n].kind;
    if (k == E_ARENA && (g_node[n].op == K_ARENA_RESET || g_node[n].op == K_ARENA_DESTROY || g_node[n].op == K_ARENA_FREE_TO)) {
        int c = g_mf_nresets[f];
        int i = 0;
        int have = 0;
        while (i < c) { if (g_mf_resets[f * 4 + i] == g_node[n].sid) { have = 1; } i = i + 1; }
        if (have == 0 && c < 4) { g_mf_resets[f * 4 + c] = g_node[n].sid; g_mf_nresets[f] = c + 1; }
    }
    if (k == E_CALL && (g_node[n].c == CK_FUNC || g_node[n].c == CK_METHOD)) {
        int g = g_node[n].d;
        if (g > 0 && g != f) {
            mf_summary(g);
            int i = 0;
            while (i < g_mf_nresets[g]) {
                int r = g_mf_resets[g * 4 + i];
                int c = g_mf_nresets[f];
                int j = 0;
                int have = 0;
                while (j < c) { if (g_mf_resets[f * 4 + j] == r) { have = 1; } j = j + 1; }
                if (have == 0 && c < 4) { g_mf_resets[f * 4 + c] = r; g_mf_nresets[f] = c + 1; }
                i = i + 1;
            }
        }
    }
    if (k == S_UNSAFE) { return; }
    // children
    if (k != E_IDENT && k != E_INT && k != E_STR && k != E_FLOAT && k != E_CHAR) {
        if (k != E_CAST && k != E_SIZEOF_T && k != E_ALIGNOF && k != E_NEW && k != S_VAR) { mf_scan_resets(f, g_node[n].c); }
        mf_scan_resets(f, g_node[n].a);
        mf_scan_resets(f, g_node[n].b);
        if (k == S_FOR) { mf_scan_resets(f, g_node[n].d); }
        int i = 0;
        while (i < g_node[n].nlist) { mf_scan_resets(f, g_list[g_node[n].list + i]); i = i + 1; }
    }
}

void mf_summary(int f) {
    if (g_mf_summ_done[f] != 0) { return; }
    g_mf_summ_done[f] = 1;
    g_mf_nresets[f] = 0;
    mf_scan_resets(f, g_node[g_func[f].node].a);
}

long mf_expr(int e);

void mf_use(int e) {
    // a read of a local reference variable
    if (g_node[e].kind != E_IDENT) { return; }
    if (g_node[e].c != SYM_LOCAL) { return; }
    int v = mf_var_of(g_node[e].d);
    if (v < 0) { return; }
    long bad = g_mf_var[v] & g_mf_inv;
    if (bad != 0L && g_mf_live == 1 && g_mf_unsafe == 0) { mf_report(e, v, bad); }
}

int g_mf_rhs_mark = -1;

long mf_expr(int e) {
    if (e == 0) { return 0L; }
    int k = g_node[e].kind;
    g_mf_rhs_mark = -1;
    if (k == E_IDENT) {
        mf_use(e);
        if (g_node[e].c == SYM_LOCAL) {
            int v = mf_var_of(g_node[e].d);
            if (v >= 0) { g_mf_rhs_mark = g_mf_mark[v]; return g_mf_var[v]; }
        }
        return 0L;
    }
    if (k == E_NEW) {
        if (g_node[e].a != 0) { mf_expr(g_node[e].a); }
        int t = mf_new_tok(g_node[e].sid);
        return 1L << (long)t;
    }
    if (k == E_ARENA) {
        int op = g_node[e].op;
        int r = g_node[e].sid;
        if (op == K_ARENA_RESET || op == K_ARENA_DESTROY) { mf_invalidate_region(r, -1); return 0L; }
        if (op == K_ARENA_MARK) {
            // the newest token seq of this rgn so far
            g_mf_rhs_mark = g_mf_seq;
            return 0L;
        }
        int a = g_node[e].a;
        int mk = -1;
        if (g_node[a].kind == E_IDENT && g_node[a].c == SYM_LOCAL) {
            int v = mf_var_of(g_node[a].d);
            if (v >= 0) { mk = g_mf_mark[v]; }
        } else { mf_expr(a); }
        mf_invalidate_region(r, mk);
        return 0L;
    }
    if (k == E_ASSIGN) {
        int lhs = g_node[e].a;
        long rv = mf_expr(g_node[e].b);
        int rm = g_mf_rhs_mark;
        if (g_node[lhs].kind == E_IDENT && g_node[lhs].c == SYM_LOCAL) {
            int v = mf_var_of(g_node[lhs].d);
            if (v >= 0) {
                if (g_node[e].op == T_ASSIGN) { g_mf_var[v] = rv; g_mf_mark[v] = rm; }
                else { mf_use(lhs); }
                return rv;
            }
            return rv;
        }
        mf_expr(lhs);
        return rv;
    }
    if (k == E_CALL) {
        int ck = g_node[e].c;
        if (ck == CK_METHOD) { mf_expr(g_node[e].b); }
        if (ck == CK_ISNULL || ck == CK_ISNONE || ck == CK_DEFAULT) { mf_expr(g_node[e].b); }
        if (ck == CK_INDIRECT) { mf_expr(g_node[e].a); }
        int i = 0;
        long first = 0L;
        while (i < g_node[e].nlist) {
            int a = g_list[g_node[e].list + i];
            long av = mf_expr(a);
            if (i == 0) { first = av; }
            i = i + 1;
        }
        if (ck == CK_FUNC || ck == CK_METHOD) {
            int f = g_node[e].d;
            if (f > 0) {
                int name = g_func[f].origsid;
                int callee = g_node[e].a;
                int cname = 0;
                if (g_node[callee].kind == E_IDENT) { cname = g_node[callee].sid; }
                if ((is_dealloc_name(name) == 1 || (cname != 0 && is_dealloc_name(cname) == 1)) && g_node[e].nlist > 0) {
                    g_mf_inv = g_mf_inv | first;
                    // heap tokens: a variable passed to dealloc gets its own token
                    int a0 = g_list[g_node[e].list];
                    if (g_node[a0].kind == E_IDENT && g_node[a0].c == SYM_LOCAL) {
                        int v = mf_var_of(g_node[a0].d);
                        if (v >= 0) {
                            if (g_mf_var[v] == 0L) { g_mf_var[v] = 1L << (long)mf_new_tok(0); }
                            g_mf_inv = g_mf_inv | g_mf_var[v];
                        }
                    }
                } else {
                    mf_summary(f);
                    int j = 0;
                    while (j < g_mf_nresets[f]) { mf_invalidate_region(g_mf_resets[f * 4 + j], -1); j = j + 1; }
                }
            }
        }
        return 0L;
    }
    if (k == E_TERNARY) {
        mf_expr(g_node[e].a);
        int s = mf_save();
        long a = mf_expr(g_node[e].b);
        int s2 = mf_save();
        mf_restore(s);
        long b = mf_expr(g_node[e].c);
        mf_join(s2);
        mf_pop(s);
        return a | b;
    }
    if (k == E_BINARY && (g_node[e].op == T_ANDAND || g_node[e].op == T_OROR)) {
        mf_expr(g_node[e].a);
        int s = mf_save();
        mf_expr(g_node[e].b);
        mf_join(s);
        mf_pop(s);
        return 0L;
    }
    if (k == E_MATCH) {
        mf_expr(g_node[e].a);
        int s = mf_save();
        int acc = -1;
        int i = 0;
        while (i < g_node[e].nlist) {
            int arm = g_list[g_node[e].list + i];
            mf_restore(s);
            mf_expr(g_node[arm].a);
            if (acc < 0) { acc = mf_save(); } else { mf_join(acc); mf_pop(acc); acc = mf_save(); }
            i = i + 1;
        }
        if (acc >= 0) { mf_restore(acc); }
        mf_pop(s);
        return 0L;
    }
    if (k == E_CAST || k == E_SIZEOF_E || k == E_UNARY || k == E_POSTFIX || k == E_ADDR || k == E_DEREF ||
        k == E_FORCE || k == E_MEMBER || k == E_TRY || k == E_DESIG) {
        if (k == E_SIZEOF_E) { return 0L; }
        long v = mf_expr(g_node[e].a);
        if (k == E_CAST || k == E_FORCE) { return v; }
        return 0L;
    }
    if (k == E_BINARY || k == E_INDEX || k == E_COMMA) {
        mf_expr(g_node[e].a);
        long v = mf_expr(g_node[e].b);
        if (k == E_COMMA) { return v; }
        return 0L;
    }
    if (k == E_SLICE) {
        mf_expr(g_node[e].a);
        mf_expr(g_node[e].b);
        mf_expr(g_node[e].c);
        return 0L;
    }
    if (k == E_INITLIST || k == E_TUPLE) {
        int i = 0;
        while (i < g_node[e].nlist) { mf_expr(g_list[g_node[e].list + i]); i = i + 1; }
        return 0L;
    }
    return 0L;
}

void mf_stmt(int s);

void mf_list(int s) {
    int i = 0;
    while (i < g_node[s].nlist) { mf_stmt(g_list[g_node[s].list + i]); i = i + 1; }
}

void mf_loop(int cond, int body, int incr, int init, int is_do) {
    if (init != 0) { mf_stmt(init); }
    // iterate to a fixpoint (states only grow)
    int iter = 0;
    while (iter < 4) {
        int before = mf_save();
        if (cond != 0 && is_do == 0) { mf_expr(cond); }
        int entry = mf_save();
        mf_stmt(body);
        if (g_mf_live == 0) { mf_restore(entry); }
        if (incr != 0) { mf_expr(incr); }
        if (cond != 0 && is_do == 1) { mf_expr(cond); }
        // join with the state at loop entry
        mf_join(before);
        int changed = 0;
        int i = 0;
        while (i < g_mf_nv) { if (g_mf_var[i] != g_mf_svar[before * MF_V + i]) { changed = 1; } i = i + 1; }
        if (g_mf_inv != g_mf_sinv[before]) { changed = 1; }
        mf_pop(before);
        g_mf_live = 1;
        if (changed == 0) { break; }
        iter = iter + 1;
    }
}

void mf_stmt(int s) {
    if (s == 0) { return; }
    int k = g_node[s].kind;
    if (k == S_BLOCK || k == S_MULTI) { mf_list(s); return; }
    if (k == S_VAR) {
        long v = 0L;
        int m = -1;
        if (g_node[s].a != 0) { v = mf_expr(g_node[s].a); m = g_mf_rhs_mark; }
        int x = mf_var_of(g_node[s].d);
        if (x >= 0) { g_mf_var[x] = v; g_mf_mark[x] = m; }
        return;
    }
    if (k == S_EXPR) { mf_expr(g_node[s].a); return; }
    if (k == S_IF) {
        mf_expr(g_node[s].a);
        int sv = mf_save();
        mf_stmt(g_node[s].b);
        int st = mf_save();
        mf_restore(sv);
        if (g_node[s].c != 0) { mf_stmt(g_node[s].c); }
        mf_join(st);
        mf_pop(sv);
        return;
    }
    if (k == S_IFCONST) {
        if (g_node[s].op == 1) { mf_stmt(g_node[s].b); }
        else if (g_node[s].op == 2) { mf_stmt(g_node[s].c); }
        return;
    }
    if (k == S_WHILE) { mf_loop(g_node[s].a, g_node[s].b, 0, 0, 0); return; }
    if (k == S_DO) { mf_loop(g_node[s].a, g_node[s].b, 0, 0, 1); return; }
    if (k == S_FOR) { mf_loop(g_node[s].b, g_node[s].d, g_node[s].c, g_node[s].a, 0); return; }
    if (k == S_RETURN) {
        if (g_node[s].a != 0) { mf_expr(g_node[s].a); }
        g_mf_live = 0;
        return;
    }
    if (k == S_BREAK || k == S_CONTINUE) { g_mf_live = 0; return; }
    if (k == S_UNSAFE) {
        g_mf_unsafe = g_mf_unsafe + 1;
        mf_stmt(g_node[s].a);
        g_mf_unsafe = g_mf_unsafe - 1;
        return;
    }
    if (k == S_MATCH || k == S_SWITCH) {
        mf_expr(g_node[s].a);
        int sv = mf_save();
        int acc = -1;
        int i = 0;
        while (i < g_node[s].nlist) {
            int arm = g_list[g_node[s].list + i];
            mf_restore(sv);
            if (k == S_MATCH) { mf_stmt(g_node[arm].a); } else { mf_stmt(g_node[arm].b); }
            if (acc < 0) { acc = mf_save(); } else { mf_join(acc); mf_pop(acc); acc = mf_save(); }
            i = i + 1;
        }
        // an arm may not match: the state before the match also flows on
        if (acc >= 0) { mf_restore(acc); mf_join(sv); }
        mf_pop(sv);
        g_mf_live = 1;
        return;
    }
    if (k == S_DEFER) { return; }
}

void memflow_function(int f) {
    int body = g_node[g_func[f].node].a;
    if (body == 0) { return; }
    g_mf_loc0 = g_func[f].loc0;
    g_mf_nv = g_func[f].nloc;
    if (g_mf_nv > MF_V) { g_mf_nv = MF_V; }
    int i = 0;
    while (i < g_mf_nv) { g_mf_var[i] = 0L; g_mf_mark[i] = -1; i = i + 1; }
    g_mf_inv = 0L;
    g_mf_live = 1;
    g_mf_unsafe = 0;
    g_mf_ntok = 0;
    g_mf_seq = 0;
    g_mf_nsave = 0;
    mf_stmt(body);
}

// run over every checked function, in declaration order
void memflow_program() {
    int f = 1;
    while (f <= g_nfunc) {
        if (g_func[f].checked != 0 && g_node[g_func[f].node].a != 0 && g_func[f].ty != 0) {
            // the runtime prelude is trusted
            if (g_tok[g_node[g_func[f].node].tok].file != g_rt_file) { memflow_function(f); }
        }
        f = f + 1;
    }
}
