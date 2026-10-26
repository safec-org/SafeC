// memflow.sc — compile-time use-after-free / use-after-reset checking, a
// simplified version of the C++ compiler's MemoryFlow pass (MemoryFlow.cpp).
//
//  * Tokens stand for objects: each allocation site ('new', a call result)
//    has two, the most recent allocation and the earlier ones (allocating
//    again folds the first into the second), and an "entry" token stands for
//    whatever a parameter path pointed to on entry ('p0' + ".data").
//  * Cells hold values: a local variable (path 0) or a field/element of one
//    (".f", "[*]"); a pointer variable's field cells stand for its pointee's
//    fields. A value is a token set plus the subset of those tokens that may
//    be dangling *as seen through this cell*, so "freed on one path and
//    replaced on the other" joins to a live value.
//  * Statements are interpreted in order; branches are joined, loops are
//    iterated to a fixpoint, defers run at returns and block ends, errdefers
//    on a failing 'try'.
//  * Calls use per-function summaries, computed over all functions until
//    they stop changing: the entry paths a function frees, reads, and writes
//    (null, a fresh allocation or a parameter), what it returns, and the
//    regions it resets or frees to a mark it was passed.
#include "scc.h"
#include "sema.h"
#include "sema_int.h"

#define MF_V 256        // tracked locals per function (one more: the call temp)
#define MF_C 384        // cells per function
#define MF_T 64         // tokens per function
#define MF_D 48         // saved states: a stack, and two slots per loop/switch target at the top
#define MF_TG 8         // nested break/continue targets tracked
#define MF_SL (MF_D - 2 * MF_TG)
#define MF_DEF 64       // pending defers
#define MF_S 8          // summary entries of each kind

int g_mf_f = 0;          // function being analysed
int g_mf_loc0 = 0;
int g_mf_nv = 0;
int g_mf_np = 0;         // its parameters (locals loc0 .. loc0+np)
int g_mf_tmpv = 0;       // the call-result temporary (a var index)
int g_mf_live = 1;
int g_mf_unsafe = 0;
int g_mf_quiet = 0;      // summary rounds: no reports
int g_mf_changed = 0;    // a summary changed this round

// cells
int g_c_var[MF_C];
int g_c_path[MF_C];
long g_c_tok[MF_C];
long g_c_dang[MF_C];
int g_c_mark[MF_C];      // marks (path 0): >= 0 seq, -1 none, <= -100 a parameter's mark
long g_c_dtok[MF_C];     // the value a cell had when it was created (its default)
long g_c_ddang[MF_C];
int g_c_dmark[MF_C];
int g_nc = 0;
int g_v_alias[MF_V + 1];   // var holds &w of local struct w: its field cells are w's
int g_v_lazy[MF_V + 1];    // -1, or a token: fields of a fresh aggregate not yet seen
long g_mf_dead = 0L;       // tokens freed/reset so far: cells created later that hold one dangle

// tokens
int g_t_kind[MF_T];      // 1 latest allocation, 2 earlier ones, 3 entry
int g_t_site[MF_T];
int g_t_rgn[MF_T];       // 0 heap, else the arena region sid
int g_t_seq[MF_T];
int g_t_param[MF_T];
int g_t_path[MF_T];
int g_t_loop[MF_T];      // a site's outermost enclosing loop (0: none)
int g_nt = 0;
int g_mf_outer = 0;      // the outermost loop being walked
int g_mf_seq = 0;

// saved states
long g_s_tok[MF_D * MF_C];
long g_s_dang[MF_D * MF_C];
int g_s_mark[MF_D * MF_C];
int g_s_alias[MF_D * (MF_V + 1)];
int g_s_lazy[MF_D * (MF_V + 1)];
int g_s_nc[MF_D];
long g_s_dead[MF_D];
int g_s_live[MF_D];
int g_mf_nsave = 0;

// defers in scope (statement nodes; flags NF_ERR for errdefer)
int g_mfd_node[MF_DEF];
int g_mfd_n = 0;

// summaries
int g_sm_nfree[NFUNC_MAX];
int g_sm_free_p[NFUNC_MAX * MF_S];
int g_sm_free_path[NFUNC_MAX * MF_S];
int g_sm_nuse[NFUNC_MAX];
int g_sm_use_p[NFUNC_MAX * MF_S];
int g_sm_use_path[NFUNC_MAX * MF_S];
int g_sm_nwr[NFUNC_MAX];
int g_sm_wr_p[NFUNC_MAX * MF_S];
int g_sm_wr_path[NFUNC_MAX * MF_S];
int g_sm_wr_kind[NFUNC_MAX * MF_S];     // 0 null, 1 fresh, 2 + j: parameter j's value
int g_sm_ret[NFUNC_MAX];                // 0 nothing tracked, 1 fresh, 2 + j: parameter j's value
int g_sm_nretf[NFUNC_MAX];
int g_sm_retf_path[NFUNC_MAX * MF_S];
int g_sm_retf_kind[NFUNC_MAX * MF_S];   // as g_sm_wr_kind
int g_sm_nreset[NFUNC_MAX];
int g_sm_reset[NFUNC_MAX * MF_S];       // region sid
int g_sm_reset_p[NFUNC_MAX * MF_S];     // -1: the whole region, else free_to(the mark param p)
int g_sm_has[NFUNC_MAX];                // 1 once computed

// this function's return values (summary rounds)
int g_mf_rkind = -1;     // -1 no return seen yet
int g_mf_rfixed = 0;

// reported (dedup)
int g_mf_rep_tok[512];
int g_mf_nrep = 0;

// ── types ────────────────────────────────────────────────────────────────────
int mf_strip(int t) {
    while (t != 0 && g_ty[t].kind == TY_NEWTYPE) { t = g_sdef[g_ty[t].n].base; }
    return t;
}

int mf_is_agg(int t) {
    t = mf_strip(t);
    if (t == 0) { return 0; }
    int k = g_ty[t].kind;
    if (k == TY_STRUCT || k == TY_ARRAY || k == TY_TUPLE) { return 1; }
    if (k == TY_OPT) { return mf_is_agg(g_ty[t].base); }
    return 0;
}

// the reference type of t (looking through ?T), or 0
int mf_ref_of(int t) {
    t = mf_strip(t);
    if (t == 0) { return 0; }
    if (g_ty[t].kind == TY_REF) { return t; }
    if (g_ty[t].kind == TY_OPT) { return mf_ref_of(g_ty[t].base); }
    return 0;
}

int mf_rgn_of_type(int t) {
    int r = mf_ref_of(t);
    if (r != 0 && g_ty[r].n == R_ARENA) { return g_ty[r].sid; }
    return 0;
}

// ── paths ────────────────────────────────────────────────────────────────────
int mf_path_depth(int p) {
    if (p == 0) { return 0; }
    int n = 0;
    int i = 0;
    while (i < g_str_len[p]) { int c = str_char(p, i); if (c == '.' || c == '[') { n = n + 1; } i = i + 1; }
    return n;
}

int mf_join_path(int a, int b) {
    if (a == 0) { return b; }
    if (b == 0) { return a; }
    tmp_reset();
    tmp_str(a);
    tmp_str(b);
    return intern_tmp();
}

int mf_field_step(int f) {
    tmp_reset();
    tmp_char('.');
    tmp_str(f);
    return intern_tmp();
}

int g_mf_elem = 0;   // "[*]"

// ── tokens ───────────────────────────────────────────────────────────────────
long mf_bit(int t) {
    if (t < 0) { return 0L; }
    return 1L << (long)t;
}

int mf_new_token(int kind) {
    if (g_nt >= MF_T) { return -1; }   // untracked from here on: never reported
    int t = g_nt;
    g_nt = t + 1;
    g_t_kind[t] = kind;
    g_t_site[t] = 0;
    g_t_rgn[t] = 0;
    g_t_seq[t] = 0;
    g_t_param[t] = -1;
    g_t_path[t] = 0;
    g_t_loop[t] = 0;
    return t;
}

int mf_entry_token(int param, int path, int rgn) {
    int t = 0;
    while (t < g_nt) {
        if (g_t_kind[t] == 3 && g_t_param[t] == param && g_t_path[t] == path) { return t; }
        t = t + 1;
    }
    if (mf_path_depth(path) > 4) { return -1; }
    t = mf_new_token(3);
    if (t < 0) { return -1; }
    g_t_param[t] = param;
    g_t_path[t] = path;
    g_t_rgn[t] = rgn;
    return t;
}

// the latest-allocation token of a site; the next two are the earlier ones
// of this iteration of its outermost loop, and those of earlier iterations
int mf_site_token(int site, int rgn) {
    int t = 0;
    while (t < g_nt) {
        if (g_t_kind[t] == 1 && g_t_site[t] == site) { return t; }
        t = t + 1;
    }
    if (g_nt + 3 > MF_T) { return -1; }
    t = mf_new_token(1);
    int k = 0;
    while (k < 3) {
        g_t_site[t + k] = site;
        g_t_rgn[t + k] = rgn;
        g_t_loop[t + k] = g_mf_outer;
        if (k > 0) { mf_new_token(2); }
        k = k + 1;
    }
    return t;
}

// ── cells ────────────────────────────────────────────────────────────────────
int mf_var_of(int l) {
    int v = l - g_mf_loc0;
    if (v < 0 || v >= g_mf_nv) { return -1; }
    return v;
}

// the value a new cell (v, path) starts with
void mf_default(int v, int path) {
    g_c_dtok[g_nc] = 0L;
    g_c_ddang[g_nc] = 0L;
    g_c_dmark[g_nc] = -1;
    if (path == 0) {
        if (v < g_mf_np) {
            int lt = g_local[g_mf_loc0 + v].ty;
            int t = mf_entry_token(v, 0, mf_rgn_of_type(lt));
            g_c_dtok[g_nc] = mf_bit(t);
            g_c_ddang[g_nc] = mf_bit(t) & g_mf_dead;
            if (g_ty[mf_strip(lt)].kind == TY_U64) { g_c_dtok[g_nc] = 0L; g_c_dmark[g_nc] = -100 - v; }
        }
        return;
    }
    // a field: of a fresh aggregate, or of what an entry token points to
    if (v <= MF_V && g_v_lazy[v] >= 0) {
        g_c_dtok[g_nc] = mf_bit(g_v_lazy[v]);
        g_c_ddang[g_nc] = g_c_dtok[g_nc] & g_mf_dead;
        return;
    }
    if (v < g_mf_np && g_v_alias[v] < 0) {
        // a field of parameter v (by value, or of what it points to)
        g_c_dtok[g_nc] = mf_bit(mf_entry_token(v, path, 0));
        g_c_ddang[g_nc] = g_c_dtok[g_nc] & g_mf_dead;
        return;
    }
}

int mf_cell(int v, int path) {
    if (v < 0) { return -1; }
    if (path != 0 && v < MF_V && g_v_alias[v] >= 0) { v = g_v_alias[v]; }
    int i = 0;
    while (i < g_nc) {
        if (g_c_var[i] == v && g_c_path[i] == path) { return i; }
        i = i + 1;
    }
    // a field of a variable that holds one entry token: that token's path
    if (path != 0 && v != g_mf_tmpv && v >= g_mf_np) {
        int base = mf_cell(v, 0);
        if (base >= 0) {
            long bt = g_c_tok[base];
            int t = 0;
            while (t < g_nt) {
                if (bt == mf_bit(t) && g_t_kind[t] == 3) {
                    if (g_nc >= MF_C) { return -1; }
                    g_c_var[g_nc] = v;
                    g_c_path[g_nc] = path;
                    g_c_dtok[g_nc] = mf_bit(mf_entry_token(g_t_param[t], mf_join_path(g_t_path[t], path), 0));
                    g_c_ddang[g_nc] = g_c_dtok[g_nc] & g_mf_dead;
                    g_c_dmark[g_nc] = -1;
                    break;
                }
                t = t + 1;
            }
            if (t < g_nt) {
                int c = g_nc;
                g_c_tok[c] = g_c_dtok[c];
                g_c_dang[c] = g_c_ddang[c];
                g_c_mark[c] = -1;
                g_nc = c + 1;
                return c;
            }
        }
    }
    if (g_nc >= MF_C) { return -1; }
    g_c_var[g_nc] = v;
    g_c_path[g_nc] = path;
    mf_default(v, path);
    int c = g_nc;
    g_c_tok[c] = g_c_dtok[c];
    g_c_dang[c] = g_c_ddang[c];
    g_c_mark[c] = g_c_dmark[c];
    g_nc = c + 1;
    return c;
}

void mf_clear_var(int v) {
    // every cell of v back to empty (a new value is being stored in v)
    int i = 0;
    while (i < g_nc) {
        if (g_c_var[i] == v) { g_c_tok[i] = 0L; g_c_dang[i] = 0L; g_c_mark[i] = -1; }
        i = i + 1;
    }
    if (v < MF_V + 1) { g_v_alias[v] = -1; g_v_lazy[v] = -1; }
}

// copy every cell of 'from' (an aggregate) into 'to'
void mf_copy_cells(int to, int from) {
    if (to < 0 || from < 0 || to == from) { return; }
    int lz = g_v_lazy[from];
    int i = 0;
    int n = g_nc;
    mf_clear_var(to);
    while (i < n) {
        if (g_c_var[i] == from && g_c_path[i] != 0) {
            int c = mf_cell(to, g_c_path[i]);
            if (c >= 0) { g_c_tok[c] = g_c_tok[i]; g_c_dang[c] = g_c_dang[i]; }
        }
        i = i + 1;
    }
    g_v_lazy[to] = lz;
}

// ── state ────────────────────────────────────────────────────────────────────
void mf_save_at(int s);

int mf_save() {
    if (g_mf_nsave >= MF_SL) { return -1; }
    int s = g_mf_nsave;
    mf_save_at(s);
    g_mf_nsave = s + 1;
    return s;
}

void mf_save_at(int s) {
    int i = 0;
    while (i < g_nc) {
        g_s_tok[s * MF_C + i] = g_c_tok[i];
        g_s_dang[s * MF_C + i] = g_c_dang[i];
        g_s_mark[s * MF_C + i] = g_c_mark[i];
        i = i + 1;
    }
    i = 0;
    while (i <= MF_V) {
        g_s_alias[s * (MF_V + 1) + i] = g_v_alias[i];
        g_s_lazy[s * (MF_V + 1) + i] = g_v_lazy[i];
        i = i + 1;
    }
    g_s_nc[s] = g_nc;
    g_s_dead[s] = g_mf_dead;
    g_s_live[s] = g_mf_live;
}

void mf_restore(int s) {
    if (s < 0) { return; }
    int i = 0;
    while (i < g_nc) {
        if (i < g_s_nc[s]) {
            g_c_tok[i] = g_s_tok[s * MF_C + i];
            g_c_dang[i] = g_s_dang[s * MF_C + i];
            g_c_mark[i] = g_s_mark[s * MF_C + i];
        } else {
            g_c_tok[i] = g_c_dtok[i];
            g_c_dang[i] = g_c_ddang[i];
            g_c_mark[i] = g_c_dmark[i];
        }
        i = i + 1;
    }
    i = 0;
    while (i <= MF_V) {
        g_v_alias[i] = g_s_alias[s * (MF_V + 1) + i];
        g_v_lazy[i] = g_s_lazy[s * (MF_V + 1) + i];
        i = i + 1;
    }
    g_mf_dead = g_s_dead[s];
    g_mf_live = g_s_live[s];
}

// current state ∪= saved state s
void mf_join(int s) {
    if (s < 0) { return; }
    if (g_s_live[s] == 0) { return; }
    if (g_mf_live == 0) { mf_restore(s); return; }
    int i = 0;
    while (i < g_nc) {
        long st = g_c_dtok[i];
        long sd = g_c_ddang[i];
        int sm = g_c_dmark[i];
        if (i < g_s_nc[s]) { st = g_s_tok[s * MF_C + i]; sd = g_s_dang[s * MF_C + i]; sm = g_s_mark[s * MF_C + i]; }
        g_c_tok[i] = g_c_tok[i] | st;
        g_c_dang[i] = g_c_dang[i] | sd;
        if (g_c_mark[i] != sm) {
            // joined marks: freeing to either one frees what follows the earlier
            if (g_c_mark[i] >= 0 && sm >= 0) { if (sm < g_c_mark[i]) { g_c_mark[i] = sm; } }
            else { g_c_mark[i] = -1; }
        }
        i = i + 1;
    }
    i = 0;
    while (i <= MF_V) {
        if (g_v_alias[i] != g_s_alias[s * (MF_V + 1) + i]) { g_v_alias[i] = -1; }
        if (g_v_lazy[i] != g_s_lazy[s * (MF_V + 1) + i]) { g_v_lazy[i] = -1; }
        i = i + 1;
    }
    g_mf_dead = g_mf_dead | g_s_dead[s];
}

void mf_pop(int s) { if (s >= 0) { g_mf_nsave = s; } }

// break/continue targets: loops and switches being walked
int g_tg_node[MF_TG];
int g_tg_label[MF_TG];
int g_tg_switch[MF_TG];
int g_ntg = 0;
int mf_slot_brk(int i) { return MF_D - 1 - 2 * i; }
int mf_slot_cont(int i) { return MF_D - 2 - 2 * i; }

void mf_slot_clear(int slot) { g_s_live[slot] = 0; g_s_nc[slot] = 0; }

// slot ∪= the current state
void mf_join_into(int slot) {
    if (g_mf_live == 0) { return; }
    if (g_s_live[slot] == 0) { mf_save_at(slot); return; }
    int t = mf_save();
    if (t < 0) { return; }
    mf_restore(slot);
    mf_join(t);
    mf_save_at(slot);
    mf_restore(t);
    mf_pop(t);
}

int mf_push_target(int node, int label, int is_switch) {
    if (g_ntg >= MF_TG) { g_ntg = g_ntg + 1; return -1; }
    int i = g_ntg;
    g_tg_node[i] = node;
    g_tg_label[i] = label;
    g_tg_switch[i] = is_switch;
    mf_slot_clear(mf_slot_brk(i));
    mf_slot_clear(mf_slot_cont(i));
    g_ntg = i + 1;
    return i;
}

void mf_jump(int is_break, int label) {
    int i = g_ntg - 1;
    if (i >= MF_TG) { i = MF_TG - 1; g_mf_live = 0; return; }
    while (i >= 0) {
        int ok = 0;
        if (label != 0) { ok = g_tg_label[i] == label; }
        else if (is_break == 1) { ok = 1; }
        else { ok = g_tg_switch[i] == 0; }
        if (ok == 1 && is_break == 0 && g_tg_switch[i] == 1) { ok = 0; }
        if (ok == 1) {
            if (is_break == 1) { mf_join_into(mf_slot_brk(i)); } else { mf_join_into(mf_slot_cont(i)); }
            g_mf_live = 0;
            return;
        }
        i = i - 1;
    }
    g_mf_live = 0;
}

// overwrites saved slot dst with saved slot src
void mf_copy_slot(int dst, int src) {
    if (dst < 0 || src < 0) { return; }
    int i = 0;
    while (i < g_s_nc[src]) {
        g_s_tok[dst * MF_C + i] = g_s_tok[src * MF_C + i];
        g_s_dang[dst * MF_C + i] = g_s_dang[src * MF_C + i];
        g_s_mark[dst * MF_C + i] = g_s_mark[src * MF_C + i];
        i = i + 1;
    }
    i = 0;
    while (i <= MF_V) {
        g_s_alias[dst * (MF_V + 1) + i] = g_s_alias[src * (MF_V + 1) + i];
        g_s_lazy[dst * (MF_V + 1) + i] = g_s_lazy[src * (MF_V + 1) + i];
        i = i + 1;
    }
    g_s_nc[dst] = g_s_nc[src];
    g_s_dead[dst] = g_s_dead[src];
    g_s_live[dst] = g_s_live[src];
}

int mf_same_as(int s) {
    if (s < 0) { return 1; }
    if (g_mf_dead != g_s_dead[s]) { return 0; }
    int i = 0;
    while (i < g_nc) {
        long st = g_c_dtok[i];
        long sd = g_c_ddang[i];
        if (i < g_s_nc[s]) { st = g_s_tok[s * MF_C + i]; sd = g_s_dang[s * MF_C + i]; }
        if (g_c_tok[i] != st || g_c_dang[i] != sd) { return 0; }
        i = i + 1;
    }
    return 1;
}

// ── summaries: recording ─────────────────────────────────────────────────────
int mf_sm_add(int f, int kind, int p, int path, int extra) {
    // kind 0 free, 1 use, 2 write, 3 reset, 4 ret field
    int i = 0;
    if (kind == 0) {
        while (i < g_sm_nfree[f]) { if (g_sm_free_p[f * MF_S + i] == p && g_sm_free_path[f * MF_S + i] == path) { return 0; } i = i + 1; }
        if (g_sm_nfree[f] >= MF_S) { return 0; }
        g_sm_free_p[f * MF_S + i] = p; g_sm_free_path[f * MF_S + i] = path; g_sm_nfree[f] = i + 1;
    } else if (kind == 1) {
        while (i < g_sm_nuse[f]) { if (g_sm_use_p[f * MF_S + i] == p && g_sm_use_path[f * MF_S + i] == path) { return 0; } i = i + 1; }
        if (g_sm_nuse[f] >= MF_S) { return 0; }
        g_sm_use_p[f * MF_S + i] = p; g_sm_use_path[f * MF_S + i] = path; g_sm_nuse[f] = i + 1;
    } else if (kind == 2) {
        while (i < g_sm_nwr[f]) {
            if (g_sm_wr_p[f * MF_S + i] == p && g_sm_wr_path[f * MF_S + i] == path) {
                if (g_sm_wr_kind[f * MF_S + i] == extra) { return 0; }
                g_sm_wr_kind[f * MF_S + i] = extra;
                g_mf_changed = 1;
                return 1;
            }
            i = i + 1;
        }
        if (g_sm_nwr[f] >= MF_S) { return 0; }
        g_sm_wr_p[f * MF_S + i] = p; g_sm_wr_path[f * MF_S + i] = path; g_sm_wr_kind[f * MF_S + i] = extra; g_sm_nwr[f] = i + 1;
    } else if (kind == 3) {
        while (i < g_sm_nreset[f]) { if (g_sm_reset[f * MF_S + i] == path && g_sm_reset_p[f * MF_S + i] == p) { return 0; } i = i + 1; }
        if (g_sm_nreset[f] >= MF_S) { return 0; }
        g_sm_reset[f * MF_S + i] = path; g_sm_reset_p[f * MF_S + i] = p; g_sm_nreset[f] = i + 1;
    } else {
        while (i < g_sm_nretf[f]) {
            if (g_sm_retf_path[f * MF_S + i] == path) {
                if (g_sm_retf_kind[f * MF_S + i] == extra) { return 0; }
                g_sm_retf_kind[f * MF_S + i] = extra;
                g_mf_changed = 1;
                return 1;
            }
            i = i + 1;
        }
        if (g_sm_nretf[f] >= MF_S) { return 0; }
        g_sm_retf_path[f * MF_S + i] = path; g_sm_retf_kind[f * MF_S + i] = extra; g_sm_nretf[f] = i + 1;
    }
    g_mf_changed = 1;
    return 1;
}

// ── effects ──────────────────────────────────────────────────────────────────
// every cell holding a token of 'mask' may now dangle
void mf_invalidate(long mask) {
    if (mask == 0L) { return; }
    g_mf_dead = g_mf_dead | mask;
    int i = 0;
    while (i < g_nc) {
        g_c_dang[i] = g_c_dang[i] | (g_c_tok[i] & mask);
        i = i + 1;
    }
}

void mf_free(long mask) {
    int t = 0;
    while (t < g_nt) {
        if ((mask & mf_bit(t)) != 0L && g_t_kind[t] == 3) { mf_sm_add(g_mf_f, 0, g_t_param[t], g_t_path[t], 0); }
        t = t + 1;
    }
    mf_invalidate(mask);
}

// arena_reset (after_seq -1) or arena_free_to (a mark): allocations of the
// region made after the mark; a reset also reaches entry objects
void mf_reset(int rgn, int after_seq) {
    long mask = 0L;
    int t = 0;
    while (t < g_nt) {
        if (g_t_rgn[t] == rgn && rgn != 0) {
            if (g_t_kind[t] == 3) { if (after_seq < 0) { mask = mask | mf_bit(t); } }
            else if (g_t_seq[t] > after_seq) { mask = mask | mf_bit(t); }
        }
        t = t + 1;
    }
    mf_invalidate(mask);
}

// renames token 'from' to 'to' everywhere in the state
void mf_rename(int from, int to) {
    long fb = mf_bit(from);
    long tb = mf_bit(to);
    int i = 0;
    while (i < g_nc) {
        if ((g_c_tok[i] & fb) != 0L) { g_c_tok[i] = (g_c_tok[i] & ~fb) | tb; }
        if ((g_c_dang[i] & fb) != 0L) { g_c_dang[i] = (g_c_dang[i] & ~fb) | tb; }
        i = i + 1;
    }
    i = 0;
    while (i <= MF_V) { if (g_v_lazy[i] == from) { g_v_lazy[i] = to; } i = i + 1; }
    if ((g_mf_dead & fb) != 0L) { g_mf_dead = (g_mf_dead & ~fb) | tb; }
    if (g_t_seq[from] > g_t_seq[to]) { g_t_seq[to] = g_t_seq[from]; }
}

// the next iteration of 'loop' starts: allocations made under it so far
// belong to earlier iterations
void mf_fold_iteration(int loop) {
    int t = 0;
    while (t + 2 < g_nt) {
        if (g_t_kind[t] == 1 && g_t_loop[t] == loop && loop != 0) {
            mf_rename(t, t + 2);
            mf_rename(t + 1, t + 2);
        }
        t = t + 1;
    }
}

// a new allocation at a site: the previous one becomes one of the earlier ones
long mf_fresh(int site, int rgn) {
    int l = mf_site_token(site, rgn);
    if (l < 0) { return 0L; }
    long lb = mf_bit(l);
    mf_rename(l, l + 1);
    g_mf_seq = g_mf_seq + 1;
    g_t_seq[l] = g_mf_seq;
    return lb;
}

// ── reports ──────────────────────────────────────────────────────────────────
int mf_once(int e) {
    if (g_mf_quiet != 0 || g_mf_unsafe != 0 || g_mf_live == 0) { return 0; }
    int i = 0;
    while (i < g_mf_nrep) { if (g_mf_rep_tok[i] == g_node[e].tok) { return 0; } i = i + 1; }
    if (g_mf_nrep < 512) { g_mf_rep_tok[g_mf_nrep] = g_node[e].tok; g_mf_nrep = g_mf_nrep + 1; }
    return 1;
}

// the region of the first dangling token in 'bad' (0: freed by dealloc)
int mf_cause(long bad) {
    int t = 0;
    while (t < g_nt) {
        if ((bad & mf_bit(t)) != 0L && g_t_rgn[t] == 0) { return 0; }
        t = t + 1;
    }
    t = 0;
    while (t < g_nt) {
        if ((bad & mf_bit(t)) != 0L) { return g_t_rgn[t]; }
        t = t + 1;
    }
    return 0;
}

void mf_after(int rgn) {
    if (rgn == 0) { msg_lit(" after std::dealloc() freed it"); return; }
    msg_lit(" after arena_reset<"); msg_str(rgn);
    msg_lit(">(), arena_destroy<"); msg_str(rgn);
    msg_lit(">(), or arena_free_to<"); msg_str(rgn);
    msg_lit(">() invalidated it");
}

// a load of a reference value: form 0 a variable, 1 a field/element
void mf_check_load(int e, int ty, long bad, int form, int name) {
    if (bad == 0L) { return; }
    int rt = mf_ref_of(ty);
    if (rt == 0) { return; }
    if (mf_once(e) == 0) { return; }
    int rgn = mf_cause(bad);
    int rr = g_ty[rt].n;
    if (rgn != 0 && rr == R_ARENA) { rgn = g_ty[rt].sid; }
    msg_reset();
    if (form == 0) {
        msg_lit("use of '");
        msg_str(name);
        msg_lit("' ");
        if (rgn == 0 && rr == R_HEAP) { msg_lit("(&heap reference)"); }
        else if (rgn != 0 && rr == R_ARENA) { msg_lit("(&arena<"); msg_str(rgn); msg_lit("> reference)"); }
        else if (rgn == 0) { msg_lit("(a reference into a &heap allocation)"); }
        else { msg_lit("(a reference into arena<"); msg_str(rgn); msg_lit(">)"); }
    } else {
        msg_lit("use of a '");
        if (rgn == 0 && rr == R_HEAP) { msg_lit("&heap T"); }
        else if (rgn != 0 && rr == R_ARENA) { msg_lit("&arena<"); msg_str(rgn); msg_lit("> T"); }
        else { msg_lit("&T"); }
        msg_lit("' value read through a struct field/array element");
    }
    mf_after(rgn);
    error_node(e);
}

// ── expressions ──────────────────────────────────────────────────────────────
long g_ev_dang = 0L;   // dangling part of the value mf_expr returned
int g_ev_mark = -1;    // the mark it carries
int g_ev_var = -1;     // an aggregate value: the variable holding its cells
int g_ev_addr = -1;    // '&w' of local w

long mf_ret(long tok, long dang) {
    g_ev_dang = dang;
    return tok;
}

long mf_expr(int e);
void mf_stmt(int s);

// the cell an lvalue designates: (var, path) in g_lv_var/g_lv_path, or var -1
int g_lv_var = -1;
int g_lv_path = 0;

void mf_lvalue(int e) {
    g_lv_var = -1;
    g_lv_path = 0;
    if (e == 0) { return; }
    int k = g_node[e].kind;
    if (k == E_IDENT) {
        if (g_node[e].c == SYM_LOCAL) { g_lv_var = mf_var_of(g_node[e].d); }
        return;
    }
    if (k == E_MEMBER) {
        int b = g_node[e].a;
        int bt = mf_strip(g_node[b].ty);
        if (bt != 0 && g_ty[bt].kind == TY_SLICE) { mf_expr(b); return; }
        if (g_node[b].kind == E_IDENT && g_node[b].c == SYM_LOCAL) {
            // through a reference variable: a read of that variable
            if (mf_is_agg(g_node[b].ty) == 0) { mf_expr(b); }
            g_lv_var = mf_var_of(g_node[b].d);
            g_lv_path = mf_field_step(g_node[e].sid);
            return;
        }
        if (g_node[b].kind == E_MEMBER || g_node[b].kind == E_INDEX) {
            if (mf_is_agg(g_node[b].ty) == 1) {
                mf_lvalue(b);
                if (g_lv_var >= 0) {
                    int p = mf_join_path(g_lv_path, mf_field_step(g_node[e].sid));
                    if (mf_path_depth(p) > 4) { g_lv_var = -1; return; }
                    g_lv_path = p;
                }
                return;
            }
        }
        mf_expr(b);
        return;
    }
    if (k == E_INDEX) {
        int b = g_node[e].a;
        mf_expr(g_node[e].b);
        if (g_node[b].kind == E_IDENT && g_node[b].c == SYM_LOCAL && g_ty[mf_strip(g_node[b].ty)].kind == TY_ARRAY) {
            g_lv_var = mf_var_of(g_node[b].d);
            g_lv_path = g_mf_elem;
            int ix = g_node[e].b;
            if (g_node[ix].kind == E_INT) {
                tmp_reset();
                tmp_char('[');
                tmp_int(g_node[ix].ival);
                tmp_char(']');
                g_lv_path = intern_tmp();
            }
            return;
        }
        if ((g_node[b].kind == E_MEMBER) && g_ty[mf_strip(g_node[b].ty)].kind == TY_ARRAY) {
            mf_lvalue(b);
            if (g_lv_var >= 0) { g_lv_path = mf_join_path(g_lv_path, g_mf_elem); }
            return;
        }
        mf_expr(b);
        return;
    }
    if (k == E_DEREF) { mf_expr(g_node[e].a); return; }
    mf_expr(e);
}

// stores a value (from mf_expr: tok/dang, g_ev_*) into variable v's own cell
void mf_store_var(int v, long tok, long dang, int mark, int aggvar, int addr) {
    if (v < 0) { return; }
    if (aggvar >= 0) { mf_copy_cells(v, aggvar); return; }
    int fromtmp = (g_mf_tmpv >= 0 && g_ev_var == -2);
    mf_clear_var(v);
    int c = mf_cell(v, 0);
    if (c >= 0) { g_c_tok[c] = tok; g_c_dang[c] = dang; g_c_mark[c] = mark; }
    if (addr >= 0) { g_v_alias[v] = addr; }
    if (fromtmp == 1) {
        // a call result: the fields the callee left in it
        int i = 0;
        int n = g_nc;
        while (i < n) {
            if (g_c_var[i] == g_mf_tmpv && g_c_path[i] != 0) {
                int d = mf_cell(v, g_c_path[i]);
                if (d >= 0) { g_c_tok[d] = g_c_tok[i]; g_c_dang[d] = g_c_dang[i]; }
            }
            i = i + 1;
        }
        g_v_lazy[v] = g_v_lazy[g_mf_tmpv];
    }
}

int is_dealloc_name(int sid) {
    int n = g_str_len[sid];
    if (str_eq_lit(sid, "dealloc") == 1 || str_eq_lit(sid, "dealloc_raw_") == 1) { return 1; }
    if (n > 9) {
        int p = n - 9;
        if (str_char(sid, p) == ':' && str_char(sid, p + 1) == ':' && str_char(sid, p + 2) == 'd' &&
            str_char(sid, p + 3) == 'e' && str_char(sid, p + 8) == 'c') { return 1; }
    }
    if (str_eq_lit(sid, "std_dealloc") == 1) { return 1; }
    return 0;
}

// a call argument: its value and the variable whose cells it designates
#define MF_A 16
long g_a_tok[MF_A];
long g_a_dang[MF_A];
int g_a_mark[MF_A];
int g_a_var[MF_A];

void mf_arg(int i, int a, int by_ref) {
    if (i >= MF_A) { mf_expr(a); return; }
    g_a_var[i] = -1;
    long t = mf_expr(a);
    g_a_tok[i] = t;
    g_a_dang[i] = g_ev_dang;
    g_a_mark[i] = g_ev_mark;
    if (g_ev_var >= 0) { g_a_var[i] = g_ev_var; }
    else if (g_ev_addr >= 0) { g_a_var[i] = g_ev_addr; }
    else if (g_node[a].kind == E_IDENT && g_node[a].c == SYM_LOCAL) {
        int v = mf_var_of(g_node[a].d);
        if (v >= 0 && g_v_alias[v] >= 0) { v = g_v_alias[v]; }
        g_a_var[i] = v;
    } else if (g_node[a].kind == E_CAST && g_node[g_node[a].a].kind == E_IDENT && g_node[g_node[a].a].c == SYM_LOCAL) {
        g_a_var[i] = mf_var_of(g_node[g_node[a].a].d);
    }
    by_ref = by_ref + 0;
}

void mf_summary(int f);

// the value of a summary's kind for this call (0 null, 1 fresh, 2+j param j)
long mf_kind_value(int kind, int site, int rgn, int nargs) {
    g_ev_dang = 0L;
    if (kind == 1) { return mf_fresh(site, rgn); }
    if (kind >= 2) {
        int j = kind - 2;
        if (j < nargs && j < MF_A) { g_ev_dang = g_a_dang[j]; return g_a_tok[j]; }
    }
    return 0L;
}

void mf_call_report(int e, int f, int what, long bad) {
    if (mf_once(e) == 0) { return; }
    int rgn = mf_cause(bad);
    msg_reset();
    msg_lit("call to '");
    msg_str(g_func[f].sid);
    if (what == 1) { msg_lit("' frees memory after std::dealloc() freed it"); error_node(e); return; }
    if (rgn == 0) { msg_lit("' reads a &heap reference after std::dealloc() freed it"); error_node(e); return; }
    msg_lit("' reads an &arena<"); msg_str(rgn); msg_lit("> reference");
    mf_after(rgn);
    error_node(e);
}

long mf_call(int e) {
    int ck = g_node[e].c;
    int f = g_node[e].d;
    int n = g_node[e].nlist;
    int off = 0;
    if (ck == CK_INDIRECT) { mf_expr(g_node[e].a); }
    if (ck == CK_ISNULL || ck == CK_ISNONE || ck == CK_DEFAULT) {
        long v = mf_expr(g_node[e].b);
        long d = g_ev_dang;
        int i = 0;
        while (i < n) { mf_expr(g_list[g_node[e].list + i]); i = i + 1; }
        if (ck == CK_DEFAULT) { return mf_ret(v, d); }
        return mf_ret(0L, 0L);
    }
    if (ck == CK_METHOD) { mf_arg(0, g_node[e].b, 1); off = 1; }
    int i = 0;
    while (i < n) { mf_arg(i + off, g_list[g_node[e].list + i], 0); i = i + 1; }
    int na = n + off;
    g_ev_var = -1;
    g_ev_addr = -1;
    g_ev_mark = -1;
    if ((ck != CK_FUNC && ck != CK_METHOD) || f <= 0) {
        if (ck == CK_INDIRECT && mf_ref_of(g_node[e].ty) != 0) { return mf_ret(mf_fresh(e, mf_rgn_of_type(g_node[e].ty)), 0L); }
        return mf_ret(0L, 0L);
    }
    int callee = g_node[e].a;
    int cname = 0;
    if (callee != 0 && g_node[callee].kind == E_IDENT) { cname = g_node[callee].sid; }
    if (g_func[f].owner == 0 && (is_dealloc_name(g_func[f].origsid) == 1 || (cname != 0 && is_dealloc_name(cname) == 1))) {
        if (na > 0) { mf_free(g_a_tok[0]); }
        return mf_ret(0L, 0L);
    }
    int body = g_node[g_func[f].node].a;
    if (body == 0 || g_tok[g_node[g_func[f].node].tok].file == g_rt_file) {
        // no body (or the runtime): trusted; a fresh, unaliased result
        if (mf_ref_of(g_node[e].ty) != 0) { return mf_ret(mf_fresh(e, mf_rgn_of_type(g_node[e].ty)), 0L); }
        if (mf_is_agg(g_node[e].ty) == 1 && g_mf_tmpv >= 0) {
            mf_clear_var(g_mf_tmpv);
            g_ev_var = g_mf_tmpv;
            return mf_ret(0L, 0L);
        }
        return mf_ret(0L, 0L);
    }
    mf_summary(f);
    // reads of entry memory in the callee are reads here
    i = 0;
    while (i < g_sm_nuse[f]) {
        int p = g_sm_use_p[f * MF_S + i];
        int path = g_sm_use_path[f * MF_S + i];
        int freed = 0;
        int j = 0;
        while (j < g_sm_nfree[f]) { if (g_sm_free_p[f * MF_S + j] == p && g_sm_free_path[f * MF_S + j] == path) { freed = 1; } j = j + 1; }
        if (freed == 0 && p < na && p < MF_A && g_a_var[p] >= 0) {
            int c = mf_cell(g_a_var[p], path);
            if (c >= 0 && g_c_dang[c] != 0L) {
                long bad = g_c_dang[c];
                mf_call_report(e, f, 0, bad);
            }
            // transitively a read of our own entry memory
            if (c >= 0) {
                int t = 0;
                while (t < g_nt) {
                    if ((g_c_tok[c] & mf_bit(t)) != 0L && g_t_kind[t] == 3 && g_t_path[t] != 0) {
                        mf_sm_add(g_mf_f, 1, g_t_param[t], g_t_path[t], 0);
                    }
                    t = t + 1;
                }
            }
        }
        i = i + 1;
    }
    // frees
    long freed_mask = 0L;
    i = 0;
    while (i < g_sm_nfree[f]) {
        int p = g_sm_free_p[f * MF_S + i];
        int path = g_sm_free_path[f * MF_S + i];
        if (p < na && p < MF_A) {
            long m = 0L;
            long d = 0L;
            if (path == 0) { m = g_a_tok[p]; d = g_a_dang[p]; }
            else if (g_a_var[p] >= 0) {
                int c = mf_cell(g_a_var[p], path);
                if (c >= 0) { m = g_c_tok[c]; d = g_c_dang[c]; }
            }
            if (d != 0L && mf_cause(d) == 0) { mf_call_report(e, f, 1, d); }
            freed_mask = freed_mask | m;
        }
        i = i + 1;
    }
    mf_free(freed_mask);
    // resets
    i = 0;
    while (i < g_sm_nreset[f]) {
        int r = g_sm_reset[f * MF_S + i];
        int p = g_sm_reset_p[f * MF_S + i];
        if (p < 0) { mf_reset(r, -1); mf_sm_add(g_mf_f, 3, -1, r, 0); }
        else if (p < na && p < MF_A) {
            int mk = g_a_mark[p];
            if (mk <= -100) { mf_sm_add(g_mf_f, 3, -100 - mk, r, 0); mf_reset(r, 0); }
            else if (mk >= 0) { mf_reset(r, mk); }
            else { mf_reset(r, -1); mf_sm_add(g_mf_f, 3, -1, r, 0); }
        }
        i = i + 1;
    }
    // what the callee left in memory it was passed
    i = 0;
    while (i < g_sm_nwr[f]) {
        int p = g_sm_wr_p[f * MF_S + i];
        int path = g_sm_wr_path[f * MF_S + i];
        int kind = g_sm_wr_kind[f * MF_S + i];
        if (p < na && p < MF_A && g_a_var[p] >= 0) {
            int c = mf_cell(g_a_var[p], path);
            long v = mf_kind_value(kind, e, 0, na);
            long d = g_ev_dang;
            if (c >= 0) { g_c_tok[c] = v; g_c_dang[c] = d; }
        }
        i = i + 1;
    }
    // the result
    int rk = g_sm_ret[f];
    if (mf_is_agg(g_node[e].ty) == 1) {
        if (g_mf_tmpv < 0) { return mf_ret(0L, 0L); }
        mf_clear_var(g_mf_tmpv);
        if (rk == 1) { int l = mf_site_token(e, 0); if (l >= 0) { mf_fresh(e, 0); g_v_lazy[g_mf_tmpv] = l; } }
        i = 0;
        while (i < g_sm_nretf[f]) {
            int c = mf_cell(g_mf_tmpv, g_sm_retf_path[f * MF_S + i]);
            long v = mf_kind_value(g_sm_retf_kind[f * MF_S + i], e, 0, na);
            if (c >= 0) { g_c_tok[c] = v; g_c_dang[c] = g_ev_dang; }
            i = i + 1;
        }
        g_ev_var = g_mf_tmpv;
        return mf_ret(0L, 0L);
    }
    long v = 0L;
    long d = 0L;
    if (rk == 1) { v = mf_fresh(e, mf_rgn_of_type(g_node[e].ty)); }
    else if (rk >= 2) { v = mf_kind_value(rk, e, 0, na); d = g_ev_dang; }
    if (g_sm_nretf[f] > 0 && g_mf_tmpv >= 0) {
        mf_clear_var(g_mf_tmpv);
        i = 0;
        while (i < g_sm_nretf[f]) {
            int c = mf_cell(g_mf_tmpv, g_sm_retf_path[f * MF_S + i]);
            long fv = mf_kind_value(g_sm_retf_kind[f * MF_S + i], e, 0, na);
            if (c >= 0) { g_c_tok[c] = fv; g_c_dang[c] = g_ev_dang; }
            i = i + 1;
        }
        g_ev_var = -2;   // a pointer whose pointee fields are in the temp
    }
    g_ev_dang = d;
    return v;
}

void mf_run_defers(int from, int with_err);

long mf_expr(int e) {
    g_ev_dang = 0L;
    g_ev_mark = -1;
    g_ev_var = -1;
    g_ev_addr = -1;
    if (e == 0) { return 0L; }
    int k = g_node[e].kind;
    if (k == E_IDENT) {
        if (g_node[e].c != SYM_LOCAL) { return 0L; }
        int v = mf_var_of(g_node[e].d);
        if (v < 0) { return 0L; }
        if (mf_is_agg(g_node[e].ty) == 1) { g_ev_var = v; return 0L; }
        int c = mf_cell(v, 0);
        if (c < 0) { return 0L; }
        mf_check_load(e, g_node[e].ty, g_c_dang[c], 0, g_node[e].sid);
        long t = g_c_tok[c];
        long d = g_c_dang[c];
        int mk = g_c_mark[c];
        int al = g_v_alias[v];
        g_ev_dang = d;
        g_ev_mark = mk;
        if (al >= 0) { g_ev_addr = al; }
        return t;
    }
    if (k == E_MEMBER || k == E_INDEX) {
        if (k == E_MEMBER) {
            int bt = mf_strip(g_node[g_node[e].a].ty);
            if (bt != 0 && g_ty[bt].kind == TY_SLICE) {
                long st = mf_expr(g_node[e].a);
                long sd = g_ev_dang;
                g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1;
                if (str_eq_lit(g_node[e].sid, "ptr") == 1) { return mf_ret(st, sd); }
                return mf_ret(0L, 0L);
            }
        }
        mf_lvalue(e);
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1; g_ev_dang = 0L;
        if (g_lv_var < 0) { return 0L; }
        if (mf_is_agg(g_node[e].ty) == 1) { return 0L; }
        int c = mf_cell(g_lv_var, g_lv_path);
        if (c < 0) { return 0L; }
        // a read of entry memory (recorded for the summary)
        int t = 0;
        while (t < g_nt) {
            if ((g_c_tok[c] & mf_bit(t)) != 0L && g_t_kind[t] == 3 && g_t_path[t] != 0 && mf_ref_of(g_node[e].ty) != 0) {
                mf_sm_add(g_mf_f, 1, g_t_param[t], g_t_path[t], 0);
            }
            t = t + 1;
        }
        long lt = g_c_tok[c];
        long ld = g_c_dang[c];
        if (g_lv_path != 0 && str_char(g_lv_path, 0) == '[') {
            // an element: [k] may also be what a store to an unknown index
            // left; [*] may be any element
            int i = 0;
            while (i < g_nc) {
                if (g_c_var[i] == g_lv_var && g_c_path[i] != 0 && str_char(g_c_path[i], 0) == '[' &&
                    (g_lv_path == g_mf_elem || g_c_path[i] == g_mf_elem)) {
                    lt = lt | g_c_tok[i]; ld = ld | g_c_dang[i];
                }
                i = i + 1;
            }
        }
        mf_check_load(e, g_node[e].ty, ld, 1, 0);
        return mf_ret(lt, ld);
    }
    if (k == E_ADDR) {
        int a = g_node[e].a;
        if (g_node[a].kind == E_IDENT && g_node[a].c == SYM_LOCAL) {
            int v = mf_var_of(g_node[a].d);
            g_ev_addr = v;
            return 0L;
        }
        mf_lvalue(a);
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1; g_ev_dang = 0L;
        return 0L;
    }
    if (k == E_NEW) {
        if (g_node[e].a != 0) { mf_expr(g_node[e].a); }
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1;
        return mf_ret(mf_fresh(e, g_node[e].sid), 0L);
    }
    if (k == E_ARENA) {
        int op = g_node[e].op;
        int r = g_node[e].sid;
        if (op == K_ARENA_RESET || op == K_ARENA_DESTROY) {
            mf_reset(r, -1);
            mf_sm_add(g_mf_f, 3, -1, r, 0);
            return mf_ret(0L, 0L);
        }
        if (op == K_ARENA_MARK) { g_ev_mark = g_mf_seq; return 0L; }
        int a = g_node[e].a;
        mf_expr(a);
        int mk = g_ev_mark;
        g_ev_mark = -1;
        if (mk <= -100) { mf_sm_add(g_mf_f, 3, -100 - mk, r, 0); mf_reset(r, 0); }
        else if (mk >= 0) { mf_reset(r, mk); }
        else { mf_reset(r, -1); mf_sm_add(g_mf_f, 3, -1, r, 0); }
        return mf_ret(0L, 0L);
    }
    if (k == E_ASSIGN) {
        int lhs = g_node[e].a;
        long rv = mf_expr(g_node[e].b);
        long rd = g_ev_dang;
        int rm = g_ev_mark;
        int ra = g_ev_var;
        int raddr = g_ev_addr;
        if (g_node[e].op != T_ASSIGN) { mf_expr(lhs); return mf_ret(0L, 0L); }
        if (g_node[lhs].kind == E_IDENT && g_node[lhs].c == SYM_LOCAL) {
            int v = mf_var_of(g_node[lhs].d);
            g_ev_var = ra;
            mf_store_var(v, rv, rd, rm, ra, raddr);
            g_ev_var = -1; g_ev_addr = -1;
            return mf_ret(rv, rd);
        }
        mf_lvalue(lhs);
        if (g_lv_var >= 0) {
            if (ra >= 0 && mf_is_agg(g_node[lhs].ty) == 1) {
                // a struct stored into a field: its cells move under the path
                int i = 0;
                int n = g_nc;
                int pre = g_lv_path;
                int lv = g_lv_var;
                while (i < n) {
                    if (g_c_var[i] == ra && g_c_path[i] != 0) {
                        int c = mf_cell(lv, mf_join_path(pre, g_c_path[i]));
                        if (c >= 0) { g_c_tok[c] = g_c_tok[i]; g_c_dang[c] = g_c_dang[i]; }
                    }
                    i = i + 1;
                }
            } else {
                int c = mf_cell(g_lv_var, g_lv_path);
                if (c >= 0) {
                    if (g_lv_path == g_mf_elem) {
                        // an unknown element: a weak update of every element
                        int i = 0;
                        while (i < g_nc) {
                            if (g_c_var[i] == g_lv_var && g_c_path[i] != 0 && str_char(g_c_path[i], 0) == '[') {
                                g_c_tok[i] = g_c_tok[i] | rv; g_c_dang[i] = g_c_dang[i] | rd;
                            }
                            i = i + 1;
                        }
                    } else {
                        g_c_tok[c] = rv; g_c_dang[c] = rd;
                    }
                }
            }
        }
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1;
        return mf_ret(rv, rd);
    }
    if (k == E_CALL) { return mf_call(e); }
    if (k == E_TERNARY) {
        int agg = mf_is_agg(g_node[e].ty);
        int s = mf_cond(g_node[e].a);
        long a = mf_expr(g_node[e].b);
        long ad = g_ev_dang;
        int am = g_ev_mark;
        int s2 = mf_save();
        mf_restore(s);
        long b = mf_expr(g_node[e].c);
        long bd = g_ev_dang;
        int bm = g_ev_mark;
        mf_join(s2);
        mf_pop(s);
        if (agg == 1 && g_mf_tmpv >= 0) {
            // joined aggregates: one fresh object of this expression
            mf_clear_var(g_mf_tmpv);
            int l = mf_site_token(e, 0);
            if (l >= 0) { mf_fresh(e, 0); g_v_lazy[g_mf_tmpv] = l; }
            g_ev_var = g_mf_tmpv;
            return mf_ret(0L, 0L);
        }
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1;
        if (am >= 0 && bm >= 0) { g_ev_mark = am; if (bm < am) { g_ev_mark = bm; } }
        return mf_ret(a | b, ad | bd);
    }
    if ((k == E_BINARY && (g_node[e].op == T_ANDAND || g_node[e].op == T_OROR)) || (k == E_UNARY && g_node[e].op == T_BANG)) {
        int s = mf_cond(e);
        mf_join(s);
        mf_pop(s);
        return mf_ret(0L, 0L);
    }
    if (k == E_MATCH) {
        mf_expr(g_node[e].a);
        int agg = mf_is_agg(g_node[e].ty);
        int s = mf_save();
        int acc = -1;
        long rv = 0L;
        long rd = 0L;
        int i = 0;
        while (i < g_node[e].nlist) {
            int arm = g_list[g_node[e].list + i];
            mf_restore(s);
            rv = rv | mf_expr(g_node[arm].a);
            rd = rd | g_ev_dang;
            if (acc < 0) { acc = mf_save(); } else { mf_join(acc); mf_pop(acc); acc = mf_save(); }
            i = i + 1;
        }
        if (acc >= 0) { mf_restore(acc); }
        mf_pop(s);
        if (agg == 1 && g_mf_tmpv >= 0) {
            mf_clear_var(g_mf_tmpv);
            int l = mf_site_token(e, 0);
            if (l >= 0) { mf_fresh(e, 0); g_v_lazy[g_mf_tmpv] = l; }
            g_ev_var = g_mf_tmpv;
            return mf_ret(0L, 0L);
        }
        g_ev_var = -1; g_ev_addr = -1; g_ev_mark = -1;
        return mf_ret(rv, rd);
    }
    if (k == E_TRY) {
        long v = mf_expr(g_node[e].a);
        long d = g_ev_dang;
        // the failure path returns: defers and errdefers run there
        int s = mf_save();
        mf_run_defers(0, 1);
        mf_restore(s);
        mf_pop(s);
        return mf_ret(v, d);
    }
    if (k == E_CAST || k == E_FORCE || k == E_DESIG) {
        long v = mf_expr(g_node[e].a);
        return mf_ret(v, g_ev_dang);
    }
    if (k == E_SIZEOF_E || k == E_SIZEOF_T || k == E_ALIGNOF || k == E_SIZEOFPACK) { return 0L; }
    if (k == E_UNARY || k == E_POSTFIX || k == E_DEREF) {
        mf_expr(g_node[e].a);
        return mf_ret(0L, 0L);
    }
    if (k == E_BINARY || k == E_COMMA) {
        mf_expr(g_node[e].a);
        long v = mf_expr(g_node[e].b);
        if (k == E_COMMA) { return mf_ret(v, g_ev_dang); }
        return mf_ret(0L, 0L);
    }
    if (k == E_SLICE) {
        mf_expr(g_node[e].a);
        mf_expr(g_node[e].b);
        mf_expr(g_node[e].c);
        return mf_ret(0L, 0L);
    }
    if (k == E_SPAWN) {
        mf_expr(g_node[e].a);
        mf_expr(g_node[e].b);
        return mf_ret(0L, 0L);
    }
    if (k == E_INITLIST || k == E_TUPLE) {
        int i = 0;
        while (i < g_node[e].nlist) { mf_expr(g_list[g_node[e].list + i]); i = i + 1; }
        return mf_ret(0L, 0L);
    }
    if (k == E_UNIONINIT) { mf_expr(g_node[e].a); return mf_ret(0L, 0L); }
    return 0L;
}

// a condition, split: afterwards the current state is the one where 'e' is
// true, and the returned saved slot (the top of the save stack) the one where
// it is false
int mf_cond(int e) {
    int k = g_node[e].kind;
    if (k == E_BINARY && g_node[e].op == T_ANDAND) {
        int f1 = mf_cond(g_node[e].a);
        int f2 = mf_cond(g_node[e].b);
        int t = mf_save();
        mf_restore(f2);
        mf_join(f1);
        int f = mf_save();
        mf_copy_slot(f1, f);
        mf_restore(t);
        mf_pop(f1 + 1);
        return f1;
    }
    if (k == E_BINARY && g_node[e].op == T_OROR) {
        int f1 = mf_cond(g_node[e].a);
        int ta = mf_save();
        mf_restore(f1);
        mf_cond(g_node[e].b);
        mf_join(ta);
        int t = mf_save();
        // the false state of b sits at ta + 1
        mf_copy_slot(f1, ta + 1);
        mf_restore(t);
        mf_pop(f1 + 1);
        return f1;
    }
    if (k == E_UNARY && g_node[e].op == T_BANG) {
        int f = mf_cond(g_node[e].a);
        int t = mf_save();
        mf_restore(f);
        mf_copy_slot(f, t);
        mf_pop(f + 1);
        return f;
    }
    if (k == E_COMMA) {
        mf_expr(g_node[e].a);
        return mf_cond(g_node[e].b);
    }
    if (k == E_INT || k == E_BOOL) {
        int f = mf_save();
        if (f >= 0) {
            if (g_node[e].ival != 0L) { g_s_live[f] = 0; } else { g_mf_live = 0; }
        }
        return f;
    }
    mf_expr(e);
    return mf_save();
}

// ── statements ───────────────────────────────────────────────────────────────
// runs the pending defers from index 'from' (innermost first); errdefers
// only when with_err
void mf_run_defers(int from, int with_err) {
    int i = g_mfd_n - 1;
    while (i >= from) {
        int d = g_mfd_node[i];
        if ((g_node[d].flags & NF_ERR) == 0 || with_err == 1) {
            int saved = g_mfd_n;
            g_mfd_n = i;            // a defer's own body sees only the outer ones
            mf_stmt(g_node[d].a);
            g_mfd_n = saved;
        }
        i = i - 1;
    }
}

// the function's return value, for its summary
void mf_note_return(int e) {
    if (e == 0) { return; }
    long v = mf_expr(e);
    int ra = g_ev_var;
    int kind = 0;
    int f = g_mf_f;
    int rt = func_ret(f);
    if (mf_is_agg(rt) == 1) {
        if (ra >= 0) {
            // fields of the returned aggregate
            int i = 0;
            int fresh = 0;
            while (i < g_nc) {
                if (g_c_var[i] == ra && g_c_path[i] != 0) {
                    long tv = g_c_tok[i];
                    int fk = -1;
                    int t = 0;
                    while (t < g_nt) {
                        if ((tv & mf_bit(t)) != 0L) {
                            if (g_t_kind[t] == 3 && g_t_path[t] == 0) { fk = 2 + g_t_param[t]; }
                            else { fresh = 1; }
                        }
                        t = t + 1;
                    }
                    if (fk >= 2) { mf_sm_add(f, 4, 0, g_c_path[i], fk); }
                }
                i = i + 1;
            }
            if (fresh == 1 || g_v_lazy[ra] >= 0) { kind = 1; }
        }
    } else {
        int t = 0;
        while (t < g_nt) {
            if ((v & mf_bit(t)) != 0L) {
                if (g_t_kind[t] == 3 && g_t_path[t] == 0) { if (kind == 0) { kind = 2 + g_t_param[t]; } else if (kind != 2 + g_t_param[t]) { kind = 1; } }
                else { kind = 1; }
            }
            t = t + 1;
        }
        // fields of a returned (fresh) pointer
        if (e != 0 && g_node[e].kind == E_IDENT && g_node[e].c == SYM_LOCAL) {
            int rv = mf_var_of(g_node[e].d);
            int i = 0;
            while (rv >= 0 && i < g_nc) {
                if (g_c_var[i] == rv && g_c_path[i] != 0) {
                    long tv = g_c_tok[i];
                    int t2 = 0;
                    while (t2 < g_nt) {
                        if ((tv & mf_bit(t2)) != 0L && g_t_kind[t2] == 3 && g_t_path[t2] == 0) { mf_sm_add(f, 4, 0, g_c_path[i], 2 + g_t_param[t2]); }
                        t2 = t2 + 1;
                    }
                }
                i = i + 1;
            }
        }
    }
    if (g_mf_rkind == -1) { g_mf_rkind = kind; }
    else if (g_mf_rkind != kind) { if (g_mf_rkind == 0) { g_mf_rkind = kind; } else if (kind != 0) { g_mf_rkind = 1; } }
}

// at the end of the function: what it leaves in memory its parameters point to
void mf_note_exit() {
    if (g_mf_live == 0) { return; }
    int f = g_mf_f;
    int i = 0;
    while (i < g_nc) {
        int v = g_c_var[i];
        int path = g_c_path[i];
        if (v < g_mf_np && path != 0 && (g_c_tok[i] != g_c_dtok[i])) {
            int pt = mf_strip(g_local[g_mf_loc0 + v].ty);
            if (pt != 0 && (g_ty[pt].kind == TY_REF || g_ty[pt].kind == TY_PTR || g_ty[pt].kind == TY_OPT)) {
                long tv = g_c_tok[i];
                int kind = 0;
                int t = 0;
                while (t < g_nt) {
                    if ((tv & mf_bit(t)) != 0L) {
                        if (g_t_kind[t] == 3 && g_t_path[t] == 0) { kind = 2 + g_t_param[t]; }
                        else if (g_t_kind[t] != 3) { kind = 1; }
                    }
                    t = t + 1;
                }
                if (tv == 0L || kind != 0) { mf_sm_add(f, 2, v, path, kind); }
            }
        }
        i = i + 1;
    }
}

void mf_list(int s) {
    int mark = g_mfd_n;
    int i = 0;
    while (i < g_node[s].nlist) { mf_stmt(g_list[g_node[s].list + i]); i = i + 1; }
    if (g_mf_live == 1 && g_mfd_n > mark) { mf_run_defers(mark, 0); }
    g_mfd_n = mark;
}

void mf_loop(int loop, int cond, int body, int incr, int init, int is_do) {
    int mark = g_mfd_n;
    if (init != 0) { mf_stmt(init); }
    int saved_outer = g_mf_outer;
    if (g_mf_outer == 0) { g_mf_outer = loop; }
    int label = 0;
    if (g_node[loop].kind != S_DO) { label = g_node[loop].sid; }
    int tg = mf_push_target(loop, label, 0);
    // iterate the head state to a fixpoint (states only grow); the exit is
    // the false-condition state joined with every break
    int iter = 0;
    while (iter < 12) {
        if (g_mf_outer == loop && iter > 0) { mf_fold_iteration(loop); }
        int before = mf_save();
        if (tg >= 0) { mf_slot_clear(mf_slot_brk(tg)); mf_slot_clear(mf_slot_cont(tg)); }
        int f = -1;
        if (cond != 0 && is_do == 0) { f = mf_cond(cond); }
        mf_stmt(body);
        if (tg >= 0) { mf_join(mf_slot_cont(tg)); }
        if (incr != 0) { mf_expr(incr); }
        if (cond != 0 && is_do == 1) { f = mf_cond(cond); }
        mf_join(before);
        int h = mf_save();
        // this iteration's exit state, kept in the break slot
        if (f >= 0) { mf_restore(f); } else { g_mf_live = 0; }
        if (cond == 0 && f < 0) { g_mf_live = 0; }
        if (tg >= 0) { mf_join(mf_slot_brk(tg)); mf_save_at(mf_slot_brk(tg)); }
        mf_restore(h);
        int same = mf_same_as(before);
        mf_pop(before);
        if (same == 1) { break; }
        iter = iter + 1;
    }
    if (tg >= 0) { mf_restore(mf_slot_brk(tg)); }
    g_ntg = g_ntg - 1;
    g_mf_outer = saved_outer;
    g_mfd_n = mark;
}

void mf_bind(int arm, long tok, long dang, int subj_var) {
    int i = 0;
    while (i < g_node[arm].nlist) {
        int p = g_list[g_node[arm].list + i];
        int k = g_node[p].kind;
        if ((k == P_SOME || k == P_VARIANT) && g_node[p].b != 0) {
            int v = mf_var_of(g_node[p].b);
            if (v >= 0) {
                if (mf_is_agg(g_local[g_node[p].b].ty) == 1 && subj_var >= 0) { mf_copy_cells(v, subj_var); }
                else {
                    mf_clear_var(v);
                    int c = mf_cell(v, 0);
                    if (c >= 0) { g_c_tok[c] = tok; g_c_dang[c] = dang; }
                    // the payload's fields are the subject's
                    if (subj_var >= 0 && k == P_SOME) { g_v_alias[v] = -1; }
                }
            }
        }
        i = i + 1;
    }
}

void mf_stmt(int s) {
    if (s == 0) { return; }
    int k = g_node[s].kind;
    if (k == S_BLOCK) { mf_list(s); return; }
    if (k == S_MULTI) {
        int i = 0;
        while (i < g_node[s].nlist) { mf_stmt(g_list[g_node[s].list + i]); i = i + 1; }
        return;
    }
    if (k == S_VAR) {
        int x = mf_var_of(g_node[s].d);
        if (g_node[s].a == 0) { if (x >= 0) { mf_clear_var(x); } return; }
        int init = g_node[s].a;
        if (g_node[init].kind == E_INITLIST) {
            mf_expr(init);
            if (x >= 0) { mf_clear_var(x); }
            return;
        }
        long v = mf_expr(init);
        if (x >= 0) { mf_store_var(x, v, g_ev_dang, g_ev_mark, g_ev_var, g_ev_addr); }
        return;
    }
    if (k == S_EXPR) { mf_expr(g_node[s].a); return; }
    if (k == S_IF) {
        int sv = mf_cond(g_node[s].a);
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
    if (k == S_WHILE) { mf_loop(s, g_node[s].a, g_node[s].b, 0, 0, 0); return; }
    if (k == S_DO) { mf_loop(s, g_node[s].a, g_node[s].b, 0, 0, 1); return; }
    if (k == S_FOR) { mf_loop(s, g_node[s].b, g_node[s].d, g_node[s].c, g_node[s].a, 0); return; }
    if (k == S_RETURN) {
        if (g_node[s].a != 0) { mf_note_return(g_node[s].a); }
        mf_run_defers(0, 0);
        mf_note_exit();
        g_mf_live = 0;
        return;
    }
    if (k == S_BREAK) { mf_jump(1, g_node[s].sid); return; }
    if (k == S_CONTINUE) { mf_jump(0, g_node[s].sid); return; }
    if (k == S_UNSAFE) {
        g_mf_unsafe = g_mf_unsafe + 1;
        mf_stmt(g_node[s].a);
        g_mf_unsafe = g_mf_unsafe - 1;
        return;
    }
    if (k == S_DEFER) {
        if (g_mfd_n < MF_DEF) { g_mfd_node[g_mfd_n] = s; g_mfd_n = g_mfd_n + 1; }
        return;
    }
    if (k == S_MATCH) {
        long st = mf_expr(g_node[s].a);
        long sd = g_ev_dang;
        int sv_var = g_ev_var;
        if (sv_var < 0 && g_node[g_node[s].a].kind == E_IDENT && g_node[g_node[s].a].c == SYM_LOCAL) { sv_var = mf_var_of(g_node[g_node[s].a].d); }
        int sv = mf_save();
        int acc = -1;
        int i = 0;
        while (i < g_node[s].nlist) {
            int arm = g_list[g_node[s].list + i];
            mf_restore(sv);
            mf_bind(arm, st, sd, sv_var);
            mf_stmt(g_node[arm].a);
            if (acc < 0) { acc = mf_save(); } else { mf_join(acc); mf_pop(acc); acc = mf_save(); }
            i = i + 1;
        }
        if (acc >= 0) { mf_restore(acc); mf_join(sv); }
        mf_pop(sv);
        return;
    }
    if (k == S_SWITCH) {
        mf_expr(g_node[s].a);
        int sv = mf_save();
        int tg = mf_push_target(s, 0, 1);
        int has_default = 0;
        int i = 0;
        while (i < g_node[s].nlist) {
            int arm = g_list[g_node[s].list + i];
            if ((g_node[arm].flags & NF_DEFAULT) != 0) { has_default = 1; }
            // entered by its label, or by falling through from the case before
            if (i == 0) { mf_restore(sv); }
            else { int t = mf_save(); mf_restore(sv); mf_join(t); mf_pop(t); }
            mf_stmt(g_node[arm].b);
            i = i + 1;
        }
        if (g_node[s].nlist == 0) { mf_restore(sv); }
        if (tg >= 0) { mf_join(mf_slot_brk(tg)); }
        if (has_default == 0) { mf_join(sv); }
        g_ntg = g_ntg - 1;
        mf_pop(sv);
        return;
    }
    if (k == S_LABEL) { mf_stmt(g_node[s].a); return; }
}

// ── driver ───────────────────────────────────────────────────────────────────
void memflow_function(int f) {
    int body = g_node[g_func[f].node].a;
    if (body == 0) { return; }
    int saved_f = g_mf_f;
    int saved_loc0 = g_mf_loc0;
    g_mf_f = f;
    g_mf_loc0 = g_func[f].loc0;
    g_mf_nv = g_func[f].nloc;
    if (g_mf_nv > MF_V) { g_mf_nv = MF_V; }
    g_mf_np = g_ty[g_func[f].ty].nlist;
    if (g_mf_np > g_mf_nv) { g_mf_np = g_mf_nv; }
    g_mf_tmpv = MF_V;
    int i = 0;
    while (i <= MF_V) { g_v_alias[i] = -1; g_v_lazy[i] = -1; i = i + 1; }
    g_nc = 0;
    g_nt = 0;
    g_mf_outer = 0;
    g_mf_dead = 0L;
    g_mf_seq = 0;
    g_mf_live = 1;
    g_mf_unsafe = 0;
    g_mf_nsave = 0;
    g_ntg = 0;
    g_mfd_n = 0;
    g_mf_rkind = -1;
    mf_stmt(body);
    if (g_mf_live == 1) {
        // falling off the end
        mf_note_exit();
    }
    if (g_mf_rkind < 0) { g_mf_rkind = 0; }
    if (g_sm_ret[f] != g_mf_rkind) { g_sm_ret[f] = g_mf_rkind; g_mf_changed = 1; }
    g_mf_f = saved_f;
    g_mf_loc0 = saved_loc0;
}

int g_mf_round = 0;

// summaries are computed in rounds over every function (memflow_program);
// a callee met here before that just uses what it has so far
void mf_summary(int f) {
    if (g_sm_has[f] == 0) { g_sm_has[f] = 1; }
}

int mf_wants(int f) {
    if (g_func[f].checked == 0 || g_node[g_func[f].node].a == 0 || g_func[f].ty == 0) { return 0; }
    // the runtime prelude is trusted
    if (g_tok[g_node[g_func[f].node].tok].file == g_rt_file) { return 0; }
    return 1;
}

// summaries until they stop changing, then one reporting pass, in
// declaration order
void memflow_program() {
    g_mf_elem = intern_lit("[*]");
    int f = 1;
    while (f <= g_nfunc) {
        g_sm_nfree[f] = 0; g_sm_nuse[f] = 0; g_sm_nwr[f] = 0; g_sm_ret[f] = 0;
        g_sm_nretf[f] = 0; g_sm_nreset[f] = 0; g_sm_has[f] = 0;
        f = f + 1;
    }
    g_mf_quiet = 1;
    int round = 0;
    while (round < 8) {
        g_mf_changed = 0;
        f = 1;
        while (f <= g_nfunc) {
            if (mf_wants(f) == 1) { memflow_function(f); }
            f = f + 1;
        }
        if (g_mf_changed == 0) { break; }
        round = round + 1;
    }
    g_mf_quiet = 0;
    g_mf_nrep = 0;
    f = 1;
    while (f <= g_nfunc) {
        if (mf_wants(f) == 1) { memflow_function(f); }
        f = f + 1;
    }
}
