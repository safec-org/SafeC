// sema.sc — declarations, scopes, type resolution and conversions, and
// statement checking. Expressions are in expr.sc. The rules (and the
// diagnostic wording) follow the C++ safec's Sema.cpp.
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"

struct Func g_func[NFUNC_MAX];
int g_nfunc = 0;
struct Glob g_glob[NGLOB_MAX];
int g_nglob = 0;
struct Local g_local[NLOCAL_MAX];
int g_nlocal = 0;

struct Sym g_sym[NSYM_MAX];
int g_nsym = 0;
int g_depth = 0;
int g_scope_mark[512];
int g_scope_unsafe[512];

int g_gkind[NSTR_MAX];      // global namespace: SYM_* kind by name
int g_gidx[NSTR_MAX];
int g_nsfb_kind[NSTR_MAX];  // unqualified fallback into namespaces
int g_nsfb_idx[NSTR_MAX];
int g_tyreg[NSTR_MAX];      // type names → type
int g_gtmpl[NSTR_MAX];      // generic struct templates: name → D_STRUCT node
int g_trait[NSTR_MAX];      // trait name → D_TRAIT node
int g_region_glob[NSTR_MAX];
int g_region_cap[NSTR_MAX];
int g_nregions = 0;
int g_main_fn = 0;

int g_cur_fn = 0;
int g_loop_depth = 0;
int g_cur_ret = 0;

// method table: (struct def, name) → func
int g_mkey_sd[16384];
int g_mkey_sid[16384];
int g_mval[16384];

// generic environment (type parameter → type), a stack
int g_env_sid[256];
int g_env_ty[256];
int g_nenv = 0;

// generic function instances
int g_inst_tmpl[NFUNC_MAX];
int g_inst_func[NFUNC_MAX];
int g_inst_args[NFUNC_MAX];   // start in g_tyl
int g_inst_nargs[NFUNC_MAX];
int g_ninst = 0;

// check queue
int g_queue[NFUNC_MAX];
int g_nqueue = 0;
int g_qpos = 0;

// borrow records
int g_br_var[4096];
int g_br_borrower[4096];
int g_br_mut[4096];
int g_br_depth[4096];
int g_br_released[4096];
int g_nbr = 0;

// ── type names (C++ Type::str) ───────────────────────────────────────────────
void msg_type(int t) {
    int k = g_ty[t].kind;
    if (k == TY_VOID) { msg_lit("void"); return; }
    if (k == TY_BOOL) { msg_lit("bool"); return; }
    if (k == TY_CHAR) { msg_lit("char"); return; }
    if (k == TY_I8) { msg_lit("int8"); return; }
    if (k == TY_I16) { msg_lit("int16"); return; }
    if (k == TY_I32) { msg_lit("int"); return; }
    if (k == TY_I64) { msg_lit("long"); return; }
    if (k == TY_U8) { msg_lit("uint8"); return; }
    if (k == TY_U16) { msg_lit("uint16"); return; }
    if (k == TY_U32) { msg_lit("unsigned int"); return; }
    if (k == TY_U64) { msg_lit("unsigned long"); return; }
    if (k == TY_F32) { msg_lit("float"); return; }
    if (k == TY_F64) { msg_lit("double"); return; }
    if (k == TY_ERROR) { msg_lit("<error>"); return; }
    if (k == TY_PTR) {
        if ((g_ty[t].flags & TYF_CONST) != 0) { msg_lit("const "); }
        msg_type(g_ty[t].base);
        msg_lit("*");
        return;
    }
    if (k == TY_REF) {
        if ((g_ty[t].flags & TYF_NULL) != 0) { msg_lit("?&"); } else { msg_lit("&"); }
        int r = g_ty[t].n;
        if (r == R_ARENA) { msg_lit("arena<"); msg_str(g_ty[t].sid); msg_lit("> "); }
        else if (r == R_STACK) { msg_lit("stack "); }
        else if (r == R_HEAP) { msg_lit("heap "); }
        else if (r == R_STATIC) { msg_lit("static "); }
        if ((g_ty[t].flags & TYF_CONST) != 0) { msg_lit("const "); }
        msg_type(g_ty[t].base);
        return;
    }
    if (k == TY_ARRAY) {
        msg_type(g_ty[t].base);
        if (g_ty[t].n < 0) { msg_lit("[]"); }
        else { msg_lit("["); msg_int((long)g_ty[t].n); msg_lit("]"); }
        return;
    }
    if (k == TY_SLICE) { msg_lit("[]"); msg_type(g_ty[t].base); return; }
    if (k == TY_OPT) { msg_lit("?"); msg_type(g_ty[t].base); return; }
    if (k == TY_FN) {
        msg_type(g_ty[t].base);
        msg_lit("(");
        int i = 0;
        while (i < g_ty[t].nlist) {
            if (i > 0) { msg_lit(", "); }
            msg_type(g_tyl[g_ty[t].list + i]);
            i = i + 1;
        }
        if ((g_ty[t].flags & TYF_VARIADIC) != 0) {
            if (g_ty[t].nlist > 0) { msg_lit(", "); }
            msg_lit("...");
        }
        msg_lit(")");
        return;
    }
    if (k == TY_TUPLE) {
        msg_lit("(");
        int i = 0;
        while (i < g_ty[t].nlist) {
            if (i > 0) { msg_lit(", "); }
            msg_type(g_tyl[g_ty[t].list + i]);
            i = i + 1;
        }
        msg_lit(")");
        return;
    }
    if (k == TY_STRUCT || k == TY_ENUM || k == TY_NEWTYPE) { msg_str(g_sdef[g_ty[t].n].sid); return; }
    if (k == TY_NAMED || k == TY_GENERIC) { msg_str(g_ty[t].sid); return; }
    if (k == TY_TYPEOF) { msg_lit("typeof(...)"); return; }
    msg_lit("?");
}

// ── scopes ───────────────────────────────────────────────────────────────────
void push_scope(int is_unsafe) {
    g_depth = g_depth + 1;
    if (g_depth >= 512) { fatal("scope depth"); g_depth = 511; }
    g_scope_mark[g_depth] = g_nsym;
    g_scope_unsafe[g_depth] = is_unsafe;
}

void pop_scope() {
    // borrows taken in this scope end with it
    int i = 0;
    int j = 0;
    while (i < g_nbr) {
        if (g_br_depth[i] < g_depth) {
            g_br_var[j] = g_br_var[i];
            g_br_borrower[j] = g_br_borrower[i];
            g_br_mut[j] = g_br_mut[i];
            g_br_depth[j] = g_br_depth[i];
            g_br_released[j] = g_br_released[i];
            j = j + 1;
        }
        i = i + 1;
    }
    g_nbr = j;
    g_nsym = g_scope_mark[g_depth];
    g_depth = g_depth - 1;
}

int in_unsafe() {
    int d = g_depth;
    while (d > 0) {
        if (g_scope_unsafe[d] == 1) { return 1; }
        d = d - 1;
    }
    return 0;
}

int define_sym(int sid, int kind, int idx, int ty, int init) {
    if (g_nsym >= NSYM_MAX) { fatal("symbols"); return 0; }
    int s = g_nsym;
    g_sym[s].sid = sid;
    g_sym[s].kind = kind;
    g_sym[s].idx = idx;
    g_sym[s].ty = ty;
    g_sym[s].depth = g_depth;
    g_sym[s].init = init;
    g_nsym = s + 1;
    return s;
}

int lookup_local(int sid) {
    int i = g_nsym - 1;
    while (i >= 0) {
        if (g_sym[i].sid == sid) { return i; }
        i = i - 1;
    }
    return -1;
}

int new_local(int sid, int ty, int flags, int node) {
    if (g_nlocal >= NLOCAL_MAX) { fatal("locals"); return 0; }
    int l = g_nlocal;
    g_local[l].sid = sid;
    g_local[l].ty = ty;
    g_local[l].flags = flags;
    g_local[l].wl = -1;
    g_local[l].off = 0;
    g_local[l].depth = g_depth;
    g_local[l].node = node;
    g_nlocal = l + 1;
    if (g_cur_fn != 0) { g_func[g_cur_fn].nloc = g_func[g_cur_fn].nloc + 1; }
    return l;
}

// ── generic environment ──────────────────────────────────────────────────────
int env_lookup(int sid) {
    int i = g_nenv - 1;
    while (i >= 0) {
        if (g_env_sid[i] == sid) { return g_env_ty[i]; }
        i = i - 1;
    }
    return 0;
}

void env_push(int sid, int ty) {
    if (g_nenv >= 256) { fatal("generic environment"); return; }
    g_env_sid[g_nenv] = sid;
    g_env_ty[g_nenv] = ty;
    g_nenv = g_nenv + 1;
}

// ── struct registry helpers ──────────────────────────────────────────────────
int new_sdef(int kind, int sid, int node) {
    if (g_nsdef + 1 >= NSTRUCT_MAX) { fatal("structs"); return 1; }
    g_nsdef = g_nsdef + 1;
    int sd = g_nsdef;
    g_sdef[sd].kind = kind;
    g_sdef[sd].sid = sid;
    g_sdef[sd].node = node;
    g_sdef[sd].field0 = g_nfield;
    g_sdef[sd].nfield = 0;
    g_sdef[sd].done = 0;
    g_sdef[sd].base = TY_I32;
    g_sdef[sd].ty = ty_struct(sd);
    return sd;
}

int new_field(int sid, int ty, int node) {
    if (g_nfield >= NFIELD_MAX) { fatal("fields"); return 0; }
    int f = g_nfield;
    g_field[f].sid = sid;
    g_field[f].ty = ty;
    g_field[f].off = 0;
    g_field[f].node = node;
    g_field[f].val = 0L;
    g_nfield = f + 1;
    return f;
}

int find_field(int sd, int sid) {
    int i = 0;
    while (i < g_sdef[sd].nfield) {
        if (g_field[g_sdef[sd].field0 + i].sid == sid) { return g_sdef[sd].field0 + i; }
        i = i + 1;
    }
    return -1;
}

int mhash(int sd, int sid) { return ilog_hash(sd, sid) & 16383; }

int lookup_method(int sd, int sid) {
    int h = mhash(sd, sid);
    int n = 0;
    while (g_mkey_sd[h] != 0 && n < 16384) {
        if (g_mkey_sd[h] == sd && g_mkey_sid[h] == sid) { return g_mval[h]; }
        h = (h + 1) & 16383;
        n = n + 1;
    }
    return 0;
}

void set_method(int sd, int sid, int f) {
    int h = mhash(sd, sid);
    int n = 0;
    while (g_mkey_sd[h] != 0 && n < 16384) {
        if (g_mkey_sd[h] == sd && g_mkey_sid[h] == sid) { g_mval[h] = f; return; }
        h = (h + 1) & 16383;
        n = n + 1;
    }
    g_mkey_sd[h] = sd;
    g_mkey_sid[h] = sid;
    g_mval[h] = f;
}

int struct_of(int t) {
    if (g_ty[t].kind == TY_STRUCT) { return g_ty[t].n; }
    return 0;
}

// struct def behind T, &T, ?&T or T*
int deref_struct(int t) {
    int k = g_ty[t].kind;
    if (k == TY_REF || k == TY_PTR) { t = g_ty[t].base; }
    return struct_of(t);
}

int new_func(int sid, int node) {
    if (g_nfunc + 1 >= NFUNC_MAX) { fatal("functions"); return 1; }
    g_nfunc = g_nfunc + 1;
    int f = g_nfunc;
    g_func[f].sid = sid;
    g_func[f].origsid = sid;
    g_func[f].node = node;
    g_func[f].ty = 0;
    g_func[f].owner = 0;
    g_func[f].flags = 0;
    if (node != 0) { g_func[f].flags = g_node[node].flags; }
    g_func[f].used = 0;
    g_func[f].widx = -1;
    g_func[f].import = 0;
    g_func[f].loc0 = 0;
    g_func[f].nloc = 0;
    g_func[f].env = 0;
    g_func[f].checked = 0;
    g_func[f].tmpl = 0;
    g_func[f].ns = 0;
    return f;
}

int fn_type_of(int t) {
    // function value types are '&static fn(...)'
    if (g_ty[t].kind == TY_REF || g_ty[t].kind == TY_PTR) { t = g_ty[t].base; }
    if (g_ty[t].kind == TY_FN) { return t; }
    return 0;
}

int func_ret(int f) { return g_ty[g_func[f].ty].base; }
int func_nparams(int f) { return g_ty[g_func[f].ty].nlist; }
int func_param(int f, int i) { return g_tyl[g_ty[g_func[f].ty].list + i]; }

// ── type resolution ──────────────────────────────────────────────────────────
int instantiate_struct(int tmpl_node, int mark);
int sema_check_expr_ty(int e);

int resolve_type(int t) {
    if (t == 0) { return 0; }
    int k = g_ty[t].kind;
    if (k <= TY_F64) { return t; }
    if (k == TY_NAMED) {
        int sid = g_ty[t].sid;
        if (g_ty[t].nlist == 0) {
            int e = env_lookup(sid);
            if (e != 0) { return e; }
            int r = g_tyreg[sid];
            if (r != 0) { return r; }
            if (g_gtmpl[sid] != 0) { return t; }
            // an undeclared name is an (empty) struct, as in the C++ compiler:
            // errors surface where its fields are used
            int sd = new_sdef(SD_STRUCT, sid, 0);
            g_tyreg[sid] = g_sdef[sd].ty;
            return g_sdef[sd].ty;
        }
        // instantiation of a generic struct
        int args[16];
        int na = g_ty[t].nlist;
        if (na > 16) { na = 16; }
        int i = 0;
        while (i < na) { args[i] = resolve_type(g_tyl[g_ty[t].list + i]); i = i + 1; }
        int gen = 0;
        i = 0;
        while (i < na) { if (mentions_generic(args[i]) == 1) { gen = 1; } i = i + 1; }
        int mark = tyl_begin();
        if (g_gtmpl[sid] == 0 || gen == 1) {
            i = 0;
            while (i < na) { tyl_push(args[i]); i = i + 1; }
            return ty_named(sid, mark, g_ty[t].flags);
        }
        i = 0;
        while (i < na) { tyl_push(args[i]); i = i + 1; }
        return instantiate_struct(g_gtmpl[sid], mark);
    }
    if (k == TY_GENERIC) {
        int e = env_lookup(g_ty[t].sid);
        if (e != 0) { return e; }
        return t;
    }
    if (k == TY_PTR) { return ty_ptr(resolve_type(g_ty[t].base), g_ty[t].flags); }
    if (k == TY_REF) { return ty_ref(resolve_type(g_ty[t].base), g_ty[t].n, g_ty[t].sid, g_ty[t].flags); }
    if (k == TY_ARRAY) {
        int n = g_ty[t].n;
        if (n <= -10) {
            int e = -10 - n;
            if (sema_const_int(e) == 1) {
                n = (int)g_cval;
                if (n < 0) { msg_reset(); msg_lit("array size must be non-negative"); error_node(e); n = 0; }
            } else {
                g_ce_bad = 0;
                sema_const_int(e);
                const_error(e);
                n = 1;
            }
        }
        return ty_array(resolve_type(g_ty[t].base), n);
    }
    if (k == TY_SLICE) { return ty_slice(resolve_type(g_ty[t].base)); }
    if (k == TY_OPT) { return ty_opt(resolve_type(g_ty[t].base)); }
    if (k == TY_FN || k == TY_TUPLE) {
        int ps[64];
        int np = g_ty[t].nlist;
        if (np > 64) { np = 64; }
        int i = 0;
        while (i < np) { ps[i] = resolve_type(g_tyl[g_ty[t].list + i]); i = i + 1; }
        int ret = 0;
        if (k == TY_FN) { ret = resolve_type(g_ty[t].base); }
        int mark = tyl_begin();
        i = 0;
        while (i < np) { tyl_push(ps[i]); i = i + 1; }
        if (k == TY_FN) { return ty_make(TY_FN, ret, 0, 0, g_ty[t].flags, mark); }
        return ty_tuple(mark);
    }
    if (k == TY_TYPEOF) { return sema_check_expr_ty(g_ty[t].n); }
    return t;
}

// does a (resolved) type still mention a type parameter?
int mentions_generic(int t) {
    int k = g_ty[t].kind;
    if (k == TY_GENERIC) { return 1; }
    if (k == TY_NAMED) {
        if (g_tyreg[g_ty[t].sid] == 0 && g_ty[t].nlist == 0) { return 1; }
        int i = 0;
        while (i < g_ty[t].nlist) { if (mentions_generic(g_tyl[g_ty[t].list + i]) == 1) { return 1; } i = i + 1; }
        return 0;
    }
    if (k == TY_PTR || k == TY_REF || k == TY_ARRAY || k == TY_SLICE || k == TY_OPT) { return mentions_generic(g_ty[t].base); }
    if (k == TY_FN || k == TY_TUPLE) {
        if (k == TY_FN && mentions_generic(g_ty[t].base) == 1) { return 1; }
        int i = 0;
        while (i < g_ty[t].nlist) { if (mentions_generic(g_tyl[g_ty[t].list + i]) == 1) { return 1; } i = i + 1; }
    }
    return 0;
}

// ── generic struct instantiation ─────────────────────────────────────────────
int g_sinst_tmpl[NSTRUCT_MAX];
int g_sinst_sd[NSTRUCT_MAX];
int g_nsinst = 0;

void collect_struct_members(int sd, int node);

int g_tyloc = 0;   // node a type error is reported at (the declaration being checked)
int g_ce_struct = 0;   // constraint_error: "... of struct 'S'"
int trait_mismatch(int t, int trn);
void constraint_error(int node, int t, int trn, int gp);

int instantiate_struct(int tmpl_node, int mark) {
    int na = g_ntyl - mark;
    // existing instance?
    int i = 0;
    while (i < g_nsinst) {
        if (g_sinst_tmpl[i] == tmpl_node) {
            int sd = g_sinst_sd[i];
            if (g_sdef[sd].ntargs == na) {
                int same = 1;
                int j = 0;
                while (j < na) {
                    if (g_tyl[g_sdef[sd].targs + j] != g_tyl[mark + j]) { same = 0; break; }
                    j = j + 1;
                }
                if (same == 1) { g_ntyl = mark; return g_sdef[sd].ty; }
            }
        }
        i = i + 1;
    }
    int gp = g_node[tmpl_node].b;
    int np = g_node[gp].nlist;
    if (np != na) {
        msg_reset();
        msg_lit("generic struct '");
        msg_str(g_node[tmpl_node].sid);
        msg_lit("' expects ");
        msg_int((long)np);
        msg_lit(" type argument(s), got ");
        msg_int((long)na);
        error_node(tmpl_node);
        g_ntyl = mark;
        return TY_ERROR;
    }
    // name: Vec_int
    tmp_reset();
    tmp_str(g_node[tmpl_node].sid);
    int j = 0;
    while (j < na) {
        tmp_char('_');
        msg_reset();
        msg_type(g_tyl[mark + j]);
        int q = 0;
        while (q < g_msg_len) {
            int c = (int)g_msg[q];
            if (c == ' ' || c == '*' || c == '&' || c == '<' || c == '>' || c == ',' || c == '?' || c == '[' || c == ']' || c == '(' || c == ')') { c = '_'; }
            tmp_char(c);
            q = q + 1;
        }
        j = j + 1;
    }
    int name = intern_tmp();
    // bounds: 'generic<T: Shape> struct Holder'
    j = 0;
    while (j < na) {
        int gpj = g_list[g_node[gp].list + j];
        int bl = g_node[gpj].b;
        int q = 0;
        while (bl != 0 && q < g_node[bl].nlist) {
            int trn = g_node[g_list[g_node[bl].list + q]].sid;
            if (trait_mismatch(g_tyl[mark + j], trn) == 1) {
                g_ce_struct = g_node[tmpl_node].sid;
                int loc = g_tyloc;
                if (loc == 0) { loc = tmpl_node; }
                constraint_error(loc, g_tyl[mark + j], trn, g_node[gpj].sid);
                g_ce_struct = 0;
            }
            q = q + 1;
        }
        j = j + 1;
    }
    int kind = SD_STRUCT;
    if ((g_node[tmpl_node].flags & NF_UNION) != 0) { kind = SD_UNION; }
    int sd = new_sdef(kind, name, tmpl_node);
    g_sdef[sd].tmpl = tmpl_node;
    g_sdef[sd].targs = mark;
    g_sdef[sd].ntargs = na;
    if (g_nsinst < NSTRUCT_MAX) { g_sinst_tmpl[g_nsinst] = tmpl_node; g_sinst_sd[g_nsinst] = sd; g_nsinst = g_nsinst + 1; }
    g_tyreg[name] = g_sdef[sd].ty;
    // fields with T bound
    int saved_env = g_nenv;
    j = 0;
    while (j < na) {
        env_push(g_node[g_list[g_node[gp].list + j]].sid, g_tyl[mark + j]);
        j = j + 1;
    }
    collect_struct_members(sd, tmpl_node);
    g_nenv = saved_env;
    return g_sdef[sd].ty;
}

// ── conversions (C++ Sema::canImplicitlyConvert & friends) ───────────────────
int ty_equal(int a, int b) {
    if (a == b) { return 1; }
    // structural equality of interned types: identical ids, except the
    // const flag on pointers is ignored (PointerType::equals)
    if (g_ty[a].kind == TY_PTR && g_ty[b].kind == TY_PTR) { return ty_equal(g_ty[a].base, g_ty[b].base); }
    return 0;
}

int is8(int t) { int k = g_ty[t].kind; return k == TY_CHAR || k == TY_I8 || k == TY_U8; }
int is8b(int t) { int k = g_ty[t].kind; return k == TY_CHAR || k == TY_I8 || k == TY_U8 || k == TY_BOOL; }
int is_void(int t) { return g_ty[t].kind == TY_VOID; }
int is_err(int t) { return t == 0 || g_ty[t].kind == TY_ERROR; }

int fn_of(int t) {
    if (g_ty[t].kind == TY_FN) { return t; }
    if (g_ty[t].kind == TY_REF && g_ty[g_ty[t].base].kind == TY_FN) { return g_ty[t].base; }
    return 0;
}

int slot_pointee(int t) {
    if (g_ty[t].kind == TY_PTR) { return g_ty[t].base; }
    if (g_ty[t].kind == TY_REF && g_ty[t].n == R_EXTERN) { return g_ty[t].base; }
    return 0;
}

int same_slot(int a, int b) {
    if (ty_equal(a, b) == 1) { return 1; }
    int pa = slot_pointee(a);
    int pb = slot_pointee(b);
    if (pa == 0 || pb == 0) { return 0; }
    if (g_ty[a].kind == g_ty[b].kind) { return 0; }
    return ty_equal(pa, pb) == 1 || is_void(pa) == 1 || is_void(pb) == 1;
}

int can_convert(int from, int to) {
    if (from == 0 || to == 0) { return 0; }
    if (is_err(from) == 1 || is_err(to) == 1) { return 1; }
    if (ty_equal(from, to) == 1) { return 1; }
    int fk = g_ty[from].kind;
    int tk = g_ty[to].kind;
    int ff = fn_of(from);
    int tf = fn_of(to);
    if (ff != 0 && tf != 0) {
        if ((g_ty[ff].flags & TYF_VARIADIC) == (g_ty[tf].flags & TYF_VARIADIC) &&
            g_ty[ff].nlist == g_ty[tf].nlist && same_slot(g_ty[ff].base, g_ty[tf].base) == 1) {
            int ok = 1;
            int i = 0;
            while (i < g_ty[ff].nlist && ok == 1) {
                ok = same_slot(g_tyl[g_ty[ff].list + i], g_tyl[g_ty[tf].list + i]);
                i = i + 1;
            }
            if (ok == 1) { return 1; }
        }
    }
    if (is8b(from) == 1 && is8b(to) == 1) { return 1; }
    if (fk == TY_CHAR && ty_is_int(to) == 1) { return 1; }
    if (ty_is_int(from) == 1 && ty_is_int(to) == 1) {
        if (ty_bits(to) >= ty_bits(from)) { return 1; }
    }
    if (ty_is_float(from) == 1 && ty_is_float(to) == 1) {
        if (ty_bits(to) >= ty_bits(from)) { return 1; }
    }
    if (ty_is_int(from) == 1 && ty_is_float(to) == 1) { return 1; }
    if (fk == TY_ARRAY && tk == TY_PTR) {
        if (ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
        if (is8(g_ty[from].base) == 1 && is8(g_ty[to].base) == 1) { return 1; }
    }
    if (fk == TY_ARRAY && tk == TY_REF) {
        if (ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
    }
    if (fk == TY_PTR && tk == TY_ARRAY) {
        if (ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
        if (is8(g_ty[from].base) == 1 && is8(g_ty[to].base) == 1) { return 1; }
    }
    if (fk == TY_ARRAY && tk == TY_ARRAY) {
        if (g_ty[to].n == -1 && ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
    }
    if (fk == TY_BOOL && ty_is_int(to) == 1) { return 1; }
    if (ty_is_int(from) == 1 && tk == TY_BOOL) { return 1; }
    if (fk == TY_PTR && tk == TY_PTR) {
        if (is_void(g_ty[from].base) == 1 || is_void(g_ty[to].base) == 1) { return 1; }
    }
    if (fk == TY_PTR && tk == TY_REF) {
        int nul = g_ty[to].flags & TYF_NULL;
        if (is_void(g_ty[from].base) == 1 && nul != 0) { return 1; }
        if (nul == 0 && in_unsafe() == 0) { return 0; }
        if (ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
        if (is8(g_ty[from].base) == 1 && is8(g_ty[to].base) == 1) { return 1; }
        return 0;
    }
    if (tk == TY_OPT) {
        if (ty_equal(from, g_ty[to].base) == 1) { return 1; }
        if (fk == TY_PTR && is_void(g_ty[from].base) == 1) { return 1; }
    }
    if (fk == TY_REF && tk == TY_PTR) {
        int r = g_ty[from].n;
        int same = ty_equal(g_ty[from].base, g_ty[to].base);
        int tv = is_void(g_ty[to].base);
        if (r == R_STATIC && (same == 1 || tv == 1)) { return 1; }
        if ((r == R_ARENA || r == R_HEAP) && (same == 1 || tv == 1)) { return 1; }
        if (r == R_EXTERN && (same == 1 || tv == 1)) { return 1; }
    }
    if (tk == TY_NEWTYPE) {
        int nb = g_sdef[g_ty[to].n].base;
        if (ty_equal(from, nb) == 1) { return 1; }
        if (can_convert(from, nb) == 1) { return 1; }
    }
    if (fk == TY_NEWTYPE) {
        if (ty_equal(g_sdef[g_ty[from].n].base, to) == 1) { return 1; }
    }
    if (fk == TY_ENUM && ty_is_int(to) == 1) { return 1; }
    if (ty_is_int(from) == 1 && tk == TY_ENUM) { return 1; }
    if (fk == TY_REF && tk == TY_REF) {
        if (ty_equal(g_ty[from].base, g_ty[to].base) == 0) { return 0; }
        int fnul = g_ty[from].flags & TYF_NULL;
        int tnul = g_ty[to].flags & TYF_NULL;
        if (g_ty[to].n == R_EXTERN) {
            if (fnul != 0 && tnul == 0) { return 0; }
            return 1;
        }
        if (g_ty[from].n != g_ty[to].n) { return 0; }
        if (fnul == 0 && tnul != 0) { return 1; }
        if (fnul != 0 && tnul == 0) { return 0; }
        return 1;
    }
    return 0;
}

int int_lit_fits(int e, int to) {
    if (to == 0) { return 0; }
    int neg = 0;
    if (g_node[e].kind == E_UNARY && g_node[e].op == T_MINUS) { e = g_node[e].a; neg = 1; }
    if (g_node[e].kind != E_INT) { return 0; }
    int k = g_ty[to].kind;
    if (!(k == TY_CHAR || k == TY_BOOL || (k >= TY_I8 && k <= TY_U64))) { return 0; }
    long v = g_node[e].ival;
    if (neg == 1) { v = 0L - v; }
    int bits = ty_bits(to);
    if (bits == 0 || bits >= 64) { return 1; }
    if (ty_is_signed(to) == 1) {
        long lo = 0L - (1L << (long)(bits - 1));
        long hi = (1L << (long)(bits - 1)) - 1L;
        return v >= lo && v <= hi;
    }
    if (v < 0L) { return 0; }
    long hu = (1L << (long)bits) - 1L;
    return v <= hu;
}

int float_lit_fits(int e, int to) {
    if (to == 0) { return 0; }
    if (g_ty[to].kind != TY_F32 && g_ty[to].kind != TY_F64) { return 0; }
    if (g_node[e].kind == E_UNARY && g_node[e].op == T_MINUS) { e = g_node[e].a; }
    return g_node[e].kind == E_FLOAT;
}

int ref_to_ptr_arg(int from, int to) {
    if (g_ty[from].kind != TY_REF || g_ty[to].kind != TY_PTR) { return 0; }
    if (ty_equal(g_ty[from].base, g_ty[to].base) == 1) { return 1; }
    return is8(g_ty[from].base) == 1 && is8(g_ty[to].base) == 1;
}

// expression e (of type from) may initialize/assign/return into 'to'
int convertible_to(int e, int from, int to) {
    if (is_err(from) == 1 || is_err(to) == 1) { return 1; }
    if (can_convert(from, to) == 1) { return 1; }
    if (int_lit_fits(e, to) == 1) { return 1; }
    if (float_lit_fits(e, to) == 1) { return 1; }
    return 0;
}

void msg_raw_ref_hint(int from, int to) {
    if (in_unsafe() == 1) { return; }
    if (g_ty[from].kind != TY_PTR || g_ty[to].kind != TY_REF) { return; }
    if ((g_ty[to].flags & TYF_NULL) != 0) { return; }
    msg_lit(" (a raw pointer becomes a non-null reference only inside 'unsafe'; convert it to '?");
    msg_type(to);
    msg_lit("' and check it with 'match')");
}

// ── borrow checker ───────────────────────────────────────────────────────────
void track_ref(int var, int is_mut, int borrower, int node) {
    int i = 0;
    while (i < g_nbr) {
        if (g_br_var[i] == var && g_br_released[i] == 0 && g_br_depth[i] >= g_depth) {
            if (is_mut == 1 || g_br_mut[i] == 1) {
                msg_reset();
                msg_lit("cannot borrow '");
                msg_str(var);
                msg_lit("' as ");
                if (is_mut == 1) { msg_lit("mutable"); } else { msg_lit("immutable"); }
                msg_lit(": already borrowed");
                if (g_br_mut[i] == 1) { msg_lit(" as mutable"); }
                error_node(node);
                return;
            }
        }
        i = i + 1;
    }
    if (g_nbr >= 4096) { return; }
    g_br_var[g_nbr] = var;
    g_br_borrower[g_nbr] = borrower;
    g_br_mut[g_nbr] = is_mut;
    g_br_depth[g_nbr] = g_depth;
    g_br_released[g_nbr] = 0;
    g_nbr = g_nbr + 1;
}

void release_borrows(int borrower) {
    int i = 0;
    while (i < g_nbr) {
        if (g_br_borrower[i] == borrower) { g_br_released[i] = 1; }
        i = i + 1;
    }
}

// ── declarations: collection ─────────────────────────────────────────────────
int func_type_from_node(int fnode, int self_ty) {
    int ps[64];
    int np = 0;
    if (self_ty != 0) { ps[0] = self_ty; np = 1; }
    int i = 0;
    while (i < g_node[fnode].nlist && np < 64) {
        int p = g_list[g_node[fnode].list + i];
        int pt = resolve_type(g_node[p].c);
        // array parameters decay to pointers (as in C)
        if (g_ty[pt].kind == TY_ARRAY) { pt = ty_ptr(g_ty[pt].base, 0); }
        ps[np] = pt;
        np = np + 1;
        i = i + 1;
    }
    int ret = resolve_type(g_node[fnode].c);
    int mark = tyl_begin();
    i = 0;
    while (i < np) { tyl_push(ps[i]); i = i + 1; }
    int variadic = 0;
    if ((g_node[fnode].flags & NF_VARIADIC) != 0) { variadic = 1; }
    return ty_fn(ret, mark, variadic);
}

int self_type(int sd, int fnode) {
    int fl = 0;
    if ((g_node[fnode].flags & NF_MCONST) != 0) { fl = TYF_CONST; }
    return ty_ref(g_sdef[sd].ty, R_STACK, 0, fl);
}

// ── out-of-line method definitions vs. their struct-body declarations ───────
// the declaration of method 'name' in struct sd's body, or 0
int body_decl(int sd, int name) {
    int sn = g_sdef[sd].node;
    if (sn == 0) { return 0; }
    int i = 0;
    while (i < g_node[sn].nlist) {
        int m = g_list[g_node[sn].list + i];
        if (g_node[m].kind == D_FUNC && g_node[m].sid == name) { return m; }
        i = i + 1;
    }
    return 0;
}

int ngparams(int fnode) {
    if (g_node[fnode].b == 0) { return 0; }
    return g_node[g_node[fnode].b].nlist;
}

// t's spelling with every "const " removed, interned
int bare_type_sid(int t) {
    msg_reset();
    msg_type(t);
    tmp_reset();
    int i = 0;
    while (i < g_msg_len) {
        if (i + 6 <= g_msg_len && g_msg[i] == 'c' && g_msg[i + 1] == 'o' && g_msg[i + 2] == 'n' && g_msg[i + 3] == 's' &&
            g_msg[i + 4] == 't' && g_msg[i + 5] == ' ') { i = i + 6; continue; }
        tmp_char((int)g_msg[i]);
        i = i + 1;
    }
    return intern_tmp();
}

int fn_type_in_pattern_env(int fnode) {
    int saved = g_nenv;
    int gpn = g_node[fnode].b;
    int i = 0;
    while (gpn != 0 && i < g_node[gpn].nlist) {
        int gs = g_node[g_list[g_node[gpn].list + i]].sid;
        env_push(gs, ty_generic(gs));
        i = i + 1;
    }
    int ft = func_type_from_node(fnode, 0);
    g_nenv = saved;
    return ft;
}

void def_mismatch(int def, int sd, int name) {
    msg_reset();
    msg_lit("definition of '");
    msg_str(g_sdef[sd].sid);
    msg_lit("::");
    msg_str(name);
    msg_lit("' doesn't match its declaration in the struct body: ");
    int i = 0;
    while (i < g_tmp_len) { msg_char((int)g_tmp[i]); i = i + 1; }
    error_node(def);
}

void check_method_def(int sd, int def) {
    int name = g_node[def].sid;
    int d = body_decl(sd, name);
    if (d == 0 || d == def) { return; }
    if (ngparams(d) != ngparams(def)) {
        msg_reset();
        msg_lit("'");
        msg_str(g_sdef[sd].sid);
        msg_lit("::");
        msg_str(name);
        msg_lit("' is declared with ");
        msg_int((long)ngparams(d));
        msg_lit(" type parameter(s) in the struct body but defined with ");
        msg_int((long)ngparams(def));
        error_node(def);
        return;
    }
    int ds = (g_node[d].flags & NF_SMETHOD) != 0;
    int fs = (g_node[def].flags & NF_SMETHOD) != 0;
    if (ds == 1) { fs = 1; }   // a definition of a static method is static
    if (ds != fs) { tmp_reset(); tmp_lit("it is declared as an instance method"); def_mismatch(def, sd, name); return; }
    int dc = (g_node[d].flags & NF_MCONST) != 0;
    int fc = (g_node[def].flags & NF_MCONST) != 0;
    if (dc != fc) {
        tmp_reset();
        if (dc == 1) { tmp_lit("it is declared const"); } else { tmp_lit("it is not declared const"); }
        def_mismatch(def, sd, name);
        return;
    }
    if (g_node[d].nlist != g_node[def].nlist) {
        tmp_reset();
        tmp_lit("it is declared with ");
        tmp_int((long)g_node[d].nlist);
        tmp_lit(" parameter(s)");
        def_mismatch(def, sd, name);
        return;
    }
    int dt = fn_type_in_pattern_env(d);
    int ft = fn_type_in_pattern_env(def);
    if (g_ty[dt].base != 0 && g_ty[ft].base != 0 && bare_type_sid(g_ty[dt].base) != bare_type_sid(g_ty[ft].base)) {
        msg_reset();
        msg_type(g_ty[dt].base);
        tmp_reset();
        tmp_lit("it is declared to return '");
        int q = 0;
        while (q < g_msg_len) { tmp_char((int)g_msg[q]); q = q + 1; }
        tmp_lit("'");
        def_mismatch(def, sd, name);
        return;
    }
    int i = 0;
    while (i < g_ty[dt].nlist && i < g_ty[ft].nlist) {
        int a = g_tyl[g_ty[dt].list + i];
        int b = g_tyl[g_ty[ft].list + i];
        if (bare_type_sid(a) != bare_type_sid(b)) {
            // "parameter N is declared 'A', defined 'B'"
            int sa = 0;
            int sb = 0;
            msg_reset(); msg_type(a); tmp_reset();
            int q = 0;
            while (q < g_msg_len) { tmp_char((int)g_msg[q]); q = q + 1; }
            sa = intern_tmp();
            msg_reset(); msg_type(b); tmp_reset();
            q = 0;
            while (q < g_msg_len) { tmp_char((int)g_msg[q]); q = q + 1; }
            sb = intern_tmp();
            tmp_reset();
            tmp_lit("parameter ");
            tmp_int((long)(i + 1));
            tmp_lit(" is declared '");
            tmp_str(sa);
            tmp_lit("', defined '");
            tmp_str(sb);
            tmp_lit("'");
            def_mismatch(def, sd, name);
            return;
        }
        i = i + 1;
    }
}

// a bare name that names no struct, typedef or generic template, looked
// through references, pointers, arrays, optionals and function types; 0 if none
int unknown_name(int t) {
    if (t == 0) { return 0; }
    int k = g_ty[t].kind;
    if (k == TY_STRUCT) {
        int sd = g_ty[t].n;
        if (g_sdef[sd].node == 0 && g_sdef[sd].tmpl == 0 && g_sdef[sd].kind == SD_STRUCT) { return g_sdef[sd].sid; }
        return 0;
    }
    if (k == TY_REF || k == TY_PTR || k == TY_ARRAY || k == TY_OPT) { return unknown_name(g_ty[t].base); }
    if (k == TY_FN) {
        int n = unknown_name(g_ty[t].base);
        int i = 0;
        while (n == 0 && i < g_ty[t].nlist) { n = unknown_name(g_tyl[g_ty[t].list + i]); i = i + 1; }
        return n;
    }
    return 0;
}

// "unknown type 'X' in the declaration of 'S::f'" for struct-body methods
void check_body_decl_types() {
    int i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        i = i + 1;
        if (g_node[n].kind != D_STRUCT || g_node[n].b != 0) { continue; }
        int j = 0;
        while (j < g_node[n].nlist) {
            int m = g_list[g_node[n].list + j];
            j = j + 1;
            if (g_node[m].kind != D_FUNC || g_node[m].b != 0) { continue; }
            int ft = func_type_from_node(m, 0);
            int bad = unknown_name(ft);
            if (bad != 0) {
                msg_reset();
                msg_lit("unknown type '");
                msg_str(bad);
                msg_lit("' in the declaration of '");
                msg_str(g_node[n].sid);
                msg_lit("::");
                msg_str(g_node[m].sid);
                msg_lit("'");
                error_node(m);
            }
        }
    }
}

// registers a method (declaration or definition) of struct sd
int add_method(int sd, int fnode) {
    int name = g_node[fnode].sid;
    int f = lookup_method(sd, name);
    int is_static = (g_node[fnode].flags & NF_SMETHOD) != 0;
    if (f != 0) {
        // a definition completes a declaration
        if (g_node[g_func[f].node].a == 0 && g_node[fnode].a != 0) {
            int decl = g_func[f].node;
            if ((g_node[decl].flags & NF_SMETHOD) != 0) { g_node[fnode].flags = g_node[fnode].flags | NF_SMETHOD; }
            g_func[f].node = fnode;
            g_func[f].flags = g_node[fnode].flags;
        }
        return f;
    }
    f = new_func(str_concat3(g_sdef[sd].sid, "_", name), fnode);
    g_func[f].origsid = name;
    g_func[f].owner = sd;
    g_func[f].flags = g_node[fnode].flags;
    set_method(sd, name, f);
    return f;
}

void finish_method_type(int f) {
    int fnode = g_func[f].node;
    int sd = g_func[f].owner;
    int st = 0;
    if ((g_func[f].flags & NF_SMETHOD) == 0) { st = self_type(sd, fnode); }
    g_func[f].ty = func_type_from_node(fnode, st);
}

void collect_struct_members(int sd, int node) {
    g_sdef[sd].field0 = g_nfield;
    g_sdef[sd].nfield = 0;
    int i = 0;
    // fields first (they must be contiguous): reserve the slots, then
    // resolve the types (which may instantiate other structs' fields)
    while (i < g_node[node].nlist) {
        int m = g_list[g_node[node].list + i];
        if (g_node[m].kind == D_FIELD) {
            new_field(g_node[m].sid, TY_VOID, m);
            g_sdef[sd].nfield = g_sdef[sd].nfield + 1;
        }
        i = i + 1;
    }
    i = 0;
    int fi = g_sdef[sd].field0;
    while (i < g_node[node].nlist) {
        int m = g_list[g_node[node].list + i];
        if (g_node[m].kind == D_FIELD) {
            int ft = resolve_type(g_node[m].c);
            g_field[fi].ty = ft;
            fi = fi + 1;
        }
        i = i + 1;
    }
    i = 0;
    while (i < g_node[node].nlist) {
        int m = g_list[g_node[node].list + i];
        if (g_node[m].kind == D_FUNC) {
            int fnode = m;
            if (g_sdef[sd].tmpl != 0) { fnode = clone_node(m); }
            int f = add_method(sd, fnode);
            g_func[f].env = 0;
            if (g_sdef[sd].tmpl != 0) {
                // the instance's methods see T bound
                g_func[f].tmpl = 1;
                g_func[f].env = sd;
            }
            finish_method_type(f);
        }
        i = i + 1;
    }
}

// out-of-line methods of generic struct templates: name → list
int g_gmeth_owner[NFUNC_MAX];
int g_gmeth_node[NFUNC_MAX];
int g_ngmeth = 0;

int push_struct_env(int sd) {
    int saved = g_nenv;
    if (g_sdef[sd].tmpl == 0) { return saved; }
    int gp = g_node[g_sdef[sd].tmpl].b;
    int j = 0;
    while (j < g_sdef[sd].ntargs) {
        env_push(g_node[g_list[g_node[gp].list + j]].sid, g_tyl[g_sdef[sd].targs + j]);
        j = j + 1;
    }
    return saved;
}

// a generic struct method (from its out-of-line definition) for instance sd
int instance_method(int sd, int name) {
    int f = lookup_method(sd, name);
    if (f != 0 && g_node[g_func[f].node].a != 0) { return f; }
    if (g_sdef[sd].tmpl == 0) { return f; }
    int tname = g_node[g_sdef[sd].tmpl].sid;
    int i = 0;
    while (i < g_ngmeth) {
        if (g_gmeth_owner[i] == tname && g_node[g_gmeth_node[i]].sid == name) {
            int c = clone_node(g_gmeth_node[i]);
            int saved = push_struct_env(sd);
            // 'generic<U> R S::m(...)': the definition's own parameter names
            // stand for the struct's type arguments, in order
            int mgp = g_node[c].b;
            if ((g_node[c].flags & NF_GENERIC) != 0 && mgp != 0 && g_node[mgp].nlist == g_sdef[sd].ntargs) {
                int j = 0;
                while (j < g_sdef[sd].ntargs) {
                    env_push(g_node[g_list[g_node[mgp].list + j]].sid, g_tyl[g_sdef[sd].targs + j]);
                    j = j + 1;
                }
                g_node[c].flags = g_node[c].flags & ~NF_GENERIC;
            }
            int nf = add_method(sd, c);
            g_func[nf].tmpl = 1;
            g_func[nf].env = sd;
            finish_method_type(nf);
            g_nenv = saved;
            return nf;
        }
        i = i + 1;
    }
    return f;
}

void register_global_name(int sid, int ns, int kind, int idx) {
    if (ns == 0) {
        g_gkind[sid] = kind;
        g_gidx[sid] = idx;
        return;
    }
    int q = str_concat3(ns, "::", sid);
    g_gkind[q] = kind;
    g_gidx[q] = idx;
    if (g_nsfb_kind[sid] == 0) { g_nsfb_kind[sid] = kind; g_nsfb_idx[sid] = idx; }
}

void collect_function(int n) {
    g_tyloc = n;
    int name = g_node[n].sid;
    int owner = g_node[n].d;
    if ((g_node[n].flags & NF_METHOD) != 0 && owner != 0) {
        if (g_gtmpl[owner] != 0) {
            // method of a generic struct template: instantiated with the struct
            if (g_ngmeth < NFUNC_MAX) { g_gmeth_owner[g_ngmeth] = owner; g_gmeth_node[g_ngmeth] = n; g_ngmeth = g_ngmeth + 1; }
            return;
        }
        int ot = g_tyreg[owner];
        int sd = 0;
        if (ot != 0) { sd = struct_of(ot); }
        if (sd == 0) {
            msg_reset();
            msg_lit("method owner struct '");
            msg_str(owner);
            msg_lit("' not declared");
            error_node(n);
            return;
        }
        if (g_node[n].a != 0) { check_method_def(sd, n); }
        int f = add_method(sd, n);
        finish_method_type(f);
        return;
    }
    // free function
    int existing = 0;
    int q = name;
    if (g_node[n].op != 0) { q = str_concat3(g_node[n].op, "::", name); }
    if (g_gkind[q] == SYM_FUNC) { existing = g_gidx[q]; }
    if (existing != 0) {
        if (g_node[g_func[existing].node].a == 0 && g_node[n].a != 0) {
            g_func[existing].node = n;
            g_func[existing].flags = g_node[n].flags;
            if ((g_node[n].flags & NF_GENERIC) == 0) { g_func[existing].ty = func_type_from_node(n, 0); }
        }
        return;
    }
    int mangled = name;
    if (g_node[n].op != 0 && (g_node[n].flags & NF_EXTERN) == 0) {
        tmp_reset();
        int ns = g_node[n].op;
        int i = 0;
        while (i < g_str_len[ns]) {
            int c = str_char(ns, i);
            if (c == ':') { tmp_char('_'); i = i + 2; continue; }
            tmp_char(c);
            i = i + 1;
        }
        tmp_char('_');
        tmp_str(name);
        mangled = intern_tmp();
    }
    int f = new_func(mangled, n);
    g_func[f].origsid = name;
    g_func[f].ns = g_node[n].op;
    if ((g_node[n].flags & NF_GENERIC) == 0) { g_func[f].ty = func_type_from_node(n, 0); }
    register_global_name(name, g_node[n].op, SYM_FUNC, f);
    if (name == S_main && g_node[n].op == 0) { g_main_fn = f; }
}

void collect_enum(int n) {
    int sd = new_sdef(SD_ENUM, g_node[n].sid, n);
    g_tyreg[g_node[n].sid] = g_sdef[sd].ty;
    if (g_node[n].c != 0) { g_sdef[sd].base = resolve_type(g_node[n].c); }
    long next = 0L;
    int i = 0;
    while (i < g_node[n].nlist) {
        int e = g_list[g_node[n].list + i];
        if (g_node[e].a != 0) {
            if (g_node[g_node[e].a].kind == E_INT || (g_node[g_node[e].a].kind == E_UNARY && g_node[g_node[e].a].op == T_MINUS &&
                g_node[g_node[g_node[e].a].a].kind == E_INT) || g_node[g_node[e].a].kind == E_CHAR) {
                if (const_fold(g_node[e].a) == 1) { next = g_fold; }
            } else if (sema_const_int(g_node[e].a) == 1) {
                next = g_cval;
            } else {
                msg_reset();
                msg_lit("expected integer value in enum");
                error_node(g_node[e].a);
            }
        }
        int f = new_field(g_node[e].sid, g_sdef[sd].ty, e);
        g_field[f].val = next;
        g_sdef[sd].nfield = g_sdef[sd].nfield + 1;
        register_global_name(g_node[e].sid, 0, SYM_ENUMC, f);
        next = next + 1L;
        i = i + 1;
    }
}

void collect_global(int n) {
    g_tyloc = n;
    int name = g_node[n].sid;
    int q = name;
    if (g_node[n].op != 0) { q = str_concat3(g_node[n].op, "::", name); }
    if (g_gkind[q] == SYM_GLOBAL) {
        int g0 = g_gidx[q];
        // 'extern T x;' followed by the definition
        if ((g_glob[g0].flags & NF_EXTERN) != 0 && (g_node[n].flags & NF_EXTERN) == 0) {
            g_glob[g0].node = n;
            g_glob[g0].flags = g_node[n].flags;
            g_glob[g0].ty = resolve_type(g_node[n].c);
        }
        return;
    }
    if (g_nglob + 1 >= NGLOB_MAX) { fatal("globals"); return; }
    g_nglob = g_nglob + 1;
    int g = g_nglob;
    g_glob[g].sid = name;
    g_glob[g].node = n;
    g_glob[g].ty = resolve_type(g_node[n].c);
    g_glob[g].flags = g_node[n].flags;
    g_glob[g].addr = 0;
    g_glob[g].done = 0;
    register_global_name(name, g_node[n].op, SYM_GLOBAL, g);
}

void sema_collect() {
    // pass 1: type names
    int i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        int k = g_node[n].kind;
        if (k == D_STRUCT) {
            if (g_node[n].b != 0) { g_gtmpl[g_node[n].sid] = n; }
            else if (g_tyreg[g_node[n].sid] == 0) {
                int kind = SD_STRUCT;
                if ((g_node[n].flags & NF_UNION) != 0) { kind = SD_UNION; }
                int sd = new_sdef(kind, g_node[n].sid, n);
                g_tyreg[g_node[n].sid] = g_sdef[sd].ty;
            }
        } else if (k == D_TRAIT) {
            g_trait[g_node[n].sid] = n;
        } else if (k == D_REGION) {
            g_region_cap[g_node[n].sid] = (int)g_node[n].ival;
            g_nregions = g_nregions + 1;
            g_region_glob[g_node[n].sid] = g_nregions;
        }
        i = i + 1;
    }
    // pass 2: in declaration order
    i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        int k = g_node[n].kind;
        if (k == D_STRUCT && g_node[n].b == 0) {
            int sd = struct_of(g_tyreg[g_node[n].sid]);
            if (sd != 0 && g_sdef[sd].node == n) { collect_struct_members(sd, n); }
        } else if (k == D_ENUM) {
            collect_enum(n);
        } else if (k == D_TYPEDEF) {
            g_tyreg[g_node[n].sid] = resolve_type(g_node[n].c);
        } else if (k == D_NEWTYPE) {
            int sd = new_sdef(SD_NEWTYPE, g_node[n].sid, n);
            g_sdef[sd].base = resolve_type(g_node[n].c);
            g_tyreg[g_node[n].sid] = g_sdef[sd].ty;
        } else if (k == D_FUNC) {
            collect_function(n);
        } else if (k == D_GLOBAL) {
            collect_global(n);
        }
        i = i + 1;
    }
    check_body_decl_types();
}

// ── constant evaluation (integers) ───────────────────────────────────────────
long g_cval = 0L;
int g_ce_bad = 0;     // the identifier that made an expression non-constant
int check_expr(int e);
int ce_call(int e);

// reports why e (expected constant) is not one
void const_error(int e) {
    if (g_ce_bad != 0) {
        msg_reset();
        msg_lit("'");
        msg_str(g_node[g_ce_bad].sid);
        msg_lit("' is not a compile-time constant");
        error_node(g_ce_bad);
        return;
    }
    msg_reset();
    msg_lit("expression is not a valid compile-time constant expression");
    error_node(e);
}

// the integer part of the double with bit pattern b (|value| < 2^63)
long f64_bits_trunc(long b) {
    long ex = ((b >> 52) & 2047L) - 1023L;
    if (ex < 0L) { return 0L; }
    if (ex > 62L) { return 0L; }
    long m = (b & 4503599627370495L) | 4503599627370496L;
    long v = 0L;
    if (ex >= 52L) { v = m << (ex - 52L); } else { v = m >> (52L - ex); }
    if (b < 0L) { v = 0L - v; }
    return v;
}

int sema_const_int(int e) {
    if (e == 0) { return 0; }
    int k = g_node[e].kind;
    if (k == E_INT || k == E_CHAR || k == E_BOOL || k == E_SIZEOFPACK) { g_cval = g_node[e].ival; return 1; }
    if (k == E_SIZEOF_T) {
        int t = resolve_type(g_node[e].c);
        if (g_ty[t].kind == TY_NAMED || g_ty[t].kind == TY_GENERIC) { return 0; }
        g_cval = (long)ty_size(t);
        return 1;
    }
    if (k == E_ALIGNOF) { g_cval = (long)ty_align(resolve_type(g_node[e].c)); return 1; }
    if (k == E_SIZEOF_E) {
        int t = g_node[g_node[e].a].ty;
        if (t == 0) { t = check_expr(g_node[e].a); }
        g_cval = (long)ty_size(t);
        return 1;
    }
    if (k == E_IDENT) {
        int sid = g_node[e].sid;
        // locals (even const ones) are never compile-time constants
        if (lookup_local(sid) >= 0) { g_ce_bad = e; return 0; }
        int gk = g_gkind[sid];
        if (gk == 0) { gk = g_nsfb_kind[sid]; if (gk != 0) { sid = sid; } }
        if (gk == SYM_ENUMC) {
            int fi = g_gidx[sid];
            if (g_gkind[sid] == 0) { fi = g_nsfb_idx[sid]; }
            g_cval = g_field[fi].val;
            return 1;
        }
        if (gk == SYM_GLOBAL) {
            int g = g_gidx[sid];
            if (g_gkind[sid] == 0) { g = g_nsfb_idx[sid]; }
            int gn = g_glob[g].node;
            if ((g_glob[g].flags & NF_CONST) != 0 && g_node[gn].a != 0) { return sema_const_int(g_node[gn].a); }
        }
        g_ce_bad = e;
        return 0;
    }
    if (k == E_CAST) {
        int t = resolve_type(g_node[e].c);
        // a newtype folds like its base type
        if (g_ty[t].kind == TY_NEWTYPE) { t = g_sdef[g_ty[t].n].base; }
        int fe = g_node[e].a;
        int fneg = 0;
        if (g_node[fe].kind == E_UNARY && g_node[fe].op == T_MINUS && g_node[g_node[fe].a].kind == E_FLOAT) { fneg = 1; fe = g_node[fe].a; }
        if (g_node[fe].kind == E_FLOAT && ty_is_int(t) == 1) {
            // (int)2.5: truncate the literal's double toward zero
            g_cval = f64_bits_trunc(g_node[fe].ival);
            if (fneg == 1) { g_cval = 0L - g_cval; }
        } else if (sema_const_int(g_node[e].a) == 0) { return 0; }
        if (ty_is_int(t) == 1) {
            int bits = ty_bits(t);
            if (bits < 64) {
                long m = (1L << (long)bits) - 1L;
                long v = g_cval & m;
                if (ty_is_signed(t) == 1 && (v & (1L << (long)(bits - 1))) != 0L) { v = v - (1L << (long)bits); }
                if (g_ty[t].kind == TY_BOOL) { if (g_cval != 0L) { v = 1L; } else { v = 0L; } }
                g_cval = v;
            }
            return 1;
        }
        if (g_ty[t].kind == TY_ENUM) { return 1; }
        return 0;
    }
    if (k == E_UNARY) {
        if (sema_const_int(g_node[e].a) == 0) { return 0; }
        int op = g_node[e].op;
        if (op == T_MINUS) { g_cval = 0L - g_cval; return 1; }
        if (op == T_PLUS) { return 1; }
        if (op == T_TILDE) { g_cval = ~g_cval; return 1; }
        if (op == T_BANG) { if (g_cval == 0L) { g_cval = 1L; } else { g_cval = 0L; } return 1; }
        return 0;
    }
    if (k == E_BINARY) {
        int op = g_node[e].op;
        if (sema_const_int(g_node[e].a) == 0) { return 0; }
        long a = g_cval;
        if (op == T_ANDAND && a == 0L) { g_cval = 0L; return 1; }
        if (op == T_OROR && a != 0L) { g_cval = 1L; return 1; }
        if (sema_const_int(g_node[e].b) == 0) { return 0; }
        long b = g_cval;
        if (op == T_PLUS || op == T_PLUSPIPE) { g_cval = a + b; return 1; }
        if (op == T_MINUS || op == T_MINUSPIPE) { g_cval = a - b; return 1; }
        if (op == T_STAR || op == T_STARPIPE) { g_cval = a * b; return 1; }
        if (op == T_SLASH) { if (b == 0L) { return 0; } g_cval = a / b; return 1; }
        if (op == T_PERCENT) { if (b == 0L) { return 0; } g_cval = a % b; return 1; }
        if (op == T_SHL) { g_cval = a << b; return 1; }
        if (op == T_SHR) { g_cval = a >> b; return 1; }
        if (op == T_AMP) { g_cval = a & b; return 1; }
        if (op == T_PIPE) { g_cval = a | b; return 1; }
        if (op == T_CARET) { g_cval = a ^ b; return 1; }
        g_cval = 0L;
        if (op == T_EQ) { if (a == b) { g_cval = 1L; } return 1; }
        if (op == T_NE) { if (a != b) { g_cval = 1L; } return 1; }
        if (op == T_LT) { if (a < b) { g_cval = 1L; } return 1; }
        if (op == T_GT) { if (a > b) { g_cval = 1L; } return 1; }
        if (op == T_LE) { if (a <= b) { g_cval = 1L; } return 1; }
        if (op == T_GE) { if (a >= b) { g_cval = 1L; } return 1; }
        if (op == T_ANDAND || op == T_OROR) { if (b != 0L) { g_cval = 1L; } return 1; }
        return 0;
    }
    if (k == E_TERNARY) {
        if (sema_const_int(g_node[e].a) == 0) { return 0; }
        if (g_cval != 0L) { return sema_const_int(g_node[e].b); }
        return sema_const_int(g_node[e].c);
    }
    if (k == E_CALL) { return ce_call(e); }
    return 0;
}

// ── const fn interpreter (integers) ──────────────────────────────────────────
// Locals of the function being interpreted live in a small frame stack.
int g_ce_sid[1024];
long g_ce_val[1024];
int g_ce_n = 0;
int g_ce_base = 0;
int g_ce_depth = 0;
int g_ce_steps = 0;
int g_ce_ret = 0;      // 1 = return executed
long g_ce_retv = 0L;
int g_ce_fail = 0;
int g_ce_brk = 0;      // 1 = break, 2 = continue

int ce_find(int sid) {
    int i = g_ce_n - 1;
    while (i >= g_ce_base) {
        if (g_ce_sid[i] == sid) { return i; }
        i = i - 1;
    }
    return -1;
}

long ce_expr(int e);

void ce_stmt(int s) {
    if (g_ce_fail == 1 || g_ce_ret == 1 || g_ce_brk != 0 || s == 0) { return; }
    g_ce_steps = g_ce_steps + 1;
    if (g_ce_steps > 1000000) { g_ce_fail = 1; return; }
    int k = g_node[s].kind;
    if (k == S_BLOCK || k == S_MULTI) {
        int mark = g_ce_n;
        int i = 0;
        while (i < g_node[s].nlist) {
            ce_stmt(g_list[g_node[s].list + i]);
            if (g_ce_fail == 1 || g_ce_ret == 1 || g_ce_brk != 0) { break; }
            i = i + 1;
        }
        if (k == S_BLOCK) { g_ce_n = mark; }
        return;
    }
    if (k == S_VAR) {
        if (g_ce_n >= 1024) { g_ce_fail = 1; return; }
        long v = 0L;
        if (g_node[s].a != 0) { v = ce_expr(g_node[s].a); }
        g_ce_sid[g_ce_n] = g_node[s].sid;
        g_ce_val[g_ce_n] = v;
        g_ce_n = g_ce_n + 1;
        return;
    }
    if (k == S_EXPR) { ce_expr(g_node[s].a); return; }
    if (k == S_RETURN) {
        if (g_node[s].a != 0) { g_ce_retv = ce_expr(g_node[s].a); }
        g_ce_ret = 1;
        return;
    }
    if (k == S_IF || k == S_IFCONST) {
        if (ce_expr(g_node[s].a) != 0L) { ce_stmt(g_node[s].b); }
        else { ce_stmt(g_node[s].c); }
        return;
    }
    if (k == S_WHILE || k == S_DO || k == S_FOR) {
        int mark = g_ce_n;
        if (k == S_FOR) { ce_stmt(g_node[s].a); }
        int body = g_node[s].b;
        if (k == S_FOR) { body = g_node[s].d; }
        int first = 1;
        while (g_ce_fail == 0) {
            if (k == S_DO && first == 1) { first = 0; }
            else {
                int c = g_node[s].a;
                if (k == S_FOR) { c = g_node[s].b; }
                if (c != 0 && ce_expr(c) == 0L) { break; }
            }
            ce_stmt(body);
            if (g_ce_ret == 1) { break; }
            if (g_ce_brk == 1) { g_ce_brk = 0; break; }
            g_ce_brk = 0;
            if (k == S_FOR && g_node[s].c != 0) { ce_expr(g_node[s].c); }
            g_ce_steps = g_ce_steps + 1;
            if (g_ce_steps > 1000000) { g_ce_fail = 1; }
        }
        g_ce_n = mark;
        return;
    }
    if (k == S_BREAK) { g_ce_brk = 1; return; }
    if (k == S_CONTINUE) { g_ce_brk = 2; return; }
    if (k == S_EMPTY || k == S_STATIC_ASSERT) { return; }
    g_ce_fail = 1;
}

long ce_expr(int e) {
    if (g_ce_fail == 1) { return 0L; }
    int k = g_node[e].kind;
    if (k == E_IDENT) {
        int v = ce_find(g_node[e].sid);
        if (v >= 0) { return g_ce_val[v]; }
        // outside the interpreted function: a compile-time constant
        int saved_base = g_ce_base;
        if (sema_const_int(e) == 1) { return g_cval; }
        g_ce_base = saved_base;
        g_ce_fail = 1;
        return 0L;
    }
    if (k == E_ASSIGN) {
        int lhs = g_node[e].a;
        if (g_node[lhs].kind != E_IDENT) { g_ce_fail = 1; return 0L; }
        int v = ce_find(g_node[lhs].sid);
        if (v < 0) { g_ce_fail = 1; return 0L; }
        long r = ce_expr(g_node[e].b);
        int op = g_node[e].op;
        long o = g_ce_val[v];
        if (op == T_ASSIGN) { o = r; }
        else if (op == T_PLUSEQ) { o = o + r; }
        else if (op == T_MINUSEQ) { o = o - r; }
        else if (op == T_STAREQ) { o = o * r; }
        else if (op == T_SLASHEQ) { if (r == 0L) { g_ce_fail = 1; return 0L; } o = o / r; }
        else if (op == T_PERCENTEQ) { if (r == 0L) { g_ce_fail = 1; return 0L; } o = o % r; }
        else if (op == T_SHLEQ) { o = o << r; }
        else if (op == T_SHREQ) { o = o >> r; }
        else if (op == T_AMPEQ) { o = o & r; }
        else if (op == T_PIPEEQ) { o = o | r; }
        else if (op == T_CARETEQ) { o = o ^ r; }
        g_ce_val[v] = o;
        return o;
    }
    if (k == E_UNARY && (g_node[e].op == T_INC || g_node[e].op == T_DEC) ||
        k == E_POSTFIX) {
        int lhs = g_node[e].a;
        if (g_node[lhs].kind != E_IDENT) { g_ce_fail = 1; return 0L; }
        int v = ce_find(g_node[lhs].sid);
        if (v < 0) { g_ce_fail = 1; return 0L; }
        long o = g_ce_val[v];
        long nv = o + 1L;
        if (g_node[e].op == T_DEC) { nv = o - 1L; }
        g_ce_val[v] = nv;
        if (k == E_POSTFIX) { return o; }
        return nv;
    }
    if (k == E_BINARY && (g_node[e].op == T_ANDAND || g_node[e].op == T_OROR)) {
        long a = ce_expr(g_node[e].a);
        if (g_node[e].op == T_ANDAND && a == 0L) { return 0L; }
        if (g_node[e].op == T_OROR && a != 0L) { return 1L; }
        if (ce_expr(g_node[e].b) != 0L) { return 1L; }
        return 0L;
    }
    if (k == E_BINARY || k == E_UNARY || k == E_TERNARY || k == E_CAST) {
        // evaluate operands, then fold through literal nodes
        if (k == E_TERNARY) {
            if (ce_expr(g_node[e].a) != 0L) { return ce_expr(g_node[e].b); }
            return ce_expr(g_node[e].c);
        }
        long a = ce_expr(g_node[e].a);
        long b = 0L;
        if (k == E_BINARY) { b = ce_expr(g_node[e].b); }
        if (g_ce_fail == 1) { return 0L; }
        int op = g_node[e].op;
        if (k == E_CAST) { return a; }
        if (k == E_UNARY) {
            if (op == T_MINUS) { return 0L - a; }
            if (op == T_TILDE) { return ~a; }
            if (op == T_BANG) { if (a == 0L) { return 1L; } return 0L; }
            return a;
        }
        if (op == T_PLUS || op == T_PLUSPIPE) { return a + b; }
        if (op == T_MINUS || op == T_MINUSPIPE) { return a - b; }
        if (op == T_STAR || op == T_STARPIPE) { return a * b; }
        if (op == T_SLASH) { if (b == 0L) { g_ce_fail = 1; return 0L; } return a / b; }
        if (op == T_PERCENT) { if (b == 0L) { g_ce_fail = 1; return 0L; } return a % b; }
        if (op == T_SHL) { return a << b; }
        if (op == T_SHR) { return a >> b; }
        if (op == T_AMP) { return a & b; }
        if (op == T_PIPE) { return a | b; }
        if (op == T_CARET) { return a ^ b; }
        if (op == T_EQ) { if (a == b) { return 1L; } return 0L; }
        if (op == T_NE) { if (a != b) { return 1L; } return 0L; }
        if (op == T_LT) { if (a < b) { return 1L; } return 0L; }
        if (op == T_GT) { if (a > b) { return 1L; } return 0L; }
        if (op == T_LE) { if (a <= b) { return 1L; } return 0L; }
        if (op == T_GE) { if (a >= b) { return 1L; } return 0L; }
        g_ce_fail = 1;
        return 0L;
    }
    if (k == E_CALL) {
        if (ce_call(e) == 0) { g_ce_fail = 1; return 0L; }
        return g_cval;
    }
    if (sema_const_int(e) == 1) { return g_cval; }
    g_ce_fail = 1;
    return 0L;
}

// a call to a const/consteval function with constant arguments
int ce_call(int e) {
    int callee = g_node[e].a;
    if (g_node[callee].kind != E_IDENT) { return 0; }
    int sid = g_node[callee].sid;
    int f = 0;
    if (g_gkind[sid] == SYM_FUNC) { f = g_gidx[sid]; }
    else if (g_nsfb_kind[sid] == SYM_FUNC) { f = g_nsfb_idx[sid]; }
    if (f == 0) { return 0; }
    int fnode = g_func[f].node;
    if ((g_node[fnode].flags & (NF_CONST | NF_CONSTEVAL)) == 0 || g_node[fnode].a == 0) { return 0; }
    if (g_ce_depth > 200) { return 0; }
    int np = g_node[fnode].nlist;
    if (np != g_node[e].nlist) { return 0; }
    // evaluate arguments in the caller's frame
    long args[32];
    int i = 0;
    int outer = g_ce_depth > 0;
    while (i < np && i < 32) {
        int a = g_list[g_node[e].list + i];
        if (outer == 1) { args[i] = ce_expr(a); if (g_ce_fail == 1) { return 0; } }
        else { if (sema_const_int(a) == 0) { return 0; } args[i] = g_cval; }
        i = i + 1;
    }
    int saved_base = g_ce_base;
    int saved_n = g_ce_n;
    int saved_ret = g_ce_ret;
    long saved_retv = g_ce_retv;
    int saved_fail = g_ce_fail;
    if (g_ce_depth == 0) { g_ce_steps = 0; g_ce_fail = 0; }
    g_ce_depth = g_ce_depth + 1;
    g_ce_base = g_ce_n;
    i = 0;
    while (i < np && g_ce_n < 1024) {
        g_ce_sid[g_ce_n] = g_node[g_list[g_node[fnode].list + i]].sid;
        g_ce_val[g_ce_n] = args[i];
        g_ce_n = g_ce_n + 1;
        i = i + 1;
    }
    g_ce_ret = 0;
    g_ce_brk = 0;
    ce_stmt(g_node[fnode].a);
    int ok = 0;
    if (g_ce_fail == 0 && g_ce_ret == 1) { ok = 1; }
    long v = g_ce_retv;
    g_ce_depth = g_ce_depth - 1;
    g_ce_base = saved_base;
    g_ce_n = saved_n;
    g_ce_ret = saved_ret;
    g_ce_retv = saved_retv;
    if (g_ce_depth == 0) { g_ce_fail = 0; } else if (ok == 0) { g_ce_fail = 1; } else { g_ce_fail = saved_fail; }
    if (ok == 1) {
        // narrow to the return type
        int rt = resolve_type(g_node[fnode].c);
        if (ty_is_int(rt) == 1 && ty_bits(rt) < 64) {
            int bits = ty_bits(rt);
            long m = (1L << (long)bits) - 1L;
            long r = v & m;
            if (ty_is_signed(rt) == 1 && (r & (1L << (long)(bits - 1))) != 0L) { r = r - (1L << (long)bits); }
            v = r;
        }
        g_cval = v;
    }
    return ok;
}
