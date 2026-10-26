// call.sc — call checking: direct/indirect/method calls, nullable
// pseudo-methods, tagged-union construction, builtins, generic inference
// and instantiation, trn constraints.
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"

void queue_func(int f) {
    if (g_func[f].checked != 0) { return; }
    if (g_func[f].tmpl == 0) { return; }   // checked in declaration order
    if (g_node[g_func[f].node].a == 0) { return; }
    g_func[f].checked = 1;
    if (g_nqueue < NFUNC_MAX) { g_queue[g_nqueue] = f; g_nqueue = g_nqueue + 1; }
}

// ── type pattern matching (generic inference) ────────────────────────────────
int g_bind_sid[32];
int g_bind_ty[32];
int g_nbind = 0;

int bind_get(int sid) {
    int i = 0;
    while (i < g_nbind) { if (g_bind_sid[i] == sid) { return g_bind_ty[i]; } i = i + 1; }
    return 0;
}

int match_type(int p, int a) {
    if (p == 0 || a == 0) { return 0; }
    int pk = g_ty[p].kind;
    int ak = g_ty[a].kind;
    if (pk == TY_GENERIC) {
        int b = bind_get(g_ty[p].sid);
        if (b != 0) { return ty_equal(b, a); }
        if (g_nbind < 32) { g_bind_sid[g_nbind] = g_ty[p].sid; g_bind_ty[g_nbind] = a; g_nbind = g_nbind + 1; }
        return 1;
    }
    if (pk == TY_REF && ak == TY_PTR) { return match_type(g_ty[p].base, g_ty[a].base); }
    if (pk == TY_NAMED && ak == TY_STRUCT) {
        int sd = g_ty[a].n;
        if (g_sdef[sd].tmpl == 0) { return 0; }
        if (g_node[g_sdef[sd].tmpl].sid != g_ty[p].sid) { return 0; }
        if (g_sdef[sd].ntargs != g_ty[p].nlist) { return 0; }
        int i = 0;
        while (i < g_ty[p].nlist) {
            if (match_type(g_tyl[g_ty[p].list + i], g_tyl[g_sdef[sd].targs + i]) == 0) { return 0; }
            i = i + 1;
        }
        return 1;
    }
    if (pk != ak) { return 0; }
    if (pk == TY_OPT || pk == TY_SLICE || pk == TY_PTR || pk == TY_REF || pk == TY_ARRAY) {
        return match_type(g_ty[p].base, g_ty[a].base);
    }
    if (pk == TY_TUPLE || pk == TY_FN) {
        if (g_ty[p].nlist != g_ty[a].nlist) { return 0; }
        if (pk == TY_FN && match_type(g_ty[p].base, g_ty[a].base) == 0) { return 0; }
        int i = 0;
        while (i < g_ty[p].nlist) {
            if (match_type(g_tyl[g_ty[p].list + i], g_tyl[g_ty[a].list + i]) == 0) { return 0; }
            i = i + 1;
        }
        return 1;
    }
    return ty_equal(p, a);
}

// ── traits ───────────────────────────────────────────────────────────────────
int is_builtin_trait(int sid) {
    return str_eq_lit(sid, "Numeric") == 1 || str_eq_lit(sid, "Eq") == 1 || str_eq_lit(sid, "Ord") == 1 ||
           str_eq_lit(sid, "Add") == 1 || str_eq_lit(sid, "Sub") == 1 || str_eq_lit(sid, "Mul") == 1 ||
           str_eq_lit(sid, "Div") == 1;
}

// reason text goes to g_tmp; returns 0 if satisfied, 1 if not (reason
// "-" = no detail)
int need_op(int sd, const char* op) {
    int f = instance_method(sd, intern_lit(op));
    if (f == 0) { tmp_reset(); tmp_lit("no '"); tmp_lit(op); tmp_lit("'"); return 1; }
    return 0;
}

int trait_mismatch(int t, int trn) {
    int k = g_ty[t].kind;
    if (str_eq_lit(trn, "Indexed") == 1) {
        if (k == TY_ARRAY || k == TY_PTR || k == TY_SLICE || k == TY_REF) { return 0; }
        tmp_reset(); tmp_lit("-"); return 1;
    }
    if (str_eq_lit(trn, "Pointer") == 1) {
        if (k == TY_PTR || k == TY_REF) { return 0; }
        tmp_reset(); tmp_lit("-"); return 1;
    }
    if (ty_is_arith(t) == 1) {
        if (is_builtin_trait(trn) == 1) { return 0; }
        tmp_reset(); tmp_lit("-"); return 1;
    }
    if (k == TY_STRUCT) {
        int sd = g_ty[t].n;
        if (str_eq_lit(trn, "Numeric") == 1) {
            if (need_op(sd, "operator+") == 1) { return 1; }
            if (need_op(sd, "operator-") == 1) { return 1; }
            if (need_op(sd, "operator*") == 1) { return 1; }
            return need_op(sd, "operator/");
        }
        if (str_eq_lit(trn, "Eq") == 1) {
            if (need_op(sd, "operator==") == 1) { return 1; }
            return need_op(sd, "operator!=");
        }
        if (str_eq_lit(trn, "Ord") == 1) {
            if (need_op(sd, "operator<") == 1) { return 1; }
            if (need_op(sd, "operator>") == 1) { return 1; }
            if (need_op(sd, "operator<=") == 1) { return 1; }
            return need_op(sd, "operator>=");
        }
        if (str_eq_lit(trn, "Add") == 1) { return need_op(sd, "operator+"); }
        if (str_eq_lit(trn, "Sub") == 1) { return need_op(sd, "operator-"); }
        if (str_eq_lit(trn, "Mul") == 1) { return need_op(sd, "operator*"); }
        if (str_eq_lit(trn, "Div") == 1) { return need_op(sd, "operator/"); }
        int tn = g_trait[trn];
        if (tn != 0) {
            int i = 0;
            while (i < g_node[tn].nlist) {
                int m = g_list[g_node[tn].list + i];
                int f = instance_method(sd, g_node[m].sid);
                if (f == 0) {
                    tmp_reset(); tmp_lit("no method '"); tmp_str(g_node[m].sid); tmp_lit("'");
                    return 1;
                }
                int fstatic = (g_func[f].flags & NF_SMETHOD) != 0;
                int mstatic = (g_node[m].flags & NF_SMETHOD) != 0;
                if (fstatic != mstatic) {
                    tmp_reset(); tmp_lit("'"); tmp_str(g_node[m].sid); tmp_lit("' is ");
                    if (mstatic == 1) { tmp_lit("static"); } else { tmp_lit("an instance method"); }
                    tmp_lit(" in the trait");
                    return 1;
                }
                int have = g_node[g_func[f].node].nlist;
                if ((g_func[f].flags & NF_VARIADIC) == 0 && have != g_node[m].nlist) {
                    tmp_reset(); tmp_lit("'"); tmp_str(g_node[m].sid); tmp_lit("' takes ");
                    tmp_int((long)have); tmp_lit(" parameter(s), the trn "); tmp_int((long)g_node[m].nlist);
                    return 1;
                }
                i = i + 1;
            }
            int sup = g_node[tn].b;
            if (sup != 0) {
                i = 0;
                while (i < g_node[sup].nlist) {
                    int s = g_node[g_list[g_node[sup].list + i]].sid;
                    if (trait_mismatch(t, s) == 1) {
                        if (g_tmp_len == 1 && g_tmp[0] == (unsigned char)'-') {
                            tmp_reset(); tmp_lit("not '"); tmp_str(s); tmp_lit("'");
                        } else {
                            tmp_lit(", from '"); tmp_str(s); tmp_lit("'");
                        }
                        return 1;
                    }
                    i = i + 1;
                }
            }
            return 0;
        }
    }
    tmp_reset(); tmp_lit("-");
    return 1;
}

// "type 'X' does not satisfy constraint 'C' for generic parameter 'T' (why)"
void constraint_error(int node, int t, int trn, int gp) {
    trait_mismatch(t, trn);
    int why = intern_tmp();
    msg_reset();
    msg_lit("type '");
    msg_type(t);
    msg_lit("' does not satisfy constraint '");
    msg_str(trn);
    msg_lit("' for generic parameter '");
    msg_str(gp);
    msg_lit("'");
    if (str_eq_lit(why, "-") == 0) { msg_lit(" ("); msg_str(why); msg_lit(")"); }
    error_node(node);
}

// checks the bounds of generic params node gpn against the current
// bindings; returns 1 if any failed
int check_bounds(int gpn, int node) {
    int failed = 0;
    int i = 0;
    while (i < g_node[gpn].nlist) {
        int gp = g_list[g_node[gpn].list + i];
        int b = g_node[gp].b;
        int t = bind_get(g_node[gp].sid);
        if (b != 0 && t != 0) {
            int j = 0;
            while (j < g_node[b].nlist) {
                int tr = g_node[g_list[g_node[b].list + j]].sid;
                if (trait_mismatch(t, tr) == 1) {
                    constraint_error(node, t, tr, g_node[gp].sid);
                    failed = 1;
                }
                j = j + 1;
            }
        }
        i = i + 1;
    }
    return failed;
}

// ── generic function instantiation ───────────────────────────────────────────
// pushes T → generic(T) for the params of gpn
void push_pattern_env(int gpn) {
    int i = 0;
    while (i < g_node[gpn].nlist) {
        int gp = g_list[g_node[gpn].list + i];
        env_push(g_node[gp].sid, ty_generic(g_node[gp].sid));
        i = i + 1;
    }
}

void push_inst_env(int inst) {
    int f = g_inst_tmpl[inst];
    int gpn = g_node[g_func[f].node].b;
    int i = 0;
    while (i < g_node[gpn].nlist && i < g_inst_nargs[inst]) {
        env_push(g_node[g_list[g_node[gpn].list + i]].sid, g_tyl[g_inst_args[inst] + i]);
        i = i + 1;
    }
}

int instantiate_func(int tf) {
    int fnode = g_func[tf].node;
    int gpn = g_node[fnode].b;
    int np = g_node[gpn].nlist;
    int args[32];
    int i = 0;
    while (i < np && i < 32) { args[i] = bind_get(g_node[g_list[g_node[gpn].list + i]].sid); i = i + 1; }
    int k = 0;
    while (k < g_ninst) {
        if (g_inst_tmpl[k] == tf && g_inst_nargs[k] == np) {
            int same = 1;
            i = 0;
            while (i < np) { if (g_tyl[g_inst_args[k] + i] != args[i]) { same = 0; break; } i = i + 1; }
            if (same == 1) { return g_inst_func[k]; }
        }
        k = k + 1;
    }
    if (g_ninst >= NFUNC_MAX) { fatal("generic instances"); return tf; }
    // name: __safec_max_int
    tmp_reset();
    tmp_lit("__safec_");
    tmp_str(g_func[tf].sid);
    i = 0;
    while (i < np) {
        tmp_char('_');
        msg_reset();
        msg_type(args[i]);
        int q = 0;
        while (q < g_msg_len) { tmp_char((int)g_msg[q]); q = q + 1; }
        i = i + 1;
    }
    int name = intern_tmp();
    int c = clone_node(fnode);
    g_node[c].flags = g_node[c].flags & ~NF_GENERIC;
    int nf = new_func(name, c);
    g_func[nf].origsid = g_func[tf].origsid;
    g_func[nf].owner = g_func[tf].owner;
    g_func[nf].tmpl = 2;
    int inst = g_ninst;
    g_ninst = g_ninst + 1;
    g_inst_tmpl[inst] = tf;
    g_inst_func[inst] = nf;
    int mark = tyl_begin();
    i = 0;
    while (i < np) { tyl_push(args[i]); i = i + 1; }
    g_inst_args[inst] = mark;
    g_inst_nargs[inst] = np;
    g_func[nf].env = inst;
    int saved = g_nenv;
    if (g_func[tf].owner != 0) { push_struct_env(g_func[tf].owner); }
    push_inst_env(inst);
    int st = 0;
    if (g_func[tf].owner != 0 && (g_node[c].flags & NF_SMETHOD) == 0) {
        int fl = 0;
        if ((g_node[c].flags & NF_MCONST) != 0) { fl = TYF_CONST; }
        st = ty_ref(g_sdef[g_func[tf].owner].ty, R_STACK, 0, fl);
    }
    g_func[nf].ty = func_type_from_node(c, st);
    g_nenv = saved;
    return nf;
}

// a generic function used as a value where a function type is expected
int coerce_generic_value(int e, int t, int target) {
    if (target == 0 || g_node[e].kind != E_IDENT || g_node[e].c != SYM_FUNC) { return t; }
    int tf = g_node[e].d;
    if ((g_func[tf].flags & NF_GENERIC) == 0) { return t; }
    int tt = target;
    if (g_ty[tt].kind == TY_OPT) { tt = g_ty[tt].base; }
    int ft = fn_type_of(tt);
    if (ft == 0) { return t; }
    int fnode = g_func[tf].node;
    if (g_ty[ft].nlist != g_node[fnode].nlist) { return t; }
    int saved = g_nenv;
    push_pattern_env(g_node[fnode].b);
    int pat = func_type_from_node(fnode, 0);
    g_nenv = saved;
    int saved_nbind = g_nbind;
    g_nbind = 0;
    int i = 0;
    while (i < g_ty[ft].nlist) {
        if (match_type(g_tyl[g_ty[pat].list + i], g_tyl[g_ty[ft].list + i]) == 0) { g_nbind = saved_nbind; return t; }
        i = i + 1;
    }
    match_type(g_ty[pat].base, g_ty[ft].base);
    int gpn = g_node[fnode].b;
    i = 0;
    while (i < g_node[gpn].nlist) {
        int gs = g_node[g_list[g_node[gpn].list + i]].sid;
        if (bind_get(gs) == 0) {
            msg_reset();
            msg_lit("cannot infer type arguments for generic function '");
            msg_str(g_func[tf].origsid);
            msg_lit("' from '");
            msg_type(target);
            msg_lit("'");
            error_node(e);
            g_nbind = saved_nbind;
            return TY_ERROR;
        }
        i = i + 1;
    }
    if (check_bounds(gpn, e) == 1) { return TY_ERROR; }
    int nf = instantiate_func(tf);
    g_node[e].d = nf;
    queue_func(nf);
    int r = func_value_type(nf);
    g_node[e].ty = r;
    return r;
}

// ── argument checking ────────────────────────────────────────────────────────
void arg_error(int a, int i, int at, int pt) {
    msg_reset();
    msg_lit("argument ");
    msg_int((long)(i + 1));
    msg_lit(": cannot convert '");
    msg_type(at);
    msg_lit("' to '");
    msg_type(pt);
    msg_lit("'");
    msg_raw_ref_hint(at, pt);
    error_node(a);
}

int arg_ok(int a, int at, int pt, int is_extern) {
    if (is_err(at) == 1 || is_err(pt) == 1) { return 1; }
    if (convertible_to(a, at, pt) == 1) { return 1; }
    if (ref_to_ptr_arg(at, pt) == 1) { return 1; }
    // C ABI: pointer-width integer parameters of extern functions
    if (is_extern == 1 && ty_is_int(at) == 1 && ty_is_int(pt) == 1 && ty_bits(pt) == 32) { return 1; }
    return 0;
}

// the out-parameter idiom 'f(&x)' initializes x
void pre_mark_out_arg(int a, int pt) {
    if (g_node[a].kind != E_ADDR) { return; }
    int o = g_node[a].a;
    if (g_node[o].kind != E_IDENT) { return; }
    int out = 0;
    if (g_ty[pt].kind == TY_PTR && (g_ty[pt].flags & TYF_CONST) == 0) { out = 1; }
    if (g_ty[pt].kind == TY_REF && (g_ty[pt].flags & TYF_CONST) == 0) { out = 1; }
    if (out == 1) {
        int s = lookup_local(g_node[o].sid);
        if (s >= 0) { g_sym[s].init = 1; }
    }
}

void check_call_regions(int e, int ft, int self_off) {
    int n = g_node[e].nlist;
    int np = g_ty[ft].nlist;
    int i = 0;
    while (i < n) {
        int pi = i + self_off;
        if (pi >= np) { break; }
        int a = g_list[g_node[e].list + i];
        int at = g_node[a].ty;
        int pt = g_tyl[g_ty[ft].list + pi];
        if (g_ty[pt].kind == TY_REF && g_ty[at].kind == TY_REF) {
            int pr = g_ty[pt].n;
            int ar = g_ty[at].n;
            if (pr == R_STATIC && (ar == R_STACK || ar == R_ARENA)) {
                msg_reset();
                msg_lit("argument ");
                msg_int((long)(i + 1));
                msg_lit(": cannot pass '");
                msg_type(at);
                msg_lit("' where '");
                msg_type(pt);
                msg_lit("' is expected (region lifetime insufficient)");
                error_node(a);
            }
            if (pr == R_HEAP && ar == R_STACK) {
                msg_reset();
                msg_lit("argument ");
                msg_int((long)(i + 1));
                msg_lit(": cannot pass '&stack ");
                msg_type(g_ty[at].base);
                msg_lit("' where '&heap ");
                msg_type(g_ty[pt].base);
                msg_lit("' is expected (stack reference may dangle)");
                error_node(a);
            }
        }
        i = i + 1;
    }
    // aliasing within one call: f(&x, &x)
    i = 0;
    while (i < n) {
        int ai = g_list[g_node[e].list + i];
        int pi = i + self_off;
        if (g_node[ai].kind == E_ADDR && g_node[g_node[ai].a].kind == E_IDENT && pi < np) {
            int pti = g_tyl[g_ty[ft].list + pi];
            int mi = g_ty[pti].kind == TY_REF && (g_ty[pti].flags & TYF_CONST) == 0;
            int j = i + 1;
            while (j < n) {
                int aj = g_list[g_node[e].list + j];
                int pj = j + self_off;
                if (g_node[aj].kind == E_ADDR && g_node[g_node[aj].a].kind == E_IDENT && pj < np &&
                    g_node[g_node[aj].a].sid == g_node[g_node[ai].a].sid) {
                    int ptj = g_tyl[g_ty[ft].list + pj];
                    int mj = g_ty[ptj].kind == TY_REF && (g_ty[ptj].flags & TYF_CONST) == 0;
                    if (mi == 1 || mj == 1) {
                        msg_reset();
                        msg_lit("cannot borrow '");
                        msg_str(g_node[g_node[aj].a].sid);
                        msg_lit("' as ");
                        if (mj == 1) { msg_lit("mutable"); } else { msg_lit("immutable"); }
                        msg_lit(": already borrowed by argument ");
                        msg_int((long)(i + 1));
                        error_node(aj);
                    }
                }
                j = j + 1;
            }
        }
        i = i + 1;
    }
    int rt = g_ty[ft].base;
    if (g_ty[rt].kind == TY_REF && g_ty[rt].n == R_STACK) {
        msg_reset();
        msg_lit("called function returns '&stack ");
        msg_type(g_ty[rt].base);
        msg_lit("': stack reference does not survive call boundary");
        error_node(e);
    }
}

// checks args of a call against fn type ft (self_off: skip self)
void check_args(int e, int ft, int self_off, int is_extern) {
    int n = g_node[e].nlist;
    int np = g_ty[ft].nlist;
    int eff = np - self_off;
    int variadic = (g_ty[ft].flags & TYF_VARIADIC) != 0;
    if (n < eff || (variadic == 0 && n > eff)) {
        msg_reset();
        msg_lit("wrong number of arguments: expected ");
        msg_int((long)eff);
        msg_lit(", got ");
        msg_int((long)n);
        error_node(e);
    }
    int i = 0;
    while (i < n) {
        int a = g_list[g_node[e].list + i];
        int pi = i + self_off;
        int pt = 0;
        if (pi < np) { pt = g_tyl[g_ty[ft].list + pi]; pre_mark_out_arg(a, pt); adopt_tuple(a, pt); }
        int at = 0;
        if (pt != 0 && g_node[a].kind == E_INITLIST) { at = check_initlist(a, pt); }
        else { at = check_expr(a); }
        if (pt != 0) {
            at = coerce_generic_value(a, at, pt);
            if (arg_ok(a, at, pt, is_extern) == 0) { arg_error(a, i, at, pt); }
        }
        i = i + 1;
    }
    check_call_regions(e, ft, self_off);
}

// ── builtins ─────────────────────────────────────────────────────────────────
int builtin_id(int sid) {
    if (str_eq_lit(sid, "__builtin_memory_size") == 1) { return BI_MEMSIZE; }
    if (str_eq_lit(sid, "__builtin_memory_grow") == 1) { return BI_MEMGROW; }
    if (str_eq_lit(sid, "__builtin_trap") == 1) { return BI_TRAP; }
    if (str_eq_lit(sid, "__builtin_unreachable") == 1) { return BI_UNREACH; }
    if (str_eq_lit(sid, "__builtin_sqrt") == 1) { return BI_SQRT; }
    if (str_eq_lit(sid, "__builtin_floor") == 1) { return BI_FLOOR; }
    if (str_eq_lit(sid, "__builtin_ceil") == 1) { return BI_CEIL; }
    if (str_eq_lit(sid, "__builtin_fabs") == 1) { return BI_FABS; }
    if (str_eq_lit(sid, "__builtin_trunc") == 1) { return BI_TRUNC; }
    if (str_eq_lit(sid, "__builtin_nearest") == 1) { return BI_NEAREST; }
    if (str_eq_lit(sid, "__builtin_heap_base") == 1) { return BI_HEAPBASE; }
    if (str_eq_lit(sid, "__builtin_clz") == 1 || str_eq_lit(sid, "__builtin_clzll") == 1) { return BI_CLZ; }
    if (str_eq_lit(sid, "__builtin_ctz") == 1 || str_eq_lit(sid, "__builtin_ctzll") == 1) { return BI_CTZ; }
    if (str_eq_lit(sid, "__builtin_popcount") == 1 || str_eq_lit(sid, "__builtin_popcountll") == 1) { return BI_POPCNT; }
    if (str_eq_lit(sid, "__builtin_f64_bits") == 1) { return BI_F64BITS; }
    if (str_eq_lit(sid, "__builtin_f64_from_bits") == 1) { return BI_F64FROM; }
    return 0;
}

int check_builtin(int e, int bi) {
    int n = g_node[e].nlist;
    int i = 0;
    int t0 = TY_VOID;
    while (i < n) {
        int a = g_list[g_node[e].list + i];
        int t = check_expr(a);
        if (i == 0) { t0 = t; }
        i = i + 1;
    }
    g_node[e].c = CK_BUILTIN;
    g_node[e].op = bi;
    if (bi == BI_MEMSIZE) { return TY_I32; }
    if (bi == BI_MEMGROW) { return TY_I32; }
    if (bi == BI_HEAPBASE) { return TY_U64; }
    if (bi == BI_TRAP || bi == BI_UNREACH) { return TY_VOID; }
    if (bi == BI_SQRT || bi == BI_FLOOR || bi == BI_CEIL || bi == BI_FABS || bi == BI_TRUNC || bi == BI_NEAREST) {
        if (n != 1 || ty_is_float(t0) == 0) {
            if (bi == BI_SQRT) { err_at(e, "__builtin_sqrt takes one float, double or vec<float/double, N>"); }
            else { err_at(e, "this builtin takes one float or double"); }
            return TY_ERROR;
        }
        return t0;
    }
    if (bi == BI_CLZ || bi == BI_CTZ || bi == BI_POPCNT) { return TY_I32; }
    if (bi == BI_F64BITS) { return TY_I64; }
    if (bi == BI_F64FROM) { return TY_F64; }
    return TY_ERROR;
}

// ── calls ────────────────────────────────────────────────────────────────────
int generic_call(int e, int tf, int recv_base);
int generic_static_call(int e, int tmpl_node, int mname);

int finish_direct(int e, int f, int self_off) {
    g_func[f].used = 1;
    queue_func(f);
    int fnode = g_func[f].node;
    if (g_pure == 1 && (g_node[fnode].flags & NF_PURE) == 0 && (g_node[fnode].flags & NF_EXTERN) == 0 && g_node[fnode].a != 0) {
        msg_reset();
        msg_lit("pure function calls non-pure function '");
        msg_str(g_func[f].sid);
        msg_lit("'");
        warn_node(e);
    }
    int is_extern = 0;
    if ((g_node[fnode].flags & NF_EXTERN) != 0 || g_node[fnode].a == 0) { is_extern = 1; }
    check_args(e, g_func[f].ty, self_off, is_extern);
    return func_ret(f);
}

int check_call(int e) {
    int expect = g_expect;
    g_expect = 0;
    int callee = g_node[e].a;
    int ck = g_node[callee].kind;
    if (ck == E_IDENT) {
        int sid = g_node[callee].sid;
        if (sid == S_va_ptr) {
            g_node[e].c = CK_VAPTR;
            return ty_ptr(TY_U8, 0);
        }
        int bi = builtin_id(sid);
        if (bi != 0 && lookup_local(sid) < 0 && g_gkind[sid] == 0) { return check_builtin(e, bi); }
    }
    if (ck == E_MEMBER && (g_node[callee].flags & NF_ARROW) == 0) {
        int fsid = g_node[callee].sid;
        int base = g_node[callee].a;
        // tagged union construction: Shape.radius(2.0)
        if (g_node[base].kind == E_IDENT && lookup_local(g_node[base].sid) < 0) {
            int ut = g_tyreg[g_node[base].sid];
            if (ut != 0 && g_ty[ut].kind == TY_STRUCT && g_sdef[g_ty[ut].n].kind == SD_UNION) {
                int sd = g_ty[ut].n;
                int f = find_field(sd, fsid);
                if (f >= 0) {
                    int i = 0;
                    while (i < g_node[e].nlist) {
                        int a = g_list[g_node[e].list + i];
                        int at = check_expr(a);
                        if (i == 0 && convertible_to(a, at, g_field[f].ty) == 0) { arg_error(a, 0, at, g_field[f].ty); }
                        i = i + 1;
                    }
                    g_node[e].c = CK_UNIONINIT;
                    g_node[e].d = sd;
                    g_node[e].op = f - g_sdef[sd].field0;
                    return ut;
                }
            }
        }
        if (fsid == S_is_null || fsid == S_is_none || fsid == S_default) {
            int bt = check_expr(base);
            int is_ptr = g_ty[bt].kind == TY_PTR;
            int is_nref = g_ty[bt].kind == TY_REF && (g_ty[bt].flags & TYF_NULL) != 0;
            int is_opt = g_ty[bt].kind == TY_OPT;
            if (is_ptr == 1 || is_nref == 1 || is_opt == 1) {
                g_node[e].b = base;
                if (fsid == S_is_null) {
                    if (is_opt == 1) { err_at(e, "'is_null()' is for pointers/nullable references; use 'is_none()' for optional (?T)"); }
                    if (g_node[e].nlist != 0) { err_at(e, "'is_null()' takes no arguments"); }
                    g_node[e].c = CK_ISNULL;
                    return TY_BOOL;
                }
                if (fsid == S_is_none) {
                    if (is_opt == 0) { err_at(e, "'is_none()' is for optional (?T); use 'is_null()' for pointers/nullable references"); }
                    if (g_node[e].nlist != 0) { err_at(e, "'is_none()' takes no arguments"); }
                    g_node[e].c = CK_ISNONE;
                    return TY_BOOL;
                }
                int inner = g_ty[bt].base;
                if (is_nref == 1) { inner = g_ty[bt].base; }
                if (g_node[e].nlist != 1) {
                    err_at(e, "'.default(value)' takes exactly 1 argument");
                    int i = 0;
                    while (i < g_node[e].nlist) { check_expr(g_list[g_node[e].list + i]); i = i + 1; }
                } else {
                    int a = g_list[g_node[e].list];
                    int at = check_expr(a);
                    if (is_err(at) == 0 && is_err(inner) == 0 && convertible_to(a, at, inner) == 0) {
                        msg_reset();
                        msg_lit("'.default(value)': fallback type '");
                        msg_type(at);
                        msg_lit("' does not match '");
                        msg_type(inner);
                        msg_lit("'");
                        error_node(e);
                    }
                }
                g_node[e].c = CK_DEFAULT;
                return inner;
            }
        }
    }
    if (ck == E_MEMBER) {
        int base = g_node[callee].a;
        int fsid = g_node[callee].sid;
        int bt = check_expr(base);
        check_nullable_deref(bt, e);
        int st = bt;
        if (g_ty[st].kind == TY_REF || g_ty[st].kind == TY_PTR) { st = g_ty[st].base; }
        int sd = struct_of(st);
        if (sd != 0) {
            int f = instance_method(sd, fsid);
            if (f != 0) {
                if ((g_func[f].flags & NF_SMETHOD) != 0) {
                    msg_reset();
                    msg_lit("'");
                    msg_str(fsid);
                    msg_lit("' is a static method of '");
                    msg_str(g_sdef[sd].sid);
                    msg_lit("'; call it as '");
                    msg_str(g_sdef[sd].sid);
                    msg_lit("::");
                    msg_str(fsid);
                    msg_lit("(...)'");
                    error_node(e);
                    int i = 0;
                    while (i < g_node[e].nlist) { check_expr(g_list[g_node[e].list + i]); i = i + 1; }
                    return TY_ERROR;
                }
                g_node[e].c = CK_METHOD;
                g_node[e].b = base;
                g_node[e].d = f;
                if (g_ty[bt].kind != TY_REF && g_ty[bt].kind != TY_PTR) { mark_addr(base); }
                if ((g_func[f].flags & NF_GENERIC) != 0) { return generic_call(e, f, base); }
                // a non-const method on a const receiver
                return finish_direct(e, f, 1);
            }
            layout_struct(sd);
            if (find_field(sd, fsid) < 0) {
                msg_reset();
                msg_lit("no method '");
                msg_str(fsid);
                msg_lit("' on type '");
                msg_str(g_sdef[sd].sid);
                msg_lit("'");
                error_node(e);
                int i = 0;
                while (i < g_node[e].nlist) { check_expr(g_list[g_node[e].list + i]); i = i + 1; }
                return TY_ERROR;
            }
        }
    }
    g_expect = expect;
    int ct = check_expr(callee);
    g_expect = 0;
    if (g_node[callee].kind == E_IDENT && g_node[callee].c == 0 && g_node[callee].d < 0) {
        // static method of a generic struct template
        int sid = g_node[callee].sid;
        int pos = str_find_colons(sid);
        int mname = substr_sid(sid, pos + 2, g_str_len[sid] - pos - 2);
        g_expect = expect;
        int r = generic_static_call(e, -g_node[callee].d, mname);
        g_expect = 0;
        return r;
    }
    if (g_node[callee].kind == E_IDENT && g_node[callee].c == SYM_FUNC) {
        int f = g_node[callee].d;
        if ((g_func[f].flags & NF_GENERIC) != 0) {
            g_expect = expect;
            int r = generic_call(e, f, 0);
            g_expect = 0;
            return r;
        }
        g_node[e].c = CK_FUNC;
        g_node[e].d = f;
        return finish_direct(e, f, 0);
    }
    int ft = fn_type_of(ct);
    if (ft == 0) {
        if (is_err(ct) == 0) { err_at(e, "expression is not callable"); }
        int i = 0;
        while (i < g_node[e].nlist) { check_expr(g_list[g_node[e].list + i]); i = i + 1; }
        return TY_ERROR;
    }
    g_node[e].c = CK_INDIRECT;
    check_args(e, ft, 0, 0);
    return g_ty[ft].base;
}

// generic function call (recv_base != 0: a generic method x.m(...))
int generic_call(int e, int tf, int recv_base) {
    int expect = g_expect;
    g_expect = 0;
    int fnode = g_func[tf].node;
    int gpn = g_node[fnode].b;
    int n = g_node[e].nlist;
    int ats[64];
    int i = 0;
    while (i < n && i < 64) {
        int a = g_list[g_node[e].list + i];
        ats[i] = check_expr(a);
        // a generic function passed as a value: bound after inference
        if (g_node[a].kind == E_IDENT && g_node[a].c == SYM_FUNC && (g_func[g_node[a].d].flags & NF_GENERIC) != 0) { ats[i] = 0; }
        i = i + 1;
    }
    int saved = g_nenv;
    if (g_func[tf].owner != 0) { push_struct_env(g_func[tf].owner); }
    push_pattern_env(gpn);
    int pat = func_type_from_node(fnode, 0);
    g_nenv = saved;
    g_nbind = 0;
    int ok = 1;
    i = 0;
    while (i < n && i < g_ty[pat].nlist) {
        int pt = g_tyl[g_ty[pat].list + i];
        if (ats[i] != 0 && mentions_generic(pt) == 1) {
            // a bare null binds nothing against a nullable reference
            int is_null = g_ty[ats[i]].kind == TY_PTR && is_void(g_ty[ats[i]].base) == 1 &&
                          g_ty[pt].kind == TY_REF && (g_ty[pt].flags & TYF_NULL) != 0;
            if (is_null == 0 && match_type(pt, ats[i]) == 0) { ok = 0; break; }
        }
        i = i + 1;
    }
    // unbound params: from the expected type
    int unbound = 0;
    i = 0;
    while (i < g_node[gpn].nlist) {
        if (bind_get(g_node[g_list[g_node[gpn].list + i]].sid) == 0) { unbound = 1; }
        i = i + 1;
    }
    if (ok == 1 && unbound == 1 && expect != 0) { match_type(g_ty[pat].base, expect); }
    i = 0;
    while (i < g_node[gpn].nlist && ok == 1) {
        if (bind_get(g_node[g_list[g_node[gpn].list + i]].sid) == 0) { ok = 0; }
        i = i + 1;
    }
    if (ok == 0) {
        msg_reset();
        msg_lit("cannot infer type arguments for generic function '");
        msg_str(g_func[tf].sid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (check_bounds(gpn, e) == 1) { return TY_ERROR; }
    int nf = instantiate_func(tf);
    g_func[nf].used = 1;
    queue_func(nf);
    int self_off = 0;
    if (recv_base != 0) { self_off = 1; g_node[e].c = CK_METHOD; g_node[e].b = recv_base; }
    else { g_node[e].c = CK_FUNC; }
    g_node[e].d = nf;
    g_node[g_node[e].a].d = nf;
    // each argument must still convert to its instantiated parameter
    int ft = g_func[nf].ty;
    int np = g_ty[ft].nlist;
    i = 0;
    while (i < n && i + self_off < np) {
        int a = g_list[g_node[e].list + i];
        int pt = g_tyl[g_ty[ft].list + i + self_off];
        int at = ats[i];
        if (at == 0) { at = coerce_generic_value(a, g_node[a].ty, pt); }
        if (arg_ok(a, at, pt, 0) == 0) { arg_error(a, i, at, pt); }
        i = i + 1;
    }
    if (n < np - self_off || ((g_ty[ft].flags & TYF_VARIADIC) == 0 && n > np - self_off)) {
        msg_reset();
        msg_lit("wrong number of arguments: expected ");
        msg_int((long)(np - self_off));
        msg_lit(", got ");
        msg_int((long)n);
        error_node(e);
    }
    return g_ty[ft].base;
}

// 'Box::of(5)' on a generic struct template: infer the struct's type
// arguments from the arguments (or the expected type), then call the
// instance's static method.
int generic_static_call(int e, int tmpl_node, int mname) {
    int expect = g_expect;
    g_expect = 0;
    int gpn = g_node[tmpl_node].b;
    // the method declaration in the template body (or its definition)
    int decl = 0;
    int i = 0;
    while (i < g_node[tmpl_node].nlist) {
        int m = g_list[g_node[tmpl_node].list + i];
        if (g_node[m].kind == D_FUNC && g_node[m].sid == mname) { decl = m; }
        i = i + 1;
    }
    if (decl == 0) {
        i = 0;
        while (i < g_ngmeth) {
            if (g_gmeth_owner[i] == g_node[tmpl_node].sid && g_node[g_gmeth_node[i]].sid == mname) { decl = g_gmeth_node[i]; }
            i = i + 1;
        }
    }
    if (decl == 0) {
        msg_reset();
        msg_lit("use of undeclared namespace member '");
        msg_str(g_node[g_node[e].a].sid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    int n = g_node[e].nlist;
    int ats[64];
    i = 0;
    while (i < n && i < 64) { ats[i] = check_expr(g_list[g_node[e].list + i]); i = i + 1; }
    int saved = g_nenv;
    push_pattern_env(gpn);
    int pat = func_type_from_node(decl, 0);
    g_nenv = saved;
    g_nbind = 0;
    i = 0;
    while (i < n && i < g_ty[pat].nlist) {
        int pt = g_tyl[g_ty[pat].list + i];
        if (mentions_generic(pt) == 1) { match_type(pt, ats[i]); }
        i = i + 1;
    }
    int unbound = 0;
    i = 0;
    while (i < g_node[gpn].nlist) {
        if (bind_get(g_node[g_list[g_node[gpn].list + i]].sid) == 0) { unbound = 1; }
        i = i + 1;
    }
    if (unbound == 1 && expect != 0) { match_type(g_ty[pat].base, expect); }
    int mark = tyl_begin();
    int bound[16];
    int nb = 0;
    i = 0;
    while (i < g_node[gpn].nlist && i < 16) {
        int b = bind_get(g_node[g_list[g_node[gpn].list + i]].sid);
        if (b == 0) {
            msg_reset();
            msg_lit("cannot infer type arguments for generic function '");
            msg_str(g_node[tmpl_node].sid);
            msg_lit("_");
            msg_str(mname);
            msg_lit("'");
            error_node(e);
            return TY_ERROR;
        }
        bound[nb] = b;
        nb = nb + 1;
        i = i + 1;
    }
    if (check_bounds(gpn, e) == 1) { return TY_ERROR; }
    mark = tyl_begin();
    i = 0;
    while (i < nb) { tyl_push(bound[i]); i = i + 1; }
    int st = instantiate_struct(tmpl_node, mark);
    int sd = struct_of(st);
    if (sd == 0) { return TY_ERROR; }
    int f = instance_method(sd, mname);
    if (f == 0) { return TY_ERROR; }
    g_node[g_node[e].a].c = SYM_FUNC;
    g_node[g_node[e].a].d = f;
    g_node[e].c = CK_FUNC;
    g_node[e].d = f;
    g_func[f].used = 1;
    queue_func(f);
    // re-check the arguments against the instance (already checked: convert only)
    int ft = g_func[f].ty;
    int np = g_ty[ft].nlist;
    if (n != np) {
        msg_reset();
        msg_lit("wrong number of arguments: expected ");
        msg_int((long)np);
        msg_lit(", got ");
        msg_int((long)n);
        error_node(e);
    }
    i = 0;
    while (i < n && i < np) {
        int a = g_list[g_node[e].list + i];
        int pt = g_tyl[g_ty[ft].list + i];
        if (arg_ok(a, ats[i], pt, 0) == 0) { arg_error(a, i, ats[i], pt); }
        i = i + 1;
    }
    return g_ty[ft].base;
}
