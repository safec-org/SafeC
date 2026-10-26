// stmt.sc — statement and function checking, and the sema driver.
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"

void check_stmt(int s);

// ── goto and labels (C++ Sema: collectLabels, checkGotoJumps) ────────────────
// the goto targets of the function being checked: labels outside 'match'
// arms and 'for' initializers
#define GL_MAX 256
int g_fl_sid[GL_MAX];
int g_fl_n = 0;

int fn_has_label(int sid) {
    int i = 0;
    while (i < g_fl_n) { if (g_fl_sid[i] == sid) { return 1; } i = i + 1; }
    return 0;
}

void collect_labels(int s) {
    if (s == 0) { return; }
    int k = g_node[s].kind;
    if (k == S_LABEL) {
        if (g_fl_n < GL_MAX) { g_fl_sid[g_fl_n] = g_node[s].sid; g_fl_n = g_fl_n + 1; }
        collect_labels(g_node[s].a);
    } else if (k == S_BLOCK) {
        int i = 0;
        while (i < g_node[s].nlist) { collect_labels(g_list[g_node[s].list + i]); i = i + 1; }
    } else if (k == S_IF) {
        collect_labels(g_node[s].b);
        collect_labels(g_node[s].c);
    } else if (k == S_WHILE || k == S_DO) {
        collect_labels(g_node[s].b);
    } else if (k == S_FOR) {
        collect_labels(g_node[s].d);
    } else if (k == S_UNSAFE) {
        collect_labels(g_node[s].a);
    } else if (k == S_SWITCH) {
        int i = 0;
        while (i < g_node[s].nlist) { collect_labels(g_node[g_list[g_node[s].list + i]].b); i = i + 1; }
    }
}

// scopes: the non-static declarations in scope while walking
#define GJ_MAX 4096
int g_gs_var[GJ_MAX];
int g_gs_n = 0;
int g_jl_sid[GL_MAX];      // labels seen: name, node, scope snapshot
int g_jl_node[GL_MAX];
int g_jl_start[GL_MAX];
int g_jl_len[GL_MAX];
int g_jl_n = 0;
int g_jg_node[GL_MAX];     // gotos: node, scope snapshot
int g_jg_start[GL_MAX];
int g_jg_len[GL_MAX];
int g_jg_n = 0;
int g_jv[GJ_MAX * 4];      // snapshot storage
int g_jv_n = 0;

int jump_snapshot() {
    int start = g_jv_n;
    int i = 0;
    while (i < g_gs_n && g_jv_n < GJ_MAX * 4) { g_jv[g_jv_n] = g_gs_var[i]; g_jv_n = g_jv_n + 1; i = i + 1; }
    return start;
}

void jump_walk(int s) {
    if (s == 0) { return; }
    int mark = g_gs_n;
    int k = g_node[s].kind;
    if (k == S_BLOCK) {
        int i = 0;
        while (i < g_node[s].nlist) { jump_walk(g_list[g_node[s].list + i]); i = i + 1; }
        g_gs_n = mark;
    } else if (k == S_VAR) {
        if ((g_node[s].flags & NF_STATIC) == 0 && g_gs_n < GJ_MAX) { g_gs_var[g_gs_n] = s; g_gs_n = g_gs_n + 1; }
    } else if (k == S_MULTI) {
        int i = 0;
        while (i < g_node[s].nlist) { jump_walk(g_list[g_node[s].list + i]); i = i + 1; }
    } else if (k == S_IF || k == S_IFCONST) {
        if (g_node[s].b != 0) { jump_walk(g_node[s].b); g_gs_n = mark; }
        if (g_node[s].c != 0) { jump_walk(g_node[s].c); g_gs_n = mark; }
    } else if (k == S_WHILE || k == S_DO) {
        jump_walk(g_node[s].b);
        g_gs_n = mark;
    } else if (k == S_FOR) {
        jump_walk(g_node[s].a);
        jump_walk(g_node[s].d);
        g_gs_n = mark;
    } else if (k == S_UNSAFE) {
        jump_walk(g_node[s].a);
        g_gs_n = mark;
    } else if (k == S_SWITCH) {
        int i = 0;
        while (i < g_node[s].nlist) {
            int c = g_list[g_node[s].list + i];
            int b = g_node[c].b;
            int j = 0;
            while (b != 0 && j < g_node[b].nlist) { jump_walk(g_list[g_node[b].list + j]); j = j + 1; }
            g_gs_n = mark;
            i = i + 1;
        }
    } else if (k == S_MATCH) {
        int i = 0;
        while (i < g_node[s].nlist) {
            jump_walk(g_node[g_list[g_node[s].list + i]].a);
            g_gs_n = mark;
            i = i + 1;
        }
    } else if (k == S_LABEL) {
        int sid = g_node[s].sid;
        int i = 0;
        int dup = -1;
        while (i < g_jl_n) { if (g_jl_sid[i] == sid) { dup = i; } i = i + 1; }
        if (dup >= 0) {
            msg_reset();
            msg_lit("duplicate label '");
            msg_str(sid);
            msg_lit("' (first defined at line ");
            msg_int((long)g_tok[g_node[g_jl_node[dup]].tok].line);
            msg_lit(")");
            error_node(s);
        } else if (g_jl_n < GL_MAX) {
            g_jl_sid[g_jl_n] = sid;
            g_jl_node[g_jl_n] = s;
            g_jl_start[g_jl_n] = jump_snapshot();
            g_jl_len[g_jl_n] = g_gs_n;
            g_jl_n = g_jl_n + 1;
        }
        jump_walk(g_node[s].a);
    } else if (k == S_GOTO) {
        if (g_jg_n < GL_MAX) {
            g_jg_node[g_jg_n] = s;
            g_jg_start[g_jg_n] = jump_snapshot();
            g_jg_len[g_jg_n] = g_gs_n;
            g_jg_n = g_jg_n + 1;
        }
    }
}

void check_goto_jumps(int body) {
    g_gs_n = 0;
    g_jl_n = 0;
    g_jg_n = 0;
    g_jv_n = 0;
    jump_walk(body);
    int g = 0;
    while (g < g_jg_n) {
        int gn = g_jg_node[g];
        int li = -1;
        int i = 0;
        while (i < g_jl_n) { if (g_jl_sid[i] == g_node[gn].sid) { li = i; } i = i + 1; }
        if (li >= 0) {
            i = 0;
            while (i < g_jl_len[li]) {
                int vd = g_jv[g_jl_start[li] + i];
                int seen = 0;
                int j = 0;
                while (j < g_jg_len[g]) { if (g_jv[g_jg_start[g] + j] == vd) { seen = 1; } j = j + 1; }
                if (seen == 0) {
                    msg_reset();
                    msg_lit("goto '");
                    msg_str(g_node[gn].sid);
                    msg_lit("' jumps over the declaration of '");
                    msg_str(g_node[vd].sid);
                    msg_lit("' (line ");
                    msg_int((long)g_tok[g_node[vd].tok].line);
                    msg_lit("), which is still in scope at the label but would never be initialized; ");
                    msg_lit("declare it before the goto, or in a block that ends before the label");
                    error_node(gn);
                    break;
                }
                i = i + 1;
            }
        }
        g = g + 1;
    }
}

void check_block_list(int s) {
    int i = 0;
    while (i < g_node[s].nlist) { check_stmt(g_list[g_node[s].list + i]); i = i + 1; }
}

void check_block(int s) {
    push_scope(0);
    check_block_list(s);
    pop_scope();
}

void check_cond(int c, int is_while) {
    int t = check_expr(c);
    int k = g_ty[t].kind;
    if (is_err(t) == 1) { return; }
    if (ty_is_int(t) == 1 || k == TY_PTR) { return; }
    if (is_while == 0 && k == TY_REF) { return; }
    if (is_while == 1) { err_at(c, "while condition must be boolean or numeric"); }
    else { err_at(c, "condition must be boolean or numeric"); }
}

int strip_arr(int t) {
    while (g_ty[t].kind == TY_ARRAY) { t = g_ty[t].base; }
    return t;
}

void check_var(int s) {
    int dt = 0;
    if (g_node[s].c != 0) { dt = resolve_type(g_node[s].c); }
    if ((g_node[s].flags & NF_VOLATILE) != 0) { dt = with_volatile_elements(dt); }
    int init = g_node[s].a;
    int it = 0;
    int is_static = (g_node[s].flags & NF_STATIC) != 0;
    if (init != 0) {
        int saved = g_expect;
        if (dt != 0 && is_err(dt) == 0) { g_expect = dt; }
        if (dt != 0) { adopt_tuple(init, dt); }
        if (g_node[init].kind == E_INITLIST) { it = check_initlist(init, dt); }
        else { it = check_expr(init); }
        g_expect = saved;
        // borrow tracking: '&R T r = &x' with the declared mutability
        if (in_unsafe() == 0 && dt != 0 && g_ty[dt].kind == TY_REF && g_node[init].kind == E_ADDR) {
            int o = g_node[init].a;
            if (g_node[o].kind == E_IDENT && (g_node[o].c == SYM_LOCAL || g_node[o].c == SYM_GLOBAL) && g_ty[dt].n != R_STATIC) {
                int is_mut = (g_ty[dt].flags & TYF_CONST) == 0;
                track_ref(g_node[o].sid, is_mut, g_node[s].sid, s);
            }
        }
        if (in_unsafe() == 0 && g_ty[it].kind == TY_REF && g_node[init].kind == E_IDENT &&
            (g_node[init].c == SYM_LOCAL || g_node[init].c == SYM_GLOBAL) && g_ty[it].n != R_STATIC) {
            int is_mut = (g_ty[it].flags & TYF_CONST) == 0;
            track_ref(g_node[init].sid, is_mut, g_node[s].sid, s);
        }
        if (dt != 0) { it = coerce_generic_value(init, it, dt); }
        if (dt != 0 && is_err(dt) == 0 && is_err(it) == 0) {
            int ok = convertible_to(init, it, dt);
            if (ok == 0) {
                msg_reset();
                msg_lit("type mismatch in variable declaration: cannot convert '");
                msg_type(it);
                msg_lit("' to '");
                msg_type(dt);
                msg_lit("'");
                msg_raw_ref_hint(it, dt);
                error_node(s);
            }
        }
    }
    if (dt == 0 || is_err(dt) == 1) {
        if (it != 0) { dt = it; } else { dt = TY_ERROR; }
    }
    // unsized array: size from the initializer
    if (g_ty[dt].kind == TY_ARRAY && g_ty[dt].n < 0 && init != 0) {
        int n = 0;
        if (g_node[init].kind == E_INITLIST) { n = (int)g_node[init].ival; }
        else if (g_node[init].kind == E_STR) { n = g_str_len[g_node[init].sid] + 1; }
        dt = ty_array(g_ty[dt].base, n);
        if (g_node[init].kind == E_INITLIST) { g_node[init].ty = dt; }
    }
    int k = g_ty[dt].kind;
    int aggregate = k == TY_STRUCT || k == TY_ARRAY;
    int flags = 0;
    int initialized = 0;
    if (init != 0 || aggregate == 1 || is_static == 1) { initialized = 1; flags = LF_INIT; }
    if ((g_node[s].flags & NF_CONST) != 0) { flags = flags | LF_CONST; }
    if (is_static == 1) { flags = flags | LF_STATIC; }
    if ((g_node[s].flags & NF_VOLATILE) != 0) { flags = flags | LF_VOLATILE; }
    int l = new_local(g_node[s].sid, dt, flags, s);
    g_node[s].d = l;
    g_node[s].ty = dt;
    define_sym(g_node[s].sid, SYM_LOCAL, l, dt, initialized);
}

void check_switch(int s) {
    int t = check_expr(g_node[s].a);
    if (is_err(t) == 0 && ty_is_int(t) == 0 && g_ty[t].kind != TY_ENUM) {
        msg_reset();
        msg_lit("switch subject must be an integer, char, or bool type, got '");
        msg_type(t);
        msg_lit("'");
        error_node(g_node[s].a);
    }
    int saw_default = 0;
    long seen[512];
    int nseen = 0;
    int i = 0;
    while (i < g_node[s].nlist) {
        int c = g_list[g_node[s].list + i];
        if ((g_node[c].flags & NF_DEFAULT) != 0) { saw_default = 1; }
        int j = 0;
        while (j < g_node[c].nlist) {
            long v = g_node[g_list[g_node[c].list + j]].ival;
            int q = 0;
            int dup = 0;
            while (q < nseen) { if (seen[q] == v) { dup = 1; } q = q + 1; }
            if (dup == 1) {
                msg_reset();
                msg_lit("duplicate case value ");
                msg_int(v);
                msg_lit(" in switch");
                error_node(c);
            } else if (nseen < 512) { seen[nseen] = v; nseen = nseen + 1; }
            j = j + 1;
        }
        push_scope(0);
        g_loop_depth = g_loop_depth + 1;
        check_block_list(g_node[c].b);
        g_loop_depth = g_loop_depth - 1;
        pop_scope();
        i = i + 1;
    }
    if (saw_default == 0) {
        msg_reset();
        msg_lit("switch statement may not be exhaustive (no 'default' label)");
        warn_node(s);
    }
}

void check_match_stmt(int s) {
    int st = check_expr(g_node[s].a);
    classify_subject(st);
    int nulllike = g_mt_nulllike;
    int opt = g_mt_opt;
    int un = g_mt_union;
    int en = g_mt_enum;
    int payload = g_mt_payload;
    g_mt_wild = 0;
    g_mt_ncov = 0;
    int i = 0;
    while (i < g_node[s].nlist) {
        int arm = g_list[g_node[s].list + i];
        g_mt_nulllike = nulllike; g_mt_opt = opt; g_mt_union = un; g_mt_enum = en; g_mt_payload = payload;
        push_scope(0);
        resolve_patterns(arm, s);
        g_loop_depth = g_loop_depth + 1;
        check_stmt(g_node[arm].a);
        g_loop_depth = g_loop_depth - 1;
        pop_scope();
        i = i + 1;
    }
    int wild = g_mt_wild;
    // null/none plus some(x) covers a nullable/optional subject
    if ((nulllike == 1 || opt == 1) && g_mt_ncov == 2) { wild = 1; }
    if (wild == 0) {
        msg_reset();
        msg_lit("match statement may not be exhaustive (no wildcard arm)");
        warn_node(s);
    }
    g_node[s].c = st;
}

void check_return(int s) {
    int v = g_node[s].a;
    int rt = g_cur_ret;
    if (v != 0) {
        if (rt != 0) { adopt_tuple(v, rt); }
        int saved = g_expect;
        if (rt != 0 && is_err(rt) == 0 && g_ty[rt].kind != TY_VOID) { g_expect = rt; }
        int vt = 0;
        if (g_node[v].kind == E_INITLIST) { vt = check_initlist(v, rt); }
        else { vt = check_expr(v); }
        g_expect = saved;
        if (rt != 0) { vt = coerce_generic_value(v, vt, rt); }
        if (g_ty[vt].kind == TY_REF && g_ty[vt].n == R_STACK) {
            msg_reset();
            msg_lit("cannot return '&stack ");
            msg_type(g_ty[vt].base);
            msg_lit("': stack reference escapes function scope");
            error_node(s);
        }
        if (rt != 0 && g_ty[rt].kind != TY_VOID && is_err(vt) == 0 && is_err(rt) == 0) {
            if (convertible_to(v, vt, rt) == 0) {
                msg_reset();
                msg_lit("return type mismatch: cannot convert '");
                msg_type(vt);
                msg_lit("' to '");
                msg_type(rt);
                msg_lit("'");
                msg_raw_ref_hint(vt, rt);
                error_node(s);
            }
        }
        if ((g_func[g_cur_fn].flags & NF_NORETURN) != 0) {
            msg_reset();
            msg_lit("noreturn function must not return a value");
            error_node(s);
        }
    } else if (rt != 0 && g_ty[rt].kind != TY_VOID) {
        err_at(s, "non-void function must return a value");
    }
}

void check_static_assert(int s) {
    int c = g_node[s].a;
    check_expr(c);
    if (sema_const_int(c) == 1) {
        if (g_cval == 0L) {
            msg_reset();
            msg_lit("static_assert failed");
            if (g_node[s].sid != 0 && g_str_len[g_node[s].sid] > 0) { msg_lit(": "); msg_str(g_node[s].sid); }
            error_node(s);
        }
    } else {
        g_ce_bad = 0;
        sema_const_int(c);
        const_error(c);
    }
}

void check_stmt(int s) {
    if (s == 0) { return; }
    g_tyloc = s;
    int k = g_node[s].kind;
    if (k == S_BLOCK) { check_block(s); return; }
    if (k == S_MULTI) { check_block_list(s); return; }
    if (k == S_VAR) { check_var(s); return; }
    if (k == S_EXPR) {
        int e = g_node[s].a;
        check_expr(e);
        if (g_node[e].kind == E_CALL && (g_node[e].c == CK_FUNC) && g_node[e].d > 0) {
            int f = g_node[e].d;
            if ((g_func[f].flags & NF_MUSTUSE) != 0) {
                msg_reset();
                msg_lit("return value of '");
                msg_str(g_func[f].sid);
                msg_lit("' should not be ignored (marked must_use)");
                warn_node(e);
            }
        }
        return;
    }
    if (k == S_IF) {
        check_cond(g_node[s].a, 0);
        check_stmt(g_node[s].b);
        if (g_node[s].c != 0) { check_stmt(g_node[s].c); }
        return;
    }
    if (k == S_IFCONST) {
        int c = g_node[s].a;
        check_expr(c);
        if (sema_const_int(c) == 1) {
            if (g_cval != 0L) { g_node[s].op = 1; check_stmt(g_node[s].b); }
            else { g_node[s].op = 2; if (g_node[s].c != 0) { check_stmt(g_node[s].c); } }
            return;
        }
        g_ce_bad = 0;
        sema_const_int(c);
        const_error(c);
        return;
    }
    if (k == S_WHILE || k == S_DO) {
        check_cond(g_node[s].a, 1);
        g_loop_depth = g_loop_depth + 1;
        check_stmt(g_node[s].b);
        g_loop_depth = g_loop_depth - 1;
        return;
    }
    if (k == S_FOR) {
        push_scope(0);
        if (g_node[s].a != 0) { check_stmt(g_node[s].a); }
        if (g_node[s].b != 0) { check_expr(g_node[s].b); }
        if (g_node[s].c != 0) { check_expr(g_node[s].c); }
        g_loop_depth = g_loop_depth + 1;
        check_stmt(g_node[s].d);
        g_loop_depth = g_loop_depth - 1;
        pop_scope();
        return;
    }
    if (k == S_RETURN) { check_return(s); return; }
    if (k == S_BREAK) {
        if (g_loop_depth == 0) { err_at(s, "'break' statement not inside a loop or match"); }
        return;
    }
    if (k == S_CONTINUE) {
        if (g_loop_depth == 0) { err_at(s, "'continue' statement not inside a loop"); }
        return;
    }
    if (k == S_UNSAFE) {
        push_scope(1);
        check_block(g_node[s].a);
        pop_scope();
        return;
    }
    if (k == S_DEFER) { check_stmt(g_node[s].a); return; }
    if (k == S_MATCH) { check_match_stmt(s); return; }
    if (k == S_SWITCH) { check_switch(s); return; }
    if (k == S_STATIC_ASSERT) { check_static_assert(s); return; }
    if (k == S_GOTO) {
        if (fn_has_label(g_node[s].sid) == 0) {
            msg_reset();
            msg_lit("use of undeclared label '");
            msg_str(g_node[s].sid);
            msg_lit("'");
            error_node(s);
        }
        return;
    }
    if (k == S_LABEL) { check_stmt(g_node[s].a); return; }
}

// ── functions ────────────────────────────────────────────────────────────────
void check_function(int f) {
    int fnode = g_func[f].node;
    int body = g_node[fnode].a;
    if (body == 0) { return; }
    int saved_fn = g_cur_fn;
    int saved_ret = g_cur_ret;
    int saved_loop = g_loop_depth;
    int saved_env = g_nenv;
    int saved_pure = g_pure;
    g_cur_fn = f;
    g_func[f].loc0 = g_nlocal;
    g_func[f].nloc = 0;
    if (g_func[f].tmpl == 1) { push_struct_env(g_func[f].env); }
    if (g_func[f].tmpl == 2) {
        int tf = g_inst_tmpl[g_func[f].env];
        if (g_func[tf].owner != 0) { push_struct_env(g_func[tf].owner); }
        // push_inst_env
        int gpn = g_node[g_func[tf].node].b;
        int i = 0;
        int inst = g_func[f].env;
        while (i < g_node[gpn].nlist && i < g_inst_nargs[inst]) {
            env_push(g_node[g_list[g_node[gpn].list + i]].sid, g_tyl[g_inst_args[inst] + i]);
            i = i + 1;
        }
    }
    g_cur_ret = func_ret(f);
    g_loop_depth = 0;
    g_pure = 0;
    if ((g_func[f].flags & NF_PURE) != 0) { g_pure = 1; }
    push_scope(0);
    int pi = 0;
    int ft = g_func[f].ty;
    if (g_func[f].owner != 0 && (g_func[f].flags & NF_SMETHOD) == 0) {
        int st = g_tyl[g_ty[ft].list];
        int l = new_local(S_self, st, LF_PARAM | LF_INIT, 0);
        define_sym(S_self, SYM_LOCAL, l, st, 1);
        pi = 1;
    }
    int i = 0;
    while (i < g_node[fnode].nlist) {
        int p = g_list[g_node[fnode].list + i];
        int pt = g_tyl[g_ty[ft].list + pi + i];
        int pfl = LF_PARAM | LF_INIT;
        if ((g_node[p].flags & NF_VOLATILE) != 0) { pfl = pfl | LF_VOLATILE; }
        int l = new_local(g_node[p].sid, pt, pfl, p);
        if (g_node[p].sid != 0) { define_sym(g_node[p].sid, SYM_LOCAL, l, pt, 1); }
        i = i + 1;
    }
    int saved_fl = g_fl_n;
    g_fl_n = 0;
    collect_labels(body);
    check_goto_jumps(body);
    check_block(body);
    g_fl_n = saved_fl;
    pop_scope();
    g_cur_fn = saved_fn;
    g_cur_ret = saved_ret;
    g_loop_depth = saved_loop;
    g_nenv = saved_env;
    g_pure = saved_pure;
}

void check_global(int g) {
    int n = g_glob[g].node;
    int init = g_node[n].a;
    if (init != 0) {
        int gt = g_glob[g].ty;
        int it = 0;
        adopt_tuple(init, gt);
        if (g_node[init].kind == E_INITLIST) { it = check_initlist(init, gt); }
        else { it = check_expr(init); }
        int ok = convertible_to(init, it, gt);
        if (ok == 0) {
            msg_reset();
            msg_lit("type mismatch in global variable initializer: cannot convert '");
            msg_type(it);
            msg_lit("' to '");
            msg_type(gt);
            msg_lit("'");
            error_node(n);
        }
        if (g_ty[gt].kind == TY_ARRAY && g_ty[gt].n < 0) {
            int cnt = 0;
            if (g_node[init].kind == E_INITLIST) { cnt = (int)g_node[init].ival; }
            else if (g_node[init].kind == E_STR) { cnt = g_str_len[g_node[init].sid] + 1; }
            g_glob[g].ty = ty_array(g_ty[gt].base, cnt);
            g_node[init].ty = g_glob[g].ty;
        }
    }
    g_glob[g].done = 1;
}

void drain_queue() {
    while (g_qpos < g_nqueue) {
        int f = g_queue[g_qpos];
        g_qpos = g_qpos + 1;
        check_function(f);
    }
}

void register_builtin_types() {
    g_tyreg[intern_lit("int8_t")] = TY_I8;
    g_tyreg[intern_lit("int16_t")] = TY_I16;
    g_tyreg[intern_lit("int32_t")] = TY_I32;
    g_tyreg[intern_lit("int64_t")] = TY_I64;
    g_tyreg[intern_lit("uint8_t")] = TY_U8;
    g_tyreg[intern_lit("uint16_t")] = TY_U16;
    g_tyreg[intern_lit("uint32_t")] = TY_U32;
    g_tyreg[intern_lit("uint64_t")] = TY_U64;
}

// supertrait cycles (C++ Sema::checkTraitSupertraits): an offending edge is
// reported and dropped (flags 1), as are unknown supertraits
int g_tr_state[NSTR_MAX];

void visit_trait(int n) {
    int t = g_node[n].sid;
    g_tr_state[t] = 1;
    int sup = g_node[n].b;
    int j = 0;
    while (sup != 0 && j < g_node[sup].nlist) {
        int el = g_list[g_node[sup].list + j];
        int s = g_node[el].sid;
        j = j + 1;
        if (g_node[el].flags == 1 || g_trait[s] == 0) { continue; }
        if (g_tr_state[s] == 1) {
            msg_reset();
            msg_lit("trait '");
            msg_str(t);
            msg_lit("' is its own supertrait (through '");
            msg_str(s);
            msg_lit("')");
            error_node(n);
            g_node[el].flags = 1;
            continue;
        }
        if (g_tr_state[s] == 0) { visit_trait(g_trait[s]); }
    }
    g_tr_state[t] = 2;
}

void check_traits() {
    int i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        if (g_node[n].kind == D_TRAIT && g_node[n].b != 0) {
            int sup = g_node[n].b;
            int j = 0;
            while (j < g_node[sup].nlist) {
                int s = g_node[g_list[g_node[sup].list + j]].sid;
                int known = g_trait[s] != 0 || str_eq_lit(s, "Numeric") == 1 || str_eq_lit(s, "Eq") == 1 ||
                            str_eq_lit(s, "Ord") == 1 || str_eq_lit(s, "Add") == 1 || str_eq_lit(s, "Sub") == 1 ||
                            str_eq_lit(s, "Mul") == 1 || str_eq_lit(s, "Div") == 1 || str_eq_lit(s, "Indexed") == 1 ||
                            str_eq_lit(s, "Pointer") == 1 || str_eq_lit(s, "Send") == 1 ||
                            str_eq_lit(s, "Sync") == 1 || str_eq_lit(s, "Drop") == 1;
                if (known == 0) {
                    msg_reset();
                    msg_lit("unknown supertrait '");
                    msg_str(s);
                    msg_lit("' of trait '");
                    msg_str(g_node[n].sid);
                    msg_lit("'");
                    error_node(n);
                    g_node[g_list[g_node[sup].list + j]].flags = 1;
                }
                j = j + 1;
            }
        }
        i = i + 1;
    }
    i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        if (g_node[n].kind == D_TRAIT && g_tr_state[g_node[n].sid] == 0) { visit_trait(n); }
        i = i + 1;
    }
}

void sema_program() {
    register_builtin_types();
    g_depth = 0;
    sema_collect();
    check_traits();
    int i = 0;
    while (i < g_ntop) {
        int n = g_top[i];
        int k = g_node[n].kind;
        if (k == D_GLOBAL) {
            int q = g_node[n].sid;
            if (g_node[n].op != 0) { q = str_concat3(g_node[n].op, "::", q); }
            int g = g_gidx[q];
            if (g_glob[g].node == n) { check_global(g); }
            else { g_glob[g].done = 1; }
        } else if (k == D_FUNC && g_node[n].a != 0 && (g_node[n].flags & NF_GENERIC) == 0) {
            int f = 0;
            if ((g_node[n].flags & NF_METHOD) != 0) {
                int owner = g_node[n].d;
                if (g_gtmpl[owner] == 0 && g_tyreg[owner] != 0) {
                    int sd = struct_of(g_tyreg[owner]);
                    if (sd != 0) { f = lookup_method(sd, g_node[n].sid); }
                }
            } else {
                int q = g_node[n].sid;
                if (g_node[n].op != 0) { q = str_concat3(g_node[n].op, "::", q); }
                if (g_gkind[q] == SYM_FUNC) { f = g_gidx[q]; }
            }
            if (f != 0 && g_func[f].node == n && g_func[f].checked == 0) {
                g_func[f].checked = 1;
                check_function(f);
            }
        } else if (k == D_STATIC_ASSERT) {
            check_static_assert(n);
        }
        i = i + 1;
    }
    drain_queue();
    memflow_program();
}
