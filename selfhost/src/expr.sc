// expr.sc — expression type checking (C++ Sema::checkExpr and friends).
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"

int g_expect = 0;        // expected type for generic inference of the next call
int g_pure = 0;          // checking a 'pure' function

int set_ty(int e, int t) { g_node[e].ty = t; return t; }
int is_lval(int e) { return (g_node[e].flags & NF_LVAL) != 0; }
void mark_lval(int e) { g_node[e].flags = g_node[e].flags | NF_LVAL; }

void err_at(int e, const char* s) { msg_reset(); msg_lit(s); error_node(e); }

int err_ty() { return TY_ERROR; }

int sema_check_expr_ty(int e) { return check_expr(e); }

// ── identifiers ──────────────────────────────────────────────────────────────
int str_find_colons(int sid) {
    int i = g_str_len[sid] - 2;
    while (i >= 0) {
        if (str_char(sid, i) == ':' && str_char(sid, i + 1) == ':') { return i; }
        i = i - 1;
    }
    return -1;
}

int substr_sid(int sid, int from, int len) {
    tmp_reset();
    int i = 0;
    while (i < len) { tmp_char(str_char(sid, from + i)); i = i + 1; }
    return intern_tmp();
}

int func_value_type(int f) {
    if (g_func[f].ty == 0) { return TY_ERROR; }
    return ty_ref(g_func[f].ty, R_STATIC, 0, 0);
}

// 'Owner::m' → static method func (0 if none); reports errors
int g_static_owner_sd = 0;
int resolve_static_method(int e, int sid) {
    int pos = str_find_colons(sid);
    int owner = substr_sid(sid, 0, pos);
    int mname = substr_sid(sid, pos + 2, g_str_len[sid] - pos - 2);
    // drop a namespace prefix on the owner ('std::Vec::new')
    int p2 = str_find_colons(owner);
    if (p2 >= 0) { owner = substr_sid(owner, p2 + 2, g_str_len[owner] - p2 - 2); }
    g_static_owner_sd = 0;
    int ot = env_lookup(owner);
    if (ot == 0) { ot = g_tyreg[owner]; }
    if (ot != 0 && struct_of(ot) != 0) {
        int sd = struct_of(ot);
        g_static_owner_sd = sd;
        int f = instance_method(sd, mname);
        if (f != 0) {
            if ((g_func[f].flags & NF_SMETHOD) == 0) {
                msg_reset();
                msg_lit("'");
                msg_str(g_sdef[sd].sid);
                msg_lit("::");
                msg_str(mname);
                msg_lit("' is not a static method; call it on a value ('x.");
                msg_str(mname);
                msg_lit("(...)')");
                error_node(e);
                return -1;
            }
            return f;
        }
    }
    if (g_gtmpl[owner] != 0) {
        // static method of a generic struct template: resolved at the call
        g_static_owner_sd = -g_gtmpl[owner];
        return 0;
    }
    return 0;
}

int check_ident(int e) {
    int sid = g_node[e].sid;
    int ls = lookup_local(sid);
    if (ls >= 0) {
        int k = g_sym[ls].kind;
        g_node[e].c = k;
        g_node[e].d = g_sym[ls].idx;
        if (g_sym[ls].init == 0) {
            msg_reset();
            msg_lit("use of possibly uninitialized variable '");
            msg_str(sid);
            msg_lit("'");
            error_node(e);
        }
        if (k == SYM_LOCAL || k == SYM_BIND) {
            int t = g_sym[ls].ty;
            if (k == SYM_LOCAL || g_ty[t].kind == TY_REF) { mark_lval(e); }
        }
        return g_sym[ls].ty;
    }
    int k = g_gkind[sid];
    int idx = g_gidx[sid];
    int qualified = str_find_colons(sid) >= 0;
    if (k == 0 && qualified == 0) { k = g_nsfb_kind[sid]; idx = g_nsfb_idx[sid]; }
    if (k == 0 && qualified == 1) {
        int f = resolve_static_method(e, sid);
        if (f < 0) { return TY_ERROR; }
        if (f > 0) {
            g_node[e].c = SYM_FUNC;
            g_node[e].d = f;
            return func_value_type(f);
        }
        if (g_static_owner_sd < 0) {
            g_node[e].c = 0;
            g_node[e].d = g_static_owner_sd;   // template marker, resolved by check_call
            return TY_ERROR;
        }
        msg_reset();
        msg_lit("use of undeclared namespace member '");
        msg_str(sid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (k == 0) {
        msg_reset();
        msg_lit("use of undeclared identifier '");
        msg_str(sid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    g_node[e].c = k;
    g_node[e].d = idx;
    if (k == SYM_FUNC) {
        if ((g_func[idx].flags & NF_GENERIC) != 0) { return TY_ERROR; }
        return func_value_type(idx);
    }
    if (k == SYM_GLOBAL) {
        if (g_glob[idx].done == 0) {
            msg_reset();
            msg_lit("use of possibly uninitialized variable '");
            msg_str(sid);
            msg_lit("'");
            error_node(e);
        }
        mark_lval(e);
        return g_glob[idx].ty;
    }
    if (k == SYM_ENUMC) { return g_field[idx].ty; }
    return TY_ERROR;
}

// marks the local behind an lvalue chain as addressable (lives in memory)
void mark_addr(int e) {
    int cur = e;
    while (cur != 0) {
        int k = g_node[cur].kind;
        if (k == E_IDENT) {
            if (g_node[cur].c == SYM_LOCAL || g_node[cur].c == SYM_BIND) {
                int l = g_node[cur].d;
                g_local[l].flags = g_local[l].flags | LF_ADDR;
            }
            return;
        }
        if (k == E_MEMBER || k == E_INDEX) {
            int b = g_node[cur].a;
            int bt = g_node[b].ty;
            if (g_ty[bt].kind == TY_REF || g_ty[bt].kind == TY_PTR || g_ty[bt].kind == TY_SLICE) { return; }
            cur = b;
            continue;
        }
        return;
    }
}

// ── unary ────────────────────────────────────────────────────────────────────
void check_nullable_deref(int t, int e) {
    if (g_ty[t].kind != TY_REF) { return; }
    if ((g_ty[t].flags & TYF_NULL) != 0 && in_unsafe() == 0) {
        msg_reset();
        msg_lit("dereference of nullable reference '?&");
        msg_type(g_ty[t].base);
        msg_lit("' requires 'unsafe', or use match/is_null()/.default(value) instead");
        error_node(e);
    }
}

// '&x' of a volatile lvalue is a reference to volatile (C++ isVolatileLValue)
int is_volatile_lval(int e) {
    int k = g_node[e].kind;
    if (k == E_IDENT) {
        if (g_node[e].c == SYM_LOCAL) { return (g_local[g_node[e].d].flags & LF_VOLATILE) != 0; }
        if (g_node[e].c == SYM_GLOBAL) { return (g_glob[g_node[e].d].flags & NF_VOLATILE) != 0; }
        return 0;
    }
    if (k == E_DEREF || k == E_FORCE) { return pointee_volatile(g_node[g_node[e].a].ty); }
    if (k == E_INDEX) {
        int bt = g_node[g_node[e].a].ty;
        if (pointee_volatile(bt) == 1) { return 1; }
        return g_ty[bt].kind == TY_ARRAY && is_volatile_lval(g_node[e].a) == 1;
    }
    if (k == E_MEMBER) {
        int b = g_node[e].a;
        int bt = g_node[b].ty;
        int indirect = g_ty[bt].kind == TY_PTR || g_ty[bt].kind == TY_REF || (g_node[e].flags & NF_ARROW) != 0;
        if (indirect == 1 && pointee_volatile(bt) == 1) { return 1; }
        if (g_node[e].d == MK_FIELD) {
            int fi = g_node[e].c;
            if (fi >= 0 && fi < g_nfield && g_field[fi].node != 0 && (g_node[g_field[fi].node].flags & NF_VOLATILE) != 0) { return 1; }
        }
        if (indirect == 0) { return is_volatile_lval(b); }
    }
    return 0;
}

int check_addr(int e) {
    int op = g_node[e].a;
    int ot = check_expr(op);
    if (g_node[op].kind == E_IDENT && g_node[op].c == SYM_FUNC) { return ot; }
    int k = g_node[op].kind;
    int designates = (k == E_INDEX || k == E_MEMBER);
    if (is_lval(op) == 0 && designates == 0) { err_at(e, "address-of operator requires lvalue"); }
    int is_mut = is_lval(op);
    // region of the result
    int rgn = R_STACK;
    int have = 0;
    int base_global = 0;
    int cur = op;
    while (cur != 0) {
        int ck = g_node[cur].kind;
        if (ck == E_IDENT) {
            if (g_node[cur].c == SYM_GLOBAL) { base_global = 1; }
            if (g_node[cur].c == SYM_LOCAL) {
                int l = g_node[cur].d;
                if ((g_local[l].flags & LF_STATIC) != 0) { base_global = 1; }
            }
            break;
        }
        if (ck == E_MEMBER || ck == E_INDEX) {
            int b = g_node[cur].a;
            int bt = g_node[b].ty;
            if (g_ty[bt].kind == TY_REF) {
                if (ck == E_MEMBER && g_node[b].kind == E_IDENT && g_node[b].sid == S_self) { rgn = R_EXTERN; }
                else { rgn = g_ty[bt].n; }
                have = 1;
                break;
            }
            if (g_ty[bt].kind == TY_PTR) { rgn = R_EXTERN; have = 1; break; }
            if (ck == E_MEMBER && (g_node[cur].flags & NF_ARROW) != 0) { rgn = R_EXTERN; have = 1; break; }
            cur = b;
            continue;
        }
        break;
    }
    if (have == 0) {
        if (base_global == 1) { rgn = R_STATIC; } else { rgn = R_STACK; }
    }
    mark_addr(op);
    int arn = 0;
    if (have == 1 && rgn == R_ARENA) {
        int b = g_node[cur].a;
        arn = g_ty[g_node[b].ty].sid;
    }
    int fl = 0;
    if (is_mut == 0) { fl = TYF_CONST; }
    if (is_volatile_lval(op) == 1) { fl = fl | TYF_VOLATILE; }
    return ty_ref(ot, rgn, arn, fl);
}

int check_deref(int e) {
    int t = check_expr(g_node[e].a);
    int k = g_ty[t].kind;
    if (k == TY_PTR) {
        if (in_unsafe() == 0) { err_at(e, "dereference of raw pointer requires 'unsafe' block"); }
        mark_lval(e);
        return g_ty[t].base;
    }
    if (k == TY_REF) {
        check_nullable_deref(t, e);
        if ((g_ty[t].flags & TYF_CONST) == 0) { mark_lval(e); }
        return g_ty[t].base;
    }
    if (is_err(t) == 1) { return TY_ERROR; }
    msg_reset();
    msg_lit("cannot dereference non-pointer type '");
    msg_type(t);
    msg_lit("'");
    error_node(e);
    return TY_ERROR;
}

int check_unary(int e) {
    int op = g_node[e].op;
    int t = check_expr(g_node[e].a);
    if (op == T_MINUS) {
        if (ty_is_arith(t) == 0 && is_err(t) == 0) { err_at(e, "unary '-' requires numeric type"); }
        return t;
    }
    if (op == T_BANG) { return TY_BOOL; }
    if (op == T_TILDE) {
        if (ty_is_int(t) == 0 && is_err(t) == 0) { err_at(e, "bitwise not requires integer type"); }
        return t;
    }
    // ++ / --
    if (ty_is_arith(t) == 0 && g_ty[t].kind != TY_PTR) {
        if (is_err(t) == 0) { err_at(e, "++ / -- requires numeric or pointer type"); }
    } else if (g_ty[t].kind == TY_PTR && in_unsafe() == 0) {
        err_at(e, "raw pointer arithmetic requires 'unsafe' block");
    }
    return t;
}

int check_force(int e) {
    int t = check_expr(g_node[e].a);
    int k = g_ty[t].kind;
    int is_ptr = k == TY_PTR;
    int is_nref = k == TY_REF && (g_ty[t].flags & TYF_NULL) != 0;
    int is_opt = k == TY_OPT;
    if (is_ptr == 0 && is_nref == 0 && is_opt == 0) {
        if (is_err(t) == 1) { return TY_ERROR; }
        msg_reset();
        msg_lit("'!' can only force-unwrap a pointer, nullable reference (?&T), or optional (?T), got '");
        msg_type(t);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (in_unsafe() == 0) {
        err_at(e, "'!' force-unwrap requires 'unsafe', or use match/is_null()/is_none()/.default(value) instead");
    }
    if (is_ptr == 1) { mark_lval(e); return g_ty[t].base; }
    if (is_nref == 1) {
        // '?fn' -> 'fn': the non-null function reference itself
        if (g_ty[g_ty[t].base].kind == TY_FN) { return ty_ref(g_ty[t].base, g_ty[t].n, g_ty[t].sid, g_ty[t].flags & ~TYF_NULL); }
        if ((g_ty[t].flags & TYF_CONST) == 0) { mark_lval(e); }
        return g_ty[t].base;
    }
    return g_ty[t].base;
}

// ── binary ───────────────────────────────────────────────────────────────────
int op_method_name(int op) {
    if (op == T_PLUS) { return intern_lit("operator+"); }
    if (op == T_MINUS) { return intern_lit("operator-"); }
    if (op == T_STAR) { return intern_lit("operator*"); }
    if (op == T_SLASH) { return intern_lit("operator/"); }
    if (op == T_PERCENT) { return intern_lit("operator%"); }
    if (op == T_EQ) { return intern_lit("operator=="); }
    if (op == T_NE) { return intern_lit("operator!="); }
    if (op == T_LT) { return intern_lit("operator<"); }
    if (op == T_GT) { return intern_lit("operator>"); }
    if (op == T_LE) { return intern_lit("operator<="); }
    if (op == T_GE) { return intern_lit("operator>="); }
    return 0;
}

int is_arith_op(int op) {
    return op == T_PLUS || op == T_MINUS || op == T_STAR || op == T_SLASH || op == T_PERCENT ||
           op == T_PLUSPIPE || op == T_MINUSPIPE || op == T_STARPIPE || op == T_PLUSPCT ||
           op == T_MINUSPCT || op == T_STARPCT;
}

int check_binary(int e) {
    int op = g_node[e].op;
    int lt = check_expr(g_node[e].a);
    int rt = check_expr(g_node[e].b);
    if (is_err(lt) == 1 || is_err(rt) == 1) { return TY_ERROR; }
    int sd = struct_of(lt);
    if (sd != 0) {
        int mn = op_method_name(op);
        if (mn != 0) {
            int f = instance_method(sd, mn);
            if (f != 0) {
                g_node[e].c = 1;
                g_node[e].d = f;
                g_func[f].used = 1;
                queue_func(f);
                int r = func_ret(f);
                if (r == 0) { r = TY_BOOL; }
                return r;
            }
        }
    }
    if (is_arith_op(op) == 1) {
        if (ty_is_arith(lt) == 0 || ty_is_arith(rt) == 0) {
            int lk = g_ty[lt].kind;
            int rk = g_ty[rt].kind;
            if (lk == TY_PTR || rk == TY_PTR) {
                if (in_unsafe() == 0) { err_at(e, "raw pointer arithmetic requires 'unsafe' block"); }
                if (op == T_MINUS && lk == TY_PTR && rk == TY_PTR) { return TY_I64; }
                if (lk == TY_PTR) { return lt; }
                return rt;
            }
            if (lk == TY_ARRAY && ty_is_int(rt) == 1) {
                if (in_unsafe() == 0) { err_at(e, "array pointer arithmetic requires 'unsafe' block"); }
                return ty_ptr(g_ty[lt].base, 0);
            }
            if (rk == TY_ARRAY && ty_is_int(lt) == 1) {
                if (in_unsafe() == 0) { err_at(e, "array pointer arithmetic requires 'unsafe' block"); }
                return ty_ptr(g_ty[rt].base, 0);
            }
            err_at(e, "arithmetic requires numeric types");
            return TY_ERROR;
        }
        int lc = is8(lt);
        int rc = is8(rt);
        if (lc == 1 && rc == 1) { return TY_I32; }
        if (lc == 1 && ty_is_int(rt) == 1) { return rt; }
        if (rc == 1 && ty_is_int(lt) == 1) { return lt; }
        if (ty_equal(lt, rt) == 0) {
            msg_reset();
            msg_lit("implicit type conversion not allowed in SafeC: '");
            msg_type(lt);
            msg_lit("' and '");
            msg_type(rt);
            msg_lit("' differ");
            error_node(e);
            return TY_ERROR;
        }
        return lt;
    }
    if (op == T_AMP || op == T_PIPE || op == T_CARET || op == T_SHL || op == T_SHR) {
        if (ty_is_int(lt) == 0 || ty_is_int(rt) == 0) {
            err_at(e, "bitwise operations require integer types");
            return TY_ERROR;
        }
        return lt;
    }
    return TY_BOOL;   // comparisons and logic
}

// ── assignment ───────────────────────────────────────────────────────────────
int base_var_sym(int e) {
    // the symbol (scope index) of the variable at the root of a '.'/'->'
    // chain ('->' is judged like '.' through a reference: by the variable
    // holding the pointer); -2 when the root is a call ('f().p': the storage
    // a returned reference designates outlives every scope of this function)
    int cur = e;
    while (cur != 0) {
        if (g_node[cur].kind == E_IDENT) { return lookup_local(g_node[cur].sid); }
        if (g_node[cur].kind == E_MEMBER) { cur = g_node[cur].a; continue; }
        if (g_node[cur].kind == E_CALL && cur != e) { return -2; }
        return -1;
    }
    return -1;
}

int ident_is_const(int e) {
    if (g_node[e].kind != E_IDENT) { return 0; }
    int c = g_node[e].c;
    if (c == SYM_LOCAL) { return (g_local[g_node[e].d].flags & LF_CONST) != 0; }
    if (c == SYM_GLOBAL) { return (g_glob[g_node[e].d].flags & NF_CONST) != 0; }
    return 0;
}

int check_assign(int e) {
    int op = g_node[e].op;
    int lhs = g_node[e].a;
    int rhs = g_node[e].b;
    if (op == T_ASSIGN && g_node[lhs].kind == E_IDENT) {
        int s = lookup_local(g_node[lhs].sid);
        if (s >= 0 && g_sym[s].kind == SYM_LOCAL) { g_sym[s].init = 1; }
    }
    int lt = check_expr(lhs);
    int saved = g_expect;
    if (op == T_ASSIGN && is_err(lt) == 0) { g_expect = lt; }
    if (op == T_ASSIGN) { adopt_tuple(rhs, lt); }
    int rt = 0;
    if (g_node[rhs].kind == E_INITLIST && op == T_ASSIGN) { rt = check_initlist(rhs, lt); }
    else { rt = check_expr(rhs); }
    g_expect = saved;
    rt = coerce_generic_value(rhs, rt, lt);
    if (is_lval(lhs) == 0 && g_node[lhs].kind == E_INDEX && g_ty[g_node[g_node[lhs].a].ty].kind == TY_SLICE) {
        msg_reset();
        msg_lit("cannot assign through a read-only slice '");
        msg_type(g_node[g_node[lhs].a].ty);
        msg_lit("'");
        error_node(e);
    } else if (is_lval(lhs) == 0) { err_at(e, "left side of assignment must be lvalue"); }
    if (ident_is_const(lhs) == 1) {
        msg_reset();
        msg_lit("cannot assign to const variable '");
        msg_str(g_node[lhs].sid);
        msg_lit("'");
        error_node(e);
    }
    if (g_ty[rt].kind == TY_REF) {
        int bs = base_var_sym(lhs);
        int r = g_ty[rt].n;
        if (bs == -2 || (bs >= 0 && (g_sym[bs].kind == SYM_LOCAL))) {
            int vdepth = bs == -2 ? -1 : g_local[g_sym[bs].idx].depth;
            if (r == R_STACK && vdepth < g_depth) {
                msg_reset();
                msg_lit("cannot assign '&stack ");
                msg_type(g_ty[rt].base);
                msg_lit("' to variable in outer scope: reference would escape");
                error_node(e);
            }
        }
        // a stack reference stored into static storage
        if (r == R_STACK && g_node[lhs].kind == E_IDENT && g_node[lhs].c == SYM_GLOBAL && in_unsafe() == 0) {
            msg_reset();
            msg_lit("cannot assign '&stack ");
            msg_type(g_ty[rt].base);
            msg_lit("' to variable in outer scope: reference would escape");
            error_node(e);
        }
    }
    int ptr_step = g_ty[lt].kind == TY_PTR && ty_is_int(rt) == 1 && (op == T_PLUSEQ || op == T_MINUSEQ);
    if (ptr_step == 1 && in_unsafe() == 0) { err_at(e, "raw pointer arithmetic requires 'unsafe' block"); }
    if (is_err(lt) == 0 && is_err(rt) == 0 && ptr_step == 0) {
        if (convertible_to(rhs, rt, lt) == 0) {
            msg_reset();
            msg_lit("type mismatch in assignment: cannot convert '");
            msg_type(rt);
            msg_lit("' to '");
            msg_type(lt);
            msg_lit("'");
            msg_raw_ref_hint(rt, lt);
            error_node(e);
        }
    }
    if (op == T_ASSIGN && g_node[lhs].kind == E_IDENT) { release_borrows(g_node[lhs].sid); }
    if (in_unsafe() == 0 && op == T_ASSIGN && g_node[lhs].kind == E_IDENT && g_node[rhs].kind == E_ADDR &&
        g_ty[lt].kind == TY_REF) {
        int ro = g_node[rhs].a;
        if (g_node[ro].kind == E_IDENT && (g_node[ro].c == SYM_LOCAL || g_node[ro].c == SYM_GLOBAL) && g_ty[lt].n != R_STATIC) {
            int is_mut = (g_ty[lt].flags & TYF_CONST) == 0;
            track_ref(g_node[ro].sid, is_mut, g_node[lhs].sid, e);
        }
    }
    if (op == T_ASSIGN && g_node[lhs].kind == E_IDENT) {
        int s = lookup_local(g_node[lhs].sid);
        if (s >= 0) { g_sym[s].init = 1; }
    }
    return lt;
}

// '(a, b)' where a tuple is expected becomes a tuple literal
void adopt_tuple(int e, int expected) {
    if (expected == 0 || g_ty[expected].kind != TY_TUPLE) { return; }
    if (g_node[e].kind != E_COMMA || (g_node[e].flags & NF_PAREN) == 0) { return; }
    int parts[64];
    int n = 0;
    int cur = e;
    // left-nested chain: ((a, b), c)
    while (g_node[cur].kind == E_COMMA && (cur == e || (g_node[cur].flags & NF_PAREN) == 0) && n < 63) {
        parts[n] = g_node[cur].b;
        n = n + 1;
        cur = g_node[cur].a;
    }
    parts[n] = cur;
    n = n + 1;
    if (n != g_ty[expected].nlist) { return; }
    g_node[e].kind = E_TUPLE;
    int mark = list_begin();
    int i = n - 1;
    while (i >= 0) { list_push(parts[i]); i = i - 1; }
    list_end(mark, e);
    g_node[e].c = expected;
}

int check_tuple(int e) {
    int target = g_node[e].c;
    int i = 0;
    int ets[64];
    int n = g_node[e].nlist;
    while (i < n && i < 64) {
        int el = g_list[g_node[e].list + i];
        int want = 0;
        if (target != 0) { want = g_tyl[g_ty[target].list + i]; adopt_tuple(el, want); }
        ets[i] = check_expr(el);
        if (target != 0 && is_err(ets[i]) == 0 && convertible_to(el, ets[i], want) == 0) {
            msg_reset();
            msg_lit("tuple element ");
            msg_int((long)(i + 1));
            msg_lit(": cannot convert '");
            msg_type(ets[i]);
            msg_lit("' to '");
            msg_type(want);
            msg_lit("'");
            error_node(el);
        }
        i = i + 1;
    }
    if (target != 0) { return target; }
    int mark = tyl_begin();
    i = 0;
    while (i < n && i < 64) { tyl_push(ets[i]); i = i + 1; }
    return ty_tuple(mark);
}

// ── initializer lists ────────────────────────────────────────────────────────
int check_initlist(int e, int target) {
    int n = g_node[e].nlist;
    int k = g_ty[target].kind;
    if (k == TY_STRUCT && g_sdef[g_ty[target].n].kind == SD_STRUCT) {
        int sd = g_ty[target].n;
        int next = 0;
        int i = 0;
        while (i < n) {
            int el = g_list[g_node[e].list + i];
            int val = el;
            int slot = next;
            if (g_node[el].kind == E_DESIG) {
                val = g_node[el].a;
                if (g_node[el].flags == 0) {
                    int f = find_field(sd, g_node[el].sid);
                    if (f < 0) {
                        msg_reset();
                        msg_lit("struct '");
                        msg_str(g_sdef[sd].sid);
                        msg_lit("' has no field '");
                        msg_str(g_node[el].sid);
                        msg_lit("'");
                        error_node(val);
                        slot = 0;
                    } else { slot = f - g_sdef[sd].field0; }
                } else {
                    msg_reset();
                    msg_lit("array designator '[");
                    msg_int(g_node[el].ival);
                    msg_lit("]' used initializing struct '");
                    msg_str(g_sdef[sd].sid);
                    msg_lit("' (expected '.field =')");
                    error_node(val);
                }
                g_node[el].d = slot;
            }
            next = slot + 1;
            if (slot >= g_sdef[sd].nfield) {
                check_expr(val);
                if (g_node[el].kind != E_DESIG) {
                    msg_reset();
                    msg_lit("too many initializers for struct '");
                    msg_str(g_sdef[sd].sid);
                    msg_lit("'");
                    error_node(e);
                }
                i = i + 1;
                continue;
            }
            int fi = g_sdef[sd].field0 + slot;
            int ft = g_field[fi].ty;
            int vt = 0;
            adopt_tuple(val, ft);
            vt = check_expr(val);
            vt = coerce_generic_value(val, vt, ft);
            if (is_err(vt) == 0 && convertible_to(val, vt, ft) == 0) {
                msg_reset();
                msg_lit("type mismatch initializing field '");
                msg_str(g_field[fi].sid);
                msg_lit("': cannot convert '");
                msg_type(vt);
                msg_lit("' to '");
                msg_type(ft);
                msg_lit("'");
                error_node(val);
            }
            if (g_node[el].kind != E_DESIG) {
                // remember the slot on a wrapper-less element
                g_node[val].flags = g_node[val].flags;
            }
            i = i + 1;
        }
        return set_ty(e, target);
    }
    if (k == TY_ARRAY) {
        int et = g_ty[target].base;
        int next = 0;
        int maxs = -1;
        int i = 0;
        while (i < n) {
            int el = g_list[g_node[e].list + i];
            int val = el;
            int slot = next;
            if (g_node[el].kind == E_DESIG) {
                val = g_node[el].a;
                if (g_node[el].flags == 1) { slot = (int)g_node[el].ival; }
                else {
                    msg_reset();
                    msg_lit("field designator '.");
                    msg_str(g_node[el].sid);
                    msg_lit("' used initializing an array (expected '[index] =')");
                    error_node(val);
                }
                g_node[el].d = slot;
            }
            next = slot + 1;
            if (slot > maxs) { maxs = slot; }
            int vt = check_expr(val);
            if (is_err(vt) == 0 && convertible_to(val, vt, et) == 0) {
                msg_reset();
                msg_lit("type mismatch in array initializer: cannot convert '");
                msg_type(vt);
                msg_lit("' to '");
                msg_type(et);
                msg_lit("'");
                error_node(val);
            }
            i = i + 1;
        }
        if (g_ty[target].n > 0 && maxs >= g_ty[target].n) {
            msg_reset();
            msg_lit("too many initializers for array of size ");
            msg_int((long)g_ty[target].n);
            error_node(e);
        }
        g_node[e].ival = (long)(maxs + 1);
        return set_ty(e, target);
    }
    int i = 0;
    while (i < n) {
        int el = g_list[g_node[e].list + i];
        if (g_node[el].kind == E_DESIG) { check_expr(g_node[el].a); } else { check_expr(el); }
        i = i + 1;
    }
    return set_ty(e, TY_VOID);
}

// ── member / index / slice ───────────────────────────────────────────────────
int check_member(int e) {
    int b = g_node[e].a;
    int bt = check_expr(b);
    int arrow = (g_node[e].flags & NF_ARROW) != 0;
    if (arrow == 0 && g_ty[bt].kind == TY_REF) {
        check_nullable_deref(bt, e);
        if ((g_ty[bt].flags & TYF_CONST) == 0) { mark_lval(e); }
        bt = g_ty[bt].base;
    }
    if (arrow == 1) {
        if (g_ty[bt].kind == TY_PTR) {
            if (in_unsafe() == 0) { err_at(e, "pointer member access requires 'unsafe' block"); }
            bt = g_ty[bt].base;
            mark_lval(e);
        } else if (g_ty[bt].kind == TY_REF) {
            check_nullable_deref(bt, e);
            if ((g_ty[bt].flags & TYF_CONST) == 0) { mark_lval(e); }
            bt = g_ty[bt].base;
        } else {
            if (is_err(bt) == 1) { return TY_ERROR; }
            msg_reset();
            msg_lit("'->' requires pointer or reference, got '");
            msg_type(bt);
            msg_lit("'");
            error_node(e);
            return TY_ERROR;
        }
    }
    int fsid = g_node[e].sid;
    if (g_ty[bt].kind == TY_TUPLE) {
        if (g_node[e].op == 1 && g_node[e].ival >= 0L && g_node[e].ival < (long)g_ty[bt].nlist) {
            if (arrow == 0 && is_lval(b) == 1) { mark_lval(e); }
            g_node[e].d = MK_TUPLE;
            g_node[e].c = (int)g_node[e].ival;
            return g_tyl[g_ty[bt].list + (int)g_node[e].ival];
        }
        msg_reset();
        msg_lit("invalid tuple field '");
        msg_str(fsid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (g_ty[bt].kind == TY_SLICE) {
        if (fsid == S_len) { g_node[e].d = MK_SLEN; return TY_U64; }
        if (fsid == S_ptr) { g_node[e].d = MK_SPTR; return ty_ptr(g_ty[bt].base, g_ty[bt].flags & TYF_CONST); }
        msg_reset();
        msg_lit("slice has no member '");
        msg_str(fsid);
        msg_lit("' (use .len or .ptr)");
        error_node(e);
        return TY_ERROR;
    }
    int sd = struct_of(bt);
    if (sd == 0) {
        if (is_err(bt) == 1) { return TY_ERROR; }
        msg_reset();
        msg_lit("member access on non-struct type '");
        msg_type(bt);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    layout_struct(sd);
    int f = find_field(sd, fsid);
    if (f < 0) {
        msg_reset();
        msg_lit("struct '");
        msg_str(g_sdef[sd].sid);
        msg_lit("' has no field '");
        msg_str(fsid);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (arrow == 0 && g_ty[g_node[b].ty].kind != TY_REF && is_lval(b) == 1) { mark_lval(e); }
    g_node[e].d = MK_FIELD;
    g_node[e].c = f;
    return g_field[f].ty;
}

int check_index(int e) {
    int bt = check_expr(g_node[e].a);
    int it = check_expr(g_node[e].b);
    if (ty_is_int(it) == 0 && is_err(it) == 0) { err_at(g_node[e].b, "array index must be integer"); }
    if (in_unsafe() == 1) { g_node[e].flags = g_node[e].flags | NF_INUNSAFE; }
    int k = g_ty[bt].kind;
    if (k == TY_ARRAY) {
        int idx = g_node[e].b;
        if (in_unsafe() == 0 && g_ty[bt].n > 0 && g_node[idx].kind == E_INT) {
            long v = g_node[idx].ival;
            if (v < 0L || v >= (long)g_ty[bt].n) {
                msg_reset();
                msg_lit("array index ");
                msg_int(v);
                msg_lit(" out of bounds for array of size ");
                msg_int((long)g_ty[bt].n);
                error_node(e);
            }
        }
        mark_lval(e);
        return g_ty[bt].base;
    }
    if (k == TY_PTR) {
        if (in_unsafe() == 0) { err_at(e, "pointer subscript requires 'unsafe' block"); }
        mark_lval(e);
        return g_ty[bt].base;
    }
    if (k == TY_SLICE) {
        // an element of a '[]const T' isn't assignable
        if ((g_ty[bt].flags & TYF_CONST) == 0) { mark_lval(e); }
        return g_ty[bt].base;
    }
    if (k == TY_REF) {
        int rb = g_ty[bt].base;
        int mut = (g_ty[bt].flags & TYF_CONST) == 0;
        if (g_ty[rb].kind == TY_ARRAY) {
            if (mut == 1) { mark_lval(e); }
            return g_ty[rb].base;
        }
        if (in_unsafe() == 0) { err_at(e, "reference subscript requires 'unsafe' block"); }
        if (mut == 1) { mark_lval(e); }
        return rb;
    }
    if (is_err(bt) == 1) { return TY_ERROR; }
    msg_reset();
    msg_lit("subscript on non-array type '");
    msg_type(bt);
    msg_lit("'");
    error_node(e);
    return TY_ERROR;
}

int check_slice(int e) {
    int bt = check_expr(g_node[e].a);
    if (g_node[e].b != 0) {
        int st = check_expr(g_node[e].b);
        if (ty_is_int(st) == 0) { err_at(g_node[e].b, "slice start index must be integer"); }
    }
    if (g_node[e].c != 0) {
        int et = check_expr(g_node[e].c);
        if (ty_is_int(et) == 0) { err_at(g_node[e].c, "slice end index must be integer"); }
    }
    if (in_unsafe() == 1) { g_node[e].flags = g_node[e].flags | NF_INUNSAFE; }
    int k = g_ty[bt].kind;
    int el = 0;
    int cfl = 0;
    if (k == TY_ARRAY) { el = g_ty[bt].base; mark_addr(g_node[e].a); }
    else if (k == TY_PTR) {
        if (in_unsafe() == 0) { err_at(e, "slicing a pointer requires 'unsafe' block"); }
        el = g_ty[bt].base;
        cfl = g_ty[bt].flags & TYF_CONST;
    } else if (k == TY_SLICE) { el = g_ty[bt].base; cfl = g_ty[bt].flags & TYF_CONST; }
    else if (k == TY_REF && g_ty[g_ty[bt].base].kind == TY_ARRAY) {
        el = g_ty[g_ty[bt].base].base;
        cfl = g_ty[bt].flags & TYF_CONST;
    }
    else {
        if (is_err(bt) == 1) { return TY_ERROR; }
        msg_reset();
        msg_lit("cannot slice type '");
        msg_type(bt);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    return ty_slice_f(el, cfl);
}

// ── cast ─────────────────────────────────────────────────────────────────────
int check_cast(int e) {
    int t = resolve_type(g_node[e].c);
    g_node[e].c = t;
    int op = g_node[e].a;
    int st = 0;
    if (g_node[op].kind == E_INITLIST) { st = check_initlist(op, t); }
    else { st = check_expr(op); }
    // a slice is a (pointer, length) pair, not a pointer
    if (g_ty[st].kind == TY_SLICE && g_ty[t].kind != TY_SLICE && is_err(t) == 0) {
        msg_reset();
        msg_lit("cannot cast slice '");
        msg_type(st);
        msg_lit("' to '");
        msg_type(t);
        msg_lit("': use its '.ptr' (and '.len'), or slice it");
        error_node(e);
        return t;
    }
    if (g_ty[st].kind == TY_REF && (g_ty[st].flags & TYF_NULL) != 0 && g_ty[t].kind == TY_REF && (g_ty[t].flags & TYF_NULL) == 0) {
        msg_reset();
        msg_lit("cannot cast nullable reference '");
        msg_type(st);
        msg_lit("' to non-null '");
        msg_type(t);
        msg_lit("': check it with 'match (x) { case null: ... case some(r): ... }' (r is the non-null reference)");
        error_node(e);
    }
    if (in_unsafe() == 0 && is_err(st) == 0) {
        if (g_ty[st].kind == TY_PTR && g_ty[t].kind == TY_REF && (g_ty[t].flags & TYF_NULL) == 0) {
            msg_reset();
            msg_lit("cast from raw pointer to non-null reference '");
            msg_type(t);
            msg_lit("' requires 'unsafe' block; use '(?");
            msg_type(t);
            msg_lit(")' and check it with 'match'");
            error_node(e);
        }
        if (g_ty[t].kind == TY_PTR) {
            int is_null = g_node[op].kind == E_NULL || (g_node[op].kind == E_INT && g_node[op].ival == 0L);
            if (ty_is_int(st) == 1 && is_null == 0) {
                err_at(e, "cast from integer to raw pointer requires 'unsafe' block");
            } else if (g_ty[st].kind == TY_PTR && is_null == 0) {
                if (is_void(g_ty[t].base) == 0 && ty_equal(g_ty[st].base, g_ty[t].base) == 0) {
                    msg_reset();
                    msg_lit("cast between raw pointer types '");
                    msg_type(st);
                    msg_lit("' and '");
                    msg_type(t);
                    msg_lit("' requires 'unsafe' block");
                    error_node(e);
                }
            }
        }
        int is_fref = g_ty[st].kind == TY_REF && g_ty[g_ty[st].base].kind == TY_FN;
        if (g_ty[st].kind == TY_REF && g_ty[t].kind == TY_PTR && is_fref == 0) {
            err_at(e, "cast from safe reference to raw pointer requires 'unsafe' block");
        }
    }
    if (g_ty[st].kind == TY_ARRAY || g_ty[st].kind == TY_STRUCT) { mark_addr(op); }
    return t;
}

// ── new / arena ──────────────────────────────────────────────────────────────
int check_new(int e) {
    int r = g_node[e].sid;
    if (r != 0 && g_region_glob[r] == 0) {
        msg_reset();
        msg_lit("unknown region '");
        msg_str(r);
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    int t = resolve_type(g_node[e].c);
    g_node[e].c = t;
    if (g_node[e].a != 0) {
        int st = check_expr(g_node[e].a);
        if (is_err(st) == 0 && ty_is_int(st) == 0) {
            msg_reset();
            msg_lit("'new' array size must be an integer, got '");
            msg_type(st);
            msg_lit("'");
            error_node(g_node[e].a);
        }
    }
    if (r == 0) { return ty_ref(t, R_HEAP, 0, 0); }
    return ty_ref(t, R_ARENA, r, 0);
}

int check_arena(int e) {
    int r = g_node[e].sid;
    int k = g_node[e].op;
    if (r == 0 || g_region_glob[r] == 0) {
        msg_reset();
        msg_lit("use of undeclared identifier '__arena_");
        if (k == K_ARENA_RESET) { msg_lit("reset_"); }
        else if (k == K_ARENA_DESTROY) { msg_lit("destroy_"); }
        else if (k == K_ARENA_MARK) { msg_lit("mark_"); }
        else { msg_lit("free_to_"); }
        if (r != 0) { msg_str(r); }
        msg_lit("'");
        error_node(e);
        return TY_ERROR;
    }
    if (k == K_ARENA_FREE_TO) {
        int a = g_node[e].a;
        int at = check_expr(a);
        if (is_err(at) == 0 && convertible_to(a, at, TY_U64) == 0) {
            msg_reset();
            msg_lit("argument 1: cannot convert '");
            msg_type(at);
            msg_lit("' to 'unsigned long'");
            error_node(a);
        }
    }
    if (k == K_ARENA_MARK) { return TY_U64; }
    return TY_VOID;
}

// ── match / try ──────────────────────────────────────────────────────────────
// classification of a match subject
int g_mt_nulllike = 0;
int g_mt_opt = 0;
int g_mt_payload = 0;
int g_mt_union = 0;   // sdef of a tagged union subject
int g_mt_enum = 0;    // sdef of an enum subject

void classify_subject(int t) {
    g_mt_nulllike = 0;
    g_mt_opt = 0;
    g_mt_payload = 0;
    g_mt_union = 0;
    g_mt_enum = 0;
    int k = g_ty[t].kind;
    if (k == TY_PTR) { g_mt_nulllike = 1; g_mt_payload = g_ty[t].base; }
    else if (k == TY_REF && (g_ty[t].flags & TYF_NULL) != 0) {
        g_mt_nulllike = 1;
        g_mt_payload = ty_ref(g_ty[t].base, g_ty[t].n, g_ty[t].sid, g_ty[t].flags & ~TYF_NULL);
    } else if (k == TY_OPT) { g_mt_opt = 1; g_mt_payload = g_ty[t].base; }
    else if (k == TY_STRUCT && g_sdef[g_ty[t].n].kind == SD_UNION) { g_mt_union = g_ty[t].n; layout_struct(g_mt_union); }
    else if (k == TY_ENUM) { g_mt_enum = g_ty[t].n; }
}

int g_mt_wild = 0;
int g_mt_cov[256];
int g_mt_ncov = 0;

void cover(int tag) {
    int i = 0;
    while (i < g_mt_ncov) { if (g_mt_cov[i] == tag) { return; } i = i + 1; }
    if (g_mt_ncov < 256) { g_mt_cov[g_mt_ncov] = tag; g_mt_ncov = g_mt_ncov + 1; }
}

// resolves one arm's patterns: P_NAME → P_NULL/P_NONE/P_SOME/P_VALUE(enum)/P_VARIANT
// binds payload names in the current scope. 'loc' is the match node.
void resolve_patterns(int arm, int loc) {
    int i = 0;
    while (i < g_node[arm].nlist) {
        int p = g_list[g_node[arm].list + i];
        int k = g_node[p].kind;
        if (k == P_WILD) { g_mt_wild = 1; }
        if ((k == P_NAME || k == P_VARIANT) && g_mt_enum != 0) {
            int f = find_field(g_mt_enum, g_node[p].sid);
            if (f >= 0) {
                g_node[p].kind = P_VALUE;
                g_node[p].ival = g_field[f].val;
                g_node[p].a = 0;
                cover((int)g_field[f].val);
            } else {
                msg_reset();
                msg_lit("unknown enumerator '");
                msg_str(g_node[p].sid);
                msg_lit("' in enum '");
                msg_str(g_sdef[g_mt_enum].sid);
                msg_lit("'");
                error_node(loc);
            }
        } else if ((k == P_NAME || k == P_VARIANT) && g_mt_union != 0) {
            int f = find_field(g_mt_union, g_node[p].sid);
            if (f >= 0) {
                g_node[p].kind = P_VARIANT;
                g_node[p].c = f;
                cover(f - g_sdef[g_mt_union].field0);
                if (g_node[p].d != 0) {
                    int l = new_local(g_node[p].d, g_field[f].ty, LF_INIT, 0);
                    g_node[p].b = l;
                    define_sym(g_node[p].d, SYM_LOCAL, l, g_field[f].ty, 1);
                }
            } else {
                msg_reset();
                msg_lit("unknown variant '");
                msg_str(g_node[p].sid);
                msg_lit("' in union '");
                msg_str(g_sdef[g_mt_union].sid);
                msg_lit("'");
                error_node(loc);
            }
        } else if (k == P_NAME && (g_mt_nulllike == 1 || g_mt_opt == 1)) {
            int empty = intern_lit("null");
            if (g_mt_opt == 1) { empty = S_none; }
            int sid = g_node[p].sid;
            if (sid == empty) {
                g_node[p].kind = P_NULL;
                cover(0);
                if (g_node[p].d != 0) {
                    msg_reset();
                    msg_lit("'");
                    msg_str(empty);
                    msg_lit("' does not carry a value to bind");
                    error_node(loc);
                }
            } else if (sid == S_some) {
                g_node[p].kind = P_SOME;
                cover(1);
                if (g_node[p].d != 0) {
                    int kind = SYM_LOCAL;
                    int l = new_local(g_node[p].d, g_mt_payload, LF_INIT, 0);
                    g_node[p].b = l;
                    define_sym(g_node[p].d, kind, l, g_mt_payload, 1);
                }
            } else {
                msg_reset();
                msg_lit("unknown pattern '");
                msg_str(sid);
                msg_lit("' — expected '");
                msg_str(empty);
                msg_lit("' or 'some(x)'");
                error_node(loc);
            }
        } else if (k == P_NAME) {
            // an enumerator constant compared against an integer subject
            int sid = g_node[p].sid;
            if (g_gkind[sid] == SYM_ENUMC) {
                g_node[p].kind = P_VALUE;
                g_node[p].ival = g_field[g_gidx[sid]].val;
                g_node[p].a = 0;
            }
            // anything else matches nothing (as in the C++ compiler's codegen)
        } else if (k == P_VALUE) {
            g_node[p].ival = g_node[g_node[p].a].ival;
        }
        i = i + 1;
    }
}

int check_match_expr(int e) {
    int st = check_expr(g_node[e].a);
    classify_subject(st);
    if (g_node[e].nlist == 0) { err_at(e, "match expression must have at least one arm"); return TY_ERROR; }
    int nulllike = g_mt_nulllike;
    int opt = g_mt_opt;
    int un = g_mt_union;
    int en = g_mt_enum;
    int payload = g_mt_payload;
    int saved_wild = g_mt_wild;
    int saved_ncov = g_mt_ncov;
    g_mt_wild = 0;
    g_mt_ncov = 0;
    int result = 0;
    int saw_error = 0;
    int i = 0;
    while (i < g_node[e].nlist) {
        int arm = g_list[g_node[e].list + i];
        g_mt_nulllike = nulllike; g_mt_opt = opt; g_mt_union = un; g_mt_enum = en; g_mt_payload = payload;
        push_scope(0);
        resolve_patterns(arm, e);
        int v = g_node[arm].a;
        int at = TY_ERROR;
        if (v != 0) {
            if (result != 0) { adopt_tuple(v, result); }
            at = check_expr(v);
        }
        pop_scope();
        if (is_err(at) == 1) { saw_error = 1; }
        else if (result == 0) { result = at; }
        else if (ty_equal(result, at) == 0) {
            msg_reset();
            msg_lit("match arms must have the same type: '");
            msg_type(result);
            msg_lit("' vs '");
            msg_type(at);
            msg_lit("'");
            error_node(v);
            saw_error = 1;
        }
        i = i + 1;
    }
    int exhaustive = g_mt_wild;
    if (un != 0 && g_mt_ncov == g_sdef[un].nfield) { exhaustive = 1; }
    if ((nulllike == 1 || opt == 1) && g_mt_ncov == 2) { exhaustive = 1; }
    if (en != 0 && g_mt_ncov == g_sdef[en].nfield) { exhaustive = 1; }
    if (exhaustive == 0) {
        msg_reset();
        msg_lit("match expression is not exhaustive: add a 'default' arm, or cover every case (every tagged-union variant; or both '");
        if (opt == 1) { msg_lit("none"); } else { msg_lit("null"); }
        msg_lit("' and 'some(x)')");
        error_node(e);
    }
    g_node[e].c = st;
    g_mt_wild = saved_wild;
    g_mt_ncov = saved_ncov;
    if (saw_error == 1 || result == 0) { return TY_ERROR; }
    return result;
}

int check_try(int e) {
    int t = check_expr(g_node[e].a);
    if (g_ty[t].kind != TY_OPT) {
        if (is_err(t) == 0) {
            msg_reset();
            msg_lit("try: operand must be an optional type '?T', got '");
            msg_type(t);
            msg_lit("'");
            error_node(e);
        }
        return TY_ERROR;
    }
    return g_ty[t].base;
}

// ── spawn / fn_eval ──────────────────────────────────────────────────────────
int check_spawn(int e, int fnx, int arg, int scoped) {
    int ft = check_expr(fnx);
    if (is_err(ft) == 0) {
        int valid = 0;
        if (g_ty[ft].kind == TY_REF && g_ty[ft].n == R_STATIC && g_ty[g_ty[ft].base].kind == TY_FN) { valid = 1; }
        if (valid == 0) {
            msg_reset();
            msg_lit("spawn requires &static function reference, got '");
            msg_type(ft);
            msg_lit("'");
            error_node(fnx);
        }
    }
    int at = check_expr(arg);
    // safec's rules on what a thread may receive, without its Send/Sync
    // analysis: spawn's (detached-capable) threads only static data (or a
    // region-less reference), spawn_scoped's anything (it is joined before
    // the function returns); a raw pointer only in 'unsafe'
    const char* what = scoped == 1 ? "spawn_scoped" : "spawn";
    if (g_ty[at].kind == TY_PTR && in_unsafe() == 0 && g_node[arg].kind != E_NULL) {
        msg_reset(); msg_lit(what);
        msg_lit(": a raw pointer argument can't be checked for thread safety; pass a reference, or write the call inside 'unsafe { }'");
        error_node(arg);
    }
    if (scoped == 0 && g_ty[at].kind == TY_REF && g_ty[at].n != R_STATIC && g_ty[at].n != R_EXTERN &&
        g_ty[g_ty[at].base].kind != TY_FN) {
        msg_reset(); msg_lit("spawn: the argument '"); msg_type(at);
        msg_lit("' refers to data the thread may outlive (a spawned thread isn't joined by its scope): pass static data, or use spawn_scoped and join it before the data goes away");
        error_node(arg);
    }
    return TY_I64;
}

int check_fn_eval(int e) {
    int ot = check_expr(g_node[e].a);
    int fx = g_node[e].b;
    if (g_node[fx].kind != E_IDENT) { err_at(e, "fn_eval: second argument must directly name a declared function"); return TY_ERROR; }
    check_expr(fx);
    if (g_node[fx].c != SYM_FUNC) {
        msg_reset(); msg_lit("fn_eval: '"); msg_str(g_node[fx].sid); msg_lit("' does not name a function"); error_node(e);
        return TY_ERROR;
    }
    int shape = g_node[fx].d;
    if (g_func[shape].owner != 0) {
        msg_reset(); msg_lit("fn_eval: '"); msg_str(g_node[fx].sid);
        msg_lit("' must be a plain function, not a method — it's used only as a name+signature key, never called");
        error_node(e);
        return TY_ERROR;
    }
    int st = ot;
    if (g_ty[st].kind == TY_PTR || g_ty[st].kind == TY_REF) { st = g_ty[st].base; }
    int sd = struct_of(st);
    if (sd == 0) {
        msg_reset(); msg_lit("fn_eval: first argument must be a struct (or a pointer/reference to one), got '");
        msg_type(ot); msg_lit("'"); error_node(e);
        return TY_ERROR;
    }
    int name = g_func[shape].origsid;
    int m = instance_method(sd, name);
    if (m == 0) {
        msg_reset(); msg_lit("fn_eval: type '"); msg_str(g_sdef[sd].sid); msg_lit("' has no method named '");
        msg_str(name); msg_lit("'"); error_node(e);
        return TY_ERROR;
    }
    int mt = g_func[m].ty;
    int sht = g_func[shape].ty;
    int ma = g_ty[mt].nlist - 1;
    if (ma != g_ty[sht].nlist) {
        msg_reset(); msg_lit("fn_eval: '"); msg_str(g_sdef[sd].sid); msg_lit("::"); msg_str(name);
        msg_lit("' takes "); msg_int((long)ma); msg_lit(" parameter(s), but '"); msg_str(g_node[fx].sid);
        msg_lit("' declares "); msg_int((long)g_ty[sht].nlist); error_node(e);
        return TY_ERROR;
    }
    if (ty_equal(g_ty[mt].base, g_ty[sht].base) == 0) {
        msg_reset(); msg_lit("fn_eval: '"); msg_str(g_sdef[sd].sid); msg_lit("::"); msg_str(name);
        msg_lit("' returns '"); msg_type(g_ty[mt].base); msg_lit("', but '"); msg_str(g_node[fx].sid);
        msg_lit("' declares '"); msg_type(g_ty[sht].base); msg_lit("'"); error_node(e);
        return TY_ERROR;
    }
    g_node[e].d = m;
    g_func[m].used = 1;
    queue_func(m);
    return func_value_type(m);
}

// ── dispatch ─────────────────────────────────────────────────────────────────
int check_expr(int e) {
    if (e == 0) { return TY_ERROR; }
    int k = g_node[e].kind;
    int t = TY_ERROR;
    if (in_unsafe() == 1) { g_node[e].flags = g_node[e].flags | NF_INUNSAFE; }
    if (k == E_INT) {
        int fl = g_node[e].flags;
        long v = g_node[e].ival;
        if ((fl & TF_LONG) != 0 && (fl & TF_UNS) != 0) { t = TY_U64; }
        else if ((fl & TF_LONG) != 0) { t = TY_I64; }
        else if ((fl & TF_UNS) != 0) {
            if (v >= 0L && v <= 4294967295L) { t = TY_U32; } else { t = TY_U64; }
        } else if (v >= -2147483648L && v <= 2147483647L) { t = TY_I32; }
        else { t = TY_I64; }
    }
    else if (k == E_FLOAT) { if ((g_node[e].flags & TF_F32) != 0) { t = TY_F32; } else { t = TY_F64; } }
    else if (k == E_CHAR) { t = TY_CHAR; }
    else if (k == E_BOOL) { t = TY_BOOL; }
    else if (k == E_STR) { t = ty_ref(TY_CHAR, R_STATIC, 0, TYF_CONST); }
    else if (k == E_NULL) { t = ty_ptr(TY_VOID, 0); }
    else if (k == E_IDENT) { t = check_ident(e); }
    else if (k == E_UNARY || k == E_POSTFIX) { t = check_unary(e); }
    else if (k == E_ADDR) { t = check_addr(e); }
    else if (k == E_DEREF) { t = check_deref(e); }
    else if (k == E_FORCE) { t = check_force(e); }
    else if (k == E_BINARY) { t = check_binary(e); }
    else if (k == E_ASSIGN) { t = check_assign(e); }
    else if (k == E_TERNARY) {
        check_expr(g_node[e].a);
        int a = check_expr(g_node[e].b);
        int b = check_expr(g_node[e].c);
        if (ty_equal(a, b) == 0) {
            msg_reset();
            msg_lit("ternary branches must have same type: '");
            msg_type(a);
            msg_lit("' vs '");
            msg_type(b);
            msg_lit("'");
            error_node(e);
            t = TY_ERROR;
        } else { t = a; }
    }
    else if (k == E_CALL) { t = check_call(e); }
    else if (k == E_INDEX) { t = check_index(e); }
    else if (k == E_SLICE) { t = check_slice(e); }
    else if (k == E_MEMBER) { t = check_member(e); }
    else if (k == E_CAST) { t = check_cast(e); }
    else if (k == E_SIZEOF_T || k == E_ALIGNOF) {
        int raw = g_node[e].c;
        int st = 0;
        // 'sizeof(x)' where x names a variable parses as a type
        if (k == E_SIZEOF_T && g_ty[raw].kind == TY_NAMED && g_ty[raw].nlist == 0 && g_tyreg[g_ty[raw].sid] == 0 &&
            env_lookup(g_ty[raw].sid) == 0 && (lookup_local(g_ty[raw].sid) >= 0 || g_gkind[g_ty[raw].sid] == SYM_GLOBAL)) {
            int id = new_node(E_IDENT, g_node[e].tok);
            g_node[id].sid = g_ty[raw].sid;
            st = check_expr(id);
        } else {
            st = resolve_type(raw);
        }
        g_node[e].c = st;
        t = TY_U64;
    }
    else if (k == E_SIZEOF_E) { check_expr(g_node[e].a); t = TY_U64; }
    else if (k == E_SIZEOFPACK) { t = TY_U64; }
    else if (k == E_FIELDCOUNT) {
        int ft = resolve_type(g_node[e].c);
        g_node[e].c = ft;
        g_node[e].ival = 0L;
        int sk = g_ty[ft].kind;
        if ((g_node[e].flags & NF_NEEDSDROP) != 0) {
            g_node[e].ival = (long)ty_needs_drop(ft, 0);
            t = TY_I32;
        } else {
            if (sk == TY_STRUCT) { g_node[e].ival = (long)g_sdef[g_ty[ft].n].nfield; }
            else if (is_err(ft) == 0) { err_at(e, "fieldcount() requires a struct type"); }
            t = TY_U64;
        }
    }
    else if (k == E_INITLIST) { t = check_initlist(e, 0); }
    else if (k == E_NEW) { t = check_new(e); }
    else if (k == E_ARENA) { t = check_arena(e); }
    else if (k == E_TUPLE) { t = check_tuple(e); }
    else if (k == E_MATCH) { t = check_match_expr(e); }
    else if (k == E_TRY) { t = check_try(e); }
    else if (k == E_COMMA) {
        check_expr(g_node[e].a);
        t = check_expr(g_node[e].b);
    }
    else if (k == E_DESIG) { t = check_expr(g_node[e].a); }
    else if (k == E_SPAWN) { t = check_spawn(e, g_node[e].a, g_node[e].b, 0); }
    else if (k == E_FNEVAL) { t = check_fn_eval(e); }
    return set_ty(e, t);
}
