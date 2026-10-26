// genx.sc — expression code generation. Integer values narrower than 32
// bits live in i32 sign-extended from their "physical width" (g_pw, which
// mirrors the C++ compiler's LLVM value widths, e.g. char+char computes at
// 8 bits); bools are 0/1. Aggregates are handled by address.
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"
#include "gen.h"
#include "gen_int.h"

int g_pw = 32;

void lget(int l) { emit(OP_LOCAL_GET); emitu((long)l); }
void lset(int l) { emit(OP_LOCAL_SET); emitu((long)l); }
void ltee(int l) { emit(OP_LOCAL_TEE); emitu((long)l); }

// canonicalize an i32 to sign-extended-from-pw
void norm(int pw) {
    if (pw == 8) { emit(OP_I32_EXTEND8_S); }
    else if (pw == 16) { emit(OP_I32_EXTEND16_S); }
    else if (pw == 1) { emit_i32(1); emit(OP_I32_AND); }
}

void mask(int pw) {
    if (pw == 8) { emit_i32(255); emit(OP_I32_AND); }
    else if (pw == 16) { emit_i32(65535); emit(OP_I32_AND); }
}

// integer width change; s: extend signed
void ext_to(int pwf, int pwt, int s) {
    if (pwf == pwt) { return; }
    if (pwf > pwt) {
        if (pwf == 64) { emit(OP_I32_WRAP_I64); }
        if (pwt < 32) { norm(pwt); }
        return;
    }
    if (pwf == 1) {
        if (pwt == 64) { emit(OP_I64_EXTEND_I32_U); }
        return;
    }
    if (s == 0) { mask(pwf); }
    if (pwt == 64) {
        if (s == 1) { emit(OP_I64_EXTEND_I32_S); } else { emit(OP_I64_EXTEND_I32_U); }
    }
}

int is_int_like(int t) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    return (k >= TY_BOOL && k <= TY_U64) || k == TY_PTR || k == TY_REF || (k == TY_OPT && is_ptr_opt(t) == 1);
}

// value conversion for assignments, arguments, returns and casts
void conv(int from, int pw, int to) {
    if (from == to) {
        // same type: restore full width if the value is narrower (char+char)
        if (is_int_like(to) == 1 && pw != 0 && pw != pw_of(to)) {
            int s = 1;
            if (pw == 1 || ty_unsigned(from) == 1) { s = 0; }
            ext_to(pw, pw_of(to), s);
        }
        return;
    }
    int fb = base_scalar(from);
    int tb = base_scalar(to);
    int fk = g_ty[fb].kind;
    int tk = g_ty[tb].kind;
    if (tk == TY_VOID) { return; }
    // optionals
    if (tk == TY_OPT && is_ptr_opt(tb) == 0) {
        if (fk == TY_OPT) { return; }
        int inner = g_ty[tb].base;
        int off = alloc_tmp(tb);
        if (fk == TY_PTR && g_ty[g_ty[fb].base].kind == TY_VOID) {
            emit(OP_DROP);
            emit_frame_addr(off);
            emit_i32(0);
            emit_i32(ty_size(tb));
            emit_fc(11); emit(0);
            emit_frame_addr(off);
            return;
        }
        conv(from, pw, inner);
        int vt = vt_of(inner);
        if (is_agg(inner) == 1) { vt = VT_I32; }
        int tmp = new_wl(vt);
        lset(tmp);
        emit_frame_addr(off);
        lget(tmp);
        emit_store(inner, 0);
        emit_frame_addr(off);
        emit_i32(1);
        emit_mem(OP_I32_STORE8, 0, ty_size(inner));
        emit_frame_addr(off);
        return;
    }
    if (is_agg(tb) == 1 || is_agg(fb) == 1) { return; }   // addresses
    if (tk == TY_F32 || tk == TY_F64) {
        if (fk == TY_F32 || fk == TY_F64) {
            if (fk == TY_F32 && tk == TY_F64) { emit(OP_F64_PROMOTE_F32); }
            if (fk == TY_F64 && tk == TY_F32) { emit(OP_F32_DEMOTE_F64); }
            return;
        }
        // int → float
        int s = 1;
        if (pw == 1 || ty_unsigned(fb) == 1) { s = 0; }
        if (pw < 32 && pw != 1) {
            if (s == 0) { mask(pw); }
        }
        if (pw == 64) {
            if (tk == TY_F64) { if (s == 1) { emit(OP_F64_CONVERT_I64_S); } else { emit(OP_F64_CONVERT_I64_U); } }
            else { if (s == 1) { emit(OP_F32_CONVERT_I64_S); } else { emit(OP_F32_CONVERT_I64_U); } }
        } else {
            if (tk == TY_F64) { if (s == 1) { emit(OP_F64_CONVERT_I32_S); } else { emit(OP_F64_CONVERT_I32_U); } }
            else { if (s == 1) { emit(OP_F32_CONVERT_I32_S); } else { emit(OP_F32_CONVERT_I32_U); } }
        }
        return;
    }
    int tw = pw_of(tb);
    if (fk == TY_F32 || fk == TY_F64) {
        // float → int (saturating: out-of-range is undefined in C)
        int s = ty_unsigned(tb) == 0;
        if (tw == 64) {
            if (fk == TY_F32) { if (s == 1) { emit_fc(4); } else { emit_fc(5); } }
            else { if (s == 1) { emit_fc(6); } else { emit_fc(7); } }
        } else {
            if (fk == TY_F32) { if (s == 1) { emit_fc(0); } else { emit_fc(1); } }
            else { if (s == 1) { emit_fc(2); } else { emit_fc(3); } }
            if (tw < 32) { norm(tw); }
        }
        return;
    }
    // integer/pointer-like
    if (pw == 0) { pw = pw_of(fb); }
    int s = 1;
    if (pw == 1 || ty_unsigned(fb) == 1) { s = 0; }
    if (tk == TY_BOOL) {
        // LLVM's trunc to i1: the low bit
        if (pw == 64) { emit(OP_I32_WRAP_I64); }
        if (pw != 1) { emit_i32(1); emit(OP_I32_AND); }
        return;
    }
    ext_to(pw, tw, s);
}

void gen_value_as(int e, int t) {
    int k = g_node[e].kind;
    if (k == E_INITLIST || (k == E_STR && g_ty[base_scalar(t)].kind == TY_ARRAY)) {
        int off = alloc_tmp(t);
        int a = new_wl(VT_I32);
        emit_frame_addr(off);
        lset(a);
        gen_init_at(a, 0, t, e);
        emit_frame_addr(off);
        g_pw = 32;
        return;
    }
    gen_expr(e);
    conv(g_node[e].ty, g_pw, t);
    g_pw = pw_of(t);
}

// ── addresses ────────────────────────────────────────────────────────────────
void gen_index_addr(int e);

void gen_member_base(int e) {
    int b = g_node[e].a;
    int bt = g_node[b].ty;
    gen_expr(b);   // aggregates and references/pointers all yield an address
    bt = bt + 0;
}

void gen_addr(int e) {
    int k = g_node[e].kind;
    if (k == E_IDENT) {
        int c = g_node[e].c;
        if (c == SYM_LOCAL || c == SYM_BIND) { emit_local_addr(g_node[e].d); return; }
        if (c == SYM_GLOBAL) { emit_i32(glob_addr(g_node[e].d)); return; }
        if (c == SYM_FUNC) { emit_i32(table_index(g_node[e].d)); return; }
    }
    if (k == E_MEMBER) {
        gen_member_base(e);
        int mk = g_node[e].d;
        if (mk == MK_FIELD) {
            int off = g_field[g_node[e].c].off;
            if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
            return;
        }
        if (mk == MK_TUPLE) {
            int bt = g_node[g_node[e].a].ty;
            if (g_ty[bt].kind == TY_REF || g_ty[bt].kind == TY_PTR) { bt = g_ty[bt].base; }
            int off = tuple_off(bt, g_node[e].c);
            if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
            return;
        }
        if (mk == MK_SLEN) { emit_i32(8); emit(OP_I32_ADD); return; }
        return;
    }
    if (k == E_INDEX) { gen_index_addr(e); return; }
    if (k == E_DEREF || k == E_FORCE) {
        gen_expr(g_node[e].a);
        return;
    }
    if (k == E_COMMA) { gen_expr_discard(g_node[e].a); gen_addr(g_node[e].b); return; }
    // anything else: materialize
    int t = g_node[e].ty;
    if (is_agg(t) == 1) { gen_expr(e); return; }
    int off = alloc_tmp(t);
    emit_frame_addr(off);
    gen_value_as(e, t);
    emit_store(t, 0);
    emit_frame_addr(off);
}

// element address with bounds checking
void index_to_i64(int it, int pw) {
    int s = 1;
    if (ty_unsigned(it) == 1) { s = 0; }
    int bt = base_scalar(it);
    if (g_ty[bt].kind == TY_ENUM) { s = 1; }
    ext_to(pw, 64, s);
}

void gen_index_addr(int e) {
    int b = g_node[e].a;
    int bt = g_node[b].ty;
    int k = g_ty[bt].kind;
    int et = g_node[e].ty;
    int es = ty_size(et);
    int checked = (g_node[e].flags & NF_INUNSAFE) == 0;
    if (k == TY_SLICE) {
        int sl = new_wl(VT_I32);
        gen_expr(b);
        lset(sl);
        gen_expr(g_node[e].b);
        index_to_i64(g_node[g_node[e].b].ty, g_pw);
        int ix = new_wl(VT_I64);
        lset(ix);
        if (checked == 1) {
            lget(ix);
            lget(sl);
            emit_mem(OP_I64_LOAD, 3, 8);
            emit(OP_I64_GE_U);
            ctl_open(OP_IF);
            emit_trap(134);
            ctl_close();
        }
        lget(sl);
        emit_mem(OP_I32_LOAD, 2, 0);
        lget(ix);
        emit(OP_I32_WRAP_I64);
        if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
        emit(OP_I32_ADD);
        return;
    }
    int n = -1;
    if (k == TY_ARRAY) { n = g_ty[bt].n; }
    if (k == TY_REF && g_ty[g_ty[bt].base].kind == TY_ARRAY) { n = g_ty[g_ty[bt].base].n; }
    gen_expr(b);    // array: its address; pointer/reference: its value
    int idx = g_node[e].b;
    if (g_node[idx].kind == E_INT && (n < 0 || (g_node[idx].ival >= 0L && g_node[idx].ival < (long)n) || checked == 0)) {
        int off = (int)g_node[idx].ival * es;
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        return;
    }
    gen_expr(idx);
    index_to_i64(g_node[idx].ty, g_pw);
    if (checked == 1 && n >= 0) {
        int ix = new_wl(VT_I64);
        ltee(ix);
        emit_i64((long)n);
        emit(OP_I64_GE_U);
        ctl_open(OP_IF);
        emit_trap(134);
        ctl_close();
        lget(ix);
    }
    emit(OP_I32_WRAP_I64);
    if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
    emit(OP_I32_ADD);
}

// ── lvalue stores ────────────────────────────────────────────────────────────
// returns the wasm local of a register local, or -1 (then the address is pushed)
int lv_begin(int e) {
    if (g_node[e].kind == E_IDENT && (g_node[e].c == SYM_LOCAL || g_node[e].c == SYM_BIND)) {
        int l = g_node[e].d;
        setup_local(l);
        if (g_lst_kind[l] == LS_WASM) { return g_lst_val[l]; }
    }
    gen_addr(e);
    return -1;
}

// ── binary operators ─────────────────────────────────────────────────────────
int is_cmp(int op) { return op == T_EQ || op == T_NE || op == T_LT || op == T_GT || op == T_LE || op == T_GE; }

int is_float_t(int t) { int k = g_ty[base_scalar(t)].kind; return k == TY_F32 || k == TY_F64; }

void emit_float_op(int op, int is64) {
    int base = OP_F32_ADD;
    if (is64 == 1) { base = OP_F64_ADD; }
    if (op == T_PLUS || op == T_PLUSPIPE || op == T_PLUSPCT || op == T_PLUSEQ) { emit(base); return; }
    if (op == T_MINUS || op == T_MINUSPIPE || op == T_MINUSPCT || op == T_MINUSEQ) { emit(base + 1); return; }
    if (op == T_STAR || op == T_STARPIPE || op == T_STARPCT || op == T_STAREQ) { emit(base + 2); return; }
    if (op == T_SLASH || op == T_SLASHEQ) { emit(base + 3); return; }
    int cb = OP_F32_EQ;
    if (is64 == 1) { cb = OP_F64_EQ; }
    if (op == T_EQ) { emit(cb); return; }
    if (op == T_NE) { emit(cb + 1); return; }
    if (op == T_LT) { emit(cb + 2); return; }
    if (op == T_GT) { emit(cb + 3); return; }
    if (op == T_LE) { emit(cb + 4); return; }
    if (op == T_GE) { emit(cb + 5); return; }
    emit(OP_DROP);   // '%' on floats (rejected by sema)
}

int op_of_compound(int op) {
    if (op == T_PLUSEQ) { return T_PLUS; }
    if (op == T_MINUSEQ) { return T_MINUS; }
    if (op == T_STAREQ) { return T_STAR; }
    if (op == T_SLASHEQ) { return T_SLASH; }
    if (op == T_PERCENTEQ) { return T_PERCENT; }
    if (op == T_AMPEQ) { return T_AMP; }
    if (op == T_PIPEEQ) { return T_PIPE; }
    if (op == T_CARETEQ) { return T_CARET; }
    if (op == T_SHLEQ) { return T_SHL; }
    if (op == T_SHREQ) { return T_SHR; }
    return op;
}

// integer op at width w (operands already at width w, masked if unsigned
// and w < 32); leaves the result (at width w, or a 0/1 for comparisons)
void emit_int_op(int op, int w, int s) {
    if (w == 64) {
        if (op == T_PLUS || op == T_PLUSPIPE) { emit(OP_I64_ADD); return; }
        if (op == T_MINUS || op == T_MINUSPIPE) { emit(OP_I64_SUB); return; }
        if (op == T_STAR || op == T_STARPIPE) { emit(OP_I64_MUL); return; }
        if (op == T_SLASH) { if (s == 1) { emit(OP_I64_DIV_S); } else { emit(OP_I64_DIV_U); } return; }
        if (op == T_PERCENT) { if (s == 1) { emit(OP_I64_REM_S); } else { emit(OP_I64_REM_U); } return; }
        if (op == T_AMP) { emit(OP_I64_AND); return; }
        if (op == T_PIPE) { emit(OP_I64_OR); return; }
        if (op == T_CARET) { emit(OP_I64_XOR); return; }
        if (op == T_SHL) { emit(OP_I64_SHL); return; }
        if (op == T_SHR) { if (s == 1) { emit(OP_I64_SHR_S); } else { emit(OP_I64_SHR_U); } return; }
        if (op == T_EQ) { emit(OP_I64_EQ); return; }
        if (op == T_NE) { emit(OP_I64_NE); return; }
        if (op == T_LT) { if (s == 1) { emit(OP_I64_LT_S); } else { emit(OP_I64_LT_U); } return; }
        if (op == T_GT) { if (s == 1) { emit(OP_I64_GT_S); } else { emit(OP_I64_GT_U); } return; }
        if (op == T_LE) { if (s == 1) { emit(OP_I64_LE_S); } else { emit(OP_I64_LE_U); } return; }
        if (op == T_GE) { if (s == 1) { emit(OP_I64_GE_S); } else { emit(OP_I64_GE_U); } return; }
        return;
    }
    if (op == T_PLUS || op == T_PLUSPIPE) { emit(OP_I32_ADD); return; }
    if (op == T_MINUS || op == T_MINUSPIPE) { emit(OP_I32_SUB); return; }
    if (op == T_STAR || op == T_STARPIPE) { emit(OP_I32_MUL); return; }
    if (op == T_SLASH) { if (s == 1) { emit(OP_I32_DIV_S); } else { emit(OP_I32_DIV_U); } return; }
    if (op == T_PERCENT) { if (s == 1) { emit(OP_I32_REM_S); } else { emit(OP_I32_REM_U); } return; }
    if (op == T_AMP) { emit(OP_I32_AND); return; }
    if (op == T_PIPE) { emit(OP_I32_OR); return; }
    if (op == T_CARET) { emit(OP_I32_XOR); return; }
    if (op == T_SHL) { emit(OP_I32_SHL); return; }
    if (op == T_SHR) { if (s == 1) { emit(OP_I32_SHR_S); } else { emit(OP_I32_SHR_U); } return; }
    if (op == T_EQ) { emit(OP_I32_EQ); return; }
    if (op == T_NE) { emit(OP_I32_NE); return; }
    if (op == T_LT) { if (s == 1) { emit(OP_I32_LT_S); } else { emit(OP_I32_LT_U); } return; }
    if (op == T_GT) { if (s == 1) { emit(OP_I32_GT_S); } else { emit(OP_I32_GT_U); } return; }
    if (op == T_LE) { if (s == 1) { emit(OP_I32_LE_S); } else { emit(OP_I32_LE_U); } return; }
    if (op == T_GE) { if (s == 1) { emit(OP_I32_GE_S); } else { emit(OP_I32_GE_U); } return; }
}

// saturating op at width w (operands canonical at w)
void emit_sat_op(int op, int w, int s) {
    int base = op;
    if (op == T_PLUSPCT) { base = T_PLUS; }
    if (op == T_MINUSPCT) { base = T_MINUS; }
    if (op == T_STARPCT) { base = T_STAR; }
    if (w == 64) {
        // runtime helpers: __rt_sat{add,sub,mul}_{s,u}(long, long)
        tmp_reset();
        tmp_lit("__rt_sat");
        if (base == T_PLUS) { tmp_lit("add"); } else if (base == T_MINUS) { tmp_lit("sub"); } else { tmp_lit("mul"); }
        if (s == 1) { tmp_lit("_s"); } else { tmp_lit("_u"); }
        int sid = intern_tmp();
        int f = 0;
        if (g_gkind[sid] == SYM_FUNC) { f = g_gidx[sid]; }
        if (f == 0) { emit(OP_DROP); return; }
        emit(OP_CALL); emitu((long)func_index(f));
        return;
    }
    // compute exactly in i64, then clamp to the range of w bits
    int b = new_wl(VT_I64);
    if (s == 1) { emit(OP_I64_EXTEND_I32_S); } else { emit(OP_I64_EXTEND_I32_U); }
    lset(b);
    if (s == 1) { emit(OP_I64_EXTEND_I32_S); } else { emit(OP_I64_EXTEND_I32_U); }
    lget(b);
    if (base == T_PLUS) { emit(OP_I64_ADD); }
    else if (base == T_MINUS) { emit(OP_I64_SUB); }
    else { emit(OP_I64_MUL); }
    long hi = (1L << (long)(w - 1)) - 1L;
    long lo = 0L - (1L << (long)(w - 1));
    if (s == 0) { hi = (1L << (long)w) - 1L; lo = 0L; }
    int r = new_wl(VT_I64);
    lset(r);
    // r = r > hi ? hi : r
    emit_i64(hi); lget(r); lget(r); emit_i64(hi); emit(OP_I64_GT_S); emit(OP_SELECT); lset(r);
    emit_i64(lo); lget(r); lget(r); emit_i64(lo); emit(OP_I64_LT_S); emit(OP_SELECT);
    emit(OP_I32_WRAP_I64);
    if (w < 32) { norm(w); }
}

int is_wrap_or_sat(int op) {
    return op == T_PLUSPIPE || op == T_MINUSPIPE || op == T_STARPIPE || op == T_PLUSPCT || op == T_MINUSPCT || op == T_STARPCT;
}

int is_sat(int op) { return op == T_PLUSPCT || op == T_MINUSPCT || op == T_STARPCT; }

// generic binary on two already-generated integer operands: the left one is
// in local l (width lw), the right one on the stack (width rw)
void int_binop_locals(int op, int l, int lw, int lt, int r, int rw) {
    int s = ty_binsigned(lt);
    int w = lw;
    if (is_wrap_or_sat(op) == 1) { if (rw < w) { w = rw; } }
    else if (rw > w) { w = rw; }
    if (w < 8 && w != 1) { w = 8; }
    lget(l);
    ext_to(lw, w, s);
    if (s == 0 && w < 32) { mask(w); }
    lget(r);
    ext_to(rw, w, s);
    if (s == 0 && w < 32) { mask(w); }
    if (is_sat(op) == 1) {
        if (s == 0 && w < 32) { /* masked: fine */ }
        emit_sat_op(op, w, s);
        g_pw = w;
        return;
    }
    emit_int_op(op, w, s);
    if (is_cmp(op) == 1) { g_pw = 1; return; }
    if (w < 32) { norm(w); }
    g_pw = w;
}

void gen_binary(int e) {
    int op = g_node[e].op;
    int a = g_node[e].a;
    int b = g_node[e].b;
    int lt = g_node[a].ty;
    int rt = g_node[b].ty;
    // operator overload: f(&lhs, rhs)
    if (g_node[e].c == 1) {
        int f = g_node[e].d;
        int ft = g_func[f].ty;
        int ret = g_ty[ft].base;
        int sret_off = 0;
        if (is_agg(ret) == 1) { sret_off = alloc_tmp(ret); emit_frame_addr(sret_off); }
        gen_addr(a);
        int pt = g_tyl[g_ty[ft].list + 1];
        gen_arg(b, pt);
        emit(OP_CALL); emitu((long)func_index(f));
        if (is_agg(ret) == 1) { emit_frame_addr(sret_off); }
        g_pw = pw_of(ret);
        return;
    }
    if (op == T_ANDAND || op == T_OROR) {
        gen_cond(a);
        emit(OP_IF); emit(VT_I32);
        g_cdepth = g_cdepth + 1;
        if (op == T_ANDAND) { gen_cond(b); emit(OP_ELSE); emit_i32(0); }
        else { emit_i32(1); emit(OP_ELSE); gen_cond(b); }
        emit(OP_END);
        g_cdepth = g_cdepth - 1;
        g_pw = 1;
        return;
    }
    int lk = g_ty[base_scalar(lt)].kind;
    int rk = g_ty[base_scalar(rt)].kind;
    // pointer arithmetic
    if ((op == T_PLUS || op == T_MINUS) && (lk == TY_PTR || lk == TY_REF || lk == TY_ARRAY) && is_int_like(rt) == 1 && rk != TY_PTR && rk != TY_REF) {
        int el = g_ty[base_scalar(lt)].base;
        int es = ty_size(el);
        gen_expr(a);
        gen_expr(b);
        index_to_i64(rt, g_pw);
        emit(OP_I32_WRAP_I64);
        if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
        if (op == T_PLUS) { emit(OP_I32_ADD); } else { emit(OP_I32_SUB); }
        g_pw = 32;
        return;
    }
    if (op == T_PLUS && (rk == TY_PTR || rk == TY_ARRAY) && is_int_like(lt) == 1) {
        int el = g_ty[base_scalar(rt)].base;
        int es = ty_size(el);
        gen_expr(b);
        gen_expr(a);
        index_to_i64(lt, g_pw);
        emit(OP_I32_WRAP_I64);
        if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
        emit(OP_I32_ADD);
        g_pw = 32;
        return;
    }
    if (op == T_MINUS && (lk == TY_PTR || lk == TY_REF) && (rk == TY_PTR || rk == TY_REF)) {
        int es = ty_size(g_ty[base_scalar(lt)].base);
        gen_expr(a);
        gen_expr(b);
        emit(OP_I32_SUB);
        emit(OP_I64_EXTEND_I32_S);
        if (es > 1) { emit_i64((long)es); emit(OP_I64_DIV_S); }
        g_pw = 64;
        return;
    }
    // floats (either operand)
    if (is_float_t(lt) == 1 || is_float_t(rt) == 1) {
        int is64 = 0;
        if (g_ty[base_scalar(lt)].kind == TY_F64 || g_ty[base_scalar(rt)].kind == TY_F64) { is64 = 1; }
        int ft = TY_F32;
        if (is64 == 1) { ft = TY_F64; }
        gen_expr(a);
        conv(lt, g_pw, ft);
        gen_expr(b);
        conv(rt, g_pw, ft);
        emit_float_op(op, is64);
        if (is_cmp(op) == 1) { g_pw = 1; } else { g_pw = 0; }
        return;
    }
    // integers / pointers
    gen_expr(a);
    int lw = g_pw;
    if (lw == 0) { lw = pw_of(lt); }
    int vl = VT_I32;
    if (lw == 64) { vl = VT_I64; }
    int l = new_wl(vl);
    lset(l);
    gen_expr(b);
    int rw = g_pw;
    if (rw == 0) { rw = pw_of(rt); }
    int vr = VT_I32;
    if (rw == 64) { vr = VT_I64; }
    int r = new_wl(vr);
    lset(r);
    int lt2 = lt;
    if (lk == TY_PTR || lk == TY_REF || lk == TY_ENUM) { lt2 = TY_U32; }
    if (lk == TY_ENUM) { lt2 = lt; }
    int_binop_locals(op, l, lw, lt2, r, rw);
}

// ── unary / increments / assignment ──────────────────────────────────────────
void gen_incdec(int e, int is_post) {
    int op = g_node[e].op;
    int a = g_node[e].a;
    int t = g_node[a].ty;
    int tb = base_scalar(t);
    int vt = vt_of(t);
    int wl = lv_begin(a);
    int addr = -1;
    if (wl < 0) { addr = new_wl(VT_I32); ltee(addr); emit_load(t, 0); }
    else { lget(wl); }
    int old = new_wl(vt);
    lset(old);
    lget(old);
    int step = 1;
    if (g_ty[tb].kind == TY_PTR) { step = ty_size(g_ty[tb].base); }
    if (vt == VT_I64) { emit_i64((long)step); if (op == T_INC) { emit(OP_I64_ADD); } else { emit(OP_I64_SUB); } }
    else if (vt == VT_F64) { emit_f64_bits(f64_bits(1.0)); if (op == T_INC) { emit(OP_F64_ADD); } else { emit(OP_F64_SUB); } }
    else if (vt == VT_F32) { emit(OP_F32_CONST); emit(0); emit(0); emit(128); emit(63); if (op == T_INC) { emit(OP_F32_ADD); } else { emit(OP_F32_SUB); } }
    else {
        emit_i32(step);
        if (op == T_INC) { emit(OP_I32_ADD); } else { emit(OP_I32_SUB); }
        int w = pw_of(t);
        if (w < 32) { norm(w); }
    }
    int nv = new_wl(vt);
    lset(nv);
    if (wl < 0) { lget(addr); lget(nv); emit_store(t, 0); }
    else { lget(nv); lset(wl); }
    if (is_post == 1) { lget(old); } else { lget(nv); }
    g_pw = pw_of(t);
}

// x op= y  (op = T_ASSIGN for plain assignment); leaves the value if want
void gen_assign(int e, int want) {
    int op = g_node[e].op;
    int a = g_node[e].a;
    int b = g_node[e].b;
    int t = g_node[a].ty;
    int vt = vt_of(t);
    if (is_agg(t) == 1) {
        // aggregate copy
        gen_addr(a);
        int dst = new_wl(VT_I32);
        ltee(dst);
        gen_value_as(b, t);
        emit_store(t, 0);
        if (want == 1) { lget(dst); }
        g_pw = 32;
        return;
    }
    int wl = lv_begin(a);
    int addr = -1;
    if (wl < 0) { addr = new_wl(VT_I32); lset(addr); }
    if (op == T_ASSIGN) {
        gen_value_as(b, t);
    } else {
        int bop = op_of_compound(op);
        int bt = g_node[b].ty;
        int tb = base_scalar(t);
        if (g_ty[tb].kind == TY_PTR && (bop == T_PLUS || bop == T_MINUS)) {
            if (wl < 0) { lget(addr); emit_load(t, 0); } else { lget(wl); }
            gen_expr(b);
            index_to_i64(bt, g_pw);
            emit(OP_I32_WRAP_I64);
            int es = ty_size(g_ty[tb].base);
            if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
            if (bop == T_PLUS) { emit(OP_I32_ADD); } else { emit(OP_I32_SUB); }
        } else if (is_float_t(t) == 1 || is_float_t(bt) == 1) {
            int is64 = g_ty[tb].kind == TY_F64;
            int ft = TY_F32;
            if (is64 == 1 || g_ty[base_scalar(bt)].kind == TY_F64) { ft = TY_F64; is64 = 1; }
            if (wl < 0) { lget(addr); emit_load(t, 0); } else { lget(wl); }
            conv(t, pw_of(t), ft);
            gen_expr(b);
            conv(bt, g_pw, ft);
            emit_float_op(bop, is64);
            conv(ft, 0, t);
        } else {
            int lw = pw_of(t);
            int l = new_wl(vt);
            if (wl < 0) { lget(addr); emit_load(t, 0); } else { lget(wl); }
            lset(l);
            gen_expr(b);
            int rw = g_pw;
            if (rw == 0) { rw = pw_of(bt); }
            int vr = VT_I32;
            if (rw == 64) { vr = VT_I64; }
            int r = new_wl(vr);
            lset(r);
            int_binop_locals(bop, l, lw, t, r, rw);
            conv(t, g_pw, t);
        }
    }
    int v = new_wl(vt);
    lset(v);
    if (wl < 0) { lget(addr); lget(v); emit_store(t, 0); }
    else { lget(v); lset(wl); }
    if (want == 1) { lget(v); }
    g_pw = pw_of(t);
}

// ── conditions ───────────────────────────────────────────────────────────────
void gen_cond(int e) {
    gen_expr(e);
    int t = base_scalar(g_node[e].ty);
    int k = g_ty[t].kind;
    if (k == TY_F32) { emit(OP_F32_CONST); emit(0); emit(0); emit(0); emit(0); emit(OP_F32_NE); return; }
    if (k == TY_F64) { emit_f64_bits(0L); emit(OP_F64_NE); return; }
    if (g_pw == 64 || vt_of(t) == VT_I64) { emit(OP_I64_EQZ); emit(OP_I32_EQZ); return; }
    if (is_agg(t) == 1) { emit(OP_DROP); emit_i32(1); return; }
    // i32: nonzero is true; normalize to 0/1 for the result
    if (g_pw != 1) { emit(OP_I32_EQZ); emit(OP_I32_EQZ); }
}

// ── calls ────────────────────────────────────────────────────────────────────
// one argument converted to param type pt; aggregates are copied
void gen_arg(int a, int pt) {
    if (is_agg(pt) == 1) {
        int off = alloc_tmp(pt);
        emit_frame_addr(off);
        gen_value_as(a, pt);
        emit_store(pt, 0);
        emit_frame_addr(off);
        return;
    }
    gen_value_as(a, pt);
}

// variadic arguments: written to an 8-byte-slot area (C default promotions)
void gen_varargs(int e, int first) {
    int n = g_node[e].nlist - first;
    if (n < 0) { n = 0; }
    int off = alloc_frame(8 * n + 8, 8);
    int i = 0;
    while (i < n) {
        int a = g_list[g_node[e].list + first + i];
        int t = g_node[a].ty;
        int tb = base_scalar(t);
        int k = g_ty[tb].kind;
        emit_frame_addr(off + 8 * i);
        gen_expr(a);
        if (k == TY_F32 || k == TY_F64) {
            conv(t, 0, TY_F64);
            emit_mem(OP_F64_STORE, 3, 0);
        } else if (is_agg(t) == 1) {
            emit(OP_I64_EXTEND_I32_U);
            emit_mem(OP_I64_STORE, 3, 0);
        } else {
            int pw = g_pw;
            if (pw == 0) { pw = pw_of(t); }
            int s = 1;
            if (pw == 1 || ty_unsigned(tb) == 1 || k == TY_PTR || k == TY_REF) { s = 0; }
            if (pw == 32 && (k == TY_PTR || k == TY_REF || k == TY_OPT)) { s = 0; }
            ext_to(pw, 64, s);
            emit_mem(OP_I64_STORE, 3, 0);
        }
        i = i + 1;
    }
    emit_frame_addr(off);
}

void gen_args_for(int e, int ft, int self_off) {
    int np = g_ty[ft].nlist;
    int n = g_node[e].nlist;
    int i = 0;
    while (i < n && i + self_off < np) {
        int a = g_list[g_node[e].list + i];
        gen_arg(a, g_tyl[g_ty[ft].list + i + self_off]);
        i = i + 1;
    }
    if ((g_ty[ft].flags & TYF_VARIADIC) != 0) { gen_varargs(e, np - self_off); }
}

void gen_builtin(int e) {
    int bi = g_node[e].op;
    int a0 = 0;
    if (g_node[e].nlist > 0) { a0 = g_list[g_node[e].list]; }
    if (bi == BI_MEMSIZE) { emit(OP_MEMORY_SIZE); emit(0); g_pw = 32; return; }
    if (bi == BI_MEMGROW) { gen_value_as(a0, TY_I32); emit(OP_MEMORY_GROW); emit(0); g_pw = 32; return; }
    if (bi == BI_HEAPBASE) { emit(OP_GLOBAL_GET); emitu(2L); emit(OP_I64_EXTEND_I32_U); g_pw = 64; return; }
    if (bi == BI_TRAP || bi == BI_UNREACH) { emit(OP_UNREACHABLE); g_pw = 0; return; }
    if (bi == BI_F64BITS) { gen_value_as(a0, TY_F64); emit(OP_I64_REINTERPRET_F64); g_pw = 64; return; }
    if (bi == BI_F64FROM) { gen_value_as(a0, TY_I64); emit(OP_F64_REINTERPRET_I64); g_pw = 0; return; }
    if (bi == BI_CLZ || bi == BI_CTZ || bi == BI_POPCNT) {
        int t = g_node[a0].ty;
        gen_expr(a0);
        int pw = g_pw;
        if (pw == 64 || vt_of(t) == VT_I64) {
            ext_to(pw, 64, 0);
            if (bi == BI_CLZ) { emit(OP_I64_CLZ); } else if (bi == BI_CTZ) { emit(OP_I64_CTZ); } else { emit(OP_I64_POPCNT); }
            emit(OP_I32_WRAP_I64);
        } else {
            if (pw < 32) { mask(pw); }
            if (bi == BI_CLZ) { emit(OP_I32_CLZ); } else if (bi == BI_CTZ) { emit(OP_I32_CTZ); } else { emit(OP_I32_POPCNT); }
        }
        g_pw = 32;
        return;
    }
    // float unary
    int t = g_node[a0].ty;
    gen_value_as(a0, t);
    int is64 = g_ty[base_scalar(t)].kind == TY_F64;
    int base = OP_F32_ABS;
    if (is64 == 1) { base = OP_F64_ABS; }
    if (bi == BI_FABS) { emit(base); }
    else if (bi == BI_CEIL) { emit(base + 2); }
    else if (bi == BI_FLOOR) { emit(base + 3); }
    else if (bi == BI_TRUNC) { emit(base + 4); }
    else if (bi == BI_NEAREST) { emit(base + 5); }
    else if (bi == BI_SQRT) { emit(base + 6); }
    g_pw = 0;
}

void gen_call(int e) {
    int ck = g_node[e].c;
    int rt = g_node[e].ty;
    if (ck == CK_BUILTIN) { gen_builtin(e); return; }
    if (ck == CK_VAPTR) {
        if (g_vaarg >= 0) { lget(g_vaarg); } else { emit_i32(0); }
        g_pw = 32;
        return;
    }
    if (ck == CK_ISNULL || ck == CK_ISNONE) {
        int b = g_node[e].b;
        int bt = g_node[b].ty;
        gen_expr(b);
        if (is_agg(bt) == 1) {
            // ?T: present flag
            emit_mem(OP_I32_LOAD8_U, 0, ty_size(g_ty[base_scalar(bt)].base));
        }
        emit(OP_I32_EQZ);
        g_pw = 1;
        return;
    }
    if (ck == CK_DEFAULT) {
        int b = g_node[e].b;
        int bt = g_node[b].ty;
        int inner = rt;
        int fb = g_list[g_node[e].list];
        int res_vt = vt_of(inner);
        if (is_agg(inner) == 1) { res_vt = VT_I32; }
        int r = new_wl(res_vt);
        int sv = new_wl(VT_I32);
        gen_expr(b);
        lset(sv);
        lget(sv);
        if (is_agg(bt) == 1) { emit_mem(OP_I32_LOAD8_U, 0, ty_size(inner)); }
        ctl_open(OP_IF);
        lget(sv);
        if (is_agg(bt) == 1) { emit_load(inner, 0); }
        else if (g_ty[base_scalar(bt)].kind == TY_REF && (g_ty[base_scalar(bt)].flags & TYF_NULL) != 0) {
            // ?&T default gives the referent value T
            emit_load(inner, 0);
        } else {
            emit_load(inner, 0);
        }
        lset(r);
        emit(OP_ELSE);
        gen_value_as(fb, inner);
        lset(r);
        ctl_close();
        lget(r);
        g_pw = pw_of(inner);
        return;
    }
    if (ck == CK_UNIONINIT) {
        int sd = g_node[e].d;
        int t = g_sdef[sd].ty;
        layout_struct(sd);
        int off = alloc_tmp(t);
        int fi = g_sdef[sd].field0 + g_node[e].op;
        emit_frame_addr(off);
        emit_i32(0);
        emit_i32(ty_size(t));
        emit_fc(11); emit(0);
        emit_frame_addr(off);
        emit_i32(g_node[e].op);
        emit_mem(OP_I32_STORE, 2, 0);
        if (g_node[e].nlist > 0) {
            emit_frame_addr(off);
            gen_value_as(g_list[g_node[e].list], g_field[fi].ty);
            emit_store(g_field[fi].ty, g_field[fi].off);
        }
        emit_frame_addr(off);
        g_pw = 32;
        return;
    }
    int ft = 0;
    int f = 0;
    if (ck == CK_FUNC || ck == CK_METHOD) { f = g_node[e].d; ft = g_func[f].ty; }
    else { ft = fn_type_of(g_node[g_node[e].a].ty); }
    int ret = g_ty[ft].base;
    int sret_off = 0;
    if (is_agg(ret) == 1) { sret_off = alloc_tmp(ret); emit_frame_addr(sret_off); }
    int self_off = 0;
    if (ck == CK_METHOD) {
        // self: the receiver's address (a reference/pointer receiver is the address)
        int b = g_node[e].b;
        gen_expr(b);
        self_off = 1;
    }
    gen_args_for(e, ft, self_off);
    if (ck == CK_INDIRECT) {
        gen_expr(g_node[e].a);
        emit(OP_CALL_INDIRECT);
        emitu((long)sig_of_fn_type(ft));
        emit(0);
    } else {
        emit(OP_CALL);
        emitu((long)func_index(f));
    }
    if (is_agg(ret) == 1) { emit_frame_addr(sret_off); }
    g_pw = pw_of(ret);
}

// ── match ────────────────────────────────────────────────────────────────────
// subject in local sv (scalar value or aggregate address); pushes the arm's condition
void gen_pattern_cond(int p, int sv, int st) {
    int k = g_node[p].kind;
    int stb = base_scalar(st);
    int is64 = vt_of(stb) == VT_I64;
    if (k == P_WILD) { emit_i32(1); return; }
    if (k == P_VALUE || k == P_RANGE) {
        int w = pw_of(stb);
        long lo = g_node[p].ival;
        long hi = lo;
        if (k == P_RANGE) { lo = g_node[g_node[p].a].ival; hi = g_node[g_node[p].b].ival; }
        // patterns compare as the subject's width (canonical sign-extended)
        if (w < 32 && w != 1) {
            long m = (1L << (long)w) - 1L;
            lo = lo & m; if ((lo & (1L << (long)(w - 1))) != 0L) { lo = lo - (1L << (long)w); }
            hi = hi & m; if ((hi & (1L << (long)(w - 1))) != 0L) { hi = hi - (1L << (long)w); }
        }
        int s = ty_is_signed(stb) == 1 || g_ty[stb].kind == TY_ENUM;
        if (k == P_VALUE) {
            lget(sv);
            if (is64 == 1) { emit_i64(lo); emit(OP_I64_EQ); } else { emit_i32((int)lo); emit(OP_I32_EQ); }
            return;
        }
        lget(sv);
        if (is64 == 1) { emit_i64(lo); if (s == 1) { emit(OP_I64_GE_S); } else { emit(OP_I64_GE_U); } }
        else { emit_i32((int)lo); if (s == 1) { emit(OP_I32_GE_S); } else { emit(OP_I32_GE_U); } }
        lget(sv);
        if (is64 == 1) { emit_i64(hi); if (s == 1) { emit(OP_I64_LE_S); } else { emit(OP_I64_LE_U); } }
        else { emit_i32((int)hi); if (s == 1) { emit(OP_I32_LE_S); } else { emit(OP_I32_LE_U); } }
        emit(OP_I32_AND);
        return;
    }
    if (k == P_NULL || k == P_SOME) {
        lget(sv);
        if (is_agg(stb) == 1) { emit_mem(OP_I32_LOAD8_U, 0, ty_size(g_ty[stb].base)); }
        if (k == P_NULL) { emit(OP_I32_EQZ); }
        else if (is_agg(stb) == 0) { emit(OP_I32_EQZ); emit(OP_I32_EQZ); }
        return;
    }
    if (k == P_VARIANT) {
        int sd = g_ty[stb].n;
        lget(sv);
        emit_mem(OP_I32_LOAD, 2, 0);
        emit_i32(g_node[p].c - g_sdef[sd].field0);
        emit(OP_I32_EQ);
        return;
    }
    emit_i32(0);   // an unresolved name matches nothing
}

void gen_pattern_bind(int p, int sv, int st) {
    int k = g_node[p].kind;
    int stb = base_scalar(st);
    if (k == P_SOME && g_node[p].d != 0) {
        int l = g_node[p].b;
        int lt = g_local[l].ty;
        setup_local(l);
        if (is_agg(stb) == 0) {
            // ?&T / T*: the binding is the (non-null) pointer itself
            if (g_lst_kind[l] == LS_WASM) { lget(sv); lset(g_lst_val[l]); }
            else { emit_local_addr(l); lget(sv); emit_store(lt, 0); }
            return;
        }
        if (g_lst_kind[l] == LS_WASM) { lget(sv); emit_load(lt, 0); lset(g_lst_val[l]); }
        else { emit_local_addr(l); lget(sv); emit_load(lt, 0); emit_store(lt, 0); }
        return;
    }
    if (k == P_VARIANT && g_node[p].d != 0) {
        int l = g_node[p].b;
        int lt = g_local[l].ty;
        int off = g_field[g_node[p].c].off;
        setup_local(l);
        if (g_lst_kind[l] == LS_WASM) { lget(sv); emit_load(lt, off); lset(g_lst_val[l]); }
        else { emit_local_addr(l); lget(sv); emit_load(lt, off); emit_store(lt, 0); }
    }
}

void gen_match(int s, int is_expr) {
    int st = g_node[s].c;
    int stb = base_scalar(st);
    int svt = vt_of(stb);
    if (is_agg(stb) == 1 || svt == 0) { svt = VT_I32; }
    int sv = new_wl(svt);
    gen_expr(g_node[s].a);
    lset(sv);
    int rt = 0;
    int r = -1;
    if (is_expr == 1) {
        rt = g_node[s].ty;
        int rvt = vt_of(rt);
        if (is_agg(rt) == 1 || rvt == 0) { rvt = VT_I32; }
        r = new_wl(rvt);
    }
    ctl_open(OP_BLOCK);
    int end = g_cdepth;
    // 'break' in an arm exits the enclosing loop; with no loop, the match
    push_loop(end, -1, 0, 2);
    int i = 0;
    while (i < g_node[s].nlist) {
        int arm = g_list[g_node[s].list + i];
        ctl_open(OP_BLOCK);
        int j = 0;
        while (j < g_node[arm].nlist) {
            gen_pattern_cond(g_list[g_node[arm].list + j], sv, st);
            if (j > 0) { emit(OP_I32_OR); }
            j = j + 1;
        }
        emit(OP_I32_EQZ);
        emit(OP_BR_IF); emitu(0L);
        if (g_node[arm].nlist == 1) { gen_pattern_bind(g_list[g_node[arm].list], sv, st); }
        if (is_expr == 1) {
            gen_value_as(g_node[arm].a, rt);
            lset(r);
        } else {
            int h = g_ndf;
            gen_stmt(g_node[arm].a);
            g_ndf = h;
        }
        emit_br_to(end);
        ctl_close();
        i = i + 1;
    }
    g_nlp = g_nlp - 1;
    ctl_close();
    if (is_expr == 1) { lget(r); g_pw = pw_of(rt); }
}

// ── new / arena ──────────────────────────────────────────────────────────────
void gen_new(int e) {
    int t = g_node[e].c;
    int r = g_node[e].sid;
    int sz = ty_size(t);
    int al = ty_align(t);
    if (r == 0) {
        // heap: __rt_new(size)
        int f = 0;
        int sid = intern_lit("__rt_new");
        if (g_gkind[sid] == SYM_FUNC) { f = g_gidx[sid]; }
        emit_i64((long)sz);
        if (g_node[e].a != 0) { gen_value_as(g_node[e].a, TY_I64); emit(OP_I64_MUL); }
        if (f != 0) { emit(OP_CALL); emitu((long)func_index(f)); }
        else { emit(OP_DROP); emit_i32(0); }
        g_pw = 32;
        return;
    }
    int buf = region_buf(r);
    int offa = region_off(r);
    int cap = g_region_cap[r];
    int start = new_wl(VT_I64);
    // start = (off + al - 1) & ~(al - 1)
    emit_i32(offa);
    emit_mem(OP_I64_LOAD, 3, 0);
    emit_i64((long)(al - 1));
    emit(OP_I64_ADD);
    emit_i64(0L - (long)al);
    emit(OP_I64_AND);
    lset(start);
    int end = new_wl(VT_I64);
    lget(start);
    emit_i64((long)sz);
    if (g_node[e].a != 0) { gen_value_as(g_node[e].a, TY_I64); emit(OP_I64_MUL); }
    emit(OP_I64_ADD);
    ltee(end);
    emit_i64((long)cap);
    emit(OP_I64_GT_U);
    ctl_open(OP_IF);
    emit_trap(134);
    ctl_close();
    emit_i32(offa);
    lget(end);
    emit_mem(OP_I64_STORE, 3, 0);
    lget(start);
    emit(OP_I32_WRAP_I64);
    emit_i32(buf);
    emit(OP_I32_ADD);
    g_pw = 32;
}

void gen_arena(int e) {
    int r = g_node[e].sid;
    int k = g_node[e].op;
    int offa = region_off(r);
    if (k == K_ARENA_MARK) {
        emit_i32(offa);
        emit_mem(OP_I64_LOAD, 3, 0);
        g_pw = 64;
        return;
    }
    emit_i32(offa);
    if (k == K_ARENA_FREE_TO) { gen_value_as(g_node[e].a, TY_U64); }
    else { emit_i64(0L); }
    emit_mem(OP_I64_STORE, 3, 0);
    g_pw = 0;
}

// ── slices ───────────────────────────────────────────────────────────────────
void gen_slice(int e) {
    int b = g_node[e].a;
    int bt = g_node[b].ty;
    int k = g_ty[bt].kind;
    int el = g_ty[g_node[e].ty].base;
    int es = ty_size(el);
    int off = alloc_frame(16, 8);
    int base = new_wl(VT_I32);
    int lo = new_wl(VT_I64);
    int hi = new_wl(VT_I64);
    long n = -1L;
    if (k == TY_ARRAY) { n = (long)g_ty[bt].n; }
    if (k == TY_REF && g_ty[g_ty[bt].base].kind == TY_ARRAY) { n = (long)g_ty[g_ty[bt].base].n; }
    if (k == TY_SLICE) {
        int sl = new_wl(VT_I32);
        gen_expr(b);
        ltee(sl);
        emit_mem(OP_I32_LOAD, 2, 0);
        lset(base);
        if (g_node[e].c == 0) { lget(sl); emit_mem(OP_I64_LOAD, 3, 8); lset(hi); }
        // bound = the slice length
        int ln = new_wl(VT_I64);
        lget(sl);
        emit_mem(OP_I64_LOAD, 3, 8);
        lset(ln);
        if (g_node[e].b != 0) { gen_expr(g_node[e].b); index_to_i64(g_node[g_node[e].b].ty, g_pw); lset(lo); }
        else { emit_i64(0L); lset(lo); }
        if (g_node[e].c != 0) { gen_expr(g_node[e].c); index_to_i64(g_node[g_node[e].c].ty, g_pw); lset(hi); }
        if ((g_node[e].flags & NF_INUNSAFE) == 0) {
            lget(lo); lget(hi); emit(OP_I64_GT_U);
            lget(hi); lget(ln); emit(OP_I64_GT_U);
            emit(OP_I32_OR);
            ctl_open(OP_IF);
            emit_trap(134);
            ctl_close();
        }
    } else {
        gen_expr(b);
        lset(base);
        if (g_node[e].b != 0) { gen_expr(g_node[e].b); index_to_i64(g_node[g_node[e].b].ty, g_pw); lset(lo); }
        else { emit_i64(0L); lset(lo); }
        if (g_node[e].c != 0) { gen_expr(g_node[e].c); index_to_i64(g_node[g_node[e].c].ty, g_pw); lset(hi); }
        else { emit_i64(n); lset(hi); }
        if ((g_node[e].flags & NF_INUNSAFE) == 0 && n >= 0L) {
            lget(lo); lget(hi); emit(OP_I64_GT_U);
            lget(hi); emit_i64(n); emit(OP_I64_GT_U);
            emit(OP_I32_OR);
            ctl_open(OP_IF);
            emit_trap(134);
            ctl_close();
        }
    }
    emit_frame_addr(off);
    lget(base);
    lget(lo);
    emit(OP_I32_WRAP_I64);
    if (es != 1) { emit_i32(es); emit(OP_I32_MUL); }
    emit(OP_I32_ADD);
    emit_mem(OP_I32_STORE, 2, 0);
    emit_frame_addr(off);
    lget(hi);
    lget(lo);
    emit(OP_I64_SUB);
    emit_mem(OP_I64_STORE, 3, 8);
    emit_frame_addr(off);
    g_pw = 32;
}

// ── expressions ──────────────────────────────────────────────────────────────
void gen_ident(int e) {
    int c = g_node[e].c;
    int t = g_node[e].ty;
    if (c == SYM_LOCAL || c == SYM_BIND) {
        int l = g_node[e].d;
        setup_local(l);
        if (g_lst_kind[l] == LS_WASM) { lget(g_lst_val[l]); }
        else { emit_local_addr(l); emit_load(t, 0); }
        g_pw = pw_of(t);
        return;
    }
    if (c == SYM_GLOBAL) {
        emit_i32(glob_addr(g_node[e].d));
        emit_load(t, 0);
        g_pw = pw_of(t);
        return;
    }
    if (c == SYM_ENUMC) {
        long v = g_field[g_node[e].d].val;
        if (vt_of(t) == VT_I64) { emit_i64(v); } else { emit_i32((int)v); }
        g_pw = pw_of(t);
        return;
    }
    if (c == SYM_FUNC) { emit_i32(table_index(g_node[e].d)); g_pw = 32; return; }
    emit_zero(t);
    g_pw = pw_of(t);
}

void gen_cast(int e) {
    int t = g_node[e].c;
    int o = g_node[e].a;
    int ot = g_node[o].ty;
    int tb = base_scalar(t);
    int ob = base_scalar(ot);
    if (g_ty[tb].kind == TY_VOID) { gen_expr_discard(o); g_pw = 0; return; }
    // (&T)lvalue of a non-pointer type: its address
    if (g_ty[tb].kind == TY_REF && is_lval(o) == 1 && g_ty[ob].kind != TY_REF && g_ty[ob].kind != TY_PTR && g_ty[ob].kind != TY_ARRAY) {
        int rb = g_ty[tb].base;
        if (g_ty[rb].kind == TY_VOID || ty_size(rb) == ty_size(ot) || ty_is_int(ot) == 0) {
            gen_addr(o);
            g_pw = 32;
            return;
        }
    }
    if (g_node[o].kind == E_INITLIST) { gen_value_as(o, t); return; }
    gen_expr(o);
    conv(ot, g_pw, t);
    g_pw = pw_of(t);
}

void gen_ternary(int e) {
    int t = g_node[e].ty;
    int vt = vt_of(t);
    if (is_agg(t) == 1 || vt == 0) { vt = VT_I32; }
    gen_cond(g_node[e].a);
    emit(OP_IF); emit(vt);
    g_cdepth = g_cdepth + 1;
    gen_value_as(g_node[e].b, t);
    emit(OP_ELSE);
    gen_value_as(g_node[e].c, t);
    emit(OP_END);
    g_cdepth = g_cdepth - 1;
    g_pw = pw_of(t);
}

void gen_tuple(int e) {
    int t = g_node[e].ty;
    int off = alloc_tmp(t);
    int i = 0;
    while (i < g_node[e].nlist) {
        int el = g_list[g_node[e].list + i];
        int et = g_tyl[g_ty[t].list + i];
        emit_frame_addr(off);
        gen_value_as(el, et);
        emit_store(et, tuple_off(t, i));
        i = i + 1;
    }
    emit_frame_addr(off);
    g_pw = 32;
}

void gen_try(int e) {
    int a = g_node[e].a;
    int ot = g_node[a].ty;
    int inner = g_ty[ot].base;
    int sv = new_wl(VT_I32);
    gen_expr(a);
    lset(sv);
    lget(sv);
    emit_mem(OP_I32_LOAD8_U, 0, ty_size(inner));
    emit(OP_I32_EQZ);
    ctl_open(OP_IF);
    gen_try_fail();
    ctl_close();
    lget(sv);
    emit_load(inner, 0);
    g_pw = pw_of(inner);
}

void gen_expr(int e) {
    int k = g_node[e].kind;
    int t = g_node[e].ty;
    g_pw = pw_of(t);
    if (k == E_INT || k == E_CHAR || k == E_BOOL) {
        long v = g_node[e].ival;
        int w = pw_of(t);
        if (vt_of(t) == VT_I64) { emit_i64(v); }
        else {
            if (w < 32 && w != 1) {
                long m = (1L << (long)w) - 1L;
                v = v & m;
                if ((v & (1L << (long)(w - 1))) != 0L) { v = v - (1L << (long)w); }
            }
            emit_i32((int)v);
        }
        return;
    }
    if (k == E_FLOAT) {
        emit_f64_bits(g_node[e].ival);
        if (g_ty[t].kind == TY_F32) { emit(OP_F32_DEMOTE_F64); }
        g_pw = 0;
        return;
    }
    if (k == E_STR) { emit_i32(str_addr(g_node[e].sid)); g_pw = 32; return; }
    if (k == E_NULL) { emit_i32(0); g_pw = 32; return; }
    if (k == E_IDENT) { gen_ident(e); return; }
    if (k == E_MEMBER) {
        int mk = g_node[e].d;
        if (mk == MK_SLEN) { gen_member_base(e); emit_mem(OP_I64_LOAD, 3, 8); g_pw = 64; return; }
        if (mk == MK_SPTR) { gen_member_base(e); emit_mem(OP_I32_LOAD, 2, 0); g_pw = 32; return; }
        gen_member_base(e);
        int off = 0;
        if (mk == MK_FIELD) { off = g_field[g_node[e].c].off; }
        else if (mk == MK_TUPLE) {
            int bt = g_node[g_node[e].a].ty;
            if (g_ty[bt].kind == TY_REF || g_ty[bt].kind == TY_PTR) { bt = g_ty[bt].base; }
            off = tuple_off(bt, g_node[e].c);
        }
        emit_load(t, off);
        g_pw = pw_of(t);
        return;
    }
    if (k == E_INDEX) { gen_index_addr(e); emit_load(t, 0); g_pw = pw_of(t); return; }
    if (k == E_SLICE) { gen_slice(e); return; }
    if (k == E_DEREF || k == E_FORCE) {
        int a = g_node[e].a;
        gen_expr(a);
        int at = base_scalar(g_node[a].ty);
        // ?T!: the payload at offset 0 of the optional's storage
        emit_load(t, 0);
        at = at + 0;
        g_pw = pw_of(t);
        return;
    }
    if (k == E_ADDR) {
        int o = g_node[e].a;
        if (g_node[o].kind == E_IDENT && g_node[o].c == SYM_FUNC) { emit_i32(table_index(g_node[o].d)); g_pw = 32; return; }
        gen_addr(o);
        g_pw = 32;
        return;
    }
    if (k == E_UNARY) {
        int op = g_node[e].op;
        if (op == T_INC || op == T_DEC) { gen_incdec(e, 0); return; }
        int a = g_node[e].a;
        if (op == T_BANG) {
            gen_cond(a);
            emit(OP_I32_EQZ);
            g_pw = 1;
            return;
        }
        gen_expr(a);
        int pw = g_pw;
        int at = base_scalar(g_node[a].ty);
        if (op == T_MINUS) {
            int fk = g_ty[at].kind;
            if (fk == TY_F32) { emit(OP_F32_NEG); g_pw = 0; return; }
            if (fk == TY_F64) { emit(OP_F64_NEG); g_pw = 0; return; }
            if (pw == 64) {
                int tmp = new_wl(VT_I64);
                lset(tmp);
                emit_i64(0L);
                lget(tmp);
                emit(OP_I64_SUB);
            } else {
                int tmp = new_wl(VT_I32);
                lset(tmp);
                emit_i32(0);
                lget(tmp);
                emit(OP_I32_SUB);
                if (pw < 32) { norm(pw); }
            }
            g_pw = pw;
            return;
        }
        if (op == T_TILDE) {
            if (pw == 64) { emit_i64(-1L); emit(OP_I64_XOR); }
            else { emit_i32(-1); emit(OP_I32_XOR); if (pw == 1) { emit_i32(1); emit(OP_I32_AND); } }
            g_pw = pw;
            return;
        }
        return;
    }
    if (k == E_POSTFIX) { gen_incdec(e, 1); return; }
    if (k == E_BINARY) { gen_binary(e); return; }
    if (k == E_ASSIGN) { gen_assign(e, 1); return; }
    if (k == E_TERNARY) { gen_ternary(e); return; }
    if (k == E_CALL) { gen_call(e); return; }
    if (k == E_CAST) { gen_cast(e); return; }
    if (k == E_SIZEOF_T || k == E_ALIGNOF) {
        int st = g_node[e].c;
        long v = (long)ty_size(st);
        if (k == E_ALIGNOF) { v = (long)ty_align(st); }
        emit_i64(v);
        g_pw = 64;
        return;
    }
    if (k == E_SIZEOF_E) { emit_i64((long)ty_size(g_node[g_node[e].a].ty)); g_pw = 64; return; }
    if (k == E_INITLIST) { gen_value_as(e, t); return; }
    if (k == E_NEW) { gen_new(e); return; }
    if (k == E_ARENA) { gen_arena(e); return; }
    if (k == E_TUPLE) { gen_tuple(e); return; }
    if (k == E_MATCH) { gen_match(e, 1); return; }
    if (k == E_TRY) { gen_try(e); return; }
    if (k == E_COMMA) {
        gen_expr_discard(g_node[e].a);
        gen_expr(g_node[e].b);
        return;
    }
    emit_zero(t);
}

void gen_expr_discard(int e) {
    int k = g_node[e].kind;
    if (k == E_ASSIGN) { gen_assign(e, 0); return; }
    if (k == E_POSTFIX || (k == E_UNARY && (g_node[e].op == T_INC || g_node[e].op == T_DEC))) {
        gen_incdec(e, 1);
        emit(OP_DROP);
        return;
    }
    if (k == E_COMMA) { gen_expr_discard(g_node[e].a); gen_expr_discard(g_node[e].b); return; }
    gen_expr(e);
    int t = g_node[e].ty;
    if (k == E_CALL) {
        int ck = g_node[e].c;
        int ret = t;
        if (ck == CK_FUNC || ck == CK_METHOD) { ret = g_ty[g_func[g_node[e].d].ty].base; }
        if (ck == CK_BUILTIN && (g_node[e].op == BI_TRAP || g_node[e].op == BI_UNREACH)) { return; }
        if (vt_of(ret) != 0 || is_agg(ret) == 1) { emit(OP_DROP); }
        return;
    }
    if (k == E_ARENA && g_node[e].op != K_ARENA_MARK) { return; }
    if (vt_of(t) != 0 || is_agg(t) == 1) { emit(OP_DROP); }
}
