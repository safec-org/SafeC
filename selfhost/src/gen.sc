// gen.sc — the WebAssembly code generator: function frames, locals,
// statements, defer/break/continue/return, data layout and constant
// initializers. Expressions are in genx.sc.
#include "scc.h"
#include "parse.h"
#include "sema.h"
#include "sema_int.h"
#include "gen.h"
#include "gen_int.h"

// ── per-function state ───────────────────────────────────────────────────────
int g_wl_ty[8192];
int g_wl_n = 0;
int g_fp = 0;          // wasm local: frame base
int g_oldsp = 0;       // wasm local: caller's stack pointer
int g_sret = -1;       // wasm local holding the sret pointer (-1: none)
int g_vaarg = -1;      // wasm local holding the vararg area pointer
int g_frame = 0;       // frame bytes
int g_cdepth = 0;      // wasm control nesting depth
int g_gen_fn = 0;
int g_gen_ret = 0;     // return type

// local storage (indexed by g_local index)
int g_lst_kind[NLOCAL_MAX];
int g_lst_val[NLOCAL_MAX];

// loops: break/continue targets (control depths) and defer heights
int g_lp_brk[256];
int g_lp_cont[256];
int g_lp_label[256];
int g_lp_defer[256];
int g_lp_ismatch[256];
int g_nlp = 0;

// defers
int g_df_node[1024];
int g_df_err[1024];
int g_ndf = 0;

// ── wasm locals and frame ────────────────────────────────────────────────────
int new_wl(int vt) {
    if (g_wl_n >= 8192) { fatal("wasm locals"); return 0; }
    g_wl_ty[g_wl_n] = vt;
    g_wl_n = g_wl_n + 1;
    return g_wl_n - 1;
}

int align_to(int v, int a) {
    if (a <= 1) { return v; }
    return (v + a - 1) / a * a;
}

int alloc_frame(int size, int al) {
    if (al < 1) { al = 1; }
    if (al > 16) { al = 16; }
    int off = align_to(g_frame, al);
    if (size < 1) { size = 1; }
    g_frame = off + size;
    return off;
}

// pushes fp + off
void emit_frame_addr(int off) {
    emit(OP_LOCAL_GET); emitu((long)g_fp);
    if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
}

// a temporary of type t in the frame; pushes its address
int alloc_tmp(int t) {
    int sz = ty_size(t);
    int off = alloc_frame(sz, ty_align(t));
    return off;
}

// ── type helpers ─────────────────────────────────────────────────────────────
int base_scalar(int t) {
    int k = g_ty[t].kind;
    if (k == TY_ENUM || k == TY_NEWTYPE) { return base_scalar(g_sdef[g_ty[t].n].base); }
    return t;
}

int vt_of(int t) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    if (k == TY_VOID || k == TY_ERROR) { return 0; }
    if (k == TY_I64 || k == TY_U64) { return VT_I64; }
    if (k == TY_F32) { return VT_F32; }
    if (k == TY_F64) { return VT_F64; }
    return VT_I32;
}

int is_agg(int t) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    if (k == TY_OPT) {
        // ?&T is a plain nullable pointer
        if (g_ty[g_ty[t].base].kind == TY_REF) { return 0; }
        return 1;
    }
    return k == TY_STRUCT || k == TY_ARRAY || k == TY_SLICE || k == TY_TUPLE;
}

int is_ptr_opt(int t) { return g_ty[t].kind == TY_OPT && g_ty[g_ty[t].base].kind == TY_REF; }

// physical integer width of a value of type t (bool = 1)
int pw_of(int t) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    if (k == TY_BOOL) { return 1; }
    if (k == TY_CHAR || k == TY_I8 || k == TY_U8) { return 8; }
    if (k == TY_I16 || k == TY_U16) { return 16; }
    if (k == TY_I64 || k == TY_U64) { return 64; }
    if (k == TY_F32 || k == TY_F64) { return 0; }
    return 32;
}

int ty_unsigned(int t) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    return k == TY_U8 || k == TY_U16 || k == TY_U32 || k == TY_U64 || k == TY_BOOL;
}

// the left-operand signedness rule of binary operators
int ty_binsigned(int t) {
    int k = g_ty[t].kind;
    return k == TY_I8 || k == TY_I16 || k == TY_I32 || k == TY_I64;
}

void emit_load(int t, int off) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    if (k == TY_BOOL) { emit_mem(OP_I32_LOAD8_U, 0, off); return; }
    if (k == TY_CHAR || k == TY_I8 || k == TY_U8) { emit_mem(OP_I32_LOAD8_S, 0, off); return; }
    if (k == TY_I16 || k == TY_U16) { emit_mem(OP_I32_LOAD16_S, 1, off); return; }
    if (k == TY_I64 || k == TY_U64) { emit_mem(OP_I64_LOAD, 3, off); return; }
    if (k == TY_F32) { emit_mem(OP_F32_LOAD, 2, off); return; }
    if (k == TY_F64) { emit_mem(OP_F64_LOAD, 3, off); return; }
    if (is_agg(t) == 1) {
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        return;
    }
    emit_mem(OP_I32_LOAD, 2, off);
}

// stack: addr, value
void emit_store(int t, int off) {
    t = base_scalar(t);
    int k = g_ty[t].kind;
    if (k == TY_BOOL || k == TY_CHAR || k == TY_I8 || k == TY_U8) { emit_mem(OP_I32_STORE8, 0, off); return; }
    if (k == TY_I16 || k == TY_U16) { emit_mem(OP_I32_STORE16, 1, off); return; }
    if (k == TY_I64 || k == TY_U64) { emit_mem(OP_I64_STORE, 3, off); return; }
    if (k == TY_F32) { emit_mem(OP_F32_STORE, 2, off); return; }
    if (k == TY_F64) { emit_mem(OP_F64_STORE, 3, off); return; }
    if (is_agg(t) == 1) {
        // [dst, src] → memory.copy(dst+off, src, size)
        int src = new_wl(VT_I32);
        emit(OP_LOCAL_SET); emitu((long)src);
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        emit(OP_LOCAL_GET); emitu((long)src);
        emit_i32(ty_size(t));
        emit_fc(10); emit(0); emit(0);
        return;
    }
    emit_mem(OP_I32_STORE, 2, off);
}

void emit_zero(int t) {
    int vt = vt_of(t);
    if (is_agg(t) == 1) {
        int off = alloc_tmp(t);
        emit_frame_addr(off);
        emit_i32(0);
        emit_i32(ty_size(t));
        emit_fc(11); emit(0);
        emit_frame_addr(off);
        return;
    }
    if (vt == VT_I32) { emit_i32(0); }
    else if (vt == VT_I64) { emit_i64(0L); }
    else if (vt == VT_F32) { emit(OP_F32_CONST); emit(0); emit(0); emit(0); emit(0); }
    else if (vt == VT_F64) { emit_f64_bits(0L); }
}

void emit_drop_ty(int t) {
    if (vt_of(t) != 0 || is_agg(t) == 1) { emit(OP_DROP); }
}

// ── traps ────────────────────────────────────────────────────────────────────
int g_rt_trap = 0;   // runtime function __rt_trap(int code)

void emit_trap(int code) {
    if (g_rt_trap != 0) {
        emit_i32(code);
        emit(OP_CALL);
        emitu((long)func_index(g_rt_trap));
    }
    emit(OP_UNREACHABLE);
}

// ── functions: indices and signatures ────────────────────────────────────────
int g_emit_queue[NFUNC_MAX];
int g_nemit = 0;
int g_emit_pos = 0;

int sig_of_fn_type(int ft) {
    int mark = g_nftype_param;
    int ret = g_ty[ft].base;
    if (is_agg(ret) == 1) { push_ftparam(VT_I32); }
    int i = 0;
    while (i < g_ty[ft].nlist) {
        int pt = g_tyl[g_ty[ft].list + i];
        int vt = vt_of(pt);
        if (is_agg(pt) == 1 || vt == 0) { vt = VT_I32; }
        push_ftparam(vt);
        i = i + 1;
    }
    if ((g_ty[ft].flags & TYF_VARIADIC) != 0) { push_ftparam(VT_I32); }
    int res = 0;
    int nres = 0;
    if (is_agg(ret) == 0 && vt_of(ret) != 0) { res = vt_of(ret); nres = 1; }
    return func_type_index(mark, nres, res);
}

void push_ftparam(int vt) {
    if (g_nftype_param >= NFUNC_MAX * 4) { fatal("function type params"); return; }
    g_ftype_param[g_nftype_param] = vt;
    g_nftype_param = g_nftype_param + 1;
}

int is_host_import(int f) {
    int s = g_func[f].sid;
    return g_str_len[s] > 7 && str_char(s, 0) == '_' && str_char(s, 1) == '_' && str_char(s, 2) == 'h' &&
           str_char(s, 3) == 'o' && str_char(s, 4) == 's' && str_char(s, 5) == 't' && str_char(s, 6) == '_';
}

// wasm index of a function, assigning it (and queueing its body) on first use
int func_index(int f) {
    if (g_func[f].widx >= 0) { return g_func[f].widx; }
    int fnode = g_func[f].node;
    if (g_node[fnode].a == 0) {
        if (is_host_import(f) == 0) {
            msg_reset();
            msg_lit("undefined function '");
            msg_str(g_func[f].origsid);
            msg_lit("': it is declared but has no definition (the playground runtime doesn't provide it)");
            error_node(fnode);
            g_func[f].widx = 0;
            return 0;
        }
        // host imports are pre-assigned; unknown host names are errors
        msg_reset();
        msg_lit("unknown host import '");
        msg_str(g_func[f].sid);
        msg_lit("'");
        error_node(fnode);
        g_func[f].widx = 0;
        return 0;
    }
    if (g_nwfunc >= NFUNC_MAX) { fatal("wasm functions"); return 0; }
    g_func[f].widx = g_nimports + g_nwfunc;
    g_wfunc[g_nwfunc] = f;
    g_wfunc_type[g_nwfunc] = sig_of_fn_type(g_func[f].ty);
    g_nwfunc = g_nwfunc + 1;
    g_emit_queue[g_nemit] = f;
    g_nemit = g_nemit + 1;
    return g_func[f].widx;
}

int table_index(int f) {
    int i = 0;
    while (i < g_ntable) { if (g_table[i] == f) { return i + 1; } i = i + 1; }
    func_index(f);
    if (g_ntable >= NFUNC_MAX) { return 0; }
    g_table[g_ntable] = f;
    g_ntable = g_ntable + 1;
    return g_ntable;
}

// ── data ─────────────────────────────────────────────────────────────────────
int g_stradr[NSTR_MAX];
int g_region_buf[NSTR_MAX];
int g_region_off[NSTR_MAX];

// zero-initialized storage lives in "bss", above the data segment's maximum
int g_bss_len = 0;
int bss_alloc(int size, int al) {
    if (al < 1) { al = 1; }
    int off = align_to(g_bss_len, al);
    if (size < 1) { size = 1; }
    g_bss_len = off + size;
    if (g_bss_len > 1073741824) { fatal("static data (bss)"); }
    return DATA_BASE + DATA_MAX + off;
}

int data_alloc(int size, int al) {
    if (al < 1) { al = 1; }
    int off = align_to(g_data_len, al);
    if (off + size >= DATA_MAX) { fatal("static data"); return DATA_BASE; }
    int i = g_data_len;
    while (i < off + size) { g_data[i] = (unsigned char)0; i = i + 1; }
    g_data_len = off + size;
    return DATA_BASE + off;
}

void data_put(int addr, int size, long v) {
    int i = 0;
    int o = addr - DATA_BASE;
    while (i < size) {
        g_data[o + i] = (unsigned char)((v >> (long)(i * 8)) & 255L);
        i = i + 1;
    }
}

int str_addr(int sid) {
    if (g_stradr[sid] != 0) { return g_stradr[sid]; }
    int n = g_str_len[sid];
    int a = data_alloc(n + 1, 1);
    int i = 0;
    while (i < n) { g_data[a - DATA_BASE + i] = (unsigned char)str_char(sid, i); i = i + 1; }
    g_stradr[sid] = a;
    return a;
}

int region_buf(int r) {
    if (g_region_buf[r] == 0) {
        g_region_off[r] = data_alloc(8, 8);
        int cap = g_region_cap[r];
        if (cap < 0) { cap = 0; }
        g_region_buf[r] = bss_alloc(cap, 16);
    }
    return g_region_buf[r];
}

int region_off(int r) { region_buf(r); return g_region_off[r]; }

int glob_addr(int g);

// writes the constant value of e (type of e: et) converted to type t at addr
void const_init(int addr, int t, int e, int gname_node) {
    int k = g_node[e].kind;
    t = base_scalar(t);
    int tk = g_ty[t].kind;
    if (k == E_INITLIST) {
        if (tk == TY_STRUCT && g_sdef[g_ty[t].n].kind == SD_STRUCT) {
            int sd = g_ty[t].n;
            layout_struct(sd);
            int next = 0;
            int i = 0;
            while (i < g_node[e].nlist) {
                int el = g_list[g_node[e].list + i];
                int val = el;
                int slot = next;
                if (g_node[el].kind == E_DESIG) { val = g_node[el].a; slot = g_node[el].d; }
                next = slot + 1;
                if (slot < g_sdef[sd].nfield) {
                    int fi = g_sdef[sd].field0 + slot;
                    const_init(addr + g_field[fi].off, g_field[fi].ty, val, gname_node);
                }
                i = i + 1;
            }
            return;
        }
        if (tk == TY_ARRAY) {
            int et = g_ty[t].base;
            int es = ty_size(et);
            int next = 0;
            int i = 0;
            while (i < g_node[e].nlist) {
                int el = g_list[g_node[e].list + i];
                int val = el;
                int slot = next;
                if (g_node[el].kind == E_DESIG) { val = g_node[el].a; slot = g_node[el].d; }
                next = slot + 1;
                if (slot < g_ty[t].n || g_ty[t].n < 0) { const_init(addr + slot * es, et, val, gname_node); }
                i = i + 1;
            }
            return;
        }
    }
    if (k == E_STR) {
        if (tk == TY_ARRAY) {
            int sid = g_node[e].sid;
            int n = g_str_len[sid];
            int i = 0;
            while (i < n && i < g_ty[t].n) { g_data[addr - DATA_BASE + i] = (unsigned char)str_char(sid, i); i = i + 1; }
            return;
        }
        data_put(addr, 4, (long)str_addr(g_node[e].sid));
        return;
    }
    if (k == E_NULL) { data_put(addr, ty_size(t), 0L); return; }
    if (k == E_ADDR) {
        int o = g_node[e].a;
        if (g_node[o].kind == E_IDENT && g_node[o].c == SYM_GLOBAL) {
            data_put(addr, 4, (long)glob_addr(g_node[o].d));
            return;
        }
        if (g_node[o].kind == E_IDENT && g_node[o].c == SYM_FUNC) {
            data_put(addr, 4, (long)table_index(g_node[o].d));
            return;
        }
    }
    if (k == E_IDENT && g_node[e].c == SYM_FUNC) {
        data_put(addr, 4, (long)table_index(g_node[e].d));
        return;
    }
    if (k == E_IDENT && g_node[e].c == SYM_GLOBAL && g_ty[g_glob[g_node[e].d].ty].kind == TY_ARRAY && (tk == TY_PTR || tk == TY_REF)) {
        data_put(addr, 4, (long)glob_addr(g_node[e].d));
        return;
    }
    if (tk == TY_F32 || tk == TY_F64) {
        int neg = 0;
        int f = e;
        if (g_node[f].kind == E_UNARY && g_node[f].op == T_MINUS) { neg = 1; f = g_node[f].a; }
        long bits = 0L;
        int ok = 0;
        if (g_node[f].kind == E_FLOAT) {
            bits = g_node[f].ival;
            if (neg == 1) { bits = bits ^ (1L << 63L); }
            ok = 1;
        } else if (sema_const_int(e) == 1) {
            bits = f64_bits((double)g_cval);
            ok = 1;
        }
        if (ok == 1) {
            if (tk == TY_F64) { data_put(addr, 8, bits); }
            else { data_put(addr, 4, f32_bits_of_f64(bits)); }
            return;
        }
    }
    if (tk == TY_OPT && g_ty[g_ty[t].base].kind != TY_REF) {
        // T → ?T, null → ?T
        if (k == E_NULL) { return; }
        const_init(addr, g_ty[t].base, e, gname_node);
        data_put(addr + ty_size(g_ty[t].base), 1, 1L);
        return;
    }
    if (sema_const_int(e) == 1) {
        long v = g_cval;
        if (tk == TY_BOOL) { v = v & 1L; }
        data_put(addr, ty_size(t), v);
        return;
    }
    msg_reset();
    msg_lit("global initializer for '");
    msg_str(g_node[gname_node].sid);
    msg_lit("' is not a compile-time constant expression");
    error_node(gname_node);
}

int glob_addr(int g) {
    if (g_glob[g].addr != 0) { return g_glob[g].addr; }
    int t = g_glob[g].ty;
    int n = g_glob[g].node;
    int a = 0;
    if (g_node[n].a != 0) { a = data_alloc(ty_size(t), ty_align(t)); }
    else { a = bss_alloc(ty_size(t), ty_align(t)); }
    g_glob[g].addr = a;
    if (g_node[n].a != 0) { const_init(a, t, g_node[n].a, n); }
    return a;
}

// ── defer / loops ────────────────────────────────────────────────────────────
void gen_stmt(int s);

// runs defers down to height h (exclusive), innermost first; err: include errdefers
void run_defers(int h, int err) {
    int i = g_ndf - 1;
    while (i >= h) {
        if (g_df_err[i] == 0 || err == 1) {
            // a defer body runs with the defer stack truncated to below it
            int saved = g_ndf;
            g_ndf = i;
            gen_stmt(g_df_node[i]);
            g_ndf = saved;
        }
        i = i - 1;
    }
}

void emit_epilogue() {
    emit(OP_LOCAL_GET); emitu((long)g_oldsp);
    emit(OP_GLOBAL_SET); emitu(0L);
}

// return with the value (if any) already on the stack / in sret
void emit_return_now() {
    emit_epilogue();
    emit(OP_RETURN);
}

// ── statements ───────────────────────────────────────────────────────────────
void gen_block_list(int s) {
    int i = 0;
    while (i < g_node[s].nlist) { gen_stmt(g_list[g_node[s].list + i]); i = i + 1; }
}

void gen_block(int s) {
    int h = g_ndf;
    gen_block_list(s);
    run_defers(h, 0);
    g_ndf = h;
}

// storage of a local, set up on first sight
void setup_local(int l) {
    if (g_lst_kind[l] != 0) { return; }
    int t = g_local[l].ty;
    int fl = g_local[l].flags;
    if ((fl & LF_STATIC) != 0) {
        g_lst_kind[l] = LS_STATIC;
        int vn = g_local[l].node;
        if (vn != 0 && g_node[vn].a != 0) { g_lst_val[l] = data_alloc(ty_size(t), ty_align(t)); }
        else { g_lst_val[l] = bss_alloc(ty_size(t), ty_align(t)); }
        if (vn != 0 && g_node[vn].a != 0) { const_init(g_lst_val[l], t, g_node[vn].a, vn); }
        return;
    }
    if (is_agg(t) == 1 || (fl & LF_ADDR) != 0) {
        g_lst_kind[l] = LS_FRAME;
        g_lst_val[l] = alloc_frame(ty_size(t), ty_align(t));
        return;
    }
    g_lst_kind[l] = LS_WASM;
    int vt = vt_of(t);
    if (vt == 0) { vt = VT_I32; }
    g_lst_val[l] = new_wl(vt);
}

// pushes the address of local l (frame/static/indirect storage)
void emit_local_addr(int l) {
    setup_local(l);
    int k = g_lst_kind[l];
    if (k == LS_FRAME) { emit_frame_addr(g_lst_val[l]); return; }
    if (k == LS_STATIC) { emit_i32(g_lst_val[l]); return; }
    if (k == LS_INDIRECT) { emit(OP_LOCAL_GET); emitu((long)g_lst_val[l]); return; }
    fatal("address of a register local");
}

void gen_var(int s) {
    int l = g_node[s].d;
    setup_local(l);
    int t = g_local[l].ty;
    int init = g_node[s].a;
    if ((g_local[l].flags & LF_STATIC) != 0) { return; }
    if (init == 0) {
        // aggregates are zeroed (C leaves them indeterminate)
        if (g_lst_kind[l] == LS_FRAME) {
            emit_local_addr(l);
            emit_i32(0);
            emit_i32(ty_size(t));
            emit_fc(11); emit(0);
        } else if (g_lst_kind[l] == LS_WASM) {
            emit_zero(t);
            emit(OP_LOCAL_SET); emitu((long)g_lst_val[l]);
        }
        return;
    }
    if (g_lst_kind[l] == LS_WASM) {
        gen_value_as(init, t);
        emit(OP_LOCAL_SET); emitu((long)g_lst_val[l]);
        return;
    }
    gen_init_at_local(l, t, init);
}

// initializes the memory of local l from e (handles initializer lists in place)
void gen_init_at_local(int l, int t, int e) {
    int a = new_wl(VT_I32);
    emit_local_addr(l);
    emit(OP_LOCAL_SET); emitu((long)a);
    gen_init_at(a, 0, t, e);
}

// *(addr_local + off) = e   (as type t)
void gen_init_at(int al, int off, int t, int e) {
    int k = g_node[e].kind;
    int bt = base_scalar(t);
    if (k == E_INITLIST && (g_ty[bt].kind == TY_STRUCT || g_ty[bt].kind == TY_ARRAY)) {
        // zero, then the given elements
        emit(OP_LOCAL_GET); emitu((long)al);
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        emit_i32(0);
        emit_i32(ty_size(t));
        emit_fc(11); emit(0);
        if (g_ty[bt].kind == TY_STRUCT) {
            int sd = g_ty[bt].n;
            layout_struct(sd);
            int next = 0;
            int i = 0;
            while (i < g_node[e].nlist) {
                int el = g_list[g_node[e].list + i];
                int val = el;
                int slot = next;
                if (g_node[el].kind == E_DESIG) { val = g_node[el].a; slot = g_node[el].d; }
                next = slot + 1;
                if (slot < g_sdef[sd].nfield) {
                    int fi = g_sdef[sd].field0 + slot;
                    gen_init_at(al, off + g_field[fi].off, g_field[fi].ty, val);
                }
                i = i + 1;
            }
        } else {
            int et = g_ty[bt].base;
            int es = ty_size(et);
            int next = 0;
            int i = 0;
            while (i < g_node[e].nlist) {
                int el = g_list[g_node[e].list + i];
                int val = el;
                int slot = next;
                if (g_node[el].kind == E_DESIG) { val = g_node[el].a; slot = g_node[el].d; }
                next = slot + 1;
                if (slot < g_ty[bt].n || g_ty[bt].n < 0) { gen_init_at(al, off + slot * es, et, val); }
                i = i + 1;
            }
        }
        return;
    }
    if (k == E_STR && g_ty[bt].kind == TY_ARRAY) {
        // char buf[N] = "text": copy the bytes (and the NUL if it fits)
        int sid = g_node[e].sid;
        int n = g_str_len[sid] + 1;
        if (n > g_ty[bt].n) { n = g_ty[bt].n; }
        emit(OP_LOCAL_GET); emitu((long)al);
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        emit_i32(0);
        emit_i32(ty_size(t));
        emit_fc(11); emit(0);
        emit(OP_LOCAL_GET); emitu((long)al);
        if (off != 0) { emit_i32(off); emit(OP_I32_ADD); }
        emit_i32(str_addr(sid));
        emit_i32(n);
        emit_fc(10); emit(0); emit(0);
        return;
    }
    emit(OP_LOCAL_GET); emitu((long)al);
    gen_value_as(e, t);
    emit_store(t, off);
}

// g_lp_ismatch: 0 loop, 1 switch (a break target, not a continue target)
// (2 = match: 'break' passes through it, unless no loop encloses the match)
int find_loop(int label, int is_cont) {
    int i = g_nlp - 1;
    int match_at = -1;
    while (i >= 0) {
        if (label != 0) {
            if (g_lp_label[i] == label) { return i; }
        } else if (g_lp_ismatch[i] == 0 || (is_cont == 0 && g_lp_ismatch[i] == 1)) { return i; }
        if (g_lp_ismatch[i] == 2 && match_at < 0) { match_at = i; }
        i = i - 1;
    }
    if (is_cont == 0 && label == 0) { return match_at; }
    return -1;
}

void push_loop(int brk, int cont, int label, int ismatch) {
    if (g_nlp >= 256) { fatal("loop nesting"); return; }
    g_lp_brk[g_nlp] = brk;
    g_lp_cont[g_nlp] = cont;
    g_lp_label[g_nlp] = label;
    g_lp_defer[g_nlp] = g_ndf;
    g_lp_ismatch[g_nlp] = ismatch;
    g_nlp = g_nlp + 1;
}

void emit_br_to(int depth) {
    emit(OP_BR);
    emitu((long)(g_cdepth - depth));
}

void ctl_open(int op) {
    emit(op);
    emit(64);         // empty block type
    g_cdepth = g_cdepth + 1;
}

void ctl_close() {
    emit(OP_END);
    g_cdepth = g_cdepth - 1;
}

// ── goto ─────────────────────────────────────────────────────────────────────
// A function with goto targets runs its body inside a dispatch loop. 'goto L'
// sets the target local to L's id and branches to the loop's head; the body
// then runs in "skip mode": each statement whose subtree doesn't hold the
// target is skipped, one that does is entered without evaluating its
// condition, and the label itself clears the target. Ids are assigned in
// preorder, so the labels under any statement form a contiguous id range.
int g_gt_mode = 0;
int g_gt_local = 0;
int g_gt_disp = 0;
int g_gl_sid[256];
int g_gl_node[256];
int g_gl_n = 0;
int g_gr_node[1024];
int g_gr_lo[1024];
int g_gr_hi[1024];
int g_gr_n = 0;
int g_rlo = 0;
int g_rhi = 0;

// assigns label ids under s (goto targets: as Sema's collectLabels)
void gt_assign(int s) {
    if (s == 0) { return; }
    int k = g_node[s].kind;
    int first = g_gl_n + 1;
    if (k == S_LABEL) {
        if (g_gl_n < 256) { g_gl_sid[g_gl_n] = g_node[s].sid; g_gl_node[g_gl_n] = s; g_gl_n = g_gl_n + 1; }
        gt_assign(g_node[s].a);
    } else if (k == S_BLOCK) {
        int i = 0;
        while (i < g_node[s].nlist) { gt_assign(g_list[g_node[s].list + i]); i = i + 1; }
    } else if (k == S_IF) {
        gt_assign(g_node[s].b);
        gt_assign(g_node[s].c);
    } else if (k == S_WHILE || k == S_DO) {
        gt_assign(g_node[s].b);
    } else if (k == S_FOR) {
        gt_assign(g_node[s].d);
    } else if (k == S_UNSAFE) {
        gt_assign(g_node[s].a);
    } else if (k == S_SWITCH) {
        int i = 0;
        while (i < g_node[s].nlist) { gt_assign(g_node[g_list[g_node[s].list + i]].b); i = i + 1; }
    } else {
        return;
    }
    if (g_gl_n + 1 > first && g_gr_n < 1024) {
        g_gr_node[g_gr_n] = s;
        g_gr_lo[g_gr_n] = first;
        g_gr_hi[g_gr_n] = g_gl_n;
        g_gr_n = g_gr_n + 1;
    }
}

// 1 (range in g_rlo..g_rhi) if s holds goto targets
int gt_range(int s) {
    if (g_gt_mode == 0 || s == 0) { return 0; }
    int i = 0;
    while (i < g_gr_n) {
        if (g_gr_node[i] == s) { g_rlo = g_gr_lo[i]; g_rhi = g_gr_hi[i]; return 1; }
        i = i + 1;
    }
    return 0;
}

int gt_label_id(int s) {
    int i = 0;
    while (i < g_gl_n) { if (g_gl_node[i] == s) { return i + 1; } i = i + 1; }
    return 0;
}

int gt_id_of_name(int sid) {
    int i = 0;
    while (i < g_gl_n) { if (g_gl_sid[i] == sid) { return i + 1; } i = i + 1; }
    return 0;
}

void gt_get() { emit(OP_LOCAL_GET); emitu((long)g_gt_local); }

// pushes (target in s's range) as an i32; 0 if s holds no targets
void gt_in_range(int s) {
    if (gt_range(s) == 0) { emit_i32(0); return; }
    gt_get(); emit_i32(g_rlo); emit(OP_I32_GE_U);
    gt_get(); emit_i32(g_rhi); emit(OP_I32_LE_U);
    emit(OP_I32_AND);
}

// opens 'if (no target)' (skip mode off)
void gt_open_normal() {
    gt_get();
    emit(OP_I32_EQZ);
    ctl_open(OP_IF);
}

void gen_return(int s) {
    int v = g_node[s].a;
    int rt = g_gen_ret;
    if (v != 0 && vt_of(rt) != 0 && is_agg(rt) == 0) {
        gen_value_as(v, rt);
        if (g_ndf > 0) {
            int tmp = new_wl(vt_of(rt));
            emit(OP_LOCAL_SET); emitu((long)tmp);
            run_defers(0, 0);
            emit(OP_LOCAL_GET); emitu((long)tmp);
        }
        emit_return_now();
        return;
    }
    if (v != 0 && is_agg(rt) == 1) {
        emit(OP_LOCAL_GET); emitu((long)g_sret);
        gen_value_as(v, rt);
        emit_store(rt, 0);
        run_defers(0, 0);
        emit_return_now();
        return;
    }
    if (v != 0) {
        // void function returning a void expression
        gen_expr_discard(v);
    }
    run_defers(0, 0);
    emit_return_now();
}

// the error exit of 'try': errdefers + defers, then return null
void gen_try_fail() {
    run_defers(0, 1);
    int rt = g_gen_ret;
    if (is_agg(rt) == 1) {
        emit(OP_LOCAL_GET); emitu((long)g_sret);
        emit_i32(0);
        emit_i32(ty_size(rt));
        emit_fc(11); emit(0);
        emit_return_now();
        return;
    }
    if (vt_of(rt) != 0) { emit_zero(rt); }
    emit_return_now();
}

void gen_if(int s) {
    if (gt_range(s) == 1) {
        // skip mode: enter the branch that holds the target
        gt_get();
        emit(OP_I32_EQZ);
        emit(OP_IF); emit(127);
        g_cdepth = g_cdepth + 1;
        gen_cond(g_node[s].a);
        emit(OP_ELSE);
        gt_in_range(g_node[s].b);
        ctl_close();
    } else {
        gen_cond(g_node[s].a);
    }
    ctl_open(OP_IF);
    gen_stmt(g_node[s].b);
    if (g_node[s].c != 0) {
        emit(OP_ELSE);
        gen_stmt(g_node[s].c);
    }
    ctl_close();
}

void gen_while(int s) {
    ctl_open(OP_BLOCK);
    int brk = g_cdepth;
    ctl_open(OP_LOOP);
    int top = g_cdepth;
    int skip = gt_range(s);
    if (skip == 1) { gt_open_normal(); }
    gen_cond(g_node[s].a);
    emit(OP_I32_EQZ);
    emit(OP_BR_IF); emitu((long)(g_cdepth - brk));
    if (skip == 1) { ctl_close(); }
    push_loop(brk, top, g_node[s].sid, 0);
    gen_stmt(g_node[s].b);
    g_nlp = g_nlp - 1;
    emit_br_to(top);
    ctl_close();
    ctl_close();
}

void gen_do(int s) {
    ctl_open(OP_BLOCK);
    int brk = g_cdepth;
    ctl_open(OP_LOOP);
    int top = g_cdepth;
    ctl_open(OP_BLOCK);
    int cont = g_cdepth;
    push_loop(brk, cont, g_node[s].sid, 0);
    gen_stmt(g_node[s].b);
    g_nlp = g_nlp - 1;
    ctl_close();
    gen_cond(g_node[s].a);
    emit(OP_BR_IF); emitu((long)(g_cdepth - top));
    ctl_close();
    ctl_close();
}

void gen_for(int s) {
    int h = g_ndf;
    if (g_node[s].a != 0) { gen_stmt(g_node[s].a); }
    ctl_open(OP_BLOCK);
    int brk = g_cdepth;
    ctl_open(OP_LOOP);
    int top = g_cdepth;
    if (g_node[s].b != 0) {
        int skip = gt_range(s);
        if (skip == 1) { gt_open_normal(); }
        gen_cond(g_node[s].b);
        emit(OP_I32_EQZ);
        emit(OP_BR_IF); emitu((long)(g_cdepth - brk));
        if (skip == 1) { ctl_close(); }
    }
    ctl_open(OP_BLOCK);
    int cont = g_cdepth;
    push_loop(brk, cont, g_node[s].sid, 0);
    gen_stmt(g_node[s].d);
    g_nlp = g_nlp - 1;
    ctl_close();
    if (g_node[s].c != 0) { gen_expr_discard(g_node[s].c); }
    emit_br_to(top);
    ctl_close();
    ctl_close();
    run_defers(h, 0);
    g_ndf = h;
}

void gen_break(int s, int is_cont) {
    int i = find_loop(g_node[s].sid, is_cont);
    if (i < 0) { emit(OP_UNREACHABLE); return; }
    run_defers(g_lp_defer[i], 0);
    if (is_cont == 1) { emit_br_to(g_lp_cont[i]); }
    else { emit_br_to(g_lp_brk[i]); }
}

void gen_switch(int s) {
    int t = g_node[g_node[s].a].ty;
    int is64 = vt_of(t) == VT_I64;
    int sv = new_wl(VT_I32);
    if (is64 == 1) { sv = new_wl(VT_I64); }
    int skip = gt_range(s);
    if (skip == 1) { gt_open_normal(); }
    gen_value_as(g_node[s].a, t);
    emit(OP_LOCAL_SET); emitu((long)sv);
    if (skip == 1) { ctl_close(); }
    int n = g_node[s].nlist;
    // block structure: block(end) block(case n-1) ... block(case 0) dispatch end body0 end body1 ...
    ctl_open(OP_BLOCK);
    int brk = g_cdepth;
    int i = 0;
    while (i < n) { ctl_open(OP_BLOCK); i = i + 1; }
    // dispatch: case i's body starts after the end of block depth (brk + 1 + n - 1 - i)... see below
    int def = -1;
    if (gt_range(s) == 1) {
        // skip mode: straight into the case that holds the target
        i = 0;
        while (i < n) {
            int c = g_list[g_node[s].list + i];
            gt_in_range(g_node[c].b);
            emit(OP_BR_IF); emitu((long)i);
            i = i + 1;
        }
    }
    i = 0;
    while (i < n) {
        int c = g_list[g_node[s].list + i];
        if ((g_node[c].flags & NF_DEFAULT) != 0) { def = i; }
        int j = 0;
        while (j < g_node[c].nlist) {
            long v = g_node[g_list[g_node[c].list + j]].ival;
            emit(OP_LOCAL_GET); emitu((long)sv);
            if (is64 == 1) { emit_i64(v); emit(OP_I64_EQ); }
            else { emit_i32((int)v); emit(OP_I32_EQ); }
            // case i's body follows the end of the block opened i-th from the inside
            emit(OP_BR_IF); emitu((long)i);
            j = j + 1;
        }
        i = i + 1;
    }
    if (def >= 0) { emit(OP_BR); emitu((long)def); }
    else { emit(OP_BR); emitu((long)n); }
    push_loop(brk, -1, 0, 1);
    i = 0;
    while (i < n) {
        ctl_close();
        int c = g_list[g_node[s].list + i];
        int h = g_ndf;
        gen_block_list(g_node[c].b);
        run_defers(h, 0);
        g_ndf = h;
        i = i + 1;
    }
    g_nlp = g_nlp - 1;
    ctl_close();
}

void gen_stmt_inner(int s);

void gen_stmt(int s) {
    if (s == 0) { return; }
    int k = g_node[s].kind;
    if (g_gt_mode == 1 && k != S_BLOCK && k != S_MULTI && k != S_UNSAFE && k != S_DEFER && k != S_EMPTY &&
        k != S_STATIC_ASSERT) {
        if (k == S_LABEL) {
            int id = gt_label_id(s);
            if (id != 0) {
                gt_get(); emit_i32(id); emit(OP_I32_EQ);
                ctl_open(OP_IF);
                emit_i32(0); emit(OP_LOCAL_SET); emitu((long)g_gt_local);
                ctl_close();
            }
            gen_stmt(g_node[s].a);
            return;
        }
        if (gt_range(s) == 0) {
            // skipped while looking for the target
            gt_open_normal();
            gen_stmt_inner(s);
            ctl_close();
            return;
        }
    }
    gen_stmt_inner(s);
}

void gen_stmt_inner(int s) {
    int k = g_node[s].kind;
    if (k == S_LABEL) { gen_stmt(g_node[s].a); return; }
    if (k == S_GOTO) {
        int id = gt_id_of_name(g_node[s].sid);
        emit_i32(id); emit(OP_LOCAL_SET); emitu((long)g_gt_local);
        emit_br_to(g_gt_disp);
        return;
    }
    if (k == S_BLOCK) { gen_block(s); return; }
    if (k == S_MULTI) { gen_block_list(s); return; }
    if (k == S_VAR) { gen_var(s); return; }
    if (k == S_EXPR) { gen_expr_discard(g_node[s].a); return; }
    if (k == S_IF) { gen_if(s); return; }
    if (k == S_IFCONST) {
        if (g_node[s].op == 1) { gen_stmt(g_node[s].b); }
        else if (g_node[s].op == 2) { gen_stmt(g_node[s].c); }
        return;
    }
    if (k == S_WHILE) { gen_while(s); return; }
    if (k == S_DO) { gen_do(s); return; }
    if (k == S_FOR) { gen_for(s); return; }
    if (k == S_RETURN) { gen_return(s); return; }
    if (k == S_BREAK) { gen_break(s, 0); return; }
    if (k == S_CONTINUE) { gen_break(s, 1); return; }
    if (k == S_UNSAFE) { gen_stmt(g_node[s].a); return; }
    if (k == S_DEFER) {
        if (g_ndf >= 1024) { fatal("defer depth"); return; }
        g_df_node[g_ndf] = g_node[s].a;
        g_df_err[g_ndf] = 0;
        if ((g_node[s].flags & NF_ERR) != 0) { g_df_err[g_ndf] = 1; }
        g_ndf = g_ndf + 1;
        return;
    }
    if (k == S_MATCH) { gen_match(s, 0); return; }
    if (k == S_SWITCH) { gen_switch(s); return; }
    // S_EMPTY, S_STATIC_ASSERT: nothing
}

// ── functions ────────────────────────────────────────────────────────────────
void gen_expr_discard(int e);

void gen_function(int f) {
    int fnode = g_func[f].node;
    int ft = g_func[f].ty;
    g_gen_fn = f;
    g_gen_ret = g_ty[ft].base;
    g_code_len = 0;
    g_wl_n = 0;
    g_frame = 0;
    g_cdepth = 0;
    g_nlp = 0;
    g_ndf = 0;
    g_sret = -1;
    g_vaarg = -1;
    int i = 0;
    while (i < g_func[f].nloc) { g_lst_kind[g_func[f].loc0 + i] = 0; i = i + 1; }
    // parameters
    if (is_agg(g_gen_ret) == 1) { g_sret = new_wl(VT_I32); }
    int np = g_ty[ft].nlist;
    int pw0 = g_wl_n;
    i = 0;
    while (i < np) {
        int pt = g_tyl[g_ty[ft].list + i];
        int vt = vt_of(pt);
        if (is_agg(pt) == 1 || vt == 0) { vt = VT_I32; }
        new_wl(vt);
        i = i + 1;
    }
    if ((g_ty[ft].flags & TYF_VARIADIC) != 0) { g_vaarg = new_wl(VT_I32); }
    int nparams = g_wl_n;
    g_fp = new_wl(VT_I32);
    g_oldsp = new_wl(VT_I32);
    // parameter locals: the first np locals of the function
    int pcount = 0;
    i = 0;
    while (i < g_func[f].nloc && pcount < np) {
        int l = g_func[f].loc0 + i;
        if ((g_local[l].flags & LF_PARAM) != 0) {
            int pt = g_local[l].ty;
            int wl = pw0 + pcount;
            if (is_agg(pt) == 1) { g_lst_kind[l] = LS_INDIRECT; g_lst_val[l] = wl; }
            else if ((g_local[l].flags & LF_ADDR) != 0) { g_lst_kind[l] = LS_FRAME; g_lst_val[l] = alloc_frame(ty_size(pt), ty_align(pt)); }
            else { g_lst_kind[l] = LS_WASM; g_lst_val[l] = wl; }
            pcount = pcount + 1;
        }
        i = i + 1;
    }
    // body
    pcount = 0;
    i = 0;
    while (i < g_func[f].nloc && pcount < np) {
        int l = g_func[f].loc0 + i;
        if ((g_local[l].flags & LF_PARAM) != 0) {
            if (g_lst_kind[l] == LS_FRAME) {
                emit_frame_addr(g_lst_val[l]);
                emit(OP_LOCAL_GET); emitu((long)(pw0 + pcount));
                emit_store(g_local[l].ty, 0);
            }
            pcount = pcount + 1;
        }
        i = i + 1;
    }
    g_gt_mode = 0;
    g_gl_n = 0;
    g_gr_n = 0;
    gt_assign(g_node[fnode].a);
    if (g_gl_n > 0) {
        g_gt_mode = 1;
        g_gt_local = new_wl(VT_I32);
        ctl_open(OP_LOOP);
        g_gt_disp = g_cdepth;
        gen_block(g_node[fnode].a);
        ctl_close();
        g_gt_mode = 0;
    } else {
        gen_block(g_node[fnode].a);
    }
    // falling off the end
    if (vt_of(g_gen_ret) != 0 && is_agg(g_gen_ret) == 0) { emit_zero(g_gen_ret); }
    emit_epilogue();
    emit(OP_END);
    // assemble: locals, prologue, body
    int body_len = g_code_len;
    if (g_codesec_len + body_len + 4096 >= OUT_MAX) { fatal("code size"); return; }
    int start = g_codesec_len;
    // local declarations (runs of equal types) after the params
    int nruns = 0;
    i = nparams;
    while (i < g_wl_n) {
        int j = i;
        while (j < g_wl_n && g_wl_ty[j] == g_wl_ty[i]) { j = j + 1; }
        nruns = nruns + 1;
        i = j;
    }
    cs_u((long)nruns);
    i = nparams;
    while (i < g_wl_n) {
        int j = i;
        while (j < g_wl_n && g_wl_ty[j] == g_wl_ty[i]) { j = j + 1; }
        cs_u((long)(j - i));
        cs_b(g_wl_ty[i]);
        i = j;
    }
    // prologue: oldsp = sp; fp = sp - frame; sp = fp; stack check
    int frame = align_to(g_frame, 16);
    cs_b(OP_GLOBAL_GET); cs_u(0L);
    cs_b(OP_LOCAL_TEE); cs_u((long)g_oldsp);
    cs_b(OP_I32_CONST); cs_s((long)frame);
    cs_b(OP_I32_SUB);
    cs_b(OP_LOCAL_TEE); cs_u((long)g_fp);
    cs_b(OP_GLOBAL_SET); cs_u(0L);
    cs_b(OP_LOCAL_GET); cs_u((long)g_fp);
    cs_b(OP_GLOBAL_GET); cs_u(1L);
    cs_b(OP_I32_LT_U);
    cs_b(OP_IF); cs_b(64);
    if (g_rt_trap != 0) {
        // room for the trap handler's own frames: the deepest 64 KB of the
        // stack is given up (the program is exiting)
        cs_b(OP_GLOBAL_GET); cs_u(1L);
        cs_b(OP_I32_CONST); cs_s(65536L);
        cs_b(OP_I32_ADD);
        cs_b(OP_GLOBAL_SET); cs_u(0L);
        cs_b(OP_I32_CONST); cs_s(3L);
        cs_b(OP_CALL); cs_u((long)func_index(g_rt_trap));
    }
    cs_b(OP_UNREACHABLE);
    cs_b(OP_END);
    i = 0;
    while (i < body_len) { cs_b((int)g_code[i]); i = i + 1; }
    int k = 0;
    while (k < g_nwfunc) {
        if (g_wfunc[k] == f) {
            g_wfunc_body[k] = start;
            g_wfunc_blen[k] = g_codesec_len - start;
        }
        k = k + 1;
    }
}

void cs_b(int b) {
    if (g_codesec_len < OUT_MAX) { g_codesec[g_codesec_len] = (unsigned char)b; g_codesec_len = g_codesec_len + 1; }
}

void cs_u(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = (long)((unsigned long)v >> 7UL);
        if (v != 0L) { cs_b(b | 128); } else { cs_b(b); break; }
    }
}

void cs_s(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = v >> 7L;
        int sign = b & 64;
        if ((v == 0L && sign == 0) || (v == -1L && sign != 0)) { cs_b(b); break; }
        cs_b(b | 128);
    }
}

// ── module ───────────────────────────────────────────────────────────────────
int find_func_by_name(const char* s) {
    int sid = intern_lit(s);
    if (g_gkind[sid] == SYM_FUNC) { return g_gidx[sid]; }
    return 0;
}

void add_import(const char* name, int f) {
    if (g_nimports >= 16) { return; }
    g_import_name[g_nimports] = intern_lit(name);
    int t = 0;
    if (f != 0) { t = sig_of_fn_type(g_func[f].ty); g_func[f].widx = g_nimports; }
    g_import_type[g_nimports] = t;
    g_nimports = g_nimports + 1;
}

void gen_module() {
    g_data_len = 0;
    // host imports (declared by the runtime prelude)
    int i = 1;
    while (i <= g_nfunc) {
        if (g_node[g_func[i].node].a == 0 && is_host_import(i) == 1 && g_func[i].ty != 0 && g_func[i].owner == 0) {
            add_import_sid(g_func[i].sid, i);
        }
        i = i + 1;
    }
    g_rt_trap = find_func_by_name("__rt_trap");
    int nexp = 0;
    i = 1;
    while (i <= g_nfunc) {
        if ((g_func[i].flags & NF_EXPORT) != 0 && g_node[g_func[i].node].a != 0 &&
            g_tok[g_node[g_func[i].node].tok].file != g_rt_file) { nexp = nexp + 1; }
        i = i + 1;
    }
    g_nexport = 0;
    if (g_main_fn == 0) {
        // a library module (exports only) needs no main
        if (nexp == 0) {
            msg_reset();
            msg_lit("no 'main' function");
            diag_at(0, 0, 0, 1);
            return;
        }
    } else {
        func_index(g_main_fn);
        g_export_sid[0] = S_main;
        g_export_func[0] = g_main_fn;
        g_nexport = 1;
    }
    // exported functions
    i = 1;
    while (i <= g_nfunc) {
        if ((g_func[i].flags & NF_EXPORT) != 0 && g_node[g_func[i].node].a != 0 && g_nexport < 256) {
            func_index(i);
            g_export_sid[g_nexport] = (int)g_node[g_func[i].node].ival;
            g_export_func[g_nexport] = i;
            g_nexport = g_nexport + 1;
        }
        i = i + 1;
    }
    if (g_rt_trap != 0) { func_index(g_rt_trap); }
    while (g_emit_pos < g_nemit) {
        int f = g_emit_queue[g_emit_pos];
        g_emit_pos = g_emit_pos + 1;
        gen_function(f);
        if (g_nerrors > 0) { return; }
    }
    // memory layout: data, bss, then the stack, then the heap
    int data_end = DATA_BASE + DATA_MAX + align_to(g_bss_len, 16);
    g_stack_limit = data_end;
    g_stack_top = data_end + STACK_SIZE;
    g_heap_base = g_stack_top;
    wasm_assemble();
}

void add_import_sid(int sid, int f) {
    if (g_nimports >= 16) { return; }
    g_import_name[g_nimports] = sid;
    g_import_type[g_nimports] = sig_of_fn_type(g_func[f].ty);
    g_func[f].widx = g_nimports;
    g_nimports = g_nimports + 1;
}
