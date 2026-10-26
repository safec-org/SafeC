// types.sc — hash-consed type table, sizes and layout (wasm32 data model:
// pointers/references 4 bytes, long 8 bytes).
#include "scc.h"

struct Ty g_ty[TY_MAX];
int g_nty = 0;
int g_tyl[TYL_MAX];
int g_ntyl = 0;
int g_tyhash[TYHASH_SIZE];

struct SDef g_sdef[NSTRUCT_MAX];
int g_nsdef = 0;
struct Field g_field[NFIELD_MAX];
int g_nfield = 0;

int tyl_begin() { return g_ntyl; }

void tyl_push(int t) {
    if (g_ntyl >= TYL_MAX) { fatal("type lists"); return; }
    g_tyl[g_ntyl] = t;
    g_ntyl = g_ntyl + 1;
}

int ty_make(int kind, int base, int n, int sid, int flags, int mark) {
    int nl = g_ntyl - mark;
    if (mark < 0) { nl = 0; mark = g_ntyl; }
    unsigned int h = (unsigned int)kind *| 31U +| (unsigned int)base;
    h = h *| 31U +| (unsigned int)n;
    h = h *| 31U +| (unsigned int)sid;
    h = h *| 31U +| (unsigned int)flags;
    int i = 0;
    while (i < nl) { h = h *| 31U +| (unsigned int)g_tyl[mark + i]; i = i + 1; }
    int b = (int)(h & (unsigned int)(TYHASH_SIZE - 1));
    int t = g_tyhash[b];
    while (t != 0) {
        if (g_ty[t].kind == kind && g_ty[t].base == base && g_ty[t].n == n &&
            g_ty[t].sid == sid && g_ty[t].flags == flags && g_ty[t].nlist == nl) {
            int same = 1;
            i = 0;
            while (i < nl) {
                if (g_tyl[g_ty[t].list + i] != g_tyl[mark + i]) { same = 0; break; }
                i = i + 1;
            }
            if (same == 1) { g_ntyl = mark; return t; }
        }
        t = g_ty[t].hnext;
    }
    if (g_nty + 1 >= TY_MAX) { fatal("types"); return TY_ERROR; }
    g_nty = g_nty + 1;
    t = g_nty;
    g_ty[t].kind = kind;
    g_ty[t].base = base;
    g_ty[t].n = n;
    g_ty[t].sid = sid;
    g_ty[t].flags = flags;
    g_ty[t].list = mark;
    g_ty[t].nlist = nl;
    g_ty[t].hnext = g_tyhash[b];
    g_tyhash[b] = t;
    return t;
}

void types_init() {
    // primitive kinds are their own type ids 1..14
    int k = 1;
    while (k <= TY_F64) {
        int t = ty_make(k, 0, 0, 0, 0, -1);
        if (t != k) { fatal("primitive type ids"); }
        k = k + 1;
    }
}

int ty_prim(int kind) { return kind; }
int ty_kind(int t) { return g_ty[t].kind; }
int ty_ptr(int base, int flags) { return ty_make(TY_PTR, base, 0, 0, flags, -1); }
int ty_ref(int base, int rgn, int arn, int flags) { return ty_make(TY_REF, base, rgn, arn, flags, -1); }
int ty_array(int base, int n) { return ty_make(TY_ARRAY, base, n, 0, 0, -1); }
int ty_slice(int base) { return ty_make(TY_SLICE, base, 0, 0, 0, -1); }
int ty_opt(int base) { return ty_make(TY_OPT, base, 0, 0, 0, -1); }
int ty_fn(int ret, int mark, int variadic) {
    int f = 0;
    if (variadic != 0) { f = TYF_VARIADIC; }
    return ty_make(TY_FN, ret, 0, 0, f, mark);
}
int ty_tuple(int mark) { return ty_make(TY_TUPLE, 0, 0, 0, 0, mark); }
int ty_named(int sid, int mark, int flags) { return ty_make(TY_NAMED, 0, 0, sid, flags, mark); }
int ty_generic(int sid) { return ty_make(TY_GENERIC, 0, 0, sid, 0, -1); }
int ty_struct(int sd) {
    int k = TY_STRUCT;
    if (g_sdef[sd].kind == SD_ENUM) { k = TY_ENUM; }
    if (g_sdef[sd].kind == SD_NEWTYPE) { k = TY_NEWTYPE; }
    return ty_make(k, 0, sd, 0, 0, -1);
}

int ty_is_int(int t) {
    int k = g_ty[t].kind;
    return k == TY_BOOL || k == TY_CHAR || (k >= TY_I8 && k <= TY_U64);
}
int ty_is_float(int t) { return g_ty[t].kind == TY_F32 || g_ty[t].kind == TY_F64; }
int ty_is_arith(int t) { return ty_is_int(t) || ty_is_float(t); }
int ty_is_signed(int t) {
    int k = g_ty[t].kind;
    return k == TY_I8 || k == TY_I16 || k == TY_I32 || k == TY_I64 || k == TY_CHAR;
}
int ty_bits(int t) {
    int k = g_ty[t].kind;
    if (k == TY_BOOL || k == TY_CHAR || k == TY_I8 || k == TY_U8) { return 8; }
    if (k == TY_I16 || k == TY_U16) { return 16; }
    if (k == TY_I32 || k == TY_U32 || k == TY_F32) { return 32; }
    if (k == TY_I64 || k == TY_U64 || k == TY_F64) { return 64; }
    return 0;
}

// a value held in a single wasm value (not in memory)
int ty_is_scalar(int t) {
    int k = g_ty[t].kind;
    if (k >= TY_BOOL && k <= TY_F64) { return 1; }
    if (k == TY_PTR || k == TY_REF || k == TY_ENUM || k == TY_NEWTYPE) {
        if (k == TY_NEWTYPE) { return ty_is_scalar(g_sdef[g_ty[t].n].base); }
        return 1;
    }
    return 0;
}

int ty_is_aggregate(int t) {
    int k = g_ty[t].kind;
    return k == TY_STRUCT || k == TY_ARRAY || k == TY_SLICE || k == TY_OPT || k == TY_TUPLE ||
           (k == TY_NEWTYPE && ty_is_scalar(t) == 0);
}

int align_up(int v, int a) {
    if (a <= 1) { return v; }
    return (v + a - 1) / a * a;
}

int ty_align(int t) {
    int k = g_ty[t].kind;
    if (k == TY_VOID || k == TY_ERROR) { return 1; }
    if (k >= TY_BOOL && k <= TY_F64) { return ty_bits(t) / 8; }
    if (k == TY_PTR || k == TY_REF) { return 4; }
    if (k == TY_ARRAY) { return ty_align(g_ty[t].base); }
    if (k == TY_SLICE) { return 8; }
    if (k == TY_OPT) {
        int b = g_ty[t].base;
        if (g_ty[b].kind == TY_REF) { return 4; }
        return ty_align(b);
    }
    if (k == TY_TUPLE) {
        int a = 1;
        int i = 0;
        while (i < g_ty[t].nlist) {
            int ea = ty_align(g_tyl[g_ty[t].list + i]);
            if (ea > a) { a = ea; }
            i = i + 1;
        }
        return a;
    }
    if (k == TY_STRUCT) { layout_struct(g_ty[t].n); return g_sdef[g_ty[t].n].align; }
    if (k == TY_ENUM || k == TY_NEWTYPE) { return ty_align(g_sdef[g_ty[t].n].base); }
    if (k == TY_FN) { return 4; }
    return 1;
}

int ty_size(int t) {
    int k = g_ty[t].kind;
    if (k == TY_VOID || k == TY_ERROR) { return 1; }
    if (k >= TY_BOOL && k <= TY_F64) { return ty_bits(t) / 8; }
    if (k == TY_PTR || k == TY_REF) { return 4; }
    if (k == TY_ARRAY) {
        int n = g_ty[t].n;
        if (n < 0) { n = 0; }
        return n * ty_size(g_ty[t].base);
    }
    if (k == TY_SLICE) { return 16; }
    if (k == TY_OPT) {
        // ?&T is a plain (nullable) pointer; ?T is { T value; bool present; }
        int b = g_ty[t].base;
        if (g_ty[b].kind == TY_REF) { return 4; }
        return align_up(ty_size(b) + 1, ty_align(b));
    }
    if (k == TY_TUPLE) {
        int off = 0;
        int i = 0;
        while (i < g_ty[t].nlist) {
            int et = g_tyl[g_ty[t].list + i];
            off = align_up(off, ty_align(et)) + ty_size(et);
            i = i + 1;
        }
        return align_up(off, ty_align(t));
    }
    if (k == TY_STRUCT) { layout_struct(g_ty[t].n); return g_sdef[g_ty[t].n].size; }
    if (k == TY_ENUM || k == TY_NEWTYPE) { return ty_size(g_sdef[g_ty[t].n].base); }
    if (k == TY_FN) { return 4; }
    return 1;
}

// offset of tuple element i
int tuple_off(int t, int idx) {
    int off = 0;
    int i = 0;
    while (i < g_ty[t].nlist) {
        int et = g_tyl[g_ty[t].list + i];
        off = align_up(off, ty_align(et));
        if (i == idx) { return off; }
        off = off + ty_size(et);
        i = i + 1;
    }
    return off;
}

// C layout for structs; a tagged union is { int tag; payload } with the
// payload aligned to the widest member.
void layout_struct(int sd) {
    if (g_sdef[sd].done != 0) { return; }
    g_sdef[sd].done = 1;
    int off = 0;
    int al = 1;
    int i = 0;
    if (g_sdef[sd].kind == SD_UNION) {
        int maxsz = 0;
        int maxal = 1;
        while (i < g_sdef[sd].nfield) {
            int f = g_sdef[sd].field0 + i;
            int ft = g_field[f].ty;
            int s = ty_size(ft);
            int a = ty_align(ft);
            if (s > maxsz) { maxsz = s; }
            if (a > maxal) { maxal = a; }
            i = i + 1;
        }
        if (maxal < 4) { al = 4; } else { al = maxal; }
        int poff = align_up(4, maxal);
        i = 0;
        while (i < g_sdef[sd].nfield) { g_field[g_sdef[sd].field0 + i].off = poff; i = i + 1; }
        g_sdef[sd].size = align_up(poff + maxsz, al);
        g_sdef[sd].align = al;
        return;
    }
    while (i < g_sdef[sd].nfield) {
        int f = g_sdef[sd].field0 + i;
        int ft = g_field[f].ty;
        int a = ty_align(ft);
        off = align_up(off, a);
        g_field[f].off = off;
        off = off + ty_size(ft);
        if (a > al) { al = a; }
        i = i + 1;
    }
    g_sdef[sd].size = align_up(off, al);
    g_sdef[sd].align = al;
}
