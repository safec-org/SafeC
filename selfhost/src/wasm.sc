// wasm.sc — WebAssembly binary encoding: LEB128, function types, and the
// final module assembly (type/import/function/table/memory/global/export/
// element/code/data sections).
#include "scc.h"
#include "gen.h"

unsigned char g_code[CODE_MAX];
int g_code_len = 0;
unsigned char g_codesec[OUT_MAX];
int g_codesec_len = 0;
unsigned char g_data[DATA_MAX];
int g_data_len = 0;

int g_ftype_param[NFUNC_MAX * 4];
int g_nftype_param = 0;
int g_ftype_p0[NFUNC_MAX];
int g_ftype_np[NFUNC_MAX];
int g_ftype_res[NFUNC_MAX];
int g_nftype = 0;

int g_wfunc[NFUNC_MAX];
int g_nwfunc = 0;
int g_wfunc_type[NFUNC_MAX];
int g_wfunc_body[NFUNC_MAX];
int g_wfunc_blen[NFUNC_MAX];
int g_nimports = 0;
int g_import_name[16];
int g_import_type[16];
int g_table[NFUNC_MAX];
int g_ntable = 0;
int g_heap_base = 0;
int g_stack_top = 0;
int g_export_sid[256];
int g_export_func[256];
int g_nexport = 0;

void emit(int b) {
    if (g_code_len < CODE_MAX) {
        g_code[g_code_len] = (unsigned char)b;
        g_code_len = g_code_len + 1;
    } else if (g_code_len == CODE_MAX) {
        fatal("function code size");
        g_code_len = g_code_len + 1;
    }
}

void emitu(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = (long)((unsigned long)v >> 7UL);
        if (v != 0L) { emit(b | 128); } else { emit(b); break; }
    }
}

void emits(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = v >> 7L;
        int sign = b & 64;
        if ((v == 0L && sign == 0) || (v == -1L && sign != 0)) { emit(b); break; }
        emit(b | 128);
    }
}

void emit_i32(int v) { emit(OP_I32_CONST); emits((long)v); }
void emit_i64(long v) { emit(OP_I64_CONST); emits(v); }

void emit_f64_bits(long bits) {
    emit(OP_F64_CONST);
    int i = 0;
    while (i < 8) {
        emit((int)((bits >> (long)(i * 8)) & 255L));
        i = i + 1;
    }
}

void emit_mem(int op, int align, int off) {
    emit(op);
    emitu((long)align);
    emitu((long)off);
}

void emit_fc(int sub) { emit(OP_PREFIX_FC); emitu((long)sub); }

// function type with params g_ftype_param[mark..] (already pushed) and an
// optional single result
int func_type_index(int mark, int nres, int res) {
    int np = g_nftype_param - mark;
    if (nres == 0) { res = 0; }
    int i = 0;
    while (i < g_nftype) {
        if (g_ftype_np[i] == np && g_ftype_res[i] == res) {
            int same = 1;
            int j = 0;
            while (j < np) {
                if (g_ftype_param[g_ftype_p0[i] + j] != g_ftype_param[mark + j]) { same = 0; break; }
                j = j + 1;
            }
            if (same == 1) { g_nftype_param = mark; return i; }
        }
        i = i + 1;
    }
    if (g_nftype >= NFUNC_MAX) { fatal("function types"); return 0; }
    g_ftype_p0[g_nftype] = mark;
    g_ftype_np[g_nftype] = np;
    g_ftype_res[g_nftype] = res;
    g_nftype = g_nftype + 1;
    return g_nftype - 1;
}

// ── module assembly into g_out ───────────────────────────────────────────────
void out_u(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = (long)((unsigned long)v >> 7UL);
        if (v != 0L) { out_byte(b | 128); } else { out_byte(b); break; }
    }
}

void out_name(int sid) {
    out_u((long)g_str_len[sid]);
    int i = 0;
    while (i < g_str_len[sid]) { out_byte(str_char(sid, i)); i = i + 1; }
}

void out_name_lit(const char* s) {
    int n = lit_len(s);
    out_u((long)n);
    int i = 0;
    while (i < n) { out_byte(lit_char(s, i)); i = i + 1; }
}

// sections are written with a 5-byte padded size placeholder
int sec_begin(int id) {
    out_byte(id);
    int at = g_out_len;
    out_byte(128); out_byte(128); out_byte(128); out_byte(128); out_byte(0);
    return at;
}

void sec_end(int at) {
    int size = g_out_len - at - 5;
    int i = 0;
    while (i < 5) {
        int b = (size >> (7 * i)) & 127;
        if (i < 4) { b = b | 128; }
        if (at + i < OUT_MAX) { g_out[at + i] = (unsigned char)b; }
        i = i + 1;
    }
}

int g_stack_limit = 0;

void out_sleb(long v) {
    while (1) {
        int b = (int)(v & 127L);
        v = v >> 7L;
        int sign = b & 64;
        if ((v == 0L && sign == 0) || (v == -1L && sign != 0)) { out_byte(b); break; }
        out_byte(b | 128);
    }
}

void out_global(int mutable, int value) {
    out_byte(VT_I32);
    out_byte(mutable);
    out_byte(OP_I32_CONST);
    out_sleb((long)value);
    out_byte(OP_END);
}

int mem_pages() {
    int need = g_heap_base + 65536;
    return (need + 65535) / 65536;
}

void wasm_assemble() {
    g_out_len = 0;
    out_byte(0); out_byte(97); out_byte(115); out_byte(109);   // \0asm
    out_byte(1); out_byte(0); out_byte(0); out_byte(0);         // version 1
    // types
    int s = sec_begin(1);
    out_u((long)g_nftype);
    int i = 0;
    while (i < g_nftype) {
        out_byte(96);
        out_u((long)g_ftype_np[i]);
        int j = 0;
        while (j < g_ftype_np[i]) { out_byte(g_ftype_param[g_ftype_p0[i] + j]); j = j + 1; }
        if (g_ftype_res[i] == 0) { out_u(0L); } else { out_u(1L); out_byte(g_ftype_res[i]); }
        i = i + 1;
    }
    sec_end(s);
    // imports
    s = sec_begin(2);
    out_u((long)g_nimports);
    i = 0;
    while (i < g_nimports) {
        out_name_lit("env");
        out_name(g_import_name[i]);
        out_byte(0);
        out_u((long)g_import_type[i]);
        i = i + 1;
    }
    sec_end(s);
    // functions
    s = sec_begin(3);
    out_u((long)g_nwfunc);
    i = 0;
    while (i < g_nwfunc) { out_u((long)g_wfunc_type[i]); i = i + 1; }
    sec_end(s);
    // table (funcref), entry 0 is the null function
    s = sec_begin(4);
    out_u(1L);
    out_byte(112);
    out_byte(1);
    out_u((long)(g_ntable + 1));
    out_u((long)(g_ntable + 1));
    sec_end(s);
    // memory: min pages, max 16384 (1 GiB)
    s = sec_begin(5);
    out_u(1L);
    out_byte(1);
    out_u((long)mem_pages());
    out_u(16384L);
    sec_end(s);
    // globals: 0 = stack pointer, 1 = stack limit, 2 = heap base
    s = sec_begin(6);
    out_u(3L);
    out_global(1, g_stack_top);
    out_global(0, g_stack_limit);
    out_global(0, g_heap_base);
    sec_end(s);
    // exports
    s = sec_begin(7);
    out_u((long)(g_nexport + 1));
    out_name_lit("memory");
    out_byte(2);
    out_u(0L);
    i = 0;
    while (i < g_nexport) {
        out_name(g_export_sid[i]);
        out_byte(0);
        out_u((long)g_func[g_export_func[i]].widx);
        i = i + 1;
    }
    sec_end(s);
    // element segment: table[1..] = functions
    if (g_ntable > 0) {
        s = sec_begin(9);
        out_u(1L);
        out_u(0L);
        out_byte(OP_I32_CONST);
        out_byte(1);
        out_byte(OP_END);
        out_u((long)g_ntable);
        i = 0;
        while (i < g_ntable) { out_u((long)g_func[g_table[i]].widx); i = i + 1; }
        sec_end(s);
    }
    // code
    s = sec_begin(10);
    out_u((long)g_nwfunc);
    i = 0;
    while (i < g_nwfunc) {
        out_u((long)g_wfunc_blen[i]);
        int j = 0;
        int b0 = g_wfunc_body[i];
        while (j < g_wfunc_blen[i]) { out_byte((int)g_codesec[b0 + j]); j = j + 1; }
        i = i + 1;
    }
    sec_end(s);
    // data: one active segment at DATA_BASE
    int dlen = g_data_len;
    while (dlen > 0 && g_data[dlen - 1] == (unsigned char)0) { dlen = dlen - 1; }
    if (dlen > 0) {
        s = sec_begin(11);
        out_u(1L);
        out_u(0L);
        out_byte(OP_I32_CONST);
        {
            long v = (long)DATA_BASE;
            while (1) {
                int b = (int)(v & 127L);
                v = v >> 7L;
                int sign = b & 64;
                if ((v == 0L && sign == 0) || (v == -1L && sign != 0)) { out_byte(b); break; }
                out_byte(b | 128);
            }
        }
        out_byte(OP_END);
        out_u((long)dlen);
        i = 0;
        while (i < dlen) { out_byte((int)g_data[i]); i = i + 1; }
        sec_end(s);
    }
    if (g_out_len >= OUT_MAX) { fatal("output module size"); }
}
