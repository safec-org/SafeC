// driver_dump.sc — bootstrap debugging: reads a source on stdin and prints
// the compiler's intermediate state (tokens, AST shape, sema results, the
// module bytes' checksum). Built natively and by scc itself, the two
// outputs must be identical.
#include "scc.h"
#include "driver.h"
#include "sema.h"

#ifdef __SCC__
extern int __host_write(int fd, const char* buf, int len);
extern int __host_read(int fd, char* buf, int len);
#else
extern long write(int fd, const void* buf, unsigned long n);
extern long read(int fd, void* buf, unsigned long n);
#endif

char g_ob[65536];
int g_obn = 0;

void o_flush() {
#ifdef __SCC__
    unsafe { __host_write(1, (const char*)&g_ob[0], g_obn); }
#else
    unsafe { write(1, (const void*)&g_ob[0], (unsigned long)g_obn); }
#endif
    g_obn = 0;
}
void o_c(int c) { if (g_obn >= 65000) { o_flush(); } g_ob[g_obn] = (char)c; g_obn = g_obn + 1; }
void o_s(const char* s) { int i = 0; int n = lit_len(s); while (i < n) { o_c(lit_char(s, i)); i = i + 1; } }
void o_sid(int sid) { int i = 0; while (i < g_str_len[sid]) { o_c(str_char(sid, i)); i = i + 1; } }
void o_i(long v) {
    tmp_reset();
    tmp_int(v);
    int i = 0;
    while (i < g_tmp_len) { o_c((int)g_tmp[i]); i = i + 1; }
}

int host_load_file(int path_sid) { return 0; }
int host_find_include(int name_sid) { return 0; }

int read_in(int at, int cap) {
    int n = 0;
#ifdef __SCC__
    unsafe { n = __host_read(0, (char*)&g_src[at], cap); }
#else
    unsafe { n = (int)read(0, (void*)&g_src[at], (unsigned long)cap); }
#endif
    return n;
}

int main() {
    scc_init();
    int start = g_src_len;
    while (g_src_len < SRC_MAX - 1) {
        int n = read_in(g_src_len, SRC_MAX - 1 - g_src_len);
        if (n <= 0) { break; }
        g_src_len = g_src_len + n;
    }
    int f = add_file(intern_lit("main.sc"), start, g_src_len - start);
    scc_preprocess(f);
    o_s("tokens "); o_i((long)g_ntok); o_c(10);
    int i = 1;
    while (i < g_ntok) {
        o_i((long)g_tok[i].kind); o_c(' ');
        if (g_tok[i].sid != 0) { o_sid(g_tok[i].sid); }
        o_c(' '); o_i((long)g_tok[i].line); o_c(':'); o_i((long)g_tok[i].col); o_c(' '); o_i(g_tok[i].ival); o_c(10);
        i = i + 1;
    }
    parse_program();
    o_s("nodes "); o_i((long)g_nnode); o_c(10);
    i = 1;
    while (i <= g_nnode) {
        o_i((long)g_node[i].kind); o_c(' '); o_i((long)g_node[i].a); o_c(' '); o_i((long)g_node[i].b); o_c(' ');
        o_i((long)g_node[i].c); o_c(' '); o_i((long)g_node[i].nlist); o_c(' '); o_i((long)g_node[i].sid); o_c(10);
        i = i + 1;
    }
    sema_program();
    o_s("funcs "); o_i((long)g_nfunc); o_s(" types "); o_i((long)g_nty); o_s(" errors "); o_i((long)g_nerrors); o_c(10);
    i = 1;
    while (i <= g_nfunc) { o_sid(g_func[i].sid); o_c(' '); o_i((long)g_func[i].ty); o_c(10); i = i + 1; }
    i = 1;
    while (i <= g_nnode) { o_i((long)g_node[i].ty); o_c(' '); i = i + 1; if (i % 32 == 0) { o_c(10); } }
    o_c(10);
    if (g_nerrors == 0) {
        gen_module();
        o_s("module "); o_i((long)g_out_len); o_c(10);
        long h = 0L;
        i = 0;
        while (i < g_out_len) { h = h *| 31L +| (long)g_out[i]; i = i + 1; }
        o_i(h); o_c(10);
    }
    i = 0;
    while (i < g_diag_len) { o_c((int)g_diag[i]); i = i + 1; }
    o_flush();
    return 0;
}
