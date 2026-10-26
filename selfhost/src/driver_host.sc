// driver_host.sc — native command-line driver (built with the C++ safec):
//   scc [-I dir]... [-o out.wasm] [--dump-tokens] input.sc
// Reads files with libc, runs the core compiler, writes the module.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scc.h"
#include "driver.h"
#include "lex.h"


int g_incdir[32];
int g_nincdir = 0;
char g_pathbuf[4096];

extern long write(int fd, const void* buf, unsigned long n);

void err_lit(const char* s) {
    unsafe { write(2, (const void*)s, (unsigned long)lit_len(s)); }
}

void err_path() {
    unsafe { write(2, (const void*)&g_pathbuf[0], (unsigned long)lit_len((const char*)&g_pathbuf[0])); }
}


// path_sid → NUL-terminated g_pathbuf
void path_to_buf(int sid) {
    int n = g_str_len[sid];
    if (n > 4000) { n = 4000; }
    int i = 0;
    while (i < n) { g_pathbuf[i] = (char)str_char(sid, i); i = i + 1; }
    g_pathbuf[n] = (char)0;
}

int g_loaded_name[NFILE_MAX];
int g_loaded_file[NFILE_MAX];
int g_nloaded = 0;

int host_load_file(int path_sid) {
    int k = 0;
    while (k < g_nloaded) {
        if (g_loaded_name[k] == path_sid) { return g_loaded_file[k]; }
        k = k + 1;
    }
    path_to_buf(path_sid);
    int start = g_src_len;
    int n = 0;
    unsafe {
        FILE* f = fopen((const char*)&g_pathbuf[0], "rb");
        if (f == null) { return 0; }
        n = (int)fread((void*)&g_src[start], 1UL, (unsigned long)(SRC_MAX - start - 1), f);
        fclose(f);
    }
    g_src_len = start + n;
    int file = add_file(path_sid, start, n);
    if (g_nloaded < NFILE_MAX) {
        g_loaded_name[g_nloaded] = path_sid;
        g_loaded_file[g_nloaded] = file;
        g_nloaded = g_nloaded + 1;
    }
    return file;
}

int host_find_include(int name_sid) {
    int i = 0;
    while (i < g_nincdir) {
        int f = host_load_file(str_concat3(g_incdir[i], "/", name_sid));
        if (f != 0) { return f; }
        i = i + 1;
    }
    return 0;
}

int arg_sid(char** argv, int i) {
    tmp_reset();
    unsafe {
        char* a = argv[i];
        int j = 0;
        while (a[j] != (char)0) { tmp_char((int)(unsigned char)a[j]); j = j + 1; }
    }
    return intern_tmp();
}

void write_diags() {
    unsafe { write(2, (const void*)&g_diag[0], (unsigned long)g_diag_len); }
}

// -D NAME[=VALUE]
char g_defname[256];
char g_defval[256];
void define_arg(int d, int eq) {
    int n = g_str_len[d];
    int nl = n;
    if (eq >= 0) { nl = eq; }
    if (nl > 255) { nl = 255; }
    int k = 0;
    while (k < nl) { g_defname[k] = (char)str_char(d, k); k = k + 1; }
    g_defname[nl] = (char)0;
    int vl = 0;
    if (eq >= 0) {
        k = eq + 1;
        while (k < n && vl < 255) { g_defval[vl] = (char)str_char(d, k); vl = vl + 1; k = k + 1; }
    } else { g_defval[0] = '1'; vl = 1; }
    g_defval[vl] = (char)0;
    unsafe { pp_define_lit((const char*)&g_defname[0], (const char*)&g_defval[0]); }
}

int main(int argc, char** argv) {
    scc_init();
    int input = 0;
    int out = 0;
    int dump = 0;
    int i = 1;
    while (i < argc) {
        int a = arg_sid(argv, i);
        if (str_eq_lit(a, "-I") == 1 && i + 1 < argc) {
            i = i + 1;
            if (g_nincdir < 32) { g_incdir[g_nincdir] = arg_sid(argv, i); g_nincdir = g_nincdir + 1; }
        } else if (str_eq_lit(a, "-o") == 1 && i + 1 < argc) {
            i = i + 1;
            out = arg_sid(argv, i);
        } else if (str_eq_lit(a, "-D") == 1 && i + 1 < argc) {
            i = i + 1;
            int d = arg_sid(argv, i);
            // NAME or NAME=VALUE
            tmp_reset();
            int k = 0;
            int eq = -1;
            while (k < g_str_len[d]) { if (str_char(d, k) == '=' && eq < 0) { eq = k; } k = k + 1; }
            define_arg(d, eq);
        } else if (str_eq_lit(a, "--dump-tokens") == 1) {
            dump = 1;
        } else if (str_eq_lit(a, "--check") == 1) {
            dump = 2;
        } else {
            input = a;
        }
        i = i + 1;
    }
    if (input == 0) {
        err_lit("usage: scc [-I dir] [-o out.wasm] input.sc\n");
        return 2;
    }
    int f = host_load_file(input);
    if (f == 0) {
        path_to_buf(input);
        err_lit("scc: cannot open '"); err_path(); err_lit("'\n");
        return 2;
    }
    if (dump == 1) {
        scc_preprocess(f);
        write_diags();
        int t = 1;
        int line = 0;
        while (t < g_ntok) {
            if (g_tok[t].line != line) { unsafe { printf("\n%d: ", g_tok[t].line); } line = g_tok[t].line; }
            tmp_reset();
            tmp_tok_spelling(g_tok[t].kind, g_tok[t].sid);
            tmp_char(' ');
            tmp_char(0);
            unsafe { printf("%s", (const char*)&g_tmp[0]); }
            t = t + 1;
        }
        unsafe { printf("\n%d tokens\n", g_ntok); }
        return 0;
    }
    int ok = scc_compile_main(f);
    write_diags();
    if (ok == 0) { return 1; }
    if (dump == 2) { return 0; }
    if (out == 0) { out = intern_lit("a.wasm"); }
    path_to_buf(out);
    unsafe {
        FILE* o = fopen((const char*)&g_pathbuf[0], "wb");
        if (o == null) { err_lit("scc: cannot write '"); err_path(); err_lit("'\n"); return 2; }
        fwrite((const void*)&g_out[0], 1UL, (unsigned long)g_out_len, o);
        fclose(o);
    }
    return 0;
}
