// driver_wasm.sc — the browser/Node driver: scc compiled to wasm32 exports
// a tiny C-ABI used by web/scc.js:
//   scc_begin()            → address to write the main source (UTF-8) to
//   scc_capacity()         → max bytes for that source
//   scc_compile(len)       → 1 ok / 0 errors; source file name "main.sc"
//   scc_output()/scc_output_len()       → the wasm module
//   scc_diagnostics()/scc_diagnostics_len() → diagnostics text
// Each compile runs on a fresh instance (the compiler keeps global state).
#include "scc.h"
#include "driver.h"
#ifndef __SCC__
#include <std/stddef.h>
#endif

int g_begun = 0;

int host_load_file(int path_sid) { return 0; }
int host_find_include(int name_sid) { return 0; }

__attribute__((export_name("scc_begin")))
int scc_begin() {
    if (g_begun == 0) { scc_init(); g_begun = 1; }
    unsafe { return (int)&g_src[g_src_len]; }
}

__attribute__((export_name("scc_capacity")))
int scc_capacity() { return SRC_MAX - g_src_len - 1; }

__attribute__((export_name("scc_compile")))
int scc_compile(int len) {
    if (g_begun == 0) { scc_begin(); }
    int start = g_src_len;
    if (len < 0) { len = 0; }
    if (start + len >= SRC_MAX) { len = SRC_MAX - start - 1; }
    g_src_len = start + len;
    int f = add_file(intern_lit("main.sc"), start, len);
    return scc_compile_main(f);
}

__attribute__((export_name("scc_output")))
int scc_output() { unsafe { return (int)&g_out[0]; } }

__attribute__((export_name("scc_output_len")))
int scc_output_len() { return g_out_len; }

__attribute__((export_name("scc_diagnostics")))
int scc_diagnostics() { unsafe { return (int)&g_diag[0]; } }

__attribute__((export_name("scc_diagnostics_len")))
int scc_diagnostics_len() { return g_diag_len; }

__attribute__((export_name("scc_error_count")))
int scc_error_count() { return g_nerrors; }

#ifndef __SCC__
// freestanding libc pieces the C++ safec's wasm output may call
&void memcpy(&void dst, const &void src, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)dst;
        const &unsigned char s = (const &unsigned char)src;
        unsigned long i = 0UL;
        while (i < (unsigned long)n) { d[i] = s[i]; i = i + 1UL; }
    }
    return dst;
}

&void memmove(&void dst, const &void src, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)dst;
        const &unsigned char s = (const &unsigned char)src;
        if ((unsigned long)dst < (unsigned long)src) {
            unsigned long i = 0UL;
            while (i < (unsigned long)n) { d[i] = s[i]; i = i + 1UL; }
        } else {
            unsigned long i = (unsigned long)n;
            while (i > 0UL) { i = i - 1UL; d[i] = s[i]; }
        }
    }
    return dst;
}

size_t strlen(const &char s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    return (size_t)n;
}

&void memset(&void p, int v, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)p;
        unsigned long i = 0UL;
        while (i < (unsigned long)n) { d[i] = (unsigned char)v; i = i + 1UL; }
    }
    return p;
}
#endif
