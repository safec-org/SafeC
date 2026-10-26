// core.sc — compiler pipeline entry points shared by both drivers.
#include "scc.h"
#include "parse.h"
#include "driver.h"

void scc_init() {
    dbg("init\n");
    util_init();
    dbg("kw\n");
    kw_init();
    types_init();
    pp_init();
    pp_define_lit("__SCC__", "1");
    pp_define_lit("__wasm__", "1");
    pp_define_lit("__wasm32__", "1");
    pp_define_lit("__SIZEOF_POINTER__", "4");
    pp_define_lit("__SAFEC_COMPAT_PREPROCESSOR__", "1");
}

// preprocess the runtime prelude, then the main file
int g_rt_file = -1;

void scc_preprocess(int main_file) {
    int rt = vfs_load(S_runtime);
    g_rt_file = rt;
    if (rt != 0) { pp_run(rt); }
    pp_run(main_file);
    pp_finish();
}

// 1: stop after semantic analysis (scc --check): no code, so no 'main'
// or definitions of declared functions are needed
int g_check_only = 0;

int scc_compile_main(int main_file) {
    dbg("pp\n");
    scc_preprocess(main_file);
    dbg("parse\n");
    if (g_nerrors > 0) { return 0; }
    parse_program();
    if (g_nerrors > 0) { return 0; }
    dbg("sema\n");
    sema_program();
    dbg("gen\n");
    if (g_nerrors > 0) { return 0; }
    if (g_check_only == 1) { return 1; }
    gen_module();
    if (g_nerrors > 0) { return 0; }
    return 1;
}
