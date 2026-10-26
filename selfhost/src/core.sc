// core.sc — compiler pipeline entry points shared by both drivers.
#include "scc.h"
#include "parse.h"
#include "driver.h"

void scc_init() {
    util_init();
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

int scc_compile_main(int main_file) {
    scc_preprocess(main_file);
    if (g_nerrors > 0) { return 0; }
    parse_program();
    if (g_nerrors > 0) { return 0; }
    sema_program();
    if (g_nerrors > 0) { return 0; }
    gen_module();
    if (g_nerrors > 0) { return 0; }
    return 1;
}
