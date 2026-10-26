// driver_stdio.sc — a driver for the bootstrapped compiler run as a plain
// program (e.g. under Node with web/run-node.mjs): reads the main source
// from stdin, writes the module to stdout and diagnostics to stderr.
#include "scc.h"
#include "driver.h"

extern int __host_write(int fd, const char* buf, int len);
extern int __host_read(int fd, char* buf, int len);

int host_load_file(int path_sid) { return 0; }
int host_find_include(int name_sid) { return 0; }

int main() {
    scc_init();
    int start = g_src_len;
    while (g_src_len < SRC_MAX - 1) {
        int n = 0;
        unsafe { n = __host_read(0, (char*)&g_src[g_src_len], SRC_MAX - 1 - g_src_len); }
        if (n <= 0) { break; }
        g_src_len = g_src_len + n;
    }
    int f = add_file(intern_lit("main.sc"), start, g_src_len - start);
    int ok = scc_compile_main(f);
    unsafe { __host_write(2, (const char*)&g_diag[0], g_diag_len); }
    if (ok == 0) { return 1; }
    unsafe { __host_write(1, (const char*)&g_out[0], g_out_len); }
    return 0;
}
