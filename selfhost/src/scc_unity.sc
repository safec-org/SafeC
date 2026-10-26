// scc_unity.sc — the whole compiler as one translation unit, for scc to
// compile itself (the C++ safec builds the files separately).
//   scc -D SCC_SMALL scc_unity.sc -o scc2.wasm
#include "util.sc"
#include "lex.sc"
#include "pp.sc"
#include "parse.sc"
#include "types.sc"
#include "sema.sc"
#include "expr.sc"
#include "call.sc"
#include "stmt.sc"
#include "wasm.sc"
#include "gen.sc"
#include "genx.sc"
#include "vfs.sc"
#include "memflow.sc"
#include "core.sc"
#include "driver_wasm.sc"
