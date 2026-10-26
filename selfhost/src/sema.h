// sema.h — semantic analysis state shared by sema.sc, expr.sc and gen.sc.
#pragma once
#include "scc.h"

// extra node flags set by sema
#define NF_LVAL     2097152    // expression designates a writable object
#define NF_INUNSAFE 4194304    // checked inside an unsafe block (no bounds checks)
#define NF_DESIG    8388608    // designates an object (element/member), maybe const

// E_CALL kinds (node.c)
#define CK_FUNC      1   // direct call: d = func
#define CK_METHOD    2   // d = func, b = receiver expression
#define CK_INDIRECT  3   // through a function value (a = callee expr)
#define CK_UNIONINIT 4   // d = sdef, op = field index (0-based)
#define CK_ISNULL    5   // b = base
#define CK_ISNONE    6
#define CK_DEFAULT   7   // b = base, list[0] = fallback
#define CK_BUILTIN   8   // op = builtin id
#define CK_VAPTR     9   // __va_ptr(): the variadic argument area

// builtins (CK_BUILTIN op)
#define BI_MEMSIZE   1   // __builtin_memory_size() → pages
#define BI_MEMGROW   2   // __builtin_memory_grow(pages) → old pages or -1
#define BI_TRAP      3   // __builtin_trap()
#define BI_SQRT      4
#define BI_FLOOR     5
#define BI_CEIL      6
#define BI_FABS      7
#define BI_TRUNC     8
#define BI_HEAPBASE  9   // __builtin_heap_base() → first free address
#define BI_CLZ      10
#define BI_CTZ      11
#define BI_POPCNT   12
#define BI_F64BITS  13   // __builtin_f64_bits(double) → long
#define BI_F64FROM  14   // __builtin_f64_from_bits(long) → double
#define BI_NEAREST  15
#define BI_UNREACH  16
#define BI_JOIN     17
#define BI_SPAWN    18

// E_MEMBER kinds (node.d)
#define MK_FIELD   1   // c = field index
#define MK_TUPLE   2   // c = element index
#define MK_SLEN    3
#define MK_SPTR    4

extern struct Sym g_sym[NSYM_MAX];
extern int g_nsym;
extern int g_cur_fn;
extern int g_mkey_sd[16384];
extern int g_region_glob[NSTR_MAX];   // region sid → arena state global
extern int g_region_cap[NSTR_MAX];
extern int g_nregions;
extern int g_main_fn;

int ty_equal(int a, int b);
int can_convert(int from, int to);
int int_lit_fits(int e, int to);
int float_lit_fits(int e, int to);
int ref_to_ptr_arg(int from, int to);
int convertible_to(int e, int from, int to);
int find_field(int sd, int sid);
int lookup_method(int sd, int sid);
int struct_of(int t);
int tuple_off(int t, int idx);
int fn_type_of(int t);
int sema_const_int(int e);
extern long g_cval;
int resolve_in_env(int t);
int func_ret(int f);
int func_nparams(int f);
int func_param(int f, int i);
int deref_struct(int t);
extern int g_tyloc;
extern int g_ce_struct;
