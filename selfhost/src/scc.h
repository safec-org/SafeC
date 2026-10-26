// scc — a SafeC compiler written in SafeC, emitting WebAssembly.
//
// Shared declarations for every translation unit of the compiler. The
// compiler keeps all of its data in fixed-size global tables addressed by
// int index (index 0 is always "none"), so it needs no allocator and no
// pointers: the same source builds natively (for testing), for wasm32 with
// the C++ safec (the browser playground), and with scc itself (bootstrap).
#pragma once

// ── Capacities ───────────────────────────────────────────────────────────────
// SCC_SMALL: the browser playground build (smaller tables, ~30 MB of memory)
#ifdef SCC_SMALL
#define SRC_MAX      1048576
#define SP_MAX       1048576
#define NSTR_MAX     65536
#define TOK_MAX      120000
#define RAW_MAX      60000
#define MTOK_MAX     30000
#define NODE_MAX     120000
#define LIST_MAX     200000
#define OUT_MAX      2097152
#define CODE_MAX     1048576
#define DATA_MAX     2097152
#endif
#ifndef SRC_MAX
#define SRC_MAX 4194304   // bytes of source text (all files)
#endif
#ifndef NFILE_MAX
#define NFILE_MAX 256
#endif
#ifndef SP_MAX
#define SP_MAX 4194304   // string pool bytes
#endif
#ifndef NSTR_MAX
#define NSTR_MAX 131072    // interned strings
#endif
#ifndef STRHASH_SIZE
#define STRHASH_SIZE 65536
#endif
#ifndef TOK_MAX
#define TOK_MAX 400000    // preprocessed tokens
#endif
#ifndef RAW_MAX
#define RAW_MAX 200000    // raw (per-file, stacked) tokens
#endif
#ifndef MTOK_MAX
#define MTOK_MAX 100000    // macro body tokens
#endif
#ifndef NMACRO_MAX
#define NMACRO_MAX 8192
#endif
#ifndef NODE_MAX
#define NODE_MAX 400000
#endif
#ifndef LIST_MAX
#define LIST_MAX 600000
#endif
#ifndef TY_MAX
#define TY_MAX 16384
#endif
#ifndef TYL_MAX
#define TYL_MAX 65536     // type component lists
#endif
#ifndef TYHASH_SIZE
#define TYHASH_SIZE 16384
#endif
#ifndef NSTRUCT_MAX
#define NSTRUCT_MAX 2048
#endif
#ifndef NFIELD_MAX
#define NFIELD_MAX 16384
#endif
#ifndef NFUNC_MAX
#define NFUNC_MAX 8192
#endif
#ifndef NGLOB_MAX
#define NGLOB_MAX 8192
#endif
#ifndef NLOCAL_MAX
#define NLOCAL_MAX 65536
#endif
#ifndef NSYM_MAX
#define NSYM_MAX 8192      // scope stack entries
#endif
#ifndef OUT_MAX
#define OUT_MAX 8388608   // output module bytes
#endif
#ifndef CODE_MAX
#define CODE_MAX 4194304   // function body staging
#endif
#ifndef DATA_MAX
#define DATA_MAX 4194304   // data segment staging
#endif
#ifndef DIAG_MAX
#define DIAG_MAX 65536
#endif
#ifndef MSG_MAX
#define MSG_MAX 4096
#endif
#ifndef TMP_MAX
#define TMP_MAX 65536
#endif

// ── Token kinds ──────────────────────────────────────────────────────────────
#define T_EOF      0
#define T_IDENT    1
#define T_INT      2
#define T_FLOAT    3
#define T_CHAR     4
#define T_STRING   5
// punctuators
#define T_LPAREN   10
#define T_RPAREN   11
#define T_LBRACE   12
#define T_RBRACE   13
#define T_LBRACKET 14
#define T_RBRACKET 15
#define T_SEMI     16
#define T_COMMA    17
#define T_DOT      18
#define T_DOTDOT   19
#define T_ELLIPSIS 20
#define T_ARROW    21
#define T_COLONCOLON 22
#define T_COLON    23
#define T_QUESTION 24
#define T_QAMP     25
#define T_BANG     26
#define T_TILDE    27
#define T_PLUS     28
#define T_MINUS    29
#define T_STAR     30
#define T_SLASH    31
#define T_PERCENT  32
#define T_AMP      33
#define T_PIPE     34
#define T_CARET    35
#define T_SHL      36
#define T_SHR      37
#define T_ASSIGN   38
#define T_EQ       39
#define T_NE       40
#define T_LT       41
#define T_GT       42
#define T_LE       43
#define T_GE       44
#define T_ANDAND   45
#define T_OROR     46
#define T_PLUSEQ   47
#define T_MINUSEQ  48
#define T_STAREQ   49
#define T_SLASHEQ  50
#define T_PERCENTEQ 51
#define T_AMPEQ    52
#define T_PIPEEQ   53
#define T_CARETEQ  54
#define T_SHLEQ    55
#define T_SHREQ    56
#define T_INC      57
#define T_DEC      58
#define T_HASH     59
#define T_HASHHASH 60
#define T_PLUSPIPE 61
#define T_MINUSPIPE 62
#define T_STARPIPE 63
#define T_PLUSPCT  64
#define T_MINUSPCT 65
#define T_STARPCT  66
#define T_FATARROW 67
#define T_INVALID  68
// keywords (assigned by the parser from interned identifiers)
#define K_FIRST    100
#define K_AUTO     100
#define K_BREAK    101
#define K_CASE     102
#define K_CHAR     103
#define K_CONST    104
#define K_CONTINUE 105
#define K_DEFAULT  106
#define K_DO       107
#define K_DOUBLE   108
#define K_ELSE     109
#define K_ENUM     110
#define K_EXTERN   111
#define K_FLOAT    112
#define K_FOR      113
#define K_GOTO     114
#define K_IF       115
#define K_INLINE   116
#define K_INT      117
#define K_LONG     118
#define K_REGISTER 119
#define K_RESTRICT 120
#define K_RETURN   121
#define K_SHORT    122
#define K_SIGNED   123
#define K_SIZEOF   124
#define K_STATIC   125
#define K_STRUCT   126
#define K_SWITCH   127
#define K_TYPEDEF  128
#define K_UNION    129
#define K_UNSIGNED 130
#define K_VOID     131
#define K_VOLATILE 132
#define K_WHILE    133
#define K_BOOL     134
#define K_TRUE     135
#define K_FALSE    136
#define K_NULL     137
#define K_REGION   138
#define K_UNSAFE   139
#define K_CONSTEVAL 140
#define K_GENERIC  141
#define K_STATIC_ASSERT 142
#define K_SELF     143
#define K_OPERATOR 144
#define K_NEW      145
#define K_ARENA_RESET 146
#define K_ARENA_DESTROY 147
#define K_ARENA_MARK 148
#define K_ARENA_FREE_TO 149
#define K_TUPLE    150
#define K_DEFER    151
#define K_ERRDEFER 152
#define K_MATCH    153
#define K_PACKED   154
#define K_TRY      155
#define K_MUST_USE 156
#define K_FN       157
#define K_ALIGNOF  158
#define K_TYPEOF   159
#define K_NORETURN 160
#define K_PURE     161
#define K_NEWTYPE  162
#define K_TRAIT    163
#define K_NAMESPACE 164
#define K_CONSTINIT 165
#define K_ASM      166
#define K_SPAWN    167
#define K_ATTRIBUTE 168
#define K_FIELDCOUNT 169
#define K_FN_EVAL  171
#define K_LAST     170

// token flags
#define TF_BOL     1     // first token on its line
#define TF_SPACE   2     // preceded by whitespace
#define TF_UNS     4     // integer literal: U suffix
#define TF_LONG    8     // integer literal: L / LL suffix
#define TF_F32     16    // float literal: f suffix
#define TF_NOEXP   32    // identifier must not be macro-expanded again

struct Tok {
    int kind;
    int sid;      // identifier / literal text (string literal: decoded bytes)
    int file;
    int line;
    int col;
    int flags;
    long ival;    // integer / char value, or IEEE bits of a float literal
};

// ── AST node kinds ───────────────────────────────────────────────────────────
// expressions
#define E_INT       1    // ival, flags TF_UNS/TF_LONG
#define E_FLOAT     2    // ival = f64 bits, flags TF_F32
#define E_CHAR      3    // ival
#define E_STR       4    // sid
#define E_BOOL      5    // ival
#define E_NULL      6
#define E_IDENT     7    // sid; sema: c = SYM_* kind, d = index
#define E_UNARY     8    // op (T_MINUS, T_BANG, T_TILDE, T_PLUS, T_INC/T_DEC pre), a
#define E_POSTFIX   9    // op (T_INC/T_DEC), a
#define E_ADDR      10   // a
#define E_DEREF     11   // a
#define E_BINARY    12   // op, a, b
#define E_ASSIGN    13   // op (T_ASSIGN or compound), a, b
#define E_TERNARY   14   // a ? b : c
#define E_CALL      15   // a = callee, list = args; sema: d = callee func, c = call kind
#define E_INDEX     16   // a[b]
#define E_SLICE     17   // a[b..c]
#define E_MEMBER    18   // a.sid  (flags NF_ARROW for ->)
#define E_CAST      19   // (c)a     c = type
#define E_SIZEOF_T  20   // c = type
#define E_SIZEOF_E  21   // a
#define E_ALIGNOF   22   // c = type
#define E_INITLIST  23   // list of elements (E_DESIG for .field = v)
#define E_DESIG     24   // sid = field, a = value
#define E_NEW       25   // c = type, sid = region (0 = heap)
#define E_ARENA     26   // op = K_ARENA_*, sid = region, a = arg
#define E_TUPLE     27   // list
#define E_MATCH     28   // a = subject, list = arms (S_ARM)
#define E_TRY       29   // a
#define E_UNIONINIT 30   // sid = field, d = union type name sid, a = payload
#define E_FORCE     31   // a!  (force unwrap)
#define E_COMMA     32   // a, b
#define E_TYPEARG   33   // placeholder
#define E_SPAWN     34   // spawn(a, b): a = function, b = argument
#define E_FNEVAL    35   // fn_eval(a, b): a = object, b = function name; sema: d = method
// statements
#define S_BLOCK     40   // list
#define S_EXPR      41   // a
#define S_VAR       42   // sid, c = declared type (0 = auto), a = init; d = local index
#define S_IF        43   // a cond, b then, c else
#define S_WHILE     44   // a cond, b body, sid = label
#define S_DO        45   // a cond, b body
#define S_FOR       46   // a init, b cond, c incr, d body, sid = label
#define S_RETURN    47   // a
#define S_BREAK     48   // sid = label
#define S_CONTINUE  49   // sid = label
#define S_UNSAFE    50   // a = block
#define S_DEFER     51   // a = stmt, flags NF_ERR for errdefer
#define S_MATCH     52   // a = subject, list = arms
#define S_ARM       53   // list = patterns, a = body (stmt or expr)
#define S_SWITCH    54   // a = subject, list = cases
#define S_CASE      55   // list = values (E_*), b = body block (S_BLOCK), flags NF_DEFAULT
#define S_EMPTY     56
#define S_STATIC_ASSERT 57 // a = cond, sid = message
#define S_IFCONST   58   // a cond, b then, c else
#define S_GOTO      59   // sid
#define S_LABEL     60   // sid, a = stmt
#define S_MULTI     61   // list of S_VAR (shares the enclosing scope)
// patterns
#define P_VALUE     70   // a = expr (int/char literal, or enumerator ident)
#define P_RANGE     71   // a .. b
#define P_WILD      72   // default / _
#define P_NULL      73
#define P_NONE      74
#define P_SOME      75   // sid = binding name (0 = none)
#define P_VARIANT   76   // sid = variant, d = binding name
#define P_NAME      77   // sid = name (enumerator, null, none, some), d = binding, flags 1 = had (...)
// declarations
#define D_FUNC      80   // sid name, c ret type, list params, a body, d = owner sid, b = generic params node
#define D_PARAM     81   // sid, c type
#define D_GLOBAL    82   // sid, c type, a init
#define D_STRUCT    83   // sid, list = members (D_FIELD / D_FUNC protos), b = generic params node
#define D_FIELD     84   // sid, c type
#define D_ENUM      85   // sid, c underlying type, list = D_ENUMC
#define D_ENUMC     86   // sid, a = value expr
#define D_TYPEDEF   87   // sid, c type
#define D_NEWTYPE   88   // sid, c type
#define D_REGION    89   // sid, ival = capacity
#define D_TRAIT     90   // sid, list = method protos, b = supertraits node (list of names)
#define D_STATIC_ASSERT 91 // a cond, sid msg
#define D_GPARAMS   92   // list = D_GPARAM
#define D_GPARAM    93   // sid, b = bounds node (list of name sids as E_IDENT)
#define D_NAMES     94   // list of E_IDENT names

// node flags
#define NF_ARROW    1
#define NF_ERR      2
#define NF_DEFAULT  4
#define NF_STATIC   8
#define NF_EXTERN   16
#define NF_CONST    32      // const var / const fn / const method
#define NF_CONSTEVAL 64
#define NF_VARIADIC 128
#define NF_PURE     256
#define NF_MUSTUSE  512
#define NF_NORETURN 1024
#define NF_INLINE   2048
#define NF_METHOD   4096
#define NF_UNION    8192
#define NF_PACKED   16384
#define NF_MCONST   32768   // const method (self is const)
#define NF_SMETHOD  65536   // static method (no self)
#define NF_EXPORT   131072  // __attribute__((export_name)) : ival = export name sid
#define NF_PAREN    262144  // parenthesized expression
#define NF_FWD      524288  // forward declaration only
#define NF_GENERIC  1048576

struct Node {
    int kind;
    int op;
    int a;
    int b;
    int c;
    int d;
    int list;    // start index in g_list
    int nlist;
    int sid;
    int ty;      // sema: resolved type of an expression
    int tok;     // source location (token index)
    int flags;
    long ival;
};

// ── Types ────────────────────────────────────────────────────────────────────
#define TY_ERROR   1
#define TY_VOID    2
#define TY_BOOL    3
#define TY_CHAR    4
#define TY_I8      5
#define TY_I16     6
#define TY_I32     7
#define TY_I64     8
#define TY_U8      9
#define TY_U16     10
#define TY_U32     11
#define TY_U64     12
#define TY_F32     13
#define TY_F64     14
#define TY_PTR     20   // base, flags TYF_CONST
#define TY_REF     21   // base, n = region, sid = arena name, flags TYF_NULL / TYF_CONST
#define TY_ARRAY   22   // base, n = length (-1 unsized)
#define TY_SLICE   23   // base
#define TY_OPT     24   // base
#define TY_FN      25   // base = return, list = params, flags TYF_VARIADIC
#define TY_TUPLE   26   // list
#define TY_STRUCT  27   // n = struct def index
#define TY_ENUM    28   // n = struct def index (kind enum)
#define TY_NEWTYPE 29   // n = struct def index (kind newtype)
#define TY_NAMED   30   // unresolved: sid, list = type args, flags TYF_STRUCTKW
#define TY_GENERIC 31   // type parameter: sid
#define TY_TYPEOF  32   // n = expression node

#define TYF_CONST    1
#define TYF_NULL     2
#define TYF_VARIADIC 4
#define TYF_STRUCTKW 8
#define TYF_UNIONKW  16
#define TYF_ENUMKW   32

#define R_EXTERN 0
#define R_STACK  1
#define R_HEAP   2
#define R_STATIC 3
#define R_ARENA  4

struct Ty {
    int kind;
    int base;
    int n;
    int list;
    int nlist;
    int sid;
    int flags;
    int hnext;
};

// struct / union / enum / newtype definitions
#define SD_STRUCT  1
#define SD_UNION   2
#define SD_ENUM    3
#define SD_NEWTYPE 4

struct SDef {
    int kind;
    int sid;        // display name ("Vec<int>" for instances)
    int node;       // D_STRUCT / D_ENUM / D_NEWTYPE node
    int field0;     // first field in g_field
    int nfield;
    int size;
    int align;
    int done;       // layout computed
    int base;       // enum underlying type / newtype base
    int tmpl;       // generic template struct def (instances)
    int targs;      // instance: type arg list start (g_tyl)
    int ntargs;
    int ty;         // the TY_STRUCT type of this def
};

struct Field {
    int sid;
    int ty;
    int off;
    int node;
    long val;       // enum constant value
};

// ── Symbols ──────────────────────────────────────────────────────────────────
#define SYM_LOCAL  1
#define SYM_GLOBAL 2
#define SYM_FUNC   3
#define SYM_ENUMC  4   // d = field index
#define SYM_BIND   5   // match binding (a local)

struct Func {
    int sid;        // mangled name
    int node;       // D_FUNC
    int ty;         // TY_FN
    int owner;      // method owner SDef (0 if free function)
    int flags;
    int used;
    int widx;       // wasm function index
    int import;     // 1 if imported from the host
    int loc0;       // locals of this function: g_local[loc0 .. loc0+nloc)
    int nloc;
    int env;        // generic substitution env used to check it
    int checked;
    int tmpl;       // generic template func (instances)
    int origsid;    // unmangled name
    int ns;         // namespace sid
};

struct Glob {
    int sid;
    int node;
    int ty;
    int addr;       // linear-memory address
    int flags;
    int done;
};

#define LF_ADDR   1   // address taken / lives in the frame
#define LF_PARAM  2
#define LF_INIT   4
#define LF_CONST  8
#define LF_STATIC 16  // function-local static: lives in data

struct Local {
    int sid;
    int ty;
    int flags;
    int wl;         // wasm local index (scalars not in memory)
    int off;        // frame offset (LF_ADDR) / global address (LF_STATIC)
    int depth;
    int node;
};

struct Sym {
    int sid;
    int kind;
    int idx;
    int ty;
    int depth;
    int init;
};

// ── Globals shared across modules ────────────────────────────────────────────
extern unsigned char g_src[SRC_MAX];
extern int g_src_len;
extern int g_file_name[NFILE_MAX];
extern int g_file_start[NFILE_MAX];
extern int g_file_len[NFILE_MAX];
extern int g_nfile;

extern char g_sp[SP_MAX];
extern int g_sp_len;
extern int g_str_off[NSTR_MAX];
extern int g_str_len[NSTR_MAX];
extern int g_str_next[NSTR_MAX];
extern int g_nstr;
extern int g_kw[NSTR_MAX];

extern struct Tok g_tok[TOK_MAX];
extern int g_ntok;

extern struct Node g_node[NODE_MAX];
extern int g_nnode;
extern int g_list[LIST_MAX];
extern int g_nlist;
extern int g_top[LIST_MAX];   // top-level declarations
extern int g_ntop;

extern struct Ty g_ty[TY_MAX];
extern int g_nty;
extern int g_tyl[TYL_MAX];
extern int g_ntyl;

extern struct SDef g_sdef[NSTRUCT_MAX];
extern int g_nsdef;
extern struct Field g_field[NFIELD_MAX];
extern int g_nfield;
extern struct Func g_func[NFUNC_MAX];
extern int g_nfunc;
extern struct Glob g_glob[NGLOB_MAX];
extern int g_nglob;
extern struct Local g_local[NLOCAL_MAX];
extern int g_nlocal;

extern unsigned char g_out[OUT_MAX];
extern int g_out_len;
extern char g_diag[DIAG_MAX];
extern int g_diag_len;
extern int g_nerrors;
extern int g_nwarnings;
extern char g_msg[MSG_MAX];
extern int g_msg_len;
extern unsigned char g_tmp[TMP_MAX];
extern int g_tmp_len;

// well-known strings
extern int S_main;
extern int S_self;
extern int S_len;
extern int S_ptr;
extern int S_stack;
extern int S_heap;
extern int S_arena;
extern int S_static;
extern int S_some;
extern int S_none;
extern int S_is_null;
extern int S_is_none;
extern int S_default;
extern int S_attribute;
extern int S_export_name;
extern int S_defined;
extern int S_va_ptr;
extern int S_capacity;
extern int S_once;
extern int S_include;
extern int S_define;
extern int S_undef;
extern int S_if;
extern int S_ifdef;
extern int S_ifndef;
extern int S_elif;
extern int S_else;
extern int S_endif;
extern int S_pragma;
extern int S_error;
extern int S_warning;
extern int S_line;
extern int S_VA_ARGS;
extern int S_FILE;
extern int S_LINE;
extern int S_runtime;
extern int S_wild;

extern int g_rt_file;

// ── util.sc ──────────────────────────────────────────────────────────────────
void util_init();
int lit_len(const char* s);
int lit_char(const char* s, int i);
int intern_bytes(int from_src, int off, int len);   // from g_src (1) or g_tmp (0)
int intern_lit(const char* s);
int intern_tmp();
int str_char(int sid, int i);
int str_eq_lit(int sid, const char* s);
int str_concat3(int a, const char* mid, int b);
void tmp_reset();
void tmp_char(int c);
void tmp_str(int sid);
void tmp_lit(const char* s);
void tmp_int(long v);
void msg_reset();
void msg_lit(const char* s);
void msg_str(int sid);
void msg_int(long v);
void msg_char(int c);
void msg_type(int ty);
void diag_at(int file, int line, int col, int is_error);
void error_tok(int tok);
void warn_tok(int tok);
void error_node(int node);
void warn_node(int node);
void fatal(const char* s);
void out_byte(int b);
int ilog_hash(int a, int b);

// ── lex.sc / pp.sc ───────────────────────────────────────────────────────────
int add_file(int name_sid, int start, int len);
int vfs_find(int name_sid);
int load_vfs_file(int name_sid);
void pp_init();
void pp_define_lit(const char* name, const char* value);
void pp_run(int file);
void pp_finish();

// ── parse.sc ─────────────────────────────────────────────────────────────────
void parse_program();
int new_node(int kind, int tok);
int list_begin();
void list_push(int v);
int list_end(int mark, int node);
int clone_node(int n);
int tok_kind(int i);

// ── types.sc ─────────────────────────────────────────────────────────────────
void types_init();
int ty_prim(int kind);
int ty_ptr(int base, int flags);
int ty_ref(int base, int rgn, int arn, int flags);
int ty_array(int base, int n);
int ty_slice(int base);
int ty_opt(int base);
int ty_fn(int ret, int mark, int variadic);
int ty_tuple(int mark);
int ty_named(int sid, int mark, int flags);
int ty_generic(int sid);
int ty_struct(int sdef);
int ty_make(int kind, int base, int n, int sid, int flags, int mark);
int tyl_begin();
void tyl_push(int t);
int ty_size(int t);
int ty_align(int t);
int ty_is_int(int t);
int ty_is_float(int t);
int ty_is_arith(int t);
int ty_is_signed(int t);
int ty_bits(int t);
int ty_is_scalar(int t);
int ty_is_aggregate(int t);
int ty_kind(int t);
void layout_struct(int sd);

// ── sema.sc ──────────────────────────────────────────────────────────────────
void sema_program();
int resolve_type(int t);

// ── gen.sc ───────────────────────────────────────────────────────────────────
void gen_module();

// ── driver core ──────────────────────────────────────────────────────────────
int scc_compile_main(int main_file);
void scc_init();
void scc_preprocess(int main_file);
