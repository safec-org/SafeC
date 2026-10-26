// parse.sc — recursive-descent parser: g_tok → AST nodes in g_node.
// Grammar and error wording follow the C++ safec's Parser.cpp.
#include "scc.h"
#include "lex.h"
#include "parse.h"

struct Node g_node[NODE_MAX];
int g_nnode = 0;
int g_list[LIST_MAX];
int g_nlist = 0;
int g_top[LIST_MAX];
int g_ntop = 0;
int g_lstack[LIST_MAX];
int g_nlstack = 0;
int g_typename[NSTR_MAX];

int g_p = 1;          // current token
int g_ns = 0;         // current namespace (qualified sid), 0 = none
int g_silent = 0;

// ── nodes and lists ──────────────────────────────────────────────────────────
int new_node(int kind, int tok) {
    if (g_nnode + 1 >= NODE_MAX) { fatal("AST nodes"); return 0; }
    g_nnode = g_nnode + 1;
    int n = g_nnode;
    g_node[n].kind = kind;
    g_node[n].op = 0;
    g_node[n].a = 0;
    g_node[n].b = 0;
    g_node[n].c = 0;
    g_node[n].d = 0;
    g_node[n].list = 0;
    g_node[n].nlist = 0;
    g_node[n].sid = 0;
    g_node[n].ty = 0;
    g_node[n].tok = tok;
    g_node[n].flags = 0;
    g_node[n].ival = 0L;
    return n;
}

int list_begin() { return g_nlstack; }

void list_push(int v) {
    if (g_nlstack >= LIST_MAX) { fatal("list stack"); return; }
    g_lstack[g_nlstack] = v;
    g_nlstack = g_nlstack + 1;
}

int list_end(int mark, int node) {
    int n = g_nlstack - mark;
    if (g_nlist + n >= LIST_MAX) { fatal("lists"); g_nlstack = mark; return 0; }
    int start = g_nlist;
    int i = 0;
    while (i < n) { g_list[start + i] = g_lstack[mark + i]; i = i + 1; }
    g_nlist = start + n;
    g_nlstack = mark;
    if (node != 0) { g_node[node].list = start; g_node[node].nlist = n; }
    return start;
}

// deep copy of a subtree (generic instantiation)
// pack expansion while cloning a generic function for an instance:
// the pack parameter 'args' (type pack g_pk_tsid with g_pk_n entries of
// types g_pk_ty[]) becomes the parameters args0..argsN-1
int g_pk_n = -1;
int g_pk_tsid = 0;
int g_pk_psid = 0;
int g_pk_ty[32];

int pack_ident(int sid, int i, int tok) {
    tmp_reset();
    tmp_str(sid);
    tmp_int((long)i);
    int id = new_node(E_IDENT, tok);
    g_node[id].sid = intern_tmp();
    return id;
}

int clone_node(int n) {
    if (n == 0) { return 0; }
    if (g_pk_n >= 0) {
        int nk = g_node[n].kind;
        // 'args[2]' with a literal index: the parameter args2
        if (nk == E_INDEX && g_node[n].a != 0 && g_node[g_node[n].a].kind == E_IDENT && g_node[g_node[n].a].sid == g_pk_psid &&
            g_node[n].b != 0 && g_node[g_node[n].b].kind == E_INT) {
            return pack_ident(g_pk_psid, (int)g_node[g_node[n].b].ival, g_node[n].tok);
        }
        if (nk == E_SIZEOFPACK) {
            int sp = new_node(E_SIZEOFPACK, g_node[n].tok);
            g_node[sp] = g_node[n];
            g_node[sp].ival = 0L;
            if (g_node[n].sid == g_pk_tsid) { g_node[sp].ival = (long)g_pk_n; }
            return sp;
        }
    }
    int c = new_node(g_node[n].kind, g_node[n].tok);
    g_node[c] = g_node[n];
    g_node[c].ty = 0;
    int k = g_node[n].kind;
    // children in a/b/c/d are nodes, except where they hold other data
    int a_is_node = 1;
    int b_is_node = 1;
    int c_is_node = 1;
    int d_is_node = 1;
    if (k == E_CAST || k == E_SIZEOF_T || k == E_ALIGNOF || k == E_NEW || k == S_VAR ||
        k == D_FUNC || k == D_PARAM || k == D_GLOBAL || k == D_FIELD || k == D_TYPEDEF ||
        k == D_NEWTYPE || k == D_ENUM) { c_is_node = 0; }
    if (k == E_IDENT || k == E_CALL || k == E_UNIONINIT || k == S_VAR || k == D_FUNC ||
        k == P_NAME || k == P_VARIANT || k == E_MEMBER) { d_is_node = 0; }
    if (k == E_CALL || k == E_IDENT) { c_is_node = 0; }
    if (k == D_FUNC) { b_is_node = 0; }
    if (a_is_node == 1) { g_node[c].a = clone_node(g_node[n].a); }
    if (b_is_node == 1) { g_node[c].b = clone_node(g_node[n].b); }
    if (c_is_node == 1) { g_node[c].c = clone_node(g_node[n].c); }
    if (d_is_node == 1) { g_node[c].d = clone_node(g_node[n].d); }
    if (k == E_IDENT || k == E_CALL) { g_node[c].c = 0; g_node[c].d = 0; }
    if (k == S_VAR) { g_node[c].d = 0; }
    if (g_node[n].nlist > 0) {
        int mark = list_begin();
        int i = 0;
        while (i < g_node[n].nlist) {
            int el = g_list[g_node[n].list + i];
            i = i + 1;
            if (g_pk_n >= 0 && k == E_CALL && g_node[el].kind == E_IDENT && g_node[el].sid == g_pk_psid) {
                // forwarding 'args' to a call: args0, args1, ...
                int j = 0;
                while (j < g_pk_n) { list_push(pack_ident(g_pk_psid, j, g_node[el].tok)); j = j + 1; }
                continue;
            }
            if (g_pk_n >= 0 && k == D_FUNC && g_node[el].kind == D_PARAM && g_node[el].flags == 1) {
                int j = 0;
                while (j < g_pk_n) {
                    int pp = new_node(D_PARAM, g_node[el].tok);
                    g_node[pp] = g_node[el];
                    g_node[pp].flags = 0;
                    g_node[pp].sid = g_node[pack_ident(g_node[el].sid, j, g_node[el].tok)].sid;
                    g_node[pp].c = g_pk_ty[j];
                    list_push(pp);
                    j = j + 1;
                }
                continue;
            }
            list_push(clone_node(el));
        }
        list_end(mark, c);
    }
    return c;
}

// ── token helpers ────────────────────────────────────────────────────────────
int tok_kind(int i) {
    if (i >= g_ntok) { return T_EOF; }
    return g_tok[i].kind;
}

int cur() { return tok_kind(g_p); }
int peek(int k) { return tok_kind(g_p + k); }
int cur_sid() { return g_tok[g_p].sid; }

void advance() { if (g_p + 1 < g_ntok) { g_p = g_p + 1; } }

int at_end() { return cur() == T_EOF; }

int accept(int k) {
    if (cur() == k) { advance(); return 1; }
    return 0;
}

void perr(const char* s) {
    if (g_silent > 0) { g_nerrors = g_nerrors + 1; return; }
    msg_reset();
    msg_lit(s);
    error_tok(g_p);
}

void perr_tok_name(int i) {
    int k = tok_kind(i);
    if (k == T_IDENT) { msg_lit("identifier"); return; }
    if (k == T_INT) { msg_lit("integer literal"); return; }
    if (k == T_FLOAT) { msg_lit("float literal"); return; }
    if (k == T_STRING) { msg_lit("string literal"); return; }
    if (k == T_CHAR) { msg_lit("char literal"); return; }
    if (k == T_EOF) { msg_lit("end of file"); return; }
    if (k >= K_FIRST) { msg_lit("'"); msg_str(g_tok[i].sid); msg_lit("'"); return; }
    tmp_reset();
    tmp_punct(k);
    msg_lit("'");
    int j = 0;
    while (j < g_tmp_len) { msg_char((int)g_tmp[j]); j = j + 1; }
    msg_lit("'");
}

// text of token i for "unexpected token 'x'" messages
void msg_tok_text(int i) {
    int k = tok_kind(i);
    if (k == T_IDENT || k == T_INT || k == T_FLOAT || k >= K_FIRST) { msg_str(g_tok[i].sid); return; }
    if (k == T_STRING || k == T_CHAR) { msg_str(g_tok[i].sid); return; }
    if (k == T_EOF) { return; }
    tmp_reset();
    tmp_punct(k);
    int j = 0;
    while (j < g_tmp_len) { msg_char((int)g_tmp[j]); j = j + 1; }
}

void expect_msg(int k, const char* s) {
    if (cur() == k) { advance(); return; }
    perr(s);
}

void expect_semi() {
    if (cur() == T_SEMI) { advance(); return; }
    if (g_silent > 0) { g_nerrors = g_nerrors + 1; return; }
    msg_reset();
    msg_lit("expected ';', got ");
    perr_tok_name(g_p);
    error_tok(g_p);
}

void expect_kind(int k) {
    if (cur() == k) { advance(); return; }
    if (g_silent > 0) { g_nerrors = g_nerrors + 1; return; }
    msg_reset();
    msg_lit("expected ");
    tmp_reset();
    tmp_punct(k);
    msg_lit("'");
    int j = 0;
    while (j < g_tmp_len) { msg_char((int)g_tmp[j]); j = j + 1; }
    msg_lit("', got ");
    perr_tok_name(g_p);
    error_tok(g_p);
}

int is_word(int i) {
    int k = tok_kind(i);
    return k == T_IDENT || (k >= K_FIRST && k != K_OPERATOR);
}

int is_attr(int i) { return tok_kind(i) == T_IDENT && g_tok[i].sid == S_attribute; }

// diagnostics checkpoint for speculative parses
int g_save_dlen = 0;
int g_save_nerr = 0;
int g_save_nwarn = 0;

// ── keywords ─────────────────────────────────────────────────────────────────
void kw(const char* s, int k) { g_kw[intern_lit(s)] = k; }

void kw_init() {
    kw("auto", K_AUTO); kw("break", K_BREAK); kw("case", K_CASE); kw("char", K_CHAR);
    kw("const", K_CONST); kw("continue", K_CONTINUE); kw("default", K_DEFAULT);
    kw("do", K_DO); kw("double", K_DOUBLE); kw("else", K_ELSE); kw("enum", K_ENUM);
    kw("extern", K_EXTERN); kw("float", K_FLOAT); kw("for", K_FOR); kw("goto", K_GOTO);
    kw("if", K_IF); kw("inline", K_INLINE); kw("int", K_INT); kw("long", K_LONG);
    kw("register", K_REGISTER); kw("restrict", K_RESTRICT); kw("return", K_RETURN);
    kw("short", K_SHORT); kw("signed", K_SIGNED); kw("sizeof", K_SIZEOF);
    kw("static", K_STATIC); kw("struct", K_STRUCT); kw("switch", K_SWITCH);
    kw("typedef", K_TYPEDEF); kw("union", K_UNION); kw("unsigned", K_UNSIGNED);
    kw("void", K_VOID); kw("volatile", K_VOLATILE); kw("while", K_WHILE);
    kw("bool", K_BOOL); kw("_Bool", K_BOOL); kw("true", K_TRUE); kw("false", K_FALSE);
    kw("null", K_NULL); kw("region", K_REGION); kw("unsafe", K_UNSAFE);
    kw("consteval", K_CONSTEVAL); kw("generic", K_GENERIC);
    kw("static_assert", K_STATIC_ASSERT); kw("_Static_assert", K_STATIC_ASSERT);
    kw("self", K_SELF); kw("operator", K_OPERATOR); kw("new", K_NEW);
    kw("arena_reset", K_ARENA_RESET); kw("arena_destroy", K_ARENA_DESTROY);
    kw("arena_mark", K_ARENA_MARK); kw("arena_free_to", K_ARENA_FREE_TO);
    kw("tuple", K_TUPLE); kw("defer", K_DEFER); kw("errdefer", K_ERRDEFER);
    kw("match", K_MATCH); kw("packed", K_PACKED); kw("try", K_TRY);
    kw("must_use", K_MUST_USE); kw("fn", K_FN); kw("alignof", K_ALIGNOF);
    kw("typeof", K_TYPEOF); kw("noreturn", K_NORETURN); kw("_Noreturn", K_NORETURN);
    kw("pure", K_PURE); kw("newtype", K_NEWTYPE); kw("trait", K_TRAIT);
    kw("namespace", K_NAMESPACE); kw("constinit", K_CONSTINIT); kw("asm", K_ASM);
    kw("spawn", K_SPAWN); kw("fieldcount", K_FIELDCOUNT); kw("fn_eval", K_FN_EVAL);
}

// ── attributes ───────────────────────────────────────────────────────────────
// __attribute__((a, b(x), ...)) — collects export_name / noreturn / pure /
// warn_unused_result; everything else is accepted and ignored.
int g_attr_flags = 0;
int g_attr_export = 0;

int parse_attributes() {
    int any = 0;
    while (is_attr(g_p)) {
        any = 1;
        advance();
        expect_msg(T_LPAREN, "expected '(' after __attribute__");
        expect_msg(T_LPAREN, "expected '((' after __attribute__");
        while (!at_end() && cur() != T_RPAREN) {
            int name = 0;
            if (is_word(g_p)) { name = cur_sid(); advance(); }
            else { advance(); continue; }
            if (str_eq_lit(name, "noreturn") == 1 || str_eq_lit(name, "__noreturn__") == 1) { g_attr_flags = g_attr_flags | NF_NORETURN; }
            if (str_eq_lit(name, "pure") == 1 || str_eq_lit(name, "const") == 1) { g_attr_flags = g_attr_flags | NF_PURE; }
            if (str_eq_lit(name, "warn_unused_result") == 1) { g_attr_flags = g_attr_flags | NF_MUSTUSE; }
            if (str_eq_lit(name, "packed") == 1) { g_attr_flags = g_attr_flags | NF_PACKED; }
            if (cur() == T_LPAREN) {
                advance();
                if (name == S_export_name) {
                    g_attr_flags = g_attr_flags | NF_EXPORT;
                    if (cur() == T_STRING) { g_attr_export = cur_sid(); }
                }
                int depth = 1;
                while (!at_end() && depth > 0) {
                    if (cur() == T_LPAREN) { depth = depth + 1; }
                    if (cur() == T_RPAREN) { depth = depth - 1; }
                    advance();
                }
            }
            if (cur() == T_COMMA) { advance(); }
        }
        expect_msg(T_RPAREN, "expected ')' closing __attribute__");
        expect_msg(T_RPAREN, "expected '))' closing __attribute__");
    }
    return any;
}

// ── types ────────────────────────────────────────────────────────────────────
int is_type_start_tok(int i) {
    int k = tok_kind(i);
    if (k == K_INT || k == K_VOID || k == K_CHAR || k == K_FLOAT || k == K_DOUBLE ||
        k == K_BOOL || k == K_LONG || k == K_SHORT || k == K_UNSIGNED || k == K_SIGNED ||
        k == K_STRUCT || k == K_UNION || k == K_ENUM || k == K_CONST || k == K_VOLATILE ||
        k == K_TUPLE || k == K_FN || k == K_TYPEOF || k == K_AUTO || k == T_AMP ||
        k == T_QAMP || k == T_QUESTION || k == T_IDENT) { return 1; }
    return 0;
}

int parse_type();
int parse_ternary();
int parse_assign();
int parse_expr();
int parse_unary();
int parse_stmt();
int parse_block();

// [N] suffixes (outermost first)
int parse_array_suffix(int base) {
    int dims[16];
    int nd = 0;
    while (cur() == T_LBRACKET) {
        advance();
        int n = -1;
        if (cur() != T_RBRACKET) {
            int e = parse_ternary();
            if (const_fold(e) == 1) { n = (int)g_fold; }
            else { n = -10 - e; }   // resolved by sema
        }
        expect_msg(T_RBRACKET, "expected ']'");
        if (nd < 16) { dims[nd] = n; nd = nd + 1; }
    }
    while (nd > 0) {
        nd = nd - 1;
        base = ty_array(base, dims[nd]);
    }
    return base;
}

int parse_declarator(int base) {
    while (cur() == T_STAR) {
        advance();
        int fl = 0;
        while (cur() == K_CONST || cur() == K_RESTRICT || cur() == K_VOLATILE) {
            if (cur() == K_CONST) { fl = TYF_CONST; }
            advance();
        }
        base = ty_ptr(base, fl);
    }
    return parse_array_suffix(base);
}

int g_base_const = 0;   // the last base type had a leading/trailing 'const'

int parse_type_args(int sid, int flags) {
    int mark = tyl_begin();
    // the args are parsed before being pushed (nested lists)
    int args[16];
    int na = 0;
    advance(); // '<'
    while (!at_end() && cur() != T_GT && cur() != T_SHR) {
        int t = parse_type();
        if (na < 16) { args[na] = t; na = na + 1; }
        if (accept(T_COMMA) == 0) { break; }
    }
    if (cur() == T_SHR) { g_tok[g_p].kind = T_GT; }
    else { expect_msg(T_GT, "expected '>' closing generic struct type arguments"); }
    mark = tyl_begin();
    int i = 0;
    while (i < na) { tyl_push(args[i]); i = i + 1; }
    return ty_named(sid, mark, flags);
}

int parse_base_type() {
    int is_unsigned = 0;
    int has_sign = 0;
    g_base_const = 0;
    while (cur() == K_CONST || cur() == K_VOLATILE || cur() == K_RESTRICT || cur() == K_REGISTER) {
        if (cur() == K_CONST) { g_base_const = 1; }
        advance();
    }
    if (cur() == K_UNSIGNED) { advance(); is_unsigned = 1; has_sign = 1; }
    else if (cur() == K_SIGNED) { advance(); has_sign = 1; }
    int k = cur();
    int t = 0;
    if (k == K_VOID) { advance(); t = TY_VOID; }
    else if (k == K_BOOL) { advance(); t = TY_BOOL; }
    else if (k == K_CHAR) { advance(); if (is_unsigned == 1) { t = TY_U8; } else if (has_sign == 1) { t = TY_I8; } else { t = TY_CHAR; } }
    else if (k == K_SHORT) { advance(); if (cur() == K_INT) { advance(); } if (is_unsigned == 1) { t = TY_U16; } else { t = TY_I16; } }
    else if (k == K_INT) { advance(); if (is_unsigned == 1) { t = TY_U32; } else { t = TY_I32; } }
    else if (k == K_LONG) {
        advance();
        if (cur() == K_LONG) { advance(); }
        if (cur() == K_INT) { advance(); }
        if (cur() == K_DOUBLE) { advance(); t = TY_F64; }
        else if (is_unsigned == 1) { t = TY_U64; } else { t = TY_I64; }
    }
    else if (k == K_FLOAT) { advance(); t = TY_F32; }
    else if (k == K_DOUBLE) { advance(); t = TY_F64; }
    else if (k == K_STRUCT || k == K_UNION) {
        int fl = TYF_STRUCTKW;
        if (k == K_UNION) { fl = TYF_UNIONKW; }
        advance();
        if (cur() == T_IDENT) {
            int name = cur_sid();
            advance();
            if (cur() == T_LT) { t = parse_type_args(name, fl); }
            else { t = ty_named(name, -1, fl); }
        } else {
            if (k == K_UNION) { perr("expected union name"); } else { perr("expected struct name"); }
            t = TY_ERROR;
        }
    }
    else if (k == K_ENUM) {
        advance();
        if (cur() == T_IDENT) { t = ty_named(cur_sid(), -1, TYF_ENUMKW); advance(); }
        else { perr("expected enum name"); t = TY_ERROR; }
    }
    else if (k == K_TUPLE) {
        advance();
        expect_msg(T_LPAREN, "expected '(' after 'tuple'");
        int el[32];
        int ne = 0;
        while (!at_end() && cur() != T_RPAREN) {
            int et = parse_type();
            if (ne < 32) { el[ne] = et; ne = ne + 1; }
            if (accept(T_COMMA) == 0) { break; }
        }
        expect_msg(T_RPAREN, "expected ')' closing tuple type");
        int mark = tyl_begin();
        int i = 0;
        while (i < ne) { tyl_push(el[i]); i = i + 1; }
        t = ty_tuple(mark);
    }
    else if (k == K_FN) {
        advance();
        int ret = 0;
        if (cur() == T_AMP || cur() == T_QAMP || cur() == T_QUESTION ||
            (cur() == K_CONST && (peek(1) == T_AMP || peek(1) == T_QAMP))) { ret = parse_type(); }
        else { ret = parse_declarator(parse_base_type()); }
        expect_msg(T_LPAREN, "expected '(' after 'fn' return type");
        int ps[64];
        int np = 0;
        int variadic = 0;
        if (cur() != T_RPAREN) {
            while (1) {
                if (cur() == T_ELLIPSIS) { advance(); variadic = 1; break; }
                if (cur() == K_VOID && peek(1) == T_RPAREN) { advance(); break; }
                int pt = parse_type();
                if (np < 64) { ps[np] = pt; np = np + 1; }
                if (cur() == T_IDENT) { advance(); }
                if (accept(T_COMMA) == 0) { break; }
            }
        }
        expect_msg(T_RPAREN, "expected ')' closing fn parameter list");
        int mark = tyl_begin();
        int i = 0;
        while (i < np) { tyl_push(ps[i]); i = i + 1; }
        int ft = ty_fn(ret, mark, variadic);
        t = ty_ref(ft, R_STATIC, 0, TYF_CONST);
    }
    else if (k == K_TYPEOF) {
        advance();
        expect_msg(T_LPAREN, "expected '(' after 'typeof'");
        int e = parse_expr();
        expect_msg(T_RPAREN, "expected ')' after typeof expression");
        t = ty_make(TY_TYPEOF, 0, e, 0, 0, -1);
    }
    else if (k == T_IDENT) {
        int name = cur_sid();
        advance();
        if (cur() == T_LT && g_typename[name] == 1) { t = parse_type_args(name, 0); }
        else { t = ty_named(name, -1, 0); }
    }
    else {
        if (has_sign == 1) {
            if (is_unsigned == 1) { t = TY_U32; } else { t = TY_I32; }
        } else {
            if (g_silent > 0) { g_nerrors = g_nerrors + 1; }
            else {
                msg_reset();
                msg_lit("expected type, got '");
                msg_tok_text(g_p);
                msg_lit("'");
                error_tok(g_p);
            }
            t = TY_ERROR;
        }
    }
    while (cur() == K_CONST || cur() == K_VOLATILE) {
        if (cur() == K_CONST) { g_base_const = 1; }
        advance();
    }
    return t;
}

int g_region_arena = 0;

int parse_region() {
    g_region_arena = 0;
    if (cur() == T_IDENT && cur_sid() == S_stack) { advance(); return R_STACK; }
    if (cur() == T_IDENT && cur_sid() == S_heap) { advance(); return R_HEAP; }
    if (cur() == K_STATIC) { advance(); return R_STATIC; }
    if (cur() == T_IDENT && cur_sid() == S_arena) {
        advance();
        expect_msg(T_LT, "expected '<' after 'arena'");
        if (cur() == T_IDENT) { g_region_arena = cur_sid(); advance(); }
        else { perr("expected arena region name"); }
        expect_msg(T_GT, "expected '>' after arena region name");
        return R_ARENA;
    }
    return R_EXTERN;
}

int at_region() {
    if (cur() == T_IDENT && (cur_sid() == S_stack || cur_sid() == S_heap || cur_sid() == S_arena)) {
        // 'arena' must be followed by '<' to be the region keyword
        if (cur_sid() == S_arena && peek(1) != T_LT) { return 0; }
        return 1;
    }
    if (cur() == K_STATIC) { return 1; }
    return 0;
}

// after '&' / '?&'
int parse_ref_type(int nullable, int lead_const) {
    int arn = 0;
    int rgn = R_EXTERN;
    if (at_region() == 1) {
        rgn = parse_region();
        arn = g_region_arena;
    }
    int is_const = lead_const;
    if (cur() == K_CONST) { advance(); is_const = 1; }
    int base = 0;
    if (cur() == T_AMP || cur() == T_QAMP || cur() == T_ANDAND) { base = parse_type(); }
    else {
        base = parse_base_type();
        if (g_base_const == 1) { is_const = 1; }
    }
    base = parse_declarator(base);
    int fl = 0;
    if (nullable == 1) { fl = fl | TYF_NULL; }
    if (is_const == 1) { fl = fl | TYF_CONST; }
    return ty_ref(base, rgn, arn, fl);
}

int parse_type() {
    if (cur() == K_CONST && (peek(1) == T_AMP || peek(1) == T_QAMP)) {
        advance();
        int nul = 0;
        if (cur() == T_QAMP) { nul = 1; }
        advance();
        return parse_ref_type(nul, 1);
    }
    if (cur() == K_VOLATILE && (peek(1) == T_AMP || peek(1) == T_QAMP)) {
        advance();
        int nul = 0;
        if (cur() == T_QAMP) { nul = 1; }
        advance();
        return parse_ref_type(nul, 0);
    }
    if (cur() == T_QAMP) { advance(); return parse_ref_type(1, 0); }
    if (cur() == T_AMP) { advance(); return parse_ref_type(0, 0); }
    if (cur() == T_ANDAND) {
        advance();
        int inner = parse_ref_type(0, 0);
        return ty_ref(inner, R_EXTERN, 0, 0);
    }
    if (cur() == T_QUESTION) {
        advance();
        int inner = parse_declarator(parse_base_type());
        return ty_opt(inner);
    }
    if (cur() == T_LBRACKET && peek(1) == T_RBRACKET) {
        advance(); advance();
        return ty_slice(parse_type());
    }
    int base = parse_base_type();
    return parse_declarator(base);
}

// 'R (*name)(P...)' declarator; returns the function-reference type or 0
int g_fp_name = 0;
int try_fnptr_declarator(int ret) {
    if (cur() != T_LPAREN || peek(1) != T_STAR || is_word(g_p + 2) == 0 || peek(3) != T_RPAREN || peek(4) != T_LPAREN) { return 0; }
    advance(); advance();
    g_fp_name = cur_sid();
    advance(); advance(); advance();
    int ps[64];
    int np = 0;
    int variadic = 0;
    if (cur() == K_VOID && peek(1) == T_RPAREN) { advance(); }
    else if (cur() != T_RPAREN) {
        while (1) {
            if (cur() == T_ELLIPSIS) { advance(); variadic = 1; break; }
            int pt = parse_type();
            if (np < 64) { ps[np] = pt; np = np + 1; }
            if (cur() == T_IDENT) { advance(); }
            if (accept(T_COMMA) == 0) { break; }
        }
    }
    expect_msg(T_RPAREN, "expected ')' closing function-pointer parameter list");
    int mark = tyl_begin();
    int i = 0;
    while (i < np) { tyl_push(ps[i]); i = i + 1; }
    return ty_ref(ty_fn(ret, mark, variadic), R_STATIC, 0, TYF_CONST);
}

// ── constant folding of literal arithmetic (array sizes) ─────────────────────
long g_fold = 0L;

// folds literal arithmetic; the value is left in g_fold
int const_fold(int e) {
    int k = g_node[e].kind;
    if (k == E_INT || k == E_CHAR) { g_fold = g_node[e].ival; return 1; }
    long a = 0L;
    long b = 0L;
    if (k == E_UNARY) {
        if (const_fold(g_node[e].a) == 0) { return 0; }
        a = g_fold;
        if (g_node[e].op == T_MINUS) { g_fold = 0L - a; return 1; }
        if (g_node[e].op == T_PLUS) { g_fold = a; return 1; }
        if (g_node[e].op == T_TILDE) { g_fold = ~a; return 1; }
        return 0;
    }
    if (k == E_BINARY) {
        if (const_fold(g_node[e].a) == 0) { return 0; }
        a = g_fold;
        if (const_fold(g_node[e].b) == 0) { return 0; }
        b = g_fold;
        int op = g_node[e].op;
        if (op == T_PLUS) { g_fold = a + b; return 1; }
        if (op == T_MINUS) { g_fold = a - b; return 1; }
        if (op == T_STAR) { g_fold = a * b; return 1; }
        if (op == T_SLASH) { if (b == 0L) { return 0; } g_fold = a / b; return 1; }
        if (op == T_PERCENT) { if (b == 0L) { return 0; } g_fold = a % b; return 1; }
        if (op == T_SHL) { g_fold = a << b; return 1; }
        if (op == T_SHR) { g_fold = a >> b; return 1; }
        if (op == T_AMP) { g_fold = a & b; return 1; }
        if (op == T_PIPE) { g_fold = a | b; return 1; }
        if (op == T_CARET) { g_fold = a ^ b; return 1; }
        return 0;
    }
    return 0;
}

// ── expressions ──────────────────────────────────────────────────────────────
int mk_binary(int op, int a, int b, int tok) {
    int n = new_node(E_BINARY, tok);
    g_node[n].op = op;
    g_node[n].a = a;
    g_node[n].b = b;
    return n;
}

int can_start_cast_operand(int k) {
    return k == T_INT || k == T_FLOAT || k == T_STRING || k == T_CHAR || k == K_TRUE ||
           k == K_FALSE || k == K_NULL || k == T_IDENT || k == K_SELF || k == T_LPAREN ||
           k == T_LBRACE || k == T_MINUS || k == T_PLUS || k == T_BANG || k == T_TILDE ||
           k == T_INC || k == T_DEC || k == T_AMP || k == T_STAR || k == K_SIZEOF || k == K_NEW;
}

int parse_initlist() {
    int n = new_node(E_INITLIST, g_p);
    advance(); // '{'
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        if (cur() == T_DOT && peek(1) == T_IDENT) {
            int d = new_node(E_DESIG, g_p);
            advance();
            g_node[d].sid = cur_sid();
            advance();
            if (cur() == T_COLON) { advance(); }
            else { expect_msg(T_ASSIGN, "expected '=' after '.field' designator"); }
            g_node[d].a = parse_assign();
            list_push(d);
        } else if (cur() == T_LBRACKET) {
            int d = new_node(E_DESIG, g_p);
            advance();
            int ie = parse_ternary();
            if (const_fold(ie) == 0) { perr("expected constant array designator"); }
            g_node[d].ival = g_fold;
            g_node[d].flags = 1;
            expect_msg(T_RBRACKET, "expected ']' closing array designator");
            if (cur() == T_COLON) { advance(); }
            else { expect_msg(T_ASSIGN, "expected '=' after array designator"); }
            g_node[d].a = parse_assign();
            list_push(d);
        } else {
            list_push(parse_assign());
        }
        if (accept(T_COMMA) == 0) { break; }
    }
    expect_msg(T_RBRACE, "expected '}' closing initializer");
    list_end(mark, n);
    return n;
}

int parse_region_arg() {
    int r = 0;
    if (accept(T_LT) == 1) {
        if (cur() == T_IDENT) { r = cur_sid(); advance(); }
        expect_msg(T_GT, "expected '>' closing region parameter");
    }
    return r;
}

int parse_match_expr();

int parse_primary() {
    int t = g_p;
    int k = cur();
    if (k == T_INT) {
        int n = new_node(E_INT, t);
        g_node[n].ival = g_tok[t].ival;
        g_node[n].flags = g_tok[t].flags & (TF_UNS | TF_LONG);
        advance();
        return n;
    }
    if (k == T_FLOAT) {
        int n = new_node(E_FLOAT, t);
        g_node[n].ival = g_tok[t].ival;
        g_node[n].flags = g_tok[t].flags & TF_F32;
        advance();
        return n;
    }
    if (k == T_STRING) {
        int n = new_node(E_STR, t);
        int s = cur_sid();
        advance();
        if (cur() == T_STRING) {
            tmp_reset();
            tmp_str(s);
            while (cur() == T_STRING) { tmp_str(cur_sid()); advance(); }
            s = intern_tmp();
        }
        g_node[n].sid = s;
        return n;
    }
    if (k == T_CHAR) {
        int n = new_node(E_CHAR, t);
        g_node[n].ival = g_tok[t].ival;
        advance();
        return n;
    }
    if (k == K_TRUE || k == K_FALSE) {
        int n = new_node(E_BOOL, t);
        if (k == K_TRUE) { g_node[n].ival = 1L; }
        advance();
        return n;
    }
    if (k == K_NULL) { advance(); return new_node(E_NULL, t); }
    if (k == T_IDENT || k == K_SELF) {
        int name = cur_sid();
        advance();
        if (cur() == T_COLONCOLON && is_word(g_p + 1)) {
            tmp_reset();
            tmp_str(name);
            while (cur() == T_COLONCOLON && is_word(g_p + 1)) {
                advance();
                tmp_lit("::");
                tmp_str(cur_sid());
                advance();
            }
            name = intern_tmp();
        }
        int n = new_node(E_IDENT, t);
        g_node[n].sid = name;
        return n;
    }
    if (k == T_LPAREN) {
        advance();
        int e = parse_assign();
        int is_comma = 0;
        while (cur() == T_COMMA) {
            int ct = g_p;
            advance();
            int r = parse_assign();
            int c = new_node(E_COMMA, ct);
            g_node[c].a = e;
            g_node[c].b = r;
            e = c;
            is_comma = 1;
        }
        expect_msg(T_RPAREN, "expected ')' closing parenthesized expression");
        g_node[e].flags = g_node[e].flags | NF_PAREN;
        return e;
    }
    if (k == T_LBRACE) { return parse_initlist(); }
    if (k == K_NEW) {
        advance();
        int n = new_node(E_NEW, t);
        g_node[n].sid = parse_region_arg();
        g_node[n].c = parse_base_type();
        if (accept(T_LBRACKET) == 1) {
            g_node[n].a = parse_expr();
            expect_msg(T_RBRACKET, "expected ']' closing 'new' array size");
        }
        return n;
    }
    if (k == K_ARENA_RESET || k == K_ARENA_DESTROY || k == K_ARENA_MARK || k == K_ARENA_FREE_TO) {
        advance();
        int n = new_node(E_ARENA, t);
        g_node[n].op = k;
        g_node[n].sid = parse_region_arg();
        expect_msg(T_LPAREN, "expected '(' for arena operation");
        if (k == K_ARENA_FREE_TO) { g_node[n].a = parse_assign(); }
        expect_msg(T_RPAREN, "expected ')' for arena operation");
        return n;
    }
    if (k == K_SPAWN || k == K_FN_EVAL) {
        advance();
        int n = new_node(E_SPAWN, t);
        if (k == K_FN_EVAL) { g_node[n].kind = E_FNEVAL; expect_msg(T_LPAREN, "expected '(' after 'fn_eval'"); }
        else { expect_msg(T_LPAREN, "expected '(' after 'spawn'"); }
        g_node[n].a = parse_assign();
        if (k == K_FN_EVAL) { expect_msg(T_COMMA, "expected ',' after fn_eval's object argument"); }
        else { expect_msg(T_COMMA, "expected ',' separating spawn arguments"); }
        g_node[n].b = parse_assign();
        if (k == K_FN_EVAL) { expect_msg(T_RPAREN, "expected ')' closing 'fn_eval'"); }
        else { expect_msg(T_RPAREN, "expected ')' closing spawn"); }
        return n;
    }
    if (g_silent > 0) { g_nerrors = g_nerrors + 1; }
    else {
        msg_reset();
        msg_lit("unexpected token '");
        msg_tok_text(g_p);
        msg_lit("' in expression");
        error_tok(g_p);
    }
    if (!at_end()) { advance(); }
    return new_node(E_INT, t);
}

int parse_postfix() {
    int e = parse_primary();
    while (1) {
        int t = g_p;
        int k = cur();
        if (k == T_LBRACKET) {
            advance();
            if (cur() == T_DOTDOT) {
                advance();
                int en = 0;
                if (cur() != T_RBRACKET) { en = parse_expr(); }
                expect_msg(T_RBRACKET, "expected ']'");
                int s = new_node(E_SLICE, t);
                g_node[s].a = e;
                g_node[s].c = en;
                e = s;
                continue;
            }
            int idx = parse_expr();
            if (cur() == T_DOTDOT) {
                advance();
                int en = 0;
                if (cur() != T_RBRACKET) { en = parse_expr(); }
                expect_msg(T_RBRACKET, "expected ']'");
                int s = new_node(E_SLICE, t);
                g_node[s].a = e;
                g_node[s].b = idx;
                g_node[s].c = en;
                e = s;
                continue;
            }
            expect_msg(T_RBRACKET, "expected ']'");
            int s = new_node(E_INDEX, t);
            g_node[s].a = e;
            g_node[s].b = idx;
            e = s;
            continue;
        }
        if (k == T_LPAREN) {
            advance();
            int c = new_node(E_CALL, t);
            g_node[c].a = e;
            int mark = list_begin();
            while (!at_end() && cur() != T_RPAREN) {
                list_push(parse_assign());
                if (accept(T_COMMA) == 0) { break; }
            }
            expect_msg(T_RPAREN, "expected ')' closing call");
            list_end(mark, c);
            e = c;
            continue;
        }
        if (k == T_DOT || k == T_ARROW) {
            advance();
            int m = new_node(E_MEMBER, t);
            g_node[m].a = e;
            if (k == T_ARROW) { g_node[m].flags = NF_ARROW; }
            if (cur() == T_IDENT || cur() == K_DEFAULT || (is_word(g_p) && peek(1) == T_LPAREN)) {
                g_node[m].sid = cur_sid();
                advance();
            } else if (cur() == T_INT && k == T_DOT) {
                g_node[m].sid = cur_sid();
                g_node[m].ival = g_tok[g_p].ival;
                g_node[m].op = 1;    // tuple index
                advance();
            } else {
                perr("expected field name");
                advance();
            }
            e = m;
            continue;
        }
        if (k == T_INC || k == T_DEC) {
            advance();
            int u = new_node(E_POSTFIX, t);
            g_node[u].op = k;
            g_node[u].a = e;
            e = u;
            continue;
        }
        if (k == T_BANG) {
            advance();
            int u = new_node(E_FORCE, t);
            g_node[u].a = e;
            e = u;
            continue;
        }
        break;
    }
    return e;
}

int parse_cast() {
    if (cur() == T_LPAREN && is_type_start_tok(g_p + 1) == 1) {
        int saved = g_p;
        if (peek(1) == T_IDENT && peek(2) == T_RPAREN && g_typename[g_tok[g_p + 1].sid] == 0) {
            int after = peek(3);
            if (after == T_PLUS || after == T_MINUS || after == T_STAR || after == T_AMP) { return parse_postfix(); }
        }
        int sdl = g_diag_len;
        int sne = g_nerrors;
        int snw = g_nwarnings;
        g_silent = g_silent + 1;
        advance();
        int t = parse_type();
        int ok = 0;
        if (cur() == T_RPAREN && can_start_cast_operand(peek(1)) == 1 && g_nerrors == sne) { ok = 1; }
        g_silent = g_silent - 1;
        g_diag_len = sdl;
        g_nerrors = sne;
        g_nwarnings = snw;
        if (ok == 1) {
            advance();
            int c = new_node(E_CAST, saved);
            g_node[c].c = t;
            g_node[c].a = parse_unary();
            return c;
        }
        g_p = saved;
    }
    return parse_postfix();
}

int parse_unary() {
    int t = g_p;
    int k = cur();
    if (k == T_MINUS || k == T_BANG || k == T_TILDE) {
        advance();
        int n = new_node(E_UNARY, t);
        g_node[n].op = k;
        g_node[n].a = parse_cast();
        return n;
    }
    if (k == T_INC || k == T_DEC) {
        advance();
        int n = new_node(E_UNARY, t);
        g_node[n].op = k;
        g_node[n].a = parse_unary();
        return n;
    }
    if (k == T_AMP) {
        advance();
        int n = new_node(E_ADDR, t);
        g_node[n].a = parse_cast();
        return n;
    }
    if (k == T_STAR) {
        advance();
        int n = new_node(E_DEREF, t);
        g_node[n].a = parse_cast();
        return n;
    }
    if (k == K_SIZEOF) {
        advance();
        if (cur() == T_ELLIPSIS) {
            // sizeof...(T): the number of entries of pack T
            advance();
            expect_msg(T_LPAREN, "expected '(' after 'sizeof...'");
            int sp = new_node(E_SIZEOFPACK, t);
            g_node[sp].sid = cur_sid();
            advance();
            expect_msg(T_RPAREN, "expected ')' after sizeof... pack name");
            return sp;
        }
        int expr_operand = 0;
        if (peek(1) == T_IDENT && (peek(2) == T_DOT || peek(2) == T_ARROW || peek(2) == T_LBRACKET)) { expr_operand = 1; }
        if (cur() == T_LPAREN && is_type_start_tok(g_p + 1) == 1 && expr_operand == 0) {
            advance();
            int n = new_node(E_SIZEOF_T, t);
            g_node[n].c = parse_type();
            expect_msg(T_RPAREN, "expected ')' after sizeof type");
            return n;
        }
        int n = new_node(E_SIZEOF_E, t);
        g_node[n].a = parse_unary();
        return n;
    }
    if (k == K_ALIGNOF) {
        advance();
        expect_msg(T_LPAREN, "expected '(' after 'alignof'");
        int n = new_node(E_ALIGNOF, t);
        g_node[n].c = parse_type();
        expect_msg(T_RPAREN, "expected ')' after alignof type");
        return n;
    }
    if (k == K_TRY) {
        advance();
        int n = new_node(E_TRY, t);
        g_node[n].a = parse_unary();
        return n;
    }
    if (k == K_MATCH) { return parse_match_expr(); }
    return parse_cast();
}

int bin_prec(int k) {
    if (k == T_STAR || k == T_SLASH || k == T_PERCENT || k == T_STARPIPE || k == T_STARPCT) { return 10; }
    if (k == T_PLUS || k == T_MINUS || k == T_PLUSPIPE || k == T_MINUSPIPE || k == T_PLUSPCT || k == T_MINUSPCT) { return 9; }
    if (k == T_SHL || k == T_SHR) { return 8; }
    if (k == T_LT || k == T_GT || k == T_LE || k == T_GE) { return 7; }
    if (k == T_EQ || k == T_NE) { return 6; }
    if (k == T_AMP) { return 5; }
    if (k == T_CARET) { return 4; }
    if (k == T_PIPE) { return 3; }
    if (k == T_ANDAND) { return 2; }
    if (k == T_OROR) { return 1; }
    return 0;
}

int parse_binary(int minp) {
    int l = parse_unary();
    while (1) {
        int k = cur();
        int p = bin_prec(k);
        if (p == 0 || p < minp) { break; }
        int t = g_p;
        advance();
        int r = parse_binary(p + 1);
        l = mk_binary(k, l, r, t);
    }
    return l;
}

int parse_ternary() {
    int c = parse_binary(1);
    if (cur() != T_QUESTION) { return c; }
    int t = g_p;
    advance();
    int a = parse_expr();
    expect_msg(T_COLON, "expected ':' in ternary expression");
    int b = parse_ternary();
    int n = new_node(E_TERNARY, t);
    g_node[n].a = c;
    g_node[n].b = a;
    g_node[n].c = b;
    return n;
}

int is_assign_op(int k) {
    return k == T_ASSIGN || k == T_PLUSEQ || k == T_MINUSEQ || k == T_STAREQ || k == T_SLASHEQ ||
           k == T_PERCENTEQ || k == T_AMPEQ || k == T_PIPEEQ || k == T_CARETEQ || k == T_SHLEQ || k == T_SHREQ;
}

int parse_assign() {
    int l = parse_ternary();
    int k = cur();
    if (is_assign_op(k) == 1) {
        int t = g_p;
        advance();
        int r = parse_assign();
        int n = new_node(E_ASSIGN, t);
        g_node[n].op = k;
        g_node[n].a = l;
        g_node[n].b = r;
        return n;
    }
    return l;
}

int parse_expr() {
    int l = parse_assign();
    while (cur() == T_COMMA) {
        int t = g_p;
        advance();
        int r = parse_assign();
        int n = new_node(E_COMMA, t);
        g_node[n].a = l;
        g_node[n].b = r;
        l = n;
    }
    return l;
}

// ── match ────────────────────────────────────────────────────────────────────
int parse_pattern() {
    int t = g_p;
    if (cur() == T_DOT && peek(1) == T_IDENT) {
        advance();
        int p = new_node(P_VARIANT, t);
        g_node[p].sid = cur_sid();
        advance();
        if (accept(T_LPAREN) == 1) {
            if (cur() == T_IDENT) { g_node[p].d = cur_sid(); advance(); }
            expect_kind(T_RPAREN);
        }
        return p;
    }
    if (cur() == T_INT || cur() == T_CHAR) {
        int kind = E_INT;
        if (cur() == T_CHAR) { kind = E_CHAR; }
        int lo = new_node(kind, t);
        g_node[lo].ival = g_tok[t].ival;
        advance();
        if (cur() == T_DOTDOT || cur() == T_ELLIPSIS) {
            advance();
            if (cur() == T_INT || cur() == T_CHAR) {
                int hk = E_INT;
                if (cur() == T_CHAR) { hk = E_CHAR; }
                int hi = new_node(hk, g_p);
                g_node[hi].ival = g_tok[g_p].ival;
                advance();
                int p = new_node(P_RANGE, t);
                g_node[p].a = lo;
                g_node[p].b = hi;
                return p;
            }
            if (kind == E_INT) { perr("expected integer after '..' in range pattern"); }
            else { perr("expected char after '..' in range pattern"); }
        }
        int p = new_node(P_VALUE, t);
        g_node[p].a = lo;
        return p;
    }
    if (cur() == T_IDENT || cur() == K_NULL) {
        int p = new_node(P_NAME, t);
        g_node[p].sid = cur_sid();
        advance();
        if (accept(T_LPAREN) == 1) {
            g_node[p].flags = 1;
            if (cur() == T_IDENT) { g_node[p].d = cur_sid(); advance(); }
            expect_kind(T_RPAREN);
        }
        return p;
    }
    perr("expected match pattern (integer, char, range, or identifier)");
    advance();
    return new_node(P_WILD, t);
}

// one arm; is_expr: the arm holds an expression
int parse_arm(int is_expr) {
    int arm = new_node(S_ARM, g_p);
    int mark = list_begin();
    if (cur() == K_DEFAULT) {
        list_push(new_node(P_WILD, g_p));
        advance();
        expect_msg(T_COLON, "expected ':' after 'default'");
    } else {
        expect_msg(K_CASE, "expected 'case' or 'default' in match arm");
        while (1) {
            list_push(parse_pattern());
            if (accept(T_COMMA) == 0) { break; }
        }
        expect_msg(T_COLON, "expected ':' after match pattern(s)");
    }
    list_end(mark, arm);
    if (is_expr == 1) { g_node[arm].a = parse_assign(); }
    else { g_node[arm].a = parse_stmt(); }
    return arm;
}

int parse_match_head(int kind) {
    int n = new_node(kind, g_p);
    advance(); // 'match'
    expect_msg(T_LPAREN, "expected '(' after 'match'");
    g_node[n].a = parse_expr();
    expect_msg(T_RPAREN, "expected ')' after match subject");
    expect_msg(T_LBRACE, "expected '{' opening match body");
    return n;
}

int parse_match_expr() {
    int n = parse_match_head(E_MATCH);
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        list_push(parse_arm(1));
        if (accept(T_COMMA) == 0) { break; }
    }
    expect_msg(T_RBRACE, "expected '}' closing match");
    list_end(mark, n);
    return n;
}

int parse_match_stmt() {
    int n = parse_match_head(S_MATCH);
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        int before = g_p;
        list_push(parse_arm(0));
        if (g_p == before) { advance(); }
    }
    expect_msg(T_RBRACE, "expected '}' closing match");
    list_end(mark, n);
    return n;
}

int parse_switch() {
    int n = new_node(S_SWITCH, g_p);
    advance();
    expect_msg(T_LPAREN, "expected '(' after 'switch'");
    g_node[n].a = parse_expr();
    expect_msg(T_RPAREN, "expected ')' after switch subject");
    expect_msg(T_LBRACE, "expected '{' opening switch body");
    int mark = list_begin();
    int saw_default = 0;
    while (!at_end() && cur() != T_RBRACE) {
        int c = new_node(S_CASE, g_p);
        int vmark = list_begin();
        int saw = 0;
        while (cur() == K_CASE || cur() == K_DEFAULT) {
            if (cur() == K_DEFAULT) {
                advance();
                expect_msg(T_COLON, "expected ':' after 'default'");
                if (saw_default == 1) { msg_reset(); msg_lit("multiple 'default' labels in one switch"); error_node(c); }
                g_node[c].flags = NF_DEFAULT;
                saw_default = 1;
            } else {
                advance();
                int neg = 0;
                if (cur() == T_MINUS) { neg = 1; advance(); }
                int v = new_node(E_INT, g_p);
                if (cur() == T_INT || cur() == T_CHAR) {
                    g_node[v].ival = g_tok[g_p].ival;
                    if (neg == 1) { g_node[v].ival = 0L - g_node[v].ival; }
                    advance();
                } else { perr("expected integer or char literal in case label"); }
                list_push(v);
                expect_msg(T_COLON, "expected ':' after case value");
            }
            saw = 1;
        }
        list_end(vmark, c);
        if (saw == 0) { perr("expected 'case' or 'default' in switch body"); break; }
        int body = new_node(S_BLOCK, g_p);
        int bmark = list_begin();
        while (!at_end() && cur() != T_RBRACE && cur() != K_CASE && cur() != K_DEFAULT) {
            list_push(parse_stmt());
        }
        list_end(bmark, body);
        g_node[c].b = body;
        list_push(c);
    }
    expect_msg(T_RBRACE, "expected '}' closing switch");
    list_end(mark, n);
    return n;
}

// ── statements ───────────────────────────────────────────────────────────────
int looks_like_var_decl() {
    int k = cur();
    if (is_type_start_tok(g_p) == 0) { return 0; }
    if (k == K_INT || k == K_VOID || k == K_CHAR || k == K_FLOAT || k == K_DOUBLE || k == K_BOOL ||
        k == K_LONG || k == K_SHORT || k == K_UNSIGNED || k == K_SIGNED || k == K_STRUCT ||
        k == K_UNION || k == K_ENUM || k == K_TUPLE || k == K_FN || k == K_TYPEOF || k == K_AUTO ||
        k == T_AMP || k == T_QAMP || k == T_QUESTION) { return 1; }
    if (k == T_IDENT && peek(1) == T_IDENT) { return 1; }
    if (k == T_IDENT && peek(1) == T_STAR && peek(2) == T_IDENT) { return 1; }
    if (k == T_IDENT && peek(1) == T_LT && g_typename[cur_sid()] == 1) { return 1; }
    return 0;
}

int strip_arrays(int t) {
    while (g_ty[t].kind == TY_ARRAY) { t = g_ty[t].base; }
    return t;
}

int mk_var(int name, int ty, int init, int tok, int is_const, int is_static) {
    int v = new_node(S_VAR, tok);
    g_node[v].sid = name;
    g_node[v].c = ty;
    g_node[v].a = init;
    if (is_const == 1 && (ty == 0 || g_ty[strip_arrays(ty)].kind != TY_PTR)) { g_node[v].flags = g_node[v].flags | NF_CONST; }
    if (is_static == 1) { g_node[v].flags = g_node[v].flags | NF_STATIC; }
    return v;
}

int parse_var_decl(int is_const, int is_static) {
    int t = g_p;
    if (cur() == K_AUTO) {
        advance();
        int name = 0;
        if (cur() == T_IDENT) { name = cur_sid(); advance(); }
        else { perr("expected variable name"); }
        int init = 0;
        if (accept(T_ASSIGN) == 0) { perr("'auto' variable must have an initializer"); }
        else { init = parse_assign(); }
        expect_msg(T_SEMI, "expected ';' after variable declaration");
        return mk_var(name, 0, init, t, is_const, is_static);
    }
    int base = parse_type();
    int ty = base;
    int name = 0;
    int fp = try_fnptr_declarator(base);
    if (fp != 0) { ty = fp; name = g_fp_name; }
    else {
        if (cur() == T_IDENT) { name = cur_sid(); advance(); }
        else { perr("expected variable name"); }
    }
    ty = parse_array_suffix(ty);
    int init = 0;
    if (accept(T_ASSIGN) == 1) { init = parse_assign(); }
    if (cur() != T_COMMA) {
        expect_msg(T_SEMI, "expected ';' after variable declaration");
        return mk_var(name, ty, init, t, is_const, is_static);
    }
    int multi = new_node(S_MULTI, t);
    int mark = list_begin();
    list_push(mk_var(name, ty, init, t, is_const, is_static));
    while (accept(T_COMMA) == 1) {
        int dt = g_p;
        int n2 = 0;
        if (cur() == T_IDENT) { n2 = cur_sid(); advance(); }
        else { perr("expected variable name"); }
        int t2 = parse_array_suffix(base);
        int i2 = 0;
        if (accept(T_ASSIGN) == 1) { i2 = parse_assign(); }
        list_push(mk_var(n2, t2, i2, dt, is_const, is_static));
    }
    expect_msg(T_SEMI, "expected ';' after variable declaration");
    list_end(mark, multi);
    return multi;
}

int parse_block() {
    int n = new_node(S_BLOCK, g_p);
    expect_msg(T_LBRACE, "expected '{'");
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        int before = g_p;
        list_push(parse_stmt());
        if (g_p == before) { advance(); }
    }
    expect_msg(T_RBRACE, "expected '}'");
    list_end(mark, n);
    return n;
}

int parse_expr_stmt() {
    int t = g_p;
    if (cur() == T_SEMI) { advance(); return new_node(S_EMPTY, t); }
    int n = new_node(S_EXPR, t);
    g_node[n].a = parse_expr();
    expect_msg(T_SEMI, "expected ';' after expression");
    return n;
}

int parse_loop(int label) {
    int t = g_p;
    int k = cur();
    advance();
    if (k == K_WHILE) {
        int n = new_node(S_WHILE, t);
        expect_kind(T_LPAREN);
        g_node[n].a = parse_expr();
        expect_kind(T_RPAREN);
        g_node[n].b = parse_stmt();
        g_node[n].sid = label;
        return n;
    }
    // for
    int n = new_node(S_FOR, t);
    g_node[n].sid = label;
    expect_kind(T_LPAREN);
    if (cur() != T_SEMI) {
        if (looks_like_var_decl() == 1) { g_node[n].a = parse_var_decl(0, 0); }
        else { g_node[n].a = parse_expr_stmt(); }
    } else { expect_kind(T_SEMI); }
    if (cur() != T_SEMI) { g_node[n].b = parse_expr(); }
    expect_kind(T_SEMI);
    if (cur() != T_RPAREN) { g_node[n].c = parse_expr(); }
    expect_kind(T_RPAREN);
    g_node[n].d = parse_stmt();
    return n;
}

int parse_stmt() {
    int t = g_p;
    int k = cur();
    if (k == T_IDENT && peek(1) == T_COLON) {
        int lbl = cur_sid();
        advance(); advance();
        if (cur() == K_FOR || cur() == K_WHILE) { return parse_loop(lbl); }
        int n = new_node(S_LABEL, t);
        g_node[n].sid = lbl;
        g_node[n].a = parse_stmt();
        return n;
    }
    if (k == T_LBRACE) { return parse_block(); }
    if (k == K_IF) {
        advance();
        int kind = S_IF;
        if (cur() == K_CONST) { advance(); kind = S_IFCONST; }
        int n = new_node(kind, t);
        if (kind == S_IFCONST) { expect_msg(T_LPAREN, "expected '(' after 'if const'"); }
        else { expect_msg(T_LPAREN, "expected '(' after 'if'"); }
        g_node[n].a = parse_expr();
        if (kind == S_IFCONST) { expect_msg(T_RPAREN, "expected ')' after if const condition"); }
        else { expect_msg(T_RPAREN, "expected ')' after if condition"); }
        g_node[n].b = parse_stmt();
        if (accept(K_ELSE) == 1) { g_node[n].c = parse_stmt(); }
        return n;
    }
    if (k == K_WHILE || k == K_FOR) { return parse_loop(0); }
    if (k == K_DO) {
        advance();
        int n = new_node(S_DO, t);
        g_node[n].b = parse_stmt();
        expect_msg(K_WHILE, "expected 'while' after do-body");
        expect_kind(T_LPAREN);
        g_node[n].a = parse_expr();
        expect_kind(T_RPAREN);
        expect_kind(T_SEMI);
        return n;
    }
    if (k == K_RETURN) {
        advance();
        int n = new_node(S_RETURN, t);
        if (cur() != T_SEMI) { g_node[n].a = parse_expr(); }
        expect_kind(T_SEMI);
        return n;
    }
    if (k == K_DEFER || k == K_ERRDEFER) {
        advance();
        int n = new_node(S_DEFER, t);
        if (k == K_ERRDEFER) { g_node[n].flags = NF_ERR; }
        g_node[n].a = parse_stmt();
        return n;
    }
    if (k == K_MATCH) { return parse_match_stmt(); }
    if (k == K_SWITCH) { return parse_switch(); }
    if (k == K_BREAK || k == K_CONTINUE) {
        advance();
        int kind = S_BREAK;
        if (k == K_CONTINUE) { kind = S_CONTINUE; }
        int n = new_node(kind, t);
        if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); advance(); }
        expect_kind(T_SEMI);
        return n;
    }
    if (k == K_GOTO) {
        advance();
        int n = new_node(S_GOTO, t);
        if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); advance(); }
        else { perr("expected label name"); }
        expect_kind(T_SEMI);
        return n;
    }
    if (k == K_UNSAFE) {
        advance();
        int n = new_node(S_UNSAFE, t);
        g_node[n].a = parse_block();
        return n;
    }
    if (k == K_STATIC_ASSERT) {
        advance();
        int n = new_node(S_STATIC_ASSERT, t);
        expect_kind(T_LPAREN);
        g_node[n].a = parse_assign();
        if (accept(T_COMMA) == 1) {
            if (cur() == T_STRING) { g_node[n].sid = cur_sid(); advance(); }
        }
        expect_kind(T_RPAREN);
        expect_kind(T_SEMI);
        return n;
    }
    if (k == K_ASM) {
        perr("inline asm is not supported by scc");
        while (!at_end() && cur() != T_SEMI) { advance(); }
        accept(T_SEMI);
        return new_node(S_EMPTY, t);
    }
    if (k == K_CONST) {
        if (peek(1) == T_AMP || peek(1) == T_QAMP) { return parse_var_decl(0, 0); }
        advance();
        int saved = g_p;
        int sdl = g_diag_len;
        int sne = g_nerrors;
        int snw = g_nwarnings;
        g_silent = g_silent + 1;
        int pt = parse_type();
        g_silent = g_silent - 1;
        g_diag_len = sdl;
        g_nerrors = sne;
        g_nwarnings = snw;
        g_p = saved;
        int is_const = 1;
        if (g_ty[pt].kind == TY_PTR) { is_const = 0; }
        return parse_var_decl(is_const, 0);
    }
    if (k == K_VOLATILE) { return parse_var_decl(0, 0); }
    if (k == K_STATIC) {
        advance();
        int is_const = 0;
        if (cur() == K_CONST && peek(1) != T_AMP && peek(1) != T_QAMP) { advance(); is_const = 1; }
        return parse_var_decl(is_const, 1);
    }
    if (k == T_LBRACKET && peek(1) == T_RBRACKET) { return parse_var_decl(0, 0); }
    if (looks_like_var_decl() == 1) { return parse_var_decl(0, 0); }
    return parse_expr_stmt();
}

// ── declarations ─────────────────────────────────────────────────────────────
int parse_generic_params() {
    int n = new_node(D_GPARAMS, g_p);
    expect_msg(T_LT, "expected '<' after 'generic'");
    int mark = list_begin();
    while (!at_end() && cur() != T_GT) {
        int p = new_node(D_GPARAM, g_p);
        if (cur() == T_IDENT) { g_node[p].sid = cur_sid(); g_typename[cur_sid()] = 1; advance(); }
        else { perr("expected generic parameter name"); }
        if (cur() == T_ELLIPSIS) { advance(); g_node[p].flags = 1; }
        if (cur() == T_COLON) {
            advance();
            int bn = new_node(D_NAMES, g_p);
            int bm = list_begin();
            while (1) {
                if (cur() != T_IDENT) { perr("expected constraint name"); break; }
                int nm = new_node(E_IDENT, g_p);
                g_node[nm].sid = cur_sid();
                advance();
                while (cur() == T_COLONCOLON && peek(1) == T_IDENT) { advance(); g_node[nm].sid = cur_sid(); advance(); }
                list_push(nm);
                if (accept(T_PLUS) == 0) { break; }
            }
            list_end(bm, bn);
            g_node[p].b = bn;
        }
        list_push(p);
        if (accept(T_COMMA) == 0) { break; }
    }
    expect_msg(T_GT, "expected '>' closing generic parameter list");
    list_end(mark, n);
    return n;
}

// parameter list after '(' ; fills fn's list, flags NF_VARIADIC
void parse_params(int fnn) {
    int mark = list_begin();
    if (cur() == K_VOID && peek(1) == T_RPAREN) { advance(); }
    while (!at_end() && cur() != T_RPAREN) {
        if (cur() == T_ELLIPSIS) { advance(); g_node[fnn].flags = g_node[fnn].flags | NF_VARIADIC; break; }
        int p = new_node(D_PARAM, g_p);
        int pt = parse_type();
        int fp = try_fnptr_declarator(pt);
        if (fp != 0) { pt = fp; g_node[p].sid = g_fp_name; }
        else {
            if (cur() == T_ELLIPSIS) { advance(); g_node[p].flags = 1; }
            if (cur() == T_IDENT || cur() == K_SELF) { g_node[p].sid = cur_sid(); advance(); }
            pt = parse_array_suffix(pt);
        }
        g_node[p].c = pt;
        list_push(p);
        if (accept(T_COMMA) == 0) { break; }
    }
    expect_msg(T_RPAREN, "expected ')' closing parameter list");
    list_end(mark, fnn);
}

int parse_operator_name() {
    // after 'operator'
    tmp_reset();
    tmp_lit("operator");
    if (cur() != T_LPAREN) {
        int k = cur();
        tmp_punct(k);
        advance();
        if ((k == T_ASSIGN || k == T_BANG || k == T_LT || k == T_GT) && cur() == T_ASSIGN) { tmp_lit("="); advance(); }
    }
    return intern_tmp();
}

int parse_struct(int is_union, int gparams) {
    int t = g_p;
    int sd = new_node(D_STRUCT, t);
    if (is_union == 1) { g_node[sd].flags = NF_UNION; }
    g_node[sd].b = gparams;
    if (cur() == T_IDENT) { g_node[sd].sid = cur_sid(); g_typename[cur_sid()] = 1; advance(); }
    expect_msg(T_LBRACE, "expected '{' in struct definition");
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        int saved = g_p;
        int mt = g_p;
        int mg = 0;
        if (cur() == K_GENERIC) { advance(); mg = parse_generic_params(); }
        int is_static = 0;
        if (cur() == K_STATIC) { advance(); is_static = 1; }
        int ty = parse_type();
        int fp = try_fnptr_declarator(ty);
        if (fp != 0) {
            int f = new_node(D_FIELD, mt);
            g_node[f].sid = g_fp_name;
            g_node[f].c = fp;
            expect_msg(T_SEMI, "expected ';' after struct field");
            list_push(f);
            continue;
        }
        if (cur() == K_OPERATOR || (is_word(g_p) == 1 && peek(1) == T_LPAREN)) {
            int m = new_node(D_FUNC, g_p);
            if (cur() == K_OPERATOR) { advance(); g_node[m].sid = parse_operator_name(); }
            else { g_node[m].sid = cur_sid(); advance(); }
            expect_msg(T_LPAREN, "expected '(' after method name");
            g_node[m].c = ty;
            g_node[m].flags = NF_METHOD | NF_FWD;
            g_node[m].d = g_node[sd].sid;
            g_node[m].b = mg;
            if (mg != 0) { g_node[m].flags = g_node[m].flags | NF_GENERIC; }
            parse_params(m);
            if (cur() == K_CONST) { advance(); g_node[m].flags = g_node[m].flags | NF_MCONST; }
            if (is_static == 1) {
                g_node[m].flags = g_node[m].flags | NF_SMETHOD;
                if ((g_node[m].flags & NF_MCONST) != 0) { msg_reset(); msg_lit("a static method has no 'self' and can't be 'const'"); error_node(m); }
            }
            parse_attributes();
            expect_msg(T_SEMI, "expected ';' after method declaration");
            list_push(m);
        } else {
            if (is_static == 1) { perr("'static' applies to methods only, not struct fields"); }
            int f = new_node(D_FIELD, g_p);
            if (cur() == T_IDENT) { g_node[f].sid = cur_sid(); advance(); }
            else if (cur() == T_SEMI && (g_ty[ty].kind == TY_NAMED)) { g_node[f].flags = 1; } // anonymous member
            else { perr("expected field name"); }
            if (cur() == T_COLON) {
                advance();
                perr("bitfields are not supported by scc");
                parse_ternary();
            } else {
                ty = parse_array_suffix(ty);
            }
            g_node[f].c = ty;
            // 'int a, b;' — several fields of one type
            while (cur() == T_COMMA && peek(1) == T_IDENT) {
                list_push(f);
                advance();
                f = new_node(D_FIELD, g_p);
                g_node[f].sid = cur_sid();
                advance();
                g_node[f].c = parse_array_suffix(ty);
            }
            expect_msg(T_SEMI, "expected ';' after struct field");
            list_push(f);
        }
        if (g_p == saved) { advance(); }
    }
    expect_msg(T_RBRACE, "expected '}' closing struct");
    list_end(mark, sd);
    accept(T_SEMI);
    return sd;
}

int parse_enum() {
    int n = new_node(D_ENUM, g_p);
    if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); g_typename[cur_sid()] = 1; advance(); }
    if (accept(T_COLON) == 1) { g_node[n].c = parse_type(); }
    expect_msg(T_LBRACE, "expected '{' in enum definition");
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        if (cur() != T_IDENT) { perr("expected enumerator name"); advance(); continue; }
        int e = new_node(D_ENUMC, g_p);
        g_node[e].sid = cur_sid();
        advance();
        if (accept(T_ASSIGN) == 1) { g_node[e].a = parse_ternary(); }
        list_push(e);
        if (accept(T_COMMA) == 0) { break; }
    }
    expect_msg(T_RBRACE, "expected '}' closing enum");
    accept(T_SEMI);
    list_end(mark, n);
    return n;
}

int parse_trait() {
    int n = new_node(D_TRAIT, g_p);
    advance();
    if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); advance(); }
    else { perr("expected trait name after 'trait'"); }
    if (accept(T_COLON) == 1) {
        int bn = new_node(D_NAMES, g_p);
        int bm = list_begin();
        while (1) {
            if (cur() != T_IDENT) { perr("expected supertrait name after ':' or '+'"); break; }
            int nm = new_node(E_IDENT, g_p);
            g_node[nm].sid = cur_sid();
            advance();
            while (cur() == T_COLONCOLON && peek(1) == T_IDENT) { advance(); g_node[nm].sid = cur_sid(); advance(); }
            list_push(nm);
            if (accept(T_PLUS) == 0) { break; }
        }
        list_end(bm, bn);
        g_node[n].b = bn;
    }
    expect_msg(T_LBRACE, "expected '{' in trait definition");
    int mark = list_begin();
    while (!at_end() && cur() != T_RBRACE) {
        int saved = g_p;
        int m = new_node(D_FUNC, g_p);
        g_node[m].c = parse_type();
        if (is_word(g_p) == 0) { perr("expected method name in trait body"); }
        else {
            g_node[m].sid = cur_sid();
            advance();
            expect_msg(T_LPAREN, "expected '(' after method name in trait body");
            parse_params(m);
            if (cur() == K_CONST) { advance(); g_node[m].flags = g_node[m].flags | NF_MCONST; }
            expect_msg(T_SEMI, "expected ';' after trait method declaration");
            list_push(m);
        }
        if (g_p == saved) { advance(); }
    }
    expect_msg(T_RBRACE, "expected '}' closing trait");
    accept(T_SEMI);
    list_end(mark, n);
    return n;
}

void add_top(int n) {
    if (n == 0) { return; }
    if (g_ntop >= LIST_MAX) { fatal("declarations"); return; }
    g_top[g_ntop] = n;
    g_ntop = g_ntop + 1;
}

void parse_top();

void parse_namespace() {
    advance();
    if (cur() != T_IDENT) { perr("expected namespace name after 'namespace'"); advance(); return; }
    int name = cur_sid();
    advance();
    expect_msg(T_LBRACE, "expected '{' after namespace name");
    int saved = g_ns;
    if (g_ns == 0) { g_ns = name; } else { g_ns = str_concat3(g_ns, "::", name); }
    while (!at_end() && cur() != T_RBRACE) {
        int before = g_p;
        parse_top();
        if (g_p == before) { advance(); }
    }
    expect_msg(T_RBRACE, "expected '}' to close namespace");
    accept(T_SEMI);
    g_ns = saved;
}

void parse_top() {
    int t = g_p;
    g_attr_flags = 0;
    g_attr_export = 0;
    parse_attributes();
    int flags = g_attr_flags;
    if (cur() == K_MUST_USE) { advance(); flags = flags | NF_MUSTUSE; }
    if (cur() == K_PACKED) { advance(); flags = flags | NF_PACKED; }
    if (cur() == K_STRUCT || cur() == K_UNION) {
        int is_union = 0;
        if (cur() == K_UNION) { is_union = 1; }
        if (peek(1) == T_LBRACE || (peek(1) == T_IDENT && peek(2) == T_LBRACE)) {
            advance();
            add_top(parse_struct(is_union, 0));
            return;
        }
        if (peek(1) == T_IDENT && peek(2) == T_SEMI) {
            // forward declaration
            g_typename[g_tok[g_p + 1].sid] = 1;
            advance(); advance(); advance();
            return;
        }
    }
    if (cur() == K_ENUM && peek(1) == T_IDENT && (peek(2) == T_LBRACE || peek(2) == T_COLON)) {
        advance();
        add_top(parse_enum());
        return;
    }
    if (cur() == K_NEWTYPE) {
        advance();
        int n = new_node(D_NEWTYPE, g_p);
        if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); g_typename[cur_sid()] = 1; advance(); }
        else { perr("expected newtype name"); }
        expect_msg(T_ASSIGN, "expected '=' in newtype declaration");
        g_node[n].c = parse_type();
        expect_semi();
        add_top(n);
        return;
    }
    if (cur() == K_TRAIT) { add_top(parse_trait()); return; }
    if (cur() == K_NAMESPACE) { parse_namespace(); return; }
    if (cur() == K_REGION) {
        advance();
        int n = new_node(D_REGION, g_p);
        if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); advance(); }
        else { perr("expected region name"); }
        expect_msg(T_LBRACE, "expected '{' in region declaration");
        while (!at_end() && cur() != T_RBRACE) {
            if (cur() == T_IDENT && cur_sid() == S_capacity) {
                advance();
                expect_msg(T_COLON, "expected ':' after 'capacity'");
                if (cur() == T_INT) { g_node[n].ival = g_tok[g_p].ival; advance(); }
                else { perr("expected integer capacity"); }
            } else {
                msg_reset();
                msg_lit("unknown region field, skipping");
                warn_tok(g_p);
                advance();
            }
            accept(T_COMMA);
        }
        expect_msg(T_RBRACE, "expected '}' closing region");
        accept(T_SEMI);
        add_top(n);
        return;
    }
    if (cur() == K_TYPEDEF) {
        advance();
        int n = new_node(D_TYPEDEF, g_p);
        int ty = 0;
        // 'typedef struct [X] { ... } Y;'
        int anon = 0;
        if ((cur() == K_STRUCT || cur() == K_UNION) && (peek(1) == T_LBRACE || (peek(1) == T_IDENT && peek(2) == T_LBRACE))) {
            int is_union = 0;
            if (cur() == K_UNION) { is_union = 1; }
            advance();
            int sd = parse_struct(is_union, 0);
            add_top(sd);
            if (g_node[sd].sid == 0) { anon = sd; }
            ty = parse_declarator(ty_named(g_node[sd].sid, -1, TYF_STRUCTKW));
        } else if (cur() == K_ENUM && (peek(1) == T_LBRACE || peek(1) == T_COLON || (peek(1) == T_IDENT && (peek(2) == T_LBRACE || peek(2) == T_COLON)))) {
            advance();
            int ed = parse_enum();
            add_top(ed);
            if (g_node[ed].sid == 0) { anon = ed; }
            ty = ty_named(g_node[ed].sid, -1, TYF_ENUMKW);
        } else {
            ty = parse_type();
        }
        if (anon != 0 && cur() == T_IDENT) {
            // the anonymous type takes the typedef's name
            g_node[anon].sid = cur_sid();
            g_typename[cur_sid()] = 1;
            g_node[n].sid = cur_sid();
            advance();
            expect_semi();
            return;
        }
        int fp = try_fnptr_declarator(ty);
        if (fp != 0) { ty = fp; g_node[n].sid = g_fp_name; }
        else if (cur() == T_IDENT) { g_node[n].sid = cur_sid(); advance(); ty = parse_array_suffix(ty); }
        else { perr("expected typedef name"); }
        g_node[n].c = ty;
        g_typename[g_node[n].sid] = 1;
        expect_semi();
        add_top(n);
        return;
    }
    if (cur() == K_STATIC_ASSERT) {
        advance();
        int n = new_node(D_STATIC_ASSERT, t);
        expect_msg(T_LPAREN, "expected '(' after static_assert");
        g_node[n].a = parse_assign();
        if (accept(T_COMMA) == 1) { if (cur() == T_STRING) { g_node[n].sid = cur_sid(); advance(); } }
        expect_msg(T_RPAREN, "expected ')' closing static_assert");
        expect_semi();
        add_top(n);
        return;
    }
    int gparams = 0;
    if (cur() == K_GENERIC) {
        advance();
        gparams = parse_generic_params();
        if ((cur() == K_STRUCT || cur() == K_UNION) && (peek(1) == T_LBRACE || (peek(1) == T_IDENT && peek(2) == T_LBRACE))) {
            int is_union = 0;
            if (cur() == K_UNION) { is_union = 1; }
            advance();
            add_top(parse_struct(is_union, gparams));
            return;
        }
    }
    // qualifiers
    while (1) {
        if (is_attr(g_p)) { g_attr_flags = 0; parse_attributes(); flags = flags | g_attr_flags; continue; }
        if (cur() == K_CONST && peek(1) != T_AMP && peek(1) != T_QAMP) { advance(); flags = flags | NF_CONST; continue; }
        if (cur() == K_CONSTEVAL) { advance(); flags = flags | NF_CONSTEVAL; continue; }
        if (cur() == K_CONSTINIT) { advance(); continue; }
        if (cur() == K_INLINE) { advance(); flags = flags | NF_INLINE; continue; }
        if (cur() == K_EXTERN) { advance(); flags = flags | NF_EXTERN; continue; }
        if (cur() == K_STATIC) { advance(); flags = flags | NF_STATIC; continue; }
        if (cur() == K_NORETURN) { advance(); flags = flags | NF_NORETURN; continue; }
        if (cur() == K_PURE) { advance(); flags = flags | NF_PURE; continue; }
        if (cur() == K_MUST_USE) { advance(); flags = flags | NF_MUSTUSE; continue; }
        if (cur() == K_VOLATILE) { advance(); continue; }
        break;
    }
    int export_sid = g_attr_export;
    if (is_type_start_tok(g_p) == 0) {
        if (g_silent == 0) {
            msg_reset();
            msg_lit("unexpected token '");
            msg_tok_text(g_p);
            msg_lit("' at top level");
            error_tok(g_p);
        }
        advance();
        return;
    }
    int ty = parse_type();
    int is_const_obj = 0;
    if ((flags & NF_CONST) != 0) { is_const_obj = 1; }
    int fp = try_fnptr_declarator(ty);
    if (fp != 0) {
        int g = new_node(D_GLOBAL, t);
        g_node[g].sid = g_fp_name;
        g_node[g].c = fp;
        g_node[g].flags = flags;
        g_node[g].op = g_ns;
        if (accept(T_ASSIGN) == 1) { g_node[g].a = parse_assign(); }
        expect_semi();
        add_top(g);
        return;
    }
    if (is_word(g_p) == 0) {
        perr("expected identifier after type");
        while (!at_end() && cur() != T_SEMI && cur() != T_RBRACE) { advance(); }
        accept(T_SEMI);
        return;
    }
    int nt = g_p;
    int name = cur_sid();
    advance();
    int owner = 0;
    if (cur() == T_COLONCOLON) {
        advance();
        owner = name;
        if (cur() == K_OPERATOR) { advance(); name = parse_operator_name(); }
        else if (is_word(g_p) == 1) { name = cur_sid(); advance(); }
        else { perr("expected method name after '::'"); return; }
    }
    if (cur() == T_LPAREN) {
        advance();
        int fnn = new_node(D_FUNC, nt);
        g_node[fnn].sid = name;
        g_node[fnn].c = ty;
        g_node[fnn].b = gparams;
        g_node[fnn].op = g_ns;
        g_node[fnn].flags = flags;
        if (gparams != 0) { g_node[fnn].flags = g_node[fnn].flags | NF_GENERIC; }
        if (owner != 0) { g_node[fnn].d = owner; g_node[fnn].flags = g_node[fnn].flags | NF_METHOD; }
        parse_params(fnn);
        if (cur() == K_CONST && is_attr(g_p + 1)) { advance(); g_node[fnn].flags = g_node[fnn].flags | NF_MCONST; }
        g_attr_flags = 0;
        g_attr_export = 0;
        if (parse_attributes() == 1) {
            g_node[fnn].flags = g_node[fnn].flags | g_attr_flags;
            if (g_attr_export != 0) { export_sid = g_attr_export; }
        }
        if (cur() == K_CONST) { advance(); g_node[fnn].flags = g_node[fnn].flags | NF_MCONST; }
        if ((g_node[fnn].flags & NF_EXPORT) != 0) {
            if (export_sid == 0) { export_sid = name; }
            g_node[fnn].ival = (long)export_sid;
        }
        if (cur() == T_LBRACE) { g_node[fnn].a = parse_block(); }
        else {
            g_node[fnn].flags = g_node[fnn].flags | NF_FWD;
            expect_msg(T_SEMI, "expected ';' or '{' after function declaration");
        }
        add_top(fnn);
        return;
    }
    // global variable(s)
    while (1) {
        int g = new_node(D_GLOBAL, nt);
        g_node[g].sid = name;
        int gt = parse_array_suffix(ty);
        g_node[g].c = gt;
        g_node[g].flags = flags & ~NF_CONST;
        if (is_const_obj == 1 && g_ty[strip_arrays(gt)].kind != TY_PTR) { g_node[g].flags = g_node[g].flags | NF_CONST; }
        g_node[g].op = g_ns;
        parse_attributes();
        if (accept(T_ASSIGN) == 1) { g_node[g].a = parse_assign(); }
        add_top(g);
        if (cur() == T_COMMA && peek(1) == T_IDENT) {
            advance();
            nt = g_p;
            name = cur_sid();
            advance();
            continue;
        }
        break;
    }
    expect_semi();
}

void parse_program() {
    g_p = 1;
    while (!at_end()) {
        int before = g_p;
        parse_top();
        if (g_p == before) { advance(); }
    }
}
