// lex.sc — the lexer: source bytes of one file → raw tokens.
#include "scc.h"
#include "lex.h"

struct Tok g_ptok[RAW_MAX];   // raw tokens, macro bodies, expansion buffers (a stack)
int g_nptok = 0;

int add_file(int name_sid, int start, int len) {
    if (g_nfile + 1 >= NFILE_MAX) { fatal("files"); return 0; }
    g_nfile = g_nfile + 1;
    g_file_name[g_nfile] = name_sid;
    g_file_start[g_nfile] = start;
    g_file_len[g_nfile] = len;
    return g_nfile;
}

int is_ident_start(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$';
}

int is_ident_char(int c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

int is_digit(int c) { return c >= '0' && c <= '9'; }

int hex_val(int c) {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

// ── IEEE-754 helpers (pure arithmetic: no type punning) ──────────────────────
long f64_bits(double d) {
    long sign = 0L;
    if (d != d) { return 9221120237041090560L; }   // 0x7FF8000000000000
    if (d < 0.0) { sign = 1L; d = 0.0 - d; }
    else if (d == 0.0) {
        if (1.0 / d < 0.0) { sign = 1L; }
        return sign << 63L;
    }
    if (d > 1.7976931348623157e308) { return (sign << 63L) | 9218868437227405312L; }
    int e = 0;
    while (d >= 2.0) { d = d / 2.0; e = e + 1; }
    while (d < 1.0) { d = d * 2.0; e = e - 1; }
    if (e < -1022) {
        // subnormal: value / 2^-1074 is an integer
        int k = e + 1074;
        double m = d;
        while (k > 0) { m = m * 2.0; k = k - 1; }
        return (sign << 63L) | (long)m;
    }
    long mant = (long)((d - 1.0) * 4503599627370496.0);
    return (sign << 63L) | ((long)(e + 1023) << 52L) | mant;
}

double pow10_d(int e) {
    double r = 1.0;
    double b = 10.0;
    while (e > 0) {
        if ((e & 1) != 0) { r = r * b; }
        b = b * b;
        e = e >> 1;
    }
    return r;
}

// ── lexing ───────────────────────────────────────────────────────────────────
int g_lx_pos = 0;
int g_lx_end = 0;
int g_lx_line = 1;
int g_lx_col = 1;
int g_lx_file = 0;

int lx_peek(int k) {
    int p = g_lx_pos + k;
    if (p >= g_lx_end) { return 0; }
    return (int)g_src[p];
}

int lx_adv() {
    int c = (int)g_src[g_lx_pos];
    g_lx_pos = g_lx_pos + 1;
    if (c == '\n') { g_lx_line = g_lx_line + 1; g_lx_col = 1; }
    else { g_lx_col = g_lx_col + 1; }
    return c;
}

int push_ptok(int kind, int sid, int line, int col, int flags, long ival) {
    if (g_nptok >= RAW_MAX - 1) { fatal("raw tokens"); return g_nptok; }
    int i = g_nptok;
    g_ptok[i].kind = kind;
    g_ptok[i].sid = sid;
    g_ptok[i].file = g_lx_file;
    g_ptok[i].line = line;
    g_ptok[i].col = col;
    g_ptok[i].flags = flags;
    g_ptok[i].ival = ival;
    g_nptok = i + 1;
    return i;
}

void lex_error(int line, int col, const char* s) {
    msg_reset();
    msg_lit(s);
    diag_at(g_lx_file, line, col, 1);
}

// Decodes one escape sequence (after the backslash) into tmp.
int lex_escape() {
    int c = lx_adv();
    if (c == 'n') { return 10; }
    if (c == 't') { return 9; }
    if (c == 'r') { return 13; }
    if (c == '0') {
        int v = 0;
        int n = 0;
        while (n < 2 && lx_peek(0) >= '0' && lx_peek(0) <= '7') { v = v * 8 + (lx_adv() - '0'); n = n + 1; }
        return v;
    }
    if (c == 'a') { return 7; }
    if (c == 'b') { return 8; }
    if (c == 'f') { return 12; }
    if (c == 'v') { return 11; }
    if (c == 'e') { return 27; }
    if (c == 'x') {
        int v = 0;
        int n = 0;
        while (n < 2 && hex_val(lx_peek(0)) >= 0) { v = v * 16 + hex_val(lx_adv()); n = n + 1; }
        if (n == 0) { lex_error(g_lx_line, g_lx_col, "\\x used with no following hex digits"); }
        return v;
    }
    return c;
}

void lex_number(int line, int col, int flags) {
    int start = g_lx_pos;
    int is_float = 0;
    long v = 0L;
    int base = 10;
    if (lx_peek(0) == '0' && (lx_peek(1) == 'x' || lx_peek(1) == 'X')) {
        lx_adv(); lx_adv();
        base = 16;
        while (hex_val(lx_peek(0)) >= 0) { v = v *| 16L +| (long)hex_val(lx_adv()); }
    } else if (lx_peek(0) == '0' && (lx_peek(1) == 'b' || lx_peek(1) == 'B')) {
        lx_adv(); lx_adv();
        base = 2;
        while (lx_peek(0) == '0' || lx_peek(0) == '1') { v = v *| 2L +| (long)(lx_adv() - '0'); }
    } else if (lx_peek(0) == '0' && (lx_peek(1) == 'o' || lx_peek(1) == 'O')) {
        lx_adv(); lx_adv();
        base = 8;
        while (lx_peek(0) >= '0' && lx_peek(0) <= '7') { v = v *| 8L +| (long)(lx_adv() - '0'); }
    } else {
        // decimal integer or float: collect significant digits
        long mant = 0L;
        int ndig = 0;
        int exp10 = 0;
        int extra = 0;
        while (is_digit(lx_peek(0))) {
            int d = lx_adv() - '0';
            v = v *| 10L +| (long)d;
            if (ndig < 18) { mant = mant * 10L + (long)d; if (mant != 0L) { ndig = ndig + 1; } }
            else { extra = extra + 1; }
        }
        if (lx_peek(0) == '.' && lx_peek(1) != '.') {
            is_float = 1;
            lx_adv();
            while (is_digit(lx_peek(0))) {
                int d = lx_adv() - '0';
                if (ndig < 18) { mant = mant * 10L + (long)d; if (mant != 0L) { ndig = ndig + 1; } exp10 = exp10 - 1; }
            }
        }
        if (lx_peek(0) == 'e' || lx_peek(0) == 'E') {
            int sgn = 1;
            int ev = 0;
            is_float = 1;
            lx_adv();
            if (lx_peek(0) == '+') { lx_adv(); }
            else if (lx_peek(0) == '-') { lx_adv(); sgn = -1; }
            while (is_digit(lx_peek(0))) { if (ev < 100000) { ev = ev * 10 + (lx_adv() - '0'); } else { lx_adv(); } }
            exp10 = exp10 + sgn * ev;
        }
        exp10 = exp10 + extra;
        if (is_float == 1) {
            double d = (double)mant;
            if (exp10 >= 0) {
                if (exp10 > 308 + 20) { d = d * 1.0e308 * 1.0e308; }
                else { d = d * pow10_d(exp10); }
            } else {
                int ne = 0 - exp10;
                if (ne > 22 && ne <= 22 + 22) { d = d / pow10_d(22); d = d / pow10_d(ne - 22); }
                else if (ne > 340) { d = 0.0; }
                else if (ne > 44) { d = d / pow10_d(ne - 300); d = d / 1.0e300; }
                else { d = d / pow10_d(ne); }
            }
            v = f64_bits(d);
        }
    }
    // suffixes
    while (1) {
        int c = lx_peek(0);
        if (c == 'u' || c == 'U') { flags = flags | TF_UNS; lx_adv(); continue; }
        if (c == 'l' || c == 'L') { flags = flags | TF_LONG; lx_adv(); continue; }
        if ((c == 'f' || c == 'F') && base != 16) {
            if (is_float == 0) {
                // '1f' — an integer literal with a float suffix
                v = f64_bits((double)v);
                is_float = 1;
            }
            flags = flags | TF_F32; lx_adv(); continue;
        }
        break;
    }
    if (is_ident_char(lx_peek(0))) {
        lex_error(line, col, "invalid suffix on numeric literal");
        while (is_ident_char(lx_peek(0))) { lx_adv(); }
    }
    int sid = intern_bytes(1, start, g_lx_pos - start);
    push_ptok(is_float == 1 ? T_FLOAT : T_INT, sid, line, col, flags, v);
}

int punct3(int c0, int c1, int c2) {
    if (c0 == '.' && c1 == '.' && c2 == '.') { return T_ELLIPSIS; }
    if (c0 == '<' && c1 == '<' && c2 == '=') { return T_SHLEQ; }
    if (c0 == '>' && c1 == '>' && c2 == '=') { return T_SHREQ; }
    return 0;
}

int punct2(int c0, int c1) {
    if (c0 == '+') {
        if (c1 == '+') { return T_INC; }
        if (c1 == '=') { return T_PLUSEQ; }
        if (c1 == '|') { return T_PLUSPIPE; }
        if (c1 == '%') { return T_PLUSPCT; }
    }
    if (c0 == '-') {
        if (c1 == '-') { return T_DEC; }
        if (c1 == '>') { return T_ARROW; }
        if (c1 == '=') { return T_MINUSEQ; }
        if (c1 == '|') { return T_MINUSPIPE; }
        if (c1 == '%') { return T_MINUSPCT; }
    }
    if (c0 == '*') {
        if (c1 == '=') { return T_STAREQ; }
        if (c1 == '|') { return T_STARPIPE; }
        if (c1 == '%') { return T_STARPCT; }
    }
    if (c0 == '/' && c1 == '=') { return T_SLASHEQ; }
    if (c0 == '%' && c1 == '=') { return T_PERCENTEQ; }
    if (c0 == '&') {
        if (c1 == '&') { return T_ANDAND; }
        if (c1 == '=') { return T_AMPEQ; }
    }
    if (c0 == '|') {
        if (c1 == '|') { return T_OROR; }
        if (c1 == '=') { return T_PIPEEQ; }
    }
    if (c0 == '^' && c1 == '=') { return T_CARETEQ; }
    if (c0 == '!' && c1 == '=') { return T_NE; }
    if (c0 == '=') {
        if (c1 == '=') { return T_EQ; }
        if (c1 == '>') { return T_FATARROW; }
    }
    if (c0 == '<') {
        if (c1 == '<') { return T_SHL; }
        if (c1 == '=') { return T_LE; }
    }
    if (c0 == '>') {
        if (c1 == '>') { return T_SHR; }
        if (c1 == '=') { return T_GE; }
    }
    if (c0 == '?' && c1 == '&') { return T_QAMP; }
    if (c0 == ':' && c1 == ':') { return T_COLONCOLON; }
    if (c0 == '.' && c1 == '.') { return T_DOTDOT; }
    if (c0 == '#' && c1 == '#') { return T_HASHHASH; }
    return 0;
}

int punct1(int c) {
    if (c == '(') { return T_LPAREN; }
    if (c == ')') { return T_RPAREN; }
    if (c == '{') { return T_LBRACE; }
    if (c == '}') { return T_RBRACE; }
    if (c == '[') { return T_LBRACKET; }
    if (c == ']') { return T_RBRACKET; }
    if (c == ';') { return T_SEMI; }
    if (c == ',') { return T_COMMA; }
    if (c == '.') { return T_DOT; }
    if (c == ':') { return T_COLON; }
    if (c == '?') { return T_QUESTION; }
    if (c == '!') { return T_BANG; }
    if (c == '~') { return T_TILDE; }
    if (c == '+') { return T_PLUS; }
    if (c == '-') { return T_MINUS; }
    if (c == '*') { return T_STAR; }
    if (c == '/') { return T_SLASH; }
    if (c == '%') { return T_PERCENT; }
    if (c == '&') { return T_AMP; }
    if (c == '|') { return T_PIPE; }
    if (c == '^') { return T_CARET; }
    if (c == '=') { return T_ASSIGN; }
    if (c == '<') { return T_LT; }
    if (c == '>') { return T_GT; }
    if (c == '#') { return T_HASH; }
    return 0;
}

// Lexes the whole file onto the top of g_ptok; returns the first index.
// The last token is T_EOF.
int lex_file(int file) {
    int first = g_nptok;
    g_lx_file = file;
    g_lx_pos = g_file_start[file];
    g_lx_end = g_file_start[file] + g_file_len[file];
    g_lx_line = 1;
    g_lx_col = 1;
    int bol = 1;
    int space = 0;
    while (g_lx_pos < g_lx_end) {
        int c = lx_peek(0);
        if (c == '\n') { lx_adv(); bol = 1; space = 0; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == 12 || c == 11) { lx_adv(); space = 1; continue; }
        if (c == '\\' && (lx_peek(1) == '\n' || (lx_peek(1) == '\r' && lx_peek(2) == '\n'))) {
            lx_adv();
            if (lx_peek(0) == '\r') { lx_adv(); }
            lx_adv();
            space = 1;
            continue;
        }
        if (c == '/' && lx_peek(1) == '/') {
            while (g_lx_pos < g_lx_end && lx_peek(0) != '\n') { lx_adv(); }
            space = 1;
            continue;
        }
        if (c == '/' && lx_peek(1) == '*') {
            int sl = g_lx_line;
            int sc = g_lx_col;
            lx_adv(); lx_adv();
            int closed = 0;
            while (g_lx_pos < g_lx_end) {
                if (lx_peek(0) == '*' && lx_peek(1) == '/') { lx_adv(); lx_adv(); closed = 1; break; }
                lx_adv();
            }
            if (closed == 0) { lex_error(sl, sc, "unterminated block comment"); }
            space = 1;
            continue;
        }
        int line = g_lx_line;
        int col = g_lx_col;
        int flags = 0;
        if (bol == 1) { flags = flags | TF_BOL; }
        if (space == 1) { flags = flags | TF_SPACE; }
        bol = 0;
        space = 0;
        if (is_digit(c) || (c == '.' && is_digit(lx_peek(1)))) {
            lex_number(line, col, flags);
            continue;
        }
        if (is_ident_start(c)) {
            int start = g_lx_pos;
            while (g_lx_pos < g_lx_end && is_ident_char(lx_peek(0))) { lx_adv(); }
            int sid = intern_bytes(1, start, g_lx_pos - start);
            push_ptok(T_IDENT, sid, line, col, flags, 0L);
            continue;
        }
        if (c == '"') {
            lx_adv();
            tmp_reset();
            int closed = 0;
            while (g_lx_pos < g_lx_end) {
                int ch = lx_peek(0);
                if (ch == '"') { lx_adv(); closed = 1; break; }
                if (ch == '\n') { break; }
                lx_adv();
                if (ch == '\\') { ch = lex_escape(); }
                tmp_char(ch);
            }
            if (closed == 0) { lex_error(line, col, "unterminated string literal"); }
            push_ptok(T_STRING, intern_tmp(), line, col, flags, 0L);
            continue;
        }
        if (c == '\'') {
            lx_adv();
            int ch = 0;
            if (lx_peek(0) == '\\') { lx_adv(); ch = lex_escape(); }
            else if (g_lx_pos < g_lx_end) { ch = lx_adv(); }
            if (lx_peek(0) != '\'') { lex_error(line, col, "expected closing ' in char literal"); }
            else { lx_adv(); }
            tmp_reset();
            tmp_char(ch);
            push_ptok(T_CHAR, intern_tmp(), line, col, flags, (long)(ch & 255));
            continue;
        }
        int k = punct3(c, lx_peek(1), lx_peek(2));
        if (k != 0) { lx_adv(); lx_adv(); lx_adv(); push_ptok(k, 0, line, col, flags, 0L); continue; }
        k = punct2(c, lx_peek(1));
        if (k != 0) { lx_adv(); lx_adv(); push_ptok(k, 0, line, col, flags, 0L); continue; }
        k = punct1(c);
        if (k != 0) { lx_adv(); push_ptok(k, 0, line, col, flags, 0L); continue; }
        lx_adv();
        msg_reset();
        msg_lit("unexpected character '");
        msg_char(c);
        msg_lit("'");
        diag_at(file, line, col, 1);
    }
    push_ptok(T_EOF, 0, g_lx_line, g_lx_col, TF_BOL, 0L);
    return first;
}

// Spelling of a punctuator kind (for diagnostics, stringification, pasting).
void tmp_punct(int k) {
    if (k == T_LPAREN) { tmp_lit("("); }
    else if (k == T_RPAREN) { tmp_lit(")"); }
    else if (k == T_LBRACE) { tmp_lit("{"); }
    else if (k == T_RBRACE) { tmp_lit("}"); }
    else if (k == T_LBRACKET) { tmp_lit("["); }
    else if (k == T_RBRACKET) { tmp_lit("]"); }
    else if (k == T_SEMI) { tmp_lit(";"); }
    else if (k == T_COMMA) { tmp_lit(","); }
    else if (k == T_DOT) { tmp_lit("."); }
    else if (k == T_DOTDOT) { tmp_lit(".."); }
    else if (k == T_ELLIPSIS) { tmp_lit("..."); }
    else if (k == T_ARROW) { tmp_lit("->"); }
    else if (k == T_COLONCOLON) { tmp_lit("::"); }
    else if (k == T_COLON) { tmp_lit(":"); }
    else if (k == T_QUESTION) { tmp_lit("?"); }
    else if (k == T_QAMP) { tmp_lit("?&"); }
    else if (k == T_BANG) { tmp_lit("!"); }
    else if (k == T_TILDE) { tmp_lit("~"); }
    else if (k == T_PLUS) { tmp_lit("+"); }
    else if (k == T_MINUS) { tmp_lit("-"); }
    else if (k == T_STAR) { tmp_lit("*"); }
    else if (k == T_SLASH) { tmp_lit("/"); }
    else if (k == T_PERCENT) { tmp_lit("%"); }
    else if (k == T_AMP) { tmp_lit("&"); }
    else if (k == T_PIPE) { tmp_lit("|"); }
    else if (k == T_CARET) { tmp_lit("^"); }
    else if (k == T_SHL) { tmp_lit("<<"); }
    else if (k == T_SHR) { tmp_lit(">>"); }
    else if (k == T_ASSIGN) { tmp_lit("="); }
    else if (k == T_EQ) { tmp_lit("=="); }
    else if (k == T_NE) { tmp_lit("!="); }
    else if (k == T_LT) { tmp_lit("<"); }
    else if (k == T_GT) { tmp_lit(">"); }
    else if (k == T_LE) { tmp_lit("<="); }
    else if (k == T_GE) { tmp_lit(">="); }
    else if (k == T_ANDAND) { tmp_lit("&&"); }
    else if (k == T_OROR) { tmp_lit("||"); }
    else if (k == T_PLUSEQ) { tmp_lit("+="); }
    else if (k == T_MINUSEQ) { tmp_lit("-="); }
    else if (k == T_STAREQ) { tmp_lit("*="); }
    else if (k == T_SLASHEQ) { tmp_lit("/="); }
    else if (k == T_PERCENTEQ) { tmp_lit("%="); }
    else if (k == T_AMPEQ) { tmp_lit("&="); }
    else if (k == T_PIPEEQ) { tmp_lit("|="); }
    else if (k == T_CARETEQ) { tmp_lit("^="); }
    else if (k == T_SHLEQ) { tmp_lit("<<="); }
    else if (k == T_SHREQ) { tmp_lit(">>="); }
    else if (k == T_INC) { tmp_lit("++"); }
    else if (k == T_DEC) { tmp_lit("--"); }
    else if (k == T_HASH) { tmp_lit("#"); }
    else if (k == T_HASHHASH) { tmp_lit("##"); }
    else if (k == T_PLUSPIPE) { tmp_lit("+|"); }
    else if (k == T_MINUSPIPE) { tmp_lit("-|"); }
    else if (k == T_STARPIPE) { tmp_lit("*|"); }
    else if (k == T_PLUSPCT) { tmp_lit("+%"); }
    else if (k == T_MINUSPCT) { tmp_lit("-%"); }
    else if (k == T_STARPCT) { tmp_lit("*%"); }
    else if (k == T_FATARROW) { tmp_lit("=>"); }
    else if (k == T_EOF) { tmp_lit("end of file"); }
}

// Appends a token's source spelling to tmp (strings/chars re-escaped).
void tmp_tok_spelling(int kind, int sid) {
    if (kind == T_IDENT || kind == T_INT || kind == T_FLOAT || kind >= K_FIRST) { tmp_str(sid); return; }
    if (kind == T_STRING || kind == T_CHAR) {
        int q = '"';
        if (kind == T_CHAR) { q = '\''; }
        tmp_char(q);
        int n = g_str_len[sid];
        int i = 0;
        while (i < n) {
            int c = str_char(sid, i);
            if (c == '\\' || c == q) { tmp_char('\\'); tmp_char(c); }
            else if (c == 10) { tmp_char('\\'); tmp_char('n'); }
            else if (c == 9) { tmp_char('\\'); tmp_char('t'); }
            else if (c == 13) { tmp_char('\\'); tmp_char('r'); }
            else if (c == 0) { tmp_char('\\'); tmp_char('0'); }
            else { tmp_char(c); }
            i = i + 1;
        }
        tmp_char(q);
        return;
    }
    tmp_punct(kind);
}
