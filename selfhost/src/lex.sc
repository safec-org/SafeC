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

// ── exact decimal → double (round to nearest even) ───────────────────────────
// big integers: base 2^32 limbs, little endian
#define BN 96
unsigned int g_bn_m[BN];   // the decimal mantissa
int g_bn_mn = 0;
unsigned int g_bn_d[BN];   // divisor 10^k
int g_bn_dn = 0;
unsigned int g_bn_q[BN];   // quotient
int g_bn_qn = 0;
unsigned int g_bn_r[BN];   // remainder
int g_bn_rn = 0;
int g_dc_ndig = 0;
int g_dc_drop = 0;         // digits beyond capacity (scale)
int g_dc_sticky = 0;       // nonzero digits dropped

void dc_reset() {
    g_bn_mn = 0;
    g_dc_ndig = 0;
    g_dc_drop = 0;
    g_dc_sticky = 0;
}

// x = x * mul + add (for x in g_bn_m / g_bn_d selected by which)
void bn_muladd(int which, unsigned int mul, unsigned int add) {
    unsigned long carry = (unsigned long)add;
    int n = g_bn_mn;
    if (which == 1) { n = g_bn_dn; }
    int i = 0;
    while (i < n) {
        unsigned long cur = 0UL;
        if (which == 0) { cur = (unsigned long)g_bn_m[i] * (unsigned long)mul + carry; g_bn_m[i] = (unsigned int)(cur & 4294967295UL); }
        else { cur = (unsigned long)g_bn_d[i] * (unsigned long)mul + carry; g_bn_d[i] = (unsigned int)(cur & 4294967295UL); }
        carry = cur >> 32UL;
        i = i + 1;
    }
    if (carry != 0UL && n < BN) {
        if (which == 0) { g_bn_m[n] = (unsigned int)carry; g_bn_mn = n + 1; }
        else { g_bn_d[n] = (unsigned int)carry; g_bn_dn = n + 1; }
    }
}

void dc_digit(int d) {
    if (g_bn_mn >= BN - 4) {
        // more digits than needed for correct rounding: scale instead
        g_dc_drop = g_dc_drop + 1;
        if (d != 0) { g_dc_sticky = 1; }
        return;
    }
    if (g_bn_mn == 0) {
        if (d == 0) { return; }
        g_bn_m[0] = (unsigned int)d;
        g_bn_mn = 1;
        return;
    }
    bn_muladd(0, 10U, (unsigned int)d);
}

int bn_bits(int which) {
    int n = g_bn_mn;
    if (which == 1) { n = g_bn_dn; }
    if (which == 2) { n = g_bn_qn; }
    while (n > 0) {
        unsigned int top = g_bn_m[n - 1];
        if (which == 1) { top = g_bn_d[n - 1]; }
        if (which == 2) { top = g_bn_q[n - 1]; }
        if (top != 0U) {
            int b = 32;
            while ((top & 2147483648U) == 0U) { top = top << 1U; b = b - 1; }
            return (n - 1) * 32 + b;
        }
        n = n - 1;
    }
    return 0;
}

int bn_qbit(int i) { return (int)((g_bn_q[i / 32] >> (unsigned int)(i % 32)) & 1U); }

// rounds q * 2^e2 (+ sticky) to the nearest double
long bn_q_to_f64(int e2, int sticky) {
    int L = bn_bits(2);
    if (L == 0) { return 0L; }
    int t = e2 + L - 53;
    if (t < -1074) { t = -1074; }
    int shift = t - e2;
    unsigned long m = 0UL;
    if (shift <= 0) {
        // exact: q fits in 53 bits
        int i = L - 1;
        while (i >= 0) { m = (m << 1UL) | (unsigned long)bn_qbit(i); i = i - 1; }
        m = m << (unsigned long)(0 - shift);
    } else {
        int i = L - 1;
        while (i >= shift) { m = (m << 1UL) | (unsigned long)bn_qbit(i); i = i - 1; }
        int half = 0;
        if (shift - 1 < L) { half = bn_qbit(shift - 1); }
        int rest = sticky;
        i = shift - 2;
        while (i >= 0 && rest == 0) { if (i < L && bn_qbit(i) == 1) { rest = 1; } i = i - 1; }
        if (half == 1 && (rest == 1 || (m & 1UL) == 1UL)) { m = m + 1UL; }
    }
    if (m >= 9007199254740992UL) { m = m >> 1UL; t = t + 1; }
    if (m == 0UL) { return 0L; }
    if (m >= 4503599627370496UL) {
        long ef = (long)(t + 52 + 1023);
        if (ef >= 2047L) { return 9218868437227405312L; }
        return (ef << 52L) | (long)(m - 4503599627370496UL);
    }
    return (long)m;   // subnormal (t == -1074)
}

// the double nearest to (decimal mantissa in g_bn_m) * 10^exp10
long dc_to_f64_bits(int exp10) {
    exp10 = exp10 + g_dc_drop;
    if (g_bn_mn == 0) { return 0L; }
    if (exp10 > 400) { return 9218868437227405312L; }
    if (exp10 < -400 - g_bn_mn * 10) { return 0L; }
    if (exp10 >= 0) {
        int k = exp10;
        while (k > 0 && g_bn_mn < BN) { bn_muladd(0, 10U, 0U); k = k - 1; }
        int i = 0;
        while (i < g_bn_mn) { g_bn_q[i] = g_bn_m[i]; i = i + 1; }
        g_bn_qn = g_bn_mn;
        return bn_q_to_f64(0, g_dc_sticky);
    }
    // q = floor(m * 2^s / 10^k), s chosen for >= 64 quotient bits
    int k = 0 - exp10;
    g_bn_d[0] = 1U;
    g_bn_dn = 1;
    int j = 0;
    while (j < k && g_bn_dn < BN - 1) { bn_muladd(1, 10U, 0U); j = j + 1; }
    int s = 66 + bn_bits(1) - bn_bits(0);
    if (s < 0) { s = 0; }
    int nb = bn_bits(0) + s;      // bits of the dividend m * 2^s
    // shift-subtract long division, one dividend bit at a time
    g_bn_rn = g_bn_dn + 1;
    int i = 0;
    while (i < g_bn_rn) { g_bn_r[i] = 0U; i = i + 1; }
    g_bn_qn = nb / 32 + 1;
    if (g_bn_qn > BN) { g_bn_qn = BN; }
    i = 0;
    while (i < g_bn_qn) { g_bn_q[i] = 0U; i = i + 1; }
    int bit = nb - 1;
    while (bit >= 0) {
        // r = r * 2 + (bit of m*2^s)
        int src = bit - s;
        unsigned int in = 0U;
        if (src >= 0) { in = (g_bn_m[src / 32] >> (unsigned int)(src % 32)) & 1U; }
        unsigned int carry = in;
        i = 0;
        while (i < g_bn_rn) {
            unsigned int top = g_bn_r[i] >> 31U;
            g_bn_r[i] = (g_bn_r[i] << 1U) | carry;
            carry = top;
            i = i + 1;
        }
        // r >= d ?
        int ge = 1;
        i = g_bn_rn - 1;
        while (i >= 0) {
            unsigned int dv = 0U;
            if (i < g_bn_dn) { dv = g_bn_d[i]; }
            if (g_bn_r[i] != dv) {
                if (g_bn_r[i] < dv) { ge = 0; }
                break;
            }
            i = i - 1;
        }
        if (ge == 1) {
            long borrow = 0L;
            i = 0;
            while (i < g_bn_rn) {
                long dv = 0L;
                if (i < g_bn_dn) { dv = (long)g_bn_d[i]; }
                long diff = (long)g_bn_r[i] - dv - borrow;
                if (diff < 0L) { diff = diff + 4294967296L; borrow = 1L; } else { borrow = 0L; }
                g_bn_r[i] = (unsigned int)diff;
                i = i + 1;
            }
            if (bit / 32 < g_bn_qn) { g_bn_q[bit / 32] = g_bn_q[bit / 32] | (1U << (unsigned int)(bit % 32)); }
        }
        bit = bit - 1;
    }
    int sticky = g_dc_sticky;
    i = 0;
    while (i < g_bn_rn) { if (g_bn_r[i] != 0U) { sticky = 1; } i = i + 1; }
    return bn_q_to_f64(0 - s, sticky);
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
        // decimal integer or float: all digits go to the exact converter
        dc_reset();
        int exp10 = 0;
        while (is_digit(lx_peek(0))) {
            int d = lx_adv() - '0';
            v = v *| 10L +| (long)d;
            dc_digit(d);
        }
        if (lx_peek(0) == '.' && lx_peek(1) != '.') {
            is_float = 1;
            lx_adv();
            while (is_digit(lx_peek(0))) { dc_digit(lx_adv() - '0'); exp10 = exp10 - 1; }
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
        if (is_float == 1) { v = dc_to_f64_bits(exp10); }
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
        if (is_digit(c)) {
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

// IEEE single bits of the double with bits 'b' (round to nearest even)
long f32_bits_of_f64(long b) {
    long sign = (b >> 63L) & 1L;
    long ex = (b >> 52L) & 2047L;
    long mant = b & 4503599627370495L;
    if (ex == 2047L) {
        if (mant != 0L) { return (sign << 31L) | 2143289344L; }   // NaN
        return (sign << 31L) | 2139095040L;                        // inf
    }
    if (ex == 0L) { return sign << 31L; }
    long e = ex - 1023L + 127L;
    long m = mant | 4503599627370496L;    // implicit 1, 53 bits
    int shift = 29;
    if (e <= 0L) {
        shift = 29 + (int)(1L - e);
        e = 0L;
        if (shift > 60) { return sign << 31L; }
    }
    long q = m >> (long)shift;
    long r = m & ((1L << (long)shift) - 1L);
    long half = 1L << (long)(shift - 1);
    if (r > half || (r == half && (q & 1L) != 0L)) { q = q + 1L; }
    if (e == 0L) {
        // subnormal (q may round up into the smallest normal)
        return (sign << 31L) | q;
    }
    if (q >= 16777216L) { q = q >> 1L; e = e + 1L; }
    if (e >= 255L) { return (sign << 31L) | 2139095040L; }
    return (sign << 31L) | (e << 23L) | (q & 8388607L);
}
