#include "safec/Lexer.h"
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <unordered_map>

namespace safec {

// ── Keyword table ─────────────────────────────────────────────────────────────
TK Lexer::keywordKind(const std::string &w) {
    static const std::unordered_map<std::string, TK> kw = {
        // C keywords
        {"auto",          TK::KW_auto},
        {"break",         TK::KW_break},
        {"case",          TK::KW_case},
        {"char",          TK::KW_char},
        {"const",         TK::KW_const},
        {"continue",      TK::KW_continue},
        {"default",       TK::KW_default},
        {"do",            TK::KW_do},
        {"double",        TK::KW_double},
        {"else",          TK::KW_else},
        {"enum",          TK::KW_enum},
        {"extern",        TK::KW_extern},
        {"float",         TK::KW_float},
        {"for",           TK::KW_for},
        {"goto",          TK::KW_goto},
        {"if",            TK::KW_if},
        {"inline",        TK::KW_inline},
        {"int",           TK::KW_int},
        {"long",          TK::KW_long},
        {"register",      TK::KW_register},
        {"restrict",      TK::KW_restrict},
        {"return",        TK::KW_return},
        {"short",         TK::KW_short},
        {"signed",        TK::KW_signed},
        {"sizeof",        TK::KW_sizeof},
        {"static",        TK::KW_static},
        {"struct",        TK::KW_struct},
        {"switch",        TK::KW_switch},
        {"typedef",       TK::KW_typedef},
        {"union",         TK::KW_union},
        {"unsigned",      TK::KW_unsigned},
        {"void",          TK::KW_void},
        {"volatile",      TK::KW_volatile},
        {"while",         TK::KW_while},
        {"bool",          TK::KW_bool},
        {"true",          TK::KW_true},
        {"false",         TK::KW_false},
        {"null",          TK::KW_null},
        // SafeC keywords
        {"region",        TK::KW_region},
        {"unsafe",        TK::KW_unsafe},
        {"consteval",     TK::KW_consteval},
        {"generic",       TK::KW_generic},
        {"static_assert", TK::KW_static_assert},
        // Contextual (we tokenize them as keywords but allow ident use in parser)
        {"stack",         TK::KW_stack},
        {"heap",          TK::KW_heap},
        {"arena",         TK::KW_arena},
        {"capacity",      TK::KW_capacity},
        // Method / object model keywords
        {"self",          TK::KW_self},
        {"operator",      TK::KW_operator},
        {"new",           TK::KW_new},
        {"arena_reset",   TK::KW_arena_reset},
        {"tuple",         TK::KW_tuple},
        {"spawn",         TK::KW_spawn},
        {"join",          TK::KW_join},
        // New SafeC features
        {"defer",         TK::KW_defer},
        {"errdefer",      TK::KW_errdefer},
        {"match",         TK::KW_match},
        {"packed",        TK::KW_packed},
        {"try",           TK::KW_try},
        {"must_use",      TK::KW_must_use},
        {"fn",            TK::KW_fn},
        {"alignof",       TK::KW_alignof},
        {"typeof",        TK::KW_typeof},
        {"fieldcount",    TK::KW_fieldcount},
    };
    auto it = kw.find(w);
    return (it != kw.end()) ? it->second : TK::Ident;
}

// ── Constructor ───────────────────────────────────────────────────────────────
Lexer::Lexer(std::string source, const char *filename, DiagEngine &diag)
    : src_(std::move(source)), filename_(filename), diag_(diag) {}

// ── Character helpers ─────────────────────────────────────────────────────────
char Lexer::peek(int offset) const {
    size_t idx = pos_ + offset;
    return (idx < src_.size()) ? src_[idx] : '\0';
}

char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') { ++line_; col_ = 1; }
    else           { ++col_; }
    return c;
}

bool Lexer::skipLineComment() {
    if (peek() == '/' && peek(1) == '/') {
        while (!atEnd() && peek() != '\n') advance();
        return true;
    }
    return false;
}

bool Lexer::skipBlockComment() {
    if (peek() == '/' && peek(1) == '*') {
        advance(); advance();
        while (!atEnd()) {
            if (peek() == '*' && peek(1) == '/') {
                advance(); advance();
                return true;
            }
            advance();
        }
        diag_.error(curLoc(), "unterminated block comment");
        return true;
    }
    return false;
}

void Lexer::skipWhitespaceAndComments() {
    while (!atEnd()) {
        char c = peek();
        if (std::isspace(c)) { advance(); continue; }
        if (skipLineComment())  continue;
        if (skipBlockComment()) continue;
        break;
    }
}

// ── Number ────────────────────────────────────────────────────────────────────
Token Lexer::lexNumber() {
    startLoc_ = curLoc();
    std::string s;
    bool isFloat = false;
    bool isHex   = false;

    if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
        s += advance(); s += advance(); // "0x"
        isHex = true;
        while (!atEnd() && std::isxdigit(peek())) s += advance();
    } else {
        while (!atEnd() && std::isdigit(peek())) s += advance();
        if (!atEnd() && peek() == '.' && std::isdigit(peek(1))) {
            isFloat = true;
            s += advance();
            while (!atEnd() && std::isdigit(peek())) s += advance();
        }
        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            isFloat = true;
            s += advance();
            if (!atEnd() && (peek() == '+' || peek() == '-')) s += advance();
            while (!atEnd() && std::isdigit(peek())) s += advance();
        }
    }

    // Save the numeric-only part for parsing before consuming the suffix.
    std::string digits = s;

    // Optional suffix: u/U (unsigned), l/L (long), ll/LL (long long), f/F (float).
    bool isUnsigned = false;
    bool isLongLong = false;
    while (!atEnd() && (peek() == 'u' || peek() == 'U' ||
                        peek() == 'l' || peek() == 'L' ||
                        peek() == 'f' || peek() == 'F')) {
        char sf = peek();
        s += advance();
        if (sf == 'f' || sf == 'F') isFloat = true;
        if (sf == 'u' || sf == 'U') isUnsigned = true;
        if (sf == 'l' || sf == 'L') {
            // Check for LL/ll suffix
            if (!atEnd() && (peek() == 'l' || peek() == 'L')) {
                s += advance(); // consume second l/L
                isLongLong = true;
            } else {
                isLongLong = true; // L alone also forces long
            }
        }
    }

    Token tok;
    tok.loc        = startLoc_;
    tok.text       = s;
    tok.isLongLong = isLongLong;
    tok.isUnsigned = isUnsigned;
    if (isFloat) {
        tok.kind     = TK::FloatLit;
        tok.floatVal = std::stod(digits);
    } else {
        tok.kind = TK::IntLit;
        // For unsigned literals (or numbers that exceed LLONG_MAX),
        // parse as unsigned long long and reinterpret the bit pattern.
        int base = isHex ? 16 : 10;
        if (isUnsigned) {
            // Parse as unsigned; reinterpret bits as int64_t.
            errno = 0;
            unsigned long long uval = std::strtoull(digits.c_str(), nullptr, base);
            tok.intVal = static_cast<int64_t>(uval);
        } else {
            // Try signed; if it overflows LLONG range, reinterpret as unsigned.
            errno = 0;
            long long sval = std::strtoll(digits.c_str(), nullptr, base);
            if (errno == ERANGE) {
                // Number exceeds signed range — parse unsigned and reinterpret.
                errno = 0;
                unsigned long long uval = std::strtoull(digits.c_str(), nullptr, base);
                tok.intVal = static_cast<int64_t>(uval);
            } else {
                tok.intVal = sval;
            }
        }
    }
    return tok;
}

// ── String ────────────────────────────────────────────────────────────────────
Token Lexer::lexString() {
    startLoc_ = curLoc();
    advance(); // consume opening "
    std::string s;
    while (!atEnd() && peek() != '"') {
        char c = advance();
        if (c == '\\') {
            if (atEnd()) break;
            char esc = advance();
            switch (esc) {
            case 'n':  s += '\n'; break;
            case 't':  s += '\t'; break;
            case 'r':  s += '\r'; break;
            case '\\': s += '\\'; break;
            case '"':  s += '"';  break;
            case '0':  s += '\0'; break;
            default:   s += '\\'; s += esc;
            }
        } else {
            s += c;
        }
    }
    if (atEnd()) diag_.error(startLoc_, "unterminated string literal");
    else advance(); // consume closing "
    return {TK::StringLit, s, startLoc_};
}

// ── Char ──────────────────────────────────────────────────────────────────────
Token Lexer::lexChar() {
    startLoc_ = curLoc();
    advance(); // consume '
    std::string s;
    char c = advance();
    if (c == '\\') {
        char esc = advance();
        switch (esc) {
        case 'n': c = '\n'; break; case 't': c = '\t'; break;
        case 'r': c = '\r'; break; case '\\': c = '\\'; break;
        case '\'': c = '\''; break; case '0': c = '\0'; break;
        default: c = esc;
        }
    }
    s += c;
    if (peek() != '\'') diag_.error(startLoc_, "expected closing ' in char literal");
    else advance();
    Token tok{TK::CharLit, s, startLoc_};
    tok.intVal = static_cast<unsigned char>(s[0]);
    return tok;
}

// ── Identifier or keyword ─────────────────────────────────────────────────────
Token Lexer::lexIdOrKeyword() {
    startLoc_ = curLoc();
    std::string s;
    while (!atEnd() && (std::isalnum(peek()) || peek() == '_')) s += advance();
    TK k = keywordKind(s);
    return {k, s, startLoc_};
}

// ── Punctuation & operators ───────────────────────────────────────────────────
Token Lexer::lexPunct() {
    startLoc_ = curLoc();
    char c = advance();
    auto mk = [&](TK k, std::string s) { return Token{k, std::move(s), startLoc_}; };
    auto next1 = [&]() { return peek(); };
    auto eat   = [&](char e) -> bool {
        if (peek() == e) { advance(); return true; }
        return false;
    };

    switch (c) {
    case '+':
        if (eat('+')) return mk(TK::PlusPlus, "++");
        if (eat('=')) return mk(TK::PlusEq,   "+=");
        return mk(TK::Plus, "+");
    case '-':
        if (eat('-')) return mk(TK::MinusMinus, "--");
        if (eat('>')) return mk(TK::Arrow,      "->");
        if (eat('=')) return mk(TK::MinusEq,    "-=");
        return mk(TK::Minus, "-");
    case '*':
        if (eat('=')) return mk(TK::StarEq, "*=");
        return mk(TK::Star, "*");
    case '/':
        if (eat('=')) return mk(TK::SlashEq, "/=");
        return mk(TK::Slash, "/");
    case '%':
        if (eat('=')) return mk(TK::PercentEq, "%=");
        return mk(TK::Percent, "%");
    case '&':
        if (eat('&')) return mk(TK::AmpAmp, "&&");
        if (eat('=')) return mk(TK::AmpEq,  "&=");
        return mk(TK::Amp, "&");
    case '|':
        if (eat('|')) return mk(TK::PipePipe, "||");
        if (eat('=')) return mk(TK::PipeEq,   "|=");
        return mk(TK::Pipe, "|");
    case '^':
        if (eat('=')) return mk(TK::CaretEq, "^=");
        return mk(TK::Caret, "^");
    case '~': return mk(TK::Tilde, "~");
    case '!':
        if (eat('=')) return mk(TK::BangEq, "!=");
        return mk(TK::Bang, "!");
    case '=':
        if (eat('=')) return mk(TK::EqEq, "==");
        if (eat('>')) return mk(TK::FatArrow, "=>");
        return mk(TK::Eq, "=");
    case '<':
        if (eat('<')) {
            if (eat('=')) return mk(TK::LShiftEq, "<<=");
            return mk(TK::LShift, "<<");
        }
        if (eat('=')) return mk(TK::LtEq, "<=");
        return mk(TK::Lt, "<");
    case '>':
        if (eat('>')) {
            if (eat('=')) return mk(TK::RShiftEq, ">>=");
            return mk(TK::RShift, ">>");
        }
        if (eat('=')) return mk(TK::GtEq, ">=");
        return mk(TK::Gt, ">");
    case '?':
        if (next1() == '&') { advance(); return mk(TK::QuestionAmp, "?&"); }
        return mk(TK::Question, "?");
    case ':':
        if (peek() == ':') { advance(); return mk(TK::ColonColon, "::"); }
        return mk(TK::Colon, ":");
    case '.':
        if (peek() == '.' && peek(1) == '.') { advance(); advance(); return mk(TK::DotDotDot, "..."); }
        return mk(TK::Dot, ".");
    case '(': return mk(TK::LParen, "(");
    case ')': return mk(TK::RParen, ")");
    case '{': return mk(TK::LBrace, "{");
    case '}': return mk(TK::RBrace, "}");
    case '[': return mk(TK::LBracket, "[");
    case ']': return mk(TK::RBracket, "]");
    case ';': return mk(TK::Semicolon, ";");
    case ',': return mk(TK::Comma, ",");
    case '#': return mk(TK::Hash, "#");
    default: {
        diag_.error(startLoc_, std::string("unexpected character '") + c + "'");
        return mk(TK::Invalid, std::string(1, c));
    }
    }
}

// ── next() ────────────────────────────────────────────────────────────────────
Token Lexer::next() {
    skipWhitespaceAndComments();
    if (atEnd()) return {TK::Eof, "", {line_, col_, filename_}};

    char c = peek();
    if (std::isdigit(c)) return lexNumber();
    if (c == '"')         return lexString();
    if (c == '\'')        return lexChar();
    if (std::isalpha(c) || c == '_') return lexIdOrKeyword();
    return lexPunct();
}

// ── lexAll() ──────────────────────────────────────────────────────────────────
std::vector<Token> Lexer::lexAll() {
    std::vector<Token> toks;
    while (true) {
        Token t = next();
        toks.push_back(t);
        if (t.is(TK::Eof)) break;
    }
    return toks;
}

// ── Token::kindName() ─────────────────────────────────────────────────────────
const char *Token::kindName() const {
    switch (kind) {
    case TK::IntLit:       return "integer literal";
    case TK::FloatLit:     return "float literal";
    case TK::StringLit:    return "string literal";
    case TK::CharLit:      return "char literal";
    case TK::Ident:        return "identifier";
    case TK::KW_int:       return "'int'";
    case TK::KW_void:      return "'void'";
    case TK::KW_return:    return "'return'";
    case TK::KW_if:        return "'if'";
    case TK::KW_else:      return "'else'";
    case TK::KW_while:     return "'while'";
    case TK::KW_for:       return "'for'";
    case TK::KW_struct:    return "'struct'";
    case TK::KW_enum:      return "'enum'";
    case TK::KW_region:    return "'region'";
    case TK::KW_unsafe:    return "'unsafe'";
    case TK::KW_generic:   return "'generic'";
    case TK::LParen:       return "'('";
    case TK::RParen:       return "')'";
    case TK::LBrace:       return "'{'";
    case TK::RBrace:       return "'}'";
    case TK::Semicolon:    return "';'";
    case TK::Comma:        return "','";
    case TK::Eq:           return "'='";
    case TK::FatArrow:     return "'=>'";
    case TK::KW_must_use:  return "'must_use'";
    case TK::KW_fn:        return "'fn'";
    case TK::Eof:          return "end of file";
    default:               return "<token>";
    }
}

} // namespace safec
