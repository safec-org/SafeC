#include "safec/Parser.h"
#include <cassert>
#include <cctype>
#include <functional>
#include <sstream>

namespace safec {

// ── Constructor ───────────────────────────────────────────────────────────────
Parser::Parser(std::vector<Token> tokens, DiagEngine &diag)
    : toks_(std::move(tokens)), diag_(diag) {
    // Ensure there's always an EOF
    if (toks_.empty() || !toks_.back().is(TK::Eof))
        toks_.push_back({TK::Eof, "", {}});
}

// ── Token stream helpers ──────────────────────────────────────────────────────
Token Parser::consume() {
    Token t = toks_[pos_];
    if (pos_ + 1 < toks_.size()) ++pos_;
    return t;
}

Token Parser::expect(TK kind, const char *msg) {
    if (cur().is(kind)) return consume();
    std::string err = msg ? msg :
        (std::string("expected ") + Token{kind, {}, {}}.kindName() +
         ", got " + cur().kindName());
    diag_.error(cur().loc, err);
    // Don't advance — return a synthetic token for error recovery
    return {kind, "", cur().loc};
}

bool Parser::match(TK k) {
    if (cur().is(k)) { consume(); return true; }
    return false;
}

bool Parser::checkIdent(const char *name) const {
    return (cur().is(TK::Ident) || cur().is(TK::KW_stack) ||
            cur().is(TK::KW_heap) || cur().is(TK::KW_arena) ||
            cur().is(TK::KW_capacity)) && cur().text == name;
}

// ── Error recovery ────────────────────────────────────────────────────────────
void Parser::syncToDecl() {
    while (!atEnd()) {
        switch (cur().kind) {
        case TK::KW_int: case TK::KW_void: case TK::KW_char:
        case TK::KW_float: case TK::KW_double: case TK::KW_bool:
        case TK::KW_long: case TK::KW_short: case TK::KW_unsigned:
        case TK::KW_signed: case TK::KW_struct: case TK::KW_enum:
        case TK::KW_region: case TK::KW_generic: case TK::KW_extern:
        case TK::KW_static: case TK::KW_inline: case TK::KW_const:
        case TK::KW_consteval: case TK::KW_typedef:
            return;
        default: consume();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPE PARSING
// ─────────────────────────────────────────────────────────────────────────────

// Returns true if current token could start a type
bool Parser::isTypeStart(const Token &t) const {
    switch (t.kind) {
    case TK::KW_int: case TK::KW_void: case TK::KW_char:
    case TK::KW_float: case TK::KW_double: case TK::KW_bool:
    case TK::KW_long: case TK::KW_short: case TK::KW_unsigned:
    case TK::KW_signed: case TK::KW_struct: case TK::KW_enum:
    case TK::KW_const:
    case TK::KW_volatile: // e.g. cast '(volatile T*)expr'
    case TK::KW_tuple:    // tuple type
    case TK::KW_fn:       // fn ReturnType(Params) — function pointer type
    case TK::KW_typeof:   // typeof(expr) — type position
    case TK::Amp:         // &stack T
    case TK::QuestionAmp: // ?&stack T
    case TK::Question:    // ?T (optional)
        return true;
    case TK::Ident:
        return true; // user-defined type name
    default: return false;
    }
}

// Could this token plausibly start a cast operand (a unary/postfix/primary
// expression)? Used to reject a speculative '(name)' cast parse when name is
// a bare identifier (always a valid type-start guess, since it might be a
// user type) but what follows makes it obvious this was actually a
// parenthesized lvalue/expression instead — e.g. '(x) = 5' or '(x).field'.
// This doesn't solve the general typedef-name ambiguity (that needs a live
// type-name table threaded through parsing); it only catches the common,
// unambiguous case where the following token can never begin an operand.
static bool canStartCastOperand(const Token &t) {
    switch (t.kind) {
    case TK::IntLit: case TK::FloatLit: case TK::StringLit: case TK::CharLit:
    case TK::KW_true: case TK::KW_false: case TK::KW_null:
    case TK::Ident: case TK::KW_stack: case TK::KW_heap:
    case TK::KW_arena: case TK::KW_capacity: case TK::KW_self:
    case TK::LParen: case TK::LBrace:
    case TK::Minus: case TK::Plus: case TK::Bang: case TK::Tilde:
    case TK::PlusPlus: case TK::MinusMinus:
    case TK::Amp: case TK::Star: case TK::KW_sizeof: case TK::KW_new:
        return true;
    default:
        return false;
    }
}

bool Parser::isTypeStart() const { return isTypeStart(cur()); }

// Parse a region qualifier: 'stack' | 'static' | 'heap' | 'arena' '<' Ident '>'
Region Parser::parseRegionQualifier(std::string &arenaName) {
    if (cur().is(TK::KW_stack)  || cur().isIdent("stack"))  { consume(); return Region::Stack; }
    if (cur().is(TK::KW_heap)   || cur().isIdent("heap"))   { consume(); return Region::Heap; }
    if (cur().is(TK::KW_static) || cur().is(TK::KW_static)) {
        if (cur().is(TK::KW_static)) { consume(); return Region::Static; }
    }
    if (checkIdent("static")) { consume(); return Region::Static; }
    if (cur().is(TK::KW_arena) || cur().isIdent("arena")) {
        consume();
        expect(TK::Lt, "expected '<' after 'arena'");
        arenaName = cur().text;
        if (!cur().is(TK::Ident) && !cur().is(TK::KW_stack)) {
            diag_.error(cur().loc, "expected arena region name");
        } else consume();
        expect(TK::Gt, "expected '>' after arena region name");
        return Region::Arena;
    }
    diag_.error(cur().loc, "expected region qualifier: stack, heap, static, or arena<R>");
    return Region::Unknown;
}

// Parse a reference type: [?]& region T
TypePtr Parser::parseReferenceType(bool nullable, bool leadingConst) {
    // We already consumed '&' or '?&' (and any 'const' preceding it)
    std::string arenaName;
    Region r = parseRegionQualifier(arenaName);

    // Parse optional const — either written before the sigil ('const &stack T')
    // or after the region ('&stack const T'); both mean the same thing.
    bool isConst = leadingConst;
    if (cur().is(TK::KW_const)) { consume(); isConst = true; }

    TypePtr base = parseBaseType();
    // Pointer / array suffixes, e.g. '&stack char* saveptr' (reference to a pointer)
    base = parseTypeDeclarator(std::move(base));
    return makeReference(std::move(base), r, nullable, !isConst, std::move(arenaName));
}

// Parse a base type (no declarator suffixes yet)
TypePtr Parser::parseBaseType() {
    bool hasSign    = false;
    bool isUnsigned = false;

    // A bare 'const' on a base type (as opposed to a declarator: '*const' or
    // '&const') is a no-op qualifier here — enforcement of immutability lives
    // on references (parseReferenceType) and local/global declarations
    // (parseVarDeclStmt), matching 'restrict' below which is likewise ignored.
    if (cur().is(TK::KW_const)) { consume(); }
    // 'restrict' and compiler-specific '__restrict' are ignored qualifiers in SafeC
    if (cur().is(TK::KW_restrict)) { consume(); }
    // 'volatile' as a bare type qualifier (e.g. imported from C headers like
    // <stdatomic.h>) is likewise a no-op here — SafeC's own volatile MMIO
    // access goes through the explicit volatile_load/volatile_store builtins.
    if (cur().is(TK::KW_volatile)) { consume(); }
    // C11 '_Atomic' qualifier-prefix form ('_Atomic int', as opposed to the
    // '_Atomic(int)' generic-selection form filtered out at header-import
    // time) — not lexed as a keyword, so it reaches here as a plain
    // identifier. SafeC has no atomic type qualifier; code that casts through
    // it (e.g. std/atomic.sc calling into <stdatomic.h>) just needs it to
    // parse as a no-op, matching const/restrict/volatile above.
    if (cur().is(TK::Ident) && cur().text == "_Atomic") { consume(); }
    if (cur().is(TK::KW_unsigned)) { consume(); isUnsigned = true; hasSign = true; }
    else if (cur().is(TK::KW_signed)) { consume(); hasSign = true; }

    auto loc = cur().loc;

    // vec<T, N> — fixed-width SIMD vector type (see std::simd). Contextual,
    // like 'align(N)' elsewhere: 'vec' isn't a reserved keyword (a plain
    // variable/field named 'vec' must keep working), so only treat it as
    // this type constructor when immediately followed by '<'.
    if (cur().isIdent("vec") && peek().is(TK::Lt)) {
        consume(); // 'vec'
        consume(); // '<'
        TypePtr elem = parseType();
        expect(TK::Comma, "expected ',' after vector element type");
        int64_t width = parseArraySizeConst();
        expect(TK::Gt, "expected '>' closing vec<T, N>");
        if (width <= 0) {
            diag_.error(loc, "vector width must be a positive constant");
            width = 1;
        }
        return makeVector(std::move(elem), static_cast<int>(width));
    }

    switch (cur().kind) {
    case TK::KW_void:   consume(); return makeVoid();
    case TK::KW_bool:   consume(); return makeBool();
    case TK::KW_char:   consume(); return isUnsigned ? makeInt(8, false) : makeChar();
    case TK::KW_short:  consume(); return makeInt(16, !isUnsigned);
    case TK::KW_int:    consume(); return makeInt(32, !isUnsigned);
    case TK::KW_long: {
        consume();
        if (cur().is(TK::KW_long)) consume(); // long long
        return makeInt(64, !isUnsigned);
    }
    case TK::KW_float:  consume(); return makeFloat(32);
    case TK::KW_double: consume(); return makeFloat(64);
    case TK::KW_struct:
    case TK::KW_union: {
        bool isUnion = cur().is(TK::KW_union);
        consume();
        // Inline body (named or anonymous): 'struct { ... }' / 'union { ...
        // }' — most commonly an anonymous struct/union member (see
        // parseStructDecl's field loop, which flags these for flattened
        // member lookup), but 'struct Tag { ... } x;' in type position is
        // also valid C. Reuses parseStructDecl unmodified (so name/'{'/
        // fields/'}' parsing stays in exactly one place) and drains the
        // result into TranslationUnit::decls (see pendingAnonStructs_) so
        // Sema's normal collectStruct()/resolveType() picks it up — this
        // function just returns the placeholder type referencing its name.
        if (cur().is(TK::LBrace) || (cur().is(TK::Ident) && peek().is(TK::LBrace))) {
            auto sd = parseStructDecl(isUnion, /*consumeTrailingSemicolon=*/false);
            if (sd->name.empty())
                sd->name = "__anon_struct_" + std::to_string(anonStructCounter_++);
            std::string bodyName = sd->name;
            pendingAnonStructs_.push_back(std::move(sd));
            return std::make_shared<StructType>(bodyName, isUnion);
        }
        std::string name;
        if (cur().is(TK::Ident)) { name = cur().text; consume(); }
        else diag_.error(cur().loc, isUnion ? "expected union name" : "expected struct name");
        // Create a named struct/union type (to be resolved by sema)
        return std::make_shared<StructType>(name, isUnion);
    }
    case TK::KW_enum: {
        consume();
        std::string name = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected enum name");
        } else consume();
        return std::make_shared<EnumType>(name);
    }
    case TK::KW_tuple: {
        // tuple(T1, T2, ...) type syntax
        consume();
        expect(TK::LParen, "expected '(' after 'tuple'");
        std::vector<TypePtr> elems;
        while (!atEnd() && !cur().is(TK::RParen)) {
            elems.push_back(parseType());
            if (!match(TK::Comma)) break;
        }
        expect(TK::RParen, "expected ')' closing tuple type");
        return makeTuple(std::move(elems));
    }
    case TK::KW_fn: {
        // fn ReturnType(ParamType, ...) — safe function pointer type
        consume(); // eat 'fn'
        TypePtr ret = parseBaseType();
        ret = parseTypeDeclarator(std::move(ret));
        expect(TK::LParen, "expected '(' after 'fn' return type");
        std::vector<TypePtr> params;
        bool variadic = false;
        if (!cur().is(TK::RParen)) {
            do {
                if (cur().is(TK::DotDotDot)) { consume(); variadic = true; break; }
                if (cur().is(TK::KW_void) && peek().is(TK::RParen)) { consume(); break; }
                params.push_back(parseType());
                // allow optional param name (ignored in type context)
                if (cur().is(TK::Ident)) consume();
            } while (match(TK::Comma));
        }
        expect(TK::RParen, "expected ')' closing fn parameter list");
        // 'fn T(Params)' names a function pointer/reference, not a bare
        // function type — wrap it in a &static reference so it matches how
        // Sema types a bare function name used as a value (also &static),
        // letting 'op = square;' and similar assignments type-check.
        return makeReference(makeFunction(std::move(ret), std::move(params), variadic),
                              Region::Static, /*nullable=*/false, /*mut=*/false);
    }
    case TK::KW_typeof: {
        // typeof(expr) — type position; Sema resolves to concrete type
        consume(); // eat 'typeof'
        expect(TK::LParen, "expected '(' after 'typeof'");
        auto e = parseExpr();
        Expr *rawExpr = e.release(); // ownership transferred to TypeofType
        expect(TK::RParen, "expected ')' after typeof expression");
        return std::make_shared<TypeofType>(static_cast<void *>(rawExpr));
    }
    case TK::Ident: {
        std::string name = cur().text;
        consume();
        // Named user type (struct/typedef)
        auto st = std::make_shared<StructType>(name, false);
        return st;
    }
    default:
        if (hasSign) return makeInt(32, !isUnsigned); // 'unsigned' alone → unsigned int
        diag_.error(loc, std::string("expected type, got '") + cur().text + "'");
        return makeError();
    }
}

// Parse a complete type including reference, pointer, array declarators
TypePtr Parser::parseType() {
    // 'const' written before the reference sigil, e.g. 'const &stack T'
    // (equivalent to '&stack const T').
    if (cur().is(TK::KW_const) && (peek().is(TK::Amp) || peek().is(TK::QuestionAmp))) {
        consume();
        bool nullable = cur().is(TK::QuestionAmp);
        consume();
        return parseReferenceType(nullable, /*leadingConst=*/true);
    }
    // ?& (nullable reference)
    if (cur().is(TK::QuestionAmp)) {
        consume();
        return parseReferenceType(true);
    }
    // & (non-null reference)
    if (cur().is(TK::Amp)) {
        consume();
        return parseReferenceType(false);
    }
    // ?T (optional type) — '?' not followed by '&'
    if (cur().is(TK::Question)) {
        consume();
        TypePtr inner = parseBaseType();
        inner = parseTypeDeclarator(std::move(inner));
        return makeOptional(std::move(inner));
    }
    // []T (slice type, parse-only)
    if (cur().is(TK::LBracket) && peek().is(TK::RBracket)) {
        consume(); consume(); // consume '[' ']'
        TypePtr elem = parseType();
        return makeSlice(std::move(elem));
    }
    // Base type
    TypePtr base = parseBaseType();

    // Pointer / array suffixes
    return parseTypeDeclarator(std::move(base));
}

// Parses a small constant integer expression: literals combined with the
// usual arithmetic operators and parentheses. This covers array sizes that
// come from preprocessor macros like '#define N (A * B)', which expand to
// a token sequence rather than a single literal.
int64_t Parser::parseArraySizeConst() {
    // Precedence-climbing over +/- (prec 1) and * / % (prec 2).
    std::function<int64_t(int)> parseLevel = [&](int minPrec) -> int64_t {
        int64_t lhs;
        if (cur().is(TK::Minus)) { consume(); lhs = -parseLevel(3); }
        else if (cur().is(TK::LParen)) {
            consume();
            lhs = parseLevel(1);
            expect(TK::RParen, "expected ')' in array size expression");
        } else if (cur().is(TK::IntLit)) {
            lhs = cur().intVal;
            consume();
        } else {
            diag_.error(cur().loc, "expected constant expression for array size");
            lhs = -1;
        }

        for (;;) {
            int prec = 0;
            if (cur().is(TK::Plus) || cur().is(TK::Minus)) prec = 1;
            else if (cur().is(TK::Star) || cur().is(TK::Slash) || cur().is(TK::Percent)) prec = 2;
            if (prec == 0 || prec < minPrec) break;

            TK op = cur().kind;
            consume();
            int64_t rhs = parseLevel(prec + 1);
            switch (op) {
            case TK::Plus:    lhs = lhs + rhs; break;
            case TK::Minus:   lhs = lhs - rhs; break;
            case TK::Star:    lhs = lhs * rhs; break;
            case TK::Slash:   lhs = rhs != 0 ? lhs / rhs : 0; break;
            case TK::Percent: lhs = rhs != 0 ? lhs % rhs : 0; break;
            default: break;
            }
        }
        return lhs;
    };
    return parseLevel(1);
}

int64_t Parser::parseArraySize(Expr **outSizeExpr) {
    *outSizeExpr = nullptr;

    // Scan ahead to the matching ']' (tracking nested brackets) to decide
    // whether this is pure literal arithmetic (the fast, common path) or
    // contains an identifier/call that needs deferred const-eval.
    bool literalOnly = true;
    {
        int depth = 0;
        for (size_t i = pos_; ; ++i) {
            const Token &t = toks_[i];
            if (t.is(TK::LBracket)) { depth++; continue; }
            if (t.is(TK::RBracket)) { if (depth == 0) break; depth--; continue; }
            if (t.is(TK::Eof)) { literalOnly = false; break; }
            if (!(t.is(TK::IntLit) || t.is(TK::Plus) || t.is(TK::Minus) ||
                  t.is(TK::Star) || t.is(TK::Slash) || t.is(TK::Percent) ||
                  t.is(TK::LParen) || t.is(TK::RParen))) {
                literalOnly = false;
            }
        }
    }
    if (literalOnly) return parseArraySizeConst();

    // General path: named constant or consteval function call, e.g.
    // 'int arr[square(3)]'. Resolved later by resolveArraySizes() once
    // function bodies are available for the const-eval interpreter.
    ExprPtr e = parseTernaryExpr();
    *outSizeExpr = e.get();
    pendingArraySizeExprs_.push_back(std::move(e));
    return -1;
}

TypePtr Parser::parseArrayDeclaratorSuffix(TypePtr base) {
    std::vector<std::pair<int64_t, Expr *>> dims;
    while (cur().is(TK::LBracket)) {
        consume();
        int64_t sz = -1;
        Expr *sizeExpr = nullptr;
        if (!cur().is(TK::RBracket)) sz = parseArraySize(&sizeExpr);
        expect(TK::RBracket, "expected ']'");
        dims.emplace_back(sz, sizeExpr);
    }
    // Build from the last-parsed dimension inward so the first bracket ends
    // up outermost (see header comment for why naive left-to-right wrapping
    // gets this backwards).
    for (auto it = dims.rbegin(); it != dims.rend(); ++it)
        base = makeArray(std::move(base), it->first, it->second);
    return base;
}

TypePtr Parser::parseTypeDeclarator(TypePtr base) {
    // C-style pointer: T* or T* const / T* restrict
    while (cur().is(TK::Star)) {
        consume();
        bool isConst = false;
        if (cur().is(TK::KW_const))    { consume(); isConst = true; }
        if (cur().is(TK::KW_restrict)) { consume(); } // ignored in SafeC
        if (cur().is(TK::KW_volatile)) { consume(); } // ignored in SafeC (see parseBaseType)
        base = makePointer(std::move(base), isConst);
    }
    // Array: T[N]
    base = parseArrayDeclaratorSuffix(std::move(base));
    return base;
}

// Parse <T: Constraint, U, ...>
std::vector<GenericParam> Parser::parseGenericParams() {
    std::vector<GenericParam> params;
    expect(TK::Lt, "expected '<' after 'generic'");
    while (!atEnd() && !cur().is(TK::Gt)) {
        GenericParam p;
        p.name = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected generic parameter name");
        } else consume();
        // T... — variadic type pack
        if (cur().is(TK::DotDotDot)) {
            consume();
            p.isPack = true;
        }
        if (cur().is(TK::Colon)) {
            consume();
            p.constraint = cur().text;
            if (!cur().is(TK::Ident)) {
                diag_.error(cur().loc, "expected constraint name");
            } else consume();
        }
        params.push_back(std::move(p));
        if (!match(TK::Comma)) break;
    }
    expect(TK::Gt, "expected '>' closing generic parameter list");
    return params;
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP-LEVEL DECLARATIONS
// ─────────────────────────────────────────────────────────────────────────────

bool Parser::tryParseCStyleFuncPtrDeclarator(TypePtr retType, std::string &outName,
                                              TypePtr &outType) {
    if (!cur().is(TK::LParen)) return false;
    size_t savedPos  = pos_;
    size_t diagMark  = diag_.checkpoint();

    consume(); // '('
    if (!cur().is(TK::Star)) { pos_ = savedPos; diag_.discardSince(diagMark); return false; }
    consume(); // '*'
    // The declarator name: accept any identifier-shaped token, not just
    // TK::Ident — SafeC keywords ('fn', 'stack', 'match', ...) are common,
    // unremarkable C parameter/variable names ('int (*fn)(int)' is the
    // single most natural name for a callback parameter), and unlike
    // expression contexts there's no other grammar production this could
    // be confused with: '(*' only ever starts this declarator pattern.
    bool nameIsIdentShaped = !cur().text.empty() &&
        (std::isalpha((unsigned char)cur().text[0]) || cur().text[0] == '_');
    if (!nameIsIdentShaped) { pos_ = savedPos; diag_.discardSince(diagMark); return false; }
    std::string name = cur().text;
    consume();
    if (!cur().is(TK::RParen)) { pos_ = savedPos; diag_.discardSince(diagMark); return false; }
    consume(); // ')'
    if (!cur().is(TK::LParen)) { pos_ = savedPos; diag_.discardSince(diagMark); return false; }
    consume(); // '(' — start of the function's own parameter list

    std::vector<TypePtr> params;
    bool variadic = false;
    if (cur().is(TK::KW_void) && peek().is(TK::RParen)) {
        consume();
    } else if (!cur().is(TK::RParen)) {
        do {
            if (cur().is(TK::DotDotDot)) { consume(); variadic = true; break; }
            params.push_back(parseType());
            if (cur().is(TK::Ident)) consume(); // optional param name, ignored in type position
        } while (match(TK::Comma));
    }
    expect(TK::RParen, "expected ')' closing function-pointer parameter list");

    outName = std::move(name);
    outType = makeReference(makeFunction(std::move(retType), std::move(params), variadic),
                             Region::Static, /*nullable=*/false, /*mut=*/false);
    return true;
}

bool Parser::parseGnuAttributes(GnuAttrs &out) {
    bool any = false;
    while (cur().isIdent("__attribute__") || cur().isIdent("__attribute")) {
        any = true;
        consume();
        expect(TK::LParen, "expected '(' after __attribute__");
        expect(TK::LParen, "expected '((' after __attribute__");

        while (!atEnd() && !cur().is(TK::RParen)) {
            // Attribute name: any identifier-shaped token, including ones
            // that collide with existing SafeC/C keywords (e.g. 'const',
            // 'packed', 'pure', 'noreturn' are all valid GCC attribute
            // names too, and still keyword tokens here, not TK::Ident).
            if (cur().text.empty() ||
                !(std::isalpha((unsigned char)cur().text[0]) || cur().text[0] == '_')) {
                break; // defensive: not attribute-name shaped, stop rather than loop forever
            }
            std::string attrName = cur().text;
            consume();
            // Normalize GCC's '__name__' spelling to plain 'name'
            if (attrName.size() > 4 && attrName.substr(0, 2) == "__" &&
                attrName.substr(attrName.size() - 2) == "__") {
                attrName = attrName.substr(2, attrName.size() - 4);
            }

            std::vector<std::string> args;
            if (cur().is(TK::LParen)) {
                consume();
                int depth = 1;
                std::string curArg;
                while (!atEnd() && depth > 0) {
                    if (cur().is(TK::LParen)) { depth++; curArg += "("; consume(); continue; }
                    if (cur().is(TK::RParen)) {
                        depth--;
                        consume();
                        if (depth == 0) break;
                        curArg += ")";
                        continue;
                    }
                    if (cur().is(TK::Comma) && depth == 1) {
                        args.push_back(curArg); curArg.clear(); consume(); continue;
                    }
                    if (cur().is(TK::StringLit)) curArg += "\"" + cur().text + "\"";
                    else if (cur().is(TK::IntLit)) curArg += std::to_string(cur().intVal);
                    else curArg += cur().text;
                    consume();
                }
                args.push_back(curArg);
            }

            // Map the common, semantically-meaningful attributes onto
            // SafeC's native equivalents; everything else (deprecated,
            // unused, used, visibility, format, nonnull, cold, hot, weak,
            // constructor, destructor, may_alias, vector_size, mode, ...)
            // is accepted syntactically and dropped — there's no SafeC
            // equivalent, and erroring on them would defeat the point of
            // tolerating this GCC/Clang extension at all.
            if (attrName == "always_inline" || attrName == "gnu_inline" ||
                attrName == "artificial") {
                out.hasInline = true;
            } else if (attrName == "noreturn") {
                out.hasNoReturn = true;
            } else if (attrName == "const" || attrName == "pure") {
                out.hasPure = true;
            } else if (attrName == "packed") {
                out.hasPacked = true;
            } else if (attrName == "warn_unused_result" || attrName == "nodiscard") {
                out.hasMustUse = true;
            } else if (attrName == "aligned") {
                if (!args.empty() && !args[0].empty() &&
                    std::isdigit((unsigned char)args[0][0])) {
                    out.alignment = std::atoi(args[0].c_str());
                } else if (out.alignment == 0) {
                    out.alignment = 16; // bare '__attribute__((aligned))': GCC picks the
                                         // target's max useful alignment; 16 is a reasonable default
                }
            } else if (attrName == "section") {
                if (!args.empty()) {
                    std::string s = args[0];
                    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
                        s = s.substr(1, s.size() - 2);
                    out.sectionName = s;
                }
            }

            if (cur().is(TK::Comma)) { consume(); continue; }
            break;
        }
        expect(TK::RParen, "expected ')' to close __attribute__ list");
        expect(TK::RParen, "expected '))' to close __attribute__");
    }
    return any;
}

std::unique_ptr<TranslationUnit> Parser::parseTranslationUnit() {
    auto tu = std::make_unique<TranslationUnit>();
    while (!atEnd()) {
        auto decl = parseTopLevelDecl();
        if (decl) tu->decls.push_back(std::move(decl));
        // Flush right here rather than once at EOF: collectDecls()/checkDecl()
        // process tu->decls in a single forward pass (Sema.cpp), so a
        // 'namespace std { typedef ...; }' block's contents must land at the
        // point in decl order where the block actually appeared, not after
        // every other top-level decl in the file — otherwise a plain
        // top-level function using a std-namespaced typedef that textually
        // precedes it would see the typedef still unregistered, since the
        // typedef's TypeAlias decl wouldn't be collected until later.
        if (!pendingNamespaceDecls_.empty()) {
            for (auto &nd : pendingNamespaceDecls_) tu->decls.push_back(std::move(nd));
            pendingNamespaceDecls_.clear();
        }
    }
    // Anonymous inline struct/union StructDecls must be visible to Sema
    // before the fields that reference them (so resolveType() finds the
    // synthesized name already registered) — prepend rather than append.
    tu->decls.insert(tu->decls.begin(),
        std::make_move_iterator(pendingAnonStructs_.begin()),
        std::make_move_iterator(pendingAnonStructs_.end()));
    pendingAnonStructs_.clear();
    tu->arraySizeExprs = std::move(pendingArraySizeExprs_);
    return tu;
}

// namespace X { decl* } — see the pendingNamespaceDecls_ comment in Parser.h
// for why this doesn't return through the normal DeclPtr path. Nested
// namespaces ('namespace a { namespace b { ... } }') fall out naturally:
// the recursive parseTopLevelDecl() call for 'namespace b' pushes its own
// (already fully "a::b"-qualified) decls into pendingNamespaceDecls_ and
// returns nullptr, so the outer loop here just skips it.
void Parser::parseNamespaceDecl() {
    consume(); // 'namespace'
    if (!cur().is(TK::Ident)) {
        diag_.error(cur().loc, "expected namespace name after 'namespace'");
        syncToDecl();
        return;
    }
    std::string name = cur().text;
    consume();
    expect(TK::LBrace, "expected '{' after namespace name");

    namespaceStack_.push_back(name);
    std::string qualified = namespaceStack_[0];
    for (size_t i = 1; i < namespaceStack_.size(); ++i) qualified += "::" + namespaceStack_[i];

    while (!atEnd() && !cur().is(TK::RBrace)) {
        auto decl = parseTopLevelDecl();
        if (!decl) continue;
        if (decl->kind == DeclKind::Function) {
            static_cast<FunctionDecl &>(*decl).namespaceName = qualified;
        } else if (decl->kind == DeclKind::GlobalVar) {
            static_cast<GlobalVarDecl &>(*decl).namespaceName = qualified;
        }
        pendingNamespaceDecls_.push_back(std::move(decl));
    }
    expect(TK::RBrace, "expected '}' to close namespace");
    match(TK::Semicolon); // tolerate an optional trailing ';'
    namespaceStack_.pop_back();
}

DeclPtr Parser::parseTopLevelDecl() {
    auto loc = cur().loc;

    // GNU/Clang '__attribute__((...))' in prefix position (before the rest
    // of the declaration) — see GnuAttrs / parseGnuAttributes for why this
    // is tolerated rather than rejected. Merged into the existing qualifier
    // locals below once they're in scope; also re-checked inside the
    // qualifier loop so it can appear interspersed with const/static/etc.
    GnuAttrs gnuAttrs;
    parseGnuAttributes(gnuAttrs);

    // must_use keyword prefix (C-style: `must_use int fn(...)`)
    bool isMustUse = gnuAttrs.hasMustUse;
    if (cur().is(TK::KW_must_use)) { isMustUse = true; consume(); }

    // packed struct
    bool isPacked = gnuAttrs.hasPacked;
    if (cur().is(TK::KW_packed)) { consume(); isPacked = true; }

    // struct / union
    if (cur().is(TK::KW_struct) || cur().is(TK::KW_union)) {
        bool isUnion = cur().is(TK::KW_union);
        consume();
        // 'struct __attribute__((packed)) Foo { ... }' — tag-position attrs
        {
            GnuAttrs tagAttrs;
            if (parseGnuAttributes(tagAttrs)) isPacked = isPacked || tagAttrs.hasPacked;
        }
        // Is this a standalone struct decl or a function returning struct?
        if (peek().is(TK::LBrace) || (cur().is(TK::Ident) && peek().is(TK::LBrace))) {
            auto sd = parseStructDecl(isUnion);
            if (isPacked && sd && sd->kind == DeclKind::Struct)
                static_cast<StructDecl&>(*sd).isPacked = true;
            // Trailing '} __attribute__((packed));' — suffix-position attrs
            GnuAttrs trailingAttrs;
            if (parseGnuAttributes(trailingAttrs) && trailingAttrs.hasPacked &&
                sd && sd->kind == DeclKind::Struct) {
                static_cast<StructDecl&>(*sd).isPacked = true;
            }
            return sd;
        }
        // Otherwise it's a function/var decl starting with struct T
        // Back up one and let the general path handle it
        --pos_;
    }

    // enum (with optional underlying type: enum Foo : uint8 { ... })
    if (cur().is(TK::KW_enum)) {
        consume();
        if (cur().is(TK::Ident) &&
            (peek().is(TK::LBrace) || peek().is(TK::Colon))) {
            return parseEnumDecl();
        }
        --pos_;
    }

    // newtype Name = BaseType;
    if (cur().is(TK::KW_newtype)) {
        consume();
        auto loc = cur().loc;
        std::string name = cur().text;
        expect(TK::Ident, "expected newtype name");
        expect(TK::Eq, "expected '=' in newtype declaration");
        TypePtr base = parseType();
        expect(TK::Semicolon);
        return std::make_unique<NewtypeDecl>(std::move(name), std::move(base), loc);
    }

    // namespace X { ... }
    if (cur().is(TK::KW_namespace)) { parseNamespaceDecl(); return nullptr; }

    // region
    if (cur().is(TK::KW_region)) { consume(); return parseRegionDecl(); }

    // typedef
    if (cur().is(TK::KW_typedef)) return parseTypedef();

    // static_assert at top level
    if (cur().is(TK::KW_static_assert)) return parseStaticAssertDecl();

    // generic<T: C> ...
    std::vector<GenericParam> genericParams;
    if (cur().is(TK::KW_generic)) {
        consume();
        genericParams = parseGenericParams();
    }

    // Qualifiers
    bool isConst    = false;
    bool isConsteval = false;
    bool isInline   = gnuAttrs.hasInline;
    bool isExtern   = false;
    bool isStatic   = false;
    // Bare-metal / effect system attributes
    bool isNaked     = false;
    bool isInterrupt = false;
    bool isNoReturn  = gnuAttrs.hasNoReturn;
    bool isPure      = gnuAttrs.hasPure;
    bool isVolatile    = false;
    bool isAtomic      = false;
    bool isThreadLocal = false;
    bool isConstinit   = false;
    std::string sectionName = gnuAttrs.sectionName;
    std::string callingConv;
    int alignment = gnuAttrs.alignment;

    while (true) {
        if (cur().isIdent("__attribute__") || cur().isIdent("__attribute")) {
            GnuAttrs more;
            parseGnuAttributes(more);
            isInline    = isInline    || more.hasInline;
            isNoReturn  = isNoReturn  || more.hasNoReturn;
            isPure      = isPure      || more.hasPure;
            isMustUse   = isMustUse   || more.hasMustUse;
            isPacked    = isPacked    || more.hasPacked;
            if (more.alignment > 0) alignment = more.alignment;
            if (!more.sectionName.empty()) sectionName = more.sectionName;
            continue;
        }
        if (cur().isIdent("align") && peek().is(TK::LParen)) {
            consume(); // consume 'align'
            expect(TK::LParen, "expected '(' after align");
            if (cur().is(TK::IntLit)) { alignment = (int)cur().intVal; consume(); }
            else diag_.error(cur().loc, "expected integer alignment value");
            expect(TK::RParen, "expected ')' after alignment value");
            continue;
        }
        if (match(TK::KW_const))     { isConst     = true; continue; }
        if (match(TK::KW_consteval)) { isConsteval = true; continue; }
        if (match(TK::KW_constinit)) { isConstinit  = true; continue; }
        if (match(TK::KW_inline))    { isInline    = true; continue; }
        if (match(TK::KW_extern))    { isExtern    = true; continue; }
        if (match(TK::KW_static))    { isStatic    = true; continue; }
        if (match(TK::KW_naked))     { isNaked     = true; continue; }
        if (match(TK::KW_interrupt)) { isInterrupt = true; continue; }
        if (match(TK::KW_noreturn))  { isNoReturn  = true; continue; }
        if (match(TK::KW_pure))      { isPure      = true; continue; }
        if (match(TK::KW_volatile))  { isVolatile  = true; continue; }
        if (match(TK::KW_atomic))    { isAtomic    = true; continue; }
        if (match(TK::KW_thread_local)) { isThreadLocal = true; continue; }
        if (match(TK::KW_stdcall))   { callingConv = "stdcall";  continue; }
        if (match(TK::KW_cdecl))     { callingConv = "cdecl";    continue; }
        if (match(TK::KW_fastcall))  { callingConv = "fastcall"; continue; }
        if (cur().is(TK::KW_section)) {
            consume(); // consume 'section'
            expect(TK::LParen, "expected '(' after section");
            if (cur().is(TK::StringLit)) {
                sectionName = cur().text;
                consume();
            } else {
                diag_.error(cur().loc, "expected string literal for section name");
            }
            expect(TK::RParen, "expected ')' after section name");
            continue;
        }
        break;
    }

    if (!isTypeStart()) {
        diag_.error(cur().loc, std::string("unexpected token '") + cur().text + "' at top level");
        consume();
        return nullptr;
    }

    auto decl = parseFunctionOrGlobalVar(isConst, isConsteval, isInline, isExtern, isStatic,
                                         std::move(genericParams));
    if (isMustUse && decl && decl->kind == DeclKind::Function)
        static_cast<FunctionDecl&>(*decl).isMustUse = true;
    // Propagate bare-metal attrs to FunctionDecl
    if (decl && decl->kind == DeclKind::Function) {
        auto &fn = static_cast<FunctionDecl&>(*decl);
        fn.isNaked     = isNaked;
        fn.isInterrupt = isInterrupt;
        // OR rather than overwrite: parseFunctionDecl may already have set
        // these from a suffix '__attribute__((...))' after the parameter
        // list (e.g. 'void f(void) __attribute__((noreturn));') — a plain
        // assignment here would silently clobber that back to false/empty.
        fn.isNoReturn  = fn.isNoReturn  || isNoReturn;
        fn.isPure      = fn.isPure      || isPure;
        fn.sectionName = fn.sectionName.empty() ? sectionName : fn.sectionName;
        fn.callingConv = callingConv;
    }
    // Propagate volatile/atomic/thread_local/section to GlobalVarDecl
    if (decl && decl->kind == DeclKind::GlobalVar) {
        auto &gv = static_cast<GlobalVarDecl&>(*decl);
        gv.isVolatile    = isVolatile;
        gv.isAtomic      = isAtomic;
        gv.isThreadLocal = isThreadLocal;
        // OR/fallback rather than overwrite — a suffix '__attribute__((...))'
        // after the declarator may already have set these (see the
        // 'Global variable' branch above).
        gv.sectionName   = gv.sectionName.empty() ? sectionName : gv.sectionName;
        gv.alignment     = gv.alignment > 0 ? gv.alignment : alignment;
        gv.isConstinit   = isConstinit;
    }
    return decl;
}

DeclPtr Parser::parseFunctionOrGlobalVar(bool isConst, bool isConsteval,
                                          bool isInline, bool isExtern, bool isStatic,
                                          std::vector<GenericParam> genericParams) {
    auto loc = cur().loc;
    TypePtr retType = parseType();
    if (!retType) return nullptr;

    // C-style function-pointer variable: 'RetType (*name)(Params);' (e.g.
    // 'void (*on_tick)(int);') — a variable of function-pointer-reference
    // type, not a function declaration, so it's handled entirely here
    // rather than falling into the method/'::'/param-list logic below.
    {
        std::string fpName;
        TypePtr fpType;
        if (tryParseCStyleFuncPtrDeclarator(retType, fpName, fpType)) {
            auto gv = std::make_unique<GlobalVarDecl>(std::move(fpName), loc);
            gv->type     = std::move(fpType);
            gv->isConst  = isConst;
            gv->isStatic = isStatic;
            gv->isExtern = isExtern;
            GnuAttrs suffixAttrs;
            if (parseGnuAttributes(suffixAttrs)) {
                if (suffixAttrs.alignment > 0) gv->alignment = suffixAttrs.alignment;
                if (!suffixAttrs.sectionName.empty()) gv->sectionName = suffixAttrs.sectionName;
            }
            if (match(TK::Eq)) gv->init = parseAssignExpr();
            expect(TK::Semicolon);
            return gv;
        }
    }

    // Expect an identifier for name
    if (!cur().is(TK::Ident)) {
        diag_.error(cur().loc, "expected identifier after type");
        syncToDecl();
        return nullptr;
    }
    std::string name = cur().text;
    consume();

    // Method definition: StructName :: methodName '(' ...
    std::string methodOwner;
    if (cur().is(TK::ColonColon)) {
        consume(); // consume '::'
        methodOwner = name;  // struct name
        if (!cur().is(TK::Ident) && !cur().is(TK::KW_operator)) {
            diag_.error(cur().loc, "expected method name after '::'");
            syncToDecl();
            return nullptr;
        }
        if (cur().is(TK::KW_operator)) {
            // operator overload: Vec2::operator+(Vec2 other)
            consume(); // consume 'operator'
            name = "operator";
            // Collect operator symbol(s)
            if (!cur().is(TK::LParen)) {
                name += cur().text; consume();
                // Two-char operators: ==, !=, <=, >=
                if ((name == "operator=" || name == "operator!" ||
                     name == "operator<" || name == "operator>") &&
                    cur().is(TK::Eq)) {
                    name += cur().text; consume();
                }
            }
        } else {
            name = cur().text;
            consume();
        }
    }

    // Function: name '(' ...
    if (cur().is(TK::LParen)) {
        consume(); // consume '('
        auto fn = parseFunctionDecl(std::move(retType), std::move(name), loc,
                                    isConst, isConsteval, isInline, isExtern,
                                    false, std::move(genericParams));
        if (!methodOwner.empty()) {
            fn->isMethod    = true;
            fn->methodOwner = std::move(methodOwner);
        }
        return fn;
    }

    // Global variable
    // C-style array dimension after name: T name[N]; — same declarator
    // position as local variables and struct fields.
    retType = parseArrayDeclaratorSuffix(std::move(retType));

    auto gv = std::make_unique<GlobalVarDecl>(std::move(name), loc);
    gv->type      = std::move(retType);
    gv->isConst   = isConst;
    gv->isStatic  = isStatic;
    gv->isExtern  = isExtern;
    // GNU/Clang suffix attributes: 'int x __attribute__((aligned(16)));'
    {
        GnuAttrs suffixAttrs;
        if (parseGnuAttributes(suffixAttrs)) {
            if (suffixAttrs.alignment > 0) gv->alignment = suffixAttrs.alignment;
            if (!suffixAttrs.sectionName.empty()) gv->sectionName = suffixAttrs.sectionName;
        }
    }
    if (match(TK::Eq)) {
        gv->init = parseAssignExpr();
    }
    expect(TK::Semicolon);
    return gv;
}

std::unique_ptr<FunctionDecl> Parser::parseFunctionDecl(
    TypePtr retType, std::string name, SourceLocation loc,
    bool isConst, bool isConsteval, bool isInline, bool isExtern,
    bool /*isVariadic_unused*/, std::vector<GenericParam> genericParams)
{
    auto fn = std::make_unique<FunctionDecl>(std::move(name), loc);
    fn->returnType   = std::move(retType);
    fn->isConst      = isConst;
    fn->isConsteval  = isConsteval;
    fn->isInline     = isInline;
    fn->isExtern     = isExtern;
    fn->genericParams = std::move(genericParams);

    // Parse parameter list (we're past the '(')
    // C-style 'f(void)' means zero parameters, not one void-typed parameter
    // (there's no such thing as a value of type void) — without this check
    // the loop below would parse 'void' as a legitimate param type, and
    // every call site would then mismatch on argument count.
    if (cur().is(TK::KW_void) && peek().is(TK::RParen)) consume();
    while (!atEnd() && !cur().is(TK::RParen)) {
        if (cur().is(TK::DotDotDot)) { consume(); fn->isVariadic = true; break; }
        ParamDecl p;
        p.loc  = cur().loc;
        p.type = parseType();
        // C-style function-pointer parameter: 'void f(int (*cb)(int))'
        if (tryParseCStyleFuncPtrDeclarator(p.type, p.name, p.type)) {
            fn->params.push_back(std::move(p));
            if (!match(TK::Comma)) break;
            continue;
        }
        // T... args — variadic pack parameter
        if (cur().is(TK::DotDotDot)) {
            consume();
            p.isPack = true;
        }
        // Optional parameter name
        if (cur().is(TK::Ident)) {
            p.name = cur().text;
            consume();
        }
        // C-style array parameter suffix: 'T name[N]' (e.g. 'const char*
        // comp_ptrs[16]'). Array parameters decay to pointers at the call
        // boundary just like C, but the declared element/size still needs
        // parsing here or the '[' desyncs the rest of the parameter list.
        p.type = parseArrayDeclaratorSuffix(std::move(p.type));
        fn->params.push_back(std::move(p));
        if (!match(TK::Comma)) break;
    }
    expect(TK::RParen, "expected ')' closing parameter list");

    // GNU/Clang suffix attributes: 'void f(void) __attribute__((noreturn));'
    {
        GnuAttrs suffixAttrs;
        if (parseGnuAttributes(suffixAttrs)) {
            if (suffixAttrs.hasNoReturn) fn->isNoReturn = true;
            if (suffixAttrs.hasPure)     fn->isPure     = true;
            if (suffixAttrs.hasMustUse)  fn->isMustUse  = true;
            if (!suffixAttrs.sectionName.empty()) fn->sectionName = suffixAttrs.sectionName;
        }
    }

    // Optional 'const' qualifier after params (for const methods: T::m() const { ... })
    if (cur().is(TK::KW_const)) {
        consume();
        fn->isConstMethod = true;
    }

    // Body or declaration
    if (cur().is(TK::LBrace)) {
        fn->body = parseCompoundStmt();
    } else {
        expect(TK::Semicolon, "expected ';' or '{' after function declaration");
    }
    return fn;
}

std::unique_ptr<StructDecl> Parser::parseStructDecl(bool isUnion, bool consumeTrailingSemicolon) {
    auto loc = cur().loc;
    std::string name;
    if (cur().is(TK::Ident)) { name = cur().text; consume(); }

    auto sd = std::make_unique<StructDecl>(std::move(name), loc);
    sd->isUnion = isUnion;

    expect(TK::LBrace, "expected '{' in struct definition");
    int fieldIdx = 0;
    while (!atEnd() && !cur().is(TK::RBrace)) {
        // Peek ahead: if we have "type name (" it's a method declaration,
        // otherwise it's a field declaration.
        // Heuristic: after parsing type + name, check for '('
        auto savedPos = pos_;

        // Try to parse as method declaration
        TypePtr maybeType = parseType();
        bool isMethodDecl = false;
        std::string memberName;

        // C-style function-pointer field: 'RetType (*name)(Params);'
        // (e.g. a vtable/callback-table entry: 'void (*on_tick)(int);').
        {
            TypePtr fpType = maybeType;
            std::string fpName;
            if (tryParseCStyleFuncPtrDeclarator(fpType, fpName, fpType)) {
                FieldDecl fd;
                fd.index = fieldIdx++;
                fd.type  = std::move(fpType);
                fd.name  = std::move(fpName);
                expect(TK::Semicolon, "expected ';' after struct field");
                sd->fields.push_back(std::move(fd));
                if (pos_ == savedPos) consume();
                continue;
            }
        }

        if (cur().is(TK::Ident)) {
            memberName = cur().text;
            // Peek ahead to see if '(' follows (method) or not (field)
            if (peek().is(TK::LParen)) {
                isMethodDecl = true;
            }
        } else if (cur().is(TK::KW_operator)) {
            // operator overload declaration: operator+, operator-, etc.
            isMethodDecl = true;
            consume(); // consume 'operator'
            // Collect the operator symbol (could be +, -, *, /, ==, !=, <, >, <=, >=)
            memberName = "operator";
            if (!cur().is(TK::LParen)) {
                memberName += cur().text;
                consume(); // consume operator symbol
                // Handle two-char operators like ==, !=, <=, >=
                if ((memberName == "operator=" && cur().is(TK::Eq)) ||
                    (memberName == "operator!" && cur().is(TK::Eq)) ||
                    (memberName == "operator<" && cur().is(TK::Eq)) ||
                    (memberName == "operator>" && cur().is(TK::Eq))) {
                    memberName += cur().text; consume();
                }
            }
            // Now fall through to isMethodDecl block — skip the "consume method name" step
            // We already consumed the full operator name
            {
                MethodDecl md;
                md.loc        = cur().loc;
                md.name       = memberName;
                md.returnType = std::move(maybeType);
                expect(TK::LParen, "expected '(' after operator name");
                if (cur().is(TK::KW_void) && peek().is(TK::RParen)) consume();
                while (!atEnd() && !cur().is(TK::RParen)) {
                    if (cur().is(TK::DotDotDot)) { consume(); break; }
                    ParamDecl p;
                    p.loc  = cur().loc;
                    p.type = parseType();
                    if (cur().is(TK::Ident)) { p.name = cur().text; consume(); }
                    p.type = parseArrayDeclaratorSuffix(std::move(p.type));
                    md.params.push_back(std::move(p));
                    if (!match(TK::Comma)) break;
                }
                expect(TK::RParen, "expected ')' closing operator parameter list");
                if (cur().is(TK::KW_const)) { consume(); md.isConst = true; }
                expect(TK::Semicolon, "expected ';' after operator declaration");
                sd->methodDecls.push_back(std::move(md));
            }
            continue; // skip the isMethodDecl block below
        }

        if (isMethodDecl) {
            // Method declaration inside struct: T method(params) [const];
            consume(); // consume method name
            consume(); // consume '('
            MethodDecl md;
            md.loc        = cur().loc;
            md.name       = memberName;
            md.returnType = std::move(maybeType);
            if (cur().is(TK::KW_void) && peek().is(TK::RParen)) consume();
            while (!atEnd() && !cur().is(TK::RParen)) {
                if (cur().is(TK::DotDotDot)) { consume(); break; }
                ParamDecl p;
                p.loc  = cur().loc;
                p.type = parseType();
                if (cur().is(TK::Ident)) { p.name = cur().text; consume(); }
                p.type = parseArrayDeclaratorSuffix(std::move(p.type));
                md.params.push_back(std::move(p));
                if (!match(TK::Comma)) break;
            }
            expect(TK::RParen, "expected ')' closing method parameter list");
            if (cur().is(TK::KW_const)) { consume(); md.isConst = true; }
            expect(TK::Semicolon, "expected ';' after method declaration");
            sd->methodDecls.push_back(std::move(md));
        } else {
            // Field declaration
            FieldDecl fd;
            fd.index = fieldIdx++;
            fd.type  = std::move(maybeType);
            fd.name  = memberName;
            // Anonymous struct/union member: 'struct S { union { int a; }; };'
            // — no name follows the (struct/union-typed) field at all, just
            // the terminating ';'. Its own members ('s.a') become reachable
            // on the enclosing struct via StructType::findFieldPath.
            if (memberName.empty() && cur().is(TK::Semicolon) &&
                fd.type && fd.type->kind == TypeKind::Struct) {
                fd.isAnonymous = true;
                fd.name = static_cast<StructType &>(*fd.type).name; // internal only
            } else if (!cur().is(TK::Ident)) {
                diag_.error(cur().loc, "expected field name");
            } else consume();
            // Bitfield: 'unsigned x : 4;' — mutually exclusive with an array
            // declarator in C grammar, so only checked when no '[' follows.
            if (!fd.isAnonymous && cur().is(TK::Colon)) {
                consume();
                fd.bitWidth = static_cast<int>(parseArraySizeConst());
                if (fd.bitWidth < 0) {
                    diag_.error(cur().loc, "bitfield width must be a non-negative constant");
                    fd.bitWidth = 0;
                }
            } else {
                // Optional array size(s)
                fd.type = parseArrayDeclaratorSuffix(std::move(fd.type));
            }
            expect(TK::Semicolon, "expected ';' after struct field");
            sd->fields.push_back(std::move(fd));
        }

        // Guarantee forward progress: if a malformed member produced only
        // errors without consuming any token, force one token forward so
        // parsing always terminates instead of looping on the same token.
        if (pos_ == savedPos) consume();
    }
    expect(TK::RBrace, "expected '}' closing struct");
    if (consumeTrailingSemicolon) match(TK::Semicolon);
    return sd;
}

std::unique_ptr<EnumDecl> Parser::parseEnumDecl() {
    auto loc = cur().loc;
    std::string name;
    if (cur().is(TK::Ident)) { name = cur().text; consume(); }

    auto ed = std::make_unique<EnumDecl>(std::move(name), loc);
    // Parse optional underlying type: enum Foo : uint8 { ... }
    if (match(TK::Colon)) {
        ed->underlyingType = parseType();
    }
    expect(TK::LBrace, "expected '{' in enum definition");
    int64_t nextVal = 0;
    while (!atEnd() && !cur().is(TK::RBrace)) {
        std::string eName = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected enumerator name");
            consume(); continue;
        }
        consume();
        std::optional<int64_t> val;
        if (match(TK::Eq)) {
            if (cur().is(TK::IntLit)) {
                val = cur().intVal; nextVal = cur().intVal + 1; consume();
            } else {
                diag_.error(cur().loc, "expected integer value in enum");
                consume();
            }
        } else {
            val = nextVal++;
        }
        ed->enumerators.push_back({std::move(eName), val});
        if (!match(TK::Comma)) break;
    }
    expect(TK::RBrace, "expected '}' closing enum");
    match(TK::Semicolon);
    return ed;
}

std::unique_ptr<RegionDecl> Parser::parseRegionDecl() {
    auto loc = cur().loc;
    std::string name = cur().text;
    if (!cur().is(TK::Ident)) {
        diag_.error(cur().loc, "expected region name");
    } else consume();

    auto rd = std::make_unique<RegionDecl>(std::move(name), loc);
    expect(TK::LBrace, "expected '{' in region declaration");
    while (!atEnd() && !cur().is(TK::RBrace)) {
        // capacity: N
        if (cur().is(TK::KW_capacity) || cur().isIdent("capacity")) {
            consume();
            expect(TK::Colon, "expected ':' after 'capacity'");
            if (cur().is(TK::IntLit)) {
                rd->capacity = cur().intVal; consume();
            } else {
                diag_.error(cur().loc, "expected integer capacity");
            }
        } else {
            diag_.warn(cur().loc, "unknown region field, skipping");
            consume();
        }
        match(TK::Comma);
    }
    expect(TK::RBrace, "expected '}' closing region");
    match(TK::Semicolon);
    return rd;
}

DeclPtr Parser::parseTypedef() {
    auto loc = cur().loc;
    consume(); // eat 'typedef'
    TypePtr t = parseType();
    // 'typedef RetType (*Name)(Params);' — C-style function-pointer typedef
    // (e.g. 'typedef void (*Callback)(int);'), the classic form real C code
    // uses instead of SafeC's native 'typedef fn RetType(Params) Name;'.
    std::string name;
    if (tryParseCStyleFuncPtrDeclarator(t, name, t)) {
        // t reassigned to the &static fn(...) reference type; name filled.
    } else {
        name = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected typedef name");
        } else consume();
    }
    expect(TK::Semicolon);
    return std::make_unique<TypeAliasDecl>(std::move(name), std::move(t), loc);
}

DeclPtr Parser::parseStaticAssertDecl() {
    auto loc = cur().loc;
    consume(); // eat 'static_assert'
    expect(TK::LParen, "expected '(' after static_assert");
    auto cond = parseAssignExpr(); // use parseAssignExpr to avoid consuming the ',' message separator
    std::string msg;
    if (match(TK::Comma)) {
        if (cur().is(TK::StringLit)) { msg = cur().text; consume(); }
    }
    expect(TK::RParen, "expected ')' closing static_assert");
    expect(TK::Semicolon);
    return std::make_unique<StaticAssertDecl>(std::move(cond), std::move(msg), loc);
}

// ─────────────────────────────────────────────────────────────────────────────
// STATEMENTS
// ─────────────────────────────────────────────────────────────────────────────

StmtPtr Parser::parseStmt() {
    auto loc = cur().loc;

    // Label: ident ':'  statement
    // Special case: if the labeled stmt is for/while, embed the label in the loop
    if (cur().is(TK::Ident) && peek().is(TK::Colon)) {
        std::string lbl = cur().text; consume(); consume();
        if (cur().is(TK::KW_for)) {
            auto fs = parseForStmt();
            static_cast<ForStmt&>(*fs).label = lbl;
            return fs;
        }
        if (cur().is(TK::KW_while)) {
            auto ws = parseWhileStmt();
            static_cast<WhileStmt&>(*ws).label = lbl;
            return ws;
        }
        return std::make_unique<LabelStmt>(std::move(lbl), parseStmt(), loc);
    }

    switch (cur().kind) {
    case TK::LBrace:           return parseCompoundStmt();
    case TK::KW_if:            return parseIfStmt();
    case TK::KW_while:         return parseWhileStmt();
    case TK::KW_do:            return parseDoWhileStmt();
    case TK::KW_for:           return parseForStmt();
    case TK::KW_return:        return parseReturnStmt();
    case TK::KW_defer:         return parseDeferStmt(false);
    case TK::KW_errdefer:      return parseDeferStmt(true);
    case TK::KW_match:         return parseMatchStmt();
    case TK::KW_break: {
        consume();
        std::string lbl;
        if (cur().is(TK::Ident)) { lbl = cur().text; consume(); }
        expect(TK::Semicolon);
        return std::make_unique<BreakStmt>(loc, std::move(lbl));
    }
    case TK::KW_continue: {
        consume();
        std::string lbl;
        if (cur().is(TK::Ident)) { lbl = cur().text; consume(); }
        expect(TK::Semicolon);
        return std::make_unique<ContinueStmt>(loc, std::move(lbl));
    }
    case TK::KW_goto: {
        consume();
        std::string lbl = cur().text;
        if (!cur().is(TK::Ident)) diag_.error(cur().loc, "expected label name");
        else consume();
        expect(TK::Semicolon);
        return std::make_unique<GotoStmt>(std::move(lbl), loc);
    }
    case TK::KW_unsafe:        return parseUnsafeStmt();
    case TK::KW_static_assert: return parseStaticAssertStmt();
    case TK::KW_asm:           return parseAsmStmt();
    case TK::KW_const: {
        consume();
        // 'const T x' binds to the BASE type: for a plain type that makes
        // the variable itself const/immutable ('const int x' — x can't be
        // reassigned), but for a pointer type it makes only the POINTEE
        // const ('const char* p' is a *mutable pointer* to const data,
        // exactly like C — 'p = other;' is legal and extremely common,
        // e.g. reassigning a string-literal pointer). Making the pointer
        // itself const additionally requires a trailing 'T* const p'.
        // Peek at the parsed type (then backtrack) to tell these apart —
        // a single leading 'const' can't be resolved without seeing
        // whether a '*' declarator follows.
        size_t savedPos = pos_;
        size_t diagMark = diag_.checkpoint();
        TypePtr peeked = parseType();
        pos_ = savedPos;
        diag_.discardSince(diagMark);
        bool variableIsConst = !(peeked && peeked->kind == TypeKind::Pointer);
        return parseVarDeclStmt(variableIsConst, false);
    }
    case TK::KW_volatile: {
        consume();
        auto vs = parseVarDeclStmt(false, false);
        if (vs && vs->kind == StmtKind::VarDecl)
            static_cast<VarDeclStmt&>(*vs).isVolatile = true;
        return vs;
    }
    case TK::KW_atomic: {
        consume();
        auto vs = parseVarDeclStmt(false, false);
        if (vs && vs->kind == StmtKind::VarDecl)
            static_cast<VarDeclStmt&>(*vs).isAtomic = true;
        return vs;
    }
    case TK::KW_thread_local: {
        consume();
        // thread_local at local scope must combine with static
        bool isStaticTL = match(TK::KW_static);
        bool isConst = false;
        if (match(TK::KW_const)) isConst = true;
        auto vs = parseVarDeclStmt(isConst, isStaticTL);
        if (vs && vs->kind == StmtKind::VarDecl)
            static_cast<VarDeclStmt&>(*vs).isThreadLocal = true;
        return vs;
    }
    case TK::KW_static: {
        consume();
        bool isConst = false;
        if (match(TK::KW_const)) isConst = true;
        return parseVarDeclStmt(isConst, true);
    }
    default:
        // align(N) Type name = init; — contextual keyword
        if (cur().isIdent("align") && peek().is(TK::LParen)) {
            consume();
            expect(TK::LParen, "expected '(' after align");
            int alignment = 0;
            if (cur().is(TK::IntLit)) { alignment = (int)cur().intVal; consume(); }
            else diag_.error(cur().loc, "expected integer alignment value");
            expect(TK::RParen, "expected ')' after alignment value");
            auto vs = parseVarDeclStmt(false, false);
            if (vs && vs->kind == StmtKind::VarDecl)
                static_cast<VarDeclStmt&>(*vs).alignment = alignment;
            return vs;
        }
        // []T varName — slice type variable declaration
        if (cur().is(TK::LBracket) && peek().is(TK::RBracket)) {
            return parseVarDeclStmt(false, false);
        }
        // Disambiguation: type-start → variable declaration
        if (isTypeStart() && !cur().is(TK::Star)) {
            // Could be a function call or a var decl.
            // Look ahead: if after the type-name we see an identifier, it's a decl.
            // Simple heuristic: if cur is a keyword type → definitely decl
            bool isKeywordType = (
                cur().is(TK::KW_int) || cur().is(TK::KW_void) ||
                cur().is(TK::KW_char) || cur().is(TK::KW_float) ||
                cur().is(TK::KW_double) || cur().is(TK::KW_bool) ||
                cur().is(TK::KW_long) || cur().is(TK::KW_short) ||
                cur().is(TK::KW_unsigned) || cur().is(TK::KW_signed) ||
                cur().is(TK::KW_struct) || cur().is(TK::KW_enum) ||
                cur().is(TK::KW_tuple) || cur().is(TK::KW_fn) ||
                cur().is(TK::KW_typeof) ||
                cur().is(TK::Amp) || cur().is(TK::QuestionAmp) ||
                cur().is(TK::Question)   // ?T optional type
            );
            if (isKeywordType) return parseVarDeclStmt(false, false);
            // vec<T, N> varName — same contextual pattern as parseBaseType's
            // vec<...> case (see there for why 'vec' isn't a keyword).
            if (cur().isIdent("vec") && peek().is(TK::Lt)) {
                return parseVarDeclStmt(false, false);
            }
            // Ident followed by another ident → var decl (TypeName varName)
            if (cur().is(TK::Ident) && peek().is(TK::Ident)) {
                return parseVarDeclStmt(false, false);
            }
            // Ident followed by * then ident → pointer var decl (TypeName* varName)
            // Disambiguate: TypeName * varName vs expr * expr
            // Heuristic: if token[2] is an ident (name), it's a declaration
            if (cur().is(TK::Ident) && peek().is(TK::Star) && peek(2).is(TK::Ident)) {
                return parseVarDeclStmt(false, false);
            }
        }
        return parseExprStmt();
    }
}

std::unique_ptr<CompoundStmt> Parser::parseCompoundStmt() {
    auto loc = cur().loc;
    expect(TK::LBrace, "expected '{'");
    auto cs = std::make_unique<CompoundStmt>(loc);
    while (!atEnd() && !cur().is(TK::RBrace)) {
        auto s = parseStmt();
        if (s) cs->body.push_back(std::move(s));
    }
    expect(TK::RBrace, "expected '}'");
    return cs;
}

StmtPtr Parser::parseIfStmt() {
    auto loc = cur().loc;
    consume(); // 'if'

    // 'if const (cond) { ... }' — compile-time conditional (CONST_EVAL.md §14)
    if (cur().isIdent("const") || cur().is(TK::KW_const)) {
        consume(); // 'const'
        expect(TK::LParen, "expected '(' after 'if const'");
        auto cond = parseExpr();
        expect(TK::RParen, "expected ')' after if const condition");
        auto then = parseStmt();
        StmtPtr else_;
        if (match(TK::KW_else)) else_ = parseStmt();
        return std::make_unique<IfConstStmt>(std::move(cond), std::move(then),
                                             std::move(else_), loc);
    }

    expect(TK::LParen, "expected '(' after 'if'");
    auto cond = parseExpr();
    expect(TK::RParen, "expected ')' after if condition");
    auto then = parseStmt();
    StmtPtr else_;
    if (match(TK::KW_else)) else_ = parseStmt();
    return std::make_unique<IfStmt>(std::move(cond), std::move(then),
                                    std::move(else_), loc);
}

StmtPtr Parser::parseWhileStmt() {
    auto loc = cur().loc;
    consume(); // 'while'
    expect(TK::LParen);
    auto cond = parseExpr();
    expect(TK::RParen);
    auto body = parseStmt();
    return std::make_unique<WhileStmt>(std::move(cond), std::move(body), false, loc);
}

StmtPtr Parser::parseDoWhileStmt() {
    auto loc = cur().loc;
    consume(); // 'do'
    auto body = parseStmt();
    expect(TK::KW_while, "expected 'while' after do-body");
    expect(TK::LParen);
    auto cond = parseExpr();
    expect(TK::RParen);
    expect(TK::Semicolon);
    return std::make_unique<WhileStmt>(std::move(cond), std::move(body), true, loc);
}

StmtPtr Parser::parseForStmt() {
    auto loc = cur().loc;
    consume(); // 'for'
    expect(TK::LParen);

    // init
    StmtPtr init;
    if (!cur().is(TK::Semicolon)) {
        if (isTypeStart()) init = parseVarDeclStmt(false, false);
        else init = parseExprStmt();
    } else {
        expect(TK::Semicolon);
    }

    // condition
    ExprPtr cond;
    if (!cur().is(TK::Semicolon)) cond = parseExpr();
    expect(TK::Semicolon);

    // increment
    ExprPtr incr;
    if (!cur().is(TK::RParen)) incr = parseExpr();
    expect(TK::RParen);

    auto body = parseStmt();
    return std::make_unique<ForStmt>(std::move(init), std::move(cond),
                                     std::move(incr), std::move(body), loc);
}

StmtPtr Parser::parseReturnStmt() {
    auto loc = cur().loc;
    consume(); // 'return'
    ExprPtr val;
    if (!cur().is(TK::Semicolon)) val = parseExpr();
    expect(TK::Semicolon);
    return std::make_unique<ReturnStmt>(std::move(val), loc);
}

StmtPtr Parser::parseUnsafeStmt() {
    auto loc = cur().loc;
    consume(); // 'unsafe'
    auto body = parseCompoundStmt();
    return std::make_unique<UnsafeStmt>(std::move(body), loc);
}

StmtPtr Parser::parseStaticAssertStmt() {
    auto loc = cur().loc;
    consume(); // 'static_assert'
    expect(TK::LParen);
    auto cond = parseAssignExpr(); // avoid consuming ',' message separator
    std::string msg;
    if (match(TK::Comma)) {
        if (cur().is(TK::StringLit)) { msg = cur().text; consume(); }
    }
    expect(TK::RParen);
    expect(TK::Semicolon);
    return std::make_unique<StaticAssertStmt>(std::move(cond), std::move(msg), loc);
}

StmtPtr Parser::parseVarDeclStmt(bool isConst, bool isStatic) {
    auto loc = cur().loc;
    TypePtr ty = parseType();
    // C-style function-pointer local: 'RetType (*name)(Params);'
    std::string name;
    if (!tryParseCStyleFuncPtrDeclarator(ty, name, ty)) {
        name = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected variable name");
        } else consume();
    }

    // C-style array dimension after name: int arr[N]
    ty = parseArrayDeclaratorSuffix(std::move(ty));

    ExprPtr init;
    if (match(TK::Eq)) init = parseAssignExpr();

    expect(TK::Semicolon, "expected ';' after variable declaration");

    auto vs = std::make_unique<VarDeclStmt>(std::move(name), std::move(ty),
                                             std::move(init), loc);
    vs->isConst  = isConst;
    vs->isStatic = isStatic;
    return vs;
}

// ── Inline assembly statement ──────────────────────────────────────────────
// asm [volatile] ( "template" [: outputs [: inputs [: clobbers]]] ) ;
// Note: ::: (empty outputs+inputs+clobbers) lexes as ColonColon + Colon
StmtPtr Parser::parseAsmStmt() {
    auto loc = cur().loc;
    consume(); // consume 'asm'
    auto stmt = std::make_unique<AsmStmt>(loc);
    if (match(TK::KW_volatile)) stmt->isVolatile = true;
    expect(TK::LParen, "expected '(' after asm");
    // Template string (adjacent string literals concatenate, same as any
    // other string-literal position — asm templates are routinely split
    // across lines, one instruction per literal, e.g. "a\n\t" "b\n\t").
    if (cur().is(TK::StringLit)) {
        stmt->asmTemplate = cur().text;
        consume();
        while (cur().is(TK::StringLit)) { stmt->asmTemplate += cur().text; consume(); }
    } else {
        diag_.error(cur().loc, "expected string literal for asm template");
    }

    // Helper: consume one colon separator (handles :: → two colons)
    // Returns how many colons were consumed (0, 1, or 2)
    int pendingColons = 0;
    auto consumeColon = [&]() -> bool {
        if (pendingColons > 0) { --pendingColons; return true; }
        if (match(TK::ColonColon)) { pendingColons = 1; return true; }
        return match(TK::Colon);
    };

    // Optional sections: outputs : inputs : clobbers
    if (consumeColon()) {
        // outputs: "constraint"(expr), ...
        while (!pendingColons && cur().is(TK::StringLit)) {
            std::string constraint = cur().text; consume();
            expect(TK::LParen);
            stmt->outputs.push_back(std::move(constraint));
            stmt->outputExprs.push_back(parseAssignExpr());
            expect(TK::RParen);
            if (!match(TK::Comma)) break;
        }
        if (consumeColon()) {
            // inputs: "constraint"(expr), ...
            while (!pendingColons && cur().is(TK::StringLit)) {
                std::string constraint = cur().text; consume();
                expect(TK::LParen);
                stmt->inputs.push_back(std::move(constraint));
                stmt->inputExprs.push_back(parseAssignExpr());
                expect(TK::RParen);
                if (!match(TK::Comma)) break;
            }
            if (consumeColon()) {
                // clobbers: "reg", "reg", ...
                while (cur().is(TK::StringLit)) {
                    stmt->clobbers.push_back(cur().text);
                    consume();
                    if (!match(TK::Comma)) break;
                }
            }
        }
    }
    expect(TK::RParen, "expected ')' after asm");
    expect(TK::Semicolon, "expected ';' after asm statement");
    return stmt;
}

StmtPtr Parser::parseExprStmt() {
    auto loc = cur().loc;
    if (cur().is(TK::Semicolon)) { consume(); return std::make_unique<ExprStmt>(nullptr, loc); }
    auto e = parseExpr();
    expect(TK::Semicolon, "expected ';' after expression");
    return std::make_unique<ExprStmt>(std::move(e), loc);
}

StmtPtr Parser::parseDeferStmt(bool isErrDefer) {
    auto loc = cur().loc;
    consume(); // eat 'defer' or 'errdefer'
    auto body = parseStmt();
    return std::make_unique<DeferStmt>(std::move(body), loc, isErrDefer);
}

// Parse a single match pattern (used inside `case pat, pat: body`)
// Handles: integer literal, char literal, N..M range, enum ident
MatchPattern Parser::parseMatchPattern() {
    MatchPattern p;
    // .variant(bind) — tagged union pattern
    if (cur().is(TK::Dot) && peek().is(TK::Ident)) {
        consume(); // eat '.'
        p.kind  = PatternKind::EnumIdent;
        p.ident = cur().text;
        consume();
        if (match(TK::LParen)) {
            if (cur().is(TK::Ident)) { p.bindName = cur().text; consume(); }
            expect(TK::RParen);
        }
        return p;
    }
    // Integer literal or N..M range
    if (cur().is(TK::IntLit)) {
        p.kind   = PatternKind::IntLit;
        p.intVal = cur().intVal;
        consume();
        // Range: N..M — DotDot token or two consecutive dots
        if (cur().is(TK::DotDot) || (cur().is(TK::Dot) && peek().is(TK::Dot))) {
            if (cur().is(TK::DotDot)) consume(); else { consume(); consume(); }
            if (cur().is(TK::IntLit)) {
                p.kind    = PatternKind::Range;
                p.intVal2 = cur().intVal;
                consume();
            } else {
                diag_.error(cur().loc, "expected integer after '..' in range pattern");
            }
        }
        return p;
    }
    // Char literal pattern: 'a'
    if (cur().is(TK::CharLit)) {
        p.kind   = PatternKind::IntLit;
        p.intVal = (unsigned char)cur().text[0];
        consume();
        return p;
    }
    // Enum ident
    if (cur().is(TK::Ident)) {
        p.kind  = PatternKind::EnumIdent;
        p.ident = cur().text;
        consume();
        if (match(TK::LParen)) {
            if (cur().is(TK::Ident)) { p.bindName = cur().text; consume(); }
            expect(TK::RParen);
        }
        return p;
    }
    diag_.error(cur().loc, "expected match pattern (integer, char, range, or identifier)");
    consume();
    return p;
}

// C-style match statement:
//   match (expr) {
//       case pat, pat:  stmt
//       case N..M:      stmt
//       default:        stmt
//   }
StmtPtr Parser::parseMatchStmt() {
    auto loc = cur().loc;
    consume(); // eat 'match'
    expect(TK::LParen, "expected '(' after 'match'");
    auto subject = parseExpr();
    expect(TK::RParen, "expected ')' after match subject");
    expect(TK::LBrace, "expected '{' opening match body");

    std::vector<MatchArm> arms;
    while (!atEnd() && !cur().is(TK::RBrace)) {
        MatchArm arm;
        if (cur().is(TK::KW_default)) {
            // default: body  — wildcard arm
            consume();
            expect(TK::Colon, "expected ':' after 'default'");
            MatchPattern p;
            p.kind = PatternKind::Wildcard;
            arm.patterns.push_back(std::move(p));
        } else {
            // case pattern, pattern, ...: body
            expect(TK::KW_case, "expected 'case' or 'default' in match arm");
            do {
                arm.patterns.push_back(parseMatchPattern());
            } while (match(TK::Comma));
            expect(TK::Colon, "expected ':' after match pattern(s)");
        }
        arm.body = parseStmt();
        arms.push_back(std::move(arm));
    }
    expect(TK::RBrace, "expected '}' closing match");
    return std::make_unique<MatchStmt>(std::move(subject), std::move(arms), loc);
}

// ─────────────────────────────────────────────────────────────────────────────
// EXPRESSIONS  (standard C precedence, recursive descent)
// ─────────────────────────────────────────────────────────────────────────────

ExprPtr Parser::parseExpr() {
    auto lhs = parseAssignExpr();
    while (cur().is(TK::Comma)) {
        auto loc = cur().loc;
        consume();
        auto rhs = parseAssignExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::Comma,
                                            std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

AssignOp Parser::tokenToAssignOp(TK k) const {
    switch (k) {
    case TK::Eq:       return AssignOp::Assign;
    case TK::PlusEq:   return AssignOp::AddAssign;
    case TK::MinusEq:  return AssignOp::SubAssign;
    case TK::StarEq:   return AssignOp::MulAssign;
    case TK::SlashEq:  return AssignOp::DivAssign;
    case TK::PercentEq:return AssignOp::ModAssign;
    case TK::AmpEq:    return AssignOp::AndAssign;
    case TK::PipeEq:   return AssignOp::OrAssign;
    case TK::CaretEq:  return AssignOp::XorAssign;
    case TK::LShiftEq: return AssignOp::ShlAssign;
    case TK::RShiftEq: return AssignOp::ShrAssign;
    default:           return AssignOp::Assign;
    }
}

ExprPtr Parser::parseAssignExpr() {
    auto lhs = parseTernaryExpr();
    // Check for assignment operators
    switch (cur().kind) {
    case TK::Eq: case TK::PlusEq: case TK::MinusEq: case TK::StarEq:
    case TK::SlashEq: case TK::PercentEq: case TK::AmpEq:
    case TK::PipeEq: case TK::CaretEq: case TK::LShiftEq: case TK::RShiftEq: {
        auto loc = cur().loc;
        AssignOp op = tokenToAssignOp(cur().kind);
        consume();
        auto rhs = parseAssignExpr(); // right-associative
        return std::make_unique<AssignExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    default: return lhs;
    }
}

ExprPtr Parser::parseTernaryExpr() {
    auto cond = parseLogOrExpr();
    if (!cur().is(TK::Question)) return cond;
    auto loc = cur().loc;
    consume();
    auto then = parseExpr();
    expect(TK::Colon, "expected ':' in ternary expression");
    auto else_ = parseTernaryExpr();
    return std::make_unique<TernaryExpr>(std::move(cond), std::move(then),
                                          std::move(else_), loc);
}

ExprPtr Parser::parseLogOrExpr() {
    auto lhs = parseLogAndExpr();
    while (cur().is(TK::PipePipe)) {
        auto loc = cur().loc; consume();
        auto rhs = parseLogAndExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::LogOr, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseLogAndExpr() {
    auto lhs = parseBitOrExpr();
    while (cur().is(TK::AmpAmp)) {
        auto loc = cur().loc; consume();
        auto rhs = parseBitOrExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::LogAnd, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseBitOrExpr() {
    auto lhs = parseBitXorExpr();
    while (cur().is(TK::Pipe)) {
        auto loc = cur().loc; consume();
        auto rhs = parseBitXorExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::BitOr, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseBitXorExpr() {
    auto lhs = parseBitAndExpr();
    while (cur().is(TK::Caret)) {
        auto loc = cur().loc; consume();
        auto rhs = parseBitAndExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::BitXor, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseBitAndExpr() {
    auto lhs = parseEqExpr();
    while (cur().is(TK::Amp)) {
        auto loc = cur().loc; consume();
        auto rhs = parseEqExpr();
        lhs = std::make_unique<BinaryExpr>(BinaryOp::BitAnd, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseEqExpr() {
    auto lhs = parseRelExpr();
    while (cur().is(TK::EqEq) || cur().is(TK::BangEq)) {
        auto loc = cur().loc;
        BinaryOp op = cur().is(TK::EqEq) ? BinaryOp::Eq : BinaryOp::NEq;
        consume();
        auto rhs = parseRelExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseRelExpr() {
    auto lhs = parseShiftExpr();
    for (;;) {
        BinaryOp op;
        switch (cur().kind) {
        case TK::Lt:  op = BinaryOp::Lt;  break;
        case TK::Gt:  op = BinaryOp::Gt;  break;
        case TK::LtEq:op = BinaryOp::LEq; break;
        case TK::GtEq:op = BinaryOp::GEq; break;
        default: return lhs;
        }
        auto loc = cur().loc; consume();
        auto rhs = parseShiftExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
}

ExprPtr Parser::parseShiftExpr() {
    auto lhs = parseAddExpr();
    while (cur().is(TK::LShift) || cur().is(TK::RShift)) {
        auto loc = cur().loc;
        BinaryOp op = cur().is(TK::LShift) ? BinaryOp::Shl : BinaryOp::Shr;
        consume();
        auto rhs = parseAddExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseAddExpr() {
    auto lhs = parseMulExpr();
    while (cur().is(TK::Plus) || cur().is(TK::Minus) ||
           cur().is(TK::PlusPipe) || cur().is(TK::MinusPipe) ||
           cur().is(TK::PlusPercent) || cur().is(TK::MinusPercent)) {
        auto loc = cur().loc;
        BinaryOp op;
        if      (cur().is(TK::Plus))         op = BinaryOp::Add;
        else if (cur().is(TK::Minus))        op = BinaryOp::Sub;
        else if (cur().is(TK::PlusPipe))     op = BinaryOp::WrapAdd;
        else if (cur().is(TK::MinusPipe))    op = BinaryOp::WrapSub;
        else if (cur().is(TK::PlusPercent))  op = BinaryOp::SatAdd;
        else                                  op = BinaryOp::SatSub;
        consume();
        auto rhs = parseMulExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseMulExpr() {
    auto lhs = parseUnaryExpr();
    while (cur().is(TK::Star) || cur().is(TK::Slash) || cur().is(TK::Percent) ||
           cur().is(TK::StarPipe) || cur().is(TK::StarPercent)) {
        auto loc = cur().loc;
        BinaryOp op;
        if      (cur().is(TK::Star))        op = BinaryOp::Mul;
        else if (cur().is(TK::Slash))       op = BinaryOp::Div;
        else if (cur().is(TK::Percent))     op = BinaryOp::Mod;
        else if (cur().is(TK::StarPipe))    op = BinaryOp::WrapMul;
        else                                 op = BinaryOp::SatMul;
        consume();
        auto rhs = parseUnaryExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseUnaryExpr() {
    auto loc = cur().loc;
    switch (cur().kind) {
    case TK::Minus: {
        consume();
        auto e = parseCastExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::Neg, std::move(e), loc);
    }
    case TK::Bang: {
        consume();
        auto e = parseCastExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::Not, std::move(e), loc);
    }
    case TK::Tilde: {
        consume();
        auto e = parseCastExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::BitNot, std::move(e), loc);
    }
    case TK::PlusPlus: {
        consume();
        auto e = parseUnaryExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::PreInc, std::move(e), loc);
    }
    case TK::MinusMinus: {
        consume();
        auto e = parseUnaryExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::PreDec, std::move(e), loc);
    }
    case TK::Amp: {
        // In safe context, & of lvalue → &stack reference
        consume();
        auto e = parseCastExpr();
        // Represent as a unary AddrOf; sema will type it as &stack T
        auto ae = std::make_unique<UnaryExpr>(UnaryOp::Neg, std::move(e), loc);
        ae->kind = ExprKind::AddrOf;  // override
        return ae;
    }
    case TK::Star: {
        // Dereference — only valid in unsafe or for heap refs
        consume();
        auto e = parseCastExpr();
        auto de = std::make_unique<UnaryExpr>(UnaryOp::Neg, std::move(e), loc);
        de->kind = ExprKind::Deref;  // override
        return de;
    }
    case TK::KW_sizeof: {
        consume();
        // sizeof...(T) — variadic pack count
        if (cur().is(TK::DotDotDot)) {
            consume(); // eat '...'
            expect(TK::LParen, "expected '(' after 'sizeof...'");
            std::string packName = cur().text;
            consume();
            expect(TK::RParen, "expected ')' after sizeof... pack name");
            return std::make_unique<SizeofPackExpr>(packName, loc);
        }
        if (cur().is(TK::LParen) && isTypeStart(peek())) {
            consume(); // '('
            auto t = parseType();
            expect(TK::RParen, "expected ')' after sizeof type");
            return std::make_unique<SizeofTypeExpr>(std::move(t), loc);
        }
        auto e = parseUnaryExpr();
        return std::make_unique<UnaryExpr>(UnaryOp::SizeofExpr, std::move(e), loc);
    }
    case TK::KW_alignof: {
        consume();
        expect(TK::LParen, "expected '(' after 'alignof'");
        auto t = parseType();
        expect(TK::RParen, "expected ')' after alignof type");
        return std::make_unique<AlignofTypeExpr>(std::move(t), loc);
    }
    case TK::KW_fieldcount: {
        consume();
        expect(TK::LParen, "expected '(' after 'fieldcount'");
        auto t = parseType();
        expect(TK::RParen, "expected ')' after fieldcount type");
        return std::make_unique<FieldCountExpr>(std::move(t), loc);
    }
    case TK::KW_c11_generic: {
        // _Generic(controlling-expr, T1: e1, T2: e2, ..., default: eN)
        consume();
        expect(TK::LParen, "expected '(' after '_Generic'");
        auto controlling = parseAssignExpr();
        expect(TK::Comma, "expected ',' after _Generic controlling expression");
        std::vector<GenericAssoc> assocs;
        do {
            GenericAssoc ga;
            if (cur().is(TK::KW_default)) {
                consume();
            } else {
                ga.type = parseType();
            }
            expect(TK::Colon, "expected ':' in _Generic association");
            ga.expr = parseAssignExpr();
            assocs.push_back(std::move(ga));
        } while (match(TK::Comma) && !cur().is(TK::RParen));
        expect(TK::RParen, "expected ')' closing '_Generic'");
        return std::make_unique<GenericSelectionExpr>(std::move(controlling),
                                                        std::move(assocs), loc);
    }
    case TK::KW_try: {
        // try expr — unwrap optional, propagate null on empty
        consume();
        auto inner = parseUnaryExpr();
        return std::make_unique<TryExpr>(std::move(inner), loc);
    }
    default:
        return parseCastExpr();
    }
}

ExprPtr Parser::parseCastExpr() {
    // Try to parse (type) expr cast
    // Heuristic: '(' followed by a type keyword → cast
    if (cur().is(TK::LParen) && isTypeStart(peek())) {
        size_t saved = pos_;
        size_t diagMark = diag_.checkpoint();
        bool wasSilent = diag_.isSilent();
        diag_.setSilent(true); // speculative — don't surface errors unless we commit
        consume(); // '('
        // Try to parse type
        TypePtr t = parseType();
        // Committing needs both a matching ')' AND a token after it that
        // could plausibly start the cast's operand — otherwise a bare
        // identifier type-guess like '(x)' in '(x) = 5' or '(x).field' would
        // "succeed" as a cast up through the ')' and only fail later trying
        // to parse '=' or '.' as an operand, instead of being recognized as
        // not a cast at all.
        bool ok = cur().is(TK::RParen) && canStartCastOperand(peek());
        diag_.setSilent(wasSilent);
        if (ok) {
            consume(); // ')'
            auto e = parseUnaryExpr();
            return std::make_unique<CastExpr>(std::move(t), std::move(e), toks_[saved].loc);
        }
        // Not a cast — restore both the token position and any diagnostics
        // emitted while speculatively parsing a type that didn't pan out.
        pos_ = saved;
        diag_.discardSince(diagMark);
    }
    return parsePostfixExpr();
}

ExprPtr Parser::parsePostfixExpr() {
    auto e = parsePrimaryExpr();
    for (;;) {
        auto loc = cur().loc;
        if (cur().is(TK::LBracket)) {
            consume();
            // Check for slice expr: arr[start..end], arr[..end], arr[start..], arr[..]
            if (cur().is(TK::DotDot)) {
                // arr[..end] or arr[..]
                consume();
                ExprPtr endE;
                if (!cur().is(TK::RBracket))
                    endE = parseExpr();
                expect(TK::RBracket, "expected ']'");
                e = std::make_unique<SliceExpr>(std::move(e), nullptr, std::move(endE), loc);
            } else {
                auto idx = parseExpr();
                if (cur().is(TK::DotDot)) {
                    // arr[start..end] or arr[start..]
                    consume();
                    ExprPtr endE;
                    if (!cur().is(TK::RBracket))
                        endE = parseExpr();
                    expect(TK::RBracket, "expected ']'");
                    e = std::make_unique<SliceExpr>(std::move(e), std::move(idx), std::move(endE), loc);
                } else {
                    expect(TK::RBracket, "expected ']'");
                    e = std::make_unique<SubscriptExpr>(std::move(e), std::move(idx), loc);
                }
            }
        } else if (cur().is(TK::LParen)) {
            consume();
            std::vector<ExprPtr> args;
            while (!atEnd() && !cur().is(TK::RParen)) {
                args.push_back(parseAssignExpr());
                if (!match(TK::Comma)) break;
            }
            expect(TK::RParen, "expected ')' closing call");
            e = std::make_unique<CallExpr>(std::move(e), std::move(args), loc);
        } else if (cur().is(TK::Dot)) {
            consume();
            std::string field;
            if (cur().is(TK::Ident)) {
                field = cur().text; consume();
            } else if (cur().is(TK::IntLit)) {
                // Tuple field access: t.0, t.1, ...
                field = std::to_string(cur().intVal); consume();
            } else {
                diag_.error(cur().loc, "expected field name");
                field = ""; consume();
            }
            e = std::make_unique<MemberExpr>(std::move(e), std::move(field), false, loc);
        } else if (cur().is(TK::Arrow)) {
            consume();
            std::string field = cur().text;
            if (!cur().is(TK::Ident)) diag_.error(cur().loc, "expected field name");
            else consume();
            e = std::make_unique<MemberExpr>(std::move(e), std::move(field), true, loc);
        } else if (cur().is(TK::PlusPlus)) {
            consume();
            e = std::make_unique<UnaryExpr>(UnaryOp::PostInc, std::move(e), loc);
        } else if (cur().is(TK::MinusMinus)) {
            consume();
            e = std::make_unique<UnaryExpr>(UnaryOp::PostDec, std::move(e), loc);
        } else {
            break;
        }
    }
    return e;
}

ExprPtr Parser::parsePrimaryExpr() {
    auto loc = cur().loc;
    switch (cur().kind) {
    case TK::IntLit: {
        int64_t v   = cur().intVal;
        bool isLL   = cur().isLongLong;
        bool isUns  = cur().isUnsigned;
        consume();
        auto expr = std::make_unique<IntLitExpr>(v, loc);
        expr->isLongLong = isLL;
        expr->isUnsigned = isUns;
        return expr;
    }
    case TK::FloatLit: {
        double v = cur().floatVal; consume();
        return std::make_unique<FloatLitExpr>(v, loc);
    }
    case TK::StringLit: {
        // Adjacent string-literal concatenation (C translation phase 6):
        // "a" "b" is one literal "ab" — used pervasively for wrapping long
        // format strings/messages across lines and by macro-generated
        // strings (e.g. token-pasting a prefix onto a literal).
        std::string s = cur().text; consume();
        while (cur().is(TK::StringLit)) { s += cur().text; consume(); }
        return std::make_unique<StringLitExpr>(std::move(s), loc);
    }
    case TK::CharLit: {
        char c = cur().text.empty() ? 0 : cur().text[0]; consume();
        return std::make_unique<CharLitExpr>(c, loc);
    }
    case TK::KW_true:  consume(); return std::make_unique<BoolLitExpr>(true, loc);
    case TK::KW_false: consume(); return std::make_unique<BoolLitExpr>(false, loc);
    case TK::KW_null:  consume(); return std::make_unique<NullLitExpr>(loc);
    case TK::Ident: {
        std::string name = cur().text; consume();
        // Namespace-qualified reference: std::foo, a::b::c, ... — joined
        // into one IdentExpr whose 'name' is the qualified string; Sema
        // resolves it against the namespace registry (see checkIdent).
        while (cur().is(TK::ColonColon) && peek().is(TK::Ident)) {
            consume(); // '::'
            name += "::";
            name += cur().text;
            consume();
        }
        return std::make_unique<IdentExpr>(std::move(name), loc);
    }
    case TK::KW_stack: case TK::KW_heap:
    case TK::KW_arena: case TK::KW_capacity:
    case TK::KW_self: {  // self is a valid expression in method bodies
        std::string name = cur().text; consume();
        return std::make_unique<IdentExpr>(std::move(name), loc);
    }
    case TK::LParen: {
        consume();
        // Detect tuple literal: (expr, expr, ...)
        auto e = parseAssignExpr();
        if (cur().is(TK::Comma)) {
            // Tuple literal
            std::vector<ExprPtr> elems;
            elems.push_back(std::move(e));
            while (match(TK::Comma)) {
                if (cur().is(TK::RParen)) break;
                elems.push_back(parseAssignExpr());
            }
            expect(TK::RParen, "expected ')' closing tuple literal");
            return std::make_unique<TupleLitExpr>(std::move(elems), loc);
        }
        expect(TK::RParen, "expected ')' closing parenthesized expression");
        return e;
    }
    case TK::LBrace: {
        // Compound initializer: { a, b, c }, with C99 designators:
        // '{.field = val, ...}' (struct) and '{[i] = val, ...}' (array).
        // GNU's older 'field: val' colon form is also accepted since it's
        // common in ported C code (e.g. some Linux kernel-style headers).
        consume();
        std::vector<ExprPtr> inits;
        std::vector<std::string> designatedFields;
        std::vector<int64_t> designatedIndices;
        while (!atEnd() && !cur().is(TK::RBrace)) {
            std::string fieldDesig;
            int64_t indexDesig = -1;
            if (cur().is(TK::Dot) && peek().is(TK::Ident)) {
                consume(); // '.'
                fieldDesig = cur().text;
                consume();
                if (cur().is(TK::Colon)) consume(); // tolerate GNU 'field: val'
                else expect(TK::Eq, "expected '=' after '.field' designator");
            } else if (cur().is(TK::LBracket)) {
                consume(); // '['
                indexDesig = parseArraySizeConst();
                expect(TK::RBracket, "expected ']' closing array designator");
                if (cur().is(TK::Colon)) consume();
                else expect(TK::Eq, "expected '=' after array designator");
            }
            designatedFields.push_back(fieldDesig);
            designatedIndices.push_back(indexDesig);
            inits.push_back(parseAssignExpr());
            if (!match(TK::Comma)) break;
        }
        expect(TK::RBrace, "expected '}' closing initializer");
        auto ci = std::make_unique<CompoundInitExpr>(std::move(inits), loc);
        ci->designatedFields  = std::move(designatedFields);
        ci->designatedIndices = std::move(designatedIndices);
        return ci;
    }
    case TK::KW_new: {
        // new<RegionName> Type
        consume();
        std::string regionName;
        if (match(TK::Lt)) {
            if (cur().is(TK::Ident)) { regionName = cur().text; consume(); }
            expect(TK::Gt, "expected '>' closing region parameter");
        }
        TypePtr allocType = parseBaseType();
        return std::make_unique<NewExpr>(std::move(regionName), std::move(allocType), loc);
    }
    case TK::KW_arena_reset: {
        // arena_reset<RegionName>()
        consume();
        std::string regionName;
        if (match(TK::Lt)) {
            if (cur().is(TK::Ident)) { regionName = cur().text; consume(); }
            expect(TK::Gt, "expected '>' closing region parameter");
        }
        expect(TK::LParen, "expected '(' for arena_reset");
        expect(TK::RParen, "expected ')' for arena_reset");
        // Represent as a CallExpr with a special callee name
        auto callee = std::make_unique<IdentExpr>("__arena_reset_" + regionName, loc);
        return std::make_unique<CallExpr>(std::move(callee), std::vector<ExprPtr>{}, loc);
    }
    case TK::KW_spawn: {
        // spawn(fn_expr, arg_expr)
        consume();
        expect(TK::LParen, "expected '(' after 'spawn'");
        auto fnExpr = parseAssignExpr();
        expect(TK::Comma, "expected ',' separating spawn arguments");
        auto argExpr = parseAssignExpr();
        expect(TK::RParen, "expected ')' closing spawn");
        return std::make_unique<SpawnExpr>(std::move(fnExpr), std::move(argExpr), loc);
    }
    case TK::KW_join: {
        // join(handle_expr)
        consume();
        expect(TK::LParen, "expected '(' after 'join'");
        auto handle = parseExpr();
        expect(TK::RParen, "expected ')' closing join");
        auto callee = std::make_unique<IdentExpr>("__safec_join", loc);
        std::vector<ExprPtr> args;
        args.push_back(std::move(handle));
        return std::make_unique<CallExpr>(std::move(callee), std::move(args), loc);
    }
    default:
        diag_.error(loc, std::string("unexpected token '") + cur().text + "' in expression");
        consume();
        return std::make_unique<IntLitExpr>(0, loc); // error recovery sentinel
    }
}

} // namespace safec
