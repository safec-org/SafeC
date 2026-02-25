#include "safec/Parser.h"
#include <cassert>
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
    case TK::KW_tuple:    // tuple type
    case TK::KW_fn:       // fn ReturnType(Params) — function pointer type
    case TK::KW_typeof:   // typeof(expr) — type position
    case TK::Amp:         // &stack T
    case TK::QuestionAmp: // ?&stack T
    case TK::Question:    // ?T (optional)
    case TK::Star:        // *T (pointer)
        return true;
    case TK::Ident:
        return true; // user-defined type name
    default: return false;
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
TypePtr Parser::parseReferenceType(bool nullable) {
    // We already consumed '&' or '?&'
    std::string arenaName;
    Region r = parseRegionQualifier(arenaName);

    // Parse optional const
    bool isConst = false;
    if (cur().is(TK::KW_const)) { consume(); isConst = true; }

    TypePtr base = parseBaseType();
    // Array suffix
    if (cur().is(TK::LBracket)) {
        consume();
        int64_t sz = -1;
        if (!cur().is(TK::RBracket)) {
            if (cur().is(TK::IntLit)) sz = cur().intVal;
            consume(); // or error
        }
        expect(TK::RBracket);
        base = makeArray(std::move(base), sz);
    }
    return makeReference(std::move(base), r, nullable, !isConst, std::move(arenaName));
}

// Parse a base type (no declarator suffixes yet)
TypePtr Parser::parseBaseType() {
    bool isConst    = false;
    bool isSigned   = true;
    bool hasSign    = false;
    bool isUnsigned = false;

    if (cur().is(TK::KW_const)) { consume(); isConst = true; }
    // 'restrict' and compiler-specific '__restrict' are ignored qualifiers in SafeC
    if (cur().is(TK::KW_restrict)) { consume(); }
    if (cur().is(TK::KW_unsigned)) { consume(); isUnsigned = true; hasSign = true; isSigned = false; }
    else if (cur().is(TK::KW_signed)) { consume(); hasSign = true; }

    auto loc = cur().loc;

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
    case TK::KW_struct: {
        consume();
        std::string name = cur().text;
        if (!cur().is(TK::Ident)) {
            diag_.error(cur().loc, "expected struct name");
        } else consume();
        // Create a named struct type (to be resolved by sema)
        auto st = std::make_shared<StructType>(name, false);
        return st;
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
        return makeFunction(std::move(ret), std::move(params), variadic);
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

TypePtr Parser::parseTypeDeclarator(TypePtr base) {
    // C-style pointer: T* or T* const / T* restrict
    while (cur().is(TK::Star)) {
        consume();
        bool isConst = false;
        if (cur().is(TK::KW_const))    { consume(); isConst = true; }
        if (cur().is(TK::KW_restrict)) { consume(); } // ignored in SafeC
        base = makePointer(std::move(base), isConst);
    }
    // Array: T[N]
    while (cur().is(TK::LBracket)) {
        consume();
        int64_t sz = -1;
        if (cur().is(TK::IntLit)) { sz = cur().intVal; consume(); }
        expect(TK::RBracket, "expected ']'");
        base = makeArray(std::move(base), sz);
    }
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

std::unique_ptr<TranslationUnit> Parser::parseTranslationUnit() {
    auto tu = std::make_unique<TranslationUnit>();
    while (!atEnd()) {
        auto decl = parseTopLevelDecl();
        if (decl) tu->decls.push_back(std::move(decl));
    }
    return tu;
}

DeclPtr Parser::parseTopLevelDecl() {
    auto loc = cur().loc;

    // must_use keyword prefix (C-style: `must_use int fn(...)`)
    bool isMustUse = false;
    if (cur().is(TK::KW_must_use)) { isMustUse = true; consume(); }

    // packed struct
    bool isPacked = false;
    if (cur().is(TK::KW_packed)) { consume(); isPacked = true; }

    // struct / union
    if (cur().is(TK::KW_struct) || cur().is(TK::KW_union)) {
        bool isUnion = cur().is(TK::KW_union);
        consume();
        // Is this a standalone struct decl or a function returning struct?
        if (peek().is(TK::LBrace) || (cur().is(TK::Ident) && peek().is(TK::LBrace))) {
            auto sd = parseStructDecl(isUnion);
            if (isPacked && sd && sd->kind == DeclKind::Struct)
                static_cast<StructDecl&>(*sd).isPacked = true;
            return sd;
        }
        // Otherwise it's a function/var decl starting with struct T
        // Back up one and let the general path handle it
        --pos_;
    }

    // enum
    if (cur().is(TK::KW_enum)) {
        consume();
        if (cur().is(TK::Ident) && peek().is(TK::LBrace)) {
            return parseEnumDecl();
        }
        --pos_;
    }

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
    bool isInline   = false;
    bool isExtern   = false;
    bool isStatic   = false;

    while (true) {
        if (match(TK::KW_const))    { isConst    = true; continue; }
        if (match(TK::KW_consteval)){ isConsteval= true; continue; }
        if (match(TK::KW_inline))   { isInline   = true; continue; }
        if (match(TK::KW_extern))   { isExtern   = true; continue; }
        if (match(TK::KW_static))   { isStatic   = true; continue; }
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
    return decl;
}

DeclPtr Parser::parseFunctionOrGlobalVar(bool isConst, bool isConsteval,
                                          bool isInline, bool isExtern, bool isStatic,
                                          std::vector<GenericParam> genericParams) {
    auto loc = cur().loc;
    TypePtr retType = parseType();
    if (!retType) return nullptr;

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
    auto gv = std::make_unique<GlobalVarDecl>(std::move(name), loc);
    gv->type      = std::move(retType);
    gv->isConst   = isConst;
    gv->isStatic  = isStatic;
    gv->isExtern  = isExtern;
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
    while (!atEnd() && !cur().is(TK::RParen)) {
        if (cur().is(TK::DotDotDot)) { consume(); fn->isVariadic = true; break; }
        ParamDecl p;
        p.loc  = cur().loc;
        p.type = parseType();
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
        fn->params.push_back(std::move(p));
        if (!match(TK::Comma)) break;
    }
    expect(TK::RParen, "expected ')' closing parameter list");

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

std::unique_ptr<StructDecl> Parser::parseStructDecl(bool isUnion) {
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
                while (!atEnd() && !cur().is(TK::RParen)) {
                    if (cur().is(TK::DotDotDot)) { consume(); break; }
                    ParamDecl p;
                    p.loc  = cur().loc;
                    p.type = parseType();
                    if (cur().is(TK::Ident)) { p.name = cur().text; consume(); }
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
            while (!atEnd() && !cur().is(TK::RParen)) {
                if (cur().is(TK::DotDotDot)) { consume(); break; }
                ParamDecl p;
                p.loc  = cur().loc;
                p.type = parseType();
                if (cur().is(TK::Ident)) { p.name = cur().text; consume(); }
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
            if (!cur().is(TK::Ident)) {
                diag_.error(cur().loc, "expected field name");
            } else consume();
            // Optional array size
            if (cur().is(TK::LBracket)) {
                consume();
                int64_t sz = -1;
                if (cur().is(TK::IntLit)) { sz = cur().intVal; consume(); }
                expect(TK::RBracket);
                fd.type = makeArray(std::move(fd.type), sz);
            }
            expect(TK::Semicolon, "expected ';' after struct field");
            sd->fields.push_back(std::move(fd));
        }
    }
    expect(TK::RBrace, "expected '}' closing struct");
    match(TK::Semicolon);
    return sd;
}

std::unique_ptr<EnumDecl> Parser::parseEnumDecl() {
    auto loc = cur().loc;
    std::string name;
    if (cur().is(TK::Ident)) { name = cur().text; consume(); }

    auto ed = std::make_unique<EnumDecl>(std::move(name), loc);
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
    std::string name = cur().text;
    if (!cur().is(TK::Ident)) {
        diag_.error(cur().loc, "expected typedef name");
    } else consume();
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
    case TK::KW_const: {
        consume();
        return parseVarDeclStmt(true, false);
    }
    case TK::KW_static: {
        consume();
        bool isConst = false;
        if (match(TK::KW_const)) isConst = true;
        return parseVarDeclStmt(isConst, true);
    }
    default:
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
    std::string name = cur().text;
    if (!cur().is(TK::Ident)) {
        diag_.error(cur().loc, "expected variable name");
    } else consume();

    // C-style array dimension after name: int arr[N]
    while (cur().is(TK::LBracket)) {
        consume();
        int64_t sz = -1;
        if (cur().is(TK::IntLit)) { sz = cur().intVal; consume(); }
        expect(TK::RBracket, "expected ']' in array declaration");
        ty = makeArray(std::move(ty), sz);
    }

    ExprPtr init;
    if (match(TK::Eq)) init = parseAssignExpr();

    expect(TK::Semicolon, "expected ';' after variable declaration");

    auto vs = std::make_unique<VarDeclStmt>(std::move(name), std::move(ty),
                                             std::move(init), loc);
    vs->isConst  = isConst;
    vs->isStatic = isStatic;
    return vs;
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
        // Range: N..M — two consecutive dots
        if (cur().is(TK::Dot) && peek().is(TK::Dot)) {
            consume(); consume(); // eat '..'
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
    while (cur().is(TK::Plus) || cur().is(TK::Minus)) {
        auto loc = cur().loc;
        BinaryOp op = cur().is(TK::Plus) ? BinaryOp::Add : BinaryOp::Sub;
        consume();
        auto rhs = parseMulExpr();
        lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
    }
    return lhs;
}

ExprPtr Parser::parseMulExpr() {
    auto lhs = parseUnaryExpr();
    while (cur().is(TK::Star) || cur().is(TK::Slash) || cur().is(TK::Percent)) {
        auto loc = cur().loc;
        BinaryOp op = cur().is(TK::Star) ? BinaryOp::Mul
                    : cur().is(TK::Slash)? BinaryOp::Div : BinaryOp::Mod;
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
        consume(); // '('
        // Try to parse type
        TypePtr t = parseType();
        if (cur().is(TK::RParen)) {
            consume(); // ')'
            auto e = parseUnaryExpr();
            return std::make_unique<CastExpr>(std::move(t), std::move(e), toks_[saved].loc);
        }
        // Not a cast — restore
        pos_ = saved;
    }
    return parsePostfixExpr();
}

ExprPtr Parser::parsePostfixExpr() {
    auto e = parsePrimaryExpr();
    for (;;) {
        auto loc = cur().loc;
        if (cur().is(TK::LBracket)) {
            consume();
            auto idx = parseExpr();
            expect(TK::RBracket, "expected ']'");
            e = std::make_unique<SubscriptExpr>(std::move(e), std::move(idx), loc);
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
        std::string s = cur().text; consume();
        return std::make_unique<StringLitExpr>(std::move(s), loc);
    }
    case TK::CharLit: {
        char c = cur().text.empty() ? 0 : cur().text[0]; consume();
        return std::make_unique<CharLitExpr>(c, loc);
    }
    case TK::KW_true:  consume(); return std::make_unique<BoolLitExpr>(true, loc);
    case TK::KW_false: consume(); return std::make_unique<BoolLitExpr>(false, loc);
    case TK::KW_null:  consume(); return std::make_unique<NullLitExpr>(loc);
    case TK::Ident:
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
        // Compound initializer: { a, b, c }
        consume();
        std::vector<ExprPtr> inits;
        while (!atEnd() && !cur().is(TK::RBrace)) {
            inits.push_back(parseAssignExpr());
            if (!match(TK::Comma)) break;
        }
        expect(TK::RBrace, "expected '}' closing initializer");
        return std::make_unique<CompoundInitExpr>(std::move(inits), loc);
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
