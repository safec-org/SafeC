#pragma once
#include "safec/AST.h"
#include "safec/Token.h"
#include "safec/Diagnostic.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace safec {

class Parser {
public:
    Parser(std::vector<Token> tokens, DiagEngine &diag);

    // Parse a full translation unit
    std::unique_ptr<TranslationUnit> parseTranslationUnit();

private:
    // ── Token stream ──────────────────────────────────────────────────────────
    const Token &cur()  const { return toks_[pos_]; }
    const Token &peek(int offset = 1) const {
        size_t idx = pos_ + offset;
        return (idx < toks_.size()) ? toks_[idx] : toks_.back();
    }
    Token consume();
    Token expect(TK kind, const char *msg = nullptr);
    bool  check(TK k)  const { return cur().is(k); }
    bool  match(TK k);
    bool  atEnd()      const { return cur().is(TK::Eof); }

    // Contextual keyword (also valid identifier in non-keyword context)
    bool checkIdent(const char *name) const;

    // ── Type parsing ──────────────────────────────────────────────────────────
    TypePtr parseType();
    TypePtr parseBaseType();           // primitive / struct / named type
    TypePtr parseTypeDeclarator(TypePtr base); // array / pointer suffixes
    TypePtr parseReferenceType(bool nullable, bool leadingConst = false); // &region T
    Region  parseRegionQualifier(std::string &arenaName);

    // Constant integer expression inside 'T[...]' (array size). Supports
    // integer literals combined with + - * / % and parentheses, so that
    // preprocessor-expanded sizes like 'N * M' work, not just bare literals.
    int64_t parseArraySizeConst();

    // Parses the array-size expression inside 'T[...]'. Fast path: pure
    // literal arithmetic is folded immediately via parseArraySizeConst() (the
    // common case, avoids building an AST + interpreter round-trip). Falls
    // back to parsing a full expression when an identifier is present — a
    // named constant or a consteval function call, e.g. 'T arr[square(3)]' —
    // which can't be evaluated until function bodies exist and const-eval
    // runs post-parse. Returns the resolved size, or -1 with '*outSizeExpr'
    // set to the deferred (owned-by-TU) expression when resolution must wait.
    int64_t parseArraySize(Expr **outSizeExpr);

    // Consumes zero or more '[size]' suffixes after a declarator (e.g. the
    // '[8][64]' in 'char parts[8][64]') and wraps 'base' in nested ArrayTypes
    // with the CORRECT C nesting order: the first bracket written is the
    // outermost array (parts has 8 elements, each a 64-char buffer), not the
    // last. A naive left-to-right wrap ('base = makeArray(base, sz)' inside
    // the loop) nests them backwards, since each iteration wraps the
    // previous result rather than the previous result wrapping it — so this
    // collects all dimensions first and builds from the last one inward.
    TypePtr parseArrayDeclaratorSuffix(TypePtr base);

    // Owns array-size expressions that couldn't be folded at parse time (see
    // parseArraySize); moved into TranslationUnit::arraySizeExprs when
    // parsing completes so the Expr*s stashed in ArrayType::sizeExpr stay valid.
    std::vector<ExprPtr> pendingArraySizeExprs_;

    // ── Namespaces ────────────────────────────────────────────────────────────
    // 'namespace X { decl* }' recurses through parseTopLevelDecl() for its
    // body, so each contained decl gets stamped with the (possibly nested,
    // "::"-joined) enclosing namespace name and collected here rather than
    // returned directly — parseTopLevelDecl() can only hand back one decl at
    // a time, but a namespace body holds many. Drained into
    // TranslationUnit::decls right after the top-level statement that
    // produced them (see parseTranslationUnit) — NOT deferred to EOF, since
    // Sema's collectDecls() is a single forward pass and needs a namespaced
    // typedef to appear at its true source position relative to later
    // top-level code that uses it.
    std::vector<DeclPtr>     pendingNamespaceDecls_;
    std::vector<std::string> namespaceStack_;
    void parseNamespaceDecl();

    // ── Anonymous inline struct/union ('struct S { union { int a; }; };') ───
    // An inline 'struct { ... }' / 'union { ... }' with no tag name, used
    // directly as a field's type, is parsed as a full StructDecl (like any
    // top-level struct) under a synthesized unique name and stashed here —
    // parseBaseType() can only return a TypePtr placeholder, not inject a
    // whole extra top-level declaration, so (mirroring
    // pendingNamespaceDecls_) it's collected and drained into
    // TranslationUnit::decls once parsing completes, letting Sema's normal
    // collectStruct()/resolveType() machinery pick it up unmodified.
    std::vector<DeclPtr> pendingAnonStructs_;
    int                  anonStructCounter_ = 0;

    std::vector<GenericParam> parseGenericParams(); // <T: Constraint, ...>

    // ── GNU/Clang __attribute__((...)) tolerance ─────────────────────────────
    // SafeC has its own native attribute keywords (inline, noreturn, pure,
    // packed, align(N), section("...")), but real-world C source (headers
    // copy-pasted into a .sc file, or code ported from C) uses GCC/Clang's
    // '__attribute__((...))' syntax pervasively. Rather than reject it
    // outright, recognize the syntax generically (arbitrary attribute-spec
    // list, arbitrary parenthesized args) and map the handful of common
    // attributes onto SafeC's existing equivalents; anything unrecognized is
    // accepted and silently dropped rather than erroring — matching "any C
    // feature not natively modeled must at least be accepted" for a claimed
    // C superset. Legal in prefix position (before a declaration) and
    // suffix position (after a declarator, before ';' or '{}'); may repeat
    // ('__attribute__((a)) __attribute__((b))'), which the caller loops for.
    struct GnuAttrs {
        bool        hasInline    = false;
        bool        hasNoReturn  = false;
        bool        hasPure      = false;
        bool        hasPacked    = false;
        bool        hasMustUse   = false;
        int         alignment    = 0;
        std::string sectionName;
    };
    // Consumes zero or more consecutive '__attribute__((...))' groups,
    // merging recognized attributes into 'out'. Returns true if at least one
    // group was consumed (false, no tokens consumed, if none present).
    bool parseGnuAttributes(GnuAttrs &out);

    // ── C-style function-pointer declarator: T (*name)(params) ──────────────
    // SafeC's native spelling is 'fn RetType(Params) name' (see the KW_fn
    // case in parseBaseType) — the parser never supported C's declarator-
    // based 'T (*name)(params)' syntax (the '*' binds to the *name*, not the
    // type, unlike every other C declarator SafeC does support). Real C code
    // uses this constantly for callback typedefs/fields/params, so this is
    // tolerated as an alternative spelling for the same underlying type
    // ('&static fn(Params) RetType', identical to what 'fn' produces) rather
    // than requiring every ported C header to be rewritten.
    // Call with the parser positioned right after the base return type has
    // already been parsed (parseBaseType + parseTypeDeclarator, stopping at
    // '(' since '*'/'[' are the only declarator suffixes recognized there).
    // On match: consumes '(*name)(params)', fills outName/outType, returns
    // true. On no match: restores the position exactly and returns false.
    bool tryParseCStyleFuncPtrDeclarator(TypePtr retType, std::string &outName,
                                         TypePtr &outType);

    // ── Top-level declarations ────────────────────────────────────────────────
    DeclPtr parseTopLevelDecl();
    DeclPtr parseFunctionOrGlobalVar(bool isConst, bool isConsteval,
                                     bool isInline, bool isExtern, bool isStatic,
                                     std::vector<GenericParam> genericParams);
    std::unique_ptr<FunctionDecl> parseFunctionDecl(
        TypePtr retType, std::string name, SourceLocation loc,
        bool isConst, bool isConsteval, bool isInline, bool isExtern,
        bool isVariadic, std::vector<GenericParam> genericParams);
    // 'consumeTrailingSemicolon': the top-level 'struct Foo { ... };' form
    // needs its own terminating ';' consumed here; an inline anonymous/named
    // struct-as-type-position use ('struct S { struct { ... } x; };' — see
    // the KW_struct/KW_union case in parseBaseType) must NOT consume it,
    // since that ';' belongs to the *enclosing* field declaration and the
    // struct-field-parsing loop expects to consume it itself.
    std::unique_ptr<StructDecl> parseStructDecl(bool isUnion,
                                                 bool consumeTrailingSemicolon = true);
    std::unique_ptr<EnumDecl>   parseEnumDecl();
    std::unique_ptr<TraitDecl>  parseTraitDecl();
    std::unique_ptr<RegionDecl> parseRegionDecl();
    DeclPtr                     parseTypedef();
    DeclPtr                     parseStaticAssertDecl();

    // ── Statements ────────────────────────────────────────────────────────────
    StmtPtr parseStmt();
    std::unique_ptr<CompoundStmt> parseCompoundStmt();
    StmtPtr parseIfStmt();
    StmtPtr parseWhileStmt();
    StmtPtr parseDoWhileStmt();
    StmtPtr parseForStmt();
    StmtPtr parseReturnStmt();
    StmtPtr parseUnsafeStmt();
    StmtPtr parseStaticAssertStmt();
    StmtPtr parseVarDeclStmt(bool isConst, bool isStatic);
    StmtPtr parseExprStmt();
    StmtPtr parseDeferStmt(bool isErrDefer = false);
    StmtPtr parseMatchStmt();
    MatchPattern parseMatchPattern();
    StmtPtr parseAsmStmt();

    // ── Expressions ───────────────────────────────────────────────────────────
    ExprPtr parseExpr();
    ExprPtr parseAssignExpr();
    ExprPtr parseTernaryExpr();
    ExprPtr parseLogOrExpr();
    ExprPtr parseLogAndExpr();
    ExprPtr parseBitOrExpr();
    ExprPtr parseBitXorExpr();
    ExprPtr parseBitAndExpr();
    ExprPtr parseEqExpr();
    ExprPtr parseRelExpr();
    ExprPtr parseShiftExpr();
    ExprPtr parseAddExpr();
    ExprPtr parseMulExpr();
    ExprPtr parseUnaryExpr();
    ExprPtr parseCastExpr();       // (type) expr
    ExprPtr parsePostfixExpr();
    ExprPtr parsePrimaryExpr();
    ExprPtr parseMatchExpr();      // match (expr) { case pat: val, ... } as a value

    // ── Helpers ───────────────────────────────────────────────────────────────
    bool isTypeStart() const;       // current token begins a type?
    bool isTypeStart(const Token &t) const;
    AssignOp tokenToAssignOp(TK k) const;
    BinaryOp tokenToBinOp(TK k)    const;

    void syncToDecl(); // error recovery: skip to next top-level decl

    std::vector<Token> toks_;
    size_t             pos_ = 0;
    DiagEngine        &diag_;
};

} // namespace safec
