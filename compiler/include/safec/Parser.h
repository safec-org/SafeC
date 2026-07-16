#pragma once
#include "safec/AST.h"
#include "safec/Token.h"
#include "safec/Diagnostic.h"
#include <vector>
#include <memory>

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

    std::vector<GenericParam> parseGenericParams(); // <T: Constraint, ...>

    // ── Top-level declarations ────────────────────────────────────────────────
    DeclPtr parseTopLevelDecl();
    DeclPtr parseFunctionOrGlobalVar(bool isConst, bool isConsteval,
                                     bool isInline, bool isExtern, bool isStatic,
                                     std::vector<GenericParam> genericParams);
    std::unique_ptr<FunctionDecl> parseFunctionDecl(
        TypePtr retType, std::string name, SourceLocation loc,
        bool isConst, bool isConsteval, bool isInline, bool isExtern,
        bool isVariadic, std::vector<GenericParam> genericParams);
    std::unique_ptr<StructDecl> parseStructDecl(bool isUnion);
    std::unique_ptr<EnumDecl>   parseEnumDecl();
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
