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
    TypePtr parseReferenceType(bool nullable); // &region T
    Region  parseRegionQualifier(std::string &arenaName);

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
