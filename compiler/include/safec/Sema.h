#pragma once
#include "safec/AST.h"
#include "safec/Clone.h"
#include "safec/Diagnostic.h"
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <functional>

namespace safec {

// ── Symbol kinds ──────────────────────────────────────────────────────────────
enum class SymKind { Variable, Function, Type, Region, Enum };

struct Symbol {
    SymKind      kind;
    std::string  name;
    TypePtr      type;
    VarDecl     *varDecl   = nullptr;
    FunctionDecl *fnDecl   = nullptr;
    int          scopeDepth = 0;
    bool         initialized = false;
};

// ── Scope ─────────────────────────────────────────────────────────────────────
class Scope {
public:
    explicit Scope(int depth, bool isUnsafe = false)
        : depth_(depth), isUnsafe_(isUnsafe) {}

    Symbol *lookup(const std::string &name);
    bool    define(Symbol sym);   // returns false on duplicate

    int  depth()    const { return depth_; }
    bool isUnsafe() const { return isUnsafe_; }

private:
    std::unordered_map<std::string, Symbol> symbols_;
    int  depth_;
    bool isUnsafe_;
};

// ── Semantic Analysis ─────────────────────────────────────────────────────────
// Performs:
//   1. Name resolution
//   2. Type checking
//   3. Region escape analysis (PLAN §4, §5.2)
//   4. Definite initialization analysis (PLAN §5.1)
//   5. Alias/mutability rules for safe references (PLAN §5.4)
//   6. Nullability enforcement (PLAN §5.5)
//   7. unsafe boundary checks (PLAN §6)
class Sema {
public:
    Sema(TranslationUnit &tu, DiagEngine &diag);

    void setFreestanding(bool v) { freestanding_ = v; }

    // Run all analyses; returns false if there were errors
    bool run();

private:
    // ── Scope management ──────────────────────────────────────────────────────
    void pushScope(bool isUnsafe = false);
    void popScope();
    Symbol *lookup(const std::string &name);
    void    define(Symbol sym);
    bool    inUnsafeScope() const;

    // ── Type registry ─────────────────────────────────────────────────────────
    void registerBuiltinTypes();
    TypePtr lookupType(const std::string &name) const;
    TypePtr resolveType(TypePtr ty); // resolve named struct/enum references

    // ── Pass 1: collect top-level declarations ────────────────────────────────
    void collectDecls(TranslationUnit &tu);
    void collectFunction(FunctionDecl &fn);
    void collectStruct(StructDecl &sd);
    void collectEnum(EnumDecl &ed);
    void collectRegion(RegionDecl &rd);
    void collectGlobalVar(GlobalVarDecl &gv);

    // ── Pass 2: check bodies ──────────────────────────────────────────────────
    void checkDecl(Decl &d);
    void checkFunction(FunctionDecl &fn);
    void checkGlobalVar(GlobalVarDecl &gv);
    void checkStaticAssert(StaticAssertDecl &sa);

    // ── Statement checking ────────────────────────────────────────────────────
    void checkStmt(Stmt &s, FunctionDecl &fn);
    void checkCompound(CompoundStmt &cs, FunctionDecl &fn);
    void checkIf(IfStmt &s, FunctionDecl &fn);
    void checkWhile(WhileStmt &s, FunctionDecl &fn);
    void checkFor(ForStmt &s, FunctionDecl &fn);
    void checkReturn(ReturnStmt &s, FunctionDecl &fn);
    void checkVarDecl(VarDeclStmt &s, FunctionDecl &fn);
    void checkUnsafe(UnsafeStmt &s, FunctionDecl &fn);
    void checkStaticAssertStmt(StaticAssertStmt &s);

    // ── Expression type-checking ──────────────────────────────────────────────
    TypePtr checkExpr(Expr &e);
    TypePtr checkIntLit(IntLitExpr &e);
    TypePtr checkFloatLit(FloatLitExpr &e);
    TypePtr checkBoolLit(BoolLitExpr &e);
    TypePtr checkStringLit(StringLitExpr &e);
    TypePtr checkIdent(IdentExpr &e);
    TypePtr checkUnary(UnaryExpr &e);
    TypePtr checkBinary(BinaryExpr &e);
    TypePtr checkTernary(TernaryExpr &e);
    TypePtr checkCall(CallExpr &e);
    TypePtr checkSubscript(SubscriptExpr &e);
    TypePtr checkMember(MemberExpr &e);
    TypePtr checkCast(CastExpr &e);
    TypePtr checkAssign(AssignExpr &e);
    TypePtr checkAddrOf(UnaryExpr &e);
    TypePtr checkDeref(UnaryExpr &e);
    TypePtr checkNew(NewExpr &e);
    TypePtr checkTupleLit(TupleLitExpr &e);
    TypePtr checkSpawn(SpawnExpr &e);

    // ── Region safety ─────────────────────────────────────────────────────────
    // Returns the scope depth a reference type originated from
    // Emits error if a reference escapes to a shorter-lived scope
    void checkRegionEscape(const TypePtr &ty, int targetScopeDepth,
                           SourceLocation loc, const char *ctx);

    // Check that a nullable reference is tested before deref
    void checkNullabilityDeref(const TypePtr &ty, SourceLocation loc);

    // ── Alias tracking ────────────────────────────────────────────────────────
    // Tracks mutable references to each object to detect aliasing violations
    struct AliasRecord {
        std::string varName;
        bool        isMutable;
        int         scopeDepth;
    };
    std::unordered_map<std::string, std::vector<AliasRecord>> aliasMap_;

    void trackRef(const std::string &targetVar, bool isMut, int depth);
    void untrackScope(int depth);

    // ── Generics monomorphization ─────────────────────────────────────────────
    struct MonoKey {
        std::string              funcName;
        std::vector<std::string> typeArgStrs; // type->str() for each generic param
        bool operator<(const MonoKey &o) const {
            if (funcName != o.funcName) return funcName < o.funcName;
            return typeArgStrs < o.typeArgStrs;
        }
    };
    std::map<MonoKey, FunctionDecl *>              monoCache_;    // non-owning
    std::vector<std::unique_ptr<FunctionDecl>>     monoFunctions_; // owns clones

    // Return true if inference succeeded, filling subs.
    bool inferTypeArgs(const std::vector<ParamDecl>      &params,
                       const std::vector<TypePtr>         &argTypes,
                       const std::vector<GenericParam>    &genericParams,
                       TypeSubst                          &subs);
    bool matchType(const TypePtr &paramTy, const TypePtr &argTy, TypeSubst &subs);

    MonoKey   makeMonoKey(const std::string &name, const TypeSubst &subs,
                           const std::vector<GenericParam> &params);
    std::string mangleName(const std::string &base, const TypeSubst &subs,
                            const std::vector<GenericParam> &params);

    // ── Bare-metal / effect system checks ────────────────────────────────────
    void checkAsmStmt(Stmt &s, FunctionDecl &fn);

    // ── Helpers ───────────────────────────────────────────────────────────────
    TypePtr unify(TypePtr a, TypePtr b, SourceLocation loc, const char *ctx);
    TypePtr resolveStruct(StructType &st);
    StructType *asStruct(TypePtr &ty);

    bool isNumeric(const TypePtr &ty) const;
    bool isIntegral(const TypePtr &ty) const;
    bool canImplicitlyConvert(const TypePtr &from, const TypePtr &to) const;

    // ── State ─────────────────────────────────────────────────────────────────
    bool freestanding_ = false;  // --freestanding mode: warn on stdlib calls
    bool checkingPure_ = false;  // true while checking a 'pure' function body
    TranslationUnit &tu_;
    DiagEngine      &diag_;

    std::vector<Scope>                      scopes_;
    std::unordered_map<std::string, TypePtr> typeRegistry_;  // struct/enum types by name
    std::unordered_map<std::string, RegionDecl*> regionRegistry_;
    // Method registry: "StructName::methodName" → FunctionDecl* (mangled)
    std::unordered_map<std::string, FunctionDecl*> methodRegistry_;
    int                                     scopeDepth_ = 0;

    // ── Trait enforcement ─────────────────────────────────────────────────────
    struct TraitDef { std::string name; std::vector<std::string> requiredOps; };
    static const TraitDef builtinTraits_[];
    bool satisfiesTrait(const TypePtr &ty, const std::string &trait) const;
    static std::string binaryOpToMethodName(BinaryOp op);

};

} // namespace safec
