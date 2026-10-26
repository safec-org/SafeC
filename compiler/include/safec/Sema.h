#pragma once
#include "safec/AST.h"
#include "safec/Clone.h"
#include "safec/Diagnostic.h"
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <string>
#include <functional>
#include <set>
#include <optional>

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
    // Target pointer width in bits (the width of size_t at the C boundary).
    void setPointerBits(unsigned b) { pointerBits_ = b; }
    // Functions the program provides itself, defined in another translation
    // unit (safec --provides=malloc,free,...): no freestanding warning.
    void addProvidedFunction(const std::string &n) { providedFns_.insert(n); }

    // Run all analyses; returns false if there were errors
    bool run();

    // ── Ownership (Drop) and thread safety (Send/Sync) ──────────────────────
    // A struct with a 'void drop()' method implements the builtin Drop
    // trait. A value is must-drop when its type implements Drop or holds a
    // must-drop value by value (a field, an array element, a tuple member,
    // an optional's payload) — references, pointers and slices only borrow.
    bool hasDropMethod(const StructType &st) const;
    bool isMustDrop(const TypePtr &ty) const;
    // drop() itself, or a method declared 'consume' in its struct body.
    bool isConsumeMethod(const FunctionDecl *fd) const;
    // A type as written in source: 'Vec<int>' rather than the mangled
    // instance name.
    std::string displayType(const TypePtr &ty) const;
    // "" when 'ty' is Send (sync=false) / Sync (sync=true), else why not.
    // 'syncMut' asks whether shared *mutable* access is race-free (only
    // types marked 'unsafe impl Sync' and aggregates of them).
    std::string threadSafety(const TypePtr &ty, bool sync, bool syncMut = false,
                             bool inOwner = false, int depth = 0) const;

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
    // goto/label rules: every label is unique within its function, and a
    // goto may not jump into the scope of a local whose declaration it
    // skips (see the definition).
    void checkGotoJumps(FunctionDecl &fn);
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
    TypePtr checkSlice(SliceExpr &e);
    TypePtr checkMember(MemberExpr &e);
    TypePtr checkCast(CastExpr &e);
    TypePtr checkAssign(AssignExpr &e);
    TypePtr checkAddrOf(UnaryExpr &e);
    // Is 'e' (an already-checked lvalue) volatile-qualified? See
    // CodeGen::isVolatileLValue, which mirrors this for code generation.
    bool isVolatileLValue(const Expr &e) const;
    TypePtr checkDeref(UnaryExpr &e);
    TypePtr checkNew(NewExpr &e);
    TypePtr checkTupleLit(TupleLitExpr &e);
    TypePtr checkSpawn(SpawnExpr &e);

    // Aggregate initializer '{a, b, c}' against a known target type (struct
    // fields or array elements, matched positionally). Falls back to
    // checking each element with no target type (result type 'void') when
    // the target isn't a struct or array — e.g. type is still being
    // inferred from the initializer.
    TypePtr checkCompoundInit(CompoundInitExpr &e, const TypePtr &targetTy);

    // ── Heap/arena lifetime analysis ─────────────────────────────────────────
    // Use-after-free, double-free and use-after-arena-reset/free_to checking
    // (MemoryFlow.cpp): a dataflow analysis over every function body checked
    // here ('checkedFns_', recorded by checkFunction), with per-function
    // summaries applied at call sites. Runs at the end of run(), once every
    // body (monomorphized instances included) is type-checked.
    void checkMemoryFlow();
    std::vector<FunctionDecl *> checkedFns_;
    mutable std::unordered_map<const StructType *, int> mustDropCache_;   // 0 no, 1 yes, 2 in progress
    // Structs whose 'void drop() if (cond);' condition is false: not Drop.
    std::unordered_set<std::string> disabledDrops_;

    // ── Region safety ─────────────────────────────────────────────────────────
    // Returns the scope depth a reference type originated from
    // Emits error if a reference escapes to a shorter-lived scope
    void checkRegionEscape(const TypePtr &ty, int targetScopeDepth,
                           SourceLocation loc, const char *ctx);

    // Check that a nullable reference is tested before deref
    void checkNullabilityDeref(const TypePtr &ty, SourceLocation loc);

    // ── Alias tracking (NLL borrow checker) ─────────────────────────────────
    // Non-Lexical Lifetimes: borrows end at last use, not scope exit.
    struct AliasRecord {
        std::string varName;
        std::string borrower;     // name of the variable holding the borrow
        bool        isMutable;
        int         scopeDepth;
        bool        released = false;  // NLL: marked true when borrow is no longer used
    };
    std::unordered_map<std::string, std::vector<AliasRecord>> aliasMap_;

    void trackRef(const std::string &targetVar, bool isMut, int depth,
                  const std::string &borrower = {}, SourceLocation loc = {});
    void untrackScope(int depth);
    void releaseUnusedBorrows(const std::string &borrower);

    // ── Inter-procedural region analysis ─────────────────────────────────────
    // Validates region constraints across function call boundaries
    void checkCallRegions(CallExpr &e, FunctionType *ft, size_t selfOffset);

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

    // ── Generic structs ────────────────────────────────────────────────────────
    // 'generic<T> struct Box { ... }' templates — collectStruct() never
    // registers these into typeRegistry_ (there's no concrete "T" to lay
    // out a real LLVM struct with); it just records the StructDecl here.
    // Instantiated lazily, the first time 'struct Box<SomeType>' is
    // actually resolved (see instantiateGenericStruct), same "monomorphize
    // on first use, cache by (name, type args) after that" shape as
    // generic functions above.
    std::unordered_map<std::string, StructDecl *>  genericStructTemplates_; // non-owning
    std::map<MonoKey, std::shared_ptr<StructType>>  genericStructCache_;
    std::vector<std::unique_ptr<StructDecl>>       monoStructs_;            // owns clones
    // Out-of-line method definitions ('inline int Box::get() const {...}')
    // whose methodOwner names a generic struct template — collectFunction
    // defers these here instead of processing them immediately (there's
    // no concrete "Box" type yet to synthesize a 'self' param against).
    // instantiateGenericStruct clones+finishes each one the first time
    // that template is actually instantiated.
    std::unordered_map<std::string, std::vector<FunctionDecl *>> genericStructMethodTemplates_;
    // Each generic struct's instances with their substitutions, and which
    // (instance, method template) pairs have been instantiated.
    std::unordered_map<std::string, std::vector<std::pair<StructDecl *, TypeSubst>>> genericStructInstances_;
    std::set<std::pair<StructDecl *, FunctionDecl *>> instantiatedStructMethods_;
    // checkBody false after a failed bound: the body would only repeat the
    // missing methods as errors.
    void instantiateStructMethod(FunctionDecl *methodTpl, StructDecl *inst, const TypeSubst &subs,
                                 bool checkBody = true);
    void completeGenericStructInstances();

    // Monomorphizes 'name<typeArgs...>' (a generic struct template
    // reference) into a concrete StructType, cloning+substituting its
    // fields and any associated methods (top-level FunctionDecls with
    // isMethod=true and methodOwner==name) the same way a generic
    // function call already clones+substitutes its body. Returns an error
    // type (already diagnosed) if 'name' isn't a known generic struct
    // template or the type-argument count doesn't match.
    TypePtr instantiateGenericStruct(const std::string &name,
                                      const std::vector<TypePtr> &typeArgs,
                                      SourceLocation loc);

    // Struct-internal method forward-declarations ('int read(...);' inside a
    // struct body, as opposed to an out-of-line 'T::read(...) { ... }'
    // definition) get synthesized into full FunctionDecl nodes so a method
    // call type-checks from the declaration alone — the definition can then
    // live in a different translation unit and be resolved at link time,
    // same as any other extern function (the '.h'/'.sc' pairing convention).
    // Owned here; appended to tu_.decls at the end of run() so CodeGen emits
    // the matching extern declaration, exactly like monoFunctions_ above.
    std::vector<std::unique_ptr<FunctionDecl>>     synthesizedMethods_;

    // Return true if inference succeeded, filling subs. 'returnType'/
    // 'expectedType' are an optional fallback (see expectedType_'s doc
    // comment below) for generic parameters that appear only in the
    // return type -- unbound after the normal per-argument pass, they're
    // matched against 'expectedType' via 'returnType' as the pattern
    // (e.g. return type 'T*' against an expected 'int*' binds T=int).
    bool inferTypeArgs(const std::vector<ParamDecl>      &params,
                       const std::vector<TypePtr>         &argTypes,
                       const std::vector<GenericParam>    &genericParams,
                       TypeSubst                          &subs,
                       const TypePtr                       &returnType = nullptr,
                       const TypePtr                       &expectedType = nullptr);
    bool matchType(const TypePtr &paramTy, const TypePtr &argTy, TypeSubst &subs);
    FunctionDecl *instantiateGeneric(FunctionDecl *fnDecl, TypeSubst &subs);
    std::optional<int64_t> tryConstInt(const Expr &e);
    TypePtr coerceGenericFnValue(Expr &e, TypePtr exprTy, const TypePtr &target);

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
    // Extra text for a failed raw T* → '&T' conversion in safe code.
    std::string rawToRefHint(const TypePtr &from, const TypePtr &to) const;
    // True if 'e' is a (possibly negative-signed) integer literal whose value
    // fits exactly in 'to' — standard C allows 'unsigned char c[] = {0x63,
    // ...}' even though each element is nominally an 'int' literal, because
    // the *value* is known at compile time to fit; canImplicitlyConvert()
    // alone can't see the value, only the two static types.
    bool intLiteralFitsType(const Expr &e, const TypePtr &to) const;
    bool floatLitFitsType(const Expr &e, const TypePtr &to) const;
    TypePtr checkGenericSelection(GenericSelectionExpr &e);
    TypePtr checkMatchExpr(MatchExpr &e);
    TypePtr checkFnEval(FnEvalExpr &e);
    bool refToPointerArgCompatible(const TypePtr &from, const TypePtr &to) const;

    // ── State ─────────────────────────────────────────────────────────────────
    unsigned pointerBits_ = 64;
    // Monomorphized generic struct name -> (template name, type arguments),
    // so inference can match 'struct Vec<T>' against a concrete 'Vec_int'.
    std::unordered_map<std::string, std::pair<std::string, std::vector<TypePtr>>> structInstanceOrigin_;
    std::unordered_set<std::string> providedFns_;
    void adoptTupleLiteral(ExprPtr &e, const TypePtr &expected);
    bool cAbiSizeArgCompatible(CallExpr &e, const TypePtr &arg, const TypePtr &param) const;
    bool freestanding_ = false;  // --freestanding mode: warn on stdlib calls
    bool checkingPure_ = false;  // true while checking a 'pure' function body
    int  loopDepth_    = 0;      // >0 when inside a loop (for break/continue validation)
    std::unordered_set<std::string> functionLabels_; // labels in current function
    TranslationUnit &tu_;
    DiagEngine      &diag_;

    std::vector<Scope>                      scopes_;
    std::unordered_map<std::string, TypePtr> typeRegistry_;  // struct/enum types by name
    std::unordered_map<std::string, RegionDecl*> regionRegistry_;
    // Fallback hint for generic-function inference when a type parameter
    // appears only in the return type (e.g. 'generic<T> T* vec_at(&stack
    // Vec v, unsigned long idx)') — inference is otherwise call-site-only
    // (reads argument types, never a target type), so a call like
    // 'vec_at(&v, 1UL)' has nothing to infer T from on its own. Set by
    // checkVarDecl (from an explicit, non-inferred declared type) and
    // checkAssign (from the LHS's resolved type) just before checking the
    // initializer/RHS expression, and consumed — cleared immediately,
    // regardless of whether it ends up used — at the top of checkCall, so
    // it never leaks into unrelated nested expression checks (the callee,
    // the call's own arguments, a deeper nested call). See inferTypeArgs's
    // 'expectedType' parameter for where it's actually applied.
    TypePtr expectedType_;
    // Method registry: "StructName::methodName" → FunctionDecl* (mangled)
    std::unordered_map<std::string, FunctionDecl*> methodRegistry_;
    // Static methods: "StructName::method" → FunctionDecl* (mangled
    // StructName_method, no self), called as 'StructName::method(...)' or
    // 'ns::StructName::method(...)'. On a generic struct the decl is a
    // generic function over the struct's parameters. staticMethodNames_
    // holds the names each struct body declares 'static', so an
    // out-of-line definition is recognized as one.
    std::unordered_map<std::string, FunctionDecl*> staticMethodRegistry_;
    std::unordered_map<std::string, std::unordered_set<std::string>> staticMethodNames_;
    // "StructName::method" → how many type parameters of its own the struct
    // body declares for it ('generic<T> void m(...);'), so an out-of-line
    // definition with a different count is an error.
    std::unordered_map<std::string, size_t> methodOwnGenericArity_;
    // "StructName::method" → the FunctionDecl synthesized from the struct
    // body's declaration, which an out-of-line definition must match.
    std::unordered_map<std::string, FunctionDecl*> methodBodyDecl_;
    // Re-resolves the struct-body method declarations once every top-level
    // declaration is collected: a typedef or struct declared after the
    // struct that mentions it. Names still unresolved are unknown types.
    void resolveMethodBodyDecls();
    // Namespace registries: "ns::name" (as written at the call site) →
    // decl* (already mangled, e.g. FunctionDecl::name == "ns_name"). See
    // collectFunction/collectGlobalVar for population and checkIdent for
    // the lookup that rewrites a qualified IdentExpr to the mangled name.
    std::unordered_map<std::string, FunctionDecl*>  namespaceFnRegistry_;
    std::unordered_map<std::string, GlobalVarDecl*> namespaceVarRegistry_;
    // Distinct namespace names seen (usually just {"std"}); used by
    // checkIdent's unqualified-name fallback so code lexically inside
    // 'namespace std { ... }' can still call siblings unqualified
    // ('chacha_qr_(...)' resolving to 'std::chacha_qr_') without every
    // internal call site needing to be rewritten to 'std::chacha_qr_(...)'.
    std::unordered_set<std::string> namespaceNames_;
    int                                     scopeDepth_ = 0;

    // ── Trait enforcement ─────────────────────────────────────────────────────
    struct TraitDef { std::string name; std::vector<std::string> requiredOps; };
    static const TraitDef builtinTraits_[];
    // User 'trait Name { ... }' declarations: trait name → required method
    // names (see TraitDecl in AST.h). Populated in collectDecls, consulted
    // by satisfiesTrait() alongside builtinTraits_.
    std::unordered_map<std::string, std::vector<std::string>> userTraits_;
    // Each user trait's supertraits ('trait Name : A + B'), and where it was
    // declared. checkTraitSupertraits() rejects unknown supertraits and
    // cycles (dropping the offending edges), so satisfiesTrait()'s recursion
    // through them always terminates.
    std::unordered_map<std::string, std::vector<std::string>> userTraitSupers_;
    std::unordered_map<std::string, SourceLocation> userTraitLocs_;
    std::unordered_map<std::string, TraitDecl*> userTraitDecls_;
    // Where the declaration or statement being checked starts: resolveType()
    // has no location of its own, so a generic struct instantiated there
    // (and a constraint it fails) is reported here.
    SourceLocation typeUseLoc_;
    void checkTraitSupertraits();
    bool isBuiltinTrait(const std::string &trait) const;
    bool satisfiesTrait(const TypePtr &ty, const std::string &trait) const {
        return traitMismatch(ty, trait).empty();
    }
    // Why 'ty' doesn't satisfy 'trait' ("" when it does): a missing method,
    // or one whose parameter count or static-ness differs from the trait's.
    std::string traitMismatch(const TypePtr &ty, const std::string &trait) const;
    // "type 'X' does not satisfy constraint 'T' <where> (<reason>)"
    std::string constraintError(const TypePtr &ty, const std::string &trait, const std::string &where) const;
    static std::string binaryOpToMethodName(BinaryOp op);

};

} // namespace safec
