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
    // Arena-reset invalidation tracking (see Sema::regionGeneration_): set
    // when this variable is bound to a '&arena<R> T' value. 'arenaBindGen'
    // is a snapshot of R's generation counter at bind time; a later read of
    // this variable is stale if R's generation has since advanced (i.e. an
    // arena_reset<R>()/arena_destroy<R>() happened in between).
    std::string  arenaRegionName;
    int          arenaBindGen = -1;
    // arena_free_to<R>(mark) invalidation tracking (see
    // Sema::regionMarkDepth_/regionScratchGeneration_): a reference bound
    // before any arena_mark<R>() call ('arenaBindMarkDepth' == 0) is a
    // "permanent" binding that only a real reset/destroy can stale — a
    // free_to() only ever rewinds *into* an active mark scope, so it can
    // never be freeing memory a depth-0 reference points into. A reference
    // bound while inside a mark scope ('arenaBindMarkDepth' > 0) is
    // checked against 'arenaBindScratchGen', a snapshot of R's
    // free_to-only generation counter at bind time.
    int          arenaBindMarkDepth  = 0;
    int          arenaBindScratchGen = -1;
    // Heap use-after-free/double-free tracking (see Sema::heapFreeGeneration_):
    // set when this variable is bound to a '&heap T' value. 'heapBindGen' is
    // a snapshot, at bind time, of the free-generation counter for *this
    // specific variable's* VarDecl — a later read is stale if that counter
    // has since advanced (i.e. std::dealloc() was called on this same
    // variable in between). -1 means not tracked (not a '&heap T' binding).
    int          heapBindGen = -1;
    // Direct-copy alias tracking: when a '&heap T' variable's initializer
    // (or a later simple assignment) is exactly another already-tracked
    // '&heap T' variable's bare identifier ('&heap T q = p;' or 'q = p;'),
    // 'heapTrackKey' is set to *that* variable's own tracking key (resolved
    // transitively through any earlier aliasing) instead of this variable's
    // own VarDecl — so std::dealloc() on either p or q bumps the one shared
    // counter both are checked against, catching a use-after-free through
    // whichever name it happens via. nullptr (the common case: not a direct
    // alias of another tracked heap variable) means "key off my own
    // VarDecl," queried through Sema::heapKeyOf(sym) rather than reading
    // this field directly. This only follows the single-hop, syntactically
    // obvious "whole variable copied from another whole variable" shape —
    // an allocation reached through a struct field, an array element, a
    // function parameter/return, or any expression more complex than a bare
    // identifier still isn't tracked (see the warning in Memory & Regions).
    const VarDecl *heapTrackKey = nullptr;
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
    // Resolves a heap-tracked symbol's actual generation-counter key (see
    // Symbol::heapTrackKey) — its own VarDecl, unless it's a direct alias of
    // another tracked '&heap T' variable, in which case that variable's key.
    static const VarDecl *heapKeyOf(const Symbol *sym);

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
    TypePtr checkSlice(SliceExpr &e);
    TypePtr checkMember(MemberExpr &e);
    TypePtr checkCast(CastExpr &e);
    TypePtr checkAssign(AssignExpr &e);
    TypePtr checkAddrOf(UnaryExpr &e);
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
    bool freestanding_ = false;  // --freestanding mode: warn on stdlib calls
    bool checkingPure_ = false;  // true while checking a 'pure' function body
    int  loopDepth_    = 0;      // >0 when inside a loop (for break/continue validation)
    std::unordered_set<std::string> functionLabels_; // labels in current function
    TranslationUnit &tu_;
    DiagEngine      &diag_;

    std::vector<Scope>                      scopes_;
    std::unordered_map<std::string, TypePtr> typeRegistry_;  // struct/enum types by name
    std::unordered_map<std::string, RegionDecl*> regionRegistry_;
    // Arena-reset invalidation: bumped every time a call to
    // '__arena_reset_<name>'/'__arena_destroy_<name>' is seen during
    // traversal (see checkCall) — see Symbol::arenaBindGen for how this is
    // consulted. Flow-*sensitive* over the function's actual control flow
    // (see the "Arena-reset flow sensitivity" comment above Sema::checkIf
    // in Sema.cpp): if/else branches are checked against independent
    // snapshots and merged conservatively afterward rather than sharing one
    // running counter, and loop bodies are pre-scanned for reset/destroy/
    // free_to calls so an earlier-iteration invalidation is visible to
    // code textually before the call within the same body. Still sound
    // (never misses a real bug) but not a full dataflow fixpoint — see
    // reference/memory.md's "Arena References Die on Reset" section for
    // the precise remaining imprecision (deeply nested marks; unusual
    // expression nesting inside a loop body).
    std::unordered_map<std::string, int> regionGeneration_;
    // arena_mark<R>()/arena_free_to<R>(mark) invalidation (see
    // Symbol::arenaBindMarkDepth's doc comment): 'regionMarkDepth_[R]' is
    // incremented by every arena_mark<R>() and decremented (floored at 0)
    // by every arena_free_to<R>(...) seen during traversal — same flow-
    // sensitive control-flow tracking as regionGeneration_ above (if/else
    // snapshot+merge, loop-body pre-scan covering free_to as well as
    // reset/destroy). 'regionScratchGeneration_[R]' is bumped by free_to
    // only, and is what actually staless a reference bound at depth > 0 —
    // depth-0 (pre-mark) references never consult it at all, which is the
    // fix for free_to() no longer nuking bindings that exist *outside* the
    // scratch scope it's rewinding. One remaining imprecision, independent
    // of flow-sensitivity: both counters are one-per-region rather than
    // one-per-nesting-level, so a deeply nested mark's free_to can over-
    // flag an outer scope's still-valid reference (documented, sound-safe
    // direction — see reference/memory.md).
    std::unordered_map<std::string, int> regionMarkDepth_;
    std::unordered_map<std::string, int> regionScratchGeneration_;
    // Heap use-after-free/double-free invalidation: bumped every time a
    // recognized 'std::dealloc(p)' call is seen during traversal, keyed by
    // *the specific variable's VarDecl* rather than by name (see
    // Symbol::heapBindGen) — heap allocations are per-variable, not a
    // shared named entity like an arena region, and keying by VarDecl*
    // (stable, unique per declaration) sidesteps the false-positive risk a
    // name-keyed map would have across shadowed/reused variable names.
    // Same flow-sensitive if/else + loop-body tracking as the arena
    // counters above (see checkIf/checkWhile/checkFor); intra-procedural
    // only — a '&heap T' value freed through a *different* variable/alias
    // than the one being read (including across a function call boundary)
    // isn't tracked, so this is sound but not exhaustive, same spirit as
    // every other checker in this section.
    std::unordered_map<const VarDecl *, int> heapFreeGeneration_;
    // Suppresses the heap-staleness check in checkIdent for exactly one
    // identifier occurrence: the argument of 'std::dealloc(p)' itself is
    // the last *valid* read of 'p', not a use-after-free — same shape of
    // exemption as suppressArenaStaleCheck_ below.
    bool suppressHeapStaleCheck_ = false;
    // Suppresses the arena-staleness check in checkIdent for exactly one
    // identifier occurrence: the LHS of 'p = <new-arena-value>;' evaluates
    // 'p' as an assignment target, not a read of its old (possibly stale)
    // value — same shape of exemption as definite-init's pre-marking of
    // 'sym->initialized = true' before checking that same LHS occurrence.
    bool suppressArenaStaleCheck_ = false;
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
    bool satisfiesTrait(const TypePtr &ty, const std::string &trait) const;
    static std::string binaryOpToMethodName(BinaryOp op);

};

} // namespace safec
