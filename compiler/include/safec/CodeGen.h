#pragma once
#include "safec/AST.h"
#include "safec/Diagnostic.h"
#include <cstdint>

// LLVM headers (C++ API)
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/InlineAsm.h"

#include <unordered_map>
#include <string>
#include <memory>

namespace llvm { class TargetMachine; }

namespace safec {

enum class DebugLevel { None, Lines, Full };

// ── Code generation environment ───────────────────────────────────────────────
// Per-function codegen state
struct FnEnv {
    llvm::Function              *fn        = nullptr;
    FunctionDecl                *fnDecl    = nullptr;
    llvm::BasicBlock            *entry     = nullptr;
    llvm::BasicBlock            *returnBB  = nullptr;
    llvm::AllocaInst            *returnSlot= nullptr; // for non-void returns
    std::unordered_map<std::string, llvm::AllocaInst *> locals;  // name → alloca

    // Labeled loop stack for break/continue
    struct LoopEntry {
        std::string       label;    // empty = unlabeled
        llvm::BasicBlock *breakBB;
        llvm::BasicBlock *continueBB;
    };
    std::vector<LoopEntry> loopStack;

    void pushLoop(llvm::BasicBlock *brk, llvm::BasicBlock *cont,
                  const std::string &lbl = "") {
        loopStack.push_back({lbl, brk, cont});
    }
    void popLoop() { if (!loopStack.empty()) loopStack.pop_back(); }

    llvm::BasicBlock *breakBB(const std::string &lbl = "") const {
        if (lbl.empty())
            return loopStack.empty() ? nullptr : loopStack.back().breakBB;
        for (auto it = loopStack.rbegin(); it != loopStack.rend(); ++it)
            if (it->label == lbl) return it->breakBB;
        return nullptr;
    }
    llvm::BasicBlock *continueBB(const std::string &lbl = "") const {
        if (lbl.empty())
            return loopStack.empty() ? nullptr : loopStack.back().continueBB;
        for (auto it = loopStack.rbegin(); it != loopStack.rend(); ++it)
            if (it->label == lbl) return it->continueBB;
        return nullptr;
    }

    // Defer tracking — LIFO list of deferred stmts
    std::vector<Stmt *> deferList;
    bool                hasError = false; // set by TryExpr on null path

    // Goto/label support: forward-declared basic blocks for labels
    std::unordered_map<std::string, llvm::BasicBlock *> gotoLabels;
};

// ── LLVM Code Generator ───────────────────────────────────────────────────────
class CodeGen {
public:
    // 'targetTriple': empty (default) keeps the previous behavior of no
    // explicit triple/data-layout — clang infers its own host default at
    // link time, same as always. A non-empty triple (e.g.
    // "x86_64-unknown-linux-gnu", "riscv64-unknown-elf") sets
    // Module::setTargetTriple/setDataLayout *before* any codegen runs,
    // since sizeof/alignment/struct-layout decisions during generate()
    // read the data layout live (see e.g. genGlobalVar's getTypeAllocSize
    // calls) — setting it after generate() would be too late for those.
    // Needed for cross-target code (e.g. std::simd, which must lower to
    // genuinely different vector-register widths per ISA) to actually
    // target something other than the host.
    CodeGen(llvm::LLVMContext &ctx, const std::string &moduleName,
            DiagEngine &diag, DebugLevel dbgLevel = DebugLevel::None,
            const std::string &targetTriple = "");
    // Declared here, defined in CodeGen.cpp: std::unique_ptr<TargetMachine>'s
    // destructor needs TargetMachine's complete definition, which only
    // CodeGen.cpp includes — everywhere else (e.g. main.cpp) only sees the
    // forward declaration above, so an implicitly-generated ~CodeGen() there
    // would fail to compile.
    ~CodeGen();

    void setFreestanding(bool v) { freestanding_ = v; }

    // Generate all top-level declarations; returns the Module
    std::unique_ptr<llvm::Module> generate(TranslationUnit &tu);

    // Non-null only when constructed with an explicit targetTriple (see
    // constructor comment). The optimization pipeline in main.cpp uses this
    // (when present) so target-aware analyses (TargetTransformInfo etc.)
    // make correct cost decisions for the requested cross-target triple
    // instead of silently falling back to host-CPU assumptions.
    llvm::TargetMachine *getTargetMachine() const { return targetMachine_.get(); }

private:
    // ── Type lowering ──────────────────────────────────────────────────────────
    // SafeC type → LLVM type
    llvm::Type *lowerType(const TypePtr &ty);
    llvm::Type *lowerPrimType(const PrimType &ty);
    llvm::StructType *lowerStructType(const StructType &ty);
    // References lower to ptr + attributes (no runtime metadata)
    llvm::PointerType *lowerRefType(const ReferenceType &ty);
    // Annotate a call/store with noalias, nonnull, dereferenceable per ref type
    void addRefAttrs(llvm::AttributeList &attrs, unsigned argIdx,
                     const ReferenceType &rt, const TypePtr &base,
                     llvm::LLVMContext &ctx);

    // ── Top-level declarations ─────────────────────────────────────────────────
    void genDecl(Decl &d);
    llvm::Function   *genFunctionProto(FunctionDecl &fn);
    void              genFunctionBody(FunctionDecl &fn, llvm::Function *llvmFn);
    llvm::GlobalVariable *genGlobalVar(GlobalVarDecl &gv);
    void              genStaticAssert(StaticAssertDecl &sa);

    // Fold a global-variable initializer expression into an LLVM constant.
    // Handles int/float literals, negation of a literal, references to other
    // globals (e.g. a static array's address), and struct/array aggregate
    // initializers (recursively). Returns nullptr if the expression isn't a
    // constant this evaluator understands, so the caller can fall back to
    // zero-initializing (with a diagnostic) rather than silently dropping
    // the initializer's actual value.
    llvm::Constant *evalConstInit(Expr &e, llvm::Type *expectedTy);

    // ── Statement codegen ──────────────────────────────────────────────────────
    void genStmt(Stmt &s, FnEnv &env);
    void collectGotoLabels(Stmt &s, FnEnv &env); // pre-create BBs for goto labels
    void genCompound(CompoundStmt &s, FnEnv &env);
    void genIf(IfStmt &s, FnEnv &env);
    void genWhile(WhileStmt &s, FnEnv &env);
    void genFor(ForStmt &s, FnEnv &env);
    void genReturn(ReturnStmt &s, FnEnv &env);
    void genVarDecl(VarDeclStmt &s, FnEnv &env);
    // Coerce a scalar int/float value to exactly match targetTy (same-kind
    // width mismatches only — int<->int via zext/trunc, float<->float via
    // fpext/fptrunc; no-op if already matching or not a scalar pair). Used
    // everywhere Sema permits a literal to narrow into a smaller declared
    // type (e.g. 'unsigned char x = 200;', 'float x = 1.0;') — Sema only
    // grants *permission*; without this, the store below would write the
    // literal's full original width (e.g. i32) into a narrower alloca
    // (e.g. i8), overrunning it by however many bytes short the true type
    // is — a real stack-corruption bug, not merely invalid-looking IR.
    // 'srcType' (optional) is the source value's SafeC-level type, used
    // only to pick sign- vs zero-extend when *widening* an integer — get
    // this wrong and widening a negative signed value (e.g. int -5 -> long
    // long) zero-extends instead of sign-extends, silently turning -5 into
    // 4294967291. Defaults to signed (matches plain 'int'/'long' — the
    // common case) when the source type isn't available/known.
    llvm::Value *coerceScalar(llvm::Value *val, llvm::Type *targetTy,
                               const TypePtr &srcType = nullptr);
    // Wraps 'val' into Optional's '{T, i1}' representation when 'targetTy'
    // is '?T' and 'val' isn't already shaped that way — Sema::
    // canImplicitlyConvert permits 'return someTValue;' / 'return null;'
    // from a '?T'-returning function (mirroring '&T' → '?&T' already being
    // implicit), a plain T value, or literal 'null', to flow anywhere a
    // '?T' is expected — but *permission* isn't the same as the actual
    // struct construction: without this, a plain T store into a '{T, i1}'
    // slot would leave the has_value bit uninitialized. No-op for every
    // non-Optional target, and a no-op when 'srcType' is already the same
    // Optional (e.g. forwarding an existing '?T' value unchanged).
    llvm::Value *coerceToOptional(llvm::Value *val, const TypePtr &srcType,
                                   const TypePtr &targetTy);
    // x.is_null() / x.is_none() / x.default(fallback) — see CallExpr::NullOp
    // in AST.h and the "safe pseudo-methods" section of Sema::checkCall.
    llvm::Value *genNullOp(CallExpr &e, FnEnv &env);
    void genUnsafe(UnsafeStmt &s, FnEnv &env);
    void genExprStmt(ExprStmt &s, FnEnv &env);
    void genMatch(MatchStmt &s, FnEnv &env);
    void genAsm(AsmStmt &s, FnEnv &env);
    void emitDefers(FnEnv &env, bool errOnly = false);

    // ── Expression codegen ────────────────────────────────────────────────────
    // Returns the LLVM value of the expression (always an rvalue / load result)
    llvm::Value *genExpr(Expr &e, FnEnv &env);
    // Returns the LLVM alloca/GEP representing the lvalue address (without load)
    llvm::Value *genLValue(Expr &e, FnEnv &env);

    // Array-to-pointer/reference decay: GEP to the address of element 0.
    // Used wherever an array-typed expression flows into a context expecting
    // a pointer or reference (call arguments, var-decl initializers,
    // assignment) — an array is an aggregate, not a pointer value, so a
    // plain genExpr()+store would try to copy the whole array (and mismatch
    // types under opaque pointers) instead of taking its address.
    llvm::Value *decayArrayToPtr(Expr &a, FnEnv &env);

    llvm::Value *genIntLit(IntLitExpr &e);
    llvm::Value *genFloatLit(FloatLitExpr &e);
    llvm::Value *genBoolLit(BoolLitExpr &e);
    llvm::Value *genStringLit(StringLitExpr &e);
    llvm::Value *genCharLit(CharLitExpr &e);
    llvm::Value *genIdent(IdentExpr &e, FnEnv &env, bool wantAddr = false);
    llvm::Value *genUnary(UnaryExpr &e, FnEnv &env);
    llvm::Value *genBinary(BinaryExpr &e, FnEnv &env);
    llvm::Value *genCall(CallExpr &e, FnEnv &env);
    llvm::Value *genSubscript(SubscriptExpr &e, FnEnv &env, bool wantAddr = false);
    llvm::Value *genSlice(SliceExpr &e, FnEnv &env);
    llvm::Value *genMember(MemberExpr &e, FnEnv &env, bool wantAddr = false);
    // Resolves e.base's type (through pointer/reference) to a struct and
    // looks up e.field by name — used by genMember's read path and
    // genAssign's write path to detect bitfield members (FieldDecl::bitWidth
    // >= 0) without duplicating the base-type-unwrapping logic twice.
    // 'outPath' receives the GEP index chain (see StructType::findFieldPath) —
    // a single index for a direct field, more for one reached through an
    // anonymous struct/union member.
    const FieldDecl *findMemberFieldDecl(MemberExpr &e, std::vector<int> &outPath);
    llvm::Value *genCast(CastExpr &e, FnEnv &env);
    llvm::Value *genAssign(AssignExpr &e, FnEnv &env);
    llvm::Value *genTernary(TernaryExpr &e, FnEnv &env);
    llvm::Value *genMatchExpr(MatchExpr &e, FnEnv &env);
    llvm::Value *genFnEval(FnEvalExpr &e);
    llvm::Value *genAddrOf(UnaryExpr &e, FnEnv &env);
    llvm::Value *genDeref(UnaryExpr &e, FnEnv &env, bool wantAddr = false);

    // ── Arena state ───────────────────────────────────────────────────────────
    struct ArenaInfo { llvm::GlobalVariable *var; llvm::StructType *ty; int64_t cap; };
    std::unordered_map<std::string, ArenaInfo> arenaStateMap_;
    void genArenaStateGlobal(RegionDecl &rd);
    llvm::Value *genNew(NewExpr &e, FnEnv &env);

    // ── Tuple ──────────────────────────────────────────────────────────────────
    llvm::Value *genTupleLit(TupleLitExpr &e, FnEnv &env);

    // ── Spawn / thread backends ────────────────────────────────────────────────
    // spawn/join/spawn_scoped no longer hardcode pthread unconditionally —
    // selectThreadBackend() picks one based on the compile target (see
    // CodeGen.cpp for the full rationale):
    //   - Pthread: default hosted POSIX targets (macOS/Linux/BSD), and the
    //     no-'--target' host-default case — unchanged from before this
    //     dispatch existed.
    //   - Win32: an explicit Windows target triple — CreateThread /
    //     WaitForSingleObject, matching std/thread.sc's own wrapper.
    //   - Hook: freestanding/bare-metal, or any target with no recognized
    //     OS thread API (wasm32-*, a vendor's own unrecognized triple,
    //     etc.) — emits calls to a small documented contract
    //     (__safec_thread_create/__safec_thread_join) that ANY runtime can
    //     satisfy without compiler changes: std/sync/bare_spawn.sc
    //     implements it via the existing cooperative TaskScheduler for
    //     bare-metal ('SafeC thread' with no OS underneath), and the same
    //     two symbols are the extension point for a WASM JS/Worker shim or
    //     a vendor/third-party thread library.
    enum class ThreadBackend { Pthread, Win32, Hook };
    ThreadBackend selectThreadBackend() const;
    llvm::Value *genThreadCreate(llvm::Value *fnVal, llvm::Value *argVal, FnEnv &env);
    void         genThreadJoin(llvm::Value *handleVal, FnEnv &env);
    llvm::Value *genSpawn(SpawnExpr &e, FnEnv &env);

    // Helpers
    llvm::AllocaInst *createEntryAlloca(FnEnv &env, llvm::Type *ty,
                                         const std::string &name);
    llvm::Value *boolify(llvm::Value *v, const TypePtr &ty);  // produce i1
    llvm::Value *applyBinaryOp(BinaryOp op, llvm::Value *l, llvm::Value *r,
                                const TypePtr &ty);

    // Whether a terminated BB still needs a branch (fallthrough guard)
    bool isTerminated(llvm::BasicBlock *bb) const {
        return bb && !bb->empty() && bb->back().isTerminator();
    }

    // ── State ─────────────────────────────────────────────────────────────────
    llvm::LLVMContext                                &ctx_;
    std::unique_ptr<llvm::Module>                    mod_;
    llvm::IRBuilder<>                                builder_;
    DiagEngine                                      &diag_;
    // Only set when constructed with a non-empty targetTriple (cross-target
    // codegen — see the constructor comment). Owns the TargetMachine so its
    // DataLayout (already installed on mod_) stays valid for the module's
    // lifetime; unused otherwise.
    std::unique_ptr<llvm::TargetMachine>             targetMachine_;

    // Struct LLVM type cache (name → StructType*)
    std::unordered_map<std::string, llvm::StructType *> structCache_;
    // Global symbol table (name → llvm::Value*)
    std::unordered_map<std::string, llvm::Value *>   globals_;
    // String literal cache
    std::unordered_map<std::string, llvm::GlobalVariable *> strLits_;

    // ── Freestanding mode ─────────────────────────────────────────────────────
    bool freestanding_ = false;

    // ── Debug info ────────────────────────────────────────────────────────────
    DebugLevel                           debugLevel_ = DebugLevel::None;
    std::unique_ptr<llvm::DIBuilder>     dib_;
    llvm::DICompileUnit                 *diCU_   = nullptr;
    llvm::DIFile                        *diFile_ = nullptr;

    llvm::DIType *diTypeFor(const TypePtr &ty);
    llvm::DISubroutineType *diSubroutineType(FunctionDecl &fn);
};

} // namespace safec
