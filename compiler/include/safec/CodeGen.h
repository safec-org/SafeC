#pragma once
#include "safec/AST.h"
#include "safec/Diagnostic.h"

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

#include <unordered_map>
#include <string>
#include <memory>

namespace safec {

// ── Code generation environment ───────────────────────────────────────────────
// Per-function codegen state
struct FnEnv {
    llvm::Function              *fn        = nullptr;
    FunctionDecl                *fnDecl    = nullptr;
    llvm::BasicBlock            *entry     = nullptr;
    llvm::BasicBlock            *returnBB  = nullptr;
    llvm::AllocaInst            *returnSlot= nullptr; // for non-void returns
    std::unordered_map<std::string, llvm::AllocaInst *> locals;  // name → alloca
    // Loop stack for break/continue
    std::vector<std::pair<llvm::BasicBlock *, llvm::BasicBlock *>> loopStack;
        // {breakBB, continueBB}

    void pushLoop(llvm::BasicBlock *brk, llvm::BasicBlock *cont) {
        loopStack.push_back({brk, cont});
    }
    void popLoop() { if (!loopStack.empty()) loopStack.pop_back(); }
    llvm::BasicBlock *breakBB()    const { return loopStack.empty() ? nullptr : loopStack.back().first; }
    llvm::BasicBlock *continueBB() const { return loopStack.empty() ? nullptr : loopStack.back().second; }
};

// ── LLVM Code Generator ───────────────────────────────────────────────────────
class CodeGen {
public:
    CodeGen(llvm::LLVMContext &ctx, const std::string &moduleName,
            DiagEngine &diag);

    // Generate all top-level declarations; returns the Module
    std::unique_ptr<llvm::Module> generate(TranslationUnit &tu);

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

    // ── Statement codegen ──────────────────────────────────────────────────────
    void genStmt(Stmt &s, FnEnv &env);
    void genCompound(CompoundStmt &s, FnEnv &env);
    void genIf(IfStmt &s, FnEnv &env);
    void genWhile(WhileStmt &s, FnEnv &env);
    void genFor(ForStmt &s, FnEnv &env);
    void genReturn(ReturnStmt &s, FnEnv &env);
    void genVarDecl(VarDeclStmt &s, FnEnv &env);
    void genUnsafe(UnsafeStmt &s, FnEnv &env);
    void genExprStmt(ExprStmt &s, FnEnv &env);

    // ── Expression codegen ────────────────────────────────────────────────────
    // Returns the LLVM value of the expression (always an rvalue / load result)
    llvm::Value *genExpr(Expr &e, FnEnv &env);
    // Returns the LLVM alloca/GEP representing the lvalue address (without load)
    llvm::Value *genLValue(Expr &e, FnEnv &env);

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
    llvm::Value *genMember(MemberExpr &e, FnEnv &env, bool wantAddr = false);
    llvm::Value *genCast(CastExpr &e, FnEnv &env);
    llvm::Value *genAssign(AssignExpr &e, FnEnv &env);
    llvm::Value *genTernary(TernaryExpr &e, FnEnv &env);
    llvm::Value *genAddrOf(UnaryExpr &e, FnEnv &env);
    llvm::Value *genDeref(UnaryExpr &e, FnEnv &env, bool wantAddr = false);

    // ── Arena state ───────────────────────────────────────────────────────────
    struct ArenaInfo { llvm::GlobalVariable *var; llvm::StructType *ty; int64_t cap; };
    std::unordered_map<std::string, ArenaInfo> arenaStateMap_;
    void genArenaStateGlobal(RegionDecl &rd);
    llvm::Value *genNew(NewExpr &e, FnEnv &env);

    // ── Tuple ──────────────────────────────────────────────────────────────────
    llvm::Value *genTupleLit(TupleLitExpr &e, FnEnv &env);

    // ── Closure ───────────────────────────────────────────────────────────────
    llvm::Value *genClosure(ClosureExpr &e, FnEnv &env);

    // ── Spawn ─────────────────────────────────────────────────────────────────
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

    // Struct LLVM type cache (name → StructType*)
    std::unordered_map<std::string, llvm::StructType *> structCache_;
    // Global symbol table (name → llvm::Value*)
    std::unordered_map<std::string, llvm::Value *>   globals_;
    // String literal cache
    std::unordered_map<std::string, llvm::GlobalVariable *> strLits_;
};

} // namespace safec
