// Post-codegen ABI lowering for large aggregate return values.
//
// CodeGen emits every struct/array/tuple-returning function as a first-class
// LLVM aggregate return ('define %T @f()' / 'call %T @f()'). That is fine for
// small values, but has two real problems for large ones:
//
//   1. It crashes LLVM. A first-class aggregate return of e.g. std::test's
//      'struct TestSuite' (~85 KB — 256 TestCase entries, each with inline
//      char arrays) makes SelectionDAG materialize one node per scalar leaf
//      on both sides of the call; clang 21's X86 DAGCombiner segfaults in
//      ReplaceAllUsesWith on '@main' for every program that calls
//      TestSuite::new(). Every 'safeguard test' target hit this.
//   2. It is not the C ABI. SysV x86-64 and AAPCS64 return aggregates over
//      16 bytes through a caller-provided buffer (the 'sret' pointer, in
//      %rdi / x8), Win64 does so for anything that isn't 1/2/4/8 bytes. A
//      raw LLVM aggregate return instead spreads the value over however
//      many registers the backend picks, so a C caller of a SafeC function
//      returning e.g. 'struct String' (24 bytes) — or SafeC calling a C
//      function that returns one — read garbage.
//
// This pass rewrites, module-wide and after all bodies exist:
//   * every function (definition or declaration) whose return type is a
//     "memory-class" aggregate into 'void @f(ptr sret(%T) noalias, ...)';
//   * every direct and indirect call with such a return type into a call
//     that passes a caller-allocated buffer;
//   * a 'ret %T (load %T, ptr %p)' into a memcpy into the sret buffer, and a
//     call result whose only use is 'store %T %r, ptr %q' into a memcpy, so
//     the huge value never exists as an SSA aggregate at all;
//   * any remaining 'store (load %T, ptr %p), ptr %q' of a large aggregate
//     (struct assignment / by-value copy) into a memcpy, for the same
//     DAG-size reason.
// Every SafeC translation unit runs the same rule for the same target, so
// SafeC-to-SafeC calls across separately compiled objects stay consistent.

#include "safec/AbiLowering.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <iterator>
#include <vector>

namespace safec {

namespace {

// Aggregates at or above this size are copied with memcpy instead of a
// first-class load/store pair (mirrors clang, which never emits aggregate
// load/store at all; 64 keeps small structs in SSA where SROA likes them).
constexpr uint64_t kLargeCopyBytes = 64;

bool isHFA(llvm::Type *ty) {
    auto *st = llvm::dyn_cast<llvm::StructType>(ty);
    if (!st) return false;
    unsigned n = st->getNumElements();
    if (n == 0 || n > 4) return false;
    llvm::Type *first = st->getElementType(0);
    if (!first->isFloatTy() && !first->isDoubleTy()) return false;
    for (unsigned i = 1; i < n; ++i)
        if (st->getElementType(i) != first) return false;
    return true;
}

struct Rules {
    const llvm::DataLayout &dl;
    bool windows;
    bool aarch64;
    bool arm32;     // AAPCS (ARM/Thumb state)
    bool x86_32;    // i386 System V
    bool riscv;
    unsigned xlenBytes;

    // Does a value of this type come back through an sret buffer under the
    // target's C ABI?
    //   Win64:         unless 1/2/4/8 bytes
    //   i386 SysV:     always (every struct/union is returned in memory)
    //   AAPCS32:       composites larger than 4 bytes
    //   RISC-V:        aggregates larger than 2*XLEN (8 on RV32, 16 on RV64)
    //   AArch64:       larger than 16 bytes, except HFAs (v0-v3)
    //   x86-64 SysV:   larger than 16 bytes
    bool returnsInMemory(llvm::Type *ty) const {
        if (!ty->isStructTy() && !ty->isArrayTy()) return false;
        if (!ty->isSized()) return false;
        uint64_t sz = dl.getTypeAllocSize(ty);
        if (windows) return !(sz == 1 || sz == 2 || sz == 4 || sz == 8);
        if (x86_32) return true;
        if (arm32) return sz > 4;
        if (riscv) return sz > 2 * (uint64_t)xlenBytes;
        if (aarch64 && isHFA(ty)) return false;  // returned in v0-v3
        return sz > 16;
    }

    bool isLargeCopy(llvm::Type *ty) const {
        return (ty->isStructTy() || ty->isArrayTy()) && ty->isSized() &&
               dl.getTypeAllocSize(ty) >= kLargeCopyBytes;
    }
};

llvm::AllocaInst *entryAlloca(llvm::Function *fn, llvm::Type *ty, const char *name) {
    llvm::IRBuilder<> b(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    return b.CreateAlloca(ty, nullptr, name);
}

void emitCopy(llvm::IRBuilder<> &b, const llvm::DataLayout &dl,
              llvm::Value *dst, llvm::Value *src, llvm::Type *ty) {
    llvm::Align al = dl.getABITypeAlign(ty);
    b.CreateMemCpy(dst, al, src, al, dl.getTypeAllocSize(ty));
}

llvm::FunctionType *sretType(llvm::FunctionType *ft) {
    std::vector<llvm::Type *> params;
    params.reserve(ft->getNumParams() + 1);
    params.push_back(llvm::PointerType::get(ft->getContext(), 0));
    for (auto *p : ft->params()) params.push_back(p);
    return llvm::FunctionType::get(llvm::Type::getVoidTy(ft->getContext()),
                                   params, ft->isVarArg());
}

// Replace one aggregate-returning call with an sret call. 'newCallee' is the
// rewritten function for direct calls, or the original pointer operand for
// indirect ones.
void rewriteCall(llvm::CallInst *ci, llvm::Value *newCallee, const Rules &r) {
    llvm::Type *retTy = ci->getType();
    llvm::Function *caller = ci->getFunction();
    llvm::FunctionType *newFT = sretType(ci->getFunctionType());

    // If the result's only use is a store, write straight through a temp
    // and memcpy into the destination (the destination can't be passed as
    // the sret buffer itself: it may alias an argument).
    llvm::StoreInst *soleStore = nullptr;
    if (ci->hasOneUse())
        if (auto *st = llvm::dyn_cast<llvm::StoreInst>(*ci->user_begin()))
            if (st->getValueOperand() == ci) soleStore = st;

    llvm::AllocaInst *tmp = entryAlloca(caller, retTy, "sret.tmp");
    llvm::IRBuilder<> b(ci);
    std::vector<llvm::Value *> args;
    args.reserve(ci->arg_size() + 1);
    args.push_back(tmp);
    for (auto &a : ci->args()) args.push_back(a.get());

    llvm::CallInst *nc = b.CreateCall(newFT, newCallee, args);
    nc->setCallingConv(ci->getCallingConv());
    nc->setDebugLoc(ci->getDebugLoc());
    nc->setTailCallKind(llvm::CallInst::TCK_None);
    // Shift any existing parameter attributes over by one.
    llvm::AttributeList old = ci->getAttributes();
    std::vector<llvm::AttributeSet> pattrs;
    pattrs.push_back(llvm::AttributeSet::get(
        ci->getContext(), {llvm::Attribute::getWithStructRetType(ci->getContext(), retTy)}));
    for (unsigned i = 0; i < ci->arg_size(); ++i) pattrs.push_back(old.getParamAttrs(i));
    nc->setAttributes(llvm::AttributeList::get(ci->getContext(), old.getFnAttrs(),
                                               llvm::AttributeSet(), pattrs));

    if (soleStore) {
        llvm::IRBuilder<> sb(soleStore);
        emitCopy(sb, r.dl, soleStore->getPointerOperand(), tmp, retTy);
        soleStore->eraseFromParent();
    } else if (!ci->use_empty()) {
        llvm::Value *v = b.CreateLoad(retTy, tmp, "sret.val");
        ci->replaceAllUsesWith(v);
    }
    ci->eraseFromParent();
}

llvm::Function *rewriteFunction(llvm::Function *f, const Rules &r) {
    llvm::Type *retTy = f->getReturnType();
    llvm::FunctionType *newFT = sretType(f->getFunctionType());
    llvm::Module *m = f->getParent();
    std::string name = f->getName().str();
    f->setName(name + ".pre_sret");

    llvm::Function *nf = llvm::Function::Create(newFT, f->getLinkage(),
                                                f->getAddressSpace(), name, m);
    nf->copyAttributesFrom(f);
    nf->setCallingConv(f->getCallingConv());
    nf->setSubprogram(f->getSubprogram());
    f->setSubprogram(nullptr);

    llvm::LLVMContext &ctx = f->getContext();
    llvm::AttributeList old = f->getAttributes();
    std::vector<llvm::AttributeSet> pattrs;
    pattrs.push_back(llvm::AttributeSet::get(ctx, {
        llvm::Attribute::getWithStructRetType(ctx, retTy),
        llvm::Attribute::get(ctx, llvm::Attribute::NoAlias)}));
    for (unsigned i = 0; i < f->arg_size(); ++i) pattrs.push_back(old.getParamAttrs(i));
    nf->setAttributes(llvm::AttributeList::get(ctx, old.getFnAttrs(),
                                               llvm::AttributeSet(), pattrs));

    llvm::Argument *sret = nf->getArg(0);
    sret->setName("agg.result");

    if (!f->isDeclaration()) {
        nf->splice(nf->begin(), f);
        auto nit = nf->arg_begin() + 1;
        for (auto &a : f->args()) {
            nit->takeName(&a);
            a.replaceAllUsesWith(&*nit);
            ++nit;
        }
        std::vector<llvm::ReturnInst *> rets;
        for (auto &bb : *nf)
            if (auto *ri = bb.empty() ? nullptr : llvm::dyn_cast<llvm::ReturnInst>(&bb.back()))
                rets.push_back(ri);
        for (auto *ri : rets) {
            llvm::IRBuilder<> b(ri);
            llvm::Value *v = ri->getReturnValue();
            auto *ld = llvm::dyn_cast_or_null<llvm::LoadInst>(v);
            if (ld && ld->getType() == retTy) {
                // Copy at the load's own position so it reads exactly the
                // memory state the load did (deferred statements may run
                // between the load and the 'ret').
                llvm::IRBuilder<> lb(ld);
                emitCopy(lb, r.dl, sret, ld->getPointerOperand(), retTy);
                if (ld->hasOneUse()) { ri->setOperand(0, llvm::PoisonValue::get(retTy)); ld->eraseFromParent(); }
            } else if (v) {
                b.CreateStore(v, sret);
            }
            auto *nr = b.CreateRetVoid();
            nr->setDebugLoc(ri->getDebugLoc());
            ri->eraseFromParent();
        }
    }

    // Direct call sites first (their callee operand is 'f' itself) …
    std::vector<llvm::CallInst *> calls;
    for (llvm::User *u : f->users())
        if (auto *ci = llvm::dyn_cast<llvm::CallInst>(u))
            if (ci->getCalledOperand() == f) calls.push_back(ci);
    for (auto *ci : calls) rewriteCall(ci, nf, r);
    // … then every other use (address taken, stored into a vtable-like
    // struct of fn pointers, …): those reach the new signature through the
    // indirect-call rewrite below.
    f->replaceAllUsesWith(nf);
    f->eraseFromParent();
    return nf;
}

} // namespace

void lowerAggregateABI(llvm::Module &m) {
    llvm::Triple triple(m.getTargetTriple());
    if (triple.getTriple().empty())
        triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    Rules r{m.getDataLayout(), triple.isOSWindows(), triple.isAArch64(),
            triple.isARM() || triple.isThumb(),
            triple.getArch() == llvm::Triple::x86 && !triple.isOSWindows() && !triple.isOSDarwin(),
            triple.isRISCV(), triple.isArch64Bit() ? 8u : 4u};

    // 1. Functions with memory-class return types.
    std::vector<llvm::Function *> fns;
    for (auto &f : m)
        if (!f.isIntrinsic() && r.returnsInMemory(f.getReturnType()))
            fns.push_back(&f);
    for (auto *f : fns) rewriteFunction(f, r);

    // 2. Remaining (indirect) calls whose declared type returns in memory.
    std::vector<llvm::CallInst *> indirect;
    for (auto &f : m)
        for (auto &bb : f)
            for (auto &inst : bb)
                if (auto *ci = llvm::dyn_cast<llvm::CallInst>(&inst))
                    if (!ci->getCalledFunction() && !ci->isInlineAsm() &&
                        r.returnsInMemory(ci->getType()))
                        indirect.push_back(ci);
    for (auto *ci : indirect) rewriteCall(ci, ci->getCalledOperand(), r);

    // 3. Large aggregate copies: store (load p), q  →  memcpy(q, p).
    std::vector<llvm::StoreInst *> copies;
    for (auto &f : m)
        for (auto &bb : f)
            for (auto &inst : bb)
                if (auto *st = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    if (auto *ld = llvm::dyn_cast<llvm::LoadInst>(st->getValueOperand()))
                        if (!st->isVolatile() && !ld->isVolatile() &&
                            ld->hasOneUse() && ld->getParent() == st->getParent() &&
                            r.isLargeCopy(ld->getType()))
                            copies.push_back(st);
    // 4. Large constant aggregate stores ('T x = {0}' / zero-init of a big
    //    local): zero → memset, anything else → memcpy from a private
    //    constant global.
    std::vector<llvm::StoreInst *> constStores;
    for (auto &f : m)
        for (auto &bb : f)
            for (auto &inst : bb)
                if (auto *st = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    if (llvm::isa<llvm::Constant>(st->getValueOperand()) && !st->isVolatile() &&
                        r.isLargeCopy(st->getValueOperand()->getType()))
                        constStores.push_back(st);
    for (auto *st : constStores) {
        auto *c = llvm::cast<llvm::Constant>(st->getValueOperand());
        llvm::Type *ty = c->getType();
        llvm::Align al = r.dl.getABITypeAlign(ty);
        llvm::IRBuilder<> b(st);
        if (c->isNullValue() || llvm::isa<llvm::UndefValue>(c)) {
            if (!llvm::isa<llvm::UndefValue>(c))
                b.CreateMemSet(st->getPointerOperand(), b.getInt8(0),
                               r.dl.getTypeAllocSize(ty), al);
        } else {
            auto *gv = new llvm::GlobalVariable(m, ty, /*isConstant=*/true,
                                                llvm::GlobalValue::PrivateLinkage, c,
                                                ".agg.init");
            gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
            gv->setAlignment(al);
            emitCopy(b, r.dl, st->getPointerOperand(), gv, ty);
        }
        st->eraseFromParent();
    }

    for (auto *st : copies) {
        auto *ld = llvm::cast<llvm::LoadInst>(st->getValueOperand());
        // The load must still observe the same memory at the store: only
        // fold when nothing between them can write memory.
        bool clobbered = false;
        for (auto it = std::next(ld->getIterator()); &*it != st; ++it)
            if (it->mayWriteToMemory()) { clobbered = true; break; }
        if (clobbered) continue;
        llvm::IRBuilder<> b(st);
        emitCopy(b, r.dl, st->getPointerOperand(), ld->getPointerOperand(), ld->getType());
        st->eraseFromParent();
        ld->eraseFromParent();
    }
}

} // namespace safec
