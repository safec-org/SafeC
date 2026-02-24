#include "safec/CodeGen.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace safec {

// ── Constructor ───────────────────────────────────────────────────────────────
CodeGen::CodeGen(llvm::LLVMContext &ctx, const std::string &moduleName,
                 DiagEngine &diag)
    : ctx_(ctx),
      mod_(std::make_unique<llvm::Module>(moduleName, ctx)),
      builder_(ctx),
      diag_(diag) {}

// ── Top-level entry ───────────────────────────────────────────────────────────
std::unique_ptr<llvm::Module> CodeGen::generate(TranslationUnit &tu) {
    // Pass 1: generate all prototypes and global variables
    for (auto &d : tu.decls) {
        if (d->kind == DeclKind::Function) {
            genFunctionProto(static_cast<FunctionDecl &>(*d));
        } else if (d->kind == DeclKind::GlobalVar) {
            genGlobalVar(static_cast<GlobalVarDecl &>(*d));
        } else if (d->kind == DeclKind::Struct) {
            // Ensure struct type is in the cache
            auto &sd = static_cast<StructDecl &>(*d);
            if (sd.type) lowerStructType(*sd.type);
        }
    }

    // Pass 2: generate function bodies
    for (auto &d : tu.decls) {
        if (d->kind == DeclKind::Function) {
            auto &fn = static_cast<FunctionDecl &>(*d);
            if (fn.body) {
                auto *llvmFn = mod_->getFunction(fn.name);
                if (llvmFn) genFunctionBody(fn, llvmFn);
            }
        } else if (d->kind == DeclKind::StaticAssert) {
            genStaticAssert(static_cast<StaticAssertDecl &>(*d));
        }
    }

    // Verify
    std::string errStr;
    llvm::raw_string_ostream errOS(errStr);
    if (llvm::verifyModule(*mod_, &errOS)) {
        diag_.error({}, "LLVM module verification failed: " + errStr);
    }
    return std::move(mod_);
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPE LOWERING
// ─────────────────────────────────────────────────────────────────────────────

llvm::Type *CodeGen::lowerType(const TypePtr &ty) {
    if (!ty) return llvm::Type::getVoidTy(ctx_);
    switch (ty->kind) {
    case TypeKind::Void:    return llvm::Type::getVoidTy(ctx_);
    case TypeKind::Bool:    return llvm::Type::getInt1Ty(ctx_);
    case TypeKind::Char:    return llvm::Type::getInt8Ty(ctx_);
    case TypeKind::Int8:    return llvm::Type::getInt8Ty(ctx_);
    case TypeKind::Int16:   return llvm::Type::getInt16Ty(ctx_);
    case TypeKind::Int32:   return llvm::Type::getInt32Ty(ctx_);
    case TypeKind::Int64:   return llvm::Type::getInt64Ty(ctx_);
    case TypeKind::UInt8:   return llvm::Type::getInt8Ty(ctx_);
    case TypeKind::UInt16:  return llvm::Type::getInt16Ty(ctx_);
    case TypeKind::UInt32:  return llvm::Type::getInt32Ty(ctx_);
    case TypeKind::UInt64:  return llvm::Type::getInt64Ty(ctx_);
    case TypeKind::Float32: return llvm::Type::getFloatTy(ctx_);
    case TypeKind::Float64: return llvm::Type::getDoubleTy(ctx_);
    case TypeKind::Pointer:
        return llvm::PointerType::get(ctx_, 0); // opaque pointer (LLVM 15+)
    case TypeKind::Reference:
        // References are raw pointers at IR level; safety is compile-time only
        return llvm::PointerType::get(ctx_, 0);
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*ty);
        auto *lt = lowerStructType(st);
        return lt;
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(*ty);
        auto *elemTy = lowerType(at.element);
        if (at.size < 0) return llvm::PointerType::get(ctx_, 0);
        return llvm::ArrayType::get(elemTy, static_cast<uint64_t>(at.size));
    }
    case TypeKind::Enum:
        return llvm::Type::getInt32Ty(ctx_); // enums are i32
    case TypeKind::Function: {
        auto &ft = static_cast<const FunctionType &>(*ty);
        std::vector<llvm::Type *> paramTys;
        for (auto &p : ft.paramTypes) paramTys.push_back(lowerType(p));
        auto *retTy = lowerType(ft.returnType);
        return llvm::FunctionType::get(retTy, paramTys, ft.variadic);
    }
    case TypeKind::Error:
    default:
        return llvm::Type::getInt32Ty(ctx_); // fallback
    }
}

llvm::StructType *CodeGen::lowerStructType(const StructType &st) {
    auto it = structCache_.find(st.name);
    if (it != structCache_.end()) return it->second;

    // Create opaque struct first (handles recursive types)
    auto *llvmSt = llvm::StructType::create(ctx_, st.name);
    structCache_[st.name] = llvmSt;

    // Fill in fields
    if (st.isDefined) {
        std::vector<llvm::Type *> fieldTys;
        for (auto &f : st.fields) fieldTys.push_back(lowerType(f.type));
        llvmSt->setBody(fieldTys, /*isPacked=*/false);
    }
    return llvmSt;
}

llvm::PointerType *CodeGen::lowerRefType(const ReferenceType &) {
    // All references → opaque pointer at IR level
    return llvm::PointerType::get(ctx_, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION PROTOTYPE
// ─────────────────────────────────────────────────────────────────────────────

llvm::Function *CodeGen::genFunctionProto(FunctionDecl &fn) {
    // Build LLVM function type
    std::vector<llvm::Type *> paramTys;
    for (auto &p : fn.params) paramTys.push_back(lowerType(p.type));
    auto *retTy  = fn.returnType ? lowerType(fn.returnType) : llvm::Type::getVoidTy(ctx_);
    auto *fnType = llvm::FunctionType::get(retTy, paramTys, fn.isVariadic);

    auto linkage = fn.isExtern
        ? llvm::Function::ExternalLinkage
        : llvm::Function::ExternalLinkage;  // default external for C ABI
    auto *llvmFn = llvm::Function::Create(fnType, linkage, fn.name, *mod_);

    if (fn.isInline) llvmFn->addFnAttr(llvm::Attribute::InlineHint);

    // Name parameters and add reference attributes
    unsigned idx = 0;
    for (auto &arg : llvmFn->args()) {
        if (idx < fn.params.size()) {
            arg.setName(fn.params[idx].name);
            // If param type is a non-null reference, add nonnull attr
            auto &pt = fn.params[idx].type;
            if (pt && pt->kind == TypeKind::Reference) {
                auto &rt = static_cast<const ReferenceType &>(*pt);
                if (!rt.nullable) {
                    llvmFn->addParamAttr(idx, llvm::Attribute::NonNull);
                }
                // For safe references: add noalias if mutable (exclusive access)
                if (rt.mut) {
                    llvmFn->addParamAttr(idx, llvm::Attribute::NoAlias);
                }
                // dereferenceable: we know the base type size statically
                // (simplified — would need DataLayout for exact size)
            }
        }
        ++idx;
    }

    globals_[fn.name] = llvmFn;
    return llvmFn;
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION BODY
// ─────────────────────────────────────────────────────────────────────────────

llvm::AllocaInst *CodeGen::createEntryAlloca(FnEnv &env, llvm::Type *ty,
                                               const std::string &name) {
    // Place all allocas in the function entry block for mem2reg
    llvm::IRBuilder<> tmpBuilder(&env.fn->getEntryBlock(),
                                  env.fn->getEntryBlock().begin());
    return tmpBuilder.CreateAlloca(ty, nullptr, name);
}

void CodeGen::genFunctionBody(FunctionDecl &fn, llvm::Function *llvmFn) {
    FnEnv env;
    env.fn     = llvmFn;
    env.fnDecl = &fn;

    // Create entry basic block
    auto *entryBB = llvm::BasicBlock::Create(ctx_, "entry", llvmFn);
    builder_.SetInsertPoint(entryBB);
    env.entry = entryBB;

    // Create return BB (single exit point pattern — cleaner IR)
    env.returnBB = llvm::BasicBlock::Create(ctx_, "return", llvmFn);

    // Allocate return value slot if non-void
    auto *retTy = lowerType(fn.returnType);
    if (!retTy->isVoidTy()) {
        env.returnSlot = createEntryAlloca(env, retTy, "retval");
    }

    // Materialize parameters into allocas
    unsigned idx = 0;
    for (auto &arg : llvmFn->args()) {
        if (idx >= fn.params.size()) break;
        auto &p    = fn.params[idx];
        auto *aTy  = lowerType(p.type);
        auto *alloca = createEntryAlloca(env, aTy, p.name + ".addr");
        builder_.CreateStore(&arg, alloca);
        env.locals[p.name] = alloca;
        ++idx;
    }

    // Generate body
    genCompound(*fn.body, env);

    // Fallthrough to return block if not already terminated
    if (!isTerminated(builder_.GetInsertBlock()))
        builder_.CreateBr(env.returnBB);

    // Emit return block
    builder_.SetInsertPoint(env.returnBB);
    if (env.returnSlot) {
        auto *retVal = builder_.CreateLoad(retTy, env.returnSlot, "ret");
        builder_.CreateRet(retVal);
    } else {
        builder_.CreateRetVoid();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL VARIABLES
// ─────────────────────────────────────────────────────────────────────────────

llvm::GlobalVariable *CodeGen::genGlobalVar(GlobalVarDecl &gv) {
    auto *ty = lowerType(gv.type);
    llvm::Constant *init = llvm::Constant::getNullValue(ty);

    if (gv.init && !ty->isVoidTy()) {
        // Compile-time constant initializer only (integer/float literals)
        if (gv.init->kind == ExprKind::IntLit) {
            int64_t v = static_cast<IntLitExpr &>(*gv.init).value;
            init = llvm::ConstantInt::get(ty, v, /*isSigned=*/true);
        } else if (gv.init->kind == ExprKind::FloatLit) {
            double v = static_cast<FloatLitExpr &>(*gv.init).value;
            init = llvm::ConstantFP::get(ty, v);
        }
        // Other initializers would need the consteval engine
    }

    auto *gvar = new llvm::GlobalVariable(
        *mod_, ty, gv.isConst,
        gv.isExtern ? llvm::GlobalValue::ExternalLinkage
                    : llvm::GlobalValue::InternalLinkage,
        gv.isExtern ? nullptr : init,
        gv.name);

    globals_[gv.name] = gvar;
    return gvar;
}

// ─────────────────────────────────────────────────────────────────────────────
// STATIC ASSERT  (compile-time evaluation — simplified)
// ─────────────────────────────────────────────────────────────────────────────

void CodeGen::genStaticAssert(StaticAssertDecl &sa) {
    // Full consteval engine is future work.
    // For now: evaluate integer literal conditions.
    if (!sa.cond) return;
    if (sa.cond->kind == ExprKind::IntLit) {
        if (static_cast<IntLitExpr &>(*sa.cond).value == 0) {
            diag_.error(sa.loc, "static_assert failed" +
                        (sa.message.empty() ? "" : ": " + sa.message));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STATEMENTS
// ─────────────────────────────────────────────────────────────────────────────

void CodeGen::genStmt(Stmt &s, FnEnv &env) {
    switch (s.kind) {
    case StmtKind::Compound:     genCompound(static_cast<CompoundStmt &>(s), env);   break;
    case StmtKind::Expr:         genExprStmt(static_cast<ExprStmt &>(s), env);       break;
    case StmtKind::If:           genIf(static_cast<IfStmt &>(s), env);               break;
    case StmtKind::While:
    case StmtKind::DoWhile:      genWhile(static_cast<WhileStmt &>(s), env);         break;
    case StmtKind::For:          genFor(static_cast<ForStmt &>(s), env);             break;
    case StmtKind::Return:       genReturn(static_cast<ReturnStmt &>(s), env);       break;
    case StmtKind::VarDecl:      genVarDecl(static_cast<VarDeclStmt &>(s), env);    break;
    case StmtKind::Unsafe:       genUnsafe(static_cast<UnsafeStmt &>(s), env);      break;
    case StmtKind::Break:
        if (env.breakBB()) builder_.CreateBr(env.breakBB());
        break;
    case StmtKind::Continue:
        if (env.continueBB()) builder_.CreateBr(env.continueBB());
        break;
    case StmtKind::StaticAssert: break; // handled at compile-time by sema
    case StmtKind::Label: {
        auto &ls = static_cast<LabelStmt &>(s);
        // Create BB for the label
        auto *bb = llvm::BasicBlock::Create(ctx_, ls.label, env.fn);
        if (!isTerminated(builder_.GetInsertBlock()))
            builder_.CreateBr(bb);
        builder_.SetInsertPoint(bb);
        genStmt(*ls.body, env);
        break;
    }
    case StmtKind::Goto: {
        // Goto is complex with SSA; simplified: emit unreachable
        diag_.warn(s.loc, "goto not fully implemented in codegen");
        break;
    }
    default: break;
    }
}

void CodeGen::genCompound(CompoundStmt &s, FnEnv &env) {
    for (auto &stmt : s.body) genStmt(*stmt, env);
}

void CodeGen::genExprStmt(ExprStmt &s, FnEnv &env) {
    if (s.expr) genExpr(*s.expr, env);
}

void CodeGen::genIf(IfStmt &s, FnEnv &env) {
    auto *condVal = genExpr(*s.cond, env);
    condVal = boolify(condVal, s.cond->type);

    auto *thenBB  = llvm::BasicBlock::Create(ctx_, "if.then", env.fn);
    auto *mergeBB = llvm::BasicBlock::Create(ctx_, "if.end",  env.fn);
    llvm::BasicBlock *elseBB = s.else_
        ? llvm::BasicBlock::Create(ctx_, "if.else", env.fn)
        : mergeBB;

    builder_.CreateCondBr(condVal, thenBB, elseBB);

    builder_.SetInsertPoint(thenBB);
    genStmt(*s.then, env);
    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(mergeBB);

    if (s.else_) {
        builder_.SetInsertPoint(elseBB);
        genStmt(*s.else_, env);
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(mergeBB);
    }

    builder_.SetInsertPoint(mergeBB);
}

void CodeGen::genWhile(WhileStmt &s, FnEnv &env) {
    auto *headerBB = llvm::BasicBlock::Create(ctx_, s.isDoWhile ? "do.cond" : "while.cond", env.fn);
    auto *bodyBB   = llvm::BasicBlock::Create(ctx_, "while.body", env.fn);
    auto *exitBB   = llvm::BasicBlock::Create(ctx_, "while.end",  env.fn);

    env.pushLoop(exitBB, headerBB);

    if (s.isDoWhile) {
        // do { body } while (cond)
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(bodyBB);
        builder_.SetInsertPoint(bodyBB);
        genStmt(*s.body, env);
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(headerBB);

        builder_.SetInsertPoint(headerBB);
        auto *condVal = genExpr(*s.cond, env);
        condVal = boolify(condVal, s.cond->type);
        builder_.CreateCondBr(condVal, bodyBB, exitBB);
    } else {
        // while (cond) { body }
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(headerBB);
        builder_.SetInsertPoint(headerBB);
        auto *condVal = genExpr(*s.cond, env);
        condVal = boolify(condVal, s.cond->type);
        builder_.CreateCondBr(condVal, bodyBB, exitBB);

        builder_.SetInsertPoint(bodyBB);
        genStmt(*s.body, env);
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(headerBB);
    }

    env.popLoop();
    builder_.SetInsertPoint(exitBB);
}

void CodeGen::genFor(ForStmt &s, FnEnv &env) {
    // for (init; cond; incr) body
    if (s.init) genStmt(*s.init, env);

    auto *headerBB = llvm::BasicBlock::Create(ctx_, "for.cond", env.fn);
    auto *bodyBB   = llvm::BasicBlock::Create(ctx_, "for.body", env.fn);
    auto *incrBB   = llvm::BasicBlock::Create(ctx_, "for.incr", env.fn);
    auto *exitBB   = llvm::BasicBlock::Create(ctx_, "for.end",  env.fn);

    env.pushLoop(exitBB, incrBB);
    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(headerBB);

    builder_.SetInsertPoint(headerBB);
    if (s.cond) {
        auto *condVal = genExpr(*s.cond, env);
        condVal = boolify(condVal, s.cond->type);
        builder_.CreateCondBr(condVal, bodyBB, exitBB);
    } else {
        builder_.CreateBr(bodyBB); // infinite loop
    }

    builder_.SetInsertPoint(bodyBB);
    genStmt(*s.body, env);
    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(incrBB);

    builder_.SetInsertPoint(incrBB);
    if (s.incr) genExpr(*s.incr, env);
    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(headerBB);

    env.popLoop();
    builder_.SetInsertPoint(exitBB);
}

void CodeGen::genReturn(ReturnStmt &s, FnEnv &env) {
    if (s.value && env.returnSlot) {
        auto *val = genExpr(*s.value, env);
        builder_.CreateStore(val, env.returnSlot);
    }
    builder_.CreateBr(env.returnBB);
}

void CodeGen::genVarDecl(VarDeclStmt &s, FnEnv &env) {
    auto *ty     = lowerType(s.resolvedType ? s.resolvedType : s.declType);
    auto *alloca = createEntryAlloca(env, ty, s.name);
    env.locals[s.name] = alloca;

    if (s.init) {
        auto *val = genExpr(*s.init, env);
        builder_.CreateStore(val, alloca);
    } else {
        // Zero-initialize (SafeC no-UB policy)
        builder_.CreateStore(llvm::Constant::getNullValue(ty), alloca);
    }
}

void CodeGen::genUnsafe(UnsafeStmt &s, FnEnv &env) {
    // Unsafe blocks don't produce special IR — safety checks are compile-time only
    genCompound(static_cast<CompoundStmt &>(*s.body), env);
}

// ─────────────────────────────────────────────────────────────────────────────
// EXPRESSIONS
// ─────────────────────────────────────────────────────────────────────────────

llvm::Value *CodeGen::genExpr(Expr &e, FnEnv &env) {
    switch (e.kind) {
    case ExprKind::IntLit:    return genIntLit(static_cast<IntLitExpr &>(e));
    case ExprKind::FloatLit:  return genFloatLit(static_cast<FloatLitExpr &>(e));
    case ExprKind::BoolLit:   return genBoolLit(static_cast<BoolLitExpr &>(e));
    case ExprKind::StringLit: return genStringLit(static_cast<StringLitExpr &>(e));
    case ExprKind::CharLit:   return genCharLit(static_cast<CharLitExpr &>(e));
    case ExprKind::NullLit:
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
    case ExprKind::Ident:     return genIdent(static_cast<IdentExpr &>(e), env);
    case ExprKind::Unary:     return genUnary(static_cast<UnaryExpr &>(e), env);
    case ExprKind::AddrOf:    return genAddrOf(static_cast<UnaryExpr &>(e), env);
    case ExprKind::Deref:     return genDeref(static_cast<UnaryExpr &>(e), env);
    case ExprKind::Binary:    return genBinary(static_cast<BinaryExpr &>(e), env);
    case ExprKind::Ternary:   return genTernary(static_cast<TernaryExpr &>(e), env);
    case ExprKind::Call:      return genCall(static_cast<CallExpr &>(e), env);
    case ExprKind::Subscript: return genSubscript(static_cast<SubscriptExpr &>(e), env);
    case ExprKind::Member:
    case ExprKind::Arrow:     return genMember(static_cast<MemberExpr &>(e), env);
    case ExprKind::Cast:      return genCast(static_cast<CastExpr &>(e), env);
    case ExprKind::Assign:    return genAssign(static_cast<AssignExpr &>(e), env);
    case ExprKind::SizeofType: {
        auto &st = static_cast<SizeofTypeExpr &>(e);
        auto *llvmTy = lowerType(st.ofType);
        // Use DataLayout-based size
        const auto &dl = mod_->getDataLayout();
        uint64_t sz = dl.getTypeAllocSize(llvmTy);
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), sz);
    }
    case ExprKind::Compound: {
        // Compound initializer: generate as struct/array aggregate (simplified)
        auto &ci = static_cast<CompoundInitExpr &>(e);
        auto *ty = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
        if (ci.inits.empty()) return llvm::Constant::getNullValue(ty);
        // For struct types: store into a temp alloca
        if (ty->isStructTy()) {
            auto *alloca = createEntryAlloca(env, ty, "compound.init");
            unsigned i = 0;
            for (auto &init : ci.inits) {
                auto *val  = genExpr(*init, env);
                auto *gep  = builder_.CreateStructGEP(ty, alloca, i++);
                builder_.CreateStore(val, gep);
            }
            return builder_.CreateLoad(ty, alloca, "compound");
        }
        // Fallback: first element
        return genExpr(*ci.inits[0], env);
    }
    default:
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }
}

llvm::Value *CodeGen::genLValue(Expr &e, FnEnv &env) {
    // Return the address of an lvalue expression
    switch (e.kind) {
    case ExprKind::Ident:
        return genIdent(static_cast<IdentExpr &>(e), env, /*wantAddr=*/true);
    case ExprKind::Subscript:
        return genSubscript(static_cast<SubscriptExpr &>(e), env, /*wantAddr=*/true);
    case ExprKind::Member:
    case ExprKind::Arrow:
        return genMember(static_cast<MemberExpr &>(e), env, /*wantAddr=*/true);
    case ExprKind::Deref:
        return genDeref(static_cast<UnaryExpr &>(e), env, /*wantAddr=*/true);
    default:
        diag_.error(e.loc, "cannot take address of this expression");
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
    }
}

// ── Literals ──────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genIntLit(IntLitExpr &e) {
    auto *ty = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
    if (!ty->isIntegerTy()) ty = llvm::Type::getInt32Ty(ctx_);
    return llvm::ConstantInt::get(static_cast<llvm::IntegerType *>(ty), e.value, true);
}

llvm::Value *CodeGen::genFloatLit(FloatLitExpr &e) {
    auto *ty = e.type ? lowerType(e.type) : llvm::Type::getDoubleTy(ctx_);
    return llvm::ConstantFP::get(ty, e.value);
}

llvm::Value *CodeGen::genBoolLit(BoolLitExpr &e) {
    return llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx_), e.value ? 1 : 0);
}

llvm::Value *CodeGen::genStringLit(StringLitExpr &e) {
    auto it = strLits_.find(e.value);
    if (it != strLits_.end()) return it->second;

    // Create a global byte array for the string
    auto *arrTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(ctx_), e.value.size() + 1);
    std::vector<llvm::Constant *> chars;
    for (unsigned char c : e.value)
        chars.push_back(llvm::ConstantInt::get(llvm::Type::getInt8Ty(ctx_), c));
    chars.push_back(llvm::ConstantInt::get(llvm::Type::getInt8Ty(ctx_), 0)); // NUL

    auto *init = llvm::ConstantArray::get(arrTy, chars);
    auto *gv   = new llvm::GlobalVariable(*mod_, arrTy, /*isConst=*/true,
                                           llvm::GlobalValue::PrivateLinkage,
                                           init, ".str");
    gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    strLits_[e.value] = gv;

    // Return pointer to first element (GEP i8*, 0, 0)
    return builder_.CreateConstInBoundsGEP2_64(arrTy, gv, 0, 0, "str");
}

llvm::Value *CodeGen::genCharLit(CharLitExpr &e) {
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(ctx_),
                                   static_cast<unsigned char>(e.value));
}

// ── Identifier ────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genIdent(IdentExpr &e, FnEnv &env, bool wantAddr) {
    // Check local variables first
    auto it = env.locals.find(e.name);
    if (it != env.locals.end()) {
        if (wantAddr) return it->second;
        auto *ty = it->second->getAllocatedType();
        return builder_.CreateLoad(ty, it->second, e.name);
    }
    // Global / function
    auto git = globals_.find(e.name);
    if (git != globals_.end()) {
        auto *gv = git->second;
        if (auto *fn = llvm::dyn_cast<llvm::Function>(gv)) return fn;
        if (wantAddr) return gv;
        if (auto *ggv = llvm::dyn_cast<llvm::GlobalVariable>(gv)) {
            return builder_.CreateLoad(ggv->getValueType(), ggv, e.name);
        }
        return gv;
    }
    // Try module lookup (handles external function declarations)
    if (auto *fn = mod_->getFunction(e.name)) return fn;
    if (auto *gv = mod_->getGlobalVariable(e.name)) {
        if (wantAddr) return gv;
        return builder_.CreateLoad(gv->getValueType(), gv, e.name);
    }
    diag_.error(e.loc, "codegen: undefined identifier '" + e.name + "'");
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
}

// ── Unary ─────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genUnary(UnaryExpr &e, FnEnv &env) {
    switch (e.op) {
    case UnaryOp::Neg: {
        auto *val = genExpr(*e.operand, env);
        if (val->getType()->isFloatingPointTy()) return builder_.CreateFNeg(val, "fneg");
        return builder_.CreateNeg(val, "neg");
    }
    case UnaryOp::Not: {
        auto *val = genExpr(*e.operand, env);
        val = boolify(val, e.operand->type);
        return builder_.CreateNot(val, "not");
    }
    case UnaryOp::BitNot: {
        auto *val = genExpr(*e.operand, env);
        return builder_.CreateNot(val, "bitnot");
    }
    case UnaryOp::PreInc: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty   = static_cast<llvm::AllocaInst *>(addr)->getAllocatedType();
        auto *val  = builder_.CreateLoad(ty, addr);
        auto *one  = llvm::ConstantInt::get(ty, 1);
        auto *inc  = builder_.CreateAdd(val, one, "inc");
        builder_.CreateStore(inc, addr);
        return inc;
    }
    case UnaryOp::PreDec: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty   = static_cast<llvm::AllocaInst *>(addr)->getAllocatedType();
        auto *val  = builder_.CreateLoad(ty, addr);
        auto *one  = llvm::ConstantInt::get(ty, 1);
        auto *dec  = builder_.CreateSub(val, one, "dec");
        builder_.CreateStore(dec, addr);
        return dec;
    }
    case UnaryOp::PostInc: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty   = static_cast<llvm::AllocaInst *>(addr)->getAllocatedType();
        auto *old  = builder_.CreateLoad(ty, addr, "old");
        auto *one  = llvm::ConstantInt::get(ty, 1);
        builder_.CreateStore(builder_.CreateAdd(old, one), addr);
        return old;
    }
    case UnaryOp::PostDec: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty   = static_cast<llvm::AllocaInst *>(addr)->getAllocatedType();
        auto *old  = builder_.CreateLoad(ty, addr, "old");
        auto *one  = llvm::ConstantInt::get(ty, 1);
        builder_.CreateStore(builder_.CreateSub(old, one), addr);
        return old;
    }
    case UnaryOp::SizeofExpr: {
        // Evaluate type, not value
        auto *ty = e.operand->type ? lowerType(e.operand->type)
                                   : llvm::Type::getInt32Ty(ctx_);
        const auto &dl = mod_->getDataLayout();
        uint64_t sz = dl.getTypeAllocSize(ty);
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), sz);
    }
    default:
        return genExpr(*e.operand, env);
    }
}

llvm::Value *CodeGen::genAddrOf(UnaryExpr &e, FnEnv &env) {
    // Address-of: return pointer to lvalue
    return genLValue(*e.operand, env);
}

llvm::Value *CodeGen::genDeref(UnaryExpr &e, FnEnv &env, bool wantAddr) {
    auto *ptr = genExpr(*e.operand, env);
    if (wantAddr) return ptr; // already a pointer
    auto *baseTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
    return builder_.CreateLoad(baseTy, ptr, "deref");
}

// ── Binary ────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::applyBinaryOp(BinaryOp op, llvm::Value *l, llvm::Value *r,
                                     const TypePtr &ty) {
    bool isFloat = ty && ty->isFloat();
    bool isSigned = !ty || (ty->kind == TypeKind::Int32 || ty->kind == TypeKind::Int64 ||
                             ty->kind == TypeKind::Int16 || ty->kind == TypeKind::Int8);
    switch (op) {
    case BinaryOp::Add:  return isFloat ? builder_.CreateFAdd(l, r, "fadd")
                                        : builder_.CreateAdd(l, r, "add");
    case BinaryOp::Sub:  return isFloat ? builder_.CreateFSub(l, r, "fsub")
                                        : builder_.CreateSub(l, r, "sub");
    case BinaryOp::Mul:  return isFloat ? builder_.CreateFMul(l, r, "fmul")
                                        : builder_.CreateMul(l, r, "mul");
    case BinaryOp::Div:  return isFloat ? builder_.CreateFDiv(l, r, "fdiv")
                                        : (isSigned ? builder_.CreateSDiv(l, r, "sdiv")
                                                    : builder_.CreateUDiv(l, r, "udiv"));
    case BinaryOp::Mod:  return isSigned ? builder_.CreateSRem(l, r, "srem")
                                         : builder_.CreateURem(l, r, "urem");
    case BinaryOp::BitAnd: return builder_.CreateAnd(l, r, "and");
    case BinaryOp::BitOr:  return builder_.CreateOr(l, r, "or");
    case BinaryOp::BitXor: return builder_.CreateXor(l, r, "xor");
    case BinaryOp::Shl:    return builder_.CreateShl(l, r, "shl");
    case BinaryOp::Shr:    return isSigned ? builder_.CreateAShr(l, r, "ashr")
                                           : builder_.CreateLShr(l, r, "lshr");
    case BinaryOp::LogAnd: return builder_.CreateAnd(l, r, "land");
    case BinaryOp::LogOr:  return builder_.CreateOr(l, r, "lor");
    // Comparisons
    case BinaryOp::Eq:
        return isFloat ? builder_.CreateFCmpOEQ(l, r, "feq")
                       : builder_.CreateICmpEQ(l, r, "eq");
    case BinaryOp::NEq:
        return isFloat ? builder_.CreateFCmpONE(l, r, "fne")
                       : builder_.CreateICmpNE(l, r, "ne");
    case BinaryOp::Lt:
        return isFloat ? builder_.CreateFCmpOLT(l, r, "flt")
                       : (isSigned ? builder_.CreateICmpSLT(l, r, "slt")
                                   : builder_.CreateICmpULT(l, r, "ult"));
    case BinaryOp::Gt:
        return isFloat ? builder_.CreateFCmpOGT(l, r, "fgt")
                       : (isSigned ? builder_.CreateICmpSGT(l, r, "sgt")
                                   : builder_.CreateICmpUGT(l, r, "ugt"));
    case BinaryOp::LEq:
        return isFloat ? builder_.CreateFCmpOLE(l, r, "fle")
                       : (isSigned ? builder_.CreateICmpSLE(l, r, "sle")
                                   : builder_.CreateICmpULE(l, r, "ule"));
    case BinaryOp::GEq:
        return isFloat ? builder_.CreateFCmpOGE(l, r, "fge")
                       : (isSigned ? builder_.CreateICmpSGE(l, r, "sge")
                                   : builder_.CreateICmpUGE(l, r, "uge"));
    case BinaryOp::Comma: return r;
    default: return l;
    }
}

llvm::Value *CodeGen::genBinary(BinaryExpr &e, FnEnv &env) {
    // Short-circuit evaluation for && and ||
    if (e.op == BinaryOp::LogAnd) {
        auto *lhsBB   = builder_.GetInsertBlock();
        auto *rhsBB   = llvm::BasicBlock::Create(ctx_, "land.rhs", env.fn);
        auto *mergeBB = llvm::BasicBlock::Create(ctx_, "land.end", env.fn);

        auto *lhsVal = genExpr(*e.left, env);
        lhsVal = boolify(lhsVal, e.left->type);
        builder_.CreateCondBr(lhsVal, rhsBB, mergeBB);
        lhsBB = builder_.GetInsertBlock();

        builder_.SetInsertPoint(rhsBB);
        auto *rhsVal = genExpr(*e.right, env);
        rhsVal = boolify(rhsVal, e.right->type);
        builder_.CreateBr(mergeBB);
        auto *rhsEndBB = builder_.GetInsertBlock();

        builder_.SetInsertPoint(mergeBB);
        auto *phi = builder_.CreatePHI(llvm::Type::getInt1Ty(ctx_), 2, "land");
        phi->addIncoming(llvm::ConstantInt::getFalse(ctx_), lhsBB);
        phi->addIncoming(rhsVal, rhsEndBB);
        return phi;
    }
    if (e.op == BinaryOp::LogOr) {
        auto *lhsBB   = builder_.GetInsertBlock();
        auto *rhsBB   = llvm::BasicBlock::Create(ctx_, "lor.rhs", env.fn);
        auto *mergeBB = llvm::BasicBlock::Create(ctx_, "lor.end", env.fn);

        auto *lhsVal = genExpr(*e.left, env);
        lhsVal = boolify(lhsVal, e.left->type);
        builder_.CreateCondBr(lhsVal, mergeBB, rhsBB);
        lhsBB = builder_.GetInsertBlock();

        builder_.SetInsertPoint(rhsBB);
        auto *rhsVal = genExpr(*e.right, env);
        rhsVal = boolify(rhsVal, e.right->type);
        builder_.CreateBr(mergeBB);
        auto *rhsEndBB = builder_.GetInsertBlock();

        builder_.SetInsertPoint(mergeBB);
        auto *phi = builder_.CreatePHI(llvm::Type::getInt1Ty(ctx_), 2, "lor");
        phi->addIncoming(llvm::ConstantInt::getTrue(ctx_), lhsBB);
        phi->addIncoming(rhsVal, rhsEndBB);
        return phi;
    }

    auto *lhs = genExpr(*e.left,  env);
    auto *rhs = genExpr(*e.right, env);

    // Pointer arithmetic (e.g., &arr[i+offset])
    if (lhs->getType()->isPointerTy() && (e.op == BinaryOp::Add || e.op == BinaryOp::Sub)) {
        auto *elemTy = llvm::Type::getInt8Ty(ctx_); // byte offset
        if (e.op == BinaryOp::Sub) rhs = builder_.CreateNeg(rhs);
        return builder_.CreateGEP(elemTy, lhs, rhs, "ptrarith");
    }

    return applyBinaryOp(e.op, lhs, rhs, e.left->type);
}

// ── Call ──────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genCall(CallExpr &e, FnEnv &env) {
    auto *calleeVal = genExpr(*e.callee, env);

    std::vector<llvm::Value *> args;
    for (auto &a : e.args) args.push_back(genExpr(*a, env));

    llvm::FunctionType *ft = nullptr;
    if (auto *fn = llvm::dyn_cast<llvm::Function>(calleeVal)) {
        ft = fn->getFunctionType();
        auto *call = builder_.CreateCall(ft, calleeVal, args);
        return call;
    }
    // Indirect call via function pointer
    if (e.callee->type && e.callee->type->kind == TypeKind::Function) {
        auto *rawFT = static_cast<llvm::FunctionType *>(lowerType(e.callee->type));
        auto *call  = builder_.CreateCall(rawFT, calleeVal, args);
        return call;
    }
    diag_.error(e.loc, "codegen: cannot determine callee function type");
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
}

// ── Subscript ─────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genSubscript(SubscriptExpr &e, FnEnv &env, bool wantAddr) {
    auto *idxVal  = genExpr(*e.index, env);
    auto *baseAddr = genLValue(*e.base, env);
    auto *baseTy  = e.base->type ? lowerType(e.base->type)
                                 : llvm::Type::getInt32Ty(ctx_);

    llvm::Value *gep = nullptr;
    if (baseTy->isArrayTy()) {
        auto *arrTy = static_cast<llvm::ArrayType *>(baseTy);
        gep = builder_.CreateInBoundsGEP(arrTy, baseAddr,
                {llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0), idxVal}, "arr.idx");
        if (wantAddr) return gep;
        return builder_.CreateLoad(arrTy->getElementType(), gep, "arr.elem");
    }
    // Pointer subscript
    auto *elemTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
    gep = builder_.CreateGEP(elemTy, baseAddr, idxVal, "ptr.idx");
    if (wantAddr) return gep;
    return builder_.CreateLoad(elemTy, gep, "ptr.elem");
}

// ── Member access ─────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genMember(MemberExpr &e, FnEnv &env, bool wantAddr) {
    llvm::Value *baseAddr;
    llvm::Type  *structTy;

    // Resolve the struct type through any pointer/reference indirection
    auto resolveStructTy = [&](TypePtr ty) -> llvm::Type * {
        if (!ty) return llvm::Type::getInt32Ty(ctx_);
        if (ty->kind == TypeKind::Pointer)
            return lowerType(static_cast<PointerType &>(*ty).base);
        if (ty->kind == TypeKind::Reference)
            return lowerType(static_cast<ReferenceType &>(*ty).base);
        return lowerType(ty);
    };

    if (e.isArrow) {
        // p->field: base is a pointer/reference; load the pointer value
        baseAddr = genExpr(*e.base, env);
        structTy = resolveStructTy(e.base->type);
    } else if (e.base->type && e.base->type->kind == TypeKind::Reference) {
        // p.field where p is a safe reference: auto-deref.
        // Load the pointer value held in the variable, then GEP into struct.
        auto &rt = static_cast<ReferenceType &>(*e.base->type);
        baseAddr = genExpr(*e.base, env);  // loads the ptr from the alloca
        structTy = lowerType(rt.base);
    } else {
        // s.field: base is a struct value; use its address directly
        baseAddr = genLValue(*e.base, env);
        structTy = e.base->type ? lowerType(e.base->type) : llvm::Type::getInt32Ty(ctx_);
    }

    // Find field index
    unsigned fieldIdx = 0;
    if (e.base->type) {
        TypePtr bty = e.base->type;
        if (bty->kind == TypeKind::Pointer)
            bty = static_cast<PointerType &>(*bty).base;
        if (bty->kind == TypeKind::Reference)
            bty = static_cast<ReferenceType &>(*bty).base;
        if (bty->kind == TypeKind::Struct) {
            auto &st = static_cast<StructType &>(*bty);
            for (auto &f : st.fields) {
                if (f.name == e.field) { fieldIdx = f.index; break; }
            }
        }
    }

    if (!structTy->isStructTy()) {
        diag_.error(e.loc, "codegen: member access on non-struct");
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }

    auto *gep = builder_.CreateStructGEP(structTy, baseAddr, fieldIdx, e.field);
    if (wantAddr) return gep;
    auto *fieldTy = static_cast<llvm::StructType *>(structTy)->getElementType(fieldIdx);
    return builder_.CreateLoad(fieldTy, gep, e.field);
}

// ── Cast ──────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genCast(CastExpr &e, FnEnv &env) {
    auto *srcVal = genExpr(*e.operand, env);
    auto *dstTy  = lowerType(e.targetType);
    auto *srcTy  = srcVal->getType();

    if (srcTy == dstTy) return srcVal;

    // Integer ↔ Integer
    if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
        unsigned srcBits = srcTy->getIntegerBitWidth();
        unsigned dstBits = dstTy->getIntegerBitWidth();
        if (dstBits > srcBits) {
            bool isSigned = e.operand->type && e.operand->type->isInteger() &&
                            static_cast<PrimType &>(*e.operand->type).isSigned();
            return isSigned ? builder_.CreateSExt(srcVal, dstTy, "sext")
                            : builder_.CreateZExt(srcVal, dstTy, "zext");
        }
        return builder_.CreateTrunc(srcVal, dstTy, "trunc");
    }
    // Integer ↔ Float
    if (srcTy->isIntegerTy() && dstTy->isFloatingPointTy()) {
        bool isSigned = e.operand->type && e.operand->type->isInteger() &&
                        static_cast<PrimType &>(*e.operand->type).isSigned();
        return isSigned ? builder_.CreateSIToFP(srcVal, dstTy, "sitofp")
                        : builder_.CreateUIToFP(srcVal, dstTy, "uitofp");
    }
    if (srcTy->isFloatingPointTy() && dstTy->isIntegerTy()) {
        bool isSigned = e.targetType && e.targetType->isInteger() &&
                        static_cast<PrimType &>(*e.targetType).isSigned();
        return isSigned ? builder_.CreateFPToSI(srcVal, dstTy, "fptosi")
                        : builder_.CreateFPToUI(srcVal, dstTy, "fptoui");
    }
    // Float ↔ Float
    if (srcTy->isFloatingPointTy() && dstTy->isFloatingPointTy()) {
        return builder_.CreateFPCast(srcVal, dstTy, "fpcast");
    }
    // Pointer ↔ Pointer (bitcast — opaque pointers are already the same type)
    if (srcTy->isPointerTy() && dstTy->isPointerTy()) return srcVal;
    // Integer ↔ Pointer
    if (srcTy->isIntegerTy() && dstTy->isPointerTy())
        return builder_.CreateIntToPtr(srcVal, dstTy, "inttoptr");
    if (srcTy->isPointerTy() && dstTy->isIntegerTy())
        return builder_.CreatePtrToInt(srcVal, dstTy, "ptrtoint");

    diag_.warn(e.loc, "codegen: unsupported cast, using bitcast");
    return builder_.CreateBitOrPointerCast(srcVal, dstTy, "bitcast");
}

// ── Assignment ────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genAssign(AssignExpr &e, FnEnv &env) {
    auto *rhs   = genExpr(*e.rhs, env);
    auto *addr  = genLValue(*e.lhs, env);
    auto *lhsTy = e.lhs->type ? lowerType(e.lhs->type) : rhs->getType();

    llvm::Value *val = rhs;
    if (e.op != AssignOp::Assign) {
        // Compound assignment: load LHS, apply op, store
        auto *cur = builder_.CreateLoad(lhsTy, addr, "cur");
        BinaryOp binOp;
        switch (e.op) {
        case AssignOp::AddAssign: binOp = BinaryOp::Add; break;
        case AssignOp::SubAssign: binOp = BinaryOp::Sub; break;
        case AssignOp::MulAssign: binOp = BinaryOp::Mul; break;
        case AssignOp::DivAssign: binOp = BinaryOp::Div; break;
        case AssignOp::ModAssign: binOp = BinaryOp::Mod; break;
        case AssignOp::AndAssign: binOp = BinaryOp::BitAnd; break;
        case AssignOp::OrAssign:  binOp = BinaryOp::BitOr;  break;
        case AssignOp::XorAssign: binOp = BinaryOp::BitXor; break;
        case AssignOp::ShlAssign: binOp = BinaryOp::Shl;    break;
        case AssignOp::ShrAssign: binOp = BinaryOp::Shr;    break;
        default:                  binOp = BinaryOp::Add;    break;
        }
        val = applyBinaryOp(binOp, cur, rhs, e.lhs->type);
    }

    // Type coercion for mismatched integer sizes (e.g., int literal → i8)
    if (val->getType() != lhsTy) {
        if (lhsTy->isIntegerTy() && val->getType()->isIntegerTy()) {
            unsigned dstBits = lhsTy->getIntegerBitWidth();
            unsigned srcBits = val->getType()->getIntegerBitWidth();
            if (dstBits < srcBits) val = builder_.CreateTrunc(val, lhsTy);
            else if (dstBits > srcBits) val = builder_.CreateSExt(val, lhsTy);
        }
    }

    builder_.CreateStore(val, addr);
    return val;
}

// ── Ternary ───────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genTernary(TernaryExpr &e, FnEnv &env) {
    auto *condVal = genExpr(*e.cond, env);
    condVal = boolify(condVal, e.cond->type);

    auto *thenBB  = llvm::BasicBlock::Create(ctx_, "tern.then", env.fn);
    auto *elseBB  = llvm::BasicBlock::Create(ctx_, "tern.else", env.fn);
    auto *mergeBB = llvm::BasicBlock::Create(ctx_, "tern.end",  env.fn);

    builder_.CreateCondBr(condVal, thenBB, elseBB);

    builder_.SetInsertPoint(thenBB);
    auto *thenVal = genExpr(*e.then, env);
    auto *thenEnd = builder_.GetInsertBlock();
    builder_.CreateBr(mergeBB);

    builder_.SetInsertPoint(elseBB);
    auto *elseVal = genExpr(*e.else_, env);
    auto *elseEnd = builder_.GetInsertBlock();
    builder_.CreateBr(mergeBB);

    builder_.SetInsertPoint(mergeBB);
    auto *ty  = thenVal->getType();
    auto *phi = builder_.CreatePHI(ty, 2, "tern");
    phi->addIncoming(thenVal, thenEnd);
    phi->addIncoming(elseVal, elseEnd);
    return phi;
}

// ── Helper: boolify ───────────────────────────────────────────────────────────
llvm::Value *CodeGen::boolify(llvm::Value *v, const TypePtr &ty) {
    if (v->getType()->isIntegerTy(1)) return v;
    if (v->getType()->isIntegerTy())
        return builder_.CreateICmpNE(v,
            llvm::ConstantInt::get(v->getType(), 0), "tobool");
    if (v->getType()->isFloatingPointTy())
        return builder_.CreateFCmpONE(v,
            llvm::ConstantFP::get(v->getType(), 0.0), "tobool");
    if (v->getType()->isPointerTy())
        return builder_.CreateICmpNE(
            builder_.CreatePtrToInt(v, llvm::Type::getInt64Ty(ctx_)),
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0), "ptrtobool");
    return v;
}

} // namespace safec
