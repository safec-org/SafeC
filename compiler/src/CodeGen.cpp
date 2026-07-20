#include "safec/CodeGen.h"
#include <cstdint>
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <cassert>
#include <cctype>

namespace safec {

// ── Constructor ───────────────────────────────────────────────────────────────
CodeGen::CodeGen(llvm::LLVMContext &ctx, const std::string &moduleName,
                 DiagEngine &diag, DebugLevel dbgLevel,
                 const std::string &targetTriple)
    : ctx_(ctx),
      mod_(std::make_unique<llvm::Module>(moduleName, ctx)),
      builder_(ctx),
      diag_(diag),
      debugLevel_(dbgLevel)
{
    if (!targetTriple.empty()) {
        llvm::Triple triple(targetTriple);
        std::string err;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
        if (!target) {
            diag_.error({}, "unknown target triple '" + targetTriple + "': " + err);
        } else {
            llvm::TargetOptions opts;
            auto *tm = target->createTargetMachine(
                triple, /*CPU=*/"generic", /*Features=*/"", opts, llvm::Reloc::PIC_);
            mod_->setTargetTriple(triple);
            mod_->setDataLayout(tm->createDataLayout());
            targetMachine_.reset(tm);
        }
    }
    if (debugLevel_ != DebugLevel::None) {
        dib_ = std::make_unique<llvm::DIBuilder>(*mod_);
        llvm::SmallString<256> absPath(moduleName);
        llvm::sys::fs::make_absolute(absPath);
        llvm::StringRef dir  = llvm::sys::path::parent_path(absPath);
        llvm::StringRef file = llvm::sys::path::filename(absPath);
        diFile_ = dib_->createFile(file, dir);
        diCU_   = dib_->createCompileUnit(
            llvm::dwarf::DW_LANG_C,
            diFile_,
            "SafeC",
            /*isOptimized=*/false, "", 0);
    }
}

CodeGen::~CodeGen() = default;

// ── Top-level entry ───────────────────────────────────────────────────────────
std::unique_ptr<llvm::Module> CodeGen::generate(TranslationUnit &tu) {
    // Pass 1: generate all prototypes and global variables
    for (auto &d : tu.decls) {
        if (d->kind == DeclKind::Function) {
            auto &fn = static_cast<FunctionDecl &>(*d);
            if (!fn.genericParams.empty() || fn.isDeferredGenericMethod) continue; // skip uninstantiated templates
            genFunctionProto(fn);
        } else if (d->kind == DeclKind::GlobalVar) {
            genGlobalVar(static_cast<GlobalVarDecl &>(*d));
        } else if (d->kind == DeclKind::Struct) {
            // Ensure struct type is in the cache
            auto &sd = static_cast<StructDecl &>(*d);
            if (sd.type) lowerStructType(*sd.type);
        } else if (d->kind == DeclKind::Region) {
            genArenaStateGlobal(static_cast<RegionDecl &>(*d));
        } else if (d->kind == DeclKind::Enum) {
            // Register enum constants as named globals (constant ints)
            // Always use at least i32 to avoid ABI issues with variadic promotion
            auto &ed = static_cast<EnumDecl &>(*d);
            if (ed.type) {
                unsigned bw = ed.type->bitWidth < 32 ? 32 : ed.type->bitWidth;
                auto *intTy = llvm::Type::getIntNTy(ctx_, bw);
                for (auto &[name, val] : ed.type->enumerators) {
                    auto *gv = new llvm::GlobalVariable(
                        *mod_, intTy, /*isConstant=*/true,
                        llvm::GlobalValue::InternalLinkage,
                        llvm::ConstantInt::get(intTy, val, ed.type->isSigned),
                        name);
                    globals_[name] = gv;
                }
            }
        }
    }

    // Pass 2: generate function bodies
    for (auto &d : tu.decls) {
        if (d->kind == DeclKind::Function) {
            auto &fn = static_cast<FunctionDecl &>(*d);
            if (!fn.genericParams.empty() || fn.isDeferredGenericMethod) continue; // skip uninstantiated templates
            // consteval functions are compile-time-only by construction —
            // Sema::checkFunctionForConstevalCalls already rejects every
            // call site that isn't itself a const-eval context (another
            // const/consteval body, a global initializer, static_assert,
            // or an array-size expression), and ConstEvalEngine interprets
            // the AST directly for all of those, never through the LLVM
            // function this would emit. Skipping the body here means a
            // consteval function's dead runtime codegen can't crash or
            // bloat a build for a path that will never execute — 'full
            // compile-time execution... to reduce overhead' includes not
            // spending overhead on code that's already guaranteed to have
            // run entirely at compile time.
            if (fn.body && !fn.isConsteval) {
                auto *llvmFn = mod_->getFunction(fn.name);
                if (llvmFn) genFunctionBody(fn, llvmFn);
            }
        } else if (d->kind == DeclKind::StaticAssert) {
            genStaticAssert(static_cast<StaticAssertDecl &>(*d));
        }
    }

    // Finalize debug info (must be before verifyModule)
    if (dib_) dib_->finalize();

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
        // size < 0: unsized array in parameter/local/global position decays
        // to a bare pointer (the C ABI rule). size == 0 is reserved
        // specifically for a struct's flexible array member (see
        // Sema::collectStruct) — a real zero-length LLVM array, not a
        // pointer, since it's embedded storage whose address IS the
        // struct field's address, not a separately-allocated pointee.
        if (at.size < 0) return llvm::PointerType::get(ctx_, 0);
        return llvm::ArrayType::get(elemTy, static_cast<uint64_t>(at.size));
    }
    case TypeKind::Enum: {
        auto &et = static_cast<const EnumType &>(*ty);
        return llvm::Type::getIntNTy(ctx_, et.bitWidth);
    }
    case TypeKind::Vector: {
        // vec<T, N> → LLVM's native fixed-width <N x T> vector IR type —
        // the whole point of std::simd being 'just' this plus a thin naming
        // layer is that every arithmetic op, load/store, etc. on this type
        // already lowers correctly through LLVM's normal (non-vector-
        // specific) instruction builders (see applyBinaryOp's scalarTy
        // unwrap), so the target's vector-register ISA (SSE/AVX/NEON/RVV/
        // wasm SIMD128) is selected entirely by LLVM's backend, not by any
        // per-architecture code in this compiler.
        auto &vt = static_cast<const VectorType &>(*ty);
        return llvm::FixedVectorType::get(lowerType(vt.element),
                                           static_cast<unsigned>(vt.width));
    }
    case TypeKind::Function: {
        auto &ft = static_cast<const FunctionType &>(*ty);
        std::vector<llvm::Type *> paramTys;
        for (auto &p : ft.paramTypes) paramTys.push_back(lowerType(p));
        auto *retTy = lowerType(ft.returnType);
        return llvm::FunctionType::get(retTy, paramTys, ft.variadic);
    }
    case TypeKind::Tuple: {
        // Tuple lowers to an anonymous LLVM struct type
        auto &tt = static_cast<const TupleType &>(*ty);
        std::vector<llvm::Type *> elemTys;
        for (auto &e : tt.elementTypes) elemTys.push_back(lowerType(e));
        return llvm::StructType::get(ctx_, elemTys);
    }
    case TypeKind::Optional: {
        // ?T  →  { T, i1 }   (value, has_value)
        auto &opt = static_cast<const OptionalType &>(*ty);
        auto *inner = lowerType(opt.inner);
        return llvm::StructType::get(ctx_, {inner, llvm::Type::getInt1Ty(ctx_)});
    }
    case TypeKind::Slice: {
        // []T  →  { T*, i64 }  (ptr, length)
        return llvm::StructType::get(ctx_,
            {llvm::PointerType::get(ctx_, 0), llvm::Type::getInt64Ty(ctx_)});
    }
    case TypeKind::Newtype: {
        // Newtype lowers to its underlying type
        auto &nt = static_cast<const NewtypeType &>(*ty);
        return lowerType(nt.base);
    }
    case TypeKind::Error:
    default:
        return llvm::Type::getInt32Ty(ctx_); // fallback
    }
}

llvm::StructType *CodeGen::lowerStructType(const StructType &st) {
    auto it = structCache_.find(st.name);
    // A cache hit only short-circuits if that entry's body was actually
    // filled in. It's possible to reach here first via an incomplete
    // forward-reference to this struct name (e.g. a sibling struct's field
    // typed with this one, resolved to a not-yet-fully-parsed StructType
    // object distinct from the real one Sema later creates when the
    // struct's own definition is parsed) — that path still calls
    // lowerStructType (with st.isDefined false / st.fields empty at the
    // time), which caches a bare opaque struct. Treating that as a
    // permanent, final result would leave every later reference to this
    // struct name silently bound to a 0-element type — including this
    // struct's own methods, whose 'self.field' accesses would then GEP
    // out of bounds and crash CodeGen (observed via std::sync::mpsc.h's
    // MpscQueue embedding std::Spinlock as a field: Spinlock's own
    // lock()/unlock() bodies crashed compiling 'self.locked = ...' because
    // the cached Spinlock type had been frozen empty by MpscQueue's field
    // resolving it first). So: still complete a stale opaque entry's body
    // below if we now have real field data, instead of returning early.
    if (it != structCache_.end() && !it->second->isOpaque()) return it->second;

    llvm::StructType *llvmSt;
    if (it != structCache_.end()) {
        llvmSt = it->second; // opaque placeholder from an earlier forward-reference
    } else {
        // Create opaque struct first (handles recursive types)
        llvmSt = llvm::StructType::create(ctx_, st.name);
        structCache_[st.name] = llvmSt;
    }

    // Fill in fields
    if (st.isDefined) {
        if (st.isTaggedUnion) {
            // Tagged union layout: { i32 tag, [maxPayloadSize x i8] payload }
            int payloadSize = st.maxPayloadSize > 0 ? st.maxPayloadSize : 8;
            std::vector<llvm::Type *> body = {
                llvm::Type::getInt32Ty(ctx_),
                llvm::ArrayType::get(llvm::Type::getInt8Ty(ctx_), payloadSize)
            };
            llvmSt->setBody(body, false);
        } else {
            // Bitfields share an LLVM struct slot across multiple SafeC
            // FieldDecls (same 'index') — emit one LLVM field per *distinct*
            // index, typed as whichever field declared that slot (they're
            // all the same underlying type within one packed run; see
            // Sema::collectStruct), not one per SafeC-level field.
            std::vector<llvm::Type *> fieldTys;
            int lastIdx = -1;
            for (auto &f : st.fields) {
                if (f.index == lastIdx) continue;
                lastIdx = f.index;
                fieldTys.push_back(lowerType(f.type));
            }
            llvmSt->setBody(fieldTys, st.isPacked);
        }
    }
    return llvmSt;
}

llvm::PointerType *CodeGen::lowerRefType(const ReferenceType &) {
    // All references → opaque pointer at IR level
    return llvm::PointerType::get(ctx_, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// ARENA STATE GLOBAL
// ─────────────────────────────────────────────────────────────────────────────

void CodeGen::genArenaStateGlobal(RegionDecl &rd) {
    // Create a global arena state: { ptr buf, i64 used, i64 cap }
    auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
    auto *PtrTy   = llvm::PointerType::get(ctx_, 0);
    auto *stateTy = llvm::StructType::get(ctx_, {PtrTy, Int64Ty, Int64Ty});
    std::string varName = "__arena_" + rd.name;
    // Avoid duplicate global
    if (mod_->getNamedGlobal(varName)) return;
    auto *gv = new llvm::GlobalVariable(
        *mod_, stateTy, /*isConst=*/false,
        llvm::GlobalValue::InternalLinkage,
        llvm::Constant::getNullValue(stateTy),
        varName);
    arenaStateMap_[rd.name] = {gv, stateTy, rd.capacity};
    globals_[varName] = gv;
}

llvm::Value *CodeGen::genNew(NewExpr &e, FnEnv &env) {
    auto *elemTy  = lowerType(e.allocType);
    auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
    auto *PtrTy   = llvm::PointerType::get(ctx_, 0);
    const auto &dl = mod_->getDataLayout();
    uint64_t elemSize = dl.getTypeAllocSize(elemTy);

    if (!e.regionName.empty()) {
        auto it = arenaStateMap_.find(e.regionName);
        if (it != arenaStateMap_.end()) {
            auto &info = it->second;
            // Load buf pointer
            auto *bufPtr  = builder_.CreateStructGEP(info.ty, info.var, 0, "arena.buf.ptr");
            auto *buf     = builder_.CreateLoad(PtrTy, bufPtr, "arena.buf");
            // Check if null → malloc
            auto *isNull  = builder_.CreateIsNull(buf, "arena.null");
            auto *mallocBB = llvm::BasicBlock::Create(ctx_, "arena.first", env.fn);
            auto *contBB   = llvm::BasicBlock::Create(ctx_, "arena.cont",  env.fn);
            builder_.CreateCondBr(isNull, mallocBB, contBB);

            builder_.SetInsertPoint(mallocBB);
            auto *capVal  = llvm::ConstantInt::get(Int64Ty, info.cap > 0 ? info.cap : 65536);
            auto mallocFn = mod_->getOrInsertFunction("malloc",
                llvm::FunctionType::get(PtrTy, {Int64Ty}, false));
            auto *newBuf  = builder_.CreateCall(mallocFn, {capVal}, "arena.newbuf");
            builder_.CreateStore(newBuf, bufPtr);
            builder_.CreateBr(contBB);

            builder_.SetInsertPoint(contBB);
            auto *bufPhi  = builder_.CreatePHI(PtrTy, 2, "arena.buf.phi");
            bufPhi->addIncoming(buf, mallocBB->getSinglePredecessor()
                                     ? mallocBB->getSinglePredecessor() : mallocBB);
            bufPhi->addIncoming(newBuf, mallocBB);

            // Load used and bump
            auto *usedPtr = builder_.CreateStructGEP(info.ty, info.var, 1, "arena.used.ptr");
            auto *used    = builder_.CreateLoad(Int64Ty, usedPtr, "arena.used");
            auto *allocPtr = builder_.CreateGEP(
                llvm::Type::getInt8Ty(ctx_), bufPhi, used, "arena.ptr");
            auto *newUsed = builder_.CreateAdd(
                used, llvm::ConstantInt::get(Int64Ty, elemSize), "arena.newused");
            builder_.CreateStore(newUsed, usedPtr);
            return allocPtr;
        }
    }

    // Fallback: heap allocation via std::alloc (mangled 'std_alloc'), NOT a
    // raw 'malloc' call — std::alloc prefixes every block with the 16-byte
    // live/freed-magic header std::dealloc() requires (see mem.sc's
    // ALLOC_HDR_SIZE_ doc comment). A plain 'malloc(size)' here would hand
    // back a pointer with no such header, so the *very first*
    // std::dealloc() call on a '&heap T' from 'new T' would misread
    // whatever bytes happen to precede the block as a bad magic value and
    // abort ("mismatched allocator or corrupted heap") — 'new'/'dealloc'
    // must share the same allocator for the compile-time UAF/double-free
    // tracking (see Sema::heapFreeGeneration_) to correspond to anything
    // real at runtime.
    auto *sizeVal = llvm::ConstantInt::get(Int64Ty, elemSize);
    auto allocFn = mod_->getOrInsertFunction("std_alloc",
        llvm::FunctionType::get(PtrTy, {Int64Ty}, false));
    return builder_.CreateCall(allocFn, {sizeVal}, "heap.alloc");
}

llvm::Value *CodeGen::genTupleLit(TupleLitExpr &e, FnEnv &env) {
    if (!e.type || e.type->kind != TypeKind::Tuple) {
        if (!e.elements.empty()) return genExpr(*e.elements[0], env);
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }
    auto *tupleTy = static_cast<llvm::StructType *>(lowerType(e.type));
    auto *alloca  = createEntryAlloca(env, tupleTy, "tuple.tmp");
    for (size_t i = 0; i < e.elements.size(); ++i) {
        auto *val = genExpr(*e.elements[i], env);
        auto *gep = builder_.CreateStructGEP(tupleTy, alloca, static_cast<unsigned>(i));
        builder_.CreateStore(val, gep);
    }
    return builder_.CreateLoad(tupleTy, alloca, "tuple");
}

// See ThreadBackend in CodeGen.h for what each case means. freestanding_
// always wins (no OS thread API exists to assume there, regardless of
// --target); otherwise dispatch on the target triple, when one was given
// (targetMachine_ is only non-null for an explicit '--target' — see the
// constructor). No explicit target at all keeps meaning "host default",
// i.e. Pthread, exactly as it always has — this dispatch changes nothing
// for the common case of compiling for the host with no --target flag.
CodeGen::ThreadBackend CodeGen::selectThreadBackend() const {
    if (freestanding_) return ThreadBackend::Hook;
    if (targetMachine_) {
        llvm::Triple triple(mod_->getTargetTriple());
        if (triple.isOSWindows()) return ThreadBackend::Win32;
        if (triple.isOSLinux() || triple.isMacOSX() || triple.isOSFreeBSD() ||
            triple.isOSNetBSD() || triple.isOSOpenBSD() || triple.isOSDragonFly() ||
            triple.isOSSolaris())
            return ThreadBackend::Pthread;
        // wasm32-*, riscv64-unknown-elf, spirv64-*, or any other triple with
        // no recognized OS-level thread API — see the Hook case comment.
        return ThreadBackend::Hook;
    }
    return ThreadBackend::Pthread;
}

llvm::Value *CodeGen::genThreadCreate(llvm::Value *fnVal, llvm::Value *argVal, FnEnv &env) {
    auto *PtrTy   = llvm::PointerType::get(ctx_, 0);
    auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
    auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);

    // Convert integer arg to pointer (e.g., 0 -> null ptr) — shared by
    // every backend below, so callers (genSpawn, spawn_scoped) no longer
    // need their own copy of this normalization.
    if (argVal->getType()->isIntegerTy())
        argVal = builder_.CreateIntToPtr(argVal, PtrTy, "arg.ptr");

    switch (selectThreadBackend()) {
    case ThreadBackend::Win32: {
        // CreateThread(secAttrs, stackSz, startAddr, param, flags, tidOut)
        // -> HANDLE. Same signature std/thread.sc's own wrapper already
        // declares and calls (thread.sc:12,44) — kept consistent so both
        // the language-level 'spawn' and the library-level 'thread_create'
        // agree on the Win32 ABI.
        auto *ctTy = llvm::FunctionType::get(PtrTy,
            {PtrTy, Int64Ty, PtrTy, PtrTy, Int32Ty, PtrTy}, false);
        auto ctFn = mod_->getOrInsertFunction("CreateThread", ctTy);
        auto *nullPtr = llvm::ConstantPointerNull::get(PtrTy);
        auto *handle = builder_.CreateCall(ctTy, ctFn.getCallee(),
            {nullPtr, llvm::ConstantInt::get(Int64Ty, 0), fnVal, argVal,
             llvm::ConstantInt::get(Int32Ty, 0), nullPtr}, "win_thread");
        return builder_.CreatePtrToInt(handle, Int64Ty, "thread_id");
    }
    case ThreadBackend::Hook: {
        // __safec_thread_create(func, arg) -> i64 handle — see the
        // ThreadBackend::Hook comment in CodeGen.h for who implements this.
        auto *hookTy = llvm::FunctionType::get(Int64Ty, {PtrTy, PtrTy}, false);
        auto hookFn  = mod_->getOrInsertFunction("__safec_thread_create", hookTy);
        return builder_.CreateCall(hookTy, hookFn.getCallee(), {fnVal, argVal}, "thread_id");
    }
    case ThreadBackend::Pthread:
    default: {
        // pthread_create(pthread_t*, attr*, void*(*)(void*), void*) -> int
        auto *handleAlloca = createEntryAlloca(env, Int64Ty, "thread_handle");
        auto *ptCreateTy = llvm::FunctionType::get(Int32Ty,
            {PtrTy, PtrTy, PtrTy, PtrTy}, false);
        auto ptCreateFn = mod_->getOrInsertFunction("pthread_create", ptCreateTy);
        auto *nullPtr = llvm::ConstantPointerNull::get(PtrTy);
        builder_.CreateCall(ptCreateTy, ptCreateFn.getCallee(),
            {handleAlloca, nullPtr, fnVal, argVal});
        return builder_.CreateLoad(Int64Ty, handleAlloca, "thread_id");
    }
    }
}

void CodeGen::genThreadJoin(llvm::Value *handleVal, FnEnv &env) {
    (void)env;
    auto *PtrTy   = llvm::PointerType::get(ctx_, 0);
    auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
    auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
    if (handleVal->getType() != Int64Ty) {
        if (handleVal->getType()->isIntegerTy())
            handleVal = builder_.CreateZExt(handleVal, Int64Ty);
        else if (handleVal->getType()->isPointerTy())
            handleVal = builder_.CreatePtrToInt(handleVal, Int64Ty);
    }

    switch (selectThreadBackend()) {
    case ThreadBackend::Win32: {
        // WaitForSingleObject(handle, INFINITE=0xFFFFFFFF)
        auto *wfsoTy = llvm::FunctionType::get(Int32Ty, {PtrTy, Int32Ty}, false);
        auto wfsoFn  = mod_->getOrInsertFunction("WaitForSingleObject", wfsoTy);
        auto *handlePtr = builder_.CreateIntToPtr(handleVal, PtrTy);
        builder_.CreateCall(wfsoTy, wfsoFn.getCallee(),
            {handlePtr, llvm::ConstantInt::get(Int32Ty, 0xFFFFFFFFu)});
        return;
    }
    case ThreadBackend::Hook: {
        auto *hookTy = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {Int64Ty}, false);
        auto hookFn  = mod_->getOrInsertFunction("__safec_thread_join", hookTy);
        builder_.CreateCall(hookTy, hookFn.getCallee(), {handleVal});
        return;
    }
    case ThreadBackend::Pthread:
    default: {
        auto *ptJoinTy = llvm::FunctionType::get(Int32Ty, {Int64Ty, PtrTy}, false);
        auto ptJoinFn  = mod_->getOrInsertFunction("pthread_join", ptJoinTy);
        builder_.CreateCall(ptJoinTy, ptJoinFn.getCallee(),
            {handleVal, llvm::ConstantPointerNull::get(PtrTy)});
        return;
    }
    }
}

llvm::Value *CodeGen::genSpawn(SpawnExpr &e, FnEnv &env) {
    llvm::Value *fnVal  = genExpr(*e.fnExpr, env);
    llvm::Value *argVal = genExpr(*e.argExpr, env);
    return genThreadCreate(fnVal, argVal, env);
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION PROTOTYPE
// ─────────────────────────────────────────────────────────────────────────────

llvm::Function *CodeGen::genFunctionProto(FunctionDecl &fn) {
    // Build LLVM function type
    std::vector<llvm::Type *> paramTys;
    for (auto &p : fn.params) {
        // Array-typed parameters decay to a pointer at the C ABI boundary
        // (same rule as C: 'void f(int x[10])' really takes 'int*') — the
        // caller already decays the argument (see genCall's array-to-pointer
        // decay), so the callee's declared signature must match or the LLVM
        // call site and the function definition disagree on the parameter's
        // LLVM type (aggregate vs pointer), which the verifier rejects.
        if (p.type && p.type->kind == TypeKind::Array)
            paramTys.push_back(llvm::PointerType::get(ctx_, 0));
        else
            paramTys.push_back(lowerType(p.type));
    }
    auto *retTy  = fn.returnType ? lowerType(fn.returnType) : llvm::Type::getVoidTy(ctx_);
    auto *fnType = llvm::FunctionType::get(retTy, paramTys, fn.isVariadic);

    // Reuse an existing declaration (forward-decl from .h already created it).
    // This prevents LLVM from auto-renaming a second Function::Create to "name.1".
    //
    // A header prototype and the '.sc' definition are two separate
    // FunctionDecl nodes for the same symbol (see the '.h'/'.sc' pairing
    // convention elsewhere in this file). Attribute-bearing modifiers like
    // 'inline'/'pure'/'noreturn' are written on the DEFINING declaration
    // (the one with a body) — if that's the second FunctionDecl processed
    // here, naively returning the already-created (attribute-less) Function
    // would silently drop them. So: apply attributes whenever 'fn' is the
    // defining occurrence, whether or not the Function already exists.
    llvm::Function *llvmFn = mod_->getFunction(fn.name);
    bool isNewFunction = (llvmFn == nullptr);
    if (isNewFunction) {
        auto linkage = fn.isExtern
            ? llvm::Function::ExternalLinkage
            : llvm::Function::ExternalLinkage;  // default external for C ABI
        llvmFn = llvm::Function::Create(fnType, linkage, fn.name, *mod_);
    }

    if (isNewFunction || fn.body) {
        if (fn.isInline) llvmFn->addFnAttr(llvm::Attribute::InlineHint);

        // ── Bare-metal / effect system attributes ────────────────────────
        if (fn.isNaked)    llvmFn->addFnAttr(llvm::Attribute::Naked);
        if (fn.isNoReturn) llvmFn->addFnAttr(llvm::Attribute::NoReturn);
        if (fn.isPure) {
            llvmFn->setMemoryEffects(llvm::MemoryEffects::readOnly());
            llvmFn->addFnAttr(llvm::Attribute::NoUnwind);
        }
        if (fn.isInterrupt) {
            llvmFn->addFnAttr("interrupt");
        }
        if (!fn.sectionName.empty()) llvmFn->setSection(fn.sectionName);
        // Calling convention
        if (fn.callingConv == "stdcall")
            llvmFn->setCallingConv(llvm::CallingConv::X86_StdCall);
        else if (fn.callingConv == "fastcall")
            llvmFn->setCallingConv(llvm::CallingConv::X86_FastCall);
        else if (fn.callingConv == "cdecl")
            llvmFn->setCallingConv(llvm::CallingConv::C);
        if (freestanding_) llvmFn->addFnAttr(llvm::Attribute::NoBuiltin);

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

    // ── Naked functions: skip prologue, emit asm only + unreachable ──────────
    if (fn.isNaked) {
        auto *entryBB = llvm::BasicBlock::Create(ctx_, "entry", llvmFn);
        builder_.SetInsertPoint(entryBB);
        env.entry = entryBB;
        // Emit only asm statements (body validated by Sema)
        if (fn.body) {
            for (auto &s : fn.body->body) {
                if (s->kind == StmtKind::Asm)
                    genAsm(static_cast<AsmStmt &>(*s), env);
            }
        }
        if (!isTerminated(builder_.GetInsertBlock()))
            builder_.CreateUnreachable();
        return;
    }

    // Debug: attach DISubprogram
    if (dib_ && diFile_) {
        auto *spTy = diSubroutineType(fn);
        unsigned line = fn.loc.line ? fn.loc.line : 1;
        auto *diSP = dib_->createFunction(
            diFile_, fn.name, "", diFile_,
            line, spTy, line,
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition);
        llvmFn->setSubprogram(diSP);
    }

    // Create entry basic block
    auto *entryBB = llvm::BasicBlock::Create(ctx_, "entry", llvmFn);
    builder_.SetInsertPoint(entryBB);
    env.entry = entryBB;

    // Set initial debug location to function start
    if (dib_ && llvmFn->getSubprogram()) {
        builder_.SetCurrentDebugLocation(
            llvm::DILocation::get(ctx_,
                fn.loc.line ? fn.loc.line : 1,
                fn.loc.col,
                llvmFn->getSubprogram()));
    }

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
        // Array-typed parameters decay to a pointer (see genFunctionProto) —
        // the incoming LLVM arg is already that pointer, so the local slot
        // holds a 'ptr' rather than the array aggregate.
        auto *aTy  = (p.type && p.type->kind == TypeKind::Array)
                         ? llvm::PointerType::get(ctx_, 0)
                         : lowerType(p.type);
        auto *alloca = createEntryAlloca(env, aTy, p.name + ".addr");
        builder_.CreateStore(&arg, alloca);
        env.locals[p.name] = alloca;
        ++idx;
    }

    // Pre-collect goto labels — create forward-declared BBs
    for (auto &stmt : fn.body->body)
        collectGotoLabels(*stmt, env);

    // Generate body
    genCompound(*fn.body, env);

    // Fallthrough to return block if not already terminated
    // Emit deferred statements before branching (same as genReturn does for explicit return)
    if (!isTerminated(builder_.GetInsertBlock())) {
        emitDefers(env, /*isErrorPath=*/false);
        builder_.CreateBr(env.returnBB);
    }

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

llvm::Constant *CodeGen::evalConstInit(Expr &e, llvm::Type *expectedTy) {
    switch (e.kind) {
    case ExprKind::IntLit: {
        int64_t v = static_cast<IntLitExpr &>(e).value;
        if (expectedTy->isIntegerTy())      return llvm::ConstantInt::get(expectedTy, (uint64_t)v, true);
        if (expectedTy->isFloatingPointTy()) return llvm::ConstantFP::get(expectedTy, (double)v);
        if (expectedTy->isPointerTy() && v == 0)
            return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(expectedTy));
        return nullptr;
    }
    case ExprKind::FloatLit:
        return expectedTy->isFloatingPointTy()
            ? llvm::ConstantFP::get(expectedTy, static_cast<FloatLitExpr &>(e).value)
            : nullptr;
    case ExprKind::BoolLit:
        return expectedTy->isIntegerTy()
            ? llvm::ConstantInt::get(expectedTy, static_cast<BoolLitExpr &>(e).value ? 1 : 0)
            : nullptr;
    case ExprKind::NullLit:
        return expectedTy->isPointerTy()
            ? llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(expectedTy))
            : nullptr;
    case ExprKind::Unary: {
        auto &ue = static_cast<UnaryExpr &>(e);
        if (ue.op != UnaryOp::Neg) return nullptr;
        auto *inner = evalConstInit(*ue.operand, expectedTy);
        if (!inner) return nullptr;
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(inner))
            return llvm::ConstantInt::get(expectedTy, (uint64_t)(-ci->getSExtValue()), true);
        if (auto *cf = llvm::dyn_cast<llvm::ConstantFP>(inner))
            return llvm::ConstantFP::get(expectedTy, -cf->getValueAPF().convertToDouble());
        return nullptr;
    }
    case ExprKind::Binary: {
        // Simple constant arithmetic (e.g. 'N - 1' from a macro-expanded
        // static-buffer declaration like RING_STATIC). Both operands are
        // evaluated against the same expected type — good enough for the
        // common case of same-width integer arithmetic in an initializer;
        // not a general constexpr interpreter.
        auto &be = static_cast<BinaryExpr &>(e);
        auto *lhs = evalConstInit(*be.left, expectedTy);
        auto *rhs = evalConstInit(*be.right, expectedTy);
        auto *lci = lhs ? llvm::dyn_cast<llvm::ConstantInt>(lhs) : nullptr;
        auto *rci = rhs ? llvm::dyn_cast<llvm::ConstantInt>(rhs) : nullptr;
        if (!lci || !rci) return nullptr;
        int64_t l = lci->getSExtValue(), r = rci->getSExtValue(), result = 0;
        switch (be.op) {
        case BinaryOp::Add:    result = l + r; break;
        case BinaryOp::Sub:    result = l - r; break;
        case BinaryOp::Mul:    result = l * r; break;
        case BinaryOp::Div:    if (r == 0) return nullptr; result = l / r; break;
        case BinaryOp::Mod:    if (r == 0) return nullptr; result = l % r; break;
        case BinaryOp::BitAnd: result = l & r; break;
        case BinaryOp::BitOr:  result = l | r; break;
        case BinaryOp::BitXor: result = l ^ r; break;
        case BinaryOp::Shl:    result = l << r; break;
        case BinaryOp::Shr:    result = l >> r; break;
        default: return nullptr;
        }
        return expectedTy->isIntegerTy()
            ? llvm::ConstantInt::get(expectedTy, (uint64_t)result, true)
            : nullptr;
    }
    case ExprKind::Cast: {
        auto &ce = static_cast<CastExpr &>(e);
        // Integer constant -> pointer, e.g. 'volatile int *UART_DATA =
        // (volatile int*)0x40001000;' — an MMIO-register-style global.
        // Recursing into the operand against 'expectedTy' (a pointer type)
        // used to be the whole body here, which only ever produced a
        // non-null result for the v==0 case (evalConstInit's IntLit arm
        // only special-cases a null pointer) — any real address silently
        // fell through to returning nullptr, reported by the caller as
        // "not a compile-time constant expression". Evaluate the operand
        // as an i64 instead and wrap it in a genuine inttoptr constant.
        if (expectedTy->isPointerTy()) {
            auto *asInt = evalConstInit(*ce.operand, llvm::Type::getInt64Ty(ctx_));
            if (auto *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(asInt)) {
                return llvm::ConstantExpr::getIntToPtr(ci, expectedTy);
            }
        }
        return evalConstInit(*ce.operand, expectedTy);
    }
    case ExprKind::Ident: {
        // Reference to another global (e.g. a static array's address) — under
        // opaque pointers the global's own address already *is* "pointer to
        // its first element", so no GEP is needed for array decay here.
        auto it = globals_.find(static_cast<IdentExpr &>(e).name);
        if (it == globals_.end()) return nullptr;
        auto *c = llvm::dyn_cast<llvm::Constant>(it->second);
        if (!c) return nullptr;
        if (expectedTy->isPointerTy() && c->getType()->isPointerTy()) return c;
        return (c->getType() == expectedTy) ? c : nullptr;
    }
    case ExprKind::Compound: {
        // Designated initializers ('{.field = v}' / '{[i] = v}') resolve to
        // a target slot per element via Sema::checkCompoundInit's
        // 'resolvedSlots' (parallel to 'inits'); plain positional literals
        // ('{a, b, c}', no designators at all) leave resolvedSlots empty,
        // so slot == source position, same as before designators existed.
        auto &ci = static_cast<CompoundInitExpr &>(e);
        if (expectedTy->isVectorTy()) {
            auto *vt = llvm::cast<llvm::FixedVectorType>(expectedTy);
            auto *elemTy = vt->getElementType();
            std::vector<llvm::Constant *> elems(vt->getNumElements(), nullptr);
            for (size_t i = 0; i < ci.inits.size(); ++i) {
                int64_t slot = ci.resolvedSlots.empty() ? (int64_t)i : ci.resolvedSlots[i];
                if (slot < 0 || slot >= (int64_t)vt->getNumElements()) return nullptr;
                auto *c = evalConstInit(*ci.inits[i], elemTy);
                if (!c) return nullptr;
                elems[(size_t)slot] = c;
            }
            for (unsigned i = 0; i < vt->getNumElements(); ++i)
                if (!elems[i]) elems[i] = llvm::Constant::getNullValue(elemTy);
            return llvm::ConstantVector::get(elems);
        }
        if (expectedTy->isStructTy()) {
            auto *st = llvm::cast<llvm::StructType>(expectedTy);
            std::vector<llvm::Constant *> fields(st->getNumElements(), nullptr);
            for (size_t i = 0; i < ci.inits.size(); ++i) {
                int64_t slot = ci.resolvedSlots.empty() ? (int64_t)i : ci.resolvedSlots[i];
                if (slot < 0 || slot >= (int64_t)st->getNumElements()) return nullptr;
                auto *c = evalConstInit(*ci.inits[i], st->getElementType((unsigned)slot));
                if (!c) return nullptr;
                fields[(size_t)slot] = c;
            }
            for (unsigned i = 0; i < st->getNumElements(); ++i)
                if (!fields[i]) fields[i] = llvm::Constant::getNullValue(st->getElementType(i));
            return llvm::ConstantStruct::get(st, fields);
        }
        if (expectedTy->isArrayTy()) {
            auto *at = llvm::cast<llvm::ArrayType>(expectedTy);
            auto *elemTy = at->getElementType();
            std::vector<llvm::Constant *> elems(at->getNumElements(), nullptr);
            for (size_t i = 0; i < ci.inits.size(); ++i) {
                int64_t slot = ci.resolvedSlots.empty() ? (int64_t)i : ci.resolvedSlots[i];
                if (slot < 0 || slot >= (int64_t)at->getNumElements()) return nullptr;
                auto *c = evalConstInit(*ci.inits[i], elemTy);
                if (!c) return nullptr;
                elems[(size_t)slot] = c;
            }
            for (unsigned i = 0; i < at->getNumElements(); ++i)
                if (!elems[i]) elems[i] = llvm::Constant::getNullValue(elemTy);
            return llvm::ConstantArray::get(at, elems);
        }
        return nullptr;
    }
    default:
        return nullptr;
    }
}

llvm::GlobalVariable *CodeGen::genGlobalVar(GlobalVarDecl &gv) {
    // Repeated 'extern T x;' declarations of the same name across
    // independently-included .sc/.h files are common in std/ (each file
    // re-declares whatever externs it needs, the same de-duplication
    // genFunction already does for extern functions via
    // mod_->getFunction() above) — reuse the existing global instead of
    // creating a second one. Without this, LLVM's Module silently
    // uniquifies the name (e.g. '__stderrp' -> '__stderrp.1') to avoid an
    // in-module collision, and that renamed symbol then fails to resolve
    // at link time since no library actually exports it under that name.
    if (gv.isExtern) {
        if (auto *existing = mod_->getNamedGlobal(gv.name)) {
            globals_[gv.name] = existing;
            return existing;
        }
    }
    auto *ty = lowerType(gv.type);
    llvm::Constant *init = llvm::Constant::getNullValue(ty);

    if (gv.init && !ty->isVoidTy()) {
        if (auto *folded = evalConstInit(*gv.init, ty)) {
            init = folded;
        } else {
            // File-scope variables require a compile-time-constant initializer
            // unconditionally — this isn't a SafeC-specific restriction to
            // relax later, it's the same constraint plain C places on
            // file-scope initializers (C11 §6.7.9p4), so every global is
            // already 'constinit' whether or not that keyword is written.
            // Silently zero-initializing here would substitute the wrong
            // value without any indication something was rejected.
            diag_.error(gv.loc,
                "global initializer for '" + gv.name +
                "' is not a compile-time constant expression");
        }
    }

    auto *gvar = new llvm::GlobalVariable(
        *mod_, ty, gv.isConst,
        gv.isExtern ? llvm::GlobalValue::ExternalLinkage
                    : llvm::GlobalValue::InternalLinkage,
        gv.isExtern ? nullptr : init,
        gv.name);

    // Bare-metal attributes: section, alignment, volatile, thread_local
    if (!gv.sectionName.empty()) gvar->setSection(gv.sectionName);
    if (gv.alignment > 0) gvar->setAlignment(llvm::Align(gv.alignment));
    // InitialExec, not the LLVM default GeneralDynamic: GeneralDynamic is
    // the model that stays correct even if the defining module could be
    // dlopen'd after the process has already started (real shared-library
    // TLS), which costs a __tls_get_addr call on every single access.
    // safec always produces a normal statically/dynamically *linked*
    // executable (or a static archive linked into one) — never something
    // safec itself dlopens — so InitialExec (resolved once at load time,
    // then a direct thread-pointer-relative load per access, no function
    // call) is safe here and meaningfully faster for any thread_local
    // that's read/written on a hot path (e.g. std/mem.sc's per-thread
    // allocator quarantine — verified: this cut single-threaded overhead
    // measurably vs. GeneralDynamic on real thread_local globals).
    if (gv.isThreadLocal)
        gvar->setThreadLocalMode(llvm::GlobalVariable::LocalExecTLSModel);

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
// DEBUG INFO HELPERS
// ─────────────────────────────────────────────────────────────────────────────

llvm::DIType *CodeGen::diTypeFor(const TypePtr &ty) {
    if (!ty || !dib_) return dib_->createUnspecifiedType("void");
    switch (ty->kind) {
    case TypeKind::Void:    return dib_->createUnspecifiedType("void");
    case TypeKind::Bool:
        return dib_->createBasicType("bool", 1,
                                     llvm::dwarf::DW_ATE_boolean);
    case TypeKind::Char:
        return dib_->createBasicType("char", 8,
                                     llvm::dwarf::DW_ATE_signed_char);
    case TypeKind::Int8:
        return dib_->createBasicType("int8_t", 8,
                                     llvm::dwarf::DW_ATE_signed);
    case TypeKind::Int16:
        return dib_->createBasicType("short", 16,
                                     llvm::dwarf::DW_ATE_signed);
    case TypeKind::Int32:
        return dib_->createBasicType("int", 32,
                                     llvm::dwarf::DW_ATE_signed);
    case TypeKind::Int64:
        return dib_->createBasicType("long long", 64,
                                     llvm::dwarf::DW_ATE_signed);
    case TypeKind::UInt8:
        return dib_->createBasicType("uint8_t", 8,
                                     llvm::dwarf::DW_ATE_unsigned_char);
    case TypeKind::UInt16:
        return dib_->createBasicType("unsigned short", 16,
                                     llvm::dwarf::DW_ATE_unsigned);
    case TypeKind::UInt32:
        return dib_->createBasicType("unsigned int", 32,
                                     llvm::dwarf::DW_ATE_unsigned);
    case TypeKind::UInt64:
        return dib_->createBasicType("unsigned long long", 64,
                                     llvm::dwarf::DW_ATE_unsigned);
    case TypeKind::Float32:
        return dib_->createBasicType("float", 32,
                                     llvm::dwarf::DW_ATE_float);
    case TypeKind::Float64:
        return dib_->createBasicType("double", 64,
                                     llvm::dwarf::DW_ATE_float);
    case TypeKind::Pointer:
    case TypeKind::Reference:
        return dib_->createPointerType(
            dib_->createUnspecifiedType("void"), 64);
    default:
        return dib_->createUnspecifiedType(ty->str());
    }
}

llvm::DISubroutineType *CodeGen::diSubroutineType(FunctionDecl &fn) {
    llvm::SmallVector<llvm::Metadata *, 8> paramTypes;
    // First element is return type
    paramTypes.push_back(diTypeFor(fn.returnType));
    for (auto &p : fn.params)
        paramTypes.push_back(diTypeFor(p.type));
    return dib_->createSubroutineType(
        dib_->getOrCreateTypeArray(paramTypes));
}

// ─────────────────────────────────────────────────────────────────────────────
// STATEMENTS
// ─────────────────────────────────────────────────────────────────────────────

void CodeGen::genStmt(Stmt &s, FnEnv &env) {
    // Attach source location for lines/full debug
    if (dib_ && env.fn->getSubprogram() && s.loc.line) {
        builder_.SetCurrentDebugLocation(
            llvm::DILocation::get(ctx_, s.loc.line, s.loc.col,
                                  env.fn->getSubprogram()));
    }
    switch (s.kind) {
    case StmtKind::Compound:     genCompound(static_cast<CompoundStmt &>(s), env);   break;
    case StmtKind::Expr:         genExprStmt(static_cast<ExprStmt &>(s), env);       break;
    case StmtKind::If:           genIf(static_cast<IfStmt &>(s), env);               break;
    case StmtKind::While:
    case StmtKind::DoWhile:      genWhile(static_cast<WhileStmt &>(s), env);         break;
    case StmtKind::For:          genFor(static_cast<ForStmt &>(s), env);             break;
    case StmtKind::Switch:       genSwitch(static_cast<SwitchStmt &>(s), env);       break;
    case StmtKind::Return:       genReturn(static_cast<ReturnStmt &>(s), env);       break;
    case StmtKind::VarDecl:      genVarDecl(static_cast<VarDeclStmt &>(s), env);    break;
    case StmtKind::MultiVarDecl: {
        auto &mv = static_cast<MultiVarDeclStmt &>(s);
        for (auto &d : mv.decls) genStmt(*d, env);
        break;
    }
    case StmtKind::Unsafe:       genUnsafe(static_cast<UnsafeStmt &>(s), env);      break;
    case StmtKind::Break: {
        auto &bs = static_cast<BreakStmt &>(s);
        auto *bb = env.breakBB(bs.label);
        if (bb) builder_.CreateBr(bb);
        break;
    }
    case StmtKind::Continue: {
        auto &cs = static_cast<ContinueStmt &>(s);
        auto *bb = env.continueBB(cs.label);
        if (bb) builder_.CreateBr(bb);
        break;
    }
    case StmtKind::Defer: {
        // Register the defer body to be emitted at return/exit
        auto &ds = static_cast<DeferStmt &>(s);
        env.deferList.push_back(&ds);
        break;
    }
    case StmtKind::Match:
        genMatch(static_cast<MatchStmt &>(s), env);
        break;
    case StmtKind::StaticAssert: break; // handled at compile-time by ConstEvalEngine
    case StmtKind::IfConst: {
        // Compile-time branch: ConstEvalEngine has set constResult.
        // Emit only the selected branch; the other is dead code.
        auto &ics = static_cast<IfConstStmt &>(s);
        if (!ics.constResult.has_value()) {
            // Fallback: evaluate as a regular if at runtime
            if (ics.cond && ics.then) {
                llvm::Value *cond = genExpr(*ics.cond, env);
                if (cond && cond->getType() != builder_.getInt1Ty())
                    cond = builder_.CreateICmpNE(cond,
                        llvm::ConstantInt::get(cond->getType(), 0), "ifconst.cond");
                auto *thenBB  = llvm::BasicBlock::Create(ctx_, "ifconst.then", env.fn);
                auto *elseBB  = llvm::BasicBlock::Create(ctx_, "ifconst.else", env.fn);
                auto *mergeBB = llvm::BasicBlock::Create(ctx_, "ifconst.merge", env.fn);
                if (cond) builder_.CreateCondBr(cond, thenBB, ics.else_ ? elseBB : mergeBB);
                builder_.SetInsertPoint(thenBB);
                genStmt(*ics.then, env);
                if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(mergeBB);
                if (ics.else_) {
                    builder_.SetInsertPoint(elseBB);
                    genStmt(*ics.else_, env);
                    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(mergeBB);
                }
                builder_.SetInsertPoint(mergeBB);
            }
        } else if (*ics.constResult) {
            if (ics.then) genStmt(*ics.then, env);
        } else {
            if (ics.else_) genStmt(*ics.else_, env);
        }
        break;
    }
    case StmtKind::Label: {
        auto &ls = static_cast<LabelStmt &>(s);
        // Use pre-collected BB if available, otherwise create new
        llvm::BasicBlock *bb = nullptr;
        auto it = env.gotoLabels.find(ls.label);
        if (it != env.gotoLabels.end()) {
            bb = it->second;
        } else {
            bb = llvm::BasicBlock::Create(ctx_, ls.label, env.fn);
            env.gotoLabels[ls.label] = bb;
        }
        if (!isTerminated(builder_.GetInsertBlock()))
            builder_.CreateBr(bb);
        builder_.SetInsertPoint(bb);
        genStmt(*ls.body, env);
        break;
    }
    case StmtKind::Goto: {
        auto &gs = static_cast<GotoStmt &>(s);
        auto it = env.gotoLabels.find(gs.label);
        if (it != env.gotoLabels.end()) {
            builder_.CreateBr(it->second);
        } else {
            // Forward goto — create BB now, will be used when label is visited
            auto *bb = llvm::BasicBlock::Create(ctx_, gs.label, env.fn);
            env.gotoLabels[gs.label] = bb;
            builder_.CreateBr(bb);
        }
        // Code after goto is unreachable — create a new BB for it
        auto *deadBB = llvm::BasicBlock::Create(ctx_, "post.goto", env.fn);
        builder_.SetInsertPoint(deadBB);
        break;
    }
    case StmtKind::Asm:
        genAsm(static_cast<AsmStmt &>(s), env);
        break;
    default: break;
    }
}

// Pre-collect labels from a statement tree to create forward-declared BBs for goto
void CodeGen::collectGotoLabels(Stmt &s, FnEnv &env) {
    switch (s.kind) {
    case StmtKind::Label: {
        auto &ls = static_cast<LabelStmt &>(s);
        if (env.gotoLabels.find(ls.label) == env.gotoLabels.end()) {
            env.gotoLabels[ls.label] =
                llvm::BasicBlock::Create(ctx_, ls.label, env.fn);
        }
        if (ls.body) collectGotoLabels(*ls.body, env);
        break;
    }
    case StmtKind::Compound: {
        auto &cs = static_cast<CompoundStmt &>(s);
        for (auto &stmt : cs.body) collectGotoLabels(*stmt, env);
        break;
    }
    case StmtKind::If: {
        auto &is = static_cast<IfStmt &>(s);
        if (is.then) collectGotoLabels(*is.then, env);
        if (is.else_) collectGotoLabels(*is.else_, env);
        break;
    }
    case StmtKind::While: case StmtKind::DoWhile: {
        auto &ws = static_cast<WhileStmt &>(s);
        if (ws.body) collectGotoLabels(*ws.body, env);
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<ForStmt &>(s);
        if (fs.body) collectGotoLabels(*fs.body, env);
        break;
    }
    case StmtKind::Unsafe: {
        auto &us = static_cast<UnsafeStmt &>(s);
        if (us.body) collectGotoLabels(*us.body, env);
        break;
    }
    case StmtKind::Switch: {
        auto &sw = static_cast<SwitchStmt &>(s);
        for (auto &c : sw.cases)
            for (auto &stmt : c.body) collectGotoLabels(*stmt, env);
        break;
    }
    default: break;
    }
}

void CodeGen::genCompound(CompoundStmt &s, FnEnv &env) {
    for (auto &stmt : s.body) {
        // Once a statement has terminated the current block (return, or an
        // if/else where both arms already return, etc.), any further
        // statements in this compound are unreachable dead code. Emitting IR
        // for them would insert instructions after the block's terminator —
        // invalid IR ("terminator found in the middle of a basic block").
        if (builder_.GetInsertBlock() && builder_.GetInsertBlock()->getTerminator())
            break;
        genStmt(*stmt, env);
    }
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

    env.pushLoop(exitBB, headerBB, s.label);

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

    env.pushLoop(exitBB, incrBB, s.label);
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

void CodeGen::genSwitch(SwitchStmt &s, FnEnv &env) {
    auto *subjectVal = genExpr(*s.controlling, env);
    auto *exitBB = llvm::BasicBlock::Create(ctx_, "switch.end", env.fn);

    // One real llvm::BasicBlock per case, in source order, so an
    // unterminated case body's natural fallthrough is just "the next
    // block in sequence" — exactly how C fallthrough already works at
    // the machine level, no special-casing needed here beyond wiring the
    // branches up that way.
    std::vector<llvm::BasicBlock *> blocks;
    blocks.reserve(s.cases.size());
    llvm::BasicBlock *defaultBB = exitBB; // no 'default:' → falls past the switch
    for (auto &c : s.cases) {
        auto *bb = llvm::BasicBlock::Create(ctx_, c.isDefault ? "switch.default" : "switch.case", env.fn);
        blocks.push_back(bb);
        if (c.isDefault) defaultBB = bb;
    }

    unsigned numDistinctValues = 0;
    for (auto &c : s.cases) numDistinctValues += (unsigned)c.values.size();
    auto *swInst = builder_.CreateSwitch(subjectVal, defaultBB, numDistinctValues);
    auto *subjectTy = subjectVal->getType();
    for (size_t i = 0; i < s.cases.size(); ++i) {
        for (int64_t v : s.cases[i].values) {
            swInst->addCase(llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(subjectTy), v, true),
                             blocks[i]);
        }
    }

    // Unlabeled 'break' exits the switch; 'continue' isn't ours to
    // consume, so it keeps referring to whatever loop (if any) already
    // encloses this switch — matches real C's "switch doesn't establish
    // a continue target" rule.
    env.pushLoop(exitBB, env.continueBB(), "");
    for (size_t i = 0; i < s.cases.size(); ++i) {
        builder_.SetInsertPoint(blocks[i]);
        for (auto &stmt : s.cases[i].body) genStmt(*stmt, env);
        if (!isTerminated(builder_.GetInsertBlock())) {
            auto *fallTo = (i + 1 < blocks.size()) ? blocks[i + 1] : exitBB;
            builder_.CreateBr(fallTo);
        }
    }
    env.popLoop();

    builder_.SetInsertPoint(exitBB);
}

llvm::Value *CodeGen::coerceScalar(llvm::Value *val, llvm::Type *targetTy,
                                    const TypePtr &srcType) {
    if (val->getType() == targetTy) return val;
    if (val->getType()->isIntegerTy() && targetTy->isIntegerTy()) {
        unsigned srcBits = val->getType()->getIntegerBitWidth();
        unsigned dstBits = targetTy->getIntegerBitWidth();
        if (dstBits < srcBits) return builder_.CreateTrunc(val, targetTy, "coerce");
        if (dstBits == srcBits) return val;
        // An i1 is always the 0/1 result of a bool expression (comparison,
        // logical op, etc.) — it must zero-extend so 'true' widens to 1, not
        // -1. This can't be folded into the srcType lookup below: Bool sits
        // outside the Int8..UInt64 range that lookup checks, so without this
        // it silently fell through to the signed-by-default case.
        if (srcBits == 1) return builder_.CreateZExt(val, targetTy, "coerce");
        bool isSigned = true; // default: plain 'int'/'long' etc. — the common case
        if (srcType && srcType->kind >= TypeKind::Int8 && srcType->kind <= TypeKind::UInt64)
            isSigned = static_cast<PrimType &>(*srcType).isSigned();
        return isSigned ? builder_.CreateSExt(val, targetTy, "coerce")
                         : builder_.CreateZExt(val, targetTy, "coerce");
    }
    if (val->getType()->isFloatingPointTy() && targetTy->isFloatingPointTy())
        return val->getType()->getPrimitiveSizeInBits() < targetTy->getPrimitiveSizeInBits()
               ? builder_.CreateFPExt(val, targetTy, "coerce")
               : builder_.CreateFPTrunc(val, targetTy, "coerce");
    // Integer → float widening (see canImplicitlyConvert's matching rule
    // in Sema.cpp): an i1 is always an unsigned 0/1 bool result (same
    // reasoning as the ZExt special-case above), everything else defaults
    // signed unless srcType says otherwise.
    if (val->getType()->isIntegerTy() && targetTy->isFloatingPointTy()) {
        bool isSigned = true;
        if (val->getType()->getIntegerBitWidth() == 1) {
            isSigned = false;
        } else if (srcType && srcType->kind >= TypeKind::Int8 && srcType->kind <= TypeKind::UInt64) {
            isSigned = static_cast<PrimType &>(*srcType).isSigned();
        }
        return isSigned ? builder_.CreateSIToFP(val, targetTy, "coerce")
                         : builder_.CreateUIToFP(val, targetTy, "coerce");
    }
    return val;
}

llvm::Value *CodeGen::coerceToOptional(llvm::Value *val, const TypePtr &srcType,
                                        const TypePtr &targetTy) {
    if (!targetTy || targetTy->kind != TypeKind::Optional) return val;
    // Already Optional-shaped (e.g. re-returning an existing '?T' local, or
    // a call that itself returns '?T') — nothing to wrap.
    if (srcType && srcType->kind == TypeKind::Optional) return val;

    auto &ot = static_cast<OptionalType &>(*targetTy);
    auto *optTy = lowerType(targetTy);

    // null → empty '?T': {zeroinitializer-of-T, false}. 'null' is always
    // typed as 'void*' (see checkIntLit's NullLit sibling in Sema), so a
    // void*-typed source is the only realistic way to reach here without
    // already being the inner type.
    bool isNull = srcType && srcType->kind == TypeKind::Pointer &&
                  static_cast<PointerType &>(*srcType).base->isVoid();
    if (isNull) {
        auto *innerTy = lowerType(ot.inner);
        llvm::Value *agg = llvm::UndefValue::get(optTy);
        agg = builder_.CreateInsertValue(agg, llvm::Constant::getNullValue(innerTy), {0}, "opt.none");
        agg = builder_.CreateInsertValue(agg, builder_.getFalse(), {1});
        return agg;
    }

    // Plain T → present '?T': {val, true}.
    llvm::Value *agg = llvm::UndefValue::get(optTy);
    agg = builder_.CreateInsertValue(agg, val, {0}, "opt.some");
    agg = builder_.CreateInsertValue(agg, builder_.getTrue(), {1});
    return agg;
}

void CodeGen::genReturn(ReturnStmt &s, FnEnv &env) {
    if (s.value && env.returnSlot) {
        auto *val     = genExpr(*s.value, env);
        auto *slotTy  = static_cast<llvm::AllocaInst *>(env.returnSlot)->getAllocatedType();
        val = coerceScalar(val, slotTy, s.value->type);
        val = coerceToOptional(val, s.value->type, env.fnDecl ? env.fnDecl->returnType : nullptr);
        builder_.CreateStore(val, env.returnSlot);
    }
    // Emit deferred statements in LIFO order before branching to return block
    emitDefers(env, /*isErrorPath=*/false);
    builder_.CreateBr(env.returnBB);
}

void CodeGen::emitDefers(FnEnv &env, bool isErrorPath) {
    for (int i = (int)env.deferList.size() - 1; i >= 0; --i) {
        auto *ds = static_cast<DeferStmt *>(env.deferList[i]);
        // Normal exit: skip errdefers (they're for the error path only).
        // Error exit: run everything — plain defers still owe their
        // cleanup, on top of whatever errdefers add.
        if (isErrorPath || !ds->isErrDefer)
            genStmt(*ds->body, env);
    }
}

// ── Inline assembly codegen ──────────────────────────────────────────────────
// GCC/Clang extended-asm source syntax (which SafeC's own asm{} statement
// mirrors) references operands as '%N' (or '%<modifier><N>', e.g. '%b0' for
// an 8-bit sub-register view of operand 0). LLVM IR's inline-asm *template*
// string uses a different operand-reference syntax entirely: '$N' (or
// '${N:modifier}') — bare '%' has no special meaning to LLVM's AsmPrinter
// at all. Real clang performs exactly this translation when lowering a C
// asm() to LLVM IR; genAsm previously passed s.asmTemplate straight through
// unmodified, so any asm block referencing an operand (almost every one
// with an input or output) produced IR whose template LLVM's own
// AsmPrinter/MC layer had no operand to substitute for — the literal text
// '%0' reached the target assembler, which naturally didn't understand it.
static std::string translateAsmTemplate(const std::string &gcc) {
    std::string out;
    out.reserve(gcc.size());
    for (size_t i = 0; i < gcc.size(); ++i) {
        char c = gcc[i];
        if (c != '%') { out += c; continue; }
        if (i + 1 >= gcc.size()) { out += c; continue; }
        char n = gcc[i + 1];
        if (n == '%') { out += '%'; ++i; continue; } // GCC '%%' -> literal '%'
        if (std::isdigit(static_cast<unsigned char>(n))) {
            size_t j = i + 1;
            while (j < gcc.size() && std::isdigit(static_cast<unsigned char>(gcc[j]))) ++j;
            out += '$';
            out += gcc.substr(i + 1, j - i - 1);
            i = j - 1;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(n)) && i + 2 < gcc.size() &&
            std::isdigit(static_cast<unsigned char>(gcc[i + 2]))) {
            size_t j = i + 2;
            while (j < gcc.size() && std::isdigit(static_cast<unsigned char>(gcc[j]))) ++j;
            out += "${";
            out += gcc.substr(i + 2, j - i - 2);
            out += ':';
            out += n;
            out += '}';
            i = j - 1;
            continue;
        }
        // Bare '%' not followed by a recognized operand form (e.g. a raw
        // '%eax'-style register name written directly in the template) —
        // not a GCC operand reference at all; pass through unchanged.
        out += c;
    }
    return out;
}

void CodeGen::genAsm(AsmStmt &s, FnEnv &env) {
    // Build constraint string: outputs,inputs,~clobbers — LLVM's InlineAsm
    // expects output constraints first (they map onto the call's return
    // type, struct-packed if there's more than one output), then input
    // constraints (mapping onto the call's argument list, in order), then
    // clobbers.
    std::string constraints;
    for (size_t i = 0; i < s.outputs.size(); ++i) {
        if (i > 0) constraints += ",";
        constraints += s.outputs[i];
    }
    for (size_t i = 0; i < s.inputs.size(); ++i) {
        if (!constraints.empty()) constraints += ",";
        constraints += s.inputs[i];
    }
    for (size_t i = 0; i < s.clobbers.size(); ++i) {
        if (!constraints.empty()) constraints += ",";
        constraints += "~{" + s.clobbers[i] + "}";
    }

    // Output operands (e.g. "=r"(v) in 'asm("mrs %0,x" : "=r"(v))') are NOT
    // call arguments in GCC/Clang extended-asm semantics — the instruction
    // *produces* a value that becomes the call's own return value (struct-
    // packed across all outputs when there's more than one), which the
    // caller then stores into each output lvalue after the call returns.
    // Only plain input operands (no '=' / '+' prefix) are passed as actual
    // call arguments. The previous implementation passed every output's
    // address as an ordinary argument and always used a void return type,
    // which mismatched any real "=r"-style constraint (LLVM's asm verifier
    // rejects the resulting operand-count mismatch) and had never worked.
    std::vector<llvm::Type *> outTys;
    for (auto &e : s.outputExprs)
        outTys.push_back(e && e->type ? lowerType(e->type) : llvm::Type::getInt32Ty(ctx_));

    std::vector<llvm::Value *> argVals;
    std::vector<llvm::Type *> argTys;
    for (auto &e : s.inputExprs) {
        if (!e) continue;
        auto *v = genExpr(*e, env);
        argVals.push_back(v);
        argTys.push_back(v->getType());
    }

    llvm::Type *resTy;
    if (outTys.empty())          resTy = llvm::Type::getVoidTy(ctx_);
    else if (outTys.size() == 1) resTy = outTys[0];
    else                          resTy = llvm::StructType::get(ctx_, outTys);

    auto *asmFnTy = llvm::FunctionType::get(resTy, argTys, false);
    auto *inlineAsm = llvm::InlineAsm::get(
        asmFnTy, translateAsmTemplate(s.asmTemplate), constraints,
        s.isVolatile || s.outputs.empty());
    auto *call = builder_.CreateCall(asmFnTy, inlineAsm, argVals);

    // Store each output register's value back into its lvalue.
    for (size_t i = 0; i < s.outputExprs.size(); ++i) {
        auto &e = s.outputExprs[i];
        if (!e) continue;
        auto *dst = genLValue(*e, env);
        llvm::Value *val = (outTys.size() == 1)
            ? static_cast<llvm::Value *>(call)
            : builder_.CreateExtractValue(call, (unsigned)i);
        builder_.CreateStore(val, dst);
    }
}

void CodeGen::genMatch(MatchStmt &s, FnEnv &env) {
    auto *subjectVal = genExpr(*s.subject, env);
    auto *exitBB = llvm::BasicBlock::Create(ctx_, "match.end", env.fn);

    // Check if subject is a tagged union (struct type with tag+payload layout)
    bool isTaggedUnion = false;
    llvm::StructType *unionTy = nullptr;
    if (s.subject->type && s.subject->type->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*s.subject->type);
        if (st.isTaggedUnion) {
            isTaggedUnion = true;
            unionTy = lowerStructType(st);
        }
    }
    // A nullable pointer/reference or Optional subject: 'null'/'some(x)' (or
    // 'none'/'some(x)') patterns behave like a two-variant tagged union
    // (tag 0 = empty, tag 1 = present) without an actual struct-with-tag
    // layout — see resolveMatchArmPatterns in Sema.cpp for the pattern
    // resolution side.
    bool isNullLike = false; // Pointer, or nullable Reference
    bool isOptional = false;
    if (s.subject->type) {
        if (s.subject->type->kind == TypeKind::Pointer) {
            isNullLike = true;
        } else if (s.subject->type->kind == TypeKind::Reference &&
                   static_cast<ReferenceType &>(*s.subject->type).nullable) {
            isNullLike = true;
        } else if (s.subject->type->kind == TypeKind::Optional) {
            isOptional = true;
        }
    }
    bool hasTag = isTaggedUnion || isNullLike || isOptional;
    // Plain 'enum' subject: EnumIdent patterns compare directly against
    // subjectVal (an integer), using pat.resolvedTag as resolved by
    // Sema's resolveMatchArmPatterns — see that function's comment for why
    // this branch exists (without it, every EnumIdent pattern against a
    // plain enum silently matched unconditionally).
    bool isPlainEnum = s.subject->type && s.subject->type->kind == TypeKind::Enum;

    // For tagged unions, store subject and extract tag
    llvm::Value *tagVal = nullptr;
    llvm::AllocaInst *subjectAlloca = nullptr;
    if (isTaggedUnion && unionTy) {
        subjectAlloca = createEntryAlloca(env, unionTy, "match.subject");
        builder_.CreateStore(subjectVal, subjectAlloca);
        auto *tagGEP = builder_.CreateStructGEP(unionTy, subjectAlloca, 0, "match.tag.ptr");
        tagVal = builder_.CreateLoad(llvm::Type::getInt32Ty(ctx_), tagGEP, "match.tag");
    } else if (isNullLike) {
        auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
        auto *notNull = builder_.CreateICmpNE(subjectVal, nullPtr, "match.notnull");
        tagVal = builder_.CreateZExt(notNull, llvm::Type::getInt32Ty(ctx_), "match.tag");
    } else if (isOptional) {
        auto *hasVal = builder_.CreateExtractValue(subjectVal, {1}, "match.hasval");
        tagVal = builder_.CreateZExt(hasVal, llvm::Type::getInt32Ty(ctx_), "match.tag");
    }

    for (auto &arm : s.arms) {
        auto *bodyBB = llvm::BasicBlock::Create(ctx_, "match.arm",  env.fn);
        auto *nextBB = llvm::BasicBlock::Create(ctx_, "match.next", env.fn);

        // Build OR-combined condition for this arm's patterns
        llvm::Value *cond = nullptr;
        for (auto &pat : arm.patterns) {
            llvm::Value *thisCond = nullptr;
            switch (pat.kind) {
            case PatternKind::Wildcard:
                thisCond = builder_.getTrue();
                break;
            case PatternKind::IntLit: {
                auto *cmpVal = hasTag ? tagVal : subjectVal;
                auto *lit = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal, true);
                thisCond  = builder_.CreateICmpEQ(cmpVal, lit, "match.eq");
                break;
            }
            case PatternKind::Range: {
                auto *cmpVal = hasTag ? tagVal : subjectVal;
                auto *lo  = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal,  true);
                auto *hi  = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal2, true);
                auto *geq = builder_.CreateICmpSGE(cmpVal, lo, "match.ge");
                auto *leq = builder_.CreateICmpSLE(cmpVal, hi, "match.le");
                thisCond  = builder_.CreateAnd(geq, leq, "match.range");
                break;
            }
            case PatternKind::EnumIdent:
                if (hasTag && pat.resolvedTag >= 0) {
                    auto *lit = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_),
                                                        pat.resolvedTag, true);
                    thisCond = builder_.CreateICmpEQ(tagVal, lit, "match.tag.eq");
                } else if (isPlainEnum && pat.resolvedTag >= 0) {
                    auto *lit = llvm::ConstantInt::get(subjectVal->getType(),
                                                        pat.resolvedTag, true);
                    thisCond = builder_.CreateICmpEQ(subjectVal, lit, "match.enum.eq");
                } else {
                    thisCond = builder_.getTrue();
                }
                break;
            }
            cond = cond ? builder_.CreateOr(cond, thisCond, "match.or") : thisCond;
        }
        if (!cond) cond = builder_.getTrue();

        builder_.CreateCondBr(cond, bodyBB, nextBB);

        builder_.SetInsertPoint(bodyBB);

        // For tagged union patterns with bind names, extract payload
        if (isTaggedUnion && subjectAlloca) {
            for (auto &pat : arm.patterns) {
                if (!pat.bindName.empty() && pat.payloadType && pat.resolvedTag >= 0) {
                    auto *payloadGEP = builder_.CreateStructGEP(
                        unionTy, subjectAlloca, 1, "match.payload.ptr");
                    auto *payloadTy = lowerType(pat.payloadType);
                    auto *payloadVal = builder_.CreateLoad(payloadTy, payloadGEP,
                                                            pat.bindName);
                    auto *bindAlloca = createEntryAlloca(env, payloadTy, pat.bindName);
                    builder_.CreateStore(payloadVal, bindAlloca);
                    env.locals[pat.bindName] = bindAlloca;
                }
            }
        } else if (isNullLike || isOptional) {
            // 'some(x)': bind x to the dereferenced pointer (nullable
            // pointer/reference) or the '{T,i1}' value field (Optional) —
            // safe to read unconditionally here since we're already inside
            // the branch this arm's own tag==1 condition selected.
            for (auto &pat : arm.patterns) {
                if (!pat.bindName.empty() && pat.payloadType && pat.resolvedTag >= 0) {
                    llvm::Value *payloadVal;
                    if (isNullLike) {
                        auto *payloadTy = lowerType(pat.payloadType);
                        payloadVal = builder_.CreateLoad(payloadTy, subjectVal, pat.bindName);
                    } else {
                        payloadVal = builder_.CreateExtractValue(subjectVal, {0}, pat.bindName);
                    }
                    auto *bindAlloca = createEntryAlloca(env, payloadVal->getType(), pat.bindName);
                    builder_.CreateStore(payloadVal, bindAlloca);
                    env.locals[pat.bindName] = bindAlloca;
                }
            }
        }

        genStmt(*arm.body, env);
        if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(exitBB);

        builder_.SetInsertPoint(nextBB);
    }

    // No arm matched → fall through to exit
    if (!isTerminated(builder_.GetInsertBlock())) builder_.CreateBr(exitBB);
    builder_.SetInsertPoint(exitBB);
}

// Same pattern-matching structure as genMatch above, but each arm produces
// a value that's merged via a PHI at the exit block instead of running a
// statement for effect. See MatchExpr::provenExhaustive in AST.h — Sema
// already rejected the program if some subject value could reach neither a
// wildcard/default arm nor a covered tagged-union variant, so the block
// left dangling after the last arm is unreachable at runtime; it's marked
// 'unreachable' rather than branched into the merge, which would otherwise
// need a PHI incoming edge with no value to offer.
llvm::Value *CodeGen::genMatchExpr(MatchExpr &e, FnEnv &env) {
    auto *subjectVal = genExpr(*e.subject, env);
    auto *exitBB = llvm::BasicBlock::Create(ctx_, "match.expr.end", env.fn);

    bool isTaggedUnion = false;
    llvm::StructType *unionTy = nullptr;
    if (e.subject->type && e.subject->type->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*e.subject->type);
        if (st.isTaggedUnion) {
            isTaggedUnion = true;
            unionTy = lowerStructType(st);
        }
    }
    // See genMatch's identical comment above — 'null'/'some(x)' and
    // 'none'/'some(x)' behave like a two-variant tagged union without an
    // actual struct-with-tag layout.
    bool isNullLike = false;
    bool isOptional = false;
    if (e.subject->type) {
        if (e.subject->type->kind == TypeKind::Pointer) {
            isNullLike = true;
        } else if (e.subject->type->kind == TypeKind::Reference &&
                   static_cast<ReferenceType &>(*e.subject->type).nullable) {
            isNullLike = true;
        } else if (e.subject->type->kind == TypeKind::Optional) {
            isOptional = true;
        }
    }
    bool hasTag = isTaggedUnion || isNullLike || isOptional;
    // See genMatch's identical comment above.
    bool isPlainEnum = e.subject->type && e.subject->type->kind == TypeKind::Enum;

    llvm::Value *tagVal = nullptr;
    llvm::AllocaInst *subjectAlloca = nullptr;
    if (isTaggedUnion && unionTy) {
        subjectAlloca = createEntryAlloca(env, unionTy, "match.expr.subject");
        builder_.CreateStore(subjectVal, subjectAlloca);
        auto *tagGEP = builder_.CreateStructGEP(unionTy, subjectAlloca, 0, "match.expr.tag.ptr");
        tagVal = builder_.CreateLoad(llvm::Type::getInt32Ty(ctx_), tagGEP, "match.expr.tag");
    } else if (isNullLike) {
        auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
        auto *notNull = builder_.CreateICmpNE(subjectVal, nullPtr, "match.expr.notnull");
        tagVal = builder_.CreateZExt(notNull, llvm::Type::getInt32Ty(ctx_), "match.expr.tag");
    } else if (isOptional) {
        auto *hasVal = builder_.CreateExtractValue(subjectVal, {1}, "match.expr.hasval");
        tagVal = builder_.CreateZExt(hasVal, llvm::Type::getInt32Ty(ctx_), "match.expr.tag");
    }

    llvm::Type *resultTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
    std::vector<std::pair<llvm::Value *, llvm::BasicBlock *>> incoming;

    for (auto &arm : e.arms) {
        auto *bodyBB = llvm::BasicBlock::Create(ctx_, "match.expr.arm",  env.fn);
        auto *nextBB = llvm::BasicBlock::Create(ctx_, "match.expr.next", env.fn);

        // Build OR-combined condition for this arm's patterns (identical
        // logic to genMatch's arm-condition loop above).
        llvm::Value *cond = nullptr;
        for (auto &pat : arm.patterns) {
            llvm::Value *thisCond = nullptr;
            switch (pat.kind) {
            case PatternKind::Wildcard:
                thisCond = builder_.getTrue();
                break;
            case PatternKind::IntLit: {
                auto *cmpVal = hasTag ? tagVal : subjectVal;
                auto *lit = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal, true);
                thisCond  = builder_.CreateICmpEQ(cmpVal, lit, "match.expr.eq");
                break;
            }
            case PatternKind::Range: {
                auto *cmpVal = hasTag ? tagVal : subjectVal;
                auto *lo  = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal,  true);
                auto *hi  = llvm::ConstantInt::get(cmpVal->getType(), pat.intVal2, true);
                auto *geq = builder_.CreateICmpSGE(cmpVal, lo, "match.expr.ge");
                auto *leq = builder_.CreateICmpSLE(cmpVal, hi, "match.expr.le");
                thisCond  = builder_.CreateAnd(geq, leq, "match.expr.range");
                break;
            }
            case PatternKind::EnumIdent:
                if (hasTag && pat.resolvedTag >= 0) {
                    auto *lit = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_),
                                                        pat.resolvedTag, true);
                    thisCond = builder_.CreateICmpEQ(tagVal, lit, "match.expr.tag.eq");
                } else if (isPlainEnum && pat.resolvedTag >= 0) {
                    auto *lit = llvm::ConstantInt::get(subjectVal->getType(),
                                                        pat.resolvedTag, true);
                    thisCond = builder_.CreateICmpEQ(subjectVal, lit, "match.expr.enum.eq");
                } else {
                    thisCond = builder_.getTrue();
                }
                break;
            }
            cond = cond ? builder_.CreateOr(cond, thisCond, "match.expr.or") : thisCond;
        }
        if (!cond) cond = builder_.getTrue();

        builder_.CreateCondBr(cond, bodyBB, nextBB);

        builder_.SetInsertPoint(bodyBB);

        // For tagged union patterns with bind names, extract payload
        if (isTaggedUnion && subjectAlloca) {
            for (auto &pat : arm.patterns) {
                if (!pat.bindName.empty() && pat.payloadType && pat.resolvedTag >= 0) {
                    auto *payloadGEP = builder_.CreateStructGEP(
                        unionTy, subjectAlloca, 1, "match.expr.payload.ptr");
                    auto *payloadTy = lowerType(pat.payloadType);
                    auto *payloadVal = builder_.CreateLoad(payloadTy, payloadGEP,
                                                            pat.bindName);
                    auto *bindAlloca = createEntryAlloca(env, payloadTy, pat.bindName);
                    builder_.CreateStore(payloadVal, bindAlloca);
                    env.locals[pat.bindName] = bindAlloca;
                }
            }
        } else if (isNullLike || isOptional) {
            // See genMatch's identical comment above.
            for (auto &pat : arm.patterns) {
                if (!pat.bindName.empty() && pat.payloadType && pat.resolvedTag >= 0) {
                    llvm::Value *payloadVal;
                    if (isNullLike) {
                        auto *payloadTy = lowerType(pat.payloadType);
                        payloadVal = builder_.CreateLoad(payloadTy, subjectVal, pat.bindName);
                    } else {
                        payloadVal = builder_.CreateExtractValue(subjectVal, {0}, pat.bindName);
                    }
                    auto *bindAlloca = createEntryAlloca(env, payloadVal->getType(), pat.bindName);
                    builder_.CreateStore(payloadVal, bindAlloca);
                    env.locals[pat.bindName] = bindAlloca;
                }
            }
        }

        auto *armVal = genExpr(*arm.value, env);
        armVal = coerceScalar(armVal, resultTy, arm.value->type);
        auto *armEnd = builder_.GetInsertBlock();
        if (!isTerminated(armEnd)) {
            builder_.CreateBr(exitBB);
            incoming.emplace_back(armVal, armEnd);
        }

        builder_.SetInsertPoint(nextBB);
    }

    builder_.CreateUnreachable();

    builder_.SetInsertPoint(exitBB);
    auto *phi = builder_.CreatePHI(resultTy, (unsigned)incoming.size(), "match.expr");
    for (auto &pair : incoming) phi->addIncoming(pair.first, pair.second);
    return phi;
}

// fn_eval(object, func) — Sema has already resolved e.matchedMethod (the
// concrete method on object's struct type); 'object'/'func' themselves are
// never evaluated here (see FnEvalExpr's comment in AST.h — object is only
// ever used for its type, same as sizeof's operand). This is exactly the
// existing "bare function name as a value" mechanism (see genIdent's
// llvm::Function* fast path) applied to a method's already-mangled symbol —
// no new runtime machinery, no vtable: the method's prototype was already
// emitted in generate()'s Pass 1 over every top-level function (methods are
// ordinary top-level FunctionDecls with isMethod/methodOwner set), so this
// is always just a lookup by then.
llvm::Value *CodeGen::genFnEval(FnEvalExpr &e) {
    if (!e.matchedMethod) {
        diag_.error(e.loc, "codegen: fn_eval did not resolve to a method");
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
    }
    if (auto *fn = mod_->getFunction(e.matchedMethod->name)) return fn;
    // Defensive fallback — shouldn't happen given Pass 1 always runs first.
    return genFunctionProto(*e.matchedMethod);
}

void CodeGen::genVarDecl(VarDeclStmt &s, FnEnv &env) {
    auto typeToUse = s.resolvedType ? s.resolvedType : s.declType;
    // Function pointer types (fn RetType(Params)) must be stored as opaque ptr,
    // since llvm::FunctionType is not a valid alloca type in LLVM IR.
    llvm::Type *allocaTy = (typeToUse && typeToUse->kind == TypeKind::Function)
        ? llvm::PointerType::get(ctx_, 0)
        : lowerType(typeToUse);

    // ── Static local variables → module-level global with internal linkage ──
    if (s.isStatic) {
        std::string globalName = env.fnDecl->name + "." + s.name;
        auto *gv = new llvm::GlobalVariable(
            *mod_, allocaTy, /*isConstant=*/false,
            llvm::GlobalValue::InternalLinkage,
            llvm::Constant::getNullValue(allocaTy),
            globalName);
        if (s.isThreadLocal)
            gv->setThreadLocalMode(llvm::GlobalVariable::LocalExecTLSModel);
        // Register in the globals_ map so genIdent finds it via the global path
        globals_[s.name] = gv;
        if (s.init) {
            // Static init: only run once (use a guard flag)
            std::string guardName = globalName + ".guard";
            auto *guardGV = new llvm::GlobalVariable(
                *mod_, llvm::Type::getInt1Ty(ctx_), false,
                llvm::GlobalValue::InternalLinkage,
                llvm::ConstantInt::getFalse(ctx_), guardName);
            auto *guardVal = builder_.CreateLoad(llvm::Type::getInt1Ty(ctx_),
                                                  guardGV, "guard");
            auto *initBB = llvm::BasicBlock::Create(ctx_, "static.init", env.fn);
            auto *contBB = llvm::BasicBlock::Create(ctx_, "static.cont", env.fn);
            builder_.CreateCondBr(guardVal, contBB, initBB);
            builder_.SetInsertPoint(initBB);
            auto *val = genExpr(*s.init, env);
            val = coerceToOptional(val, s.init->type, typeToUse);
            builder_.CreateStore(val, gv);
            builder_.CreateStore(llvm::ConstantInt::getTrue(ctx_), guardGV);
            builder_.CreateBr(contBB);
            builder_.SetInsertPoint(contBB);
        }
        return;
    }

    // Variable-length array ('unsafe { int arr[n]; }', n not a compile-time
    // constant — see ArrayType::size==-2 in ConstEval.cpp/Sema.cpp). 'alloca'
    // here already gets the pointer type lowerType() gives *any* size<0
    // array (parameter-decay and VLA share that rule), so the only new work
    // is computing the runtime length and dynamically sizing the backing
    // storage — the variable itself still behaves like a plain T* for
    // subscripting, decay, etc., exactly like an unsized-array parameter.
    if (typeToUse && typeToUse->kind == TypeKind::Array &&
        static_cast<ArrayType &>(*typeToUse).size == -2) {
        auto &at = static_cast<ArrayType &>(*typeToUse);
        auto *sizeExpr = static_cast<Expr *>(at.sizeExpr);
        auto *countVal = genExpr(*sizeExpr, env);
        auto *elemTy   = lowerType(at.element);
        countVal = builder_.CreateIntCast(countVal, llvm::Type::getInt64Ty(ctx_),
                                           /*isSigned=*/true, "vla.count");
        auto *dataPtr = builder_.CreateAlloca(elemTy, countVal, "vla.data");
        auto *ptrSlot = createEntryAlloca(env, allocaTy, s.name);
        builder_.CreateStore(dataPtr, ptrSlot);
        env.locals[s.name] = ptrSlot;
        return;
    }

    auto *alloca = createEntryAlloca(env, allocaTy, s.name);
    if (s.alignment > 0)
        alloca->setAlignment(llvm::Align(s.alignment));
    env.locals[s.name] = alloca;

    // Full debug: emit DILocalVariable + dbg.declare
    if (debugLevel_ == DebugLevel::Full && dib_ && env.fn->getSubprogram()) {
        auto *diVar = dib_->createAutoVariable(
            env.fn->getSubprogram(), s.name, diFile_,
            s.loc.line ? s.loc.line : 1,
            diTypeFor(typeToUse),
            /*alwaysPreserve=*/false);
        dib_->insertDeclare(
            alloca, diVar,
            dib_->createExpression(),
            llvm::DILocation::get(ctx_,
                s.loc.line ? s.loc.line : 1,
                s.loc.col,
                env.fn->getSubprogram()),
            builder_.GetInsertBlock());
    }

    if (s.init) {
        // Array-typed initializer flowing into a pointer/reference-typed
        // declaration (e.g. '&static T x = some_array;') needs decay to the
        // array's address, same as a call argument would.
        bool needsDecay = s.init->type && s.init->type->kind == TypeKind::Array &&
                          typeToUse && (typeToUse->kind == TypeKind::Pointer ||
                                        typeToUse->kind == TypeKind::Reference);
        auto *val = needsDecay ? decayArrayToPtr(*s.init, env) : genExpr(*s.init, env);
        val = coerceScalar(val, allocaTy, s.init->type);
        val = coerceToOptional(val, s.init->type, typeToUse);
        builder_.CreateStore(val, alloca);
    } else {
        // Zero-initialize (null function pointer)
        builder_.CreateStore(llvm::Constant::getNullValue(allocaTy), alloca);
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
    case ExprKind::Match:     return genMatchExpr(static_cast<MatchExpr &>(e), env);
    case ExprKind::FnEval:    return genFnEval(static_cast<FnEvalExpr &>(e));
    case ExprKind::Call:      return genCall(static_cast<CallExpr &>(e), env);
    case ExprKind::Subscript: return genSubscript(static_cast<SubscriptExpr &>(e), env);
    case ExprKind::Slice:     return genSlice(static_cast<SliceExpr &>(e), env);
    case ExprKind::Member:
    case ExprKind::Arrow:     return genMember(static_cast<MemberExpr &>(e), env);
    case ExprKind::Cast:      return genCast(static_cast<CastExpr &>(e), env);
    case ExprKind::Assign:    return genAssign(static_cast<AssignExpr &>(e), env);
    case ExprKind::SizeofType: {
        auto &st = static_cast<SizeofTypeExpr &>(e);
        auto *llvmTy = lowerType(st.ofType);
        const auto &dl = mod_->getDataLayout();
        uint64_t sz = dl.getTypeAllocSize(llvmTy);
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), sz);
    }
    case ExprKind::AlignofType: {
        auto &ae = static_cast<AlignofTypeExpr &>(e);
        auto *llvmTy = lowerType(ae.ofType);
        const auto &dl = mod_->getDataLayout();
        uint64_t al = dl.getABITypeAlign(llvmTy).value();
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), al);
    }
    case ExprKind::FieldCount: {
        auto &fc = static_cast<FieldCountExpr &>(e);
        if (fc.ofType && fc.ofType->kind == TypeKind::Struct) {
            auto &st = static_cast<StructType &>(*fc.ofType);
            return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_),
                                           st.fields.size());
        }
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0);
    }
    case ExprKind::SizeofPack: {
        auto &sp = static_cast<SizeofPackExpr &>(e);
        return llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), sp.resolvedCount);
    }
    case ExprKind::TaggedUnionInit: {
        // Should not appear directly — handled via CallExpr with taggedUnionTag
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }
    case ExprKind::GenericSelection: {
        // Compile-time selection (Sema::checkGenericSelection already
        // picked the association) — only the winning expression is ever
        // generated; the others don't need to be codegen-able at all
        // (real C11 doesn't evaluate them either), matching how e.g. a
        // 'const'-folded 'if const' branch elides its other side.
        auto &gs = static_cast<GenericSelectionExpr &>(e);
        if (gs.selectedIndex < 0 || gs.selectedIndex >= (int)gs.associations.size())
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        return genExpr(*gs.associations[gs.selectedIndex].expr, env);
    }
    case ExprKind::Compound: {
        // Aggregate initializer: build via a temp alloca + per-element
        // GEP+store. Handles designators via Sema's resolvedSlots (empty
        // resolvedSlots = purely positional, slot == source position,
        // identical to the pre-designator behavior). Also fixes a real bug
        // in array-typed compound inits: this used to fall through to
        // 'return genExpr(ci.inits[0])' for anything that wasn't a struct,
        // silently leaving every element after the first as uninitialized
        // stack garbage instead of actually storing each one.
        auto &ci = static_cast<CompoundInitExpr &>(e);
        auto *ty = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
        if (ci.inits.empty()) return llvm::Constant::getNullValue(ty);
        // vec<T,N> v = {a, b, ...} — built directly as an SSA value via
        // chained insertelement, no memory round-trip (unlike struct/array
        // below): a SIMD register never needed an address to begin with.
        if (ty->isVectorTy()) {
            llvm::Value *vecVal = llvm::Constant::getNullValue(ty);
            auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
            auto *elemTy  = llvm::cast<llvm::VectorType>(ty)->getElementType();
            for (size_t i = 0; i < ci.inits.size(); ++i) {
                int64_t slot = ci.resolvedSlots.empty() ? (int64_t)i : ci.resolvedSlots[i];
                auto *val = coerceScalar(genExpr(*ci.inits[i], env), elemTy, ci.inits[i]->type);
                vecVal = builder_.CreateInsertElement(
                    vecVal, val, llvm::ConstantInt::get(Int32Ty, (uint64_t)slot), "vec.init");
            }
            return vecVal;
        }
        if (ty->isStructTy() || ty->isArrayTy()) {
            auto *alloca = createEntryAlloca(env, ty, "compound.init");
            // Zero-init first: designators can skip slots, and a shorter
            // initializer list than the aggregate's size is valid C
            // ('int arr[100] = {0};' zero-fills elements 1..99).
            builder_.CreateStore(llvm::Constant::getNullValue(ty), alloca);
            auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
            for (size_t i = 0; i < ci.inits.size(); ++i) {
                int64_t slot = ci.resolvedSlots.empty() ? (int64_t)i : ci.resolvedSlots[i];
                auto *val = genExpr(*ci.inits[i], env);
                llvm::Value *gep = ty->isStructTy()
                    ? builder_.CreateStructGEP(ty, alloca, (unsigned)slot)
                    : builder_.CreateGEP(ty, alloca,
                          { llvm::ConstantInt::get(Int32Ty, 0),
                            llvm::ConstantInt::get(Int32Ty, (uint64_t)slot) });
                auto *slotTy = ty->isStructTy() ? ty->getStructElementType((unsigned)slot)
                                                 : llvm::cast<llvm::ArrayType>(ty)->getElementType();
                val = coerceScalar(val, slotTy, ci.inits[i]->type);
                builder_.CreateStore(val, gep);
            }
            return builder_.CreateLoad(ty, alloca, "compound");
        }
        // Fallback: first element
        return genExpr(*ci.inits[0], env);
    }
    case ExprKind::New:
        return genNew(static_cast<NewExpr &>(e), env);
    case ExprKind::TupleLit:
        return genTupleLit(static_cast<TupleLitExpr &>(e), env);
    case ExprKind::Spawn:
        return genSpawn(static_cast<SpawnExpr &>(e), env);
    case ExprKind::Try: {
        // try expr — unwrap ?T or early-return null optional
        auto &te = static_cast<TryExpr &>(e);
        auto *optVal = genExpr(*te.inner, env);
        // Extract has_value bit (index 1 in the { T, i1 } struct)
        auto *hasVal = builder_.CreateExtractValue(optVal, {1}, "try.has");
        auto *okBB  = llvm::BasicBlock::Create(ctx_, "try.ok",  env.fn);
        auto *errBB = llvm::BasicBlock::Create(ctx_, "try.err", env.fn);
        builder_.CreateCondBr(hasVal, okBB, errBB);

        // Error path: return a zero/null optional from this function
        builder_.SetInsertPoint(errBB);
        env.hasError = true;
        emitDefers(env, /*isErrorPath=*/true);
        if (env.returnSlot) {
            auto *slotTy = static_cast<llvm::AllocaInst *>(env.returnSlot)->getAllocatedType();
            builder_.CreateStore(llvm::Constant::getNullValue(slotTy), env.returnSlot);
        }
        builder_.CreateBr(env.returnBB);

        // Success path: extract inner value (index 0)
        builder_.SetInsertPoint(okBB);
        return builder_.CreateExtractValue(optVal, {0}, "try.val");
    }
    default:
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }
}

llvm::Value *CodeGen::decayArrayToPtr(Expr &a, FnEnv &env) {
    auto *arrAddr = genLValue(a, env);
    auto *arrTy   = lowerType(a.type);
    auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
    return builder_.CreateGEP(arrTy, arrAddr,
        { llvm::ConstantInt::get(Int64Ty, 0), llvm::ConstantInt::get(Int64Ty, 0) },
        "arraydecay");
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
    case ExprKind::Unary: {
        auto &ue = static_cast<UnaryExpr &>(e);
        // 'x!' on a pointer/nullable-reference (Sema only sets isLValue in
        // that case, never for Optional — see checkUnary) is the same
        // address as a direct '*x': both lower to a plain opaque ptr, so
        // the pointer value itself already *is* that address, letting
        // 'x! = value' reuse this exactly like '*x = value' does above.
        if (ue.op == UnaryOp::ForceUnwrap) return genExpr(*ue.operand, env);
        diag_.error(e.loc, "cannot take address of this expression");
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
    }
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
        if (wantAddr) {
            // Decayed array parameter (see genFunctionProto/genFunctionBody):
            // Sema still types 'e' as Array, but the alloca holds a POINTER
            // to the caller's storage, not the array itself — so the
            // "address" callers need for GEP-based subscripting is the
            // pointer VALUE (a load), not this alloca's own address. A real
            // array-typed local's alloca IS the storage, so its address is
            // correct as-is; distinguish by checking what got allocated.
            // Excludes a VLA (ArrayType::size == -2, see genVarDecl): that
            // alloca is *itself* a plain pointer-typed local variable slot
            // (holding the address of the separately-allocated dynamic
            // storage), exactly like an ordinary 'T* p' — its own address
            // is what genSubscript's pointer-fallback path expects to load
            // the pointer value from, same one level of indirection as any
            // other pointer variable, not two.
            bool isVLA = e.type && e.type->kind == TypeKind::Array &&
                         static_cast<ArrayType &>(*e.type).size == -2;
            if (e.type && e.type->kind == TypeKind::Array && !isVLA &&
                !it->second->getAllocatedType()->isArrayTy()) {
                return builder_.CreateLoad(it->second->getAllocatedType(), it->second, e.name);
            }
            return it->second;
        }
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
        // e.operand->type (Sema's resolved type), not
        // 'static_cast<AllocaInst*>(addr)->getAllocatedType()' — addr is
        // only actually an AllocaInst for a plain local; for 'arr[i]++' or
        // 's.field++', genLValue returns a GEP into that storage instead,
        // and static_cast'ing a non-AllocaInst Value* to AllocaInst* to
        // call an AllocaInst-only accessor is undefined behavior (in
        // practice: reading whichever type happened to sit at that
        // reinterpreted field, e.g. the *whole array's* type instead of
        // one element's — which then fails IR verification when used in
        // an add).
        auto *ty = e.operand->type ? lowerType(e.operand->type) : llvm::Type::getInt32Ty(ctx_);
        auto *val = builder_.CreateLoad(ty, addr);
        llvm::Value *inc = ty->isFloatingPointTy()
            ? builder_.CreateFAdd(val, llvm::ConstantFP::get(ty, 1.0), "inc")
            : builder_.CreateAdd(val, llvm::ConstantInt::get(ty, 1), "inc");
        builder_.CreateStore(inc, addr);
        return inc;
    }
    case UnaryOp::PreDec: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty = e.operand->type ? lowerType(e.operand->type) : llvm::Type::getInt32Ty(ctx_);
        auto *val = builder_.CreateLoad(ty, addr);
        llvm::Value *dec = ty->isFloatingPointTy()
            ? builder_.CreateFSub(val, llvm::ConstantFP::get(ty, 1.0), "dec")
            : builder_.CreateSub(val, llvm::ConstantInt::get(ty, 1), "dec");
        builder_.CreateStore(dec, addr);
        return dec;
    }
    case UnaryOp::PostInc: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty = e.operand->type ? lowerType(e.operand->type) : llvm::Type::getInt32Ty(ctx_);
        auto *old = builder_.CreateLoad(ty, addr, "old");
        llvm::Value *inc = ty->isFloatingPointTy()
            ? builder_.CreateFAdd(old, llvm::ConstantFP::get(ty, 1.0))
            : builder_.CreateAdd(old, llvm::ConstantInt::get(ty, 1));
        builder_.CreateStore(inc, addr);
        return old;
    }
    case UnaryOp::PostDec: {
        auto *addr = genLValue(*e.operand, env);
        auto *ty = e.operand->type ? lowerType(e.operand->type) : llvm::Type::getInt32Ty(ctx_);
        auto *old = builder_.CreateLoad(ty, addr, "old");
        llvm::Value *dec = ty->isFloatingPointTy()
            ? builder_.CreateFSub(old, llvm::ConstantFP::get(ty, 1.0))
            : builder_.CreateSub(old, llvm::ConstantInt::get(ty, 1));
        builder_.CreateStore(dec, addr);
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
    case UnaryOp::ForceUnwrap: {
        // No runtime has_value/null check — '!' trusts the caller (Sema
        // already required 'unsafe' to reach here), matching every other
        // unsafe-gated operation's zero-runtime-cost philosophy elsewhere
        // in this compiler (e.g. raw pointer deref).
        auto &operandTy = e.operand->type;
        if (operandTy && operandTy->kind == TypeKind::Optional) {
            // Optional lowers to '{T, i1}' — extract the value directly.
            auto *val = genExpr(*e.operand, env);
            return builder_.CreateExtractValue(val, {0}, "force_unwrap");
        }
        // Pointer or nullable Reference: both lower to a plain opaque ptr
        // (see lowerType) — '!' just dereferences it, same as '*x'.
        auto *ptrVal = genExpr(*e.operand, env);
        llvm::Type *innerTy = llvm::Type::getInt32Ty(ctx_);
        if (operandTy && operandTy->kind == TypeKind::Pointer)
            innerTy = lowerType(static_cast<PointerType &>(*operandTy).base);
        else if (operandTy && operandTy->kind == TypeKind::Reference)
            innerTy = lowerType(static_cast<ReferenceType &>(*operandTy).base);
        return builder_.CreateLoad(innerTy, ptrVal, "force_unwrap");
    }
    default:
        return genExpr(*e.operand, env);
    }
}

llvm::Value *CodeGen::genAddrOf(UnaryExpr &e, FnEnv &env) {
    // '&funcName' — see checkAddrOf's matching exemption in Sema.cpp: a
    // bare function name is not an lvalue (genLValue has no alloca to find
    // an address of), but already evaluates to the function's pointer
    // value directly via genExpr, same as the cast-less/'&'-less form.
    if (e.operand->kind == ExprKind::Ident &&
        static_cast<IdentExpr &>(*e.operand).resolvedFn) {
        return genExpr(*e.operand, env);
    }
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
    // Comma: 'l' was only evaluated for its side effects and is discarded —
    // unlike every other case below, this is not "combine two operands of a
    // shared type," so it must bypass the integer/float width-unification
    // logic just below. That logic keys off 'ty' (the *left* operand's
    // type, passed in by genBinary for every op) and would otherwise widen
    // 'r' up to 'l's width whenever the left operand's type happens to be
    // wider (e.g. '(someLongLong, someInt)') — corrupting the comma's
    // result to a bit-width that doesn't match its actual (right-operand)
    // static type, exactly the "doesn't work like C" gap this exists to
    // close: in C, '(a, b)' is always simply b's value and type.
    if (op == BinaryOp::Comma) return r;

    // vec<T,N> (std::simd): float-vs-integer dispatch must look at the
    // element type T, not the vector type itself (VectorType::isFloat() is
    // always false — its own 'kind' is TypeKind::Vector — so a
    // vec<float,4> add would otherwise wrongly take the integer CreateAdd
    // path below instead of CreateFAdd).
    const TypePtr &scalarTy = (ty && ty->kind == TypeKind::Vector)
        ? static_cast<VectorType &>(*ty).element : ty;
    bool isFloat = scalarTy && scalarTy->isFloat();
    bool isSigned = !scalarTy || (scalarTy->kind == TypeKind::Int32 || scalarTy->kind == TypeKind::Int64 ||
                             scalarTy->kind == TypeKind::Int16 || scalarTy->kind == TypeKind::Int8);

    // Widen integer operands to the same bit-width (C integer promotion).
    // This handles cases like `long long x < 0` where 0 is i32 but x is i64.
    //
    // Wrap/saturate ops are the deliberate exception: their entire point is
    // to overflow/clamp at a *specific* declared width (e.g. 'uint8_t x +%
    // 10' saturating at 255), and Sema types a bare int literal like '10'
    // as plain 'int' (i32) — widening the i8 operand up to i32 before the
    // op would make the add/saturate run at i32 width, where an i8-range
    // value can never overflow, silently turning "saturate at 255" into
    // "never saturates, then truncate() at the result's cast back to i8
    // wraps instead of clamping". Truncating the *wider* operand down to
    // the narrower one's width instead keeps the op running at the width
    // that was actually declared.
    bool isWrapOrSat = op == BinaryOp::WrapAdd || op == BinaryOp::WrapSub || op == BinaryOp::WrapMul ||
                        op == BinaryOp::SatAdd  || op == BinaryOp::SatSub  || op == BinaryOp::SatMul;
    if (!isFloat && l->getType()->isIntegerTy() && r->getType()->isIntegerTy()) {
        unsigned lw = l->getType()->getIntegerBitWidth();
        unsigned rw = r->getType()->getIntegerBitWidth();
        if (isWrapOrSat && lw != rw) {
            if (lw > rw)      l = builder_.CreateTrunc(l, r->getType(), "sat.trunc");
            else               r = builder_.CreateTrunc(r, l->getType(), "sat.trunc");
        } else if (lw < rw) {
            l = isSigned ? builder_.CreateSExt(l, r->getType(), "sext")
                         : builder_.CreateZExt(l, r->getType(), "zext");
        } else if (rw < lw) {
            r = isSigned ? builder_.CreateSExt(r, l->getType(), "sext")
                         : builder_.CreateZExt(r, l->getType(), "zext");
        }
    }
    // Widen float operands to a common precision (e.g., float vs double).
    if (isFloat && l->getType()->isFloatingPointTy() && r->getType()->isFloatingPointTy()) {
        if (l->getType() != r->getType()) {
            auto *wider = (l->getType()->getPrimitiveSizeInBits() >=
                           r->getType()->getPrimitiveSizeInBits())
                          ? l->getType() : r->getType();
            if (l->getType() != wider) l = builder_.CreateFPExt(l, wider, "fpext");
            if (r->getType() != wider) r = builder_.CreateFPExt(r, wider, "fpext");
        }
    }

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
        // Unordered-or-not-equal, not ordered-not-equal: C/IEEE754 '!=' is
        // the negation of '==', and '==' is false for any NaN operand, so
        // '!=' must be true whenever either operand is NaN (including the
        // classic 'x != x' NaN-detection idiom, e.g. isnan_d/isnan_f) —
        // 'one' (ordered) incorrectly returns false when a NaN is involved.
        return isFloat ? builder_.CreateFCmpUNE(l, r, "fne")
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
    // ── Wrapping arithmetic (no undefined behavior, modular) ────────────────
    case BinaryOp::WrapAdd:
        return builder_.CreateAdd(l, r, "wrapadd", /*NUW=*/false, /*NSW=*/false);
    case BinaryOp::WrapSub:
        return builder_.CreateSub(l, r, "wrapsub", /*NUW=*/false, /*NSW=*/false);
    case BinaryOp::WrapMul:
        return builder_.CreateMul(l, r, "wrapmul", /*NUW=*/false, /*NSW=*/false);
    // ── Saturating arithmetic (clamp at min/max) ────────────────────────────
    case BinaryOp::SatAdd: {
        auto id = isSigned ? llvm::Intrinsic::sadd_sat : llvm::Intrinsic::uadd_sat;
        return builder_.CreateBinaryIntrinsic(id, l, r, nullptr, "satadd");
    }
    case BinaryOp::SatSub: {
        auto id = isSigned ? llvm::Intrinsic::ssub_sat : llvm::Intrinsic::usub_sat;
        return builder_.CreateBinaryIntrinsic(id, l, r, nullptr, "satsub");
    }
    case BinaryOp::SatMul: {
        // No direct LLVM sat mul intrinsic — use overflow intrinsic + select
        auto id = isSigned ? llvm::Intrinsic::smul_with_overflow
                           : llvm::Intrinsic::umul_with_overflow;
        auto *result = builder_.CreateBinaryIntrinsic(id, l, r);
        auto *val = builder_.CreateExtractValue(result, 0, "mul.val");
        auto *ovf = builder_.CreateExtractValue(result, 1, "mul.ovf");
        llvm::Value *maxVal;
        unsigned bw = l->getType()->getIntegerBitWidth();
        if (isSigned)
            maxVal = llvm::ConstantInt::get(l->getType(),
                llvm::APInt::getSignedMaxValue(bw));
        else
            maxVal = llvm::ConstantInt::get(l->getType(),
                llvm::APInt::getMaxValue(bw));
        return builder_.CreateSelect(ovf, maxVal, val, "satmul");
    }
    default: return l;
    }
}

llvm::Value *CodeGen::genBinary(BinaryExpr &e, FnEnv &env) {
    // Operator overload via resolvedOperator (M1)
    if (e.resolvedOperator) {
        auto *fn = mod_->getFunction(e.resolvedOperator->name);
        if (!fn) fn = genFunctionProto(*e.resolvedOperator);
        // Same rule as a regular method call's self (see the methodBase
        // handling below): the operator's receiver needs an address, but
        // 'e.left' isn't always addressable on its own -- a chained call
        // like 'complex_new(x,0.0) * zInv' has a temporary as its left
        // operand. Materialize it into a fresh alloca in that case instead
        // of failing through genLValue's "not addressable" default case.
        llvm::Value *lhs = e.left->isLValue ? genLValue(*e.left, env) : nullptr;
        if (!lhs) {
            lhs = builder_.CreateAlloca(lowerType(e.left->type), nullptr, "opovl.lhs.tmp");
            builder_.CreateStore(genExpr(*e.left, env), lhs);
        }
        auto *rhs = genExpr(*e.right, env);
        return builder_.CreateCall(fn, {lhs, rhs}, "opovl");
    }

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

    // Array+int arithmetic: T[N] + int → T* via GEP (array decay)
    if (e.left->type && e.left->type->kind == TypeKind::Array &&
        (e.op == BinaryOp::Add || e.op == BinaryOp::Sub)) {
        auto *arrAddr  = genLValue(*e.left, env);
        auto *arrTy    = lowerType(e.left->type);
        auto *Int64Ty  = llvm::Type::getInt64Ty(ctx_);
        auto *arrDecay = builder_.CreateGEP(arrTy, arrAddr,
            { llvm::ConstantInt::get(Int64Ty, 0), llvm::ConstantInt::get(Int64Ty, 0) },
            "arrdecay");
        auto *rhs = genExpr(*e.right, env);
        // Extend index to i64 if needed
        if (!rhs->getType()->isIntegerTy(64))
            rhs = builder_.CreateSExt(rhs, Int64Ty, "idx64");
        if (e.op == BinaryOp::Sub) rhs = builder_.CreateNeg(rhs);
        auto &at = static_cast<ArrayType &>(*e.left->type);
        return builder_.CreateGEP(lowerType(at.element), arrDecay, rhs, "ptrarith");
    }

    auto *lhs = genExpr(*e.left,  env);
    auto *rhs = genExpr(*e.right, env);

    // Pointer - pointer = ptrdiff_t (element distance), e.g. 'found - base'.
    // Must be distinguished from pointer +/- integer (GEP offset) below: both
    // operands here are addresses, so the result is their byte distance
    // (scaled by pointee size), not a new address — treating rhs as a GEP
    // index would try to integer-negate a pointer value, which isn't valid IR.
    if (lhs->getType()->isPointerTy() && rhs->getType()->isPointerTy() &&
        e.op == BinaryOp::Sub) {
        auto *i64Ty  = llvm::Type::getInt64Ty(ctx_);
        auto *lhsInt = builder_.CreatePtrToInt(lhs, i64Ty, "ptrdiff.lhs");
        auto *rhsInt = builder_.CreatePtrToInt(rhs, i64Ty, "ptrdiff.rhs");
        auto *diff   = builder_.CreateSub(lhsInt, rhsInt, "ptrdiff");
        TypePtr pointee;
        if (e.left->type) {
            if (e.left->type->kind == TypeKind::Pointer)
                pointee = static_cast<PointerType &>(*e.left->type).base;
            else if (e.left->type->kind == TypeKind::Reference)
                pointee = static_cast<ReferenceType &>(*e.left->type).base;
        }
        if (pointee) {
            uint64_t elemSize = mod_->getDataLayout().getTypeAllocSize(lowerType(pointee));
            if (elemSize > 1)
                diff = builder_.CreateSDiv(diff, llvm::ConstantInt::get(i64Ty, elemSize), "ptrdiff.scaled");
        }
        return diff;
    }

    // Pointer arithmetic (e.g., &arr[i+offset]). The GEP element type must
    // be the pointee's actual type so the index gets scaled by sizeof(T) —
    // using i8 here would silently turn 'p + i' into a raw byte offset
    // instead of the i-th element's address for any T wider than 1 byte.
    if (lhs->getType()->isPointerTy() && (e.op == BinaryOp::Add || e.op == BinaryOp::Sub)) {
        TypePtr pointee;
        if (e.left->type) {
            if (e.left->type->kind == TypeKind::Pointer)
                pointee = static_cast<PointerType &>(*e.left->type).base;
            else if (e.left->type->kind == TypeKind::Reference)
                pointee = static_cast<ReferenceType &>(*e.left->type).base;
        }
        auto *elemTy = pointee ? lowerType(pointee) : llvm::Type::getInt8Ty(ctx_);
        if (!rhs->getType()->isIntegerTy(64))
            rhs = builder_.CreateSExt(rhs, llvm::Type::getInt64Ty(ctx_), "idx64");
        if (e.op == BinaryOp::Sub) rhs = builder_.CreateNeg(rhs);
        return builder_.CreateGEP(elemTy, lhs, rhs, "ptrarith");
    }

    return applyBinaryOp(e.op, lhs, rhs, e.left->type);
}

// AArch64 AAPCS64: a struct argument that is a "Homogeneous Floating-point
// Aggregate" (all members the same float/double type, 1-4 members total,
// e.g. Metal/CoreGraphics's CGRect = 4 doubles) is passed in consecutive
// SIMD/FP registers regardless of size. Every other aggregate over 16 bytes
// is passed "indirectly": the caller copies it to its own stack and passes
// a pointer in a general-purpose register/stack slot instead of the value
// itself. Only the second case needs 'byval' below — an HFA struct already
// round-trips correctly through this file's existing raw-aggregate-value
// call codegen (confirmed working today for gui_cocoa.sc's CGRect), and
// forcing byval onto it would wrongly switch it from FP-register passing to
// indirect-via-pointer passing, actively breaking a case that isn't broken.
static bool isHomogeneousFPAggregate(llvm::StructType *st) {
    unsigned n = st->getNumElements();
    if (n == 0 || n > 4) return false;
    llvm::Type *first = st->getElementType(0);
    if (!first->isFloatTy() && !first->isDoubleTy()) return false;
    for (unsigned i = 1; i < n; ++i) {
        if (st->getElementType(i) != first) return false;
    }
    return true;
}

// ── Call ──────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genCall(CallExpr &e, FnEnv &env) {
    if (e.nullOp != CallExpr::NullOp::None) {
        return genNullOp(e, env);
    }
    // ── Tagged union init: Shape.radius(5.0) ──────────────────────────────────
    if (e.taggedUnionTag >= 0 && !e.taggedUnionName.empty()) {
        // Find the LLVM struct type for the tagged union
        auto it = structCache_.find(e.taggedUnionName);
        llvm::StructType *unionTy = nullptr;
        if (it != structCache_.end()) {
            unionTy = it->second;
        } else if (e.type) {
            unionTy = static_cast<llvm::StructType *>(lowerType(e.type));
        }
        if (!unionTy) {
            diag_.error(e.loc, "codegen: unknown tagged union type '" + e.taggedUnionName + "'");
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        auto *alloca = createEntryAlloca(env, unionTy, "union.init");
        // Store tag (field 0: i32)
        auto *tagGEP = builder_.CreateStructGEP(unionTy, alloca, 0, "union.tag");
        builder_.CreateStore(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), e.taggedUnionTag),
            tagGEP);
        // Store payload (field 1: [N x i8]) — store value via pointer cast
        if (!e.args.empty()) {
            auto *payloadGEP = builder_.CreateStructGEP(unionTy, alloca, 1, "union.payload");
            auto *val = genExpr(*e.args[0], env);
            // With opaque pointers (LLVM 15+), payloadGEP is `ptr` so we can
            // store any type through it. The payload area is large enough.
            builder_.CreateStore(val, payloadGEP);
        }
        return builder_.CreateLoad(unionTy, alloca, "union.val");
    }

    // Handle __arena_reset_<R>() builtin
    if (e.callee->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.callee);
        if (ident.name.substr(0, 14) == "__arena_reset_") {
            std::string regionName = ident.name.substr(14);
            auto it = arenaStateMap_.find(regionName);
            if (it != arenaStateMap_.end()) {
                auto &info = it->second;
                auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
                // Reset used = 0
                auto *usedPtr = builder_.CreateStructGEP(info.ty, info.var, 1, "arena.used.ptr");
                builder_.CreateStore(llvm::ConstantInt::get(Int64Ty, 0), usedPtr);
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
            }
        }
        // Handle __arena_destroy_<R>() builtin: frees the region's backing
        // buffer (unlike __arena_reset_ above, which only rewinds the bump
        // pointer for reuse). Safe to call more than once or before the
        // region's first allocation — buf is null-checked first, and
        // nulled out afterward so a subsequent new<R> re-mallocs cleanly
        // (mirroring genNew's own null-check-then-malloc logic) and a
        // second arena_destroy<R>() call is a no-op instead of a double free.
        if (ident.name.substr(0, 16) == "__arena_destroy_") {
            std::string regionName = ident.name.substr(16);
            auto it = arenaStateMap_.find(regionName);
            if (it != arenaStateMap_.end()) {
                auto &info = it->second;
                auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
                auto *PtrTy   = llvm::PointerType::get(ctx_, 0);

                auto *bufPtr = builder_.CreateStructGEP(info.ty, info.var, 0, "arena.buf.ptr");
                auto *buf    = builder_.CreateLoad(PtrTy, bufPtr, "arena.buf");
                auto *isNull = builder_.CreateIsNull(buf, "arena.destroy.null");
                auto *freeBB = llvm::BasicBlock::Create(ctx_, "arena.destroy.free", env.fn);
                auto *contBB = llvm::BasicBlock::Create(ctx_, "arena.destroy.cont", env.fn);
                builder_.CreateCondBr(isNull, contBB, freeBB);

                builder_.SetInsertPoint(freeBB);
                auto freeFn = mod_->getOrInsertFunction("free",
                    llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {PtrTy}, false));
                builder_.CreateCall(freeFn, {buf});
                builder_.CreateStore(llvm::ConstantPointerNull::get(PtrTy), bufPtr);
                auto *usedPtr = builder_.CreateStructGEP(info.ty, info.var, 1, "arena.used.ptr");
                builder_.CreateStore(llvm::ConstantInt::get(Int64Ty, 0), usedPtr);
                builder_.CreateBr(contBB);

                builder_.SetInsertPoint(contBB);
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
            }
        }
        // Handle __arena_mark_<R>() builtin: returns the region's current
        // byte offset ('used') as a checkpoint for __arena_free_to_<R>().
        if (ident.name.substr(0, 13) == "__arena_mark_") {
            std::string regionName = ident.name.substr(13);
            auto it = arenaStateMap_.find(regionName);
            if (it != arenaStateMap_.end()) {
                auto &info = it->second;
                auto *usedPtr = builder_.CreateStructGEP(info.ty, info.var, 1, "arena.used.ptr");
                return builder_.CreateLoad(llvm::Type::getInt64Ty(ctx_), usedPtr, "arena.mark");
            }
        }
        // Handle __arena_free_to_<R>(mark) builtin: rewinds 'used' to
        // min(used, mark) — a partial free that keeps everything allocated
        // before the checkpoint. The min() clamp means passing a mark
        // larger than the current 'used' (e.g. a stale mark from before an
        // intervening arena_reset<R>()) is a harmless no-op rather than
        // corrupting 'used' with a bogus larger value.
        if (ident.name.substr(0, 16) == "__arena_free_to_") {
            std::string regionName = ident.name.substr(16);
            auto it = arenaStateMap_.find(regionName);
            if (it != arenaStateMap_.end() && !e.args.empty()) {
                auto &info = it->second;
                auto *Int64Ty = llvm::Type::getInt64Ty(ctx_);
                auto *usedPtr = builder_.CreateStructGEP(info.ty, info.var, 1, "arena.used.ptr");
                auto *used    = builder_.CreateLoad(Int64Ty, usedPtr, "arena.used");
                auto *mark    = genExpr(*e.args[0], env);
                auto *le      = builder_.CreateICmpULE(used, mark, "arena.freeto.le");
                auto *newUsed = builder_.CreateSelect(le, used, mark, "arena.freeto.new");
                builder_.CreateStore(newUsed, usedPtr);
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
            }
        }
        // ── volatile_load / volatile_store built-ins ─────────────────────────
        if (ident.name == "volatile_load" && !e.args.empty()) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *loadTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
            auto *ld = builder_.CreateLoad(loadTy, ptr, "vol.load");
            ld->setVolatile(true);
            return ld;
        }
        if (ident.name == "volatile_store" && e.args.size() >= 2) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *val = genExpr(*e.args[1], env);
            auto *st = builder_.CreateStore(val, ptr);
            st->setVolatile(true);
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        // ── Atomic built-ins ─────────────────────────────────────────────────
        auto parseOrdering = [&](int argIdx) -> llvm::AtomicOrdering {
            if (argIdx < (int)e.args.size() && e.args[argIdx]->kind == ExprKind::StringLit) {
                auto &str = static_cast<StringLitExpr &>(*e.args[argIdx]).value;
                if (str == "relaxed")  return llvm::AtomicOrdering::Monotonic;
                if (str == "acquire")  return llvm::AtomicOrdering::Acquire;
                if (str == "release")  return llvm::AtomicOrdering::Release;
                if (str == "acq_rel")  return llvm::AtomicOrdering::AcquireRelease;
            }
            return llvm::AtomicOrdering::SequentiallyConsistent;
        };
        if (ident.name == "atomic_load" && !e.args.empty()) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *loadTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
            auto *ld = builder_.CreateLoad(loadTy, ptr, "atomic.load");
            ld->setAtomic(parseOrdering(1));
            return ld;
        }
        if (ident.name == "atomic_store" && e.args.size() >= 2) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *val = genExpr(*e.args[1], env);
            auto *st = builder_.CreateStore(val, ptr);
            st->setAtomic(parseOrdering(2));
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        if ((ident.name == "atomic_fetch_add" || ident.name == "atomic_fetch_sub") &&
            e.args.size() >= 2) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *val = genExpr(*e.args[1], env);
            auto op = (ident.name == "atomic_fetch_add")
                ? llvm::AtomicRMWInst::Add : llvm::AtomicRMWInst::Sub;
            return builder_.CreateAtomicRMW(op, ptr, val, llvm::MaybeAlign(),
                                             parseOrdering(2));
        }
        if ((ident.name == "atomic_fetch_and" || ident.name == "atomic_fetch_or" ||
             ident.name == "atomic_fetch_xor") && e.args.size() >= 2) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *val = genExpr(*e.args[1], env);
            auto op = llvm::AtomicRMWInst::And;
            if (ident.name == "atomic_fetch_or")  op = llvm::AtomicRMWInst::Or;
            if (ident.name == "atomic_fetch_xor") op = llvm::AtomicRMWInst::Xor;
            return builder_.CreateAtomicRMW(op, ptr, val, llvm::MaybeAlign(),
                                             parseOrdering(2));
        }
        if (ident.name == "atomic_compare_exchange_strong" && e.args.size() >= 3) {
            auto *ptr         = genExpr(*e.args[0], env);
            auto *expectedPtr = genExpr(*e.args[1], env);
            auto *desired     = genExpr(*e.args[2], env);
            auto *elemTy      = desired->getType();
            auto *expectedVal = builder_.CreateLoad(elemTy, expectedPtr, "cas.expected");
            auto ord = parseOrdering(3);
            auto *cmpxchg = builder_.CreateAtomicCmpXchg(
                ptr, expectedVal, desired, llvm::MaybeAlign(), ord, ord);
            auto *oldVal  = builder_.CreateExtractValue(cmpxchg, 0, "cas.old");
            auto *success = builder_.CreateExtractValue(cmpxchg, 1, "cas.ok");
            // C11 semantics: the 'expected' out-pointer is updated with the
            // actual prior value (a no-op on success, since old == expected).
            builder_.CreateStore(oldVal, expectedPtr);
            return success;
        }
        if (ident.name == "atomic_exchange" && e.args.size() >= 2) {
            auto *ptr = genExpr(*e.args[0], env);
            auto *val = genExpr(*e.args[1], env);
            return builder_.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg, ptr, val,
                                             llvm::MaybeAlign(), parseOrdering(2));
        }
        if (ident.name == "atomic_cas" && e.args.size() >= 3) {
            auto *ptr      = genExpr(*e.args[0], env);
            auto *expected = genExpr(*e.args[1], env);
            auto *desired  = genExpr(*e.args[2], env);
            auto ord = parseOrdering(3);
            auto *cmpxchg = builder_.CreateAtomicCmpXchg(
                ptr, expected, desired, llvm::MaybeAlign(), ord, ord);
            // Extract success flag (element 1 of { T, i1 })
            return builder_.CreateExtractValue(cmpxchg, 1, "cas.ok");
        }
        if (ident.name == "atomic_fence") {
            auto ord = parseOrdering(0);
            builder_.CreateFence(ord);
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        // ── ARM Cortex-M4/M7 DSP-extension built-ins ─────────────────────────
        // 1:1 mapping onto LLVM's 'llvm.arm.*' target intrinsics — all fixed
        // (i32,i32[,i32])->i32 signatures, verified against real compiler
        // output (llc -mtriple=thumbv7em-none-eabi -mcpu=cortex-m4 emits the
        // real SADD16/SMLAD/USAD8/SSAT/... instructions from these calls,
        // not a libcall or software emulation).
        if (ident.name.rfind("__arm_dsp_", 0) == 0) {
            static const std::pair<const char *, llvm::Intrinsic::ID> kArmDspIds[] = {
                {"__arm_dsp_qadd",    llvm::Intrinsic::arm_qadd},
                {"__arm_dsp_qsub",    llvm::Intrinsic::arm_qsub},
                {"__arm_dsp_qadd16",  llvm::Intrinsic::arm_qadd16},
                {"__arm_dsp_qadd8",   llvm::Intrinsic::arm_qadd8},
                {"__arm_dsp_qsub16",  llvm::Intrinsic::arm_qsub16},
                {"__arm_dsp_qsub8",   llvm::Intrinsic::arm_qsub8},
                {"__arm_dsp_sadd16",  llvm::Intrinsic::arm_sadd16},
                {"__arm_dsp_sadd8",   llvm::Intrinsic::arm_sadd8},
                {"__arm_dsp_ssub16",  llvm::Intrinsic::arm_ssub16},
                {"__arm_dsp_ssub8",   llvm::Intrinsic::arm_ssub8},
                {"__arm_dsp_uqadd16", llvm::Intrinsic::arm_uqadd16},
                {"__arm_dsp_uqadd8",  llvm::Intrinsic::arm_uqadd8},
                {"__arm_dsp_uqsub16", llvm::Intrinsic::arm_uqsub16},
                {"__arm_dsp_uqsub8",  llvm::Intrinsic::arm_uqsub8},
                {"__arm_dsp_smlad",   llvm::Intrinsic::arm_smlad},
                {"__arm_dsp_smladx",  llvm::Intrinsic::arm_smladx},
                {"__arm_dsp_smlsd",   llvm::Intrinsic::arm_smlsd},
                {"__arm_dsp_smlsdx",  llvm::Intrinsic::arm_smlsdx},
                {"__arm_dsp_smuad",   llvm::Intrinsic::arm_smuad},
                {"__arm_dsp_smuadx",  llvm::Intrinsic::arm_smuadx},
                {"__arm_dsp_smusd",   llvm::Intrinsic::arm_smusd},
                {"__arm_dsp_smusdx",  llvm::Intrinsic::arm_smusdx},
                {"__arm_dsp_usad8",   llvm::Intrinsic::arm_usad8},
                {"__arm_dsp_usada8",  llvm::Intrinsic::arm_usada8},
                {"__arm_dsp_ssat",    llvm::Intrinsic::arm_ssat},
                {"__arm_dsp_usat",    llvm::Intrinsic::arm_usat},
                {"__arm_dsp_ssat16",  llvm::Intrinsic::arm_ssat16},
                {"__arm_dsp_usat16",  llvm::Intrinsic::arm_usat16},
                {"__arm_dsp_sxtab16", llvm::Intrinsic::arm_sxtab16},
                {"__arm_dsp_uxtab16", llvm::Intrinsic::arm_uxtab16},
            };
            llvm::Intrinsic::ID id = llvm::Intrinsic::not_intrinsic;
            for (auto &p : kArmDspIds) if (ident.name == p.first) { id = p.second; break; }
            if (id == llvm::Intrinsic::not_intrinsic) {
                diag_.error(e.loc, "unknown ARM DSP builtin '" + ident.name + "'");
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
            }
            auto &triple = mod_->getTargetTriple();
            if (!triple.isARM() && !triple.isThumb()) {
                diag_.error(e.loc, "'" + ident.name +
                    "' requires an ARM target (compile with --target thumbv7em-... "
                    "or thumbv8m.main-...); current target is '" +
                    mod_->getTargetTriple().str() + "'");
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
            }
            auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
            std::vector<llvm::Value *> args;
            for (auto &a : e.args) {
                auto *v = genExpr(*a, env);
                if (v->getType() != Int32Ty && v->getType()->isIntegerTy()) {
                    bool argUnsigned = a->type &&
                        (a->type->kind == TypeKind::UInt8  || a->type->kind == TypeKind::UInt16 ||
                         a->type->kind == TypeKind::UInt32  || a->type->kind == TypeKind::UInt64);
                    v = builder_.CreateIntCast(v, Int32Ty, !argUnsigned, "dsp.arg");
                }
                args.push_back(v);
            }
            auto *fn = llvm::Intrinsic::getOrInsertDeclaration(mod_.get(), id, {});
            return builder_.CreateCall(fn, args, "dsp");
        }
        // ── GCC/Clang bit-manipulation built-ins (std/bit.sc) ────────────────
        // These map 1:1 onto LLVM intrinsics; the '32'/'ll' suffix just picks
        // the operand width.
        if (ident.name == "__builtin_popcount" || ident.name == "__builtin_popcountll") {
            auto *val = genExpr(*e.args[0], env);
            auto *fn  = llvm::Intrinsic::getOrInsertDeclaration(
                mod_.get(), llvm::Intrinsic::ctpop, {val->getType()});
            auto *r   = builder_.CreateCall(fn, {val}, "popcount");
            return builder_.CreateIntCast(r, llvm::Type::getInt32Ty(ctx_), false);
        }
        if (ident.name == "__builtin_clz" || ident.name == "__builtin_clzll" ||
            ident.name == "__builtin_ctz" || ident.name == "__builtin_ctzll") {
            auto *val = genExpr(*e.args[0], env);
            bool isClz = ident.name == "__builtin_clz" || ident.name == "__builtin_clzll";
            auto id = isClz ? llvm::Intrinsic::ctlz : llvm::Intrinsic::cttz;
            auto *fn = llvm::Intrinsic::getOrInsertDeclaration(mod_.get(), id, {val->getType()});
            // Matches GCC/Clang semantics: undefined on a zero input (callers
            // in std/bit.sc already special-case x == 0 before calling this).
            auto *isZeroUndef = llvm::ConstantInt::getTrue(ctx_);
            auto *r = builder_.CreateCall(fn, {val, isZeroUndef}, isClz ? "clz" : "ctz");
            return builder_.CreateIntCast(r, llvm::Type::getInt32Ty(ctx_), false);
        }
        if (ident.name == "__builtin_bswap32" || ident.name == "__builtin_bswap64") {
            auto *val = genExpr(*e.args[0], env);
            auto *fn  = llvm::Intrinsic::getOrInsertDeclaration(
                mod_.get(), llvm::Intrinsic::bswap, {val->getType()});
            return builder_.CreateCall(fn, {val}, "bswap");
        }
        // Handle __safec_join(handle) — dispatches per selectThreadBackend()
        if (ident.name == "__safec_join" && !e.args.empty()) {
            auto *handleVal = genExpr(*e.args[0], env);
            genThreadJoin(handleVal, env);
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        // ── Channel built-ins ───────────────────────────────────────────────
        // Channels are implemented as extern C functions (provided by std/thread.h
        // or a user-supplied runtime). The compiler emits calls to:
        //   void* __safec_chan_create(int capacity);
        //   bool  __safec_chan_send(void* ch, void* val_ptr);
        //   bool  __safec_chan_recv(void* ch, void* out_ptr);
        //   void  __safec_chan_close(void* ch);
        if (ident.name == "chan_create" && e.args.size() == 1) {
            auto *PtrTy   = llvm::PointerType::get(ctx_, 0);
            auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
            auto *fnTy = llvm::FunctionType::get(PtrTy, {Int32Ty}, false);
            auto fn = mod_->getOrInsertFunction("__safec_chan_create", fnTy);
            auto *capVal = genExpr(*e.args[0], env);
            if (capVal->getType() != Int32Ty)
                capVal = builder_.CreateIntCast(capVal, Int32Ty, true);
            return builder_.CreateCall(fnTy, fn.getCallee(), {capVal}, "chan");
        }
        if (ident.name == "chan_send" && e.args.size() == 2) {
            auto *PtrTy  = llvm::PointerType::get(ctx_, 0);
            auto *BoolTy = llvm::Type::getInt1Ty(ctx_);
            auto *fnTy = llvm::FunctionType::get(BoolTy, {PtrTy, PtrTy}, false);
            auto fn = mod_->getOrInsertFunction("__safec_chan_send", fnTy);
            auto *chVal  = genExpr(*e.args[0], env);
            auto *valPtr = genExpr(*e.args[1], env);
            return builder_.CreateCall(fnTy, fn.getCallee(), {chVal, valPtr}, "sent");
        }
        if (ident.name == "chan_recv" && e.args.size() == 2) {
            auto *PtrTy  = llvm::PointerType::get(ctx_, 0);
            auto *BoolTy = llvm::Type::getInt1Ty(ctx_);
            auto *fnTy = llvm::FunctionType::get(BoolTy, {PtrTy, PtrTy}, false);
            auto fn = mod_->getOrInsertFunction("__safec_chan_recv", fnTy);
            auto *chVal  = genExpr(*e.args[0], env);
            auto *outPtr = genExpr(*e.args[1], env);
            return builder_.CreateCall(fnTy, fn.getCallee(), {chVal, outPtr}, "recvd");
        }
        if (ident.name == "chan_close" && e.args.size() == 1) {
            auto *PtrTy  = llvm::PointerType::get(ctx_, 0);
            auto *VoidTy = llvm::Type::getVoidTy(ctx_);
            auto *fnTy = llvm::FunctionType::get(VoidTy, {PtrTy}, false);
            auto fn = mod_->getOrInsertFunction("__safec_chan_close", fnTy);
            auto *chVal = genExpr(*e.args[0], env);
            builder_.CreateCall(fnTy, fn.getCallee(), {chVal});
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
        }
        // ── spawn_scoped: same as spawn but emits a deferred join ───────────
        // (The defer itself is handled by Sema inserting a DeferStmt with
        // __safec_join — this just needs to produce the same handle spawn
        // does, via the same backend-dispatching path.)
        if (ident.name == "spawn_scoped" && e.args.size() == 2) {
            auto *fnVal  = genExpr(*e.args[0], env);
            auto *argVal = genExpr(*e.args[1], env);
            return genThreadCreate(fnVal, argVal, env);
        }
    }

    auto *calleeVal = genExpr(*e.callee, env);

    std::vector<llvm::Value *> args;

    // Method call: prepend 'self' pointer (address of base object)
    if (e.methodBase) {
        llvm::Value *selfPtr = nullptr;
        bool baseIsIndirection = e.methodBase->type &&
            (e.methodBase->type->kind == TypeKind::Pointer ||
             e.methodBase->type->kind == TypeKind::Reference);
        if (baseIsIndirection) {
            // 'p->method()', or 'self.method()' from inside another method
            // (self's type is '&stack T' — a Reference) — the base is
            // already a pointer/reference to the object, so its *value* (not
            // the address of the pointer/reference variable) is self. Using
            // genLValue here would add a spurious extra indirection — self
            // ends up pointing at the stack slot holding the pointer rather
            // than at the object, and every field read is garbage.
            selfPtr = genExpr(*e.methodBase, env);
        } else if (e.methodBase->isLValue) {
            selfPtr = genLValue(*e.methodBase, env);
        } else {
            // Base is a value expression — allocate temp storage
            selfPtr = builder_.CreateAlloca(lowerType(e.methodBase->type), nullptr, "self.tmp");
            builder_.CreateStore(genExpr(*e.methodBase, env), selfPtr);
        }
        args.push_back(selfPtr);
    }

    for (auto &a : e.args) {
        // Perform automatic array-to-pointer/reference decay for array arguments
        if (a->type && a->type->kind == TypeKind::Array)
            args.push_back(decayArrayToPtr(*a, env));
        else
            args.push_back(genExpr(*a, env));
    }

    // Helper: coerce argument types to match the function signature.
    // This fixes i1/i32 mismatches that arise when boolean comparisons
    // (ICmpXX → i1) are passed to functions expecting i32.
    unsigned argOffset = e.methodBase ? 1 : 0; // args[0] is 'self', not e.args[0]
    auto coerceArgs = [&](llvm::FunctionType *fty) {
        for (unsigned i = 0; i < fty->getNumParams() && i < args.size(); ++i) {
            auto *paramTy = fty->getParamType(i);
            auto *argTy   = args[i]->getType();
            if (argTy == paramTy) continue;
            if (argTy->isIntegerTy() && paramTy->isIntegerTy()) {
                unsigned aw = argTy->getIntegerBitWidth();
                unsigned pw = paramTy->getIntegerBitWidth();
                if (pw > aw) {
                    // Same i1-is-always-unsigned rule as coerceScalar above:
                    // a bool comparison result must zero-extend (true -> 1,
                    // not -1) regardless of what the srcTy lookup below finds
                    // (Bool sits outside the Int8..UInt64 range it checks).
                    bool isSigned;
                    if (aw == 1) {
                        isSigned = false;
                    } else {
                        TypePtr srcTy = (i >= argOffset && (i - argOffset) < e.args.size())
                            ? e.args[i - argOffset]->type : nullptr;
                        isSigned = true;
                        if (srcTy && srcTy->kind >= TypeKind::Int8 && srcTy->kind <= TypeKind::UInt64)
                            isSigned = static_cast<PrimType &>(*srcTy).isSigned();
                    }
                    args[i] = isSigned ? builder_.CreateSExt(args[i], paramTy, "argsext")
                                        : builder_.CreateZExt(args[i], paramTy, "argzext");
                } else if (pw < aw)
                    args[i] = builder_.CreateTrunc(args[i], paramTy, "argtrunc");
            } else if (argTy->isFloatingPointTy() && paramTy->isFloatingPointTy()) {
                // e.g. a double-valued literal narrowing into a 'float'
                // parameter (Sema's floatLitFitsType bypass) — without
                // this, the call would pass the argument's full original
                // width, reinterpreting the wrong bit pattern in the
                // narrower parameter register/slot.
                args[i] = coerceScalar(args[i], paramTy);
            } else if (argTy->isIntegerTy() && paramTy->isFloatingPointTy()) {
                // Integer argument -> floating-point parameter (see
                // canImplicitlyConvert's int-to-float widening rule in
                // Sema.cpp) — coerceScalar has the actual SIToFP/UIToFP
                // logic; this call site just needed to reach it, same as
                // the int-int and float-float branches around it.
                TypePtr srcTy = (i >= argOffset && (i - argOffset) < e.args.size())
                    ? e.args[i - argOffset]->type : nullptr;
                args[i] = coerceScalar(args[i], paramTy, srcTy);
            } else if (argTy->isPointerTy() && paramTy->isPointerTy()) {
                // opaque ptrs already match
            } else if (paramTy->isStructTy() && paramTy->getStructNumElements() == 2 &&
                       paramTy->getStructElementType(1)->isIntegerTy(1)) {
                // Optional-shaped parameter ('{T, i1}') receiving a plain T
                // argument, or a null-literal 'ptr' — same wrapping
                // coerceToOptional does for return/var-decl/assignment,
                // needed here too since this lambda only sees LLVM types
                // (Sema::canImplicitlyConvert already granted the call).
                auto *innerTy = paramTy->getStructElementType(0);
                if (argTy == innerTy) {
                    llvm::Value *agg = llvm::UndefValue::get(paramTy);
                    agg = builder_.CreateInsertValue(agg, args[i], {0});
                    agg = builder_.CreateInsertValue(agg, builder_.getTrue(), {1});
                    args[i] = agg;
                } else if (argTy->isPointerTy()) {
                    llvm::Value *agg = llvm::UndefValue::get(paramTy);
                    agg = builder_.CreateInsertValue(agg, llvm::Constant::getNullValue(innerTy), {0});
                    agg = builder_.CreateInsertValue(agg, builder_.getFalse(), {1});
                    args[i] = agg;
                }
            }
        }
        // C's default argument promotion for the '...' tail of a variadic
        // call: 'float' arguments must be widened to 'double' before the
        // call (C11 §6.5.2.2p6) — printf's va_arg(ap, double) for '%f'
        // reads 8 bytes regardless of what the caller's *source* type was,
        // so passing a raw (unpromoted) float32 here fed the wrong bit
        // pattern in and silently printed garbage/zero.
        if (fty->isVarArg()) {
            for (unsigned i = fty->getNumParams(); i < args.size(); ++i) {
                if (args[i]->getType()->isFloatTy()) {
                    args[i] = builder_.CreateFPExt(
                        args[i], llvm::Type::getDoubleTy(ctx_), "varargpromote");
                }
            }
        }
    };

    llvm::FunctionType *ft = nullptr;
    if (auto *fn = llvm::dyn_cast<llvm::Function>(calleeVal)) {
        ft = fn->getFunctionType();
        coerceArgs(ft);
        auto *call = builder_.CreateCall(ft, calleeVal, args);
        // Propagate calling convention from the callee function
        call->setCallingConv(fn->getCallingConv());
        return call;
    }
    // Emits an indirect call through 'calleeVal' against declared type
    // 'rawFT', first rewriting any struct-by-value argument over 16 bytes
    // that isn't an HFA (see isHomogeneousFPAggregate above) into the
    // AAPCS64-correct indirect form. Verified against a real clang -S
    // -emit-llvm reference on this exact struct shape: for a large non-HFA
    // aggregate, AAPCS64 erases the parameter to a *plain* 'ptr' at the
    // LLVM level — no 'byval' attribute at all (that attribute means
    // something different: "copy this onto MY outgoing stack argument
    // area," the x86 convention). The caller allocas a fresh copy, stores
    // the value into it, and passes that address as an ordinary pointer
    // argument; the real, externally-compiled callee just dereferences it
    // directly (confirmed in its disassembly: 'ldr x14, [x2]' etc., no
    // extra indirection). An earlier attempt using LLVM's 'byval' attribute
    // here compiled but was still wrong: LLVM's AArch64 backend lowered it
    // as raw bytes memcpy'd onto the outgoing stack (x86-style), never
    // touching x2/x3 at all, so the callee dereferenced garbage. Plain
    // SafeC-to-SafeC calls (direct calls, above) are untouched — this path
    // exists because every extern C/Objective-C interop call in this
    // codebase (objc_msgSend et al.) is reached through exactly this kind
    // of cast function pointer, and a real, externally-compiled function on
    // the other end expects the real ABI, not this file's usual (self-
    // consistent SafeC-to-SafeC, but not standards-compliant) raw-
    // aggregate-value passing.
    auto emitIndirectCall = [&](llvm::FunctionType *rawFT) -> llvm::CallInst * {
        coerceArgs(rawFT);
        const auto &dl = mod_->getDataLayout();
        std::vector<llvm::Type *> paramTys;
        paramTys.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            auto *argTy = args[i]->getType();
            auto *st = llvm::dyn_cast<llvm::StructType>(argTy);
            if (st && dl.getTypeAllocSize(st) > 16 && !isHomogeneousFPAggregate(st)) {
                auto *slot = builder_.CreateAlloca(st, nullptr, "indirect.tmp");
                builder_.CreateStore(args[i], slot);
                args[i] = slot;
            }
            paramTys.push_back(args[i]->getType());
        }
        auto *callFT = llvm::FunctionType::get(rawFT->getReturnType(), paramTys, rawFT->isVarArg());
        return builder_.CreateCall(callFT, calleeVal, args);
    };
    // Indirect call via function pointer (fn type variable or closure)
    if (e.callee->type && e.callee->type->kind == TypeKind::Function) {
        auto *rawFT = static_cast<llvm::FunctionType *>(lowerType(e.callee->type));
        return emitIndirectCall(rawFT);
    }
    // Same, but through a '&static' reference to a function type — this is
    // how 'fn T(Params)'-typed variables are represented (see parseBaseType's
    // KW_fn case), so calling one needs to unwrap the reference first.
    if (e.callee->type && e.callee->type->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*e.callee->type);
        if (rt.base && rt.base->kind == TypeKind::Function) {
            auto *rawFT = static_cast<llvm::FunctionType *>(lowerType(rt.base));
            return emitIndirectCall(rawFT);
        }
    }
    diag_.error(e.loc, "codegen: cannot determine callee function type");
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
}

// x.is_null() / x.is_none() / x.default(fallback) — see CallExpr::NullOp
// (AST.h) and Sema::checkCall's "safe pseudo-methods" section.
llvm::Value *CodeGen::genNullOp(CallExpr &e, FnEnv &env) {
    TypePtr baseTy = e.nullOpBase ? e.nullOpBase->type : nullptr;
    bool isPointer  = baseTy && baseTy->kind == TypeKind::Pointer;
    bool isOptional = baseTy && baseTy->kind == TypeKind::Optional;
    // The remaining case (not Pointer, not Optional) is a nullable
    // reference — the only other receiver Sema accepts here.

    switch (e.nullOp) {
    case CallExpr::NullOp::IsNull: {
        // Pointer and (nullable) Reference both lower to a plain opaque
        // ptr (see lowerType) — the same null check works for either.
        auto *val = genExpr(*e.nullOpBase, env);
        auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
        return builder_.CreateICmpEQ(val, nullPtr, "is_null");
    }
    case CallExpr::NullOp::IsNone: {
        // Optional lowers to '{T, i1}' (see lowerType) — has_value is
        // field 1; is_none is just its negation.
        auto *val = genExpr(*e.nullOpBase, env);
        auto *hasVal = builder_.CreateExtractValue(val, {1}, "opt.has");
        return builder_.CreateNot(hasVal, "is_none");
    }
    case CallExpr::NullOp::Default: {
        auto *fallback = genExpr(*e.args[0], env);
        if (isOptional) {
            auto *val = genExpr(*e.nullOpBase, env);
            auto *hasVal = builder_.CreateExtractValue(val, {1}, "opt.has");
            auto *inner  = builder_.CreateExtractValue(val, {0}, "opt.val");
            fallback = coerceScalar(fallback, inner->getType(), e.args[0]->type);
            return builder_.CreateSelect(hasVal, inner, fallback, "opt.default");
        }
        // Pointer / nullable reference: '.default(fallback)' dereferences
        // when non-null, returns 'fallback' otherwise — genuine branching
        // (not a naive load-then-select), since a null pointer must never
        // actually be dereferenced, even speculatively.
        TypePtr innerTy = isPointer ? static_cast<PointerType &>(*baseTy).base
                                     : static_cast<ReferenceType &>(*baseTy).base;
        auto *innerLLVMTy = lowerType(innerTy);
        fallback = coerceScalar(fallback, innerLLVMTy, e.args[0]->type);

        auto *ptrVal  = genExpr(*e.nullOpBase, env);
        auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_, 0));
        auto *isNull  = builder_.CreateICmpEQ(ptrVal, nullPtr, "default.isnull");

        auto *loadBB  = llvm::BasicBlock::Create(ctx_, "default.load",     env.fn);
        auto *fallBB  = llvm::BasicBlock::Create(ctx_, "default.fallback", env.fn);
        auto *mergeBB = llvm::BasicBlock::Create(ctx_, "default.end",      env.fn);
        builder_.CreateCondBr(isNull, fallBB, loadBB);

        builder_.SetInsertPoint(loadBB);
        auto *loaded  = builder_.CreateLoad(innerLLVMTy, ptrVal, "default.val");
        auto *loadEnd = builder_.GetInsertBlock();
        builder_.CreateBr(mergeBB);

        builder_.SetInsertPoint(fallBB);
        auto *fallEnd = builder_.GetInsertBlock();
        builder_.CreateBr(mergeBB);

        builder_.SetInsertPoint(mergeBB);
        auto *phi = builder_.CreatePHI(innerLLVMTy, 2, "default.result");
        phi->addIncoming(loaded, loadEnd);
        phi->addIncoming(fallback, fallEnd);
        return phi;
    }
    case CallExpr::NullOp::None:
        break;
    }
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
}

// ── Subscript ─────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genSubscript(SubscriptExpr &e, FnEnv &env, bool wantAddr) {
    // v[i] on a vec<T,N> (std::simd): extractelement straight from the SSA
    // vector value — no address/GEP involved, unlike every other
    // subscriptable kind below. Writes ('v[i] = x') are handled separately
    // in genAssign (read-modify-write via insertelement, mirroring how
    // bitfield writes work). 'wantAddr' has no real meaning for a register-
    // resident vector lane; fall back to spilling the element to a fresh
    // temp so callers expecting *some* address (e.g. '&v[i]') don't crash,
    // at the cost of that address not aliasing the original vector.
    if (e.base->type && e.base->type->kind == TypeKind::Vector) {
        auto *vecVal = genExpr(*e.base, env);
        auto *idx    = genExpr(*e.index, env);
        auto *elem   = builder_.CreateExtractElement(vecVal, idx, "vec.extract");
        if (!wantAddr) return elem;
        auto *tmp = createEntryAlloca(env, elem->getType(), "vec.elem.tmp");
        builder_.CreateStore(elem, tmp);
        return tmp;
    }

    auto *idxVal   = genExpr(*e.index, env);
    auto *baseTy   = e.base->type ? lowerType(e.base->type)
                                  : llvm::Type::getInt32Ty(ctx_);
    // Array/Slice subscripting needs the aggregate's own address (to GEP
    // into its inline storage), which only exists for an lvalue base.
    // Plain-pointer subscripting doesn't: the base is only ever a pointer
    // VALUE, which may come from an lvalue (a pointer variable — load the
    // current value from its alloca) or an arbitrary rvalue pointer
    // expression (e.g. a cast like '(unsigned char*)pkt.data', or a call
    // returning a pointer) that has no addressable storage of its own.
    // Only compute an lvalue address when the base actually needs/has one.
    auto *baseAddr = (baseTy->isArrayTy() ||
                      (e.base->type && e.base->type->kind == TypeKind::Slice) ||
                      e.base->isLValue)
                         ? genLValue(*e.base, env) : nullptr;

    llvm::Value *gep = nullptr;
    if (baseTy->isArrayTy()) {
        auto *arrTy = static_cast<llvm::ArrayType *>(baseTy);

        // Runtime bounds check for arrays with statically known size
        if (!e.boundsCheckOmit && e.base->type &&
            e.base->type->kind == TypeKind::Array) {
            auto &at = static_cast<const ArrayType &>(*e.base->type);
            if (at.size > 0) {
                auto *sizeCst = llvm::ConstantInt::get(
                    llvm::Type::getInt64Ty(ctx_), at.size);
                auto *idx64 = builder_.CreateZExtOrTrunc(
                    idxVal, llvm::Type::getInt64Ty(ctx_));
                auto *oob = builder_.CreateICmpUGE(idx64, sizeCst, "oob");
                auto *curFn  = builder_.GetInsertBlock()->getParent();
                auto *trapBB = llvm::BasicBlock::Create(ctx_, "bounds.trap", curFn);
                auto *okBB   = llvm::BasicBlock::Create(ctx_, "bounds.ok",   curFn);
                builder_.CreateCondBr(oob, trapBB, okBB);
                // Trap path: call abort() and mark unreachable
                builder_.SetInsertPoint(trapBB);
                auto abortFn = mod_->getOrInsertFunction(
                    "abort",
                    llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), false));
                builder_.CreateCall(abortFn);
                builder_.CreateUnreachable();
                // Safe path continues here
                builder_.SetInsertPoint(okBB);
            }
        }

        gep = builder_.CreateInBoundsGEP(arrTy, baseAddr,
                {llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0), idxVal},
                "arr.idx");
        if (wantAddr) return gep;
        return builder_.CreateLoad(arrTy->getElementType(), gep, "arr.elem");
    }
    // Slice subscript: baseAddr is the alloca for the {T*, i64} struct
    if (e.base->type && e.base->type->kind == TypeKind::Slice) {
        auto *sliceTy = baseTy; // {T*, i64}
        auto *i64Ty   = llvm::Type::getInt64Ty(ctx_);
        auto *slPtr = builder_.CreateLoad(llvm::PointerType::get(ctx_, 0),
            builder_.CreateStructGEP(sliceTy, baseAddr, 0), "sl.ptr");
        // Runtime bounds check
        if (!e.boundsCheckOmit) {
            auto *slLen = builder_.CreateLoad(i64Ty,
                builder_.CreateStructGEP(sliceTy, baseAddr, 1), "sl.len");
            auto *idx64 = builder_.CreateZExtOrTrunc(idxVal, i64Ty);
            auto *oob = builder_.CreateICmpUGE(idx64, slLen, "sl.oob");
            auto *curFn  = builder_.GetInsertBlock()->getParent();
            auto *trapBB = llvm::BasicBlock::Create(ctx_, "sl.trap", curFn);
            auto *okBB   = llvm::BasicBlock::Create(ctx_, "sl.ok",   curFn);
            builder_.CreateCondBr(oob, trapBB, okBB);
            builder_.SetInsertPoint(trapBB);
            auto abortFn = mod_->getOrInsertFunction(
                "abort", llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), false));
            builder_.CreateCall(abortFn);
            builder_.CreateUnreachable();
            builder_.SetInsertPoint(okBB);
        }
        auto *elemTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
        gep = builder_.CreateGEP(elemTy, slPtr, idxVal, "sl.idx");
        if (wantAddr) return gep;
        return builder_.CreateLoad(elemTy, gep, "sl.elem");
    }

    // Pointer subscript: if the base is an lvalue (e.g. a pointer variable),
    // baseAddr is that variable's own alloca — load the current pointer
    // value from it. Otherwise the base is an rvalue pointer expression
    // (cast result, call, ...) with no alloca of its own; evaluate it
    // directly to get the pointer value.
    auto *ptrVal = baseAddr
        ? builder_.CreateLoad(llvm::PointerType::get(ctx_, 0), baseAddr, "ptr.load")
        : genExpr(*e.base, env);
    auto *elemTy = e.type ? lowerType(e.type) : llvm::Type::getInt32Ty(ctx_);
    gep = builder_.CreateGEP(elemTy, ptrVal, idxVal, "ptr.idx");
    if (wantAddr) return gep;
    return builder_.CreateLoad(elemTy, gep, "ptr.elem");
}

// ── Slice construction: arr[start..end] → {T*, i64} ────────────────────────
llvm::Value *CodeGen::genSlice(SliceExpr &e, FnEnv &env) {
    auto *i64Ty  = llvm::Type::getInt64Ty(ctx_);
    auto *ptrTy  = llvm::PointerType::get(ctx_, 0);
    auto *sliceST = lowerType(e.type); // {T*, i64}

    // Determine base pointer and array length
    llvm::Value *basePtr = nullptr;
    llvm::Value *arrLen  = nullptr;

    if (e.base->type->kind == TypeKind::Array) {
        auto &at = static_cast<ArrayType &>(*e.base->type);
        auto *arrAddr = genLValue(*e.base, env);
        auto *arrTy   = lowerType(e.base->type);
        // Pointer to element 0
        basePtr = builder_.CreateGEP(arrTy, arrAddr,
            {llvm::ConstantInt::get(i64Ty, 0), llvm::ConstantInt::get(i64Ty, 0)},
            "slice.base");
        arrLen = llvm::ConstantInt::get(i64Ty, at.size);
    } else if (e.base->type->kind == TypeKind::Slice) {
        // Re-slicing a slice
        auto *sliceVal = genLValue(*e.base, env);
        auto *slBaseTy = lowerType(e.base->type);
        basePtr = builder_.CreateLoad(ptrTy,
            builder_.CreateStructGEP(slBaseTy, sliceVal, 0), "reslice.ptr");
        arrLen = builder_.CreateLoad(i64Ty,
            builder_.CreateStructGEP(slBaseTy, sliceVal, 1), "reslice.len");
    } else if (e.base->type->kind == TypeKind::Pointer) {
        basePtr = genExpr(*e.base, env);
        arrLen = nullptr; // unknown length for pointers
    } else if (e.base->type->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*e.base->type);
        if (rt.base->kind == TypeKind::Array) {
            auto &at = static_cast<ArrayType &>(*rt.base);
            auto *refPtr = genExpr(*e.base, env); // loads the pointer
            auto *arrTy  = lowerType(rt.base);
            basePtr = builder_.CreateGEP(arrTy, refPtr,
                {llvm::ConstantInt::get(i64Ty, 0), llvm::ConstantInt::get(i64Ty, 0)},
                "slice.refbase");
            arrLen = llvm::ConstantInt::get(i64Ty, at.size);
        } else {
            basePtr = genExpr(*e.base, env);
            arrLen = nullptr;
        }
    } else {
        basePtr = genExpr(*e.base, env);
        arrLen = nullptr;
    }

    auto *elemTy = lowerType(static_cast<SliceType &>(*e.type).element);

    // Compute start
    llvm::Value *startVal = e.start ? builder_.CreateZExtOrTrunc(
        genExpr(*e.start, env), i64Ty) : llvm::ConstantInt::get(i64Ty, 0);
    // Compute end
    llvm::Value *endVal = e.end ? builder_.CreateZExtOrTrunc(
        genExpr(*e.end, env), i64Ty) : arrLen;

    if (!endVal) {
        // Pointer with no end — can't determine length; use 0 as fallback
        endVal = llvm::ConstantInt::get(i64Ty, 0);
    }

    // Slice pointer: basePtr + start
    auto *slicePtr = builder_.CreateGEP(elemTy, basePtr, startVal, "slice.ptr");
    // Slice length: end - start
    auto *sliceLen = builder_.CreateSub(endVal, startVal, "slice.len");

    // Construct {T*, i64} struct
    auto *alloca = builder_.CreateAlloca(sliceST, nullptr, "slice.tmp");
    builder_.CreateStore(slicePtr,
        builder_.CreateStructGEP(sliceST, alloca, 0));
    builder_.CreateStore(sliceLen,
        builder_.CreateStructGEP(sliceST, alloca, 1));
    return builder_.CreateLoad(sliceST, alloca, "slice.val");
}

// ── Member access ─────────────────────────────────────────────────────────────
const FieldDecl *CodeGen::findMemberFieldDecl(MemberExpr &e, std::vector<int> &outPath) {
    if (!e.base->type) return nullptr;
    TypePtr bty = e.base->type;
    if (bty->kind == TypeKind::Pointer)   bty = static_cast<PointerType &>(*bty).base;
    if (bty->kind == TypeKind::Reference) bty = static_cast<ReferenceType &>(*bty).base;
    if (bty->kind != TypeKind::Struct) return nullptr;
    auto &st = static_cast<StructType &>(*bty);
    return st.findFieldPath(e.field, outPath);
}

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

    // Slice member access: .ptr → index 0, .len → index 1
    if (e.base->type) {
        TypePtr bty = e.base->type;
        if (bty->kind == TypeKind::Reference)
            bty = static_cast<ReferenceType &>(*bty).base;
        if (bty->kind == TypeKind::Slice) {
            // Slice is {T*, i64}. baseAddr is alloca for the slice struct.
            auto *sliceTy = lowerType(bty);
            llvm::Value *sliceAddr;
            if (e.isArrow || (e.base->type->kind == TypeKind::Reference)) {
                sliceAddr = baseAddr;
            } else {
                sliceAddr = genLValue(*e.base, env);
            }
            unsigned idx = (e.field == "ptr") ? 0 : 1; // ptr=0, len=1
            auto *gep = builder_.CreateStructGEP(sliceTy, sliceAddr, idx, "slice." + e.field);
            if (wantAddr) return gep;
            auto *fty = static_cast<llvm::StructType *>(sliceTy)->getElementType(idx);
            return builder_.CreateLoad(fty, gep, "slice." + e.field + ".val");
        }
    }

    // Find field index (or index chain, for a field reached through one or
    // more anonymous struct/union members — see StructType::findFieldPath).
    std::vector<int> fieldPath;
    if (e.base->type) {
        TypePtr bty = e.base->type;
        if (bty->kind == TypeKind::Pointer)
            bty = static_cast<PointerType &>(*bty).base;
        if (bty->kind == TypeKind::Reference)
            bty = static_cast<ReferenceType &>(*bty).base;
        // Tuple field access: field name is a decimal index string
        if (bty->kind == TypeKind::Tuple) {
            unsigned tupleIdx = 0;
            if (!e.field.empty()) {
                bool valid = true;
                for (char c : e.field) if (!isdigit((unsigned char)c)) { valid = false; break; }
                if (valid) tupleIdx = static_cast<unsigned>(std::stoul(e.field));
            }
            if (!structTy->isStructTy()) {
                structTy = lowerType(bty);
                baseAddr = genLValue(*e.base, env);
            }
            auto *gep = builder_.CreateStructGEP(
                static_cast<llvm::StructType *>(structTy), baseAddr, tupleIdx, "tuple.field");
            if (wantAddr) return gep;
            auto *fieldTy = static_cast<llvm::StructType *>(structTy)->getElementType(tupleIdx);
            return builder_.CreateLoad(fieldTy, gep, "tuple.elem");
        }
        if (bty->kind == TypeKind::Struct) {
            auto &st = static_cast<StructType &>(*bty);
            st.findFieldPath(e.field, fieldPath);
        }
    }
    if (fieldPath.empty()) fieldPath.push_back(0);

    if (!structTy->isStructTy()) {
        diag_.error(e.loc, "codegen: member access on non-struct");
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0);
    }

    // A single-element path is the common case (a direct field): use the
    // simple single-level GEP. A longer path (through an anonymous member)
    // needs a full multi-index GEP descending through each nested struct —
    // valid in one 'getelementptr' as long as every intermediate type is
    // itself a struct, which anonymous members always are.
    llvm::Value *gep;
    llvm::Type  *fieldTy;
    if (fieldPath.size() == 1) {
        gep = builder_.CreateStructGEP(structTy, baseAddr, (unsigned)fieldPath[0], e.field);
        fieldTy = static_cast<llvm::StructType *>(structTy)->getElementType((unsigned)fieldPath[0]);
    } else {
        auto *Int32Ty = llvm::Type::getInt32Ty(ctx_);
        std::vector<llvm::Value *> indices = { llvm::ConstantInt::get(Int32Ty, 0) };
        llvm::Type *cur = structTy;
        for (int idx : fieldPath) {
            indices.push_back(llvm::ConstantInt::get(Int32Ty, (uint64_t)idx));
            cur = static_cast<llvm::StructType *>(cur)->getElementType((unsigned)idx);
        }
        gep = builder_.CreateGEP(structTy, baseAddr, indices, e.field);
        fieldTy = cur;
    }
    if (wantAddr) return gep;

    // Bitfield read: extract via shift-left-then-shift-right, which lands
    // the field's bits at position 0 and sign/zero-extends in one step
    // (arithmetic shift for signed fields, logical for unsigned) — the
    // standard bitfield-extraction trick. 'gep' already points at the whole
    // shared storage unit (see the packing scheme in Sema::collectStruct),
    // not a per-bit address, since LLVM has no bitfield concept natively.
    std::vector<int> ignoredPath;
    if (const FieldDecl *fd = findMemberFieldDecl(e, ignoredPath); fd && fd->bitWidth >= 0) {
        auto *raw = builder_.CreateLoad(fieldTy, gep, e.field + ".unit");
        unsigned unitBits  = fieldTy->getIntegerBitWidth();
        unsigned leftShift = unitBits - fd->bitOffset - fd->bitWidth;
        unsigned rightShift = unitBits - fd->bitWidth;
        auto *shiftedUp = builder_.CreateShl(raw,
            llvm::ConstantInt::get(fieldTy, leftShift));
        bool isSigned = fd->type && fd->type->isInteger() &&
                        static_cast<PrimType &>(*fd->type).isSigned();
        return isSigned
            ? builder_.CreateAShr(shiftedUp, llvm::ConstantInt::get(fieldTy, rightShift), e.field)
            : builder_.CreateLShr(shiftedUp, llvm::ConstantInt::get(fieldTy, rightShift), e.field);
    }

    return builder_.CreateLoad(fieldTy, gep, e.field);
}

// ── Cast ──────────────────────────────────────────────────────────────────────
llvm::Value *CodeGen::genCast(CastExpr &e, FnEnv &env) {
    auto *dstTy = lowerType(e.targetType);

    // Array → Pointer decay: (T*)arr where arr is T[N]
    // Must use the address of the array, not the loaded value.
    if (dstTy->isPointerTy() && e.operand->type &&
        e.operand->type->kind == TypeKind::Array) {
        auto *arrAddr  = genLValue(*e.operand, env);
        auto *arrTy    = lowerType(e.operand->type);
        // GEP to element 0: &arr[0] — gives a ptr to the first element.
        return builder_.CreateGEP(arrTy, arrAddr,
            { llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0),
              llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 0) },
            "arraydecay");
    }

    // Scalar lvalue → Reference: (&region T)lvalueExpr, e.g. '(&stack u8)data[0]'.
    // The operand is a plain value of type T (not itself a reference/pointer/
    // array), so casting to a reference means "take this lvalue's address"
    // (matching '&expr' semantics) — NOT "reinterpret the loaded scalar value
    // as a pointer", which would treat e.g. an element's runtime value (1)
    // as if it were the address 0x1.
    if (e.targetType->kind == TypeKind::Reference && e.operand->isLValue &&
        e.operand->type &&
        e.operand->type->kind != TypeKind::Reference &&
        e.operand->type->kind != TypeKind::Pointer &&
        e.operand->type->kind != TypeKind::Array) {
        return genLValue(*e.operand, env);
    }

    auto *srcVal = genExpr(*e.operand, env);
    auto *srcTy  = srcVal->getType();

    if (srcTy == dstTy) return srcVal;

    // '(void)expr' — the classic C idiom for "evaluate for side effects,
    // discard the result" (e.g. silencing an unused-parameter warning).
    // There's nothing to cast to: void isn't a real SSA value type, so
    // the fallback bitcast below would be malformed IR.
    if (dstTy->isVoidTy()) return srcVal;

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
    // Vector-lane assignment: 'v[i] = x' on a vec<T,N> (std::simd). Same
    // shape as the bitfield case below — read the whole SSA value, modify
    // one lane via insertelement, write the whole value back — since a
    // single vector lane isn't independently addressable the way a plain
    // array element is (see genSubscript's extractelement-only read path).
    if (e.lhs->kind == ExprKind::Subscript) {
        auto &se = static_cast<SubscriptExpr &>(*e.lhs);
        if (se.base->type && se.base->type->kind == TypeKind::Vector) {
            auto *vecAddr = genLValue(*se.base, env);
            auto *vecTy   = lowerType(se.base->type);
            auto *idx     = genExpr(*se.index, env);
            auto *oldVec  = builder_.CreateLoad(vecTy, vecAddr, "vec.old");
            auto *elemTy  = llvm::cast<llvm::VectorType>(vecTy)->getElementType();
            auto *rhs     = coerceScalar(genExpr(*e.rhs, env), elemTy, e.rhs->type);
            llvm::Value *newVal = rhs;
            if (e.op != AssignOp::Assign) {
                auto *cur = builder_.CreateExtractElement(oldVec, idx, "vec.cur");
                BinaryOp binOp;
                switch (e.op) {
                case AssignOp::AddAssign: binOp = BinaryOp::Add; break;
                case AssignOp::SubAssign: binOp = BinaryOp::Sub; break;
                case AssignOp::MulAssign: binOp = BinaryOp::Mul; break;
                case AssignOp::DivAssign: binOp = BinaryOp::Div; break;
                default:                  binOp = BinaryOp::Add; break;
                }
                newVal = applyBinaryOp(binOp, cur, rhs, se.type);
            }
            auto *newVec = builder_.CreateInsertElement(oldVec, newVal, idx, "vec.new");
            builder_.CreateStore(newVec, vecAddr);
            return newVal;
        }
    }

    // Bitfield assignment: read-modify-write on the shared storage unit.
    // A plain store here (the path below, for ordinary fields) would
    // clobber sibling bitfields packed into the same LLVM struct slot — see
    // the packing scheme in Sema::collectStruct and genMember's read-side
    // extraction, which this mirrors for writes.
    if (e.lhs->kind == ExprKind::Member || e.lhs->kind == ExprKind::Arrow) {
        auto &me = static_cast<MemberExpr &>(*e.lhs);
        std::vector<int> ignoredPath;
        if (const FieldDecl *fd = findMemberFieldDecl(me, ignoredPath); fd && fd->bitWidth >= 0) {
            auto *unitAddr = genLValue(*e.lhs, env); // GEP to the shared storage unit
            auto *unitTy   = lowerType(fd->type);
            auto *rawUnit  = builder_.CreateLoad(unitTy, unitAddr, "bf.unit");

            unsigned unitBits   = unitTy->getIntegerBitWidth();
            unsigned leftShift  = unitBits - fd->bitOffset - fd->bitWidth;
            unsigned rightShift = unitBits - fd->bitWidth;
            bool isSigned = fd->type && fd->type->isInteger() &&
                            static_cast<PrimType &>(*fd->type).isSigned();
            auto *shiftedUp = builder_.CreateShl(rawUnit, llvm::ConstantInt::get(unitTy, leftShift));
            auto *curVal = isSigned
                ? builder_.CreateAShr(shiftedUp, llvm::ConstantInt::get(unitTy, rightShift))
                : builder_.CreateLShr(shiftedUp, llvm::ConstantInt::get(unitTy, rightShift));

            auto *rhs = genExpr(*e.rhs, env);
            if (rhs->getType() != unitTy && rhs->getType()->isIntegerTy() && unitTy->isIntegerTy()) {
                unsigned dstBits = unitTy->getIntegerBitWidth();
                unsigned srcBits = rhs->getType()->getIntegerBitWidth();
                if (dstBits < srcBits) rhs = builder_.CreateTrunc(rhs, unitTy);
                else if (dstBits > srcBits) rhs = builder_.CreateSExt(rhs, unitTy);
            }

            llvm::Value *newVal = rhs;
            if (e.op != AssignOp::Assign) {
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
                default:                  binOp = BinaryOp::Add;   break;
                }
                newVal = applyBinaryOp(binOp, curVal, rhs, fd->type);
            }

            uint64_t maskVal = (fd->bitWidth >= 64) ? ~0ULL : ((1ULL << fd->bitWidth) - 1);
            auto *mask       = llvm::ConstantInt::get(unitTy, maskVal);
            auto *maskedNew  = builder_.CreateAnd(newVal, mask);
            auto *shiftedNew = builder_.CreateShl(maskedNew,
                llvm::ConstantInt::get(unitTy, (uint64_t)fd->bitOffset));
            auto *clearMask  = builder_.CreateNot(builder_.CreateShl(mask,
                llvm::ConstantInt::get(unitTy, (uint64_t)fd->bitOffset)));
            auto *cleared    = builder_.CreateAnd(rawUnit, clearMask);
            auto *merged     = builder_.CreateOr(cleared, shiftedNew);
            builder_.CreateStore(merged, unitAddr);
            return newVal;
        }
    }

    // Same array-to-pointer/reference decay as call arguments and var-decl
    // initializers — an array RHS assigned into a pointer/reference LHS
    // needs its address, not an attempted load of the whole array.
    bool rhsNeedsDecay = e.rhs->type && e.rhs->type->kind == TypeKind::Array &&
                         e.lhs->type && (e.lhs->type->kind == TypeKind::Pointer ||
                                         e.lhs->type->kind == TypeKind::Reference);
    auto *rhs   = rhsNeedsDecay ? decayArrayToPtr(*e.rhs, env) : genExpr(*e.rhs, env);
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

    // Type coercion for mismatched scalar sizes (e.g., int literal → i8,
    // or a double-valued literal narrowing into a 'float' lhs). Signedness
    // for widening comes from rhs on a plain assign; a compound assign's
    // 'val' is applyBinaryOp's result, whose natural type tracks lhs.
    val = coerceScalar(val, lhsTy, e.op == AssignOp::Assign ? e.rhs->type : e.lhs->type);
    if (e.op == AssignOp::Assign)
        val = coerceToOptional(val, e.rhs->type, e.lhs->type);

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
