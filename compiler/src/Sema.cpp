#include "safec/Sema.h"
#include <cassert>
#include <algorithm>

namespace safec {

// ── Scope ─────────────────────────────────────────────────────────────────────
Symbol *Scope::lookup(const std::string &name) {
    auto it = symbols_.find(name);
    return (it != symbols_.end()) ? &it->second : nullptr;
}

bool Scope::define(Symbol sym) {
    auto [it, inserted] = symbols_.emplace(sym.name, std::move(sym));
    return inserted;
}

// ── Sema ──────────────────────────────────────────────────────────────────────
Sema::Sema(TranslationUnit &tu, DiagEngine &diag) : tu_(tu), diag_(diag) {
    registerBuiltinTypes();
}

void Sema::registerBuiltinTypes() {
    typeRegistry_["void"]           = makeVoid();
    typeRegistry_["bool"]           = makeBool();
    typeRegistry_["char"]           = makeChar();
    typeRegistry_["int"]            = makeInt(32);
    typeRegistry_["long"]           = makeInt(64);
    typeRegistry_["short"]          = makeInt(16);
    typeRegistry_["unsigned"]       = makeInt(32, false);
    typeRegistry_["unsigned int"]   = makeInt(32, false);
    typeRegistry_["unsigned long"]  = makeInt(64, false);
    typeRegistry_["float"]          = makeFloat(32);
    typeRegistry_["double"]         = makeFloat(64);
    typeRegistry_["int8_t"]         = makeInt(8);
    typeRegistry_["int16_t"]        = makeInt(16);
    typeRegistry_["int32_t"]        = makeInt(32);
    typeRegistry_["int64_t"]        = makeInt(64);
    typeRegistry_["uint8_t"]        = makeInt(8,  false);
    typeRegistry_["uint16_t"]       = makeInt(16, false);
    typeRegistry_["uint32_t"]       = makeInt(32, false);
    typeRegistry_["uint64_t"]       = makeInt(64, false);
}

void Sema::pushScope(bool isUnsafe) {
    scopes_.emplace_back(++scopeDepth_, isUnsafe);
}

void Sema::popScope() {
    untrackScope(scopeDepth_);
    scopes_.pop_back();
    --scopeDepth_;
}

Symbol *Sema::lookup(const std::string &name) {
    for (int i = (int)scopes_.size() - 1; i >= 0; --i) {
        if (auto *s = scopes_[i].lookup(name)) return s;
    }
    return nullptr;
}

void Sema::define(Symbol sym) {
    sym.scopeDepth = scopeDepth_;
    if (scopes_.empty()) return;
    if (!scopes_.back().define(sym)) {
        diag_.error({}, "redefinition of '" + sym.name + "'");
    }
}

bool Sema::inUnsafeScope() const {
    for (auto &s : scopes_) if (s.isUnsafe()) return true;
    return false;
}

TypePtr Sema::lookupType(const std::string &name) const {
    auto it = typeRegistry_.find(name);
    return (it != typeRegistry_.end()) ? it->second : nullptr;
}

TypePtr Sema::resolveType(TypePtr ty) {
    if (!ty) return makeError();
    if (ty->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*ty);
        if (!st.isDefined) {
            auto it = typeRegistry_.find(st.name);
            if (it != typeRegistry_.end()) return it->second;
        }
        return ty;
    }
    if (ty->kind == TypeKind::Enum) {
        auto &et = static_cast<EnumType &>(*ty);
        auto it = typeRegistry_.find(et.name);
        if (it != typeRegistry_.end()) return it->second;
        return ty;
    }
    if (ty->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*ty);
        auto resolvedBase = resolveType(rt.base);
        if (resolvedBase != rt.base)
            return makeReference(resolvedBase, rt.region, rt.nullable, rt.mut, rt.arenaName);
    }
    if (ty->kind == TypeKind::Pointer) {
        auto &pt = static_cast<PointerType &>(*ty);
        auto resolvedBase = resolveType(pt.base);
        if (resolvedBase != pt.base)
            return makePointer(resolvedBase, pt.isConst);
    }
    if (ty->kind == TypeKind::Array) {
        auto &at = static_cast<ArrayType &>(*ty);
        auto resolvedElem = resolveType(at.element);
        if (resolvedElem != at.element)
            return makeArray(resolvedElem, at.size);
    }
    return ty;
}

// ── Pass 1: Collect top-level declarations ─────────────────────────────────────
bool Sema::run() {
    // Global scope
    pushScope();
    collectDecls(tu_);
    for (auto &d : tu_.decls) checkDecl(*d);
    popScope();
    return !diag_.hasErrors();
}

void Sema::collectDecls(TranslationUnit &tu) {
    for (auto &d : tu.decls) {
        switch (d->kind) {
        case DeclKind::Function:   collectFunction(static_cast<FunctionDecl &>(*d)); break;
        case DeclKind::Struct:     collectStruct(static_cast<StructDecl &>(*d));     break;
        case DeclKind::Enum:       collectEnum(static_cast<EnumDecl &>(*d));         break;
        case DeclKind::Region:     collectRegion(static_cast<RegionDecl &>(*d));     break;
        case DeclKind::GlobalVar:  collectGlobalVar(static_cast<GlobalVarDecl &>(*d));break;
        default: break;
        }
    }
}

void Sema::collectFunction(FunctionDecl &fn) {
    std::vector<TypePtr> paramTys;
    for (auto &p : fn.params) {
        p.type = resolveType(p.type);
        paramTys.push_back(p.type);
    }
    fn.returnType = resolveType(fn.returnType);
    auto ft = makeFunction(fn.returnType, std::move(paramTys), fn.isVariadic);
    Symbol sym;
    sym.kind   = SymKind::Function;
    sym.name   = fn.name;
    sym.type   = ft;
    sym.fnDecl = &fn;
    sym.initialized = true;
    define(std::move(sym));
}

void Sema::collectStruct(StructDecl &sd) {
    auto st = std::make_shared<StructType>(sd.name, sd.isUnion);
    st->isDefined = true;
    int idx = 0;
    for (auto &f : sd.fields) {
        FieldDecl fd;
        fd.name  = f.name;
        fd.type  = resolveType(f.type);
        fd.index = idx++;
        st->fields.push_back(std::move(fd));
    }
    sd.type = st;
    typeRegistry_[sd.name] = st;

    Symbol sym;
    sym.kind  = SymKind::Type;
    sym.name  = sd.name;
    sym.type  = st;
    sym.initialized = true;
    define(std::move(sym));
}

void Sema::collectEnum(EnumDecl &ed) {
    auto et = std::make_shared<EnumType>(ed.name);
    int64_t nextVal = 0;
    for (auto &[name, val] : ed.enumerators) {
        int64_t v = val ? *val : nextVal;
        et->enumerators.push_back({name, v});
        nextVal = v + 1;

        // Each enumerator is a constant in the scope
        Symbol sym;
        sym.kind  = SymKind::Variable;
        sym.name  = name;
        sym.type  = makeInt(32);
        sym.initialized = true;
        define(std::move(sym));
    }
    ed.type = et;
    typeRegistry_[ed.name] = et;

    Symbol sym;
    sym.kind  = SymKind::Type;
    sym.name  = ed.name;
    sym.type  = et;
    sym.initialized = true;
    define(std::move(sym));
}

void Sema::collectRegion(RegionDecl &rd) {
    regionRegistry_[rd.name] = &rd;
}

void Sema::collectGlobalVar(GlobalVarDecl &gv) {
    gv.type = resolveType(gv.type);
    Symbol sym;
    sym.kind        = SymKind::Variable;
    sym.name        = gv.name;
    sym.type        = gv.type;
    sym.initialized = (gv.init != nullptr || gv.isExtern);
    define(std::move(sym));
}

// ── Pass 2: Check bodies ──────────────────────────────────────────────────────
void Sema::checkDecl(Decl &d) {
    switch (d.kind) {
    case DeclKind::Function:
        checkFunction(static_cast<FunctionDecl &>(d));
        break;
    case DeclKind::GlobalVar:
        checkGlobalVar(static_cast<GlobalVarDecl &>(d));
        break;
    case DeclKind::StaticAssert:
        checkStaticAssert(static_cast<StaticAssertDecl &>(d));
        break;
    default: break;
    }
}

void Sema::checkFunction(FunctionDecl &fn) {
    if (!fn.body) return; // declaration only

    pushScope();
    // Register parameters
    for (auto &p : fn.params) {
        p.type = resolveType(p.type);
        VarDecl *vd = new VarDecl{};  // leaked intentionally for demo
        vd->name        = p.name;
        vd->type        = p.type;
        vd->isParam     = true;
        vd->initialized = true;
        vd->scopeDepth  = scopeDepth_;

        Symbol sym;
        sym.kind        = SymKind::Variable;
        sym.name        = p.name;
        sym.type        = p.type;
        sym.varDecl     = vd;
        sym.initialized = true;
        define(std::move(sym));
    }
    checkCompound(*fn.body, fn);
    popScope();
}

void Sema::checkGlobalVar(GlobalVarDecl &gv) {
    if (gv.init) {
        TypePtr initTy = checkExpr(*gv.init);
        if (!gv.type->isError() && !initTy->isError()) {
            if (!canImplicitlyConvert(initTy, gv.type)) {
                diag_.error(gv.loc,
                    "type mismatch in global variable initializer: cannot convert '"
                    + initTy->str() + "' to '" + gv.type->str() + "'");
            }
        }
    }
    // Update symbol initialization status
    if (auto *sym = lookup(gv.name)) sym->initialized = true;
}

void Sema::checkStaticAssert(StaticAssertDecl &sa) {
    if (sa.cond) checkExpr(*sa.cond);
    // Actual compile-time evaluation is handled by consteval engine (future)
}

// ── Statement checking ────────────────────────────────────────────────────────
void Sema::checkStmt(Stmt &s, FunctionDecl &fn) {
    switch (s.kind) {
    case StmtKind::Compound:     checkCompound(static_cast<CompoundStmt &>(s), fn);    break;
    case StmtKind::If:           checkIf(static_cast<IfStmt &>(s), fn);                break;
    case StmtKind::While:
    case StmtKind::DoWhile:      checkWhile(static_cast<WhileStmt &>(s), fn);          break;
    case StmtKind::For:          checkFor(static_cast<ForStmt &>(s), fn);              break;
    case StmtKind::Return:       checkReturn(static_cast<ReturnStmt &>(s), fn);        break;
    case StmtKind::VarDecl:      checkVarDecl(static_cast<VarDeclStmt &>(s), fn);     break;
    case StmtKind::Expr:
        if (static_cast<ExprStmt &>(s).expr)
            checkExpr(*static_cast<ExprStmt &>(s).expr);
        break;
    case StmtKind::Unsafe:       checkUnsafe(static_cast<UnsafeStmt &>(s), fn);       break;
    case StmtKind::StaticAssert: checkStaticAssertStmt(static_cast<StaticAssertStmt &>(s)); break;
    case StmtKind::IfConst: {
        // Treat like a regular if for type-checking; ConstEvalEngine evaluates condition later
        auto &ics = static_cast<IfConstStmt &>(s);
        if (ics.cond) checkExpr(*ics.cond);
        if (ics.then)  checkStmt(*ics.then, fn);
        if (ics.else_) checkStmt(*ics.else_, fn);
        break;
    }
    case StmtKind::Break:
    case StmtKind::Continue:     break; // TODO: check we're inside a loop
    case StmtKind::Goto:         break; // TODO: check label exists
    case StmtKind::Label:
        checkStmt(*static_cast<LabelStmt &>(s).body, fn);
        break;
    default: break;
    }
}

void Sema::checkCompound(CompoundStmt &cs, FunctionDecl &fn) {
    pushScope();
    for (auto &stmt : cs.body) checkStmt(*stmt, fn);
    popScope();
}

void Sema::checkIf(IfStmt &s, FunctionDecl &fn) {
    TypePtr condTy = checkExpr(*s.cond);
    if (!condTy->isError() && !condTy->isInteger() && !condTy->isBool() &&
        condTy->kind != TypeKind::Pointer && condTy->kind != TypeKind::Reference) {
        diag_.error(s.cond->loc, "condition must be boolean or numeric");
    }
    checkStmt(*s.then, fn);
    if (s.else_) checkStmt(*s.else_, fn);
}

void Sema::checkWhile(WhileStmt &s, FunctionDecl &fn) {
    TypePtr condTy = checkExpr(*s.cond);
    if (!condTy->isError() && !condTy->isInteger() && !condTy->isBool() &&
        condTy->kind != TypeKind::Pointer) {
        diag_.error(s.cond->loc, "while condition must be boolean or numeric");
    }
    checkStmt(*s.body, fn);
}

void Sema::checkFor(ForStmt &s, FunctionDecl &fn) {
    pushScope();
    if (s.init) checkStmt(*s.init, fn);
    if (s.cond) checkExpr(*s.cond);
    if (s.incr) checkExpr(*s.incr);
    checkStmt(*s.body, fn);
    popScope();
}

void Sema::checkReturn(ReturnStmt &s, FunctionDecl &fn) {
    if (s.value) {
        TypePtr valTy = checkExpr(*s.value);

        // Region escape check: returned reference must not be &stack T
        if (valTy && valTy->kind == TypeKind::Reference) {
            auto &rt = static_cast<ReferenceType &>(*valTy);
            if (rt.region == Region::Stack) {
                diag_.error(s.loc,
                    "cannot return '&stack " + rt.base->str() +
                    "': stack reference escapes function scope");
            }
        }

        // Type compatibility
        if (fn.returnType && !fn.returnType->isVoid() &&
            !valTy->isError() && !fn.returnType->isError()) {
            if (!canImplicitlyConvert(valTy, fn.returnType)) {
                diag_.error(s.loc,
                    "return type mismatch: cannot convert '" + valTy->str() +
                    "' to '" + fn.returnType->str() + "'");
            }
        }
    } else {
        if (fn.returnType && !fn.returnType->isVoid()) {
            diag_.error(s.loc, "non-void function must return a value");
        }
    }
}

void Sema::checkVarDecl(VarDeclStmt &s, FunctionDecl &fn) {
    s.declType = resolveType(s.declType);

    TypePtr initTy;
    if (s.init) {
        initTy = checkExpr(*s.init);

        // Region escape: if init is a &stack ref, the var must not be promoted
        if (initTy && initTy->kind == TypeKind::Reference) {
            auto &rt = static_cast<ReferenceType &>(*initTy);
            if (rt.region == Region::Stack) {
                // This is fine as long as it's a local — stack references
                // can be stored locally, just not returned or stored in globals.
                // Actual escape to outer scope is detected at assignment time.
            }
        }

        if (s.declType && !s.declType->isError() && !initTy->isError()) {
            if (!canImplicitlyConvert(initTy, s.declType)) {
                diag_.error(s.loc,
                    "type mismatch in variable declaration: cannot convert '"
                    + initTy->str() + "' to '" + s.declType->str() + "'");
            }
        }
    }

    // Resolve type from initializer if none declared
    if (!s.declType || s.declType->isError()) {
        s.declType = initTy ? initTy : makeError();
    }
    s.resolvedType = s.declType;

    // Register in current scope
    VarDecl *vd = new VarDecl{};
    vd->name        = s.name;
    vd->type        = s.resolvedType;
    vd->isConst     = s.isConst;
    vd->isStatic    = s.isStatic;
    // Struct/array types are considered "initialized" even without an explicit
    // initializer because their fields can be individually assigned before use.
    // Only scalar types strictly require initialization before reading.
    bool isAggregate = s.resolvedType &&
                       (s.resolvedType->kind == TypeKind::Struct ||
                        s.resolvedType->kind == TypeKind::Array);
    vd->initialized = (s.init != nullptr) || isAggregate;
    vd->scopeDepth  = scopeDepth_;

    Symbol sym;
    sym.kind        = SymKind::Variable;
    sym.name        = s.name;
    sym.type        = s.resolvedType;
    sym.varDecl     = vd;
    sym.initialized = vd->initialized;
    define(std::move(sym));
}

void Sema::checkUnsafe(UnsafeStmt &s, FunctionDecl &fn) {
    pushScope(true); // unsafe scope
    checkCompound(static_cast<CompoundStmt &>(*s.body), fn);
    popScope();
}

void Sema::checkStaticAssertStmt(StaticAssertStmt &s) {
    if (s.cond) checkExpr(*s.cond);
}

// ── Expression type-checking ──────────────────────────────────────────────────
TypePtr Sema::checkExpr(Expr &e) {
    TypePtr ty;
    switch (e.kind) {
    case ExprKind::IntLit:    ty = checkIntLit(static_cast<IntLitExpr &>(e));     break;
    case ExprKind::FloatLit:  ty = checkFloatLit(static_cast<FloatLitExpr &>(e)); break;
    case ExprKind::BoolLit:   ty = checkBoolLit(static_cast<BoolLitExpr &>(e));   break;
    case ExprKind::StringLit: ty = checkStringLit(static_cast<StringLitExpr &>(e));break;
    case ExprKind::CharLit:   ty = makeChar();                                     break;
    case ExprKind::NullLit:   ty = makePointer(makeVoid());                        break;
    case ExprKind::Ident:     ty = checkIdent(static_cast<IdentExpr &>(e));       break;
    case ExprKind::Unary:     ty = checkUnary(static_cast<UnaryExpr &>(e));       break;
    case ExprKind::AddrOf:    ty = checkAddrOf(static_cast<UnaryExpr &>(e));      break;
    case ExprKind::Deref:     ty = checkDeref(static_cast<UnaryExpr &>(e));       break;
    case ExprKind::Binary:    ty = checkBinary(static_cast<BinaryExpr &>(e));     break;
    case ExprKind::Ternary:   ty = checkTernary(static_cast<TernaryExpr &>(e));   break;
    case ExprKind::Call:      ty = checkCall(static_cast<CallExpr &>(e));         break;
    case ExprKind::Subscript: ty = checkSubscript(static_cast<SubscriptExpr &>(e));break;
    case ExprKind::Member:
    case ExprKind::Arrow:     ty = checkMember(static_cast<MemberExpr &>(e));     break;
    case ExprKind::Cast:      ty = checkCast(static_cast<CastExpr &>(e));         break;
    case ExprKind::Assign:    ty = checkAssign(static_cast<AssignExpr &>(e));     break;
    case ExprKind::SizeofType:
        static_cast<SizeofTypeExpr &>(e).ofType =
            resolveType(static_cast<SizeofTypeExpr &>(e).ofType);
        ty = makeInt(64, false);
        break;
    case ExprKind::Compound: {
        auto &ci = static_cast<CompoundInitExpr &>(e);
        for (auto &init : ci.inits) checkExpr(*init);
        ty = makeVoid(); // resolved by context
        break;
    }
    default:
        ty = makeError();
    }
    e.type = ty;
    return ty;
}

TypePtr Sema::checkIntLit(IntLitExpr &e) {
    // Choose smallest fitting type
    if (e.value >= -2147483648LL && e.value <= 2147483647LL) return makeInt(32);
    return makeInt(64);
}

TypePtr Sema::checkFloatLit(FloatLitExpr &e) {
    return makeFloat(64); // default double
}

TypePtr Sema::checkBoolLit(BoolLitExpr &) { return makeBool(); }

TypePtr Sema::checkStringLit(StringLitExpr &e) {
    // String literal → &static char (zero-terminated C string as static ref)
    return makeReference(makeInt(8), Region::Static, false, false);
}

TypePtr Sema::checkIdent(IdentExpr &e) {
    auto *sym = lookup(e.name);
    if (!sym) {
        diag_.error(e.loc, "use of undeclared identifier '" + e.name + "'");
        return makeError();
    }
    // Definite initialization check
    if (sym->kind == SymKind::Variable && !sym->initialized) {
        diag_.error(e.loc, "use of possibly uninitialized variable '" + e.name + "'");
    }
    if (sym->varDecl)   { e.resolved   = sym->varDecl;  e.isLValue = true; }
    if (sym->fnDecl)    { e.resolvedFn = sym->fnDecl; }
    return sym->type ? sym->type : makeError();
}

TypePtr Sema::checkUnary(UnaryExpr &e) {
    TypePtr operandTy = checkExpr(*e.operand);
    switch (e.op) {
    case UnaryOp::Neg:
        if (!operandTy->isArithmetic())
            diag_.error(e.loc, "unary '-' requires numeric type");
        return operandTy;
    case UnaryOp::Not:
        return makeBool();
    case UnaryOp::BitNot:
        if (!operandTy->isInteger())
            diag_.error(e.loc, "bitwise not requires integer type");
        return operandTy;
    case UnaryOp::PreInc: case UnaryOp::PreDec:
    case UnaryOp::PostInc: case UnaryOp::PostDec:
        if (!operandTy->isArithmetic() && operandTy->kind != TypeKind::Pointer)
            diag_.error(e.loc, "++ / -- requires numeric or pointer type");
        return operandTy;
    case UnaryOp::SizeofExpr:
        return makeInt(64, false);
    default:
        return operandTy;
    }
}

TypePtr Sema::checkAddrOf(UnaryExpr &e) {
    TypePtr operandTy = checkExpr(*e.operand);
    if (!e.operand->isLValue) {
        diag_.error(e.loc, "address-of operator requires lvalue");
    }
    // In safe code, & of a stack variable yields &stack T
    return makeReference(std::move(operandTy), Region::Stack, false, true);
}

TypePtr Sema::checkDeref(UnaryExpr &e) {
    TypePtr operandTy = checkExpr(*e.operand);
    if (operandTy->kind == TypeKind::Pointer) {
        if (!inUnsafeScope()) {
            diag_.error(e.loc,
                "dereference of raw pointer requires 'unsafe' block");
        }
        return static_cast<PointerType &>(*operandTy).base;
    }
    if (operandTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*operandTy);
        checkNullabilityDeref(operandTy, e.loc);
        e.isLValue = rt.mut;
        return rt.base;
    }
    diag_.error(e.loc, "cannot dereference non-pointer type '" + operandTy->str() + "'");
    return makeError();
}

TypePtr Sema::checkBinary(BinaryExpr &e) {
    TypePtr lhsTy = checkExpr(*e.left);
    TypePtr rhsTy = checkExpr(*e.right);

    if (lhsTy->isError() || rhsTy->isError()) return makeError();

    switch (e.op) {
    // Arithmetic
    case BinaryOp::Add: case BinaryOp::Sub:
    case BinaryOp::Mul: case BinaryOp::Div: case BinaryOp::Mod:
        if (!lhsTy->isArithmetic() || !rhsTy->isArithmetic()) {
            // Allow pointer arithmetic for raw pointers (unsafe)
            if ((lhsTy->kind == TypeKind::Pointer || rhsTy->kind == TypeKind::Pointer)) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "pointer arithmetic requires 'unsafe' block");
                return lhsTy->kind == TypeKind::Pointer ? lhsTy : rhsTy;
            }
            diag_.error(e.loc, "arithmetic requires numeric types");
            return makeError();
        }
        // SafeC: no implicit conversions — both sides must match
        if (!typeEqual(lhsTy, rhsTy)) {
            diag_.error(e.loc,
                "implicit type conversion not allowed in SafeC: '" +
                lhsTy->str() + "' and '" + rhsTy->str() + "' differ");
            return makeError();
        }
        return lhsTy;
    // Bit ops
    case BinaryOp::BitAnd: case BinaryOp::BitOr:
    case BinaryOp::BitXor: case BinaryOp::Shl: case BinaryOp::Shr:
        if (!lhsTy->isInteger() || !rhsTy->isInteger()) {
            diag_.error(e.loc, "bitwise operations require integer types");
            return makeError();
        }
        return lhsTy;
    // Comparison → bool
    case BinaryOp::Eq: case BinaryOp::NEq:
    case BinaryOp::Lt: case BinaryOp::Gt:
    case BinaryOp::LEq: case BinaryOp::GEq:
        return makeBool();
    // Logical → bool
    case BinaryOp::LogAnd: case BinaryOp::LogOr:
        return makeBool();
    case BinaryOp::Comma:
        return rhsTy;
    default:
        return lhsTy;
    }
}

TypePtr Sema::checkTernary(TernaryExpr &e) {
    checkExpr(*e.cond);
    TypePtr thenTy = checkExpr(*e.then);
    TypePtr elseTy = checkExpr(*e.else_);
    if (!typeEqual(thenTy, elseTy)) {
        diag_.error(e.loc,
            "ternary branches must have same type: '" +
            thenTy->str() + "' vs '" + elseTy->str() + "'");
        return makeError();
    }
    return thenTy;
}

TypePtr Sema::checkCall(CallExpr &e) {
    TypePtr calleeTy = checkExpr(*e.callee);

    // Resolve function type
    FunctionType *ft = nullptr;
    if (calleeTy->kind == TypeKind::Function) {
        ft = static_cast<FunctionType *>(calleeTy.get());
    }

    if (!ft) {
        if (!calleeTy->isError())
            diag_.error(e.loc, "expression is not callable");
        for (auto &a : e.args) checkExpr(*a);
        return makeError();
    }

    // Check arg count
    size_t expected = ft->paramTypes.size();
    size_t provided = e.args.size();
    if (provided < expected || (!ft->variadic && provided > expected)) {
        diag_.error(e.loc,
            "wrong number of arguments: expected " + std::to_string(expected) +
            ", got " + std::to_string(provided));
    }

    for (size_t i = 0; i < e.args.size(); ++i) {
        TypePtr argTy = checkExpr(*e.args[i]);
        if (i < expected && !argTy->isError() && !ft->paramTypes[i]->isError()) {
            if (!canImplicitlyConvert(argTy, ft->paramTypes[i])) {
                diag_.error(e.args[i]->loc,
                    "argument " + std::to_string(i+1) + ": cannot convert '" +
                    argTy->str() + "' to '" + ft->paramTypes[i]->str() + "'");
            }
        }
    }
    return ft->returnType;
}

TypePtr Sema::checkSubscript(SubscriptExpr &e) {
    TypePtr baseTy  = checkExpr(*e.base);
    TypePtr idxTy   = checkExpr(*e.index);

    if (!idxTy->isInteger()) {
        diag_.error(e.index->loc, "array index must be integer");
    }

    if (baseTy->kind == TypeKind::Array) {
        e.isLValue = true;
        return static_cast<ArrayType &>(*baseTy).element;
    }
    if (baseTy->kind == TypeKind::Pointer) {
        if (!inUnsafeScope())
            diag_.error(e.loc, "pointer subscript requires 'unsafe' block");
        e.isLValue = true;
        return static_cast<PointerType &>(*baseTy).base;
    }
    if (baseTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*baseTy);
        if (rt.base->kind == TypeKind::Array) {
            e.isLValue = rt.mut;
            return static_cast<ArrayType &>(*rt.base).element;
        }
    }
    diag_.error(e.loc, "subscript on non-array type '" + baseTy->str() + "'");
    return makeError();
}

TypePtr Sema::checkMember(MemberExpr &e) {
    TypePtr baseTy = checkExpr(*e.base);

    // Auto-dereference references for dot access (SafeC references are non-null
    // and always valid — dot access on a reference implicitly dereferences it)
    if (!e.isArrow && baseTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*baseTy);
        checkNullabilityDeref(baseTy, e.loc);
        e.isLValue = rt.mut;
        baseTy = rt.base;
    }

    // Arrow: base must be pointer or reference to struct
    if (e.isArrow) {
        if (baseTy->kind == TypeKind::Pointer) {
            if (!inUnsafeScope())
                diag_.error(e.loc, "pointer member access requires 'unsafe' block");
            baseTy = static_cast<PointerType &>(*baseTy).base;
        } else if (baseTy->kind == TypeKind::Reference) {
            auto &rt = static_cast<ReferenceType &>(*baseTy);
            checkNullabilityDeref(baseTy, e.loc);
            baseTy = rt.base;
            e.isLValue = rt.mut;
        } else {
            diag_.error(e.loc, "'->' requires pointer or reference, got '" + baseTy->str() + "'");
            return makeError();
        }
    }

    StructType *st = asStruct(baseTy);
    if (!st) {
        diag_.error(e.loc, "member access on non-struct type '" + baseTy->str() + "'");
        return makeError();
    }
    const FieldDecl *fd = st->findField(e.field);
    if (!fd) {
        diag_.error(e.loc, "struct '" + st->name + "' has no field '" + e.field + "'");
        return makeError();
    }
    if (!e.isArrow) e.isLValue = e.base->isLValue;
    return fd->type;
}

TypePtr Sema::checkCast(CastExpr &e) {
    e.targetType = resolveType(e.targetType);
    TypePtr srcTy = checkExpr(*e.operand);

    // In safe code, cast from reference to pointer is not allowed
    if (!inUnsafeScope()) {
        if (srcTy->kind == TypeKind::Reference &&
            e.targetType->kind == TypeKind::Pointer) {
            diag_.error(e.loc,
                "cast from safe reference to raw pointer requires 'unsafe' block");
        }
    }
    // Numeric casts are always explicit (safe in SafeC)
    return e.targetType;
}

TypePtr Sema::checkAssign(AssignExpr &e) {
    TypePtr lhsTy = checkExpr(*e.lhs);
    TypePtr rhsTy = checkExpr(*e.rhs);

    if (!e.lhs->isLValue) {
        diag_.error(e.loc, "left side of assignment must be lvalue");
    }

    // Check const assignment
    if (e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        if (ident.resolved && ident.resolved->isConst) {
            diag_.error(e.loc, "cannot assign to const variable '" + ident.name + "'");
        }
    }

    // Region escape: RHS &stack can't be assigned to variable with wider scope
    if (rhsTy && rhsTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*rhsTy);
        if (rt.region == Region::Stack && e.lhs->kind == ExprKind::Ident) {
            auto &ident = static_cast<IdentExpr &>(*e.lhs);
            if (ident.resolved && ident.resolved->scopeDepth < scopeDepth_) {
                diag_.error(e.loc,
                    "cannot assign '&stack " + rt.base->str() +
                    "' to variable in outer scope: reference would escape");
            }
        }
    }

    // Type compatibility
    if (!lhsTy->isError() && !rhsTy->isError()) {
        if (!canImplicitlyConvert(rhsTy, lhsTy)) {
            diag_.error(e.loc,
                "type mismatch in assignment: cannot convert '" +
                rhsTy->str() + "' to '" + lhsTy->str() + "'");
        }
    }

    // Mark variable as initialized
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        if (auto *sym = lookup(ident.name)) sym->initialized = true;
        if (ident.resolved) ident.resolved->initialized = true;
    }

    return lhsTy;
}

// ── Region safety helpers ─────────────────────────────────────────────────────
void Sema::checkRegionEscape(const TypePtr &ty, int targetScopeDepth,
                              SourceLocation loc, const char *ctx) {
    if (!ty || ty->kind != TypeKind::Reference) return;
    auto &rt = static_cast<const ReferenceType &>(*ty);
    if (rt.region == Region::Stack && targetScopeDepth < scopeDepth_) {
        diag_.error(loc, std::string(ctx) + ": '&stack " + rt.base->str() +
                    "' cannot escape its scope");
    }
}

void Sema::checkNullabilityDeref(const TypePtr &ty, SourceLocation loc) {
    if (!ty || ty->kind != TypeKind::Reference) return;
    auto &rt = static_cast<const ReferenceType &>(*ty);
    if (rt.nullable) {
        diag_.error(loc,
            "dereference of nullable reference '?" + std::string("&") + rt.base->str() +
            "' without null check; use if-null-check before dereferencing");
    }
}

// ── Alias tracking ────────────────────────────────────────────────────────────
void Sema::trackRef(const std::string &targetVar, bool isMut, int depth) {
    auto &records = aliasMap_[targetVar];
    if (isMut) {
        // Check: no other mutable or immutable refs at same depth
        for (auto &r : records) {
            if (r.scopeDepth == depth) {
                diag_.error({},
                    "cannot create mutable reference to '" + targetVar +
                    "': already referenced in same scope");
                return;
            }
        }
    }
    records.push_back({targetVar, isMut, depth});
}

void Sema::untrackScope(int depth) {
    for (auto &[name, records] : aliasMap_) {
        records.erase(std::remove_if(records.begin(), records.end(),
            [depth](const AliasRecord &r){ return r.scopeDepth >= depth; }),
            records.end());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
bool Sema::isNumeric(const TypePtr &ty) const {
    return ty && ty->isArithmetic();
}

bool Sema::isIntegral(const TypePtr &ty) const {
    return ty && ty->isInteger();
}

bool Sema::canImplicitlyConvert(const TypePtr &from, const TypePtr &to) const {
    if (!from || !to) return false;
    if (from->isError() || to->isError()) return true; // suppress cascaded errors
    if (typeEqual(from, to)) return true;
    // void* → any pointer is allowed in unsafe (checked by caller)
    // bool ↔ integer (narrowing check needed but simplified here)
    if (from->isBool() && to->isInteger()) return true;
    if (from->isInteger() && to->isBool()) return true;
    // null literal (void*) → any pointer/reference
    if (from->kind == TypeKind::Pointer && to->kind == TypeKind::Pointer) {
        auto &fp = static_cast<const PointerType &>(*from);
        if (fp.base->isVoid()) return true;
    }
    if (from->kind == TypeKind::Pointer && to->kind == TypeKind::Reference) return false;
    if (from->kind == TypeKind::Reference && to->kind == TypeKind::Reference) {
        auto &fr = static_cast<const ReferenceType &>(*from);
        auto &tr = static_cast<const ReferenceType &>(*to);
        // Same region, same base, nullable compat: non-null → nullable is OK
        if (!typeEqual(fr.base, tr.base)) return false;
        if (fr.region != tr.region) return false;
        if (!fr.nullable && tr.nullable) return true;  // &T → ?&T is OK
        if (fr.nullable && !tr.nullable) return false; // ?&T → &T is not OK
        return true;
    }
    return false;
}

StructType *Sema::asStruct(TypePtr &ty) {
    if (ty && ty->kind == TypeKind::Struct) return static_cast<StructType *>(ty.get());
    // Look up by name
    if (ty && ty->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*ty);
        auto it = typeRegistry_.find(st.name);
        if (it != typeRegistry_.end() && it->second->kind == TypeKind::Struct) {
            ty = it->second;
            return static_cast<StructType *>(ty.get());
        }
    }
    return nullptr;
}

TypePtr Sema::unify(TypePtr a, TypePtr b, SourceLocation loc, const char *ctx) {
    if (typeEqual(a, b)) return a;
    if (a->isError()) return b;
    if (b->isError()) return a;
    diag_.error(loc, std::string(ctx) + ": type mismatch between '" +
                a->str() + "' and '" + b->str() + "'");
    return makeError();
}

} // namespace safec
