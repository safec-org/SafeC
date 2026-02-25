#include "safec/Sema.h"
#include "safec/Clone.h"
#include <cassert>
#include <algorithm>

namespace safec {

// ── Operator overload helper ──────────────────────────────────────────────────
std::string Sema::binaryOpToMethodName(BinaryOp op) {
    switch (op) {
    case BinaryOp::Add: return "operator+";
    case BinaryOp::Sub: return "operator-";
    case BinaryOp::Mul: return "operator*";
    case BinaryOp::Div: return "operator/";
    case BinaryOp::Mod: return "operator%";
    case BinaryOp::Eq:  return "operator==";
    case BinaryOp::NEq: return "operator!=";
    case BinaryOp::Lt:  return "operator<";
    case BinaryOp::Gt:  return "operator>";
    case BinaryOp::LEq: return "operator<=";
    case BinaryOp::GEq: return "operator>=";
    default: return "";
    }
}

const Sema::TraitDef Sema::builtinTraits_[] = {
    { "Numeric",    {"operator+", "operator-", "operator*", "operator/"} },
    { "Eq",         {"operator==", "operator!="} },
    { "Ord",        {"operator<", "operator>", "operator<=", "operator>="} },
    { "Add",        {"operator+"} },
    { "Sub",        {"operator-"} },
    { "Mul",        {"operator*"} },
    { "Div",        {"operator/"} },
    { "", {} }  // sentinel
};

bool Sema::satisfiesTrait(const TypePtr &ty, const std::string &trait) const {
    if (trait.empty()) return true;
    // Primitive numeric types satisfy Numeric/Ord/Eq/Add/Sub/Mul/Div
    if (ty->isArithmetic()) return true;
    // Struct types: check if they have the required operator methods
    if (ty->kind == TypeKind::Struct) {
        auto &st = static_cast<const StructType &>(*ty);
        for (auto *td = builtinTraits_; !td->name.empty(); ++td) {
            if (td->name == trait) {
                for (auto &op : td->requiredOps) {
                    std::string key = st.name + "::" + op;
                    if (methodRegistry_.find(key) == methodRegistry_.end())
                        return false;
                }
                return true;
            }
        }
    }
    return false;
}

// ── Scope ─────────────────────────────────────────────────────────────────────
Symbol *Scope::lookup(const std::string &name) {
    auto it = symbols_.find(name);
    return (it != symbols_.end()) ? &it->second : nullptr;
}

bool Scope::define(Symbol sym) {
    auto [it, inserted] = symbols_.emplace(sym.name, sym);
    if (inserted) return true;
    // Allow a function definition to follow a forward declaration:
    // if the existing entry has no body and the new one does, update it.
    if (it->second.kind == SymKind::Function && sym.kind == SymKind::Function) {
        bool existingHasBody = it->second.fnDecl && it->second.fnDecl->body;
        bool newHasBody      = sym.fnDecl && sym.fnDecl->body;
        if (!existingHasBody && newHasBody) {
            it->second = std::move(sym);
            return true;
        }
        // Two declarations (both without body) for the same function are fine
        // (idempotent re-declaration; e.g. header included multiple times).
        if (!existingHasBody && !newHasBody) return true;
    }
    return false;
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

    // Append monomorphized function instances to the TU so CodeGen sees them
    for (auto &fn : monoFunctions_)
        tu_.decls.emplace_back(std::move(fn));
    monoFunctions_.clear();

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
        case DeclKind::TypeAlias: {
            // Register the typedef so that resolveType() can resolve names like
            // `size_t`, `FILE`, `dev_t`, etc. imported from C headers.
            auto &ta = static_cast<TypeAliasDecl &>(*d);
            TypePtr resolved = resolveType(ta.aliasedType);
            typeRegistry_[ta.name] = resolved ? resolved : ta.aliasedType;
            break;
        }
        default: break;
        }
    }
}

// Replace StructType{name} with GenericType{name} when 'name' is in the
// generic parameter list.  Recurses into pointer/reference/array/function types.
static TypePtr resolveGenericNames(const TypePtr &ty,
                                    const std::vector<GenericParam> &gps) {
    if (!ty || gps.empty()) return ty;
    if (ty->kind == TypeKind::Struct) {
        auto &st = static_cast<const StructType &>(*ty);
        if (!st.isDefined) {
            for (auto &gp : gps)
                if (gp.name == st.name)
                    return std::make_shared<GenericType>(gp.name, gp.constraint);
        }
    }
    if (ty->kind == TypeKind::Pointer) {
        auto &pt = static_cast<const PointerType &>(*ty);
        auto nb = resolveGenericNames(pt.base, gps);
        return (nb == pt.base) ? ty : makePointer(nb, pt.isConst);
    }
    if (ty->kind == TypeKind::Reference) {
        auto &rt = static_cast<const ReferenceType &>(*ty);
        auto nb = resolveGenericNames(rt.base, gps);
        return (nb == rt.base) ? ty
             : makeReference(nb, rt.region, rt.nullable, rt.mut, rt.arenaName);
    }
    if (ty->kind == TypeKind::Array) {
        auto &at = static_cast<const ArrayType &>(*ty);
        auto ne = resolveGenericNames(at.element, gps);
        return (ne == at.element) ? ty : makeArray(ne, at.size);
    }
    return ty;
}

void Sema::collectFunction(FunctionDecl &fn) {
    // Method support (OBJECT.md §4): T::m(P...) → T_m(T* self, P...)
    if (fn.isMethod) {
        // Look up owner struct type
        auto it = typeRegistry_.find(fn.methodOwner);
        if (it == typeRegistry_.end()) {
            diag_.error(fn.loc,
                "method owner struct '" + fn.methodOwner + "' not declared");
        } else {
            // Build self param: &stack T (reference) so self.field works in safe code.
            // Const methods get immutable reference; mutable methods get mutable reference.
            TypePtr selfTy = makeReference(it->second, Region::Stack,
                                            /*nullable=*/false, /*mut=*/!fn.isConstMethod);
            ParamDecl selfParam;
            selfParam.name = "self";
            selfParam.type = selfTy;
            selfParam.loc  = fn.loc;
            fn.params.insert(fn.params.begin(), std::move(selfParam));
        }
        // Mangle name: T_m
        std::string originalName = fn.name;
        fn.name = fn.methodOwner + "_" + fn.name;
        // Register in method registry under original qualified name
        methodRegistry_[fn.methodOwner + "::" + originalName] = &fn;
    }

    std::vector<TypePtr> paramTys;
    for (auto &p : fn.params) {
        p.type = resolveGenericNames(resolveType(p.type), fn.genericParams);
        paramTys.push_back(p.type);
    }
    fn.returnType = resolveGenericNames(resolveType(fn.returnType), fn.genericParams);
    auto ft = makeFunction(fn.returnType, std::move(paramTys), fn.isVariadic);
    // Function names are &static references to their function type
    auto refTy = makeReference(ft, Region::Static);
    Symbol sym;
    sym.kind   = SymKind::Function;
    sym.name   = fn.name;
    sym.type   = refTy;
    sym.fnDecl = &fn;
    sym.initialized = true;
    define(std::move(sym));
}

void Sema::collectStruct(StructDecl &sd) {
    auto st = std::make_shared<StructType>(sd.name, sd.isUnion);
    st->isDefined = true;
    st->isPacked  = sd.isPacked;
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
    rd.declScopeDepth = scopeDepth_;
    regionRegistry_[rd.name] = &rd;
    // Register __arena_reset_<name> as a callable function in global scope
    {
        auto fn = std::make_unique<FunctionDecl>("__arena_reset_" + rd.name, rd.loc);
        fn->returnType = makeVoid();
        fn->isExtern = false;
        Symbol sym;
        sym.kind = SymKind::Function;
        sym.name = fn->name;
        sym.type = makeReference(fn->funcType(), Region::Static);
        sym.fnDecl = fn.get();
        // Store in a stable location (leak intentionally — lives as long as compiler)
        static std::vector<std::unique_ptr<FunctionDecl>> arenaResetFns;
        arenaResetFns.push_back(std::move(fn));
        sym.fnDecl = arenaResetFns.back().get();
        define(sym);
    }
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
    // Generic templates: body is only type-checked on concrete instantiations
    if (!fn.genericParams.empty()) return;

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
    case StmtKind::Defer: {
        auto &ds = static_cast<DeferStmt &>(s);
        if (ds.body) checkStmt(*ds.body, fn);
        break;
    }
    case StmtKind::Match: {
        auto &ms = static_cast<MatchStmt &>(s);
        checkExpr(*ms.subject);
        bool hasWildcard = false;
        for (auto &arm : ms.arms) {
            for (auto &pat : arm.patterns)
                if (pat.kind == PatternKind::Wildcard) hasWildcard = true;
            if (arm.body) checkStmt(*arm.body, fn);
        }
        if (!hasWildcard)
            diag_.warn(ms.loc, "match statement may not be exhaustive (no wildcard arm)");
        break;
    }
    case StmtKind::Expr: {
        auto &es = static_cast<ExprStmt &>(s);
        if (es.expr) {
            checkExpr(*es.expr);
            // must_use warning: discarding return value of a must_use function
            if (es.expr->kind == ExprKind::Call) {
                auto &call = static_cast<CallExpr &>(*es.expr);
                if (call.callee && call.callee->kind == ExprKind::Ident) {
                    auto &id = static_cast<IdentExpr &>(*call.callee);
                    if (id.resolvedFn && id.resolvedFn->isMustUse)
                        diag_.warn(es.expr->loc,
                            "return value of '" + id.resolvedFn->name +
                            "' should not be ignored (marked must_use)");
                }
            }
        }
        break;
    }
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

        // Region escape check: returned reference must not be &stack or &arena T
        if (valTy && valTy->kind == TypeKind::Reference) {
            auto &rt = static_cast<ReferenceType &>(*valTy);
            if (rt.region == Region::Stack) {
                diag_.error(s.loc,
                    "cannot return '&stack " + rt.base->str() +
                    "': stack reference escapes function scope");
            } else if (rt.region == Region::Arena) {
                diag_.error(s.loc,
                    "cannot return '&arena<" + rt.arenaName + "> " +
                    rt.base->str() + "': arena reference escapes function scope");
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

        // Borrow checker: if declaring a reference var from address-of, track the borrow
        if (!inUnsafeScope() && s.declType && s.declType->kind == TypeKind::Reference) {
            auto &rt = static_cast<ReferenceType &>(*s.declType);
            // If init is address-of a named variable, trackRef was already called
            // in checkAddrOf. For explicit reference bindings (&stack T r = &x),
            // the trackRef was already done. Nothing extra needed here.
        }
        // Also track if initTy is a reference to a named ident (direct ref binding)
        if (!inUnsafeScope() && initTy && initTy->kind == TypeKind::Reference &&
            s.init && s.init->kind == ExprKind::Ident) {
            auto &rt    = static_cast<ReferenceType &>(*initTy);
            auto *ident = static_cast<IdentExpr *>(s.init.get());
            if (ident->resolved && rt.region != Region::Static) {
                trackRef(ident->name, rt.mut, scopeDepth_);
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
    case ExprKind::New:
        ty = checkNew(static_cast<NewExpr &>(e));
        break;
    case ExprKind::TupleLit:
        ty = checkTupleLit(static_cast<TupleLitExpr &>(e));
        break;
    case ExprKind::Spawn:
        ty = checkSpawn(static_cast<SpawnExpr &>(e));
        break;
    case ExprKind::Try: {
        auto &te = static_cast<TryExpr &>(e);
        TypePtr innerTy = checkExpr(*te.inner);
        if (!innerTy || innerTy->kind != TypeKind::Optional) {
            if (innerTy && !innerTy->isError())
                diag_.error(e.loc, "try: operand must be an optional type '?T', got '"
                            + innerTy->str() + "'");
            ty = makeError();
        } else {
            // try ?T → T (unwrapped inner type)
            ty = static_cast<OptionalType &>(*innerTy).inner;
        }
        break;
    }
    default:
        ty = makeError();
    }
    e.type = ty;
    return ty;
}

TypePtr Sema::checkIntLit(IntLitExpr &e) {
    // LL suffix forces 64-bit; U suffix forces unsigned
    if (e.isLongLong && e.isUnsigned) return makeInt(64, false);
    if (e.isLongLong)                 return makeInt(64, true);
    if (e.isUnsigned) {
        // U suffix: choose smallest unsigned type
        uint64_t uval = static_cast<uint64_t>(e.value);
        if (uval <= 4294967295ULL) return makeInt(32, false);
        return makeInt(64, false);
    }
    // No suffix: choose smallest fitting signed type
    if (e.value >= -2147483648LL && e.value <= 2147483647LL) return makeInt(32);
    return makeInt(64);
}

TypePtr Sema::checkFloatLit(FloatLitExpr &e) {
    return makeFloat(64); // default double
}

TypePtr Sema::checkBoolLit(BoolLitExpr &) { return makeBool(); }

TypePtr Sema::checkStringLit(StringLitExpr &e) {
    // String literal → &static const char (zero-terminated C string)
    return makeReference(makeChar(), Region::Static, false, false);
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
    // Track for borrow checker
    if (!inUnsafeScope() && e.operand && e.operand->kind == ExprKind::Ident) {
        auto *ident = static_cast<IdentExpr *>(e.operand.get());
        if (ident->resolved) {
            trackRef(ident->name, /*isMut=*/true, scopeDepth_);
        }
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
        e.isLValue = true; // *ptr is always an lvalue (writable through raw pointer)
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

    // Check for struct operator overloads (M1)
    if (lhsTy->kind == TypeKind::Struct) {
        auto &st = static_cast<const StructType &>(*lhsTy);
        std::string methodName = binaryOpToMethodName(e.op);
        if (!methodName.empty()) {
            std::string key = st.name + "::" + methodName;
            auto it = methodRegistry_.find(key);
            if (it != methodRegistry_.end()) {
                e.resolvedOperator = it->second;
                auto *fn = it->second;
                // Return type of operator
                return fn->returnType ? fn->returnType : makeBool();
            }
        }
    }

    switch (e.op) {
    // Arithmetic
    case BinaryOp::Add: case BinaryOp::Sub:
    case BinaryOp::Mul: case BinaryOp::Div: case BinaryOp::Mod:
        if (!lhsTy->isArithmetic() || !rhsTy->isArithmetic()) {
            // Allow pointer arithmetic for raw pointers (unsafe)
            if (lhsTy->kind == TypeKind::Pointer || rhsTy->kind == TypeKind::Pointer) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "pointer arithmetic requires 'unsafe' block");
                return lhsTy->kind == TypeKind::Pointer ? lhsTy : rhsTy;
            }
            // Array arithmetic: T[N] + int → T* (array decays to pointer)
            if (lhsTy->kind == TypeKind::Array && rhsTy->isInteger()) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "array pointer arithmetic requires 'unsafe' block");
                return makePointer(static_cast<ArrayType &>(*lhsTy).element);
            }
            if (rhsTy->kind == TypeKind::Array && lhsTy->isInteger()) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "array pointer arithmetic requires 'unsafe' block");
                return makePointer(static_cast<ArrayType &>(*rhsTy).element);
            }
            diag_.error(e.loc, "arithmetic requires numeric types");
            return makeError();
        }
        // Char promotion: char + integer → int (standard C character arithmetic)
        {
            bool lIsChar = (lhsTy->kind == TypeKind::Char || lhsTy->kind == TypeKind::Int8
                            || lhsTy->kind == TypeKind::UInt8);
            bool rIsChar = (rhsTy->kind == TypeKind::Char || rhsTy->kind == TypeKind::Int8
                            || rhsTy->kind == TypeKind::UInt8);
            // If both sides are 8-bit compatible, allow and return int (promoted)
            if (lIsChar && rIsChar) return makeInt(32);
            // One side 8-bit, other side integer: promote char to match
            if (lIsChar && rhsTy->isInteger()) return rhsTy;
            if (rIsChar && lhsTy->isInteger()) return lhsTy;
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

TypePtr Sema::checkNew(NewExpr &e) {
    // Verify region exists
    if (!e.regionName.empty()) {
        auto it = regionRegistry_.find(e.regionName);
        if (it == regionRegistry_.end()) {
            diag_.error(e.loc, "unknown region '" + e.regionName + "'");
            return makeError();
        }
    }
    e.allocType = resolveType(e.allocType);
    // Returns a reference to the allocated type in the specified arena
    Region r = e.regionName.empty() ? Region::Heap : Region::Arena;
    return makeReference(e.allocType, r, false, true, e.regionName);
}

TypePtr Sema::checkTupleLit(TupleLitExpr &e) {
    std::vector<TypePtr> elemTypes;
    for (auto &elem : e.elements)
        elemTypes.push_back(checkExpr(*elem));
    return makeTuple(std::move(elemTypes));
}

TypePtr Sema::checkSpawn(SpawnExpr &e) {
    // Check the function expression — must be &static reference to a function
    TypePtr fnTy = checkExpr(*e.fnExpr);
    if (fnTy && !fnTy->isError()) {
        bool valid = false;
        if (fnTy->kind == TypeKind::Reference) {
            auto &ref = static_cast<ReferenceType &>(*fnTy);
            if (ref.region == Region::Static && ref.base &&
                ref.base->kind == TypeKind::Function)
                valid = true;
        }
        if (!valid) {
            diag_.error(e.fnExpr->loc,
                "spawn requires &static function reference, got '" +
                fnTy->str() + "'");
        }
    }
    // Check the argument expression
    checkExpr(*e.argExpr);
    // spawn returns a pthread_t (represented as signed i64)
    return makeInt(64, true);  // pthread_t handle (long long)
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
    // ── Builtin __safec_join ──────────────────────────────────────────────────
    if (e.callee->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.callee);
        if (ident.name == "__safec_join") {
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeVoid();
            return makeVoid();
        }
    }

    // ── Method call detection: x.m(args) → T_m(self=&x, args) ───────────────
    if (e.callee->kind == ExprKind::Member || e.callee->kind == ExprKind::Arrow) {
        auto &mem = static_cast<MemberExpr &>(*e.callee);
        // Type-check the base to get the struct type
        TypePtr baseTy = checkExpr(*mem.base);
        // Unwrap references/pointers to get struct type
        TypePtr structTy = baseTy;
        if (structTy->kind == TypeKind::Reference)
            structTy = static_cast<ReferenceType &>(*structTy).base;
        if (structTy->kind == TypeKind::Pointer)
            structTy = static_cast<PointerType &>(*structTy).base;
        if (structTy->kind == TypeKind::Struct) {
            auto &st = static_cast<StructType &>(*structTy);
            std::string key = st.name + "::" + mem.field;
            auto mit = methodRegistry_.find(key);
            if (mit != methodRegistry_.end()) {
                // Found a method — transform: replace callee, store base for CodeGen
                FunctionDecl *methodFn = mit->second;
                auto newCallee = std::make_unique<IdentExpr>(methodFn->name, e.loc);
                newCallee->resolvedFn = methodFn;
                newCallee->type       = methodFn->funcType();
                // Store base expression for CodeGen to emit self pointer
                e.methodBase = std::move(mem.base);
                e.callee     = std::move(newCallee);
                // Now type-check as regular call (callee is now an IdentExpr)
                // Fall through to regular call checking below.
            } else if (!st.findField(mem.field)) {
                // Not a field and not a method — error
                diag_.error(e.loc,
                    "no method '" + mem.field + "' on type '" + st.name + "'");
                for (auto &a : e.args) checkExpr(*a);
                return makeError();
            }
        }
    }

    TypePtr calleeTy = checkExpr(*e.callee);

    // Find the FunctionDecl (for generic detection)
    FunctionDecl *fnDecl = nullptr;
    if (e.callee->kind == ExprKind::Ident)
        fnDecl = static_cast<IdentExpr &>(*e.callee).resolvedFn;

    // ── Generic function instantiation ────────────────────────────────────────
    if (fnDecl && !fnDecl->genericParams.empty()) {
        // Collect argument types for inference
        std::vector<TypePtr> argTypes;
        for (auto &a : e.args) argTypes.push_back(checkExpr(*a));

        TypeSubst subs;
        if (!inferTypeArgs(fnDecl->params, argTypes, fnDecl->genericParams, subs)) {
            diag_.error(e.loc,
                "cannot infer type arguments for generic function '" +
                fnDecl->name + "'");
            return makeError();
        }

        MonoKey key = makeMonoKey(fnDecl->name, subs, fnDecl->genericParams);
        if (!monoCache_.count(key)) {
            // Create a concrete clone
            auto clone  = cloneFunctionDecl(*fnDecl, subs);
            clone->name = mangleName(fnDecl->name, subs, fnDecl->genericParams);
            clone->genericParams.clear();

            // Register the monomorphized name in the global scope
            std::vector<TypePtr> paramTys;
            for (auto &p : clone->params) paramTys.push_back(p.type);
            auto monoFT = makeFunction(clone->returnType, paramTys, clone->isVariadic);
            Symbol sym;
            sym.kind        = SymKind::Function;
            sym.name        = clone->name;
            sym.type        = makeReference(monoFT, Region::Static);
            sym.fnDecl      = clone.get();
            sym.initialized = true;
            if (!scopes_.empty()) scopes_[0].define(std::move(sym));

            // Type-check the instantiation
            checkFunction(*clone);

            monoCache_[key] = clone.get();
            monoFunctions_.push_back(std::move(clone));
        }

        FunctionDecl *monoFn = monoCache_[key];
        // Redirect the call to the monomorphized version
        if (e.callee->kind == ExprKind::Ident) {
            auto &ident    = static_cast<IdentExpr &>(*e.callee);
            ident.resolvedFn = monoFn;
            ident.name       = monoFn->name;
        }
        return monoFn->returnType;
    }

    // ── Regular (non-generic) call ─────────────────────────────────────────────
    // Unwrap &static reference to get function type (function names are &static fn refs)
    TypePtr unwrappedTy = calleeTy;
    if (unwrappedTy->kind == TypeKind::Reference)
        unwrappedTy = static_cast<ReferenceType &>(*unwrappedTy).base;
    if (unwrappedTy->kind == TypeKind::Pointer)
        unwrappedTy = static_cast<PointerType &>(*unwrappedTy).base;
    FunctionType *ft = nullptr;
    if (unwrappedTy->kind == TypeKind::Function)
        ft = static_cast<FunctionType *>(unwrappedTy.get());

    if (!ft) {
        if (!calleeTy->isError())
            diag_.error(e.loc, "expression is not callable");
        for (auto &a : e.args) checkExpr(*a);
        return makeError();
    }

    // For method calls, the first param is the implicit 'self' — skip it for
    // arg count/type checking (self is passed by CodeGen from methodBase).
    size_t selfOffset = (e.methodBase ? 1 : 0);

    // Check arg count
    size_t expected = ft->paramTypes.size();
    size_t provided = e.args.size();
    size_t effExpected = expected - selfOffset;
    if (provided < effExpected || (!ft->variadic && provided > effExpected)) {
        diag_.error(e.loc,
            "wrong number of arguments: expected " + std::to_string(effExpected) +
            ", got " + std::to_string(provided));
    }

    for (size_t i = 0; i < e.args.size(); ++i) {
        TypePtr argTy = checkExpr(*e.args[i]);
        size_t  paramIdx = i + selfOffset;
        if (paramIdx < expected &&
                !argTy->isError() && !ft->paramTypes[paramIdx]->isError()) {
            if (!canImplicitlyConvert(argTy, ft->paramTypes[paramIdx])) {
                diag_.error(e.args[i]->loc,
                    "argument " + std::to_string(i+1) + ": cannot convert '" +
                    argTy->str() + "' to '" + ft->paramTypes[paramIdx]->str() + "'");
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
        auto &at = static_cast<ArrayType &>(*baseTy);
        // Static bounds check: constant index + known array size
        if (!inUnsafeScope() && at.size > 0 &&
            e.index && e.index->kind == ExprKind::IntLit) {
            auto *lit = static_cast<IntLitExpr *>(e.index.get());
            if (lit->value < 0 || lit->value >= at.size) {
                diag_.error(e.loc,
                    "array index " + std::to_string(lit->value) +
                    " out of bounds for array of size " + std::to_string(at.size));
            }
        }
        e.boundsCheckOmit = inUnsafeScope();
        e.isLValue = true;
        return at.element;
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

    // Tuple field access: tuple.0, tuple.1, ...
    if (baseTy->kind == TypeKind::Tuple) {
        auto &tt = static_cast<const TupleType &>(*baseTy);
        // Field name is a decimal integer string — parse without exceptions
        bool validIdx = !e.field.empty();
        for (char c : e.field) if (!isdigit((unsigned char)c)) { validIdx = false; break; }
        if (validIdx) {
            size_t idx = static_cast<size_t>(std::stoul(e.field));
            if (idx < tt.elementTypes.size()) {
                e.isLValue = e.base->isLValue;
                return tt.elementTypes[idx];
            }
        }
        diag_.error(e.loc, "invalid tuple field '" + e.field + "'");
        return makeError();
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
    // For simple assignment (=), writing to a previously-uninitialized variable
    // is valid and should initialize it. Pre-mark the symbol as initialized so
    // checkIdent doesn't fire a false "uninitialized" error on the LHS.
    // Compound assignments (+=, etc.) read before writing — no pre-marking.
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        if (auto *sym = lookup(ident.name))
            if (sym->kind == SymKind::Variable)
                sym->initialized = true;
    }
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

    // Region escape: RHS &stack/&arena can't be assigned to variable with wider scope
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
        if (rt.region == Region::Arena && e.lhs->kind == ExprKind::Ident) {
            auto &ident = static_cast<IdentExpr &>(*e.lhs);
            if (ident.resolved) {
                auto it = regionRegistry_.find(rt.arenaName);
                int minDepth = (it != regionRegistry_.end()) ? it->second->declScopeDepth : 0;
                if (ident.resolved->scopeDepth < minDepth) {
                    diag_.error(e.loc,
                        "cannot assign '&arena<" + rt.arenaName + "> " + rt.base->str() +
                        "' to variable in outer scope: arena reference would escape");
                }
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
    if (rt.region == Region::Arena) {
        RegionDecl *rd = regionRegistry_.count(rt.arenaName)
                            ? regionRegistry_[rt.arenaName] : nullptr;
        int minDepth = rd ? rd->declScopeDepth : 0;
        if (targetScopeDepth < minDepth) {
            diag_.error(loc, std::string(ctx) + ": '&arena<" + rt.arenaName +
                        "> " + rt.base->str() + "' cannot escape its arena scope");
        }
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
void Sema::trackRef(const std::string &var, bool isMut, int depth) {
    auto &recs = aliasMap_[var];
    for (auto &r : recs) {
        if (r.scopeDepth < depth) continue; // outer scope — no conflict
        if (isMut) {
            diag_.error({},
                "cannot borrow '" + var + "' as mutable: already borrowed");
        } else if (r.isMutable) {
            diag_.error({},
                "cannot borrow '" + var + "' as immutable: already mutably borrowed");
        }
        return;
    }
    recs.push_back({var, isMut, depth});
}

void Sema::untrackScope(int depth) {
    for (auto &[name, records] : aliasMap_) {
        records.erase(std::remove_if(records.begin(), records.end(),
            [depth](const AliasRecord &r){ return r.scopeDepth >= depth; }),
            records.end());
    }
}

// ── Generics monomorphization helpers ─────────────────────────────────────────

bool Sema::matchType(const TypePtr &paramTy, const TypePtr &argTy, TypeSubst &subs) {
    if (!paramTy || !argTy) return false;
    if (paramTy->kind == TypeKind::Generic) {
        auto &gt = static_cast<const GenericType &>(*paramTy);
        auto it = subs.find(gt.name);
        if (it != subs.end()) return typeEqual(it->second, argTy);
        subs[gt.name] = argTy;
        return true;
    }
    if (paramTy->kind != argTy->kind) return false;
    switch (paramTy->kind) {
    case TypeKind::Pointer:
        return matchType(static_cast<const PointerType &>(*paramTy).base,
                         static_cast<const PointerType &>(*argTy).base, subs);
    case TypeKind::Reference:
        return matchType(static_cast<const ReferenceType &>(*paramTy).base,
                         static_cast<const ReferenceType &>(*argTy).base, subs);
    case TypeKind::Array:
        return matchType(static_cast<const ArrayType &>(*paramTy).element,
                         static_cast<const ArrayType &>(*argTy).element, subs);
    default:
        return typeEqual(paramTy, argTy);
    }
}

bool Sema::inferTypeArgs(const std::vector<ParamDecl>   &params,
                          const std::vector<TypePtr>      &argTypes,
                          const std::vector<GenericParam> &genericParams,
                          TypeSubst                       &subs) {
    // Helper: does this type tree contain a generic param?
    auto hasGeneric = [&](const auto &self, const TypePtr &ty) -> bool {
        if (!ty) return false;
        if (ty->kind == TypeKind::Generic) return true;
        if (ty->kind == TypeKind::Pointer)
            return self(self, static_cast<const PointerType &>(*ty).base);
        if (ty->kind == TypeKind::Reference)
            return self(self, static_cast<const ReferenceType &>(*ty).base);
        if (ty->kind == TypeKind::Array)
            return self(self, static_cast<const ArrayType &>(*ty).element);
        return false;
    };

    for (size_t i = 0; i < params.size() && i < argTypes.size(); ++i) {
        // Only fail inference if a *generic* position cannot be matched.
        // Mismatches on concrete (non-generic) params are left to the
        // regular argument-type-check that runs after monomorphization.
        if (hasGeneric(hasGeneric, params[i].type)) {
            if (!matchType(params[i].type, argTypes[i], subs))
                return false;
        }
    }
    // Verify all generic params are bound
    for (auto &gp : genericParams)
        if (!subs.count(gp.name)) return false;
    return true;
}

Sema::MonoKey Sema::makeMonoKey(const std::string &name, const TypeSubst &subs,
                                  const std::vector<GenericParam> &params) {
    MonoKey key;
    key.funcName = name;
    for (auto &gp : params) {
        auto it = subs.find(gp.name);
        key.typeArgStrs.push_back(it != subs.end() ? it->second->str() : "?");
    }
    return key;
}

std::string Sema::mangleName(const std::string &base, const TypeSubst &subs,
                               const std::vector<GenericParam> &params) {
    std::string result = "__safec_" + base;
    for (auto &gp : params) {
        auto it = subs.find(gp.name);
        result += "_" + (it != subs.end() ? it->second->str() : "unknown");
    }
    return result;
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
    // Char ↔ Int8 ↔ UInt8: 8-bit types interop (char, signed char, unsigned char)
    auto is8bit = [](const TypePtr &t) {
        return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
               t->kind == TypeKind::UInt8 || t->kind == TypeKind::Bool;
    };
    if (is8bit(from) && is8bit(to)) return true;
    // Character promotion: char → any integer type (widening, always safe)
    if (from->kind == TypeKind::Char && to->isInteger()) return true;
    // Array-to-pointer decay: T[N] → T*
    if (from->kind == TypeKind::Array && to->kind == TypeKind::Pointer) {
        auto &at = static_cast<const ArrayType &>(*from);
        auto &pt = static_cast<const PointerType &>(*to);
        if (typeEqual(at.element, pt.base)) return true;
        // also allow 8-bit element type compatibility (char[] → char*, etc.)
        auto is8 = [](const TypePtr &t) {
            return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
                   t->kind == TypeKind::UInt8;
        };
        if (is8(at.element) && is8(pt.base)) return true;
    }
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
    // README §9.2: &static T is always safe — escape allowed.
    // Passing a static reference to a raw C pointer parameter does not require unsafe{}.
    if (from->kind == TypeKind::Reference && to->kind == TypeKind::Pointer) {
        auto &fr = static_cast<const ReferenceType &>(*from);
        auto &tp = static_cast<const PointerType &>(*to);
        if (fr.region == Region::Static && typeEqual(fr.base, tp.base)) return true;
        // void* target accepts any static reference (e.g. &static char → void*)
        if (fr.region == Region::Static && tp.base->isVoid()) return true;
        // Arena/Heap references → raw pointer: coerce to the base type pointer
        // (Arena pointers are just raw memory addresses from the arena bump allocator)
        if ((fr.region == Region::Arena || fr.region == Region::Heap) &&
            typeEqual(fr.base, tp.base)) return true;
        if ((fr.region == Region::Arena || fr.region == Region::Heap) &&
            tp.base->isVoid()) return true;
    }
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
