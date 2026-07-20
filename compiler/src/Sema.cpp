#include "safec/Sema.h"
#include "safec/Clone.h"
#include <cassert>
#include <cstdint>
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
    if (!ty) return false;

    // Structural traits: judged by TypeKind, not by looking up a method —
    // SafeC has no 'operator[]' overload mechanism, so these apply only to
    // the built-in kinds that are natively indexable/pointer-like. Checked
    // before the isArithmetic() shortcut below so 'int' doesn't spuriously
    // satisfy 'Indexed'/'Pointer' just for being numeric. A struct can only
    // satisfy them via a user 'trait' that names an explicit accessor
    // method instead (e.g. 'trait Indexable { int get(int i); }').
    if (trait == "Indexed") {
        return ty->kind == TypeKind::Array   || ty->kind == TypeKind::Pointer ||
               ty->kind == TypeKind::Slice   || ty->kind == TypeKind::Vector  ||
               ty->kind == TypeKind::Reference;
    }
    if (trait == "Pointer") {
        return ty->kind == TypeKind::Pointer || ty->kind == TypeKind::Reference;
    }

    // Primitive numeric types satisfy Numeric/Ord/Eq/Add/Sub/Mul/Div
    if (ty->isArithmetic()) return true;
    // Struct types: check if they have the required operator/method names —
    // whether 'trait' names a builtin trait or a user 'trait Name { ... }'
    // declaration, both are just a list of required method names.
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
        auto uit = userTraits_.find(trait);
        if (uit != userTraits_.end()) {
            for (auto &methodName : uit->second) {
                std::string key = st.name + "::" + methodName;
                if (methodRegistry_.find(key) == methodRegistry_.end())
                    return false;
            }
            return true;
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
        // A definition followed by a later bodiless re-declaration is also
        // fine (e.g. '.sc' pairs with a '.h' the way '.c' pairs with '.h':
        // a module defines its own functions after '#include'-ing its own
        // header, and callers that include both the header and the '.sc'
        // implementation in one TU will see the header's prototype again
        // after the real definition). Keep the existing (defined) entry.
        if (existingHasBody && !newHasBody) return true;
    }
    // Same '.h'/'.sc' pairing tolerance as above, for global variables:
    // 'extern struct Foo x;' in a header followed (or preceded) by the real
    // 'struct Foo x;' definition in the '.sc' file — whichever order they're
    // seen in, an extern-declaration and a real definition of the same name
    // are compatible, not a redefinition. Prefer keeping the real definition.
    if (it->second.kind == SymKind::Variable && sym.kind == SymKind::Variable) {
        bool existingIsExtern = it->second.varDecl && it->second.varDecl->isExtern;
        bool newIsExtern      = sym.varDecl && sym.varDecl->isExtern;
        if (existingIsExtern && !newIsExtern) { it->second = std::move(sym); return true; }
        if (!existingIsExtern && newIsExtern) return true;
        if (existingIsExtern && newIsExtern) return true;
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

const VarDecl *Sema::heapKeyOf(const Symbol *sym) {
    return sym->heapTrackKey ? sym->heapTrackKey : sym->varDecl;
}

TypePtr Sema::lookupType(const std::string &name) const {
    auto it = typeRegistry_.find(name);
    return (it != typeRegistry_.end()) ? it->second : nullptr;
}

TypePtr Sema::resolveType(TypePtr ty) {
    if (!ty) return makeError();
    if (ty->kind == TypeKind::Typeof) {
        auto &tt = static_cast<TypeofType &>(*ty);
        if (tt.expr) {
            auto *expr = static_cast<Expr *>(tt.expr);
            return checkExpr(*expr);
        }
        return makeError();
    }
    if (ty->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*ty);
        // 'struct Box<int>' — a reference to a generic struct template
        // with concrete type arguments (see StructType::typeArgs's doc
        // comment). Resolve each argument first (so 'Box<Pair<int,int>>'-
        // style nesting works), then monomorphize.
        if (!st.typeArgs.empty()) {
            std::vector<TypePtr> resolvedArgs;
            resolvedArgs.reserve(st.typeArgs.size());
            for (auto &arg : st.typeArgs) resolvedArgs.push_back(resolveType(arg));
            return instantiateGenericStruct(st.name, resolvedArgs, SourceLocation{});
        }
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
    if (ty->kind == TypeKind::Vector) {
        auto &vt = static_cast<VectorType &>(*ty);
        auto resolvedElem = resolveType(vt.element);
        if (resolvedElem != vt.element)
            return makeVector(resolvedElem, vt.width);
    }
    return ty;
}

// ── Pass 1: Collect top-level declarations ─────────────────────────────────────
bool Sema::run() {
    // Global scope
    pushScope();
    collectDecls(tu_);
    for (auto &d : tu_.decls) checkDecl(*d);

    // Append monomorphized generic struct instances (before the function
    // append just below, so a monomorphized struct's own out-of-line
    // method clones — which reference it by mangled name — land after
    // their owning struct's LLVM type, though CodeGen resolves struct
    // types by name lookup regardless and doesn't strictly require this
    // order).
    for (auto &sd : monoStructs_)
        tu_.decls.emplace_back(std::move(sd));
    monoStructs_.clear();

    // Append monomorphized function instances to the TU so CodeGen sees them
    for (auto &fn : monoFunctions_)
        tu_.decls.emplace_back(std::move(fn));
    monoFunctions_.clear();

    // Same for struct-internal method forward-declarations synthesized in
    // collectStruct() — bodyless, so CodeGen just emits an extern
    // declaration for them (the definition is resolved at link time,
    // possibly from a different translation unit).
    for (auto &fn : synthesizedMethods_)
        tu_.decls.emplace_back(std::move(fn));
    synthesizedMethods_.clear();

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
        case DeclKind::Newtype: {
            auto &nt = static_cast<NewtypeDecl &>(*d);
            TypePtr base = resolveType(nt.baseType);
            auto ntt = std::make_shared<NewtypeType>(nt.name, base ? base : nt.baseType);
            typeRegistry_[nt.name] = ntt;
            break;
        }
        case DeclKind::Trait: {
            // Only method *names* are recorded (see TraitDecl's comment in
            // AST.h) — satisfiesTrait() checks the same way builtinTraits_
            // does: a same-named method must exist on the candidate struct,
            // full signature compatibility isn't verified.
            auto &td = static_cast<TraitDecl &>(*d);
            std::vector<std::string> methodNames;
            for (auto &m : td.methods) methodNames.push_back(m.name);
            userTraits_[td.name] = std::move(methodNames);
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
                    return std::make_shared<GenericType>(gp.name, gp.constraints);
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
        // 'inline int Box::get() const {...}' where "Box" names a
        // generic<T> struct template (see collectStruct's early-return
        // branch — a template is never registered into typeRegistry_,
        // since there's no concrete "T" to synthesize a 'self' param
        // against yet). Defer whole cloth: instantiateGenericStruct
        // clones+finishes this the first time 'Box<SomeType>' is actually
        // instantiated, substituting the concrete struct type in for
        // 'self' at that point instead of here.
        auto genTplIt = genericStructTemplates_.find(fn.methodOwner);
        if (genTplIt != genericStructTemplates_.end()) {
            fn.isDeferredGenericMethod = true;
            // 'Box::get()' has no generic<...> prefix of its own — its "T"
            // is the *owning struct's* generic parameter, so (unlike a
            // real generic function template, whose own genericParams
            // collectFunction already resolves names against a few lines
            // below) nothing has resolved this method's bare "T"
            // references from StructType("T") to GenericType("T") yet.
            // Do it now, against the template struct's genericParams —
            // matching collectStruct's identical treatment of in-body
            // method forward-declarations — so substituteType has
            // GenericType nodes to find at instantiation time instead of
            // silently leaving them as an unrelated (and unresolvable)
            // struct type reference literally named "T".
            const auto &ownerParams = genTplIt->second->genericParams;
            fn.returnType = resolveGenericNames(resolveType(fn.returnType), ownerParams);
            for (auto &p : fn.params)
                p.type = resolveGenericNames(resolveType(p.type), ownerParams);
            genericStructMethodTemplates_[fn.methodOwner].push_back(&fn);
            return;
        }
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

    // Namespace support: 'namespace std { void foo(); }' → mangled symbol
    // 'std_foo', callable as 'std::foo(...)'. Mirrors the method-mangling
    // scheme just above (qualified name for lookup, mangled name for the
    // actual emitted/linked symbol) so CodeGen needs no namespace-specific
    // logic — it already emits/looks up functions by FunctionDecl::name.
    //
    // 'extern' declarations skip *mangling* (an extern decl's whole point
    // is to bind to a pre-existing external symbol by its exact C name —
    // e.g. 'extern void* memcpy(...)' inside 'namespace std { ... }' must
    // stay plain 'memcpy', not 'std_memcpy', or linkage against the real
    // libc symbol breaks) but are still registered for *qualified lookup*
    // — 'std::printf' (an extern decl) resolving is no different from
    // 'std::nvic_init()' (a non-extern one) resolving; only the mangling
    // decision differs, so it shouldn't also gate whether 'std::name' is
    // even recognized as a name at all. Methods stay fully excluded from
    // both — they're dispatched entirely through the separate
    // obj.method() / methodOwner mechanism above; nothing calls them by
    // 'std::'-qualified name, so registering them here would be inert.
    if (!fn.namespaceName.empty() && !fn.isMethod) {
        std::string originalName = fn.name;
        if (!fn.isExtern) {
            std::string nsPrefix = fn.namespaceName;
            size_t p;
            while ((p = nsPrefix.find("::")) != std::string::npos) nsPrefix.replace(p, 2, "_");
            fn.name = nsPrefix + "_" + fn.name;
        }
        namespaceFnRegistry_[fn.namespaceName + "::" + originalName] = &fn;
        namespaceNames_.insert(fn.namespaceName);
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

// Helper: estimate byte size of a type (for tagged union payload sizing)
static int sizeOfType(const TypePtr &ty) {
    if (!ty) return 0;
    switch (ty->kind) {
    case TypeKind::Void:    return 0;
    case TypeKind::Bool:    return 1;
    case TypeKind::Char:
    case TypeKind::Int8:
    case TypeKind::UInt8:   return 1;
    case TypeKind::Int16:
    case TypeKind::UInt16:  return 2;
    case TypeKind::Int32:
    case TypeKind::UInt32:
    case TypeKind::Float32: return 4;
    case TypeKind::Int64:
    case TypeKind::UInt64:
    case TypeKind::Float64: return 8;
    case TypeKind::Pointer:
    case TypeKind::Reference: return 8;
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*ty);
        int total = 0;
        for (auto &f : st.fields) total += sizeOfType(f.type);
        return total;
    }
    default: return 8;
    }
}

void Sema::collectStruct(StructDecl &sd) {
    // 'generic<T> struct Box { ... }' — a template, not a concrete type.
    // Field types still reference the bare name "T" (the parser has no
    // notion of generics; only Sema resolves a name against an enclosing
    // genericParams list — see resolveGenericNames), so there is no
    // concrete layout to register yet. Record the template and stop;
    // instantiateGenericStruct clones+substitutes+collectStruct()s a
    // concrete copy the first time 'Box<SomeType>' is actually resolved.
    if (!sd.genericParams.empty()) {
        // Resolve each field's raw parsed type (the parser has no notion
        // of generics — 'T value;' parses its type as an ordinary
        // StructType("T") reference, same as any other unknown type name)
        // against this struct's own generic parameter list, in place —
        // matching collectFunction's identical one-time resolution of a
        // generic function template's own param/return types. Once done,
        // substituteType (used by cloneStructDecl at instantiation time)
        // can find the GenericType nodes it needs to replace.
        for (auto &f : sd.fields)
            f.type = resolveGenericNames(resolveType(f.type), sd.genericParams);
        for (auto &md : sd.methodDecls) {
            md.returnType = resolveGenericNames(resolveType(md.returnType), sd.genericParams);
            for (auto &p : md.params)
                p.type = resolveGenericNames(resolveType(p.type), sd.genericParams);
        }
        genericStructTemplates_[sd.name] = &sd;
        return;
    }

    auto st = std::make_shared<StructType>(sd.name, sd.isUnion);
    st->isDefined = true;
    st->isPacked  = sd.isPacked;

    // Register the struct *before* resolving its fields — a self-referential
    // field like 'struct Node* next;' inside 'struct Node' needs
    // typeRegistry_["Node"] to already point at this (shared, still being
    // populated) StructType. Otherwise resolveType() falls back to the
    // parse-time placeholder (fieldless, isDefined=false) for the pointee,
    // and any later '->next'-style chained access on it fails with a
    // spurious "struct 'Node' has no field 'next'".
    sd.type = st;
    typeRegistry_[sd.name] = st;
    {
        Symbol sym;
        sym.kind  = SymKind::Type;
        sym.name  = sd.name;
        sym.type  = st;
        sym.initialized = true;
        define(std::move(sym));
    }

    int idx = 0;
    // Bitfield packing: consecutive fields with the same declared type and
    // bitWidth >= 0 share one storage slot (LLVM struct field), packed LSB-
    // first, until the unit is full or a differently-typed/non-bitfield
    // field breaks the run — the common real-world C-compiler convention.
    // 'openUnitIdx' is the shared slot index for the currently-open run of
    // same-typed bitfields; -1 means no run is open.
    int openUnitIdx = -1;
    TypePtr openUnitType;
    int openUnitUsedBits = 0;
    for (auto &f : sd.fields) {
        FieldDecl fd;
        fd.name  = f.name;
        fd.type  = resolveType(f.type);
        fd.bitWidth = f.bitWidth;
        fd.isAnonymous = f.isAnonymous;
        if (fd.bitWidth >= 0) {
            int unitBits = sizeOfType(fd.type) * 8;
            bool canReuse = openUnitIdx >= 0 && typeEqual(openUnitType, fd.type) &&
                            openUnitUsedBits + fd.bitWidth <= unitBits;
            if (!canReuse) {
                openUnitIdx = idx++;
                openUnitType = fd.type;
                openUnitUsedBits = 0;
            }
            fd.index     = openUnitIdx;
            fd.bitOffset = openUnitUsedBits;
            openUnitUsedBits += fd.bitWidth;
        } else {
            openUnitIdx = -1; // a plain field closes any open bitfield run
            fd.index = idx++;
        }
        st->fields.push_back(std::move(fd));
    }

    // Flexible array member: 'struct S { int len; char data[]; };' — an
    // unsized-array field is only meaningful as the struct's last field
    // (C11 §6.7.2.1p18). The *same* ArrayType{size=-1} also means "this
    // parameter decays to a pointer" everywhere else (function params,
    // locals, globals) — those two meanings need different CodeGen lowering
    // (a real embedded zero-length array here vs. a bare pointer there), so
    // give the struct-field occurrence a distinct size marker (0, vs. -1
    // for decay-to-pointer) that only ever appears on a FieldDecl::type,
    // never on the shared parsed-type node other contexts see. lowerType's
    // Array case treats size==0 as a genuine (zero-length) LLVM array.
    for (size_t i = 0; i < st->fields.size(); ++i) {
        auto &f = st->fields[i];
        if (f.type && f.type->kind == TypeKind::Array &&
            static_cast<ArrayType &>(*f.type).size < 0) {
            if (i + 1 != st->fields.size()) {
                diag_.error(sd.loc,
                    "flexible array member '" + f.name + "' must be the last field in '" +
                    sd.name + "'");
            } else {
                auto &at = static_cast<ArrayType &>(*f.type);
                f.type = makeArray(at.element, 0);
            }
        }
    }

    // Tagged union: compute max payload size and set tag indices
    if (sd.isUnion) {
        st->isTaggedUnion = true;
        int maxSz = 0;
        for (auto &f : st->fields) {
            int sz = sizeOfType(f.type);
            if (sz > maxSz) maxSz = sz;
        }
        st->maxPayloadSize = maxSz;
    }

    // Struct-internal method forward-declarations ('int read(...);' inside
    // the struct body) previously had no effect at all on call-site type
    // checking — only an out-of-line 'T::read(...) { body }' definition
    // registered anything in methodRegistry_, so calling a method whose
    // definition lives in another translation unit (the normal '.h'
    // declares / '.sc' defines split) failed with "no method 'read' on type
    // 'T'" even though that's exactly the intended usage. Synthesize an
    // equivalent bodyless FunctionDecl and run it through the same
    // isMethod-handling collectFunction() already uses for out-of-line
    // definitions, so the declaration alone is enough to type-check calls;
    // if a same-TU out-of-line definition also exists, whichever is
    // processed second simply overwrites methodRegistry_'s entry (and the
    // existing '.h'/'.sc' pairing logic in Scope::define/genFunctionProto
    // already merges the two FunctionDecl nodes' attributes correctly).
    for (auto &md : sd.methodDecls) {
        auto synth = std::make_unique<FunctionDecl>(md.name, md.loc);
        synth->returnType    = md.returnType;
        synth->params        = md.params;
        synth->isMethod      = true;
        synth->methodOwner   = sd.name;
        synth->isConstMethod = md.isConst;
        synth->isExtern      = true; // no body here; defined elsewhere and linked
        collectFunction(*synth);
        synthesizedMethods_.push_back(std::move(synth));
    }
}

TypePtr Sema::instantiateGenericStruct(const std::string &name,
                                        const std::vector<TypePtr> &typeArgs,
                                        SourceLocation loc) {
    auto tplIt = genericStructTemplates_.find(name);
    if (tplIt == genericStructTemplates_.end()) {
        diag_.error(loc, "'" + name + "' is not a generic struct template");
        return makeError();
    }
    StructDecl *tpl = tplIt->second;

    if (typeArgs.size() != tpl->genericParams.size()) {
        diag_.error(loc, "generic struct '" + name + "' expects " +
            std::to_string(tpl->genericParams.size()) + " type argument(s), got " +
            std::to_string(typeArgs.size()));
        return makeError();
    }

    TypeSubst subs;
    for (size_t i = 0; i < tpl->genericParams.size(); ++i) {
        auto &gp = tpl->genericParams[i];
        subs[gp.name] = typeArgs[i];
        for (auto &constraint : gp.constraints) {
            if (!satisfiesTrait(typeArgs[i], constraint)) {
                diag_.error(loc, "type '" + typeArgs[i]->str() +
                    "' does not satisfy constraint '" + constraint +
                    "' for generic parameter '" + gp.name + "' of struct '" + name + "'");
            }
        }
    }

    MonoKey key = makeMonoKey(name, subs, tpl->genericParams);
    auto cacheIt = genericStructCache_.find(key);
    if (cacheIt != genericStructCache_.end()) {
        return cacheIt->second;
    }

    // Clone the template's fields/in-body method forward-decls, substituting
    // T -> the concrete type argument(s), and give the clone a mangled,
    // collision-proof name ("Box_int") — reusing exactly the same
    // mangleName scheme generic *functions* already use, since MonoKey/
    // mangleName never assumed "name" was specifically a function name.
    auto clone = cloneStructDecl(*tpl, subs);
    clone->name = mangleName(name, subs, tpl->genericParams);
    clone->genericParams.clear();

    // collectStruct() takes the now-fully-concrete clone through the exact
    // same path an ordinary (non-generic) struct goes through — field
    // resolution, bitfield packing, tagged-union sizing, and (crucially)
    // synthesizing bodyless FunctionDecls for any in-body method forward-
    // declarations, all "just work" unmodified since clone.genericParams
    // is now empty.
    StructDecl *clonePtr = clone.get();
    collectStruct(*clonePtr);
    monoStructs_.push_back(std::move(clone));

    genericStructCache_[key] = clonePtr->type;

    // Out-of-line method *definitions* ('inline int Box::get() const {
    // ... }') deferred by collectFunction — clone each, substitute types,
    // and finish the same self-synthesis + mangling + registration
    // collectFunction() does for an ordinary method, then type-check the
    // body via checkFunction() exactly like a monomorphized generic
    // *function* already does (see the CallExpr generic-instantiation
    // code below).
    auto methodsIt = genericStructMethodTemplates_.find(name);
    if (methodsIt != genericStructMethodTemplates_.end()) {
        for (FunctionDecl *methodTpl : methodsIt->second) {
            auto methodClone = cloneFunctionDecl(*methodTpl, subs);
            std::string originalName = methodTpl->name;
            methodClone->methodOwner = clonePtr->name;

            TypePtr selfTy = makeReference(clonePtr->type, Region::Stack,
                                            /*nullable=*/false, /*mut=*/!methodClone->isConstMethod);
            ParamDecl selfParam;
            selfParam.name = "self";
            selfParam.type = selfTy;
            selfParam.loc  = methodClone->loc;
            methodClone->params.insert(methodClone->params.begin(), std::move(selfParam));

            methodClone->name = clonePtr->name + "_" + originalName;
            methodRegistry_[clonePtr->name + "::" + originalName] = methodClone.get();

            std::vector<TypePtr> paramTys;
            for (auto &p : methodClone->params) paramTys.push_back(p.type);
            auto monoFT = makeFunction(methodClone->returnType, paramTys, methodClone->isVariadic);
            Symbol sym;
            sym.kind        = SymKind::Function;
            sym.name        = methodClone->name;
            sym.type        = makeReference(monoFT, Region::Static);
            sym.fnDecl      = methodClone.get();
            sym.initialized = true;
            if (!scopes_.empty()) scopes_[0].define(std::move(sym));

            bool savedPure = checkingPure_;
            int savedLoopDepth = loopDepth_;
            auto savedLabels = functionLabels_;
            checkFunction(*methodClone);
            checkingPure_ = savedPure;
            loopDepth_ = savedLoopDepth;
            functionLabels_ = savedLabels;

            monoFunctions_.push_back(std::move(methodClone));
        }
    }

    return clonePtr->type;
}

void Sema::collectEnum(EnumDecl &ed) {
    auto et = std::make_shared<EnumType>(ed.name);
    // Determine underlying type from explicit annotation or default i32
    int bitWidth = 32;
    bool isSigned = true;
    if (ed.underlyingType) {
        auto &ut = *ed.underlyingType;
        if (ut.isInteger() && ut.kind >= TypeKind::Bool && ut.kind <= TypeKind::UInt64) {
            auto &pt = static_cast<const PrimType &>(ut);
            bitWidth = pt.bitWidth();
            isSigned = pt.isSigned();
        }
    }
    et->bitWidth = bitWidth;
    et->isSigned = isSigned;
    int64_t nextVal = 0;
    for (auto &[name, val] : ed.enumerators) {
        int64_t v = val ? *val : nextVal;
        et->enumerators.push_back({name, v});
        nextVal = v + 1;

        // Each enumerator is a constant in the scope
        Symbol sym;
        sym.kind  = SymKind::Variable;
        sym.name  = name;
        sym.type  = makeInt(bitWidth, isSigned);
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
    // Register __arena_reset_<name> and __arena_destroy_<name> as callable
    // functions in global scope (arena_reset<R>()/arena_destroy<R>() parse
    // to calls to these — see Parser.cpp's KW_arena_reset/KW_arena_destroy
    // cases).
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
    {
        auto fn = std::make_unique<FunctionDecl>("__arena_destroy_" + rd.name, rd.loc);
        fn->returnType = makeVoid();
        fn->isExtern = false;
        Symbol sym;
        sym.kind = SymKind::Function;
        sym.name = fn->name;
        sym.type = makeReference(fn->funcType(), Region::Static);
        sym.fnDecl = fn.get();
        static std::vector<std::unique_ptr<FunctionDecl>> arenaDestroyFns;
        arenaDestroyFns.push_back(std::move(fn));
        sym.fnDecl = arenaDestroyFns.back().get();
        define(sym);
    }
    // __arena_mark_<name>() -> unsigned long: snapshot of the region's
    // current byte offset, for __arena_free_to_<name>(mark) (below) to
    // rewind to — see Parser.cpp's KW_arena_mark/KW_arena_free_to cases.
    {
        auto fn = std::make_unique<FunctionDecl>("__arena_mark_" + rd.name, rd.loc);
        fn->returnType = makeInt(64, false);
        fn->isExtern = false;
        Symbol sym;
        sym.kind = SymKind::Function;
        sym.name = fn->name;
        sym.type = makeReference(fn->funcType(), Region::Static);
        sym.fnDecl = fn.get();
        static std::vector<std::unique_ptr<FunctionDecl>> arenaMarkFns;
        arenaMarkFns.push_back(std::move(fn));
        sym.fnDecl = arenaMarkFns.back().get();
        define(sym);
    }
    // __arena_free_to_<name>(mark: unsigned long) -> void: rewinds the
    // region's bump offset back to a checkpoint from __arena_mark_<name>(),
    // freeing everything allocated after it (but not before it) — a
    // partial alternative to __arena_reset_<name>()'s always-rewind-to-zero.
    {
        auto fn = std::make_unique<FunctionDecl>("__arena_free_to_" + rd.name, rd.loc);
        fn->returnType = makeVoid();
        fn->isExtern = false;
        ParamDecl markParam;
        markParam.name = "mark";
        markParam.type = makeInt(64, false);
        markParam.loc  = rd.loc;
        fn->params.push_back(markParam);
        Symbol sym;
        sym.kind = SymKind::Function;
        sym.name = fn->name;
        sym.type = makeReference(fn->funcType(), Region::Static);
        sym.fnDecl = fn.get();
        static std::vector<std::unique_ptr<FunctionDecl>> arenaFreeToFns;
        arenaFreeToFns.push_back(std::move(fn));
        sym.fnDecl = arenaFreeToFns.back().get();
        define(sym);
    }
}

void Sema::collectGlobalVar(GlobalVarDecl &gv) {
    // Namespace support — see the matching block in collectFunction for the
    // mangling scheme ("ns::name" lookup key → mangled "ns_name" symbol).
    // 'extern' globals skip *mangling* for the same linkage-preserving
    // reason extern functions do (see collectFunction) — an extern decl's
    // whole point is binding to a pre-existing symbol by its exact C name.
    // They're still registered for *qualified lookup* though: mangling and
    // lookup-registration are separate concerns that used to share one
    // gate, which meant 'std::nvic' (referencing an extern global declared
    // inside 'namespace std { }', e.g. std/hal/cortex_m.h's NVIC/SysTick/
    // SCB instances) failed to resolve at all even though the identical
    // function case ('std::nvic_init()') and a non-extern namespaced
    // global ('std::counter') both worked — an asymmetry with no reason
    // to exist, since the qualified name just needs to redirect to
    // whatever (mangled or not) symbol name the decl actually uses.
    if (!gv.namespaceName.empty()) {
        std::string originalName = gv.name;
        if (!gv.isExtern) {
            std::string nsPrefix = gv.namespaceName;
            size_t p;
            while ((p = nsPrefix.find("::")) != std::string::npos) nsPrefix.replace(p, 2, "_");
            gv.name = nsPrefix + "_" + gv.name;
        }
        namespaceVarRegistry_[gv.namespaceName + "::" + originalName] = &gv;
        namespaceNames_.insert(gv.namespaceName);
    }

    gv.type = resolveType(gv.type);

    // Without a backing VarDecl, checkIdent never sets isLValue for this
    // symbol (see its 'if (sym->varDecl) ... e.isLValue = true' check) —
    // globals would be permanently unassignable, including through struct
    // field access ('g.field = x'), which just inherits the base's isLValue.
    VarDecl *vd = new VarDecl{}; // leaked intentionally, matches param/local pattern
    vd->name        = gv.name;
    vd->type        = gv.type;
    vd->isConst     = gv.isConst;
    vd->isStatic    = true;
    vd->isGlobal    = true;
    vd->isExtern    = gv.isExtern;
    vd->initialized = (gv.init != nullptr || gv.isExtern);

    Symbol sym;
    sym.kind        = SymKind::Variable;
    sym.name        = gv.name;
    sym.type        = gv.type;
    sym.varDecl     = vd;
    sym.initialized = vd->initialized;
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
    // Out-of-line generic-struct method template — see
    // isDeferredGenericMethod's doc comment; only its cloned,
    // fully-substituted instantiations (built by instantiateGenericStruct)
    // ever get checked.
    if (fn.isDeferredGenericMethod) return;

    // ── Calling convention validation ────────────────────────────────────────
    if (!fn.callingConv.empty()) {
        if (fn.callingConv != "stdcall" && fn.callingConv != "cdecl" &&
            fn.callingConv != "fastcall") {
            diag_.error(fn.loc, "unknown calling convention '" + fn.callingConv + "'");
        }
    }

    // ── Bare-metal attribute validation ─────────────────────────────────────
    // interrupt: must be void(void)
    if (fn.isInterrupt) {
        if (fn.returnType && !fn.returnType->isVoid())
            diag_.error(fn.loc, "interrupt function must return void");
        if (!fn.params.empty())
            diag_.error(fn.loc, "interrupt function must take no parameters");
    }

    // naked: body must contain only asm statements
    if (fn.isNaked && fn.body) {
        for (auto &s : fn.body->body) {
            if (s->kind != StmtKind::Asm) {
                diag_.error(s->loc, "naked function body may only contain asm statements");
                break;
            }
        }
    }

    // noreturn: error on any return statement in body
    if (fn.isNoReturn && fn.body) {
        std::function<void(Stmt&)> checkNoRet = [&](Stmt &s) {
            if (s.kind == StmtKind::Return) {
                auto &rs = static_cast<ReturnStmt&>(s);
                if (rs.value)
                    diag_.error(s.loc, "noreturn function must not return a value");
            }
            if (s.kind == StmtKind::Compound) {
                for (auto &sub : static_cast<CompoundStmt&>(s).body)
                    checkNoRet(*sub);
            }
            if (s.kind == StmtKind::If) {
                auto &is = static_cast<IfStmt&>(s);
                if (is.then) checkNoRet(*is.then);
                if (is.else_) checkNoRet(*is.else_);
            }
        };
        checkNoRet(*fn.body);
    }

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
    // Set pure checking mode
    bool prevPure = checkingPure_;
    if (fn.isPure) checkingPure_ = true;

    // Collect all labels in this function for goto validation
    functionLabels_.clear();
    loopDepth_ = 0;
    std::function<void(Stmt&)> collectLabels = [&](Stmt &s) {
        if (s.kind == StmtKind::Label) {
            functionLabels_.insert(static_cast<LabelStmt&>(s).label);
            collectLabels(*static_cast<LabelStmt&>(s).body);
        } else if (s.kind == StmtKind::Compound) {
            for (auto &c : static_cast<CompoundStmt&>(s).body) collectLabels(*c);
        } else if (s.kind == StmtKind::If) {
            auto &i = static_cast<IfStmt&>(s);
            if (i.then) collectLabels(*i.then);
            if (i.else_) collectLabels(*i.else_);
        } else if (s.kind == StmtKind::While || s.kind == StmtKind::DoWhile) {
            collectLabels(*static_cast<WhileStmt&>(s).body);
        } else if (s.kind == StmtKind::For) {
            collectLabels(*static_cast<ForStmt&>(s).body);
        } else if (s.kind == StmtKind::Unsafe) {
            collectLabels(*static_cast<UnsafeStmt&>(s).body);
        } else if (s.kind == StmtKind::Switch) {
            for (auto &c : static_cast<SwitchStmt&>(s).cases)
                for (auto &sub : c.body) collectLabels(*sub);
        }
    };
    collectLabels(*fn.body);

    checkCompound(*fn.body, fn);

    checkingPure_ = prevPure;
    functionLabels_.clear();
    popScope();
}

void Sema::checkGlobalVar(GlobalVarDecl &gv) {
    if (gv.init) {
        TypePtr initTy = (gv.init->kind == ExprKind::Compound)
            ? checkCompoundInit(static_cast<CompoundInitExpr &>(*gv.init), gv.type)
            : checkExpr(*gv.init);
        if (!gv.type->isError() && !initTy->isError()) {
            if (!canImplicitlyConvert(initTy, gv.type) &&
                !intLiteralFitsType(*gv.init, gv.type) &&
                !floatLitFitsType(*gv.init, gv.type)) {
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

// Determines whether 'subjectTy' has the built-in nullable-pointer/
// reference or Optional match shape resolveMatchArmPatterns understands
// (null/some(x), none/some(x)), and what type a 'some(x)' pattern's
// payload binds to. Shared by the match-statement and match-expression
// subject-classification code below.
static void classifyNullMatchSubject(const TypePtr &subjectTy, bool &isNullLike,
                                      bool &isOptional, TypePtr &payloadTy) {
    isNullLike = false;
    isOptional = false;
    payloadTy  = nullptr;
    if (!subjectTy) return;
    if (subjectTy->kind == TypeKind::Pointer) {
        isNullLike = true;
        payloadTy  = static_cast<const PointerType &>(*subjectTy).base;
    } else if (subjectTy->kind == TypeKind::Reference &&
               static_cast<const ReferenceType &>(*subjectTy).nullable) {
        isNullLike = true;
        payloadTy  = static_cast<const ReferenceType &>(*subjectTy).base;
    } else if (subjectTy->kind == TypeKind::Optional) {
        isOptional = true;
        payloadTy  = static_cast<const OptionalType &>(*subjectTy).inner;
    }
}

// ── Match pattern resolution (shared by match-statement and match-expression) ──
// Resolves EnumIdent patterns against a tagged union's variant fields, or
// (when 'isNullLike'/'isOptional') against the built-in two-variant
// null/some(x) or none/some(x) shape a nullable pointer/reference or
// Optional subject has — filling resolvedTag/payloadType either way,
// reporting unknown variant/pattern names, and noting whether a
// wildcard/default arm was seen. 'coveredTags', when non-null, collects
// every resolved tag so the caller can prove exhaustiveness (match-
// expression needs this; match-statement doesn't since falling through
// with no value produced is harmless there).
static void resolveMatchArmPatterns(std::vector<MatchPattern> &patterns,
                                     bool isTaggedUnion, StructType *unionSt,
                                     bool isNullLike, bool isOptional,
                                     const TypePtr &nullPayloadTy,
                                     DiagEngine &diag, SourceLocation loc,
                                     bool &hasWildcard,
                                     std::unordered_set<int> *coveredTags,
                                     EnumType *enumSubject = nullptr) {
    for (auto &pat : patterns) {
        if (pat.kind == PatternKind::Wildcard) hasWildcard = true;
        // Plain 'enum' subject (not a tagged union/nullable/optional): an
        // EnumIdent pattern names one of the enum's own enumerators
        // ('case Monday:') — resolve it to that enumerator's integer value
        // so genMatch has something to compare the subject against.
        // Without this resolution, every EnumIdent pattern against a plain
        // enum subject silently fell through to the "not a tagged union"
        // branch below both here and in genMatch, which treated it as an
        // unconditional match — the first arm always "matched," regardless
        // of the subject's actual value.
        if (enumSubject && pat.kind == PatternKind::EnumIdent) {
            bool found = false;
            for (auto &en : enumSubject->enumerators) {
                if (en.first == pat.ident) {
                    pat.resolvedTag = (int)en.second;
                    if (coveredTags) coveredTags->insert(pat.resolvedTag);
                    found = true;
                    break;
                }
            }
            if (!found) {
                diag.error(loc, "unknown enumerator '" + pat.ident +
                    "' in enum '" + enumSubject->name + "'");
            }
            continue;
        }
        if (isTaggedUnion && pat.kind == PatternKind::EnumIdent) {
            const FieldDecl *variantField = unionSt->findField(pat.ident);
            if (variantField) {
                pat.resolvedTag = variantField->index;
                pat.payloadType = variantField->type;
                if (coveredTags) coveredTags->insert(pat.resolvedTag);
            } else {
                diag.error(loc, "unknown variant '" + pat.ident +
                    "' in union '" + unionSt->name + "'");
            }
        } else if ((isNullLike || isOptional) && pat.kind == PatternKind::EnumIdent) {
            const char *emptyName = isOptional ? "none" : "null";
            if (pat.ident == emptyName) {
                pat.resolvedTag = 0;
                if (coveredTags) coveredTags->insert(0);
                if (!pat.bindName.empty())
                    diag.error(loc, std::string("'") + emptyName +
                        "' does not carry a value to bind");
            } else if (pat.ident == "some") {
                pat.resolvedTag = 1;
                pat.payloadType = nullPayloadTy;
                if (coveredTags) coveredTags->insert(1);
            } else {
                diag.error(loc, "unknown pattern '" + pat.ident + "' — expected '" +
                    emptyName + "' or 'some(x)'");
            }
        }
    }
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
    case StmtKind::MultiVarDecl: {
        // No pushScope()/popScope() here — see MultiVarDeclStmt's header
        // comment in AST.h: each declarator must land in this statement's
        // own (enclosing) scope, same as if it had been written as its
        // own separate 'T name = init;' statement.
        auto &mv = static_cast<MultiVarDeclStmt &>(s);
        for (auto &d : mv.decls) checkStmt(*d, fn);
        break;
    }
    case StmtKind::Defer: {
        auto &ds = static_cast<DeferStmt &>(s);
        if (ds.body) checkStmt(*ds.body, fn);
        break;
    }
    case StmtKind::Match: {
        auto &ms = static_cast<MatchStmt &>(s);
        TypePtr subjectTy = checkExpr(*ms.subject);
        bool hasWildcard = false;
        // Check if subject is a tagged union
        bool isTaggedUnion = false;
        StructType *unionSt = nullptr;
        if (subjectTy && subjectTy->kind == TypeKind::Struct) {
            unionSt = static_cast<StructType *>(subjectTy.get());
            isTaggedUnion = unionSt->isTaggedUnion;
        }
        EnumType *enumSubject = nullptr;
        if (subjectTy && subjectTy->kind == TypeKind::Enum) {
            enumSubject = static_cast<EnumType *>(subjectTy.get());
        }
        bool isNullLike = false, isOptional = false;
        TypePtr nullPayloadTy;
        classifyNullMatchSubject(subjectTy, isNullLike, isOptional, nullPayloadTy);
        for (auto &arm : ms.arms) {
            resolveMatchArmPatterns(arm.patterns, isTaggedUnion, unionSt,
                                     isNullLike, isOptional, nullPayloadTy, diag_,
                                     ms.loc, hasWildcard, nullptr, enumSubject);
            // If pattern has a bind name, add it to scope for the arm body
            pushScope();
            for (auto &pat : arm.patterns) {
                if (!pat.bindName.empty() && pat.payloadType) {
                    Symbol sym;
                    sym.kind = SymKind::Variable;
                    sym.name = pat.bindName;
                    sym.type = pat.payloadType;
                    sym.initialized = true;
                    define(std::move(sym));
                }
            }
            ++loopDepth_; // break is valid inside match arms
            if (arm.body) checkStmt(*arm.body, fn);
            --loopDepth_;
            popScope();
        }
        if (!hasWildcard)
            diag_.warn(ms.loc, "match statement may not be exhaustive (no wildcard arm)");
        break;
    }
    case StmtKind::Switch: {
        auto &sw = static_cast<SwitchStmt &>(s);
        TypePtr subjectTy = checkExpr(*sw.controlling);
        if (subjectTy && !subjectTy->isError() && !subjectTy->isInteger() &&
            !subjectTy->isBool() && subjectTy->kind != TypeKind::Char) {
            diag_.error(sw.controlling->loc,
                "switch subject must be an integer, char, or bool type, got '" +
                subjectTy->str() + "'");
        }
        bool sawDefault = false;
        std::unordered_set<int64_t> seenValues;
        for (auto &c : sw.cases) {
            if (c.isDefault) {
                sawDefault = true;
            } else {
                for (int64_t v : c.values) {
                    if (!seenValues.insert(v).second)
                        diag_.error(c.loc, "duplicate case value " + std::to_string(v) + " in switch");
                }
            }
            // Own scope per case body (locals declared in one case aren't
            // visible in the next, even though execution can fall through
            // into it — matches real C: a fallthrough runs the next case's
            // statements, but goto-into-scope rules already forbid jumping
            // past a declaration with an initializer, so this doesn't hide
            // any real fallthrough-initialization footgun beyond what C
            // itself has).
            pushScope();
            ++loopDepth_; // 'break' exits the switch — see genSwitch's own break target
            for (auto &stmt : c.body) checkStmt(*stmt, fn);
            --loopDepth_;
            popScope();
        }
        if (!sawDefault)
            diag_.warn(sw.loc, "switch statement may not be exhaustive (no 'default' label)");
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
        if (loopDepth_ == 0)
            diag_.error(s.loc, "'break' statement not inside a loop or match");
        break;
    case StmtKind::Continue:
        if (loopDepth_ == 0)
            diag_.error(s.loc, "'continue' statement not inside a loop");
        break;
    case StmtKind::Goto: {
        auto &gs = static_cast<GotoStmt &>(s);
        if (functionLabels_.find(gs.label) == functionLabels_.end())
            diag_.error(s.loc, "use of undeclared label '" + gs.label + "'");
        break;
    }
    case StmtKind::Label:
        checkStmt(*static_cast<LabelStmt &>(s).body, fn);
        break;
    case StmtKind::Asm:
        checkAsmStmt(s, fn);
        break;
    default: break;
    }
}

void Sema::checkCompound(CompoundStmt &cs, FunctionDecl &fn) {
    pushScope();
    for (auto &stmt : cs.body) checkStmt(*stmt, fn);
    popScope();
}

// ── Arena-reset flow sensitivity (if/else branches, loop bodies) ───────────────
// regionGeneration_/regionMarkDepth_/regionScratchGeneration_ (see their doc
// comments in Sema.h) are plain counters bumped as checkStmt/checkExpr walks
// the AST in textual order. Two gaps that used to leave in the "flow-
// insensitive" caveat those comments documented:
//
//  1. if/else branch bleed-through: checkIf used to check 'then' then
//     'else' sequentially against the *same* counters, so a reset inside
//     'then' made 'else' see an already-bumped generation even though only
//     one of the two branches can run at a time. mergeRegionCountersMax
//     plus the snapshot/restore in checkIf below fix this: each branch now
//     starts from the same pre-if state, and the state after the whole
//     if-statement is the conservative (max) merge of what either branch
//     could have done -- still sound (a reset on either path must still be
//     assumed possible downstream), but no longer cross-contaminates the
//     *other* branch's own checking.
//
//  2. Loop-iteration gap: a reset/free_to call near the *end* of a loop body
//     used to only affect generation for code checked *after* it was
//     textually seen -- code earlier in the same body (which, on a later
//     iteration, runs *after* the previous iteration's reset/free_to) wasn't
//     flagged. collectArenaResetRegions does a best-effort syntactic
//     pre-scan of a loop body for reset/destroy/free_to calls anywhere
//     inside it and bumps the matching regions' generation
//     (reset/destroy -> regionGeneration_) or scratch generation
//     (free_to -> regionScratchGeneration_) *before* the real check pass, so
//     the whole body -- including code textually before the call -- sees
//     the "possibly invalidated by an earlier iteration" state. It's a
//     name-based scan (these calls parse to "__arena_reset_<R>"/
//     "__arena_destroy_<R>"/"__arena_free_to_<R>", see collectRegion/
//     checkCall), not a type-checking pass, so it needs no symbol
//     resolution and has no side effects of its own to worry about
//     duplicating. It covers the statement/expression forms a void-
//     returning call can plausibly appear in (blocks, if/while/for/unsafe/
//     defer/match/switch bodies, var-decl initializers, and the common
//     binary/unary/ternary/assign/cast expression nestings) rather than
//     being exhaustive over every ExprKind — deeply unusual nesting can
//     still fall back to the old (safe, just less precise) behavior. This
//     does not attempt to track free_to's mark-nesting *depth* precisely
//     across iterations (see Sema.h's regionScratchGeneration_ doc comment
//     for why nested marks remain conservative independent of
//     flow-sensitivity) -- it only closes the specific "an unpaired
//     free_to in a loop body invalidates an outer reference used earlier
//     in the same body" gap.
static void mergeRegionCountersMax(std::unordered_map<std::string, int> &dst,
                                    const std::unordered_map<std::string, int> &other) {
    for (auto &kv : other) {
        auto it = dst.find(kv.first);
        if (it == dst.end() || it->second < kv.second) dst[kv.first] = kv.second;
    }
}

// Same element-wise-max merge as above, for heapFreeGeneration_'s VarDecl*-
// keyed map (see its doc comment in Sema.h for why heap tracking is keyed
// by declaration identity rather than name).
static void mergeRegionCountersMax(std::unordered_map<const VarDecl *, int> &dst,
                                    const std::unordered_map<const VarDecl *, int> &other) {
    for (auto &kv : other) {
        auto it = dst.find(kv.first);
        if (it == dst.end() || it->second < kv.second) dst[kv.first] = kv.second;
    }
}

// Output of the syntactic loop-body pre-scan below. 'resetRegions' and
// 'freeToRegions' collect arena region names (from '__arena_reset_<R>'/
// '__arena_destroy_<R>' and '__arena_free_to_<R>' respectively — kept
// separate since they invalidate different counters, regionGeneration_ vs.
// regionScratchGeneration_). 'heapFreeVarNames' collects the *variable
// names* passed as the sole argument to a recognized 'std::dealloc(p)'
// call — unlike the region cases, there's no naming convention to extract
// a target from the callee itself, so the call site (checkWhile/checkFor)
// resolves each name via lookup() to find the VarDecl to pre-bump in
// heapFreeGeneration_; the scan itself stays purely syntactic (no symbol
// resolution, no Sema access) to match the rest of this pre-scan.
struct ArenaLoopScan {
    std::unordered_set<std::string> resetRegions;
    std::unordered_set<std::string> freeToRegions;
    std::unordered_set<std::string> heapFreeVarNames;
};

static void collectArenaResetRegions(const Expr *e, ArenaLoopScan &out) {
    if (!e) return;
    switch (e->kind) {
    case ExprKind::Call: {
        auto &ce = static_cast<const CallExpr &>(*e);
        if (ce.callee && ce.callee->kind == ExprKind::Ident) {
            auto &ident = static_cast<const IdentExpr &>(*ce.callee);
            static const std::string kResetPrefix   = "__arena_reset_";
            static const std::string kDestroyPrefix = "__arena_destroy_";
            static const std::string kFreeToPrefix  = "__arena_free_to_";
            if (ident.name.rfind(kResetPrefix, 0) == 0) {
                out.resetRegions.insert(ident.name.substr(kResetPrefix.size()));
            } else if (ident.name.rfind(kDestroyPrefix, 0) == 0) {
                out.resetRegions.insert(ident.name.substr(kDestroyPrefix.size()));
            } else if (ident.name.rfind(kFreeToPrefix, 0) == 0) {
                out.freeToRegions.insert(ident.name.substr(kFreeToPrefix.size()));
            } else if ((ident.name == "dealloc" ||
                        (ident.name.size() > 9 &&
                         ident.name.compare(ident.name.size() - 9, 9, "::dealloc") == 0)) &&
                       ce.args.size() == 1 && ce.args[0] &&
                       ce.args[0]->kind == ExprKind::Ident) {
                out.heapFreeVarNames.insert(static_cast<const IdentExpr &>(*ce.args[0]).name);
            }
        }
        collectArenaResetRegions(ce.callee.get(), out);
        for (auto &a : ce.args) collectArenaResetRegions(a.get(), out);
        break;
    }
    case ExprKind::Binary: {
        auto &be = static_cast<const BinaryExpr &>(*e);
        collectArenaResetRegions(be.left.get(), out);
        collectArenaResetRegions(be.right.get(), out);
        break;
    }
    case ExprKind::Unary:
        collectArenaResetRegions(static_cast<const UnaryExpr &>(*e).operand.get(), out);
        break;
    case ExprKind::Ternary: {
        auto &te = static_cast<const TernaryExpr &>(*e);
        collectArenaResetRegions(te.cond.get(), out);
        collectArenaResetRegions(te.then.get(), out);
        collectArenaResetRegions(te.else_.get(), out);
        break;
    }
    case ExprKind::Assign: {
        auto &ae = static_cast<const AssignExpr &>(*e);
        collectArenaResetRegions(ae.lhs.get(), out);
        collectArenaResetRegions(ae.rhs.get(), out);
        break;
    }
    case ExprKind::Cast:
        collectArenaResetRegions(static_cast<const CastExpr &>(*e).operand.get(), out);
        break;
    default:
        break; // other ExprKinds can't plausibly hold a void-returning reset/free call
    }
}

static void collectArenaResetRegions(const Stmt *s, ArenaLoopScan &out) {
    if (!s) return;
    switch (s->kind) {
    case StmtKind::Compound:
        for (auto &st : static_cast<const CompoundStmt &>(*s).body)
            collectArenaResetRegions(st.get(), out);
        break;
    case StmtKind::Expr:
        collectArenaResetRegions(static_cast<const ExprStmt &>(*s).expr.get(), out);
        break;
    case StmtKind::If: {
        auto &is = static_cast<const IfStmt &>(*s);
        collectArenaResetRegions(is.cond.get(), out);
        collectArenaResetRegions(is.then.get(), out);
        collectArenaResetRegions(is.else_.get(), out);
        break;
    }
    case StmtKind::While:
    case StmtKind::DoWhile: {
        auto &ws = static_cast<const WhileStmt &>(*s);
        collectArenaResetRegions(ws.cond.get(), out);
        collectArenaResetRegions(ws.body.get(), out);
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<const ForStmt &>(*s);
        collectArenaResetRegions(fs.init.get(), out);
        collectArenaResetRegions(fs.cond.get(), out);
        collectArenaResetRegions(fs.incr.get(), out);
        collectArenaResetRegions(fs.body.get(), out);
        break;
    }
    case StmtKind::VarDecl:
        collectArenaResetRegions(static_cast<const VarDeclStmt &>(*s).init.get(), out);
        break;
    case StmtKind::MultiVarDecl:
        for (auto &d : static_cast<const MultiVarDeclStmt &>(*s).decls)
            collectArenaResetRegions(d.get(), out);
        break;
    case StmtKind::Unsafe:
        collectArenaResetRegions(static_cast<const UnsafeStmt &>(*s).body.get(), out);
        break;
    case StmtKind::Defer:
        collectArenaResetRegions(static_cast<const DeferStmt &>(*s).body.get(), out);
        break;
    case StmtKind::Match: {
        auto &ms = static_cast<const MatchStmt &>(*s);
        collectArenaResetRegions(ms.subject.get(), out);
        for (auto &arm : ms.arms) collectArenaResetRegions(arm.body.get(), out);
        break;
    }
    case StmtKind::Switch: {
        auto &sw = static_cast<const SwitchStmt &>(*s);
        collectArenaResetRegions(sw.controlling.get(), out);
        for (auto &c : sw.cases)
            for (auto &st : c.body) collectArenaResetRegions(st.get(), out);
        break;
    }
    case StmtKind::Return:
        collectArenaResetRegions(static_cast<const ReturnStmt &>(*s).value.get(), out);
        break;
    default:
        break; // Break/Continue/Goto/Label/StaticAssert/IfConst/Asm: no reset call to find
    }
}

void Sema::checkIf(IfStmt &s, FunctionDecl &fn) {
    TypePtr condTy = checkExpr(*s.cond);
    if (!condTy->isError() && !condTy->isInteger() && !condTy->isBool() &&
        condTy->kind != TypeKind::Pointer && condTy->kind != TypeKind::Reference) {
        diag_.error(s.cond->loc, "condition must be boolean or numeric");
    }

    // See the "Arena-reset flow sensitivity" comment above item 1. Heap
    // free tracking (heapFreeGeneration_) gets the same snapshot/restore/
    // merge treatment for the same reason: a dealloc() inside 'then' must
    // not bleed into 'else''s own checking.
    auto genBefore     = regionGeneration_;
    auto markBefore    = regionMarkDepth_;
    auto scratchBefore = regionScratchGeneration_;
    auto heapBefore    = heapFreeGeneration_;

    checkStmt(*s.then, fn);
    auto genThen     = std::move(regionGeneration_);
    auto markThen    = std::move(regionMarkDepth_);
    auto scratchThen = std::move(regionScratchGeneration_);
    auto heapThen    = std::move(heapFreeGeneration_);

    regionGeneration_        = genBefore;
    regionMarkDepth_         = markBefore;
    regionScratchGeneration_ = scratchBefore;
    heapFreeGeneration_      = heapBefore;

    if (s.else_) checkStmt(*s.else_, fn);

    mergeRegionCountersMax(regionGeneration_, genThen);
    mergeRegionCountersMax(regionMarkDepth_, markThen);
    mergeRegionCountersMax(regionScratchGeneration_, scratchThen);
    mergeRegionCountersMax(heapFreeGeneration_, heapThen);
}

void Sema::checkWhile(WhileStmt &s, FunctionDecl &fn) {
    TypePtr condTy = checkExpr(*s.cond);
    if (!condTy->isError() && !condTy->isInteger() && !condTy->isBool() &&
        condTy->kind != TypeKind::Pointer) {
        diag_.error(s.cond->loc, "while condition must be boolean or numeric");
    }

    // See the "Arena-reset flow sensitivity" comment above item 2. Also
    // pre-bumps regionScratchGeneration_ for any '__arena_free_to_<R>' found
    // in the body, closing the "unpaired free_to in a loop" gap: an outer
    // reference bound before the loop (with an active mark scope already
    // open) that's read earlier in the body than the free_to call must still
    // be treated as possibly invalidated by a *previous* iteration's call.
    // Same reasoning extends heapFreeGeneration_ to std::dealloc(p) calls —
    // 'heapFreeVarNames' are resolved to VarDecls here (not inside the
    // scan itself, which stays symbol-resolution-free) via a plain lookup().
    ArenaLoopScan scan;
    collectArenaResetRegions(s.body.get(), scan);
    for (auto &r : scan.resetRegions) regionGeneration_[r]++;
    for (auto &r : scan.freeToRegions) regionScratchGeneration_[r]++;
    for (auto &name : scan.heapFreeVarNames) {
        if (auto *sym = lookup(name))
            if (sym->kind == SymKind::Variable && sym->varDecl)
                heapFreeGeneration_[heapKeyOf(sym)]++;
    }

    ++loopDepth_;
    checkStmt(*s.body, fn);
    --loopDepth_;
}

void Sema::checkFor(ForStmt &s, FunctionDecl &fn) {
    pushScope();
    if (s.init) checkStmt(*s.init, fn);
    if (s.cond) checkExpr(*s.cond);
    if (s.incr) checkExpr(*s.incr);

    // See the "Arena-reset flow sensitivity" comment above item 2, and the
    // free_to/heap-free note on checkWhile's pre-scan above.
    ArenaLoopScan scan;
    collectArenaResetRegions(s.body.get(), scan);
    for (auto &r : scan.resetRegions) regionGeneration_[r]++;
    for (auto &r : scan.freeToRegions) regionScratchGeneration_[r]++;
    for (auto &name : scan.heapFreeVarNames) {
        if (auto *sym = lookup(name))
            if (sym->kind == SymKind::Variable && sym->varDecl)
                heapFreeGeneration_[heapKeyOf(sym)]++;
    }

    ++loopDepth_;
    checkStmt(*s.body, fn);
    --loopDepth_;
    popScope();
}

void Sema::checkReturn(ReturnStmt &s, FunctionDecl &fn) {
    if (s.value) {
        TypePtr valTy = checkExpr(*s.value);

        // Region escape check: a returned '&stack T' refers to a local that's
        // gone the moment the function returns, so it can never be valid.
        // '&arena<R> T' is fine to return — regions are only ever declared
        // at the top level (see parseTopLevelDecl), so every arena already
        // outlives any individual function call; there's no such thing as a
        // arena scoped more narrowly than "the whole program".
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
            if (!canImplicitlyConvert(valTy, fn.returnType) &&
                !intLiteralFitsType(*s.value, fn.returnType) &&
                !floatLitFitsType(*s.value, fn.returnType)) {
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

    // Variable-length array: 'unsafe { int arr[n]; }' where 'n' isn't a
    // compile-time constant (ConstEvalEngine::resolveArraySizes marked this
    // with size==-2 rather than erroring, since it ran before any scope
    // existed to resolve 'n' against). Check the stored size expression now,
    // against this function's real local scope, so 'n' resolves properly
    // and a non-integer size is still caught.
    if (s.declType && s.declType->kind == TypeKind::Array) {
        auto &at = static_cast<ArrayType &>(*s.declType);
        if (at.size == -2 && at.sizeExpr) {
            auto *sizeExpr = static_cast<Expr *>(at.sizeExpr);
            TypePtr sizeTy = checkExpr(*sizeExpr);
            if (!sizeTy->isError() && !sizeTy->isInteger()) {
                diag_.error(sizeExpr->loc,
                    "variable-length array size must be an integer, got '" +
                    sizeTy->str() + "'");
            }
        }
    }

    TypePtr initTy;
    if (s.init) {
        // See expectedType_'s doc comment: an explicit declared type (not
        // auto-inferred, not already an error) is a fallback hint for
        // generic-function inference when the initializer is a call whose
        // type parameter appears only in its return type, e.g.
        // 'int* p = std::vec_at(&v, 1UL);'. checkCall consumes/clears this
        // immediately, but only if the initializer actually reaches a call
        // — always restored below regardless, so a non-call initializer
        // ('int* p = some_var;') can't leave it dangling for whatever's
        // checked next.
        TypePtr savedExpectedType = expectedType_;
        if (s.declType && !s.declType->isError()) expectedType_ = s.declType;
        initTy = (s.init->kind == ExprKind::Compound)
            ? checkCompoundInit(static_cast<CompoundInitExpr &>(*s.init), s.declType)
            : checkExpr(*s.init);
        expectedType_ = savedExpectedType;

        // Borrow checker: declaring a reference var from address-of ('&region T r = &x')
        // registers the borrow with the *declared* mutability — not always mutable —
        // so that e.g. two '&stack const T' (shared) borrows of the same variable
        // can coexist, matching the mutable-XOR-shared aliasing rule.
        if (!inUnsafeScope() && s.declType && s.declType->kind == TypeKind::Reference &&
            s.init && s.init->kind == ExprKind::AddrOf) {
            auto &rt = static_cast<ReferenceType &>(*s.declType);
            auto &ue = static_cast<UnaryExpr &>(*s.init);
            if (ue.operand && ue.operand->kind == ExprKind::Ident) {
                auto *ident = static_cast<IdentExpr *>(ue.operand.get());
                if (ident->resolved && rt.region != Region::Static) {
                    trackRef(ident->name, rt.mut, scopeDepth_, /*borrower=*/s.name, s.loc);
                }
            }
        }
        // Also track if initTy is a reference to a named ident (direct ref binding)
        // NLL: record borrower name so borrow can be released at last use
        if (!inUnsafeScope() && initTy && initTy->kind == TypeKind::Reference &&
            s.init && s.init->kind == ExprKind::Ident) {
            auto &rt    = static_cast<ReferenceType &>(*initTy);
            auto *ident = static_cast<IdentExpr *>(s.init.get());
            if (ident->resolved && rt.region != Region::Static) {
                trackRef(ident->name, rt.mut, scopeDepth_, /*borrower=*/s.name, s.loc);
            }
        }

        if (s.declType && !s.declType->isError() && !initTy->isError()) {
            if (!canImplicitlyConvert(initTy, s.declType) &&
                !intLiteralFitsType(*s.init, s.declType) &&
                !floatLitFitsType(*s.init, s.declType)) {
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

    // thread_local local variable must be static
    if (s.isThreadLocal && !s.isStatic) {
        diag_.error(s.loc, "thread_local local variable must be static");
    }

    // Register in current scope
    VarDecl *vd = new VarDecl{};
    vd->name        = s.name;
    vd->type        = s.resolvedType;
    vd->isConst     = s.isConst;
    vd->isStatic    = s.isStatic;
    // Struct/array/vector types are considered "initialized" even without an
    // explicit initializer because their fields/lanes can be individually
    // assigned before use (a vec<T,N> local built lane-by-lane via 'v[i] = x'
    // — see std/simd/simd.sc — is exactly this pattern, same as an array).
    // Only scalar types strictly require initialization before reading.
    bool isAggregate = s.resolvedType &&
                       (s.resolvedType->kind == TypeKind::Struct ||
                        s.resolvedType->kind == TypeKind::Array ||
                        s.resolvedType->kind == TypeKind::Vector);
    vd->initialized = (s.init != nullptr) || isAggregate;
    vd->scopeDepth  = scopeDepth_;

    Symbol sym;
    sym.kind        = SymKind::Variable;
    sym.name        = s.name;
    sym.type        = s.resolvedType;
    sym.varDecl     = vd;
    sym.initialized = vd->initialized;
    if (sym.type && sym.type->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*sym.type);
        if (rt.region == Region::Arena) {
            sym.arenaRegionName    = rt.arenaName;
            sym.arenaBindGen       = regionGeneration_[rt.arenaName]; // 0 if unseen yet
            sym.arenaBindMarkDepth = regionMarkDepth_[rt.arenaName];
            sym.arenaBindScratchGen = regionScratchGeneration_[rt.arenaName];
        } else if (rt.region == Region::Heap) {
            // Direct-copy alias: '&heap T q = p;' where 'p' is itself a
            // currently-tracked heap variable — key q's tracking off the
            // SAME VarDecl p already uses (transitively, via heapKeyOf), so
            // std::dealloc() on either name invalidates both. See
            // Symbol::heapTrackKey's doc comment; anything other than a
            // bare identifier initializer falls through to the untracked-
            // alias default below, exactly like before this existed.
            if (s.init && s.init->kind == ExprKind::Ident) {
                auto &initIdent = static_cast<IdentExpr &>(*s.init);
                if (auto *initSym = lookup(initIdent.name)) {
                    if (initSym->kind == SymKind::Variable && initSym->varDecl &&
                        initSym->heapBindGen >= 0) {
                        sym.heapTrackKey = heapKeyOf(initSym);
                    }
                }
            }
            sym.heapBindGen = heapFreeGeneration_[heapKeyOf(&sym)]; // 0 if unseen yet
        }
    }
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

void Sema::checkAsmStmt(Stmt &s, FunctionDecl &fn) {
    auto &as = static_cast<AsmStmt&>(s);
    // asm requires unsafe{} unless inside a naked function
    if (!fn.isNaked && !inUnsafeScope()) {
        diag_.error(as.loc, "inline asm requires unsafe{} block (or naked function)");
    }
    // pure functions cannot contain asm (side effects)
    if (checkingPure_) {
        diag_.error(as.loc, "asm statement not allowed in pure function");
    }
    // Type-check output and input expressions. A write-only output operand
    // ('"=r"(var)', as opposed to a read-write '"+r"(var)') is how the
    // definite-initialization checker learns 'var' has a value at all — an
    // asm statement is opaque to it otherwise, so 'int result;' followed
    // by '"=r"(result)' would (before this) look exactly like a read of an
    // uninitialized variable. Pre-mark it initialized, the same way a
    // plain 'var = ...;' assignment's LHS does (see checkAssign), before
    // the checkExpr() below does its own (now-satisfied) read of that
    // state — a '"+r"(var)' operand is a genuine read too and must NOT be
    // exempted this way.
    for (size_t i = 0; i < as.outputExprs.size(); ++i) {
        auto &e = as.outputExprs[i];
        if (!e) continue;
        bool writeOnly = i < as.outputs.size() && !as.outputs[i].empty() &&
                         as.outputs[i][0] == '=';
        if (writeOnly && e->kind == ExprKind::Ident) {
            auto &ident = static_cast<IdentExpr &>(*e);
            if (auto *sym = lookup(ident.name)) sym->initialized = true;
        }
        checkExpr(*e);
        if (writeOnly && e->kind == ExprKind::Ident) {
            auto &ident = static_cast<IdentExpr &>(*e);
            if (ident.resolved) ident.resolved->initialized = true;
        }
    }
    for (auto &e : as.inputExprs) if (e) checkExpr(*e);
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
    case ExprKind::Slice:     ty = checkSlice(static_cast<SliceExpr &>(e));       break;
    case ExprKind::Member:
    case ExprKind::Arrow:     ty = checkMember(static_cast<MemberExpr &>(e));     break;
    case ExprKind::Cast:      ty = checkCast(static_cast<CastExpr &>(e));         break;
    case ExprKind::Assign:    ty = checkAssign(static_cast<AssignExpr &>(e));     break;
    case ExprKind::SizeofType:
        static_cast<SizeofTypeExpr &>(e).ofType =
            resolveType(static_cast<SizeofTypeExpr &>(e).ofType);
        ty = makeInt(64, false);
        break;
    case ExprKind::AlignofType:
        static_cast<AlignofTypeExpr &>(e).ofType =
            resolveType(static_cast<AlignofTypeExpr &>(e).ofType);
        ty = makeInt(64, false);
        break;
    case ExprKind::FieldCount: {
        auto &fc = static_cast<FieldCountExpr &>(e);
        fc.ofType = resolveType(fc.ofType);
        if (fc.ofType && fc.ofType->kind == TypeKind::Struct) {
            ty = makeInt(64, false);
        } else {
            diag_.error(e.loc, "fieldcount() requires a struct type");
            ty = makeInt(64, false);
        }
        break;
    }
    case ExprKind::SizeofPack:
        // Resolved during monomorphization; type is always i64
        ty = makeInt(64, false);
        break;
    case ExprKind::TaggedUnionInit: {
        auto &tui = static_cast<TaggedUnionInitExpr &>(e);
        if (tui.payload) checkExpr(*tui.payload);
        auto it = typeRegistry_.find(tui.unionName);
        if (it != typeRegistry_.end()) ty = it->second;
        else ty = makeError();
        break;
    }
    case ExprKind::Compound: {
        // No target type available here (e.g. reached via a generic
        // checkExpr call rather than through checkVarDecl/checkGlobalVar,
        // which call checkCompoundInit directly once they know the
        // declared type) — nothing to match fields/elements against yet.
        auto &ci = static_cast<CompoundInitExpr &>(e);
        for (auto &init : ci.inits) checkExpr(*init);
        ty = makeVoid();
        break;
    }
    case ExprKind::GenericSelection:
        ty = checkGenericSelection(static_cast<GenericSelectionExpr &>(e));
        break;
    case ExprKind::New:
        ty = checkNew(static_cast<NewExpr &>(e));
        break;
    case ExprKind::TupleLit:
        ty = checkTupleLit(static_cast<TupleLitExpr &>(e));
        break;
    case ExprKind::Spawn:
        ty = checkSpawn(static_cast<SpawnExpr &>(e));
        break;
    case ExprKind::Match:
        ty = checkMatchExpr(static_cast<MatchExpr &>(e));
        break;
    case ExprKind::FnEval:
        ty = checkFnEval(static_cast<FnEvalExpr &>(e));
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
    // Namespace-qualified reference (e.g. 'std::foo', parsed as one IdentExpr
    // whose name is the literal "std::foo" — see Parser::parsePrimaryExpr).
    // Resolve against the namespace registries and rewrite e.name to the
    // mangled symbol (e.g. "std_foo") so every later stage — the normal
    // Scope lookup below, and CodeGen's genIdent, which looks functions/
    // globals up by e.name — needs no namespace-specific handling at all.
    if (e.name.find("::") != std::string::npos) {
        auto fit = namespaceFnRegistry_.find(e.name);
        if (fit != namespaceFnRegistry_.end()) {
            e.name = fit->second->name;
            e.resolvedFn = fit->second;
            auto *sym = lookup(e.name);
            return sym && sym->type ? sym->type : makeError();
        }
        auto vit = namespaceVarRegistry_.find(e.name);
        if (vit != namespaceVarRegistry_.end()) {
            e.name = vit->second->name;
        } else {
            diag_.error(e.loc, "use of undeclared namespace member '" + e.name + "'");
            return makeError();
        }
    }
    auto *sym = lookup(e.name);
    if (!sym && e.name.find("::") == std::string::npos) {
        // Unqualified fallback: code lexically inside 'namespace std { ... }'
        // calls its own siblings unqualified ('chacha_qr_(...)', not
        // 'std::chacha_qr_(...)') — migrating the *declarations* into the
        // namespace didn't rewrite every internal call site to match, so
        // a plain name that fails normal lookup gets one more try against
        // each known namespace before being declared undeclared. Ambiguous
        // only if the same bare name were namespaced under two different
        // namespaces, which doesn't happen for the single "std" namespace.
        for (auto &ns : namespaceNames_) {
            std::string qualified = ns + "::" + e.name;
            auto fit = namespaceFnRegistry_.find(qualified);
            if (fit != namespaceFnRegistry_.end()) {
                e.name = fit->second->name;
                e.resolvedFn = fit->second;
                sym = lookup(e.name);
                break;
            }
            auto vit = namespaceVarRegistry_.find(qualified);
            if (vit != namespaceVarRegistry_.end()) {
                e.name = vit->second->name;
                sym = lookup(e.name);
                break;
            }
        }
    }
    if (!sym) {
        diag_.error(e.loc, "use of undeclared identifier '" + e.name + "'");
        return makeError();
    }
    // Definite initialization check
    if (sym->kind == SymKind::Variable && !sym->initialized) {
        diag_.error(e.loc, "use of possibly uninitialized variable '" + e.name + "'");
    }
    // Arena-reset invalidation check (see Sema::regionGeneration_'s doc
    // comment for the flow-insensitive caveat). Skipped inside 'unsafe' —
    // same escape hatch as every other region/aliasing check — and for the
    // one exempted occurrence (assignment target being rebound; see
    // suppressArenaStaleCheck_'s doc comment).
    if (sym->kind == SymKind::Variable && !sym->arenaRegionName.empty() &&
        !suppressArenaStaleCheck_ && !inUnsafeScope()) {
        int curGen = regionGeneration_[sym->arenaRegionName];
        bool staleByReset = (curGen != sym->arenaBindGen);
        // free_to() only ever rewinds into an active mark scope, so it can
        // never invalidate a reference bound *before* any arena_mark<R>()
        // call (arenaBindMarkDepth == 0) — see Symbol::arenaBindMarkDepth.
        bool staleByFreeTo = false;
        if (sym->arenaBindMarkDepth > 0) {
            int curScratchGen = regionScratchGeneration_[sym->arenaRegionName];
            staleByFreeTo = (curScratchGen != sym->arenaBindScratchGen);
        }
        if (staleByReset || staleByFreeTo) {
            diag_.error(e.loc,
                "use of '" + e.name + "' (&arena<" + sym->arenaRegionName +
                "> reference) after arena_reset<" + sym->arenaRegionName +
                ">(), arena_destroy<" + sym->arenaRegionName +
                ">(), or arena_free_to<" + sym->arenaRegionName +
                ">() invalidated it");
        }
    }
    // Heap use-after-free/double-free check (see Sema::heapFreeGeneration_'s
    // doc comment) — same shape as the arena check above, keyed by this
    // variable's own VarDecl instead of a shared region name. Skipped inside
    // 'unsafe' and for the one exempted occurrence (std::dealloc(p)'s own
    // argument; see suppressHeapStaleCheck_'s doc comment and checkCall).
    if (sym->kind == SymKind::Variable && sym->varDecl && sym->heapBindGen >= 0 &&
        !suppressHeapStaleCheck_ && !inUnsafeScope()) {
        int curGen = heapFreeGeneration_[heapKeyOf(sym)];
        if (curGen != sym->heapBindGen) {
            diag_.error(e.loc,
                "use of '" + e.name + "' (&heap reference) after "
                "std::dealloc() freed it");
        }
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
    case UnaryOp::ForceUnwrap: {
        bool isPointer = operandTy->kind == TypeKind::Pointer;
        bool isNullableRef = operandTy->kind == TypeKind::Reference &&
                              static_cast<ReferenceType &>(*operandTy).nullable;
        bool isOptional = operandTy->kind == TypeKind::Optional;
        if (!isPointer && !isNullableRef && !isOptional) {
            diag_.error(e.loc,
                "'!' can only force-unwrap a pointer, nullable reference (?&T), "
                "or optional (?T), got '" + operandTy->str() + "'");
            return makeError();
        }
        // The one and only gate: 'match'/'is_null()'/'is_none()'/
        // '.default(value)' are always available without 'unsafe' (see
        // Sema::checkCall's pseudo-methods section); a blind force-unwrap
        // that trusts the value is present is not.
        if (!inUnsafeScope()) {
            diag_.error(e.loc,
                "'!' force-unwrap requires 'unsafe', or use "
                "match/is_null()/is_none()/.default(value) instead");
        }
        if (isPointer) {
            e.isLValue = true;
            return static_cast<PointerType &>(*operandTy).base;
        }
        if (isNullableRef) {
            auto &rt = static_cast<ReferenceType &>(*operandTy);
            e.isLValue = rt.mut;
            return rt.base;
        }
        return static_cast<OptionalType &>(*operandTy).inner;
    }
    default:
        return operandTy;
    }
}

TypePtr Sema::checkAddrOf(UnaryExpr &e) {
    TypePtr operandTy = checkExpr(*e.operand);
    // '&funcName' — a bare function name is never an lvalue (there's no
    // storage location holding it the way a variable has), but C and every
    // C-like language still allow taking a function's address; a bare
    // function-name IdentExpr already evaluates to '&static fn(...)' (see
    // checkIdent's SymKind::Function case), so '&funcName' is just
    // 'funcName' again — the '&' is a no-op here, not an error.
    if (e.operand->kind == ExprKind::Ident &&
        static_cast<IdentExpr &>(*e.operand).resolvedFn) {
        return operandTy;
    }
    if (!e.operand->isLValue) {
        diag_.error(e.loc, "address-of operator requires lvalue");
    }
    // Borrow tracking happens at the binding site (var decl, assignment, or
    // call argument), not here — '&x' alone doesn't know whether the
    // resulting reference will be bound mutably or as a shared/const
    // reference, and blindly assuming 'mutable' rejected perfectly valid
    // multiple shared borrows of the same variable.

    // '&x' on a static/global variable (or a '.'-chain rooted in one, e.g.
    // '&g_holder.field') has program lifetime, not stack lifetime — it
    // should produce '&static T', not '&stack T'. Getting this wrong meant
    // e.g. taking the address of a top-level 'static T arr[N];' could never
    // be stored anywhere that expected '&static T' (region-escape rejected
    // it as if it were a dangling stack reference).
    VarDecl *baseVd = nullptr;
    for (Expr *cur = e.operand.get(); cur; ) {
        if (cur->kind == ExprKind::Ident) { baseVd = static_cast<IdentExpr *>(cur)->resolved; break; }
        if (cur->kind == ExprKind::Member) { cur = static_cast<MemberExpr *>(cur)->base.get(); continue; }
        break;
    }
    Region r = (baseVd && (baseVd->isStatic || baseVd->isGlobal)) ? Region::Static : Region::Stack;
    return makeReference(std::move(operandTy), r, false, true);
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
    case BinaryOp::WrapAdd: case BinaryOp::WrapSub: case BinaryOp::WrapMul:
    case BinaryOp::SatAdd:  case BinaryOp::SatSub:  case BinaryOp::SatMul:
        // SIMD vector arithmetic (std::simd): elementwise op across the
        // whole vec<T,N> in one instruction — CodeGen just hands both
        // operands straight to the same CreateAdd/CreateFAdd/etc it already
        // uses for scalars, since LLVM's IRBuilder binary-op calls are
        // generic over vector-typed operands with zero extra code needed.
        if (lhsTy->kind == TypeKind::Vector && rhsTy->kind == TypeKind::Vector) {
            if (!typeEqual(lhsTy, rhsTy)) {
                diag_.error(e.loc,
                    "vector arithmetic requires matching vector types: '" +
                    lhsTy->str() + "' and '" + rhsTy->str() + "'");
                return makeError();
            }
            return lhsTy;
        }
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
    TypePtr argTy = checkExpr(*e.argExpr);
    // ── Spawn region isolation: reject mutable non-static refs ──────────────
    if (argTy && argTy->kind == TypeKind::Reference) {
        auto &ref = static_cast<ReferenceType &>(*argTy);
        if (ref.mut && ref.region != Region::Static) {
            diag_.error(e.argExpr->loc,
                "spawn argument must not pass mutable non-static reference "
                "(region isolation violation)");
        }
    }
    // spawn returns a pthread_t (represented as signed i64)
    return makeInt(64, true);  // pthread_t handle (long long)
}

TypePtr Sema::checkGenericSelection(GenericSelectionExpr &e) {
    // The controlling expression's value is never evaluated for selection
    // purposes in real C11 either (only its type matters) — but SafeC has
    // no "unevaluated context" concept, so just type-check it normally;
    // side effects in the controlling expression are extremely unusual and
    // this matches how e.g. sizeof(expr) is already handled here.
    TypePtr controllingTy = checkExpr(*e.controlling);

    int defaultIdx = -1;
    int matchIdx   = -1;
    for (size_t i = 0; i < e.associations.size(); ++i) {
        auto &assoc = e.associations[i];
        if (!assoc.type) {
            if (defaultIdx >= 0)
                diag_.error(e.loc, "duplicate 'default' association in '_Generic'");
            defaultIdx = (int)i;
            continue;
        }
        assoc.type = resolveType(assoc.type);
        if (typeEqual(controllingTy, assoc.type)) {
            if (matchIdx >= 0)
                diag_.error(e.loc, "'_Generic' controlling expression type matches "
                                   "more than one association");
            matchIdx = (int)i;
        }
    }

    e.selectedIndex = (matchIdx >= 0) ? matchIdx : defaultIdx;
    if (e.selectedIndex < 0) {
        diag_.error(e.loc,
            "'_Generic' controlling expression type '" + controllingTy->str() +
            "' matches no association and there is no 'default'");
    }

    // Still type-check every association (rather than only the selected
    // one) so unrelated errors elsewhere in an association's expression
    // are still caught — SafeC doesn't have C11's "unevaluated" branches.
    TypePtr selectedTy = makeError();
    for (size_t i = 0; i < e.associations.size(); ++i) {
        TypePtr t = checkExpr(*e.associations[i].expr);
        if ((int)i == e.selectedIndex) selectedTy = t;
    }
    e.type = selectedTy;
    return selectedTy;
}

TypePtr Sema::checkCompoundInit(CompoundInitExpr &e, const TypePtr &targetTy) {
    bool hasDesignators = !e.designatedFields.empty();
    e.resolvedSlots.assign(e.inits.size(), 0);

    if (targetTy && targetTy->kind == TypeKind::Vector) {
        // vec<T,N> v = {a, b, c, ...} — no designators (there's no notion of
        // '.field' or a meaningful sparse '[i] = v' for a SIMD register;
        // every lane must be given explicitly, matching real SIMD intrinsic
        // init conventions like _mm_set_ps).
        auto &vt = static_cast<VectorType &>(*targetTy);
        if ((int64_t)e.inits.size() != vt.width) {
            diag_.error(e.loc, "vec<" + vt.element->str() + ", " +
                                std::to_string(vt.width) + "> initializer needs exactly " +
                                std::to_string(vt.width) + " element(s), got " +
                                std::to_string(e.inits.size()));
        }
        for (size_t i = 0; i < e.inits.size(); ++i) {
            e.resolvedSlots[i] = (int64_t)i;
            TypePtr initTy = checkExpr(*e.inits[i]);
            if (vt.element && !initTy->isError() && !vt.element->isError() &&
                !canImplicitlyConvert(initTy, vt.element) &&
                !intLiteralFitsType(*e.inits[i], vt.element) &&
                !floatLitFitsType(*e.inits[i], vt.element)) {
                diag_.error(e.inits[i]->loc,
                    "type mismatch in vector initializer: cannot convert '" +
                    initTy->str() + "' to '" + vt.element->str() + "'");
            }
        }
        e.type = targetTy;
        return targetTy;
    }
    if (targetTy && targetTy->kind == TypeKind::Struct) {
        auto &st = static_cast<StructType &>(*targetTy);
        int64_t nextSlot = 0;
        for (size_t i = 0; i < e.inits.size(); ++i) {
            int64_t slot = nextSlot;
            if (hasDesignators && !e.designatedFields[i].empty()) {
                const FieldDecl *fd = st.findField(e.designatedFields[i]);
                if (!fd) {
                    diag_.error(e.inits[i]->loc,
                        "struct '" + st.name + "' has no field '" +
                        e.designatedFields[i] + "'");
                    slot = 0;
                } else {
                    // Field index in st.fields may not equal its declared
                    // position for bitfield-packed structs, but positional
                    // designated-init advancement is about *declaration
                    // order*, so use the field's vector position, not
                    // FieldDecl::index (which can repeat for packed
                    // bitfields sharing a storage slot).
                    slot = &(*fd) - &st.fields[0];
                }
            } else if (hasDesignators && e.designatedIndices[i] >= 0) {
                diag_.error(e.inits[i]->loc,
                    "array designator '[" + std::to_string(e.designatedIndices[i]) +
                    "]' used initializing struct '" + st.name + "' (expected '.field =')");
            }
            e.resolvedSlots[i] = slot;
            nextSlot = slot + 1;

            TypePtr initTy = checkExpr(*e.inits[i]);
            if (slot < 0 || slot >= (int64_t)st.fields.size()) {
                if (!hasDesignators)
                    diag_.error(e.loc, "too many initializers for struct '" + st.name + "'");
                continue;
            }
            const TypePtr &fieldTy = st.fields[slot].type;
            if (fieldTy && !initTy->isError() && !fieldTy->isError() &&
                !canImplicitlyConvert(initTy, fieldTy) &&
                !intLiteralFitsType(*e.inits[i], fieldTy) &&
                !floatLitFitsType(*e.inits[i], fieldTy)) {
                diag_.error(e.inits[i]->loc,
                    "type mismatch initializing field '" + st.fields[slot].name +
                    "': cannot convert '" + initTy->str() + "' to '" + fieldTy->str() + "'");
            }
        }
        e.type = targetTy;
        return targetTy;
    }
    if (targetTy && targetTy->kind == TypeKind::Array) {
        auto &at = static_cast<ArrayType &>(*targetTy);
        int64_t nextSlot = 0;
        int64_t maxSlot = -1;
        for (size_t i = 0; i < e.inits.size(); ++i) {
            int64_t slot = nextSlot;
            if (hasDesignators && e.designatedIndices[i] >= 0) {
                slot = e.designatedIndices[i];
            } else if (hasDesignators && !e.designatedFields[i].empty()) {
                diag_.error(e.inits[i]->loc,
                    "field designator '." + e.designatedFields[i] +
                    "' used initializing an array (expected '[index] =')");
            }
            e.resolvedSlots[i] = slot;
            nextSlot = slot + 1;
            if (slot > maxSlot) maxSlot = slot;

            TypePtr initTy = checkExpr(*e.inits[i]);
            if (at.element && !initTy->isError() && !at.element->isError() &&
                !canImplicitlyConvert(initTy, at.element) &&
                !intLiteralFitsType(*e.inits[i], at.element) &&
                !floatLitFitsType(*e.inits[i], at.element)) {
                diag_.error(e.inits[i]->loc,
                    "type mismatch in array initializer: cannot convert '" +
                    initTy->str() + "' to '" + at.element->str() + "'");
            }
        }
        if (at.size > 0 && maxSlot >= at.size) {
            diag_.error(e.loc, "too many initializers for array of size " +
                                std::to_string(at.size));
        }
        e.type = targetTy;
        return targetTy;
    }
    // No usable target type (e.g. still being inferred) — check each element
    // with no expected type and leave the compound literal untyped, matching
    // prior behavior.
    for (auto &init : e.inits) checkExpr(*init);
    e.type = makeVoid();
    return e.type;
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

TypePtr Sema::checkMatchExpr(MatchExpr &e) {
    TypePtr subjectTy = checkExpr(*e.subject);
    bool isTaggedUnion = false;
    StructType *unionSt = nullptr;
    if (subjectTy && subjectTy->kind == TypeKind::Struct) {
        unionSt = static_cast<StructType *>(subjectTy.get());
        isTaggedUnion = unionSt->isTaggedUnion;
    }
    EnumType *enumSubject = nullptr;
    if (subjectTy && subjectTy->kind == TypeKind::Enum) {
        enumSubject = static_cast<EnumType *>(subjectTy.get());
    }
    bool isNullLike = false, isOptional = false;
    TypePtr nullPayloadTy;
    classifyNullMatchSubject(subjectTy, isNullLike, isOptional, nullPayloadTy);
    if (e.arms.empty()) {
        diag_.error(e.loc, "match expression must have at least one arm");
        return makeError();
    }

    bool hasWildcard = false;
    std::unordered_set<int> coveredTags;
    TypePtr resultTy;
    bool sawError = false;

    for (auto &arm : e.arms) {
        resolveMatchArmPatterns(arm.patterns, isTaggedUnion, unionSt,
                                 isNullLike, isOptional, nullPayloadTy, diag_,
                                 e.loc, hasWildcard, &coveredTags, enumSubject);

        // Payload bindings are only in scope for this arm's value expr.
        pushScope();
        for (auto &pat : arm.patterns) {
            if (!pat.bindName.empty() && pat.payloadType) {
                Symbol sym;
                sym.kind = SymKind::Variable;
                sym.name = pat.bindName;
                sym.type = pat.payloadType;
                sym.initialized = true;
                define(std::move(sym));
            }
        }
        TypePtr armTy = arm.value ? checkExpr(*arm.value) : makeError();
        popScope();

        if (!armTy || armTy->isError()) { sawError = true; continue; }
        if (!resultTy) {
            resultTy = armTy;
        } else if (!typeEqual(resultTy, armTy)) {
            diag_.error(arm.value->loc,
                "match arms must have the same type: '" + resultTy->str() +
                "' vs '" + armTy->str() + "'");
            sawError = true;
        }
    }

    // Unlike the statement form (which only warns — falling through with no
    // value produced is harmless there), the expression form needs an
    // actual value out of every reachable path: CodeGen merges arm values
    // with a PHI whose incoming edges are exactly the arms, so an uncovered
    // subject value would leave the merge with nothing to produce.
    bool exhaustive = hasWildcard ||
        (isTaggedUnion && unionSt && coveredTags.size() == unionSt->fields.size()) ||
        ((isNullLike || isOptional) && coveredTags.size() == 2) ||
        (enumSubject && coveredTags.size() == enumSubject->enumerators.size());
    e.provenExhaustive = exhaustive;
    if (!exhaustive) {
        diag_.error(e.loc,
            "match expression is not exhaustive: add a 'default' arm, or cover every "
            "case (every tagged-union variant; or both '" +
            std::string(isOptional ? "none" : "null") + "' and 'some(x)')");
    }

    if (sawError || !resultTy) return makeError();
    return resultTy;
}

// fn_eval(object, func) — see FnEvalExpr in AST.h for the design rationale.
// 'object' is only checked for its type (never evaluated/codegen'd, same as
// sizeof's operand); 'func' must directly name an already-declared plain
// (non-method) function, whose name+signature is the lookup key against
// object's struct type's own methodRegistry_ entries.
TypePtr Sema::checkFnEval(FnEvalExpr &e) {
    TypePtr objTy = checkExpr(*e.object);
    checkExpr(*e.func);

    if (e.func->kind != ExprKind::Ident) {
        diag_.error(e.loc,
            "fn_eval: second argument must directly name a declared function");
        return makeError();
    }
    auto &funcId = static_cast<IdentExpr &>(*e.func);
    FunctionDecl *shapeFn = funcId.resolvedFn;
    if (!shapeFn) {
        diag_.error(e.loc, "fn_eval: '" + funcId.name + "' does not name a function");
        return makeError();
    }
    if (shapeFn->isMethod) {
        diag_.error(e.loc,
            "fn_eval: '" + funcId.name + "' must be a plain function, not a method — "
            "it's used only as a name+signature key, never called");
        return makeError();
    }

    TypePtr structTy = objTy;
    if (structTy && (structTy->kind == TypeKind::Pointer || structTy->kind == TypeKind::Reference)) {
        structTy = structTy->kind == TypeKind::Pointer
            ? static_cast<PointerType &>(*structTy).base
            : static_cast<ReferenceType &>(*structTy).base;
    }
    if (!structTy || structTy->kind != TypeKind::Struct) {
        diag_.error(e.loc,
            "fn_eval: first argument must be a struct (or a pointer/reference to one), got '" +
            (objTy ? objTy->str() : std::string("?")) + "'");
        return makeError();
    }
    auto &st = static_cast<StructType &>(*structTy);

    auto it = methodRegistry_.find(st.name + "::" + shapeFn->name);
    if (it == methodRegistry_.end()) {
        diag_.error(e.loc,
            "fn_eval: type '" + st.name + "' has no method named '" + shapeFn->name + "'");
        return makeError();
    }
    FunctionDecl *method = it->second;

    // method->params[0] is the Sema-inserted 'self' (see collectFunction) —
    // everything after it must match shapeFn's params 1:1, since shapeFn
    // (a plain function) has no self slot of its own.
    size_t methodArity = method->params.empty() ? 0 : method->params.size() - 1;
    if (methodArity != shapeFn->params.size()) {
        diag_.error(e.loc,
            "fn_eval: '" + st.name + "::" + shapeFn->name + "' takes " +
            std::to_string(methodArity) + " parameter(s), but '" + funcId.name +
            "' declares " + std::to_string(shapeFn->params.size()));
        return makeError();
    }
    for (size_t i = 0; i < shapeFn->params.size(); ++i) {
        const TypePtr &methodParamTy = method->params[i + 1].type;
        const TypePtr &shapeParamTy  = shapeFn->params[i].type;
        if (!typeEqual(methodParamTy, shapeParamTy)) {
            diag_.error(e.loc,
                "fn_eval: parameter " + std::to_string(i + 1) + " of '" + st.name + "::" +
                shapeFn->name + "' is '" + (methodParamTy ? methodParamTy->str() : "?") +
                "', but '" + funcId.name + "' declares '" +
                (shapeParamTy ? shapeParamTy->str() : "?") + "'");
            return makeError();
        }
    }
    if (!typeEqual(method->returnType, shapeFn->returnType)) {
        diag_.error(e.loc,
            "fn_eval: '" + st.name + "::" + shapeFn->name + "' returns '" +
            (method->returnType ? method->returnType->str() : "?") + "', but '" +
            funcId.name + "' declares '" +
            (shapeFn->returnType ? shapeFn->returnType->str() : "?") + "'");
        return makeError();
    }

    e.matchedMethod = method;
    // Same shape as any other bare-function-name value (see checkIdent's
    // function-symbol case) — a '&static fn(...)' reference, self included
    // in the signature since the caller must still pass it explicitly.
    e.type = makeReference(method->funcType(), Region::Static);
    return e.type;
}

TypePtr Sema::checkCall(CallExpr &e) {
    // RAII so the heap-free bump (and un-suppressing the staleness check)
    // happens on every exit path of this function, regardless of which of
    // checkCall's many special-case branches or early returns is taken
    // below — see the 'std::dealloc(p)' detection right after the arena
    // block for what populates 'target'/'wasSuppressedHere'. The bump must
    // happen *after* the normal argument-checking below has had a chance to
    // see 'p' at its pre-free generation (that read is the last valid use,
    // not a use-after-free), which a guard destructor gives for free
    // without needing to touch every return statement in this function.
    struct HeapFreeGuard {
        Sema *self;
        bool wasSuppressedHere = false;
        const VarDecl *target = nullptr;
        ~HeapFreeGuard() {
            if (wasSuppressedHere) self->suppressHeapStaleCheck_ = false;
            if (target) self->heapFreeGeneration_[target]++;
        }
    } heapFreeGuard{this};

    // Consume the pending expected-type hint (see expectedType_'s doc
    // comment in Sema.h) right away so it can't leak into checking this
    // call's own callee/arguments/any nested call below — only the
    // generic-inference fallback near the bottom of this function
    // (matching 'vec_at(&v, i)' against an 'int* p = ...' target) uses it.
    TypePtr callExpectedType = std::move(expectedType_);
    expectedType_ = nullptr;

    // ── Builtin __safec_join ──────────────────────────────────────────────────
    if (e.callee->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.callee);
        if (ident.name == "__safec_join") {
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeVoid();
            return makeVoid();
        }
        // ── std::dealloc(p): the one recognized heap-free call (see
        // heapFreeGeneration_'s doc comment in Sema.h for the intra-
        // procedural, VarDecl*-keyed scope of this check). Only a bare
        // identifier argument is tracked — same "syntactic, not a full
        // points-to analysis" restriction as the arena checks above.
        // Suppressed here (via the guard above) so the normal argument-
        // checking pass below sees this as the last *valid* read of 'p',
        // not the use-after-free itself; the actual bump happens in the
        // guard's destructor, after that pass has run. Suppression is
        // withheld when 'p' is *already* stale (a prior free already
        // bumped its generation past what it was bound against) so the
        // normal argument-checking pass's staleness check still fires —
        // this is exactly what turns a second 'dealloc(p)' into a
        // detected double-free instead of a silently-suppressed no-op.
        if ((ident.name == "dealloc" ||
             (ident.name.size() > 9 &&
              ident.name.compare(ident.name.size() - 9, 9, "::dealloc") == 0)) &&
            e.args.size() == 1 && e.args[0] &&
            e.args[0]->kind == ExprKind::Ident) {
            auto &argIdent = static_cast<IdentExpr &>(*e.args[0]);
            if (auto *sym = lookup(argIdent.name)) {
                if (sym->kind == SymKind::Variable && sym->varDecl &&
                    sym->type && sym->type->kind == TypeKind::Reference &&
                    static_cast<ReferenceType &>(*sym->type).region == Region::Heap) {
                    heapFreeGuard.target = heapKeyOf(sym);
                    bool alreadyStale = heapFreeGeneration_[heapKeyOf(sym)] != sym->heapBindGen;
                    if (!alreadyStale && !suppressHeapStaleCheck_) {
                        suppressHeapStaleCheck_ = true;
                        heapFreeGuard.wasSuppressedHere = true;
                    }
                }
            }
        }
        // ── arena_reset<R>()/arena_destroy<R>() invalidate *every*
        // outstanding '&arena<R> T' reference — bump R's generation
        // counter so any later read of a reference bound before this call
        // is flagged as stale (see regionGeneration_'s doc comment for the
        // flow-insensitive caveat, and checkIdent for where this is
        // consulted). Doesn't return early: these still resolve to real
        // synthesized functions (see collectRegion) that need normal call
        // type-checking below, same as any other call.
        //
        // arena_mark<R>()/arena_free_to<R>(mark) are narrower: free_to only
        // rewinds *into* the scratch scope opened by the most recent
        // mark(), so it must not stale a reference bound before any mark()
        // was taken (see Symbol::arenaBindMarkDepth/regionMarkDepth_'s doc
        // comments) — tracked with its own depth counter and generation
        // instead of reusing regionGeneration_, which previously made
        // free_to() indistinguishable from a full reset/destroy.
        {
            static const std::string kResetPrefix   = "__arena_reset_";
            static const std::string kDestroyPrefix = "__arena_destroy_";
            static const std::string kMarkPrefix    = "__arena_mark_";
            static const std::string kFreeToPrefix  = "__arena_free_to_";
            if (ident.name.rfind(kResetPrefix, 0) == 0) {
                regionGeneration_[ident.name.substr(kResetPrefix.size())]++;
            } else if (ident.name.rfind(kDestroyPrefix, 0) == 0) {
                regionGeneration_[ident.name.substr(kDestroyPrefix.size())]++;
            } else if (ident.name.rfind(kMarkPrefix, 0) == 0) {
                regionMarkDepth_[ident.name.substr(kMarkPrefix.size())]++;
            } else if (ident.name.rfind(kFreeToPrefix, 0) == 0) {
                const std::string region = ident.name.substr(kFreeToPrefix.size());
                int &depth = regionMarkDepth_[region];
                if (depth > 0) depth--;
                regionScratchGeneration_[region]++;
            }
        }
        // ── volatile_load / volatile_store built-ins ─────────────────────────
        if (ident.name == "volatile_load") {
            if (e.args.size() != 1)
                diag_.error(e.loc, "volatile_load expects exactly 1 argument (pointer)");
            for (auto &a : e.args) checkExpr(*a);
            if (!e.args.empty() && e.args[0]->type &&
                e.args[0]->type->kind == TypeKind::Pointer) {
                auto &pt = static_cast<PointerType &>(*e.args[0]->type);
                e.type = pt.base;
                return pt.base;
            }
            e.type = makeInt(32, true);
            return makeInt(32, true);
        }
        if (ident.name == "volatile_store") {
            if (e.args.size() != 2)
                diag_.error(e.loc, "volatile_store expects 2 arguments (pointer, value)");
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeVoid();
            return makeVoid();
        }
        // ── Atomic built-ins ─────────────────────────────────────────────────
        if (ident.name == "atomic_load" || ident.name == "atomic_store" ||
            ident.name == "atomic_fetch_add" || ident.name == "atomic_fetch_sub" ||
            ident.name == "atomic_fetch_and" || ident.name == "atomic_fetch_or" ||
            ident.name == "atomic_fetch_xor" ||
            ident.name == "atomic_exchange" || ident.name == "atomic_cas") {
            for (auto &a : e.args) checkExpr(*a);
            if (ident.name == "atomic_store") {
                e.type = makeVoid();
                return makeVoid();
            }
            if (ident.name == "atomic_cas") {
                e.type = makeBool();
                return makeBool();
            }
            // load/fetch_*/exchange: return the base type of the pointer
            if (!e.args.empty() && e.args[0]->type &&
                e.args[0]->type->kind == TypeKind::Pointer) {
                auto &pt = static_cast<PointerType &>(*e.args[0]->type);
                e.type = pt.base;
                return pt.base;
            }
            e.type = makeInt(32, true);
            return makeInt(32, true);
        }
        if (ident.name == "atomic_compare_exchange_strong") {
            // (ptr, &expected, desired) → bool (true on success)
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeBool();
            return makeBool();
        }
        // ── GCC/Clang bit-manipulation built-ins (std/bit.sc) ────────────────
        if (ident.name == "__builtin_popcount"  || ident.name == "__builtin_popcountll" ||
            ident.name == "__builtin_clz"       || ident.name == "__builtin_clzll" ||
            ident.name == "__builtin_ctz"        || ident.name == "__builtin_ctzll") {
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeInt(32, true);
            return e.type;
        }
        if (ident.name == "__builtin_bswap32" || ident.name == "__builtin_bswap64") {
            for (auto &a : e.args) checkExpr(*a);
            // Result has the same width as the operand being byte-swapped.
            if (!e.args.empty() && e.args[0]->type) {
                e.type = e.args[0]->type;
                return e.type;
            }
            e.type = makeInt(32, false);
            return e.type;
        }
        if (ident.name == "atomic_fence") {
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeVoid();
            return makeVoid();
        }
        // ── ARM Cortex-M4/M7 DSP-extension built-ins ─────────────────────────
        // Thin wrappers over the DSP extension's packed-SIMD/saturating
        // instructions (SADD16, SMLAD, USAD8, SSAT, ...). LLVM does not
        // auto-vectorize generic vec<T,N> IR into these (see std/simd/
        // cortex_m.h's DSP caveat — confirmed by inspecting compiler output:
        // a vec<short,2> add on thumbv7em lowers to two scalar 'add's, not
        // 'sadd16'), so — exactly like atomic_* above expose hardware
        // primitives no portable SafeC construct reaches — they're exposed
        // directly as builtins. CodeGen rejects them on non-ARM targets.
        if (ident.name.rfind("__arm_dsp_", 0) == 0) {
            struct ArmDspSig { const char *name; int arity; bool immArg2; bool isUnsigned; };
            static const ArmDspSig kArmDsp[] = {
                {"__arm_dsp_qadd",    2, false, false},
                {"__arm_dsp_qsub",    2, false, false},
                {"__arm_dsp_qadd16",  2, false, false},
                {"__arm_dsp_qadd8",   2, false, false},
                {"__arm_dsp_qsub16",  2, false, false},
                {"__arm_dsp_qsub8",   2, false, false},
                {"__arm_dsp_sadd16",  2, false, false},
                {"__arm_dsp_sadd8",   2, false, false},
                {"__arm_dsp_ssub16",  2, false, false},
                {"__arm_dsp_ssub8",   2, false, false},
                {"__arm_dsp_uqadd16", 2, false, true},
                {"__arm_dsp_uqadd8",  2, false, true},
                {"__arm_dsp_uqsub16", 2, false, true},
                {"__arm_dsp_uqsub8",  2, false, true},
                {"__arm_dsp_smlad",   3, false, false},
                {"__arm_dsp_smladx",  3, false, false},
                {"__arm_dsp_smlsd",   3, false, false},
                {"__arm_dsp_smlsdx",  3, false, false},
                {"__arm_dsp_smuad",   2, false, false},
                {"__arm_dsp_smuadx",  2, false, false},
                {"__arm_dsp_smusd",   2, false, false},
                {"__arm_dsp_smusdx",  2, false, false},
                {"__arm_dsp_usad8",   2, false, true},
                {"__arm_dsp_usada8",  3, false, true},
                {"__arm_dsp_ssat",    2, true,  false},
                {"__arm_dsp_usat",    2, true,  true},
                {"__arm_dsp_ssat16",  2, true,  false},
                {"__arm_dsp_usat16",  2, true,  true},
                {"__arm_dsp_sxtab16", 2, false, false},
                {"__arm_dsp_uxtab16", 2, false, true},
            };
            const ArmDspSig *found = nullptr;
            for (auto &b : kArmDsp) if (ident.name == b.name) { found = &b; break; }
            for (auto &a : e.args) checkExpr(*a);
            if (!found) {
                diag_.error(e.loc, "unknown ARM DSP builtin '" + ident.name + "'");
                e.type = makeInt(32, true);
                return e.type;
            }
            if ((int)e.args.size() != found->arity) {
                diag_.error(e.loc, ident.name + " expects " + std::to_string(found->arity) +
                    " argument(s), got " + std::to_string(e.args.size()));
            }
            if (found->immArg2 && e.args.size() >= 2 && e.args[1]->kind != ExprKind::IntLit) {
                diag_.error(e.args[1]->loc, ident.name +
                    "'s saturation-width argument must be a compile-time integer constant");
            }
            e.type = makeInt(32, !found->isUnsigned);
            return e.type;
        }
        // ── Channel built-ins ───────────────────────────────────────────────
        if (ident.name == "chan_create") {
            // chan_create(capacity) → void* (opaque channel handle)
            if (e.args.size() != 1)
                diag_.error(e.loc, "chan_create expects 1 argument (capacity)");
            for (auto &a : e.args) checkExpr(*a);
            e.type = makePointer(makeVoid());
            return e.type;
        }
        if (ident.name == "chan_send") {
            // chan_send(channel, value_ptr) → bool (true = success)
            if (e.args.size() != 2)
                diag_.error(e.loc, "chan_send expects 2 arguments (channel, value_ptr)");
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeBool();
            return e.type;
        }
        if (ident.name == "chan_recv") {
            // chan_recv(channel, out_ptr) → bool (true = received, false = closed)
            if (e.args.size() != 2)
                diag_.error(e.loc, "chan_recv expects 2 arguments (channel, out_ptr)");
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeBool();
            return e.type;
        }
        if (ident.name == "chan_close") {
            // chan_close(channel) → void
            if (e.args.size() != 1)
                diag_.error(e.loc, "chan_close expects 1 argument (channel)");
            for (auto &a : e.args) checkExpr(*a);
            e.type = makeVoid();
            return makeVoid();
        }
        // ── Scoped spawn: guarantees join before scope exit ─────────────────
        if (ident.name == "spawn_scoped") {
            // spawn_scoped(fn, arg) → long long (thread handle)
            // Like spawn, but Sema registers a deferred join at current scope.
            // Region isolation: mutable non-static refs are still rejected.
            if (e.args.size() != 2)
                diag_.error(e.loc, "spawn_scoped expects 2 arguments (function, arg)");
            for (auto &a : e.args) {
                TypePtr argTy = checkExpr(*a);
                if (argTy && argTy->kind == TypeKind::Reference) {
                    auto &ref = static_cast<ReferenceType &>(*argTy);
                    if (ref.mut && ref.region != Region::Static) {
                        diag_.error(a->loc,
                            "spawn_scoped argument must not pass mutable non-static "
                            "reference (region isolation violation)");
                    }
                }
            }
            e.type = makeInt(64, true);
            return e.type;
        }
        // ── Freestanding mode: warn on common stdlib calls ───────────────────
        if (freestanding_) {
            static const char *stdlibFns[] = {
                "malloc", "free", "calloc", "realloc",
                "printf", "fprintf", "sprintf", "snprintf",
                "puts", "putchar", "getchar",
                "fopen", "fclose", "fread", "fwrite",
                "exit", "abort", "atexit",
                nullptr
            };
            for (const char **p = stdlibFns; *p; ++p) {
                if (ident.name == *p) {
                    diag_.warn(e.loc,
                        "call to '" + ident.name + "' in freestanding mode "
                        "(no hosted runtime available)");
                    break;
                }
            }
        }
        // ── Pure function check: reject calls to non-pure functions ──────────
        if (checkingPure_ && ident.resolvedFn && !ident.resolvedFn->isPure &&
            !ident.resolvedFn->isExtern) {
            // Allow calls to other pure functions only
            diag_.warn(e.loc,
                "pure function calls non-pure function '" + ident.name + "'");
        }
    }

    // ── Tagged union construction: Shape.radius(5.0) ──────────────────────────
    if (e.callee->kind == ExprKind::Member) {
        auto &mem = static_cast<MemberExpr &>(*e.callee);
        if (mem.base->kind == ExprKind::Ident) {
            auto &baseIdent = static_cast<IdentExpr &>(*mem.base);
            auto tit = typeRegistry_.find(baseIdent.name);
            if (tit != typeRegistry_.end() && tit->second->kind == TypeKind::Struct) {
                auto &st = static_cast<StructType &>(*tit->second);
                if (st.isTaggedUnion) {
                    const FieldDecl *variantField = st.findField(mem.field);
                    if (variantField) {
                        for (auto &a : e.args) checkExpr(*a);
                        e.taggedUnionTag  = variantField->index;
                        e.taggedUnionName = st.name;
                        e.type = tit->second;
                        return tit->second;
                    }
                }
            }
        }
    }

    // ── Nullable pointer (T*) / nullable reference (?&T) / Optional (?T)
    // safe pseudo-methods ────────────────────────────────────────────────
    // x.is_null() / x.is_none() / x.default(fallback) — the *only*
    // sanctioned way to inspect or unwrap one of these outside 'unsafe'
    // (see checkNullabilityDeref/checkDeref's pointer case and the '!'
    // ForceUnwrap operator, both of which require 'unsafe'). Resolved
    // directly here since there's no real FunctionDecl / methodRegistry_
    // entry backing them. 'is_null'/'default' cover raw pointers (always
    // potentially null in this language, unlike a plain '&T' reference)
    // and nullable references (?&T); 'is_none' covers Optional (?T) —
    // named separately since a bare pointer/reference isn't "empty" the
    // way an Optional is, it's a different value entirely.
    if ((e.callee->kind == ExprKind::Member || e.callee->kind == ExprKind::Arrow) &&
        !static_cast<MemberExpr &>(*e.callee).isArrow) {
        auto &mem = static_cast<MemberExpr &>(*e.callee);
        if (mem.field == "is_null" || mem.field == "is_none" || mem.field == "default") {
            TypePtr baseTy = checkExpr(*mem.base);
            bool isPointer = baseTy && baseTy->kind == TypeKind::Pointer;
            bool isNullableRef = baseTy && baseTy->kind == TypeKind::Reference &&
                                  static_cast<ReferenceType &>(*baseTy).nullable;
            bool isOptional = baseTy && baseTy->kind == TypeKind::Optional;
            bool isNullLike = isPointer || isNullableRef;
            if (isNullLike || isOptional) {
                if (mem.field == "is_null") {
                    if (isOptional)
                        diag_.error(e.loc, "'is_null()' is for pointers/nullable references; "
                                            "use 'is_none()' for optional (?T)");
                    if (!e.args.empty())
                        diag_.error(e.loc, "'is_null()' takes no arguments");
                    for (auto &a : e.args) checkExpr(*a);
                    e.nullOp     = CallExpr::NullOp::IsNull;
                    e.nullOpBase = std::move(mem.base);
                    e.type       = makeBool();
                    return e.type;
                }
                if (mem.field == "is_none") {
                    if (isNullLike)
                        diag_.error(e.loc, "'is_none()' is for optional (?T); "
                                            "use 'is_null()' for pointers/nullable references");
                    if (!e.args.empty())
                        diag_.error(e.loc, "'is_none()' takes no arguments");
                    for (auto &a : e.args) checkExpr(*a);
                    e.nullOp     = CallExpr::NullOp::IsNone;
                    e.nullOpBase = std::move(mem.base);
                    e.type       = makeBool();
                    return e.type;
                }
                // "default"
                TypePtr innerTy = isPointer ? static_cast<PointerType &>(*baseTy).base
                    : isNullableRef ? static_cast<ReferenceType &>(*baseTy).base
                                    : static_cast<OptionalType &>(*baseTy).inner;
                if (e.args.size() != 1) {
                    diag_.error(e.loc, "'.default(value)' takes exactly 1 argument");
                    for (auto &a : e.args) checkExpr(*a);
                } else {
                    TypePtr argTy = checkExpr(*e.args[0]);
                    if (!argTy->isError() && !innerTy->isError() &&
                        !canImplicitlyConvert(argTy, innerTy) &&
                        !intLiteralFitsType(*e.args[0], innerTy) &&
                        !floatLitFitsType(*e.args[0], innerTy)) {
                        diag_.error(e.loc,
                            "'.default(value)': fallback type '" + argTy->str() +
                            "' does not match '" + innerTy->str() + "'");
                    }
                }
                e.nullOp     = CallExpr::NullOp::Default;
                e.nullOpBase = std::move(mem.base);
                e.type       = innerTy;
                return e.type;
            }
            // Not a pointer/nullable-ref/Optional receiver ('mem.base' is
            // still intact — only moved out on the return paths above) —
            // fall through to the ordinary method/field-call path below,
            // e.g. some unrelated struct genuinely has an 'is_null' method.
        }
    }

    // ── Method call detection: x.m(args) → T_m(self=&x, args) ───────────────
    if (e.callee->kind == ExprKind::Member || e.callee->kind == ExprKind::Arrow) {
        auto &mem = static_cast<MemberExpr &>(*e.callee);
        // Type-check the base to get the struct type
        TypePtr baseTy = checkExpr(*mem.base);
        // 'x.method(...)' on a nullable reference dereferences 'x' to find
        // and invoke the method against it exactly as much as 'x.field'
        // does — checkMember (the non-call member-access path) already
        // enforces this same check; without it here, 'f.bump()' on a
        // '?&Foo f' silently compiled and dereferenced a possibly-null
        // pointer at runtime with no 'unsafe' in sight, a real, confirmed
        // null-pointer-dereference gap (the pseudo-methods 'is_null()'/
        // 'is_none()'/'default(...)' just above this block are resolved
        // and returned before reaching here, so they're unaffected).
        checkNullabilityDeref(baseTy, e.loc);
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
        if (!inferTypeArgs(fnDecl->params, argTypes, fnDecl->genericParams, subs,
                           fnDecl->returnType, callExpectedType)) {
            diag_.error(e.loc,
                "cannot infer type arguments for generic function '" +
                fnDecl->name + "'");
            return makeError();
        }

        // Enforce trait constraints on inferred type arguments. Stacked
        // constraints ('T: Numeric + Indexed') are AND semantics — every
        // one must be satisfied, so report each unmet constraint rather
        // than stopping at the first (matches how a missing-field struct
        // literal reports every missing field, not just one).
        for (auto &gp : fnDecl->genericParams) {
            if (gp.constraints.empty()) continue;
            auto it = subs.find(gp.name);
            if (it == subs.end()) continue;
            for (auto &constraint : gp.constraints) {
                if (!satisfiesTrait(it->second, constraint)) {
                    diag_.error(e.loc,
                        "type '" + it->second->str() + "' does not satisfy constraint '" +
                        constraint + "' for generic parameter '" + gp.name + "'");
                }
            }
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

            // Type-check the instantiation. checkFunction() unconditionally
            // resets loopDepth_/functionLabels_/checkingPure_ at its start
            // (correct when it's called once per function from the top of
            // the TU) — but monomorphization happens lazily, on the first
            // call site actually reached, which is frequently *inside*
            // another function's own body currently being checked (e.g. a
            // generic<T> call inside a loop). Without saving/restoring
            // here, checking the monomorphized clone would clobber the
            // enclosing function's in-progress state — observed as, e.g.,
            // a 'break' later in that same enclosing loop wrongly
            // reported as "not inside a loop or match", because
            // loopDepth_ got reset to 0 by the nested checkFunction call
            // and never recovered its outer value.
            bool savedPure = checkingPure_;
            int savedLoopDepth = loopDepth_;
            auto savedLabels = functionLabels_;
            checkFunction(*clone);
            checkingPure_ = savedPure;
            loopDepth_ = savedLoopDepth;
            functionLabels_ = savedLabels;

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
        // Out-parameter idiom: 'foo(&x)' where foo's parameter is a
        // non-const pointer/mutable reference is how C-style code
        // initializes 'x' (e.g. 'unsigned long len; get_len(&len);').
        // Pre-mark 'x' as initialized before checkExpr() below walks into
        // the AddrOf and flags it as read-before-init — a *const*-qualified
        // parameter (read-only intent) is deliberately excluded so a
        // genuinely uninitialized read still gets caught.
        size_t peekParamIdx = i + selfOffset;
        if (peekParamIdx < expected && e.args[i]->kind == ExprKind::AddrOf) {
            auto &ue = static_cast<UnaryExpr &>(*e.args[i]);
            if (ue.operand && ue.operand->kind == ExprKind::Ident) {
                auto &paramTy = ft->paramTypes[peekParamIdx];
                bool isOutParam = false;
                if (paramTy && paramTy->kind == TypeKind::Pointer &&
                        !static_cast<PointerType &>(*paramTy).isConst) {
                    isOutParam = true;
                } else if (paramTy && paramTy->kind == TypeKind::Reference &&
                        static_cast<ReferenceType &>(*paramTy).mut) {
                    isOutParam = true;
                }
                if (isOutParam) {
                    auto &ident = static_cast<IdentExpr &>(*ue.operand);
                    if (auto *sym = lookup(ident.name)) sym->initialized = true;
                    if (ident.resolved) ident.resolved->initialized = true;
                }
            }
        }
        TypePtr argTy = checkExpr(*e.args[i]);
        size_t  paramIdx = i + selfOffset;
        if (paramIdx < expected &&
                !argTy->isError() && !ft->paramTypes[paramIdx]->isError()) {
            if (!canImplicitlyConvert(argTy, ft->paramTypes[paramIdx]) &&
                !intLiteralFitsType(*e.args[i], ft->paramTypes[paramIdx]) &&
                !floatLitFitsType(*e.args[i], ft->paramTypes[paramIdx]) &&
                !refToPointerArgCompatible(argTy, ft->paramTypes[paramIdx])) {
                diag_.error(e.args[i]->loc,
                    "argument " + std::to_string(i+1) + ": cannot convert '" +
                    argTy->str() + "' to '" + ft->paramTypes[paramIdx]->str() + "'");
            }
        }
    }

    // ── Inter-procedural region analysis ────────────────────────────────────
    checkCallRegions(e, ft, selfOffset);

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
    if (baseTy->kind == TypeKind::Vector) {
        // v[i] on a vec<T,N> (std::simd): extractelement for reads,
        // insertelement for writes — CodeGen (genSubscript) does both
        // directly on the SSA vector value, no address/GEP involved, unlike
        // every other subscriptable kind here.
        auto &vt = static_cast<VectorType &>(*baseTy);
        if (!inUnsafeScope() && e.index && e.index->kind == ExprKind::IntLit) {
            auto *lit = static_cast<IntLitExpr *>(e.index.get());
            if (lit->value < 0 || lit->value >= vt.width) {
                diag_.error(e.loc,
                    "vector index " + std::to_string(lit->value) +
                    " out of bounds for vec<" + vt.element->str() + ", " +
                    std::to_string(vt.width) + ">");
            }
        }
        e.boundsCheckOmit = inUnsafeScope();
        e.isLValue = e.base->isLValue;
        return vt.element;
    }
    if (baseTy->kind == TypeKind::Pointer) {
        if (!inUnsafeScope())
            diag_.error(e.loc, "pointer subscript requires 'unsafe' block");
        e.isLValue = true;
        return static_cast<PointerType &>(*baseTy).base;
    }
    if (baseTy->kind == TypeKind::Slice) {
        e.boundsCheckOmit = inUnsafeScope();
        e.isLValue = true;
        return static_cast<SliceType &>(*baseTy).element;
    }
    if (baseTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*baseTy);
        if (rt.base->kind == TypeKind::Array) {
            e.isLValue = rt.mut;
            return static_cast<ArrayType &>(*rt.base).element;
        }
        // A reference to a scalar (e.g. '&static unsigned char buf') used as
        // a buffer pointer — the same C idiom as indexing a raw pointer, and
        // needs the same 'unsafe' bar: there's no bounds information to
        // check against, unlike a real array or slice.
        if (!inUnsafeScope())
            diag_.error(e.loc, "reference subscript requires 'unsafe' block");
        e.isLValue = rt.mut;
        return rt.base;
    }
    diag_.error(e.loc, "subscript on non-array type '" + baseTy->str() + "'");
    return makeError();
}

TypePtr Sema::checkSlice(SliceExpr &e) {
    TypePtr baseTy = checkExpr(*e.base);
    if (e.start) {
        auto st = checkExpr(*e.start);
        if (!st->isInteger())
            diag_.error(e.start->loc, "slice start index must be integer");
    }
    if (e.end) {
        auto et = checkExpr(*e.end);
        if (!et->isInteger())
            diag_.error(e.end->loc, "slice end index must be integer");
    }

    TypePtr elemTy;
    if (baseTy->kind == TypeKind::Array) {
        elemTy = static_cast<ArrayType &>(*baseTy).element;
    } else if (baseTy->kind == TypeKind::Pointer) {
        if (!inUnsafeScope())
            diag_.error(e.loc, "slicing a pointer requires 'unsafe' block");
        elemTy = static_cast<PointerType &>(*baseTy).base;
    } else if (baseTy->kind == TypeKind::Slice) {
        elemTy = static_cast<SliceType &>(*baseTy).element;
    } else if (baseTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*baseTy);
        if (rt.base->kind == TypeKind::Array)
            elemTy = static_cast<ArrayType &>(*rt.base).element;
        else {
            diag_.error(e.loc, "cannot slice type '" + baseTy->str() + "'");
            return makeError();
        }
    } else {
        diag_.error(e.loc, "cannot slice type '" + baseTy->str() + "'");
        return makeError();
    }
    return makeSlice(elemTy);
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
            e.isLValue = true; // ptr->field is always an lvalue (writable through raw pointer)
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

    // Slice member access: .len → i64, .ptr → T*
    if (baseTy->kind == TypeKind::Slice) {
        auto &sl = static_cast<SliceType &>(*baseTy);
        if (e.field == "len") return makeInt(64, false);
        if (e.field == "ptr") return makePointer(sl.element, false);
        diag_.error(e.loc, "slice has no member '" + e.field + "' (use .len or .ptr)");
        return makeError();
    }

    StructType *st = asStruct(baseTy);
    if (!st) {
        diag_.error(e.loc, "member access on non-struct type '" + baseTy->str() + "'");
        return makeError();
    }
    // findFieldPath reaches through anonymous struct/union members
    // ('struct S { union { int a; }; };' — 's.a' isn't a direct field of
    // S). CodeGen re-derives the same path via findMemberFieldDecl to
    // build the (possibly multi-level) GEP chain.
    std::vector<int> path;
    const FieldDecl *fd = st->findFieldPath(e.field, path);
    if (!fd) {
        diag_.error(e.loc, "struct '" + st->name + "' has no field '" + e.field + "'");
        return makeError();
    }
    if (!e.isArrow) e.isLValue = e.base->isLValue;
    return fd->type;
}

TypePtr Sema::checkCast(CastExpr &e) {
    e.targetType = resolveType(e.targetType);

    // Compound literal: '(Type){...}' parses as a cast applied to a
    // CompoundInitExpr operand (canStartCastOperand() already accepts '{'
    // as a valid cast-operand start) — but a generic checkExpr() on that
    // operand has no target type to check fields/elements against (that's
    // otherwise only threaded through by checkVarDecl/checkGlobalVar,
    // which call checkCompoundInit directly). The cast's own target type
    // IS that type, so use it here instead of falling through to the
    // generic (target-type-less) dispatch.
    TypePtr srcTy;
    if (e.operand->kind == ExprKind::Compound) {
        srcTy = checkCompoundInit(static_cast<CompoundInitExpr &>(*e.operand), e.targetType);
    } else {
        srcTy = checkExpr(*e.operand);
    }

    // In safe code, cast from reference to pointer is not allowed — except
    // a '&static fn(...)' function reference (the type a bare function
    // name always evaluates to; see checkIdent's SymKind::Function case).
    // The whole point of the reference-to-pointer restriction is dangling/
    // aliasing safety for *data* — a function's code address is neither
    // freed nor moved for the life of the program, so '(void*)someFunc' (a
    // function-pointer table entry, an ISR vector table, etc.) carries none
    // of the risk 'unsafe' exists to flag.
    if (!inUnsafeScope()) {
        bool isFuncRef = srcTy->kind == TypeKind::Reference &&
            static_cast<ReferenceType &>(*srcTy).base &&
            static_cast<ReferenceType &>(*srcTy).base->kind == TypeKind::Function;
        if (srcTy->kind == TypeKind::Reference &&
            e.targetType->kind == TypeKind::Pointer && !isFuncRef) {
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
    // Also suppress the arena/heap-staleness checks for this same LHS
    // occurrence — rebinding a stale reference to a fresh value is exactly
    // how you recover from staleness, not a read of the old (possibly-
    // invalid) value; the rebind below (after rhsTy is known) sets the real
    // arenaBindGen/heapBindGen for whatever the RHS actually is.
    bool suppressedHere = false;
    bool suppressedHeapHere = false;
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        if (auto *sym = lookup(ident.name))
            if (sym->kind == SymKind::Variable) {
                sym->initialized = true;
                if (!suppressArenaStaleCheck_) {
                    suppressArenaStaleCheck_ = true;
                    suppressedHere = true;
                }
                if (!suppressHeapStaleCheck_) {
                    suppressHeapStaleCheck_ = true;
                    suppressedHeapHere = true;
                }
            }
    }
    TypePtr lhsTy = checkExpr(*e.lhs);
    if (suppressedHere) suppressArenaStaleCheck_ = false;
    if (suppressedHeapHere) suppressHeapStaleCheck_ = false;
    // See expectedType_'s doc comment: 'p = std::vec_at(&v, 1UL);' (p
    // already declared 'int*') gets the same fallback hint a var-decl's
    // explicit type does. Simple assignment only, matching the rebind
    // logic below this uses lhsTy for.
    TypePtr savedExpectedType = expectedType_;
    if (e.op == AssignOp::Assign && lhsTy && !lhsTy->isError())
        expectedType_ = lhsTy;
    TypePtr rhsTy = checkExpr(*e.rhs);
    expectedType_ = savedExpectedType;

    // Rebind arena tracking: 'p = <arena-ref-value>;' establishes a fresh
    // binding, so p's staleness is judged against the region's generation
    // *now*, not whatever it was bound to before. Same idea for heap
    // tracking: 'p = new int;' after a free recovers p (fresh allocation,
    // fresh VarDecl-keyed generation snapshot).
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident &&
        rhsTy && rhsTy->kind == TypeKind::Reference) {
        auto &rrt = static_cast<ReferenceType &>(*rhsTy);
        if (rrt.region == Region::Arena) {
            auto &ident = static_cast<IdentExpr &>(*e.lhs);
            if (auto *sym = lookup(ident.name)) {
                if (sym->kind == SymKind::Variable) {
                    sym->arenaRegionName    = rrt.arenaName;
                    sym->arenaBindGen       = regionGeneration_[rrt.arenaName];
                    sym->arenaBindMarkDepth = regionMarkDepth_[rrt.arenaName];
                    sym->arenaBindScratchGen = regionScratchGeneration_[rrt.arenaName];
                }
            }
        } else if (rrt.region == Region::Heap) {
            auto &ident = static_cast<IdentExpr &>(*e.lhs);
            if (auto *sym = lookup(ident.name)) {
                if (sym->kind == SymKind::Variable && sym->varDecl) {
                    // Same direct-copy alias detection as checkVarDecl's
                    // '&heap T q = p;' case, for 'q = p;' reassignment.
                    sym->heapTrackKey = nullptr;
                    if (e.rhs->kind == ExprKind::Ident) {
                        auto &rhsIdent = static_cast<IdentExpr &>(*e.rhs);
                        if (auto *rhsSym = lookup(rhsIdent.name)) {
                            if (rhsSym->kind == SymKind::Variable && rhsSym->varDecl &&
                                rhsSym->heapBindGen >= 0 && rhsSym != sym) {
                                sym->heapTrackKey = heapKeyOf(rhsSym);
                            }
                        }
                    }
                    sym->heapBindGen = heapFreeGeneration_[heapKeyOf(sym)];
                }
            }
        }
    }

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

    // Region escape: RHS &stack/&arena can't be assigned to variable with wider
    // scope. This also has to cover assignment through struct fields
    // ('h.ref = &x') — not just plain identifiers — by walking the '.' chain
    // down to its base variable ('->' breaks the chain: it indirects through
    // a pointer/reference with its own, independent lifetime).
    if (rhsTy && rhsTy->kind == TypeKind::Reference) {
        auto &rt = static_cast<ReferenceType &>(*rhsTy);
        VarDecl *baseVd = nullptr;
        for (Expr *cur = e.lhs.get(); cur; ) {
            if (cur->kind == ExprKind::Ident) { baseVd = static_cast<IdentExpr *>(cur)->resolved; break; }
            if (cur->kind == ExprKind::Member) { cur = static_cast<MemberExpr *>(cur)->base.get(); continue; }
            break;
        }
        if (rt.region == Region::Stack && baseVd) {
            if (baseVd->scopeDepth < scopeDepth_) {
                diag_.error(e.loc,
                    "cannot assign '&stack " + rt.base->str() +
                    "' to variable in outer scope: reference would escape");
            }
        }
        if (rt.region == Region::Arena && baseVd) {
            auto it = regionRegistry_.find(rt.arenaName);
            int minDepth = (it != regionRegistry_.end()) ? it->second->declScopeDepth : 0;
            if (baseVd->scopeDepth < minDepth) {
                diag_.error(e.loc,
                    "cannot assign '&arena<" + rt.arenaName + "> " + rt.base->str() +
                    "' to variable in outer scope: arena reference would escape");
            }
        }
    }

    // Type compatibility
    if (!lhsTy->isError() && !rhsTy->isError()) {
        if (!canImplicitlyConvert(rhsTy, lhsTy) &&
            !intLiteralFitsType(*e.rhs, lhsTy) &&
            !floatLitFitsType(*e.rhs, lhsTy)) {
            diag_.error(e.loc,
                "type mismatch in assignment: cannot convert '" +
                rhsTy->str() + "' to '" + lhsTy->str() + "'");
        }
    }

    // NLL: if LHS held a borrow and is being reassigned, release the old borrow
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        releaseUnusedBorrows(ident.name);
    }

    // Borrow checker: rebinding an existing reference variable ('r = &x;')
    // registers the new borrow with r's declared mutability, same as at
    // declaration time — see checkVarDecl for why this can't be done inside
    // checkAddrOf itself.
    if (!inUnsafeScope() && e.op == AssignOp::Assign &&
        e.lhs->kind == ExprKind::Ident && e.rhs->kind == ExprKind::AddrOf &&
        lhsTy && lhsTy->kind == TypeKind::Reference) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        auto &rt    = static_cast<ReferenceType &>(*lhsTy);
        auto &ue    = static_cast<UnaryExpr &>(*e.rhs);
        if (ue.operand && ue.operand->kind == ExprKind::Ident) {
            auto *rhsIdent = static_cast<IdentExpr *>(ue.operand.get());
            if (rhsIdent->resolved && rt.region != Region::Static) {
                trackRef(rhsIdent->name, rt.mut, scopeDepth_,
                         /*borrower=*/ident.name, e.loc);
            }
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
    // Direct dereference of a nullable reference is only permitted inside
    // 'unsafe' (matching raw-pointer dereference just above in checkDeref)
    // — everywhere else, go through 'match', 'is_null()', or
    // '.default(value)' (see Sema::checkCall's pseudo-methods section),
    // or the unsafe-gated '!' force-unwrap (UnaryOp::ForceUnwrap).
    if (rt.nullable && !inUnsafeScope()) {
        diag_.error(loc,
            "dereference of nullable reference '?&" + rt.base->str() +
            "' requires 'unsafe', or use match/is_null()/.default(value) instead");
    }
}

// ── Alias tracking ────────────────────────────────────────────────────────────
void Sema::trackRef(const std::string &var, bool isMut, int depth,
                    const std::string &borrower, SourceLocation loc) {
    auto &recs = aliasMap_[var];
    bool conflict = false;
    // Check *every* live record, not just the first — otherwise a borrow that
    // happens to sort before a real conflict lets the conflict slip through
    // unnoticed (and the new borrow used to never get registered either way).
    for (auto &r : recs) {
        if (r.released) continue;  // NLL: already released
        if (r.scopeDepth < depth) continue; // outer scope — no conflict
        if (isMut || r.isMutable) {
            diag_.error(loc,
                "cannot borrow '" + var + "' as " + (isMut ? "mutable" : "immutable") +
                ": already borrowed" + (r.isMutable ? " as mutable" : ""));
            conflict = true;
            break; // one diagnostic is enough; further records won't change the outcome
        }
        // Two shared (non-mutable) borrows of the same variable are fine —
        // keep scanning the rest of the records regardless.
    }
    if (!conflict) {
        recs.push_back({var, borrower, isMut, depth, /*released=*/false});
    }
}

// NLL: release borrows held by a specific borrower variable (at its last use)
void Sema::releaseUnusedBorrows(const std::string &borrower) {
    if (borrower.empty()) return;
    for (auto &[name, records] : aliasMap_) {
        for (auto &r : records) {
            if (r.borrower == borrower && !r.released) {
                r.released = true;
            }
        }
    }
}

void Sema::untrackScope(int depth) {
    for (auto &[name, records] : aliasMap_) {
        records.erase(std::remove_if(records.begin(), records.end(),
            [depth](const AliasRecord &r){ return r.scopeDepth >= depth; }),
            records.end());
    }
}

// ── Inter-procedural region analysis ──────────────────────────────────────────
void Sema::checkCallRegions(CallExpr &e, FunctionType *ft, size_t selfOffset) {
    // Check argument regions: passing &stack/&arena T to a function that expects
    // a longer-lived reference is a region violation.
    for (size_t i = 0; i < e.args.size(); ++i) {
        size_t paramIdx = i + selfOffset;
        if (paramIdx >= ft->paramTypes.size()) break;
        auto &argExpr = e.args[i];
        TypePtr argTy = argExpr->type;
        TypePtr paramTy = ft->paramTypes[paramIdx];
        if (!argTy || !paramTy) continue;

        // If param expects &static T but arg is &stack/&arena T → error
        if (paramTy->kind == TypeKind::Reference && argTy->kind == TypeKind::Reference) {
            auto &paramRef = static_cast<ReferenceType &>(*paramTy);
            auto &argRef = static_cast<ReferenceType &>(*argTy);
            if (paramRef.region == Region::Static &&
                (argRef.region == Region::Stack || argRef.region == Region::Arena)) {
                diag_.error(argExpr->loc,
                    "argument " + std::to_string(i+1) +
                    ": cannot pass '" + argTy->str() +
                    "' where '" + paramTy->str() +
                    "' is expected (region lifetime insufficient)");
            }
            if (paramRef.region == Region::Heap && argRef.region == Region::Stack) {
                diag_.error(argExpr->loc,
                    "argument " + std::to_string(i+1) +
                    ": cannot pass '&stack " + argRef.base->str() +
                    "' where '&heap " + paramRef.base->str() +
                    "' is expected (stack reference may dangle)");
            }
        }
    }

    // Aliasing within a single call: 'foo(&x, &x)' hands the callee two
    // references to the same variable. That's fine if both parameters are
    // shared/const references, but a conflict (mutable XOR shared) the
    // moment either side is mutable. This is scoped to the call's own
    // arguments rather than going through trackRef/aliasMap_, since the
    // borrow here only lives for the duration of the call.
    for (size_t i = 0; i < e.args.size(); ++i) {
        if (e.args[i]->kind != ExprKind::AddrOf) continue;
        auto &ue_i = static_cast<UnaryExpr &>(*e.args[i]);
        if (!ue_i.operand || ue_i.operand->kind != ExprKind::Ident) continue;
        auto &ident_i = static_cast<IdentExpr &>(*ue_i.operand);
        size_t paramIdx_i = i + selfOffset;
        if (paramIdx_i >= ft->paramTypes.size()) continue;
        auto &pty_i = ft->paramTypes[paramIdx_i];
        bool mut_i = pty_i && pty_i->kind == TypeKind::Reference &&
                     static_cast<ReferenceType &>(*pty_i).mut;

        for (size_t j = i + 1; j < e.args.size(); ++j) {
            if (e.args[j]->kind != ExprKind::AddrOf) continue;
            auto &ue_j = static_cast<UnaryExpr &>(*e.args[j]);
            if (!ue_j.operand || ue_j.operand->kind != ExprKind::Ident) continue;
            auto &ident_j = static_cast<IdentExpr &>(*ue_j.operand);
            if (ident_i.name != ident_j.name) continue;

            size_t paramIdx_j = j + selfOffset;
            if (paramIdx_j >= ft->paramTypes.size()) continue;
            auto &pty_j = ft->paramTypes[paramIdx_j];
            bool mut_j = pty_j && pty_j->kind == TypeKind::Reference &&
                         static_cast<ReferenceType &>(*pty_j).mut;

            if (mut_i || mut_j) {
                diag_.error(e.args[j]->loc,
                    "cannot borrow '" + ident_j.name + "' as " +
                    (mut_j ? "mutable" : "immutable") +
                    ": already borrowed by argument " + std::to_string(i + 1));
            }
        }
    }

    // Check return region: if function returns &stack T or &arena T, those
    // refer to the callee's stack — invalid in caller's context.
    // Only &static, &heap, and caller-scoped arena refs are valid returns.
    if (ft->returnType && ft->returnType->kind == TypeKind::Reference) {
        auto &retRef = static_cast<ReferenceType &>(*ft->returnType);
        if (retRef.region == Region::Stack) {
            diag_.error(e.loc,
                "called function returns '&stack " + retRef.base->str() +
                "': stack reference does not survive call boundary");
        }
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
                          TypeSubst                       &subs,
                          const TypePtr                   &returnType,
                          const TypePtr                   &expectedType) {
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

    // Check if there's a variadic pack parameter
    bool hasPack = false;
    std::string packName;
    for (auto &gp : genericParams) {
        if (gp.isPack) { hasPack = true; packName = gp.name; break; }
    }

    if (hasPack) {
        // Match non-pack params first
        size_t nonPackParams = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].isPack) break;
            nonPackParams = i + 1;
            if (i < argTypes.size() && hasGeneric(hasGeneric, params[i].type)) {
                if (!matchType(params[i].type, argTypes[i], subs))
                    return false;
            }
        }
        // Remaining args are pack entries
        for (size_t i = nonPackParams; i < argTypes.size(); ++i) {
            std::string key = packName + "__" + std::to_string(i - nonPackParams);
            subs[key] = argTypes[i];
        }
        // Store pack count
        subs[packName + "__count"] = makeInt(64, false); // sentinel
        int packCount = (int)argTypes.size() - (int)nonPackParams;
        subs[packName + "__count_val"] = makeInt(packCount, false); // actual count as Int type with bits=count
    } else {
        for (size_t i = 0; i < params.size() && i < argTypes.size(); ++i) {
            if (hasGeneric(hasGeneric, params[i].type)) {
                if (!matchType(params[i].type, argTypes[i], subs))
                    return false;
            }
        }
    }
    // Fallback for a generic param that appears only in the return type
    // (e.g. 'generic<T> T* vec_at(&stack Vec v, unsigned long idx)') and so
    // has nothing to bind against in the loops above: match the function's
    // return type, as a pattern, against the caller's expected type (see
    // expectedType_'s doc comment in Sema.h for where that comes from).
    // Only attempted when something is still unbound, so a call that
    // already inferred everything from its arguments is unaffected.
    bool anyUnbound = false;
    for (auto &gp : genericParams) {
        if (!gp.isPack && !subs.count(gp.name)) { anyUnbound = true; break; }
    }
    if (anyUnbound && returnType && expectedType) {
        matchType(returnType, expectedType, subs); // best-effort; still verified below
    }
    // Verify all non-pack generic params are bound
    for (auto &gp : genericParams) {
        if (gp.isPack) continue;
        if (!subs.count(gp.name)) return false;
    }
    return true;
}

Sema::MonoKey Sema::makeMonoKey(const std::string &name, const TypeSubst &subs,
                                  const std::vector<GenericParam> &params) {
    MonoKey key;
    key.funcName = name;
    for (auto &gp : params) {
        if (gp.isPack) {
            // Iterate pack entries
            for (int i = 0; ; ++i) {
                auto it = subs.find(gp.name + "__" + std::to_string(i));
                if (it == subs.end()) break;
                key.typeArgStrs.push_back(it->second->str());
            }
        } else {
            auto it = subs.find(gp.name);
            key.typeArgStrs.push_back(it != subs.end() ? it->second->str() : "?");
        }
    }
    return key;
}

std::string Sema::mangleName(const std::string &base, const TypeSubst &subs,
                               const std::vector<GenericParam> &params) {
    std::string result = "__safec_" + base;
    for (auto &gp : params) {
        if (gp.isPack) {
            for (int i = 0; ; ++i) {
                auto it = subs.find(gp.name + "__" + std::to_string(i));
                if (it == subs.end()) break;
                result += "_" + it->second->str();
            }
        } else {
            auto it = subs.find(gp.name);
            result += "_" + (it != subs.end() ? it->second->str() : "unknown");
        }
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

bool Sema::intLiteralFitsType(const Expr &e, const TypePtr &to) const {
    if (!to) return false;
    // Unwrap a leading unary minus so '-1' is checked as the value -1, not
    // just its magnitude (still an IntLitExpr underneath in this parser).
    const Expr *inner = &e;
    bool negate = false;
    if (inner->kind == ExprKind::Unary) {
        auto &u = static_cast<const UnaryExpr &>(*inner);
        if (u.op == UnaryOp::Neg && u.operand) {
            inner = u.operand.get();
            negate = true;
        }
    }
    if (inner->kind != ExprKind::IntLit) return false;
    bool isIntTarget = to->kind == TypeKind::Char  || to->kind == TypeKind::Bool ||
                       (to->kind >= TypeKind::Int8 && to->kind <= TypeKind::UInt64);
    if (!isIntTarget) return false;
    auto &pt = static_cast<const PrimType &>(*to);
    int64_t v = static_cast<const IntLitExpr &>(*inner).value;
    if (negate) v = -v;
    unsigned bits = pt.bitWidth();
    if (bits == 0 || bits >= 64) return true; // bool / 64-bit: literal always fits an int64_t
    if (pt.isSigned()) {
        int64_t lo = -(int64_t(1) << (bits - 1));
        int64_t hi = (int64_t(1) << (bits - 1)) - 1;
        return v >= lo && v <= hi;
    }
    if (v < 0) return false;
    uint64_t hi = (uint64_t(1) << bits) - 1;
    return static_cast<uint64_t>(v) <= hi;
}

// A double-valued literal narrowing to 'float' is always allowed — matches
// real C ('float x = 1.0;' is ordinary, valid C), unlike intLiteralFitsType
// there's no "does the magnitude fit" question: a float can represent any
// double's rounded value, possibly losing precision or saturating to
// +-inf for extreme magnitudes, exactly as plain C already permits without
// requiring an explicit cast.
bool Sema::floatLitFitsType(const Expr &e, const TypePtr &to) const {
    if (!to || to->kind != TypeKind::Float32) return false;
    const Expr *inner = &e;
    if (inner->kind == ExprKind::Unary) {
        auto &u = static_cast<const UnaryExpr &>(*inner);
        if (u.op == UnaryOp::Neg && u.operand) inner = u.operand.get();
    }
    return inner->kind == ExprKind::FloatLit;
}

// &stack T argument → T* parameter, call-argument-only leniency.
// canImplicitlyConvert() deliberately refuses this conversion in general
// (assigning a &stack reference into a raw pointer variable could let it
// outlive the stack frame it points into), but a CALL ARGUMENT can't
// outlive the call: the callee's frame nests entirely inside the caller's,
// so passing '&stack T' where a 'T*' out-parameter is expected (the classic
// C out-param idiom, e.g. 'get_len(&len)') is exactly as safe as passing it
// where a '&stack T' reference parameter is expected, which is already
// allowed. Kept separate from canImplicitlyConvert so non-call contexts
// (assignments, initializers) keep the stricter rule.
bool Sema::refToPointerArgCompatible(const TypePtr &from, const TypePtr &to) const {
    if (!from || !to) return false;
    if (from->kind != TypeKind::Reference || to->kind != TypeKind::Pointer) return false;
    auto &fr = static_cast<const ReferenceType &>(*from);
    auto &tp = static_cast<const PointerType &>(*to);
    if (typeEqual(fr.base, tp.base)) return true;
    auto is8 = [](const TypePtr &t) {
        return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
               t->kind == TypeKind::UInt8;
    };
    return is8(fr.base) && is8(tp.base);
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
    // Integer widening: smaller → larger (always safe)
    if (from->isInteger() && to->isInteger()) {
        auto &fp = static_cast<const PrimType &>(*from);
        auto &tp = static_cast<const PrimType &>(*to);
        if (tp.bitWidth() >= fp.bitWidth()) return true;
    }
    // Float widening: float → double (same "smaller → larger is always
    // safe" rule as integer widening just above — every float value is
    // exactly representable as a double, so nothing is lost). Narrowing
    // (double → float) still needs an explicit cast, same as int narrowing.
    if (from->isFloat() && to->isFloat()) {
        auto &fp = static_cast<const PrimType &>(*from);
        auto &tp = static_cast<const PrimType &>(*to);
        if (tp.bitWidth() >= fp.bitWidth()) return true;
    }
    // Integer → float widening: any integer type may implicitly widen to
    // any floating-point type (matches C's usual arithmetic conversions —
    // 'double d = someInt;' or passing an int where a function expects a
    // double is exactly the "smaller numeric type → bigger numeric type"
    // rule this whole block implements, just crossing the int/float
    // category line instead of staying within it). The reverse
    // (float → integer) is a narrowing, lossy conversion and still needs
    // an explicit cast.
    if (from->isInteger() && to->isFloat()) return true;
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
    // Array-to-reference decay: T[N] → &region T (same idea as array-to-
    // pointer decay above, for the reference side of the type system — e.g.
    // a 'static unsigned char storage[N];' binding into a '&static
    // unsigned char' field/parameter, exactly like C lets an array decay to
    // a pointer to its first element).
    if (from->kind == TypeKind::Array && to->kind == TypeKind::Reference) {
        auto &at = static_cast<const ArrayType &>(*from);
        auto &rt = static_cast<const ReferenceType &>(*to);
        if (typeEqual(at.element, rt.base)) return true;
    }
    // Pointer → Array parameter: the reverse of array-to-pointer decay.
    // Array-typed PARAMETERS decay to a pointer at the LLVM ABI boundary
    // (see genFunctionProto), so 'void f(unsigned char mac[6])' really takes
    // a pointer — passing an already-decayed 'unsigned char*' argument
    // (e.g. an explicit '(const unsigned char*)arr' cast) must be accepted
    // the same way plain C accepts it.
    if (from->kind == TypeKind::Pointer && to->kind == TypeKind::Array) {
        auto &fp = static_cast<const PointerType &>(*from);
        auto &tt = static_cast<const ArrayType &>(*to);
        if (typeEqual(fp.base, tt.element)) return true;
        auto is8 = [](const TypePtr &t) {
            return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
                   t->kind == TypeKind::UInt8;
        };
        if (is8(fp.base) && is8(tt.element)) return true;
    }
    // Multi-dimensional array outer-dimension decay: T[N][M] → T[][M], the
    // same C rule that lets 'char parts[8][64]' bind to a 'char[][64]'
    // parameter — only the outermost dimension is elidable (a function
    // parameter 'T x[][M]' is really 'T (*x)[M]'), so the inner dimensions
    // must still match exactly while the outer size is unconstrained.
    if (from->kind == TypeKind::Array && to->kind == TypeKind::Array) {
        auto &at = static_cast<const ArrayType &>(*from);
        auto &tt = static_cast<const ArrayType &>(*to);
        if (tt.size == -1 && typeEqual(at.element, tt.element)) return true;
    }
    // void* → any pointer is allowed in unsafe (checked by caller)
    // bool ↔ integer (narrowing check needed but simplified here)
    if (from->isBool() && to->isInteger()) return true;
    if (from->isInteger() && to->isBool()) return true;
    // null literal (void*) → any pointer/reference, and (matching plain C)
    // any T* → void* — e.g. passing a 'char*' where 'memcpy(void*, ...)'
    // expects 'void*'. Only one side needs to be void for the conversion to
    // be a safe, information-preserving widening (both-void is handled by
    // the typeEqual() fast path above already).
    if (from->kind == TypeKind::Pointer && to->kind == TypeKind::Pointer) {
        auto &fp = static_cast<const PointerType &>(*from);
        auto &tp = static_cast<const PointerType &>(*to);
        if (fp.base->isVoid() || tp.base->isVoid()) return true;
    }
    // Pointer → Reference parameter: mirrors Pointer → Array above. Raw
    // pointers (e.g. a 'const unsigned char*' function parameter that
    // already decayed from some caller's array, or any C-interop pointer)
    // must be accepted where a '&region T' reference parameter is expected,
    // the same way plain C lets a pointer be passed to a function taking a
    // pointer to its pointee type — the callee can only use it within
    // 'unsafe{}' regardless, so the region annotation carries no additional
    // safety obligation here.
    if (from->kind == TypeKind::Pointer && to->kind == TypeKind::Reference) {
        auto &fp = static_cast<const PointerType &>(*from);
        auto &tr = static_cast<const ReferenceType &>(*to);
        // null → ?&T: 'null' is always typed as 'void*' (see checkExpr's
        // NullLit case) and is the only realistic source of a genuinely
        // void*-typed value flowing into a reference-typed slot — this is
        // what actually makes '?&T x = null;' legal at all; only the
        // *nullable* reference kind may receive it, matching how &T → ?&T
        // is already implicit just below but T → &T is not.
        if (fp.base->isVoid() && tr.nullable) return true;
        if (typeEqual(fp.base, tr.base)) return true;
        auto is8 = [](const TypePtr &t) {
            return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
                   t->kind == TypeKind::UInt8;
        };
        if (is8(fp.base) && is8(tr.base)) return true;
        return false;
    }
    // T → ?T (implicit "always present" wrap) and null → ?T (the empty
    // case) — mirrors &T → ?&T being implicit above: a function declared
    // to return '?T' can just 'return someTValue;' or 'return null;'
    // without a separate explicit wrap/empty-construction step.
    if (to->kind == TypeKind::Optional) {
        auto &ot = static_cast<const OptionalType &>(*to);
        if (typeEqual(from, ot.inner)) return true;
        if (from->kind == TypeKind::Pointer &&
            static_cast<const PointerType &>(*from).base->isVoid())
            return true;
    }
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
        // Extern (outliving) references → raw pointer: always safe, since
        // '?&T'/'&T' (Region::Extern) never claimed any lifetime guarantee
        // in the first place — handing the same pointer value back to
        // another extern-facing T*/void* parameter carries no new risk.
        if (fr.region == Region::Extern &&
            (typeEqual(fr.base, tp.base) || tp.base->isVoid())) return true;
    }
    // Newtype ← base type: allow initialization from the underlying type
    if (to->kind == TypeKind::Newtype) {
        auto &nt = static_cast<const NewtypeType &>(*to);
        if (typeEqual(from, nt.base)) return true;
        // Also allow if the base types are implicitly convertible
        if (canImplicitlyConvert(from, nt.base)) return true;
    }
    // Newtype → base type: allow extracting the underlying value
    if (from->kind == TypeKind::Newtype) {
        auto &nt = static_cast<const NewtypeType &>(*from);
        if (typeEqual(nt.base, to)) return true;
    }
    // Enum → integer widening: enum values can be used as integers
    if (from->kind == TypeKind::Enum && to->isInteger()) return true;
    // integer → Enum: allow integer literals to initialize enum variables
    if (from->isInteger() && to->kind == TypeKind::Enum) return true;
    if (from->kind == TypeKind::Reference && to->kind == TypeKind::Reference) {
        auto &fr = static_cast<const ReferenceType &>(*from);
        auto &tr = static_cast<const ReferenceType &>(*to);
        // Same region, same base, nullable compat: non-null → nullable is OK
        if (!typeEqual(fr.base, tr.base)) return false;
        // Any region → Region::Extern ('&T'/'?&T', no declared region):
        // an outliving reference makes no lifetime claim, so it accepts
        // a reference from any other region (see Type.h's Region::Extern
        // doc comment) — the one pairing exempt from the same-region
        // rule below. Nullability still applies normally: a nullable
        // source can't silently become non-nullable.
        if (tr.region == Region::Extern) {
            if (fr.nullable && !tr.nullable) return false;
            return true;
        }
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
