#include "safec/Sema.h"
#include <functional>
#include "safec/Clone.h"
#include <cassert>
#include <cstdint>
#include <algorithm>

// 'dealloc(p)' / 'std::dealloc(p)' (the generic std/mem.h API) and its
// type-erased 'dealloc_raw_' form (what malloc_defer desugars to) both free
// a heap block; the UAF/double-free analysis (MemoryFlow.cpp) treats them,
// and an allocator's 'x.dealloc(p)' method, as freeing their argument.
static bool isDeallocCallName(const std::string &n) {
    auto endsWith = [&](const char *suf) {
        size_t k = std::char_traits<char>::length(suf);
        return n.size() >= k && n.compare(n.size() - k, k, suf) == 0;
    };
    return n == "dealloc" || n == "dealloc_raw_" ||
           endsWith("::dealloc") || endsWith("::dealloc_raw_");
}


namespace safec {

bool typeMentionsGeneric(const TypePtr &ty);

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

bool Sema::isBuiltinTrait(const std::string &trait) const {
    for (auto *td = builtinTraits_; !td->name.empty(); ++td)
        if (td->name == trait) return true;
    return false;
}

// Supertraits must name a trait (user or builtin) and must not lead back to
// the trait itself. An offending edge is reported and dropped, so the
// remaining graph is acyclic.
void Sema::checkTraitSupertraits() {
    std::unordered_map<std::string, int> state;   // 0 unvisited, 1 on the path, 2 done
    std::function<void(const std::string &)> visit = [&](const std::string &t) {
        state[t] = 1;
        auto &supers = userTraitSupers_[t];
        for (size_t i = 0; i < supers.size();) {
            const std::string &s = supers[i];
            bool known = userTraits_.count(s) || isBuiltinTrait(s) || s == "Indexed" || s == "Pointer";
            if (!known) {
                diag_.error(userTraitLocs_[t], "unknown supertrait '" + s + "' of trait '" + t + "'");
                supers.erase(supers.begin() + (long)i);
                continue;
            }
            if (userTraits_.count(s)) {
                if (state[s] == 1) {
                    diag_.error(userTraitLocs_[t], "trait '" + t + "' is its own supertrait (through '" + s + "')");
                    supers.erase(supers.begin() + (long)i);
                    continue;
                }
                if (state[s] == 0) visit(s);
            }
            ++i;
        }
        state[t] = 2;
    };
    for (auto &[name, _] : userTraits_)
        if (state[name] == 0) visit(name);
}

std::string Sema::constraintError(const TypePtr &ty, const std::string &trait, const std::string &where) const {
    std::string why = traitMismatch(ty, trait);
    return "type '" + ty->str() + "' does not satisfy constraint '" + trait + "' " + where +
           (why == "-" ? std::string() : " (" + why + ")");
}

std::string Sema::traitMismatch(const TypePtr &ty, const std::string &trait) const {
    if (trait.empty()) return "";
    if (!ty) return "-";

    // Structural traits: judged by TypeKind, not by looking up a method —
    // SafeC has no 'operator[]' overload mechanism, so these apply only to
    // the built-in kinds that are natively indexable/pointer-like. Checked
    // before the isArithmetic() shortcut below so 'int' doesn't spuriously
    // satisfy 'Indexed'/'Pointer' just for being numeric. A struct can only
    // satisfy them via a user 'trait' that names an explicit accessor
    // method instead (e.g. 'trait Indexable { int get(int i); }').
    if (trait == "Indexed") {
        bool ok = ty->kind == TypeKind::Array   || ty->kind == TypeKind::Pointer ||
                  ty->kind == TypeKind::Slice   || ty->kind == TypeKind::Vector  ||
                  ty->kind == TypeKind::Reference;
        return ok ? "" : "-";
    }
    if (trait == "Pointer") {
        return ty->kind == TypeKind::Pointer || ty->kind == TypeKind::Reference ? "" : "-";
    }

    // Primitive numeric types satisfy Numeric/Ord/Eq/Add/Sub/Mul/Div — not
    // user traits, which name methods a primitive doesn't have.
    if (ty->isArithmetic()) return isBuiltinTrait(trait) ? "" : "-";
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
                        return "no '" + op + "'";
                }
                return "";
            }
        }
        auto uit = userTraits_.find(trait);
        if (uit != userTraits_.end()) {
            auto dit = userTraitDecls_.find(trait);
            for (auto &methodName : uit->second) {
                std::string key = st.name + "::" + methodName;
                auto mit = methodRegistry_.find(key);
                if (mit == methodRegistry_.end())
                    return "no method '" + methodName + "'";
                // The trait's declaration fixes the shape: static or not and
                // how many parameters (types aren't compared: one trait can
                // serve several sample types, say).
                if (dit == userTraitDecls_.end()) continue;
                for (auto &m : dit->second->methods) {
                    if (m.name != methodName) continue;
                    const FunctionDecl &f = *mit->second;
                    if (m.isStatic != f.isStaticMethod)
                        return "'" + methodName + "' is " + (m.isStatic ? "static" : "an instance method") +
                               " in the trait";
                    size_t have = f.params.size() - (f.isStaticMethod ? 0 : 1);
                    if (!f.isVariadic && have != m.params.size())
                        return "'" + methodName + "' takes " + std::to_string(have) +
                               " parameter(s), the trait " + std::to_string(m.params.size());
                }
            }
            auto sit = userTraitSupers_.find(trait);
            if (sit != userTraitSupers_.end())
                for (auto &super : sit->second) {
                    std::string why = traitMismatch(ty, super);
                    if (!why.empty()) return why == "-" ? "not '" + super + "'" : why + ", from '" + super + "'";
                }
            return "";
        }
    }
    return "-";
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
            // Still mentions a type parameter (a generic function's
            // 'struct Vec<T>' return type): keep it symbolic until
            // monomorphization substitutes T, instead of instantiating a
            // bogus 'Vec_T'.
            // An unknown bare name ('T' parsed as a struct name inside a
            // template) counts too, at any depth: 'Pool<Node<T>>' must stay
            // symbolic, not instantiate 'Pool' over a half-resolved 'Node'.
            auto mentionsPlaceholder = [&](const auto &self, const TypePtr &a) -> bool {
                if (!a) return false;
                // through '&Node<T>', 'Node<T>*', '?&Node<T>', 'Node<T>[4]'
                if (a->kind == TypeKind::Reference) return self(self, static_cast<ReferenceType &>(*a).base);
                if (a->kind == TypeKind::Pointer) return self(self, static_cast<PointerType &>(*a).base);
                if (a->kind == TypeKind::Optional) return self(self, static_cast<OptionalType &>(*a).inner);
                if (a->kind == TypeKind::Array) return self(self, static_cast<ArrayType &>(*a).element);
                if (a->kind != TypeKind::Struct) return false;
                auto &as = static_cast<StructType &>(*a);
                if (as.typeArgs.empty())
                    return !as.isDefined && !typeRegistry_.count(as.name) &&
                           !genericStructTemplates_.count(as.name);
                for (auto &x : as.typeArgs) if (self(self, x)) return true;
                return false;
            };
            for (auto &a : resolvedArgs)
                if (typeMentionsGeneric(a) || mentionsPlaceholder(mentionsPlaceholder, a)) {
                    auto nt = std::make_shared<StructType>(st.name, st.isUnion);
                    nt->typeArgs = resolvedArgs;
                    return nt;
                }
            return instantiateGenericStruct(st.name, resolvedArgs, typeUseLoc_);
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
            return makeReference(resolvedBase, rt.region, rt.nullable, rt.mut, rt.arenaName, rt.pointeeVolatile);
    }
    if (ty->kind == TypeKind::Pointer) {
        auto &pt = static_cast<PointerType &>(*ty);
        auto resolvedBase = resolveType(pt.base);
        if (resolvedBase != pt.base)
            return makePointer(resolvedBase, pt.isConst, pt.pointeeVolatile);
    }
    if (ty->kind == TypeKind::Array) {
        auto &at = static_cast<ArrayType &>(*ty);
        auto resolvedElem = resolveType(at.element);
        if (resolvedElem != at.element) {
            auto na = makeArray(resolvedElem, at.size);
            static_cast<ArrayType &>(*na).elementVolatile = at.elementVolatile;
            return na;
        }
    }
    if (ty->kind == TypeKind::Vector) {
        auto &vt = static_cast<VectorType &>(*ty);
        auto resolvedElem = resolveType(vt.element);
        if (resolvedElem != vt.element)
            return makeVector(resolvedElem, vt.width);
    }
    // 'fn void(&Node<F16>)': resolve the parameter and return types too, so a
    // generic struct named there is the same instance as everywhere else.
    if (ty->kind == TypeKind::Function) {
        auto &ft = static_cast<FunctionType &>(*ty);
        bool changed = false;
        TypePtr ret = ft.returnType ? resolveType(ft.returnType) : ft.returnType;
        changed = ret != ft.returnType;
        std::vector<TypePtr> ps;
        for (auto &p : ft.paramTypes) {
            ps.push_back(p ? resolveType(p) : p);
            if (ps.back() != p) changed = true;
        }
        if (changed) return makeFunction(ret, ps, ft.variadic);
    }
    return ty;
}

void Sema::resolveMethodBodyDecls() {
    // A bare name parsed as a struct that names no struct, typedef or
    // generic template (looked through references, pointers, arrays,
    // optionals and function types).
    auto unknownName = [&](const auto &self, const TypePtr &t) -> std::string {
        if (!t) return "";
        switch (t->kind) {
        case TypeKind::Struct: {
            auto &st = static_cast<const StructType &>(*t);
            if (!st.isDefined && st.typeArgs.empty() && !typeRegistry_.count(st.name) &&
                !genericStructTemplates_.count(st.name))
                return st.name;
            return "";
        }
        case TypeKind::Reference: return self(self, static_cast<const ReferenceType &>(*t).base);
        case TypeKind::Pointer:   return self(self, static_cast<const PointerType &>(*t).base);
        case TypeKind::Array:     return self(self, static_cast<const ArrayType &>(*t).element);
        case TypeKind::Optional:  return self(self, static_cast<const OptionalType &>(*t).inner);
        case TypeKind::Function: {
            auto &ft = static_cast<const FunctionType &>(*t);
            std::string n = self(self, ft.returnType);
            for (auto &p : ft.paramTypes) if (n.empty()) n = self(self, p);
            return n;
        }
        default: return "";
        }
    };
    // Trait method signatures: resolved too, so an unknown type in one is
    // reported instead of silently ignored (conformance itself compares
    // names, static-ness and parameter counts; see traitMismatch).
    for (auto &[tname, td] : userTraitDecls_) {
        for (auto &m : td->methods) {
            std::string bad;
            if (m.returnType) { m.returnType = resolveType(m.returnType); bad = unknownName(unknownName, m.returnType); }
            for (auto &p : m.params) {
                if (!p.type) continue;
                p.type = resolveType(p.type);
                if (bad.empty()) bad = unknownName(unknownName, p.type);
            }
            if (!bad.empty())
                diag_.error(m.loc, "unknown type '" + bad + "' in trait '" + tname + "' method '" + m.name + "'");
        }
    }
    for (auto &[key, d] : methodBodyDecl_) {
        bool changed = false;
        if (d->returnType) { auto r = resolveType(d->returnType); changed |= r != d->returnType; d->returnType = r; }
        for (auto &p : d->params) {
            if (!p.type) continue;
            auto r = resolveType(p.type); changed |= r != p.type; p.type = r;
        }
        std::string bad = unknownName(unknownName, d->returnType);
        for (auto &p : d->params) if (bad.empty()) bad = unknownName(unknownName, p.type);
        if (!bad.empty()) {
            diag_.error(d->loc, "unknown type '" + bad + "' in the declaration of '" + key + "'");
            continue;
        }
        if (changed) {
            // Calls type-check against the symbol's function type.
            std::vector<TypePtr> pts;
            for (auto &p : d->params) pts.push_back(p.type);
            if (auto *sym = lookup(d->name))
                if (sym->fnDecl == d) sym->type = makeReference(makeFunction(d->returnType, pts, d->isVariadic), Region::Static);
        }
    }
}

// ── Pass 1: Collect top-level declarations ─────────────────────────────────────
bool Sema::run() {
    // Global scope
    pushScope();
    collectDecls(tu_);
    resolveMethodBodyDecls();
    checkTraitSupertraits();
    completeGenericStructInstances();
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

    checkMemoryFlow();

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
            userTraitSupers_[td.name] = td.supertraits;
            userTraitLocs_[td.name] = td.loc;
            userTraitDecls_[td.name] = &td;
            break;
        }
        default: break;
        }
    }
}

// Replace StructType{name} with GenericType{name} when 'name' is in the
// generic parameter list.  Recurses into pointer/reference/array/function types.
// Does this type tree still contain an unsubstituted generic parameter?
bool typeMentionsGeneric(const TypePtr &ty) {
    if (!ty) return false;
    switch (ty->kind) {
    case TypeKind::Generic: return true;
    case TypeKind::Pointer: return typeMentionsGeneric(static_cast<const PointerType &>(*ty).base);
    case TypeKind::Reference: return typeMentionsGeneric(static_cast<const ReferenceType &>(*ty).base);
    case TypeKind::Array: return typeMentionsGeneric(static_cast<const ArrayType &>(*ty).element);
    case TypeKind::Optional: return typeMentionsGeneric(static_cast<const OptionalType &>(*ty).inner);
    case TypeKind::Slice: return typeMentionsGeneric(static_cast<const SliceType &>(*ty).element);
    case TypeKind::Vector: return typeMentionsGeneric(static_cast<const VectorType &>(*ty).element);
    case TypeKind::Tuple:
        for (auto &e : static_cast<const TupleType &>(*ty).elementTypes)
            if (typeMentionsGeneric(e)) return true;
        return false;
    case TypeKind::Function: {
        auto &ft = static_cast<const FunctionType &>(*ty);
        if (typeMentionsGeneric(ft.returnType)) return true;
        for (auto &p : ft.paramTypes) if (typeMentionsGeneric(p)) return true;
        return false;
    }
    case TypeKind::Struct:
        for (auto &a : static_cast<const StructType &>(*ty).typeArgs)
            if (typeMentionsGeneric(a)) return true;
        return false;
    default: return false;
    }
}

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
        return (nb == pt.base) ? ty : makePointer(nb, pt.isConst, pt.pointeeVolatile);
    }
    if (ty->kind == TypeKind::Reference) {
        auto &rt = static_cast<const ReferenceType &>(*ty);
        auto nb = resolveGenericNames(rt.base, gps);
        return (nb == rt.base) ? ty
             : makeReference(nb, rt.region, rt.nullable, rt.mut, rt.arenaName, rt.pointeeVolatile);
    }
    if (ty->kind == TypeKind::Array) {
        auto &at = static_cast<const ArrayType &>(*ty);
        auto ne = resolveGenericNames(at.element, gps);
        if (ne == at.element) return ty;
        auto na = makeArray(ne, at.size);
        static_cast<ArrayType &>(*na).elementVolatile = at.elementVolatile;
        return na;
    }
    if (ty->kind == TypeKind::Struct) {   // 'struct Vec<T>' in a generic signature
        auto &st = static_cast<const StructType &>(*ty);
        if (st.typeArgs.empty()) return ty;
        bool changed = false;
        std::vector<TypePtr> na;
        for (auto &a : st.typeArgs) {
            na.push_back(resolveGenericNames(a, gps));
            if (na.back() != a) changed = true;
        }
        if (!changed) return ty;
        auto nt = std::make_shared<StructType>(st.name, st.isUnion);
        nt->typeArgs = std::move(na);
        return nt;
    }
    if (ty->kind == TypeKind::Optional) {
        auto &ot = static_cast<const OptionalType &>(*ty);
        auto ni = resolveGenericNames(ot.inner, gps);
        return (ni == ot.inner) ? ty : makeOptional(ni);
    }
    if (ty->kind == TypeKind::Slice) {
        auto &sl = static_cast<const SliceType &>(*ty);
        auto ne = resolveGenericNames(sl.element, gps);
        return (ne == sl.element) ? ty : makeSlice(ne);
    }
    if (ty->kind == TypeKind::Vector) {
        auto &vt = static_cast<const VectorType &>(*ty);
        auto ne = resolveGenericNames(vt.element, gps);
        return (ne == vt.element) ? ty : makeVector(ne, vt.width);
    }
    if (ty->kind == TypeKind::Tuple) {
        auto &tt = static_cast<const TupleType &>(*ty);
        bool changed = false;
        std::vector<TypePtr> ne;
        for (auto &e : tt.elementTypes) {
            ne.push_back(resolveGenericNames(e, gps));
            if (ne.back() != e) changed = true;
        }
        return changed ? makeTuple(std::move(ne)) : ty;
    }
    if (ty->kind == TypeKind::Function) {
        auto &ft = static_cast<const FunctionType &>(*ty);
        auto nr = resolveGenericNames(ft.returnType, gps);
        bool changed = nr != ft.returnType;
        std::vector<TypePtr> np;
        for (auto &p : ft.paramTypes) {
            np.push_back(resolveGenericNames(p, gps));
            if (np.back() != p) changed = true;
        }
        return changed ? makeFunction(nr, std::move(np), ft.variadic) : ty;
    }
    return ty;
}

void Sema::collectFunction(FunctionDecl &fn) {
    // Remember libc-named functions this TU itself implements (a bare-metal
    // runtime defining malloc/free/...): calling them in freestanding mode
    // is not a missing-runtime hazard.
    if (fn.body && fn.namespaceName.empty()) providedFns_.insert(fn.name);
    // Method support (OBJECT.md §4): T::m(P...) → T_m(T* self, P...)
    if (fn.isMethod) {
        // 'Alias::m' with Alias a typedef of a struct (or of one instance of
        // a generic struct): a method of that struct (or of that instance only).
        if (!genericStructTemplates_.count(fn.methodOwner)) {
            auto tit = typeRegistry_.find(fn.methodOwner);
            if (tit != typeRegistry_.end() && tit->second && tit->second->kind == TypeKind::Struct) {
                const std::string &sname = static_cast<StructType &>(*tit->second).name;
                if (sname != fn.methodOwner) fn.methodOwner = sname;
            }
        }
        if (fn.body) {
            auto ait = methodOwnGenericArity_.find(fn.methodOwner + "::" + fn.name);
            if (ait != methodOwnGenericArity_.end() &&
                genericStructTemplates_.find(fn.methodOwner) == genericStructTemplates_.end() &&
                ait->second != fn.genericParams.size())
                diag_.error(fn.loc, "'" + fn.methodOwner + "::" + fn.name + "' is declared with " +
                            std::to_string(ait->second) + " type parameter(s) in the struct body but defined with " +
                            std::to_string(fn.genericParams.size()));
        }
        {
            auto sit = staticMethodNames_.find(fn.methodOwner);
            if (sit != staticMethodNames_.end() && sit->second.count(fn.name))
                fn.isStaticMethod = true;
        }
        if (fn.isStaticMethod) {
            // No self. On a generic struct template it becomes a generic
            // function over the struct's parameters (inferred from the
            // arguments and the call's target type, like any generic
            // function); below, the ordinary function path resolves its
            // types against them.
            auto genTplIt = genericStructTemplates_.find(fn.methodOwner);
            if (genTplIt != genericStructTemplates_.end() && fn.genericParams.empty())
                fn.genericParams = genTplIt->second->genericParams;
            if (genTplIt == genericStructTemplates_.end() &&
                typeRegistry_.find(fn.methodOwner) == typeRegistry_.end())
                diag_.error(fn.loc, "method owner struct '" + fn.methodOwner + "' not declared");
            std::string originalName = fn.name;
            fn.name = fn.methodOwner + "_" + fn.name;
            std::string key = fn.methodOwner + "::" + originalName;
            // A bodyless declaration never replaces a generic template's
            // definition: instantiation needs the body.
            auto prev = staticMethodRegistry_.find(key);
            if (prev == staticMethodRegistry_.end() || fn.body || !prev->second->body ||
                prev->second->genericParams.empty()) {
                staticMethodRegistry_[key] = &fn;
                methodRegistry_[key] = &fn;
            }
        } else {
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
        // Register in method registry under original qualified name (a
        // bodyless declaration never replaces a generic template's body)
        {
            std::string key = fn.methodOwner + "::" + originalName;
            auto prev = methodRegistry_.find(key);
            if (prev == methodRegistry_.end() || fn.body || !prev->second->body ||
                prev->second->genericParams.empty())
                methodRegistry_[key] = &fn;
        }
        }
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

    // An out-of-line method definition must match its declaration in the
    // struct body (same parameters, return type, const-ness, static-ness).
    if (fn.isMethod && fn.body) {
        std::string owner = fn.methodOwner;
        std::string mname = fn.name.substr(owner.size() + 1);   // mangled Owner_name
        auto dit = methodBodyDecl_.find(owner + "::" + mname);
        if (dit != methodBodyDecl_.end() && dit->second != &fn &&
            dit->second->genericParams.size() == fn.genericParams.size()) {   // else reported above
            FunctionDecl &d = *dit->second;
            // The declaration was resolved when its struct was collected; a
            // typedef declared after the struct only resolves now.
            if (d.returnType) d.returnType = resolveType(d.returnType);
            for (auto &p : d.params) if (p.type) p.type = resolveType(p.type);
            // Compared without const qualifiers: a leading 'const' on an
            // out-of-line definition ('inline const &char S::f() const') is
            // parsed as the function's const flag, not part of the type.
            auto bare = [](const TypePtr &t) {
                std::string s = t->str();
                for (size_t p; (p = s.find("const ")) != std::string::npos;) s.erase(p, 6);
                return s;
            };
            std::string what;
            if (d.isStaticMethod != fn.isStaticMethod)
                what = d.isStaticMethod ? "it is declared static" : "it is declared as an instance method";
            else if (d.isConstMethod != fn.isConstMethod)
                what = d.isConstMethod ? "it is declared const" : "it is not declared const";
            else if (d.params.size() != fn.params.size())
                what = "it is declared with " + std::to_string(d.params.size() - (d.isStaticMethod ? 0 : 1)) +
                       " parameter(s)";
            else if (d.returnType && fn.returnType && bare(d.returnType) != bare(fn.returnType))
                what = "it is declared to return '" + d.returnType->str() + "'";
            else {
                for (size_t i = 0; i < d.params.size(); ++i) {
                    if (!d.params[i].type || !fn.params[i].type) continue;
                    if (bare(d.params[i].type) != bare(fn.params[i].type)) {
                        size_t n = d.isStaticMethod ? i + 1 : i;
                        what = "parameter " + std::to_string(n) + " is declared '" + d.params[i].type->str() +
                               "', defined '" + fn.params[i].type->str() + "'";
                        break;
                    }
                }
            }
            if (!what.empty())
                diag_.error(fn.loc, "definition of '" + owner + "::" + mname +
                            "' doesn't match its declaration in the struct body: " + what);
        }
    }
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
    for (auto &md : sd.methodDecls) {
        if (md.isStatic) staticMethodNames_[sd.name].insert(md.name);
        methodOwnGenericArity_[sd.name + "::" + md.name] = md.genericParams.size();
        if (!sd.genericParams.empty() && !md.genericParams.empty() && !md.isStatic)
            diag_.error(md.loc, "a method of a generic struct can't have type parameters of its own yet");
    }
    if (!sd.genericParams.empty()) {
        // Resolve each field's raw parsed type (the parser has no notion
        // of generics — 'T value;' parses its type as an ordinary
        // StructType("T") reference, same as any other unknown type name)
        // against this struct's own generic parameter list, in place —
        // matching collectFunction's identical one-time resolution of a
        // generic function template's own param/return types. Once done,
        // substituteType (used by cloneStructDecl at instantiation time)
        // can find the GenericType nodes it needs to replace.
        // Registered first: a field or method may name the template itself
        // ('struct Vec<&Node<T>> children;' inside 'Node').
        genericStructTemplates_[sd.name] = &sd;
        for (auto &f : sd.fields)
            f.type = resolveGenericNames(resolveType(f.type), sd.genericParams);
        for (auto &md : sd.methodDecls) {
            md.returnType = resolveGenericNames(resolveType(md.returnType), sd.genericParams);
            for (auto &p : md.params)
                p.type = resolveGenericNames(resolveType(p.type), sd.genericParams);
        }
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
        fd.isVolatile  = f.isVolatile;
        if (fd.isVolatile) fd.type = withVolatileElements(fd.type);
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
        synth->isStaticMethod = md.isStatic;
        synth->genericParams = md.genericParams;
        synth->isExtern      = true; // no body here; defined elsewhere and linked
        collectFunction(*synth);
        methodBodyDecl_[sd.name + "::" + md.name] = synth.get();
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
    bool boundFailed = false;
    for (size_t i = 0; i < tpl->genericParams.size(); ++i) {
        auto &gp = tpl->genericParams[i];
        subs[gp.name] = typeArgs[i];
        for (auto &constraint : gp.constraints) {
            if (!satisfiesTrait(typeArgs[i], constraint)) {
                diag_.error(loc, constraintError(typeArgs[i], constraint, "for generic parameter '" + gp.name +
                                                 "' of struct '" + name + "'"));
                boundFailed = true;
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
    // Re-entered while this very instance is still being collected (its
    // own methods mention 'Vec<T>', e.g. clone()): collectStruct has
    // already registered the in-progress type under its mangled name.
    {
        std::string mname = mangleName(name, subs, tpl->genericParams);
        auto inProgress = typeRegistry_.find(mname);
        if (inProgress != typeRegistry_.end() && inProgress->second &&
            inProgress->second->kind == TypeKind::Struct)
            return inProgress->second;
    }
    auto clone = cloneStructDecl(*tpl, subs);
    clone->name = mangleName(name, subs, tpl->genericParams);
    // Static methods of a generic struct are generic functions (see
    // collectFunction), not per-instance methods.
    clone->methodDecls.erase(std::remove_if(clone->methodDecls.begin(), clone->methodDecls.end(),
                                            [](const MethodDecl &m) { return m.isStatic; }),
                             clone->methodDecls.end());
    structInstanceOrigin_[clone->name] = {name, typeArgs};
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
    genericStructInstances_[name].push_back({clonePtr, subs});
    auto methodsIt = genericStructMethodTemplates_.find(name);
    if (methodsIt != genericStructMethodTemplates_.end())
        for (FunctionDecl *methodTpl : methodsIt->second)
            instantiateStructMethod(methodTpl, clonePtr, subs, /*checkBody=*/!boundFailed);

    return clonePtr->type;
}

// One method template of a generic struct for one instance: clone,
// substitute, add self, mangle, register and type-check — what
// collectFunction does for an ordinary method. Each (instance, template)
// pair is done once; a bodyless declaration is skipped when the same
// method also has a definition.
void Sema::instantiateStructMethod(FunctionDecl *methodTpl, StructDecl *clonePtr, const TypeSubst &subs,
                                   bool checkBody) {
    if (!instantiatedStructMethods_.insert({clonePtr, methodTpl}).second) return;
    if (!methodTpl->body) {
        auto it = genericStructMethodTemplates_.find(methodTpl->methodOwner);
        if (it != genericStructMethodTemplates_.end())
            for (FunctionDecl *other : it->second)
                if (other != methodTpl && other->body && other->name == methodTpl->name) return;
    }
    auto methodClone = cloneFunctionDecl(*methodTpl, subs);
    std::string originalName = methodTpl->name;
    methodClone->methodOwner = clonePtr->name;
    methodClone->isMonomorphized = true;
    if (methodClone->returnType) methodClone->returnType = resolveType(methodClone->returnType);
    for (auto &p : methodClone->params) if (p.type) p.type = resolveType(p.type);

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

    if (checkBody) {
        bool savedPure = checkingPure_;
        int savedLoopDepth = loopDepth_;
        auto savedLabels = functionLabels_;
        checkFunction(*methodClone);
        checkingPure_ = savedPure;
        loopDepth_ = savedLoopDepth;
        functionLabels_ = savedLabels;
    }

    monoFunctions_.push_back(std::move(methodClone));
}

// Methods of a generic struct can be declared or defined after one of its
// instances was created (an instance named by a typedef in one header, a
// method added in another): give every instance the templates it missed.
void Sema::completeGenericStructInstances() {
    for (auto &[name, instances] : genericStructInstances_) {
        auto methodsIt = genericStructMethodTemplates_.find(name);
        if (methodsIt == genericStructMethodTemplates_.end()) continue;
        auto tpls = methodsIt->second;
        auto insts = instances;
        for (auto &[inst, subs] : insts)
            for (FunctionDecl *tpl : tpls) instantiateStructMethod(tpl, inst, subs);
    }
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
        // Store in a stable location (leak intentionally — lives as long as
        // this thread). thread_local, not a plain process-wide static: with
        // parallel compilation (main.cpp's -j), two threads compiling two
        // different files that both declare a region would otherwise
        // push_back into the SAME shared vector with no synchronization — a
        // real data race, not hypothetical (this codebase's own std/mem.sc,
        // std/collections/*.sc etc. all declare regions, so any real
        // parallel build hits this constantly). thread_local keeps each
        // compiling thread's FunctionDecls in its own vector; still safe to
        // read from CodeGen afterward as long as Sema and CodeGen for the
        // same file run on the same thread, which main.cpp's compileOneFile
        // always does (the whole per-file pipeline is sequential within one
        // thread, only different files' pipelines run concurrently).
        thread_local static std::vector<std::unique_ptr<FunctionDecl>> arenaResetFns;
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
        thread_local static std::vector<std::unique_ptr<FunctionDecl>> arenaDestroyFns;
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
        thread_local static std::vector<std::unique_ptr<FunctionDecl>> arenaMarkFns;
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
        thread_local static std::vector<std::unique_ptr<FunctionDecl>> arenaFreeToFns;
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
        auto prior = namespaceVarRegistry_.find(gv.namespaceName + "::" + originalName);
        if (!gv.isExtern && prior != namespaceVarRegistry_.end() && prior->second->isExtern) {
            // The definition of a variable an earlier header declared
            // 'extern' (namespace std { extern struct Vfs vfs_root; } in
            // vfs.h, defined in vfs.sc): same entity, so same symbol —
            // the declaration's unmangled name.
            gv.name = prior->second->name;
        } else if (!gv.isExtern) {
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
    if (gv.isVolatile) gv.type = withVolatileElements(gv.type);
    VarDecl *vd = new VarDecl{}; // leaked intentionally, matches param/local pattern
    vd->name        = gv.name;
    vd->isVolatile  = gv.isVolatile;
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
    if (d.loc.line) typeUseLoc_ = d.loc;
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

    if (std::find(checkedFns_.begin(), checkedFns_.end(), &fn) == checkedFns_.end())
        checkedFns_.push_back(&fn);

    pushScope();
    // Register parameters
    fn.paramVars.clear();
    for (auto &p : fn.params) {
        p.type = resolveType(p.type);
        if (p.isVolatile) p.type = withVolatileElements(p.type);
        VarDecl *vd = new VarDecl{};  // leaked intentionally for demo
        vd->name        = p.name;
        vd->type        = p.type;
        vd->isParam     = true;
        vd->isVolatile  = p.isVolatile;
        vd->initialized = true;
        vd->scopeDepth  = scopeDepth_;
        fn.paramVars.push_back(vd);

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
        // 'some(r)' on a nullable reference binds r as the non-null
        // reference itself (same object, same region) — the sanctioned way
        // to narrow '?&T' to '&T' after the null check; a cast can't do it.
        isNullLike = true;
        auto &rt = static_cast<const ReferenceType &>(*subjectTy);
        payloadTy  = makeReference(rt.base, rt.region, /*nullable=*/false, rt.mut, rt.arenaName,
                                   rt.pointeeVolatile);
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
    if (s.loc.line) typeUseLoc_ = s.loc;
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
        // Type-checked here; it runs at each return (see MemoryFlow.cpp's
        // doReturn for how its frees/resets are ordered).
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
        std::unordered_set<int> nullTagsSeen;
        for (auto &arm : ms.arms) {
            resolveMatchArmPatterns(arm.patterns, isTaggedUnion, unionSt,
                                     isNullLike, isOptional, nullPayloadTy, diag_,
                                     ms.loc, hasWildcard,
                                     (isNullLike || isOptional) ? &nullTagsSeen : nullptr,
                                     enumSubject);
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
        // null/none plus some(x) covers a nullable/optional subject completely.
        if ((isNullLike || isOptional) && nullTagsSeen.count(0) && nullTagsSeen.count(1))
            hasWildcard = true;
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
        // A condition Sema can evaluate now (literals, sizeof, comparisons,
        // logic) selects its branch here, and only that branch is checked:
        // in a generic body the other one may not type-check for this
        // instantiation ('if const (sizeof(E) == 4UL) { float path } else
        // { E::load(...) }'). Otherwise both are checked and
        // ConstEvalEngine decides later.
        auto &ics = static_cast<IfConstStmt &>(s);
        if (ics.cond) checkExpr(*ics.cond);
        std::optional<int64_t> v;
        if (ics.cond) v = tryConstInt(*ics.cond);
        if (v) {
            ics.constResult = *v != 0;
            if (*v) { if (ics.then) checkStmt(*ics.then, fn); }
            else if (ics.else_) checkStmt(*ics.else_, fn);
            break;
        }
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

// The value of an integer constant expression Sema can evaluate on its own:
// literals, sizeof(type), unary - and !, arithmetic, comparisons and logic.
std::optional<int64_t> Sema::tryConstInt(const Expr &e) {
    switch (e.kind) {
    case ExprKind::IntLit: return static_cast<const IntLitExpr &>(e).value;
    case ExprKind::SizeofType: {
        auto &se = static_cast<const SizeofTypeExpr &>(e);
        TypePtr t = resolveType(se.ofType);
        if (!t || t->isError() || t->kind == TypeKind::Generic) return std::nullopt;
        return (int64_t)sizeOfType(t);
    }
    case ExprKind::Unary: {
        auto &ue = static_cast<const UnaryExpr &>(e);
        auto v = tryConstInt(*ue.operand);
        if (!v) return std::nullopt;
        if (ue.op == UnaryOp::Neg) return -*v;
        if (ue.op == UnaryOp::Not) return (int64_t)(*v == 0);
        return std::nullopt;
    }
    case ExprKind::Binary: {
        auto &be = static_cast<const BinaryExpr &>(e);
        auto a = tryConstInt(*be.left);
        if (!a) return std::nullopt;
        if (be.op == BinaryOp::LogAnd && !*a) return (int64_t)0;
        if (be.op == BinaryOp::LogOr && *a) return (int64_t)1;
        auto b = tryConstInt(*be.right);
        if (!b) return std::nullopt;
        switch (be.op) {
        case BinaryOp::Add: return *a + *b;
        case BinaryOp::Sub: return *a - *b;
        case BinaryOp::Mul: return *a * *b;
        case BinaryOp::Div: if (*b == 0) return std::nullopt; return *a / *b;
        case BinaryOp::Mod: if (*b == 0) return std::nullopt; return *a % *b;
        case BinaryOp::Eq:  return (int64_t)(*a == *b);
        case BinaryOp::NEq: return (int64_t)(*a != *b);
        case BinaryOp::Lt:  return (int64_t)(*a < *b);
        case BinaryOp::Gt:  return (int64_t)(*a > *b);
        case BinaryOp::LEq: return (int64_t)(*a <= *b);
        case BinaryOp::GEq: return (int64_t)(*a >= *b);
        case BinaryOp::LogAnd: return (int64_t)(*b != 0);
        case BinaryOp::LogOr:  return (int64_t)(*b != 0);
        default: return std::nullopt;
        }
    }
    default: return std::nullopt;
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

    ++loopDepth_;
    checkStmt(*s.body, fn);
    --loopDepth_;
}

void Sema::checkFor(ForStmt &s, FunctionDecl &fn) {
    pushScope();
    if (s.init) checkStmt(*s.init, fn);
    if (s.cond) checkExpr(*s.cond);
    if (s.incr) checkExpr(*s.incr);

    ++loopDepth_;
    checkStmt(*s.body, fn);
    --loopDepth_;
    popScope();
}

void Sema::checkReturn(ReturnStmt &s, FunctionDecl &fn) {
    if (s.value) {
        if (fn.returnType) adoptTupleLiteral(s.value, fn.returnType);
        // The return type is the target a generic call infers from, the same
        // as a declared variable's ('return Vec::new();').
        TypePtr savedExpectedType = expectedType_;
        if (fn.returnType && !fn.returnType->isError() && fn.returnType->kind != TypeKind::Void)
            expectedType_ = fn.returnType;
        TypePtr valTy = checkExpr(*s.value);
        expectedType_ = savedExpectedType;
        if (fn.returnType) valTy = coerceGenericFnValue(*s.value, valTy, fn.returnType);

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
                    "' to '" + fn.returnType->str() + "'" +
                    rawToRefHint(valTy, fn.returnType));
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
        if (s.declType) adoptTupleLiteral(s.init, resolveType(s.declType));
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

        if (s.declType) initTy = coerceGenericFnValue(*s.init, initTy, s.declType);
        if (s.declType && !s.declType->isError() && !initTy->isError()) {
            if (!canImplicitlyConvert(initTy, s.declType) &&
                !intLiteralFitsType(*s.init, s.declType) &&
                !floatLitFitsType(*s.init, s.declType)) {
                diag_.error(s.loc,
                    "type mismatch in variable declaration: cannot convert '"
                    + initTy->str() + "' to '" + s.declType->str() + "'" +
                    rawToRefHint(initTy, s.declType));
            }
        }
    }

    // Resolve type from initializer if none declared
    if (!s.declType || s.declType->isError()) {
        s.declType = initTy ? initTy : makeError();
    }
    s.resolvedType = s.isVolatile ? withVolatileElements(s.declType) : s.declType;

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
    vd->isVolatile  = s.isVolatile;
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
    s.decl = vd;

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

// Instantiates a generic function for 'subs' (cached), registering and
// type-checking the clone the first time.
FunctionDecl *Sema::instantiateGeneric(FunctionDecl *fnDecl, TypeSubst &subs) {
        MonoKey key = makeMonoKey(fnDecl->name, subs, fnDecl->genericParams);
    if (!monoCache_.count(key)) {
        // Create a concrete clone
        auto clone  = cloneFunctionDecl(*fnDecl, subs);
        clone->name = mangleName(fnDecl->name, subs, fnDecl->genericParams);
        clone->genericParams.clear();
        clone->isMonomorphized = true;
        // Substitution can leave generic-struct references symbolic
        // ('struct Vec<int>'): resolve them to the concrete instance now.
        if (clone->returnType) clone->returnType = resolveType(clone->returnType);
        for (auto &p : clone->params) if (p.type) p.type = resolveType(p.type);

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
        // Cache the instantiation *before* checking its body: a
        // recursive call inside it (directly or through another
        // generic) must resolve to this same clone instead of
        // instantiating it again, forever.
        FunctionDecl *cloneRaw = clone.get();
        monoCache_[key] = cloneRaw;
        monoFunctions_.push_back(std::move(clone));

        bool savedPure = checkingPure_;
        int savedLoopDepth = loopDepth_;
        auto savedLabels = functionLabels_;
        checkFunction(*cloneRaw);
        checkingPure_ = savedPure;
        loopDepth_ = savedLoopDepth;
        functionLabels_ = savedLabels;
    }
    return monoCache_[key];
}

// A generic function used as a value where a function type is expected
// ('fn void(&Box<F16>) f = drop_box;', an argument, a return value, a
// field): its type arguments come from that function type and 'e' is
// redirected to the instantiation. Returns e's (possibly new) type.
TypePtr Sema::coerceGenericFnValue(Expr &e, TypePtr exprTy, const TypePtr &target) {
    if (!target || e.kind != ExprKind::Ident) return exprTy;
    auto &ident = static_cast<IdentExpr &>(e);
    FunctionDecl *fnDecl = ident.resolvedFn;
    if (!fnDecl || fnDecl->genericParams.empty()) return exprTy;
    TypePtr t = target;
    if (t->kind == TypeKind::Optional) t = static_cast<OptionalType &>(*t).inner;
    if (t->kind == TypeKind::Reference) t = static_cast<ReferenceType &>(*t).base;
    if (t->kind == TypeKind::Pointer) t = static_cast<PointerType &>(*t).base;
    if (t->kind != TypeKind::Function) return exprTy;
    auto &ft = static_cast<FunctionType &>(*t);
    if (ft.paramTypes.size() != fnDecl->params.size()) return exprTy;
    TypeSubst subs;
    for (size_t i = 0; i < ft.paramTypes.size(); ++i)
        if (!matchType(fnDecl->params[i].type, ft.paramTypes[i], subs)) return exprTy;
    if (fnDecl->returnType && ft.returnType) matchType(fnDecl->returnType, ft.returnType, subs);
    for (auto &gp : fnDecl->genericParams) {
        auto it = subs.find(gp.name);
        if (it == subs.end()) {
            diag_.error(e.loc, "cannot infer type arguments for generic function '" + fnDecl->name +
                        "' from '" + target->str() + "'");
            return makeError();
        }
        bool failed = false;
        for (auto &constraint : gp.constraints)
            if (!satisfiesTrait(it->second, constraint)) {
                diag_.error(e.loc, constraintError(it->second, constraint, "for generic parameter '" + gp.name + "'"));
                failed = true;
            }
        if (failed) return makeError();
    }
    FunctionDecl *monoFn = instantiateGeneric(fnDecl, subs);
    ident.resolvedFn = monoFn;
    ident.name = monoFn->name;
    auto *sym = lookup(monoFn->name);
    TypePtr ty = sym && sym->type ? sym->type : exprTy;
    e.type = ty;
    return ty;
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
            // 'T::m' / 'ns::T::m': a static method. Methods are keyed by the
            // plain struct name, so drop any namespace qualification.
            std::string key = e.name;
            size_t last = key.rfind("::");
            size_t prev = last == 0 ? std::string::npos : key.rfind("::", last - 1);
            if (prev != std::string::npos) key = key.substr(prev + 2);
            // 'Alias::m' where Alias is a typedef: of a plain struct, the
            // struct's static method; of a generic struct's instance
            // ('typedef struct Box<F32> FloatBox;'), the template's static
            // method instantiated with the instance's type arguments.
            {
                size_t sep = key.find("::");
                std::string prefix = key.substr(0, sep);
                std::string sname;
                auto tit = typeRegistry_.find(prefix);
                if (tit != typeRegistry_.end() && tit->second && tit->second->kind == TypeKind::Struct)
                    sname = static_cast<StructType &>(*tit->second).name;
                else if (structInstanceOrigin_.count(prefix))
                    sname = prefix;   // 'T::m' in a generic body, T bound to an instance
                if (!sname.empty()) {
                    std::string mname = key.substr(sep + 2);
                    auto oit = structInstanceOrigin_.find(sname);
                    auto gtit = oit != structInstanceOrigin_.end() ? genericStructTemplates_.find(oit->second.first)
                                                                   : genericStructTemplates_.end();
                    if (gtit != genericStructTemplates_.end()) {
                        auto sit2 = staticMethodRegistry_.find(oit->second.first + "::" + mname);
                        const auto &gps = gtit->second->genericParams;
                        if (sit2 != staticMethodRegistry_.end() &&
                            sit2->second->genericParams.size() == gps.size() &&
                            gps.size() == oit->second.second.size()) {
                            TypeSubst subs;
                            for (size_t i = 0; i < gps.size(); ++i) subs[gps[i].name] = oit->second.second[i];
                            FunctionDecl *mono = instantiateGeneric(sit2->second, subs);
                            e.name = mono->name;
                            e.resolvedFn = mono;
                            auto *sym = lookup(e.name);
                            return sym && sym->type ? sym->type : makeError();
                        }
                    } else if (sname != key.substr(0, sep)) {
                        key = sname + "::" + mname;
                    }
                }
            }
            auto sit = staticMethodRegistry_.find(key);
            if (sit != staticMethodRegistry_.end()) {
                e.name = sit->second->name;
                e.resolvedFn = sit->second;
                auto *sym = lookup(e.name);
                return sym && sym->type ? sym->type : makeError();
            }
            if (methodRegistry_.count(key)) {
                diag_.error(e.loc, "'" + key + "' is not a static method; call it on a value ('x." +
                            key.substr(key.rfind("::") + 2) + "(...)')");
                return makeError();
            }
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
    if (sym->varDecl)   { e.resolved   = sym->varDecl;  e.isLValue = true; }
    // A 'some(r)' binding of a nullable reference is the matched reference
    // itself: writes through it reach the referent.
    else if (sym->kind == SymKind::Variable && sym->type &&
             sym->type->kind == TypeKind::Reference) { e.isLValue = true; }
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
        else if (operandTy->kind == TypeKind::Pointer && !inUnsafeScope())
            diag_.error(e.loc, "raw pointer arithmetic requires 'unsafe' block");
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
    // Element/member/deref expressions designate an object even when it
    // isn't assignable (an element of a 'const &T' buffer): its address is
    // still meaningful, it just yields a const reference.
    bool designatesObject = e.operand->kind == ExprKind::Subscript ||
                            e.operand->kind == ExprKind::Member ||
                            e.operand->kind == ExprKind::Arrow;
    if (!e.operand->isLValue && !designatesObject) {
        diag_.error(e.loc, "address-of operator requires lvalue");
    }
    bool resultMut = e.operand->isLValue;
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
    // The same walk continues through subscripts of an array ('&g_arr[i]'
    // is as static as g_arr), and stops at an element reached *through* a
    // reference or pointer: that element lives wherever the referent does,
    // i.e. the reference's own region (region-less for a pointer).
    VarDecl *baseVd = nullptr;
    bool haveRegion = false;
    Region r = Region::Stack;
    for (Expr *cur = e.operand.get(); cur; ) {
        if (cur->kind == ExprKind::Ident) { baseVd = static_cast<IdentExpr *>(cur)->resolved; break; }
        if (cur->kind == ExprKind::Member) {
            // A field reached through a reference lives wherever the
            // referent does. 'self' is typed '&stack T' only so methods can
            // use it freely; the receiver itself can be anywhere, so a field
            // of it is region-less, like any other '&T' referent.
            Expr *b = static_cast<MemberExpr *>(cur)->base.get();
            if (b->type && b->type->kind == TypeKind::Reference) {
                bool isSelf = b->kind == ExprKind::Ident &&
                              static_cast<IdentExpr *>(b)->name == "self";
                r = isSelf ? Region::Extern : static_cast<ReferenceType &>(*b->type).region;
                haveRegion = true; break;
            }
            if (b->type && b->type->kind == TypeKind::Pointer) { r = Region::Extern; haveRegion = true; break; }
            cur = b;
            continue;
        }
        if (cur->kind == ExprKind::Subscript) {
            Expr *b = static_cast<SubscriptExpr *>(cur)->base.get();
            if (b->type && b->type->kind == TypeKind::Reference) {
                r = static_cast<ReferenceType &>(*b->type).region; haveRegion = true; break;
            }
            if (b->type && b->type->kind == TypeKind::Pointer) { r = Region::Extern; haveRegion = true; break; }
            cur = b;
            continue;
        }
        break;
    }
    if (!haveRegion)
        r = (baseVd && (baseVd->isStatic || baseVd->isGlobal)) ? Region::Static : Region::Stack;
    // '&x' of a volatile lvalue is a reference to volatile, as in C.
    return makeReference(std::move(operandTy), r, false, resultMut, {},
                         isVolatileLValue(*e.operand));
}

bool Sema::isVolatileLValue(const Expr &e) const {
    switch (e.kind) {
    case ExprKind::Ident: {
        auto *vd = static_cast<const IdentExpr &>(e).resolved;
        return vd && vd->isVolatile;
    }
    case ExprKind::Deref:
        return pointeeIsVolatile(static_cast<const UnaryExpr &>(e).operand->type);
    case ExprKind::Unary: {
        auto &ue = static_cast<const UnaryExpr &>(e);
        return ue.op == UnaryOp::ForceUnwrap && pointeeIsVolatile(ue.operand->type);
    }
    case ExprKind::Subscript: {
        auto &se = static_cast<const SubscriptExpr &>(e);
        if (!se.base->type) return false;
        if (pointeeIsVolatile(se.base->type)) return true;
        return se.base->type->kind == TypeKind::Array && isVolatileLValue(*se.base);
    }
    case ExprKind::Member:
    case ExprKind::Arrow: {
        auto &me = static_cast<const MemberExpr &>(e);
        if (!me.base->type) return false;
        TypePtr bty = me.base->type;
        bool indirect = me.isArrow || bty->kind == TypeKind::Pointer ||
                        bty->kind == TypeKind::Reference;
        if (indirect && pointeeIsVolatile(bty)) return true;
        if (bty->kind == TypeKind::Pointer)   bty = static_cast<const PointerType &>(*bty).base;
        if (bty->kind == TypeKind::Reference) bty = static_cast<const ReferenceType &>(*bty).base;
        if (bty && bty->kind == TypeKind::Struct) {
            std::vector<int> path;
            auto *fd = static_cast<const StructType &>(*bty).findFieldPath(me.field, path);
            if (fd && fd->isVolatile) return true;
        }
        return !indirect && isVolatileLValue(*me.base);
    }
    default:
        return false;
    }
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
            // Raw pointer arithmetic requires 'unsafe': it's how an
            // out-of-bounds address is made, and safe code may only reach
            // memory through references (see checkCast for the matching
            // pointer-forging casts).
            if (lhsTy->kind == TypeKind::Pointer || rhsTy->kind == TypeKind::Pointer) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "raw pointer arithmetic requires 'unsafe' block");
                // 'p - q' is the element distance (C's ptrdiff_t), not a pointer.
                if (e.op == BinaryOp::Sub && lhsTy->kind == TypeKind::Pointer &&
                    rhsTy->kind == TypeKind::Pointer)
                    return makeInt(64, true);
                return lhsTy->kind == TypeKind::Pointer ? lhsTy : rhsTy;
            }
            // Array arithmetic: T[N] + int → T* (array decays to pointer)
            if (lhsTy->kind == TypeKind::Array && rhsTy->isInteger()) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "array pointer arithmetic requires 'unsafe' block");
                return makePointer(static_cast<ArrayType &>(*lhsTy).element, false,
                                   static_cast<ArrayType &>(*lhsTy).elementVolatile);
            }
            if (rhsTy->kind == TypeKind::Array && lhsTy->isInteger()) {
                if (!inUnsafeScope())
                    diag_.error(e.loc, "array pointer arithmetic requires 'unsafe' block");
                return makePointer(static_cast<ArrayType &>(*rhsTy).element, false,
                                   static_cast<ArrayType &>(*rhsTy).elementVolatile);
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
    if (e.arraySize) {
        TypePtr sizeTy = checkExpr(*e.arraySize);
        if (sizeTy && !sizeTy->isError() && !sizeTy->isInteger()) {
            diag_.error(e.arraySize->loc,
                "'new' array size must be an integer, got '" + sizeTy->str() + "'");
        }
    }
    // Returns a reference to the allocated type in the specified arena
    Region r = e.regionName.empty() ? Region::Heap : Region::Arena;
    return makeReference(e.allocType, r, false, true, e.regionName);
}

TypePtr Sema::checkTupleLit(TupleLitExpr &e) {
    std::vector<TypePtr> elemTypes;
    for (auto &elem : e.elements)
        elemTypes.push_back(checkExpr(*elem));
    if (e.targetType) {
        // Written against a known tuple type: each element converts to the
        // corresponding member type (literals narrow like any initializer).
        auto &tt = static_cast<TupleType &>(*e.targetType);
        for (size_t i = 0; i < e.elements.size(); ++i) {
            const TypePtr &want = tt.elementTypes[i];
            if (!elemTypes[i]->isError() && !canImplicitlyConvert(elemTypes[i], want) &&
                !intLiteralFitsType(*e.elements[i], want) &&
                !floatLitFitsType(*e.elements[i], want)) {
                diag_.error(e.elements[i]->loc,
                    "tuple element " + std::to_string(i) + ": cannot convert '" +
                    elemTypes[i]->str() + "' to '" + want->str() + "'");
            }
        }
        return e.targetType;
    }
    return makeTuple(std::move(elemTypes));
}

// '(a, b, c)' where a tuple type is expected is a tuple literal, not the C
// comma operator: rewrite the parenthesized comma chain in place. Arity must
// match; otherwise it's left alone (and type-checks as a comma expression).
void Sema::adoptTupleLiteral(ExprPtr &e, const TypePtr &expected) {
    if (!e || !expected || expected->kind != TypeKind::Tuple) return;
    if (e->kind != ExprKind::Binary) return;
    auto &be = static_cast<BinaryExpr &>(*e);
    if (be.op != BinaryOp::Comma || !be.parenComma) return;
    std::vector<ExprPtr> parts;
    std::function<void(ExprPtr &)> flatten = [&](ExprPtr &x) {
        if (x->kind == ExprKind::Binary) {
            auto &b = static_cast<BinaryExpr &>(*x);
            if (b.op == BinaryOp::Comma && (&x == &e || !b.parenComma)) {
                flatten(b.left);
                parts.push_back(std::move(b.right));
                return;
            }
        }
        parts.push_back(std::move(x));
    };
    auto &tt = static_cast<TupleType &>(*expected);
    // Count first, so a mismatched arity leaves the expression untouched.
    size_t n = 1;
    for (Expr *x = e.get(); x->kind == ExprKind::Binary &&
         static_cast<BinaryExpr *>(x)->op == BinaryOp::Comma &&
         (x == e.get() || !static_cast<BinaryExpr *>(x)->parenComma);
         x = static_cast<BinaryExpr *>(x)->left.get())
        ++n;
    if (n != tt.elementTypes.size()) return;
    SourceLocation loc = e->loc;
    flatten(e);
    auto lit = std::make_unique<TupleLitExpr>(std::move(parts), loc);
    lit->targetType = expected;
    e = std::move(lit);
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
            if (fieldTy) initTy = coerceGenericFnValue(*e.inits[i], initTy, fieldTy);
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

// Calls into C ('extern' functions): an integer argument converts
// implicitly to a pointer-width integer parameter — C's own size_t
// conversion. std declares libc's size_t parameters with the pointer-width
// size_t (32-bit on 32-bit targets) while SafeC's 'unsigned long' is always
// 64-bit, so 'memcpy(d, s, n)' with an 'unsigned long n' must still be
// accepted there. Only at the C boundary, only for that exact width.
bool Sema::cAbiSizeArgCompatible(CallExpr &e, const TypePtr &arg, const TypePtr &param) const {
    if (!arg || !param || !arg->isInteger() || !param->isInteger()) return false;
    if (static_cast<const PrimType &>(*param).bitWidth() != pointerBits_) return false;
    if (!e.callee || e.callee->kind != ExprKind::Ident) return false;
    auto *fn = static_cast<IdentExpr &>(*e.callee).resolvedFn;
    return fn && fn->isExtern;
}

TypePtr Sema::checkCall(CallExpr &e) {
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
        // ── Calls the heap/arena lifetime analysis treats as primitives
        // (see CallExpr::memOp and MemoryFlow.cpp): std::dealloc() and its
        // raw form free their argument; arena_reset<R>()/arena_destroy<R>()
        // invalidate every '&arena<R> T' reference, arena_mark<R>() takes a
        // checkpoint and arena_free_to<R>(mark) invalidates exactly the
        // references allocated after that checkpoint. These still resolve
        // to real functions (see collectRegion) and are type-checked below
        // like any other call.
        {
            static const std::string kResetPrefix   = "__arena_reset_";
            static const std::string kDestroyPrefix = "__arena_destroy_";
            static const std::string kMarkPrefix    = "__arena_mark_";
            static const std::string kFreeToPrefix  = "__arena_free_to_";
            auto memOp = [&](CallExpr::MemOp op, size_t prefixLen) {
                e.memOp = op;
                e.memRegion = ident.name.substr(prefixLen);
            };
            if (isDeallocCallName(ident.name) && !e.args.empty())
                e.memOp = CallExpr::MemOp::Dealloc;
            else if (ident.name.rfind(kResetPrefix, 0) == 0)   memOp(CallExpr::MemOp::ArenaReset, kResetPrefix.size());
            else if (ident.name.rfind(kDestroyPrefix, 0) == 0) memOp(CallExpr::MemOp::ArenaDestroy, kDestroyPrefix.size());
            else if (ident.name.rfind(kMarkPrefix, 0) == 0)    memOp(CallExpr::MemOp::ArenaMark, kMarkPrefix.size());
            else if (ident.name.rfind(kFreeToPrefix, 0) == 0)  memOp(CallExpr::MemOp::ArenaFreeTo, kFreeToPrefix.size());
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
                // bool atomic_cas(T *ptr, T expected, T desired[, order]):
                // 'expected' is a value, compared against *ptr and never
                // written back. For C11's write-back semantics (expected
                // passed by address) use atomic_compare_exchange_strong.
                if (e.args.size() != 3 && e.args.size() != 4)
                    diag_.error(e.loc, "atomic_cas expects 3 arguments (pointer, expected, desired) "
                                       "and an optional memory order");
                TypePtr ptrTy = e.args.empty() ? nullptr : e.args[0]->type;
                TypePtr elemTy;
                if (ptrTy && ptrTy->kind == TypeKind::Pointer)
                    elemTy = static_cast<PointerType &>(*ptrTy).base;
                else if (ptrTy && ptrTy->kind == TypeKind::Reference)
                    elemTy = static_cast<ReferenceType &>(*ptrTy).base;
                else if (ptrTy && !ptrTy->isError())
                    diag_.error(e.args[0]->loc, "atomic_cas: first argument must be a pointer, not '" +
                                ptrTy->str() + "'");
                for (size_t i = 1; elemTy && i < 3 && i < e.args.size(); ++i) {
                    const TypePtr &argTy = e.args[i]->type;
                    if (!argTy || argTy->isError() || elemTy->isError()) continue;
                    if (canImplicitlyConvert(argTy, elemTy) ||
                        intLiteralFitsType(*e.args[i], elemTy) ||
                        floatLitFitsType(*e.args[i], elemTy))
                        continue;
                    std::string hint;
                    if (i == 1 && (argTy->kind == TypeKind::Pointer ||
                                   argTy->kind == TypeKind::Reference))
                        hint = " (atomic_cas takes 'expected' by value; use "
                               "atomic_compare_exchange_strong(ptr, &expected, desired) "
                               "to have the current value written back on failure)";
                    diag_.error(e.args[i]->loc,
                        std::string("atomic_cas: cannot convert ") +
                        (i == 1 ? "'expected'" : "'desired'") + " of type '" +
                        argTy->str() + "' to '" + elemTy->str() + "'" + hint);
                }
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
        // __builtin_fmuladd(a, b, c): a*b + c, fused into one FMA where the
        // target has one and as a separate multiply and add where it doesn't
        // (llvm.fmuladd). For float, double and vec<float/double, N>.
        // __builtin_sqrt(x): IEEE square root (llvm.sqrt), one instruction
        // on targets with a hardware sqrt, and vectorizable in loops. For
        // float, double and vec<float/double, N>.
        if (ident.name == "__builtin_sqrt") {
            for (auto &a : e.args) checkExpr(*a);
            TypePtr t = e.args.size() == 1 ? e.args[0]->type : nullptr;
            bool ok = t && (t->isFloat() ||
                (t->kind == TypeKind::Vector &&
                 static_cast<VectorType &>(*t).element->isFloat()));
            if (!ok) {
                diag_.error(e.loc, "__builtin_sqrt takes one float, double or vec<float/double, N>");
                e.type = makeError();
                return e.type;
            }
            e.type = t;
            return t;
        }
        if (ident.name == "__builtin_fmuladd") {
            for (auto &a : e.args) checkExpr(*a);
            if (e.args.size() != 3) {
                diag_.error(e.loc, "__builtin_fmuladd takes 3 arguments");
                e.type = makeError();
                return e.type;
            }
            TypePtr t = e.args[0]->type;
            bool ok = t && (t->isFloat() ||
                (t->kind == TypeKind::Vector &&
                 static_cast<VectorType &>(*t).element->isFloat()));
            for (int i = 1; ok && i < 3; ++i)
                ok = e.args[i]->type && e.args[i]->type->equals(*t);
            if (!ok) {
                diag_.error(e.loc, "__builtin_fmuladd: arguments must all have one type: "
                                   "float, double or vec<float/double, N>");
                e.type = makeError();
                return e.type;
            }
            e.type = t;
            return t;
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
                if (ident.name == *p && !providedFns_.count(ident.name)) {
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
                if (methodFn->isStaticMethod) {
                    diag_.error(e.loc, "'" + mem.field + "' is a static method of '" + st.name +
                                "'; call it as '" + st.name + "::" + mem.field + "(...)'");
                    for (auto &a : e.args) checkExpr(*a);
                    return makeError();
                }
                // An allocator's 'a.dealloc(p)' frees 'p' (see isDeallocCallName).
                if (mem.field == "dealloc" && !e.args.empty()) e.memOp = CallExpr::MemOp::Dealloc;
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
        // A generic function passed as a value ('link_(&n, back_relu_)') has
        // no concrete type yet: infer from the other arguments, then
        // instantiate it for its parameter's type (below).
        std::vector<TypePtr> inferTypes = argTypes;
        for (size_t i = 0; i < e.args.size(); ++i)
            if (e.args[i]->kind == ExprKind::Ident) {
                auto *rf = static_cast<IdentExpr &>(*e.args[i]).resolvedFn;
                if (rf && !rf->genericParams.empty()) inferTypes[i] = nullptr;
            }

        TypeSubst subs;
        // x.m(args) on a generic method: self comes from the base, not args
        std::vector<ParamDecl> inferParams = fnDecl->params;
        size_t selfSkip = 0;
        if (e.methodBase && !fnDecl->isStaticMethod && !inferParams.empty()) {
            inferParams.erase(inferParams.begin());
            selfSkip = 1;
        }
        if (!inferTypeArgs(inferParams, inferTypes, fnDecl->genericParams, subs,
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
        bool boundFailed = false;
        for (auto &gp : fnDecl->genericParams) {
            if (gp.constraints.empty()) continue;
            auto it = subs.find(gp.name);
            if (it == subs.end()) continue;
            for (auto &constraint : gp.constraints) {
                if (!satisfiesTrait(it->second, constraint)) {
                    diag_.error(e.loc, constraintError(it->second, constraint, "for generic parameter '" + gp.name + "'"));
                    boundFailed = true;
                }
            }
        }
        // Don't instantiate past a failed bound (its body would only repeat
        // the missing methods as errors).
        if (boundFailed) return makeError();

        FunctionDecl *monoFn = instantiateGeneric(fnDecl, subs);
        for (size_t i = 0; i < e.args.size() && i + selfSkip < monoFn->params.size(); ++i)
            if (!inferTypes[i])
                argTypes[i] = coerceGenericFnValue(*e.args[i], argTypes[i], monoFn->params[i + selfSkip].type);
        // Redirect the call to the monomorphized version
        if (e.callee->kind == ExprKind::Ident) {
            auto &ident    = static_cast<IdentExpr &>(*e.callee);
            ident.resolvedFn = monoFn;
            ident.name       = monoFn->name;
        }
        // Inference only proves the arguments *match the pattern*; each one
        // must still convert to its instantiated parameter type, exactly as
        // for a non-generic call (a struct value can't bind to '&T', etc.).
        bool hasPackParam = false;
        for (auto &gp : fnDecl->genericParams) if (gp.isPack) hasPackParam = true;
        if (!hasPackParam && !e.methodBase) {
            for (size_t i = 0; i < e.args.size() && i < monoFn->params.size(); ++i) {
                const TypePtr &paramTy = monoFn->params[i].type;
                const TypePtr &argTy = argTypes[i];
                if (!paramTy || !argTy || argTy->isError() || paramTy->isError()) continue;
                if (!canImplicitlyConvert(argTy, paramTy) &&
                    !cAbiSizeArgCompatible(e, argTy, paramTy) &&
                    !intLiteralFitsType(*e.args[i], paramTy) &&
                    !floatLitFitsType(*e.args[i], paramTy) &&
                    !refToPointerArgCompatible(argTy, paramTy)) {
                    diag_.error(e.args[i]->loc,
                        "argument " + std::to_string(i+1) + ": cannot convert '" +
                        argTy->str() + "' to '" + paramTy->str() + "'" +
                        rawToRefHint(argTy, paramTy));
                }
            }
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
        size_t  paramIdx = i + selfOffset;
        if (paramIdx < expected) adoptTupleLiteral(e.args[i], ft->paramTypes[paramIdx]);
        TypePtr argTy = checkExpr(*e.args[i]);
        if (paramIdx < expected) argTy = coerceGenericFnValue(*e.args[i], argTy, ft->paramTypes[paramIdx]);
        if (paramIdx < expected &&
                !argTy->isError() && !ft->paramTypes[paramIdx]->isError()) {
            if (!canImplicitlyConvert(argTy, ft->paramTypes[paramIdx]) &&
                !cAbiSizeArgCompatible(e, argTy, ft->paramTypes[paramIdx]) &&
                !intLiteralFitsType(*e.args[i], ft->paramTypes[paramIdx]) &&
                !floatLitFitsType(*e.args[i], ft->paramTypes[paramIdx]) &&
                !refToPointerArgCompatible(argTy, ft->paramTypes[paramIdx])) {
                diag_.error(e.args[i]->loc,
                    "argument " + std::to_string(i+1) + ": cannot convert '" +
                    argTy->str() + "' to '" + ft->paramTypes[paramIdx]->str() + "'" +
                    rawToRefHint(argTy, ft->paramTypes[paramIdx]));
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

    // A cast can't drop nullability: '(&T)q' on a '?&T' would skip the null
    // check nullable references always require (in safe and unsafe code
    // alike). '(?&T)q' — keeping it nullable — is fine. Narrow with
    // 'match (q) { case some(r): ... }', which binds r as '&T'.
    if (srcTy && srcTy->kind == TypeKind::Reference &&
        static_cast<ReferenceType &>(*srcTy).nullable &&
        e.targetType->kind == TypeKind::Reference &&
        !static_cast<ReferenceType &>(*e.targetType).nullable) {
        diag_.error(e.loc,
            "cannot cast nullable reference '" + srcTy->str() + "' to non-null '" +
            e.targetType->str() + "': check it with 'match (x) { case null: ... "
            "case some(r): ... }' (r is the non-null reference)");
    }

    // In safe code, cast from reference to pointer is not allowed — except
    // a '&static fn(...)' function reference (the type a bare function
    // name always evaluates to; see checkIdent's SymKind::Function case).
    // The whole point of the reference-to-pointer restriction is dangling/
    // aliasing safety for *data* — a function's code address is neither
    // freed nor moved for the life of the program, so '(void*)someFunc' (a
    // function-pointer table entry, an ISR vector table, etc.) carries none
    // of the risk 'unsafe' exists to flag.
    if (!inUnsafeScope() && srcTy && !srcTy->isError()) {
        // Raw pointer → reference: '(&T)q' asserts q is non-null and valid
        // with nothing checking it, so it's 'unsafe' only. '(?&T)q' is
        // allowed: it keeps the null case, which must be matched away
        // before the reference can be used, and its region stays open
        // ('?&T', no region keyword).
        if (srcTy->kind == TypeKind::Pointer &&
            e.targetType->kind == TypeKind::Reference &&
            !static_cast<ReferenceType &>(*e.targetType).nullable) {
            diag_.error(e.loc,
                "cast from raw pointer to non-null reference '" + e.targetType->str() +
                "' requires 'unsafe' block; use '(?" + e.targetType->str() +
                ")' and check it with 'match'");
        }
        // Making a raw pointer out of an integer, or reinterpreting one
        // pointer type as another, forges an address/type the compiler
        // can't vouch for. Widening to 'void*' (or adding const) and the
        // 'null' literal are the safe exceptions.
        if (e.targetType->kind == TypeKind::Pointer) {
            auto &tp = static_cast<PointerType &>(*e.targetType);
            // 'null' and C's null pointer constant '(T*)0'.
            bool isNullLit = e.operand->kind == ExprKind::NullLit ||
                (e.operand->kind == ExprKind::IntLit &&
                 static_cast<IntLitExpr &>(*e.operand).value == 0);
            if (srcTy->isInteger() && !isNullLit) {
                diag_.error(e.loc, "cast from integer to raw pointer requires 'unsafe' block");
            } else if (srcTy->kind == TypeKind::Pointer && !isNullLit) {
                auto &sp = static_cast<PointerType &>(*srcTy);
                if (!tp.base->isVoid() && !typeEqual(sp.base, tp.base))
                    diag_.error(e.loc,
                        "cast between raw pointer types '" + srcTy->str() + "' and '" +
                        e.targetType->str() + "' requires 'unsafe' block");
            }
        }
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
    if (e.op == AssignOp::Assign && e.lhs->kind == ExprKind::Ident) {
        auto &ident = static_cast<IdentExpr &>(*e.lhs);
        if (auto *sym = lookup(ident.name))
            if (sym->kind == SymKind::Variable) sym->initialized = true;
    }
    TypePtr lhsTy = checkExpr(*e.lhs);
    // See expectedType_'s doc comment: 'p = std::vec_at(&v, 1UL);' (p
    // already declared 'int*') gets the same fallback hint a var-decl's
    // explicit type does. Simple assignment only.
    TypePtr savedExpectedType = expectedType_;
    if (e.op == AssignOp::Assign && lhsTy && !lhsTy->isError())
        expectedType_ = lhsTy;
    if (e.op == AssignOp::Assign) adoptTupleLiteral(e.rhs, lhsTy);
    TypePtr rhsTy = checkExpr(*e.rhs);
    expectedType_ = savedExpectedType;
    rhsTy = coerceGenericFnValue(*e.rhs, rhsTy, lhsTy);

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

    // Pointer compound arithmetic 'p += n' / 'p -= n' — C semantics (scaled
    // by the element size); 'unsafe' only, like 'p + n'.
    bool ptrStep = lhsTy->kind == TypeKind::Pointer && rhsTy->isInteger() &&
                   (e.op == AssignOp::AddAssign || e.op == AssignOp::SubAssign);
    if (ptrStep && !inUnsafeScope())
        diag_.error(e.loc, "raw pointer arithmetic requires 'unsafe' block");

    // Type compatibility
    if (!lhsTy->isError() && !rhsTy->isError() && !ptrStep) {
        if (!canImplicitlyConvert(rhsTy, lhsTy) &&
            !intLiteralFitsType(*e.rhs, lhsTy) &&
            !floatLitFitsType(*e.rhs, lhsTy)) {
            diag_.error(e.loc,
                "type mismatch in assignment: cannot convert '" +
                rhsTy->str() + "' to '" + lhsTy->str() + "'" + rawToRefHint(rhsTy, lhsTy));
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
    // A raw pointer passed where a reference is expected ('const &T' given
    // a 'const double*'): infer from the pointee, as a non-generic call
    // accepts the pointer there; the instantiated parameter is then checked
    // like any other argument.
    if (paramTy->kind == TypeKind::Reference && argTy->kind == TypeKind::Pointer)
        return matchType(static_cast<const ReferenceType &>(*paramTy).base,
                         static_cast<const PointerType &>(*argTy).base, subs);
    if (paramTy->kind != argTy->kind) return false;
    switch (paramTy->kind) {
    case TypeKind::Optional:
        return matchType(static_cast<const OptionalType &>(*paramTy).inner,
                         static_cast<const OptionalType &>(*argTy).inner, subs);
    case TypeKind::Slice:
        return matchType(static_cast<const SliceType &>(*paramTy).element,
                         static_cast<const SliceType &>(*argTy).element, subs);
    case TypeKind::Vector:
        return matchType(static_cast<const VectorType &>(*paramTy).element,
                         static_cast<const VectorType &>(*argTy).element, subs);
    case TypeKind::Tuple: {
        auto &pt = static_cast<const TupleType &>(*paramTy);
        auto &at = static_cast<const TupleType &>(*argTy);
        if (pt.elementTypes.size() != at.elementTypes.size()) return false;
        for (size_t i = 0; i < pt.elementTypes.size(); ++i)
            if (!matchType(pt.elementTypes[i], at.elementTypes[i], subs)) return false;
        return true;
    }
    case TypeKind::Function: {
        auto &pf = static_cast<const FunctionType &>(*paramTy);
        auto &af = static_cast<const FunctionType &>(*argTy);
        if (pf.paramTypes.size() != af.paramTypes.size() || pf.variadic != af.variadic) return false;
        if (!matchType(pf.returnType, af.returnType, subs)) return false;
        for (size_t i = 0; i < pf.paramTypes.size(); ++i)
            if (!matchType(pf.paramTypes[i], af.paramTypes[i], subs)) return false;
        return true;
    }
    case TypeKind::Struct: {
        // A generic-struct pattern ('struct Vec<T>') against a concrete
        // instance ('Vec_int'): match type arguments through the instance's
        // recorded origin.
        auto &ps = static_cast<const StructType &>(*paramTy);
        auto &as = static_cast<const StructType &>(*argTy);
        if (ps.typeArgs.empty()) return typeEqual(paramTy, argTy);
        const std::vector<TypePtr> *args = nullptr;
        std::string tplName = as.name;
        if (!as.typeArgs.empty()) {
            args = &as.typeArgs;
        } else {
            auto it = structInstanceOrigin_.find(as.name);
            if (it == structInstanceOrigin_.end()) return false;
            tplName = it->second.first;
            args = &it->second.second;
        }
        if (tplName != ps.name || args->size() != ps.typeArgs.size()) return false;
        for (size_t i = 0; i < ps.typeArgs.size(); ++i)
            if (!matchType(ps.typeArgs[i], (*args)[i], subs)) return false;
        return true;
    }
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
        if (ty->kind == TypeKind::Optional)
            return self(self, static_cast<const OptionalType &>(*ty).inner);
        if (ty->kind == TypeKind::Slice)
            return self(self, static_cast<const SliceType &>(*ty).element);
        if (ty->kind == TypeKind::Vector)
            return self(self, static_cast<const VectorType &>(*ty).element);
        if (ty->kind == TypeKind::Tuple) {
            for (auto &e : static_cast<const TupleType &>(*ty).elementTypes)
                if (self(self, e)) return true;
            return false;
        }
        if (ty->kind == TypeKind::Function) {
            auto &ft = static_cast<const FunctionType &>(*ty);
            if (self(self, ft.returnType)) return true;
            for (auto &p : ft.paramTypes) if (self(self, p)) return true;
            return false;
        }
        if (ty->kind == TypeKind::Struct) {
            for (auto &a : static_cast<const StructType &>(*ty).typeArgs)
                if (self(self, a)) return true;
            return false;
        }
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
        // A bare 'null' (typed 'void*') carries no type information: let it
        // stand for any nullable reference parameter and bind T from the
        // other arguments ('thread_create(&tid, work, null)'). The call's
        // argument conversion check still runs against the instantiation.
        auto isNullArg = [](const TypePtr &paramTy, const TypePtr &argTy) {
            if (!argTy || argTy->kind != TypeKind::Pointer) return false;
            if (!static_cast<const PointerType &>(*argTy).base->isVoid()) return false;
            return paramTy->kind == TypeKind::Reference &&
                   static_cast<const ReferenceType &>(*paramTy).nullable;
        };
        for (size_t i = 0; i < params.size() && i < argTypes.size(); ++i) {
            if (!argTypes[i]) continue;   // a generic function value: bound after inference
            if (hasGeneric(hasGeneric, params[i].type)) {
                if (isNullArg(params[i].type, argTypes[i])) continue;
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
    if (dropsVolatile(from, to)) return false;
    auto &fr = static_cast<const ReferenceType &>(*from);
    auto &tp = static_cast<const PointerType &>(*to);
    if (typeEqual(fr.base, tp.base)) return true;
    auto is8 = [](const TypePtr &t) {
        return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
               t->kind == TypeKind::UInt8;
    };
    return is8(fr.base) && is8(tp.base);
}

std::string Sema::rawToRefHint(const TypePtr &from, const TypePtr &to) const {
    if (!from || !to || inUnsafeScope()) return "";
    if (from->kind != TypeKind::Pointer || to->kind != TypeKind::Reference) return "";
    auto &tr = static_cast<const ReferenceType &>(*to);
    if (tr.nullable) return "";
    return " (a raw pointer becomes a non-null reference only inside 'unsafe'; "
           "convert it to '?" + to->str() + "' and check it with 'match')";
}

bool Sema::canImplicitlyConvert(const TypePtr &from, const TypePtr &to) const {
    if (!from || !to) return false;
    if (from->isError() || to->isError()) return true; // suppress cascaded errors
    // 'volatile T*' → 'T*' (or a volatile array decaying to 'T*') would let
    // later accesses skip the volatile semantics; that takes an explicit cast.
    if (dropsVolatile(from, to)) return false;
    if (typeEqual(from, to)) return true;
    // A function whose signature differs from the target function type only
    // by raw 'T*' vs region-less '&T'/'?&T' in some parameter/return slots
    // is the same function at the ABI level (both are one pointer), and
    // those two already convert implicitly as values (see the
    // Pointer↔Reference rules below). So a callback written with safe
    // references — 'HttpResponse handler(&HttpRequest req)' — binds to a
    // C-style callback type such as 'fn HttpResponse(HttpRequest*)'.
    auto fnOf = [](const TypePtr &t) -> const FunctionType * {
        if (t->kind == TypeKind::Function) return static_cast<const FunctionType *>(t.get());
        if (t->kind == TypeKind::Reference) {
            auto &r = static_cast<const ReferenceType &>(*t);
            if (r.base && r.base->kind == TypeKind::Function)
                return static_cast<const FunctionType *>(r.base.get());
        }
        return nullptr;
    };
    if (auto *ff = fnOf(from)) if (auto *tf = fnOf(to)) {
        auto pointee = [](const TypePtr &t) -> TypePtr {
            if (!t) return nullptr;
            if (t->kind == TypeKind::Pointer) return static_cast<const PointerType &>(*t).base;
            if (t->kind == TypeKind::Reference) {
                auto &r = static_cast<const ReferenceType &>(*t);
                if (r.region == Region::Extern) return r.base;
            }
            return nullptr;
        };
        auto sameSlot = [&](const TypePtr &a, const TypePtr &b) {
            if (typeEqual(a, b)) return true;
            TypePtr pa = pointee(a), pb = pointee(b);
            if (!pa || !pb) return false;
            if (a->kind == b->kind) return false;   // two refs / two pointers must match exactly
            return typeEqual(pa, pb) || pa->kind == TypeKind::Void || pb->kind == TypeKind::Void;
        };
        if (ff->variadic == tf->variadic && ff->paramTypes.size() == tf->paramTypes.size() &&
            sameSlot(ff->returnType, tf->returnType)) {
            bool ok = true;
            for (size_t i = 0; i < ff->paramTypes.size() && ok; ++i)
                ok = sameSlot(ff->paramTypes[i], tf->paramTypes[i]);
            if (ok) return true;
        }
    }
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
    // Pointer → Reference: a raw pointer (any C-interop pointer, or one
    // that decayed from an array) can become a *nullable* reference
    // anywhere — the null case stays visible and must be matched away
    // before use. Becoming a non-null '&T' asserts the pointer is valid
    // with nothing checking it, so that direction is 'unsafe' only.
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
        // Raw T* → non-null '&T' asserts non-null with no check: 'unsafe'
        // only. T* → '?&T' is fine anywhere (the null case stays visible).
        if (!tr.nullable && !inUnsafeScope()) return false;
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
