#include "safec/ConstEval.h"
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace safec {

// =============================================================================
// ConstValue helpers
// =============================================================================

int64_t ConstValue::toInt() const {
    switch (kind) {
    case Int:   return intVal;
    case Float: return (int64_t)floatVal;
    case Bool:  return boolVal ? 1 : 0;
    default:    return 0;
    }
}

double ConstValue::toFloat() const {
    switch (kind) {
    case Float: return floatVal;
    case Int:   return (double)intVal;
    case Bool:  return boolVal ? 1.0 : 0.0;
    default:    return 0.0;
    }
}

bool ConstValue::toBool() const {
    switch (kind) {
    case Bool:  return boolVal;
    case Int:   return intVal != 0;
    case Float: return floatVal != 0.0;
    default:    return false;
    }
}

std::string ConstValue::typeStr() const {
    switch (kind) {
    case Void:   return "void";
    case Int:    return "int";
    case Float:  return "float";
    case Bool:   return "bool";
    case String: return "string";
    case Array:  return "array";
    case Struct: return "struct";
    }
    return "?";
}

std::string ConstValue::repr() const {
    switch (kind) {
    case Void:   return "<void>";
    case Int:    return std::to_string(intVal);
    case Float:  return std::to_string(floatVal);
    case Bool:   return boolVal ? "true" : "false";
    case String: return '"' + strVal + '"';
    case Array:
    case Struct: {
        std::string s = (kind == Array) ? "[" : "{";
        for (size_t i = 0; i < elems.size(); ++i) {
            if (i) s += ", ";
            s += elems[i].repr();
        }
        s += (kind == Array) ? "]" : "}";
        return s;
    }
    }
    return "?";
}

// =============================================================================
// ConstEvalEngine
// =============================================================================

ConstEvalEngine::ConstEvalEngine(TranslationUnit &tu, DiagEngine &diag)
    : tu_(tu), diag_(diag) {}

ConstValue ConstEvalEngine::zeroValue(const TypePtr &ty) {
    if (!ty) return ConstValue::mkInt(0);
    switch (ty->kind) {
    case TypeKind::Float32:
    case TypeKind::Float64:
        return ConstValue::mkFloat(0.0);
    case TypeKind::Bool:
        return ConstValue::mkBool(false);
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*ty);
        std::vector<ConstValue> fields;
        fields.reserve(st.fields.size());
        for (auto &f : st.fields) fields.push_back(zeroValue(f.type));
        return ConstValue::mkStruct(std::move(fields));
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(*ty);
        int64_t n = at.size > 0 ? at.size : 0;
        std::vector<ConstValue> elems;
        elems.reserve((size_t)n);
        for (int64_t i = 0; i < n; ++i) elems.push_back(zeroValue(at.element));
        return ConstValue::mkArray(std::move(elems));
    }
    default:
        return ConstValue::mkInt(0);
    }
}

// Synthesizes an AST literal from a computed ConstValue, for folding a
// global initializer back into something CodeGen::evalConstInit can lower
// without re-running this engine (see the GlobalVar case in run() below).
// Struct/Array values become a plain positional CompoundInitExpr (empty
// resolvedSlots — slot == source position), which evalConstInit already
// knows how to walk recursively for both kinds.
static ExprPtr constValueToExpr(const ConstValue &v, SourceLocation loc) {
    switch (v.kind) {
    case ConstValue::Bool:
    case ConstValue::Int:
        return std::make_unique<IntLitExpr>(v.toInt(), loc);
    case ConstValue::Float:
        return std::make_unique<FloatLitExpr>(v.floatVal, loc);
    case ConstValue::String:
        return std::make_unique<StringLitExpr>(v.strVal, loc);
    case ConstValue::Struct:
    case ConstValue::Array: {
        std::vector<ExprPtr> inits;
        inits.reserve(v.elems.size());
        for (auto &el : v.elems) inits.push_back(constValueToExpr(el, loc));
        return std::make_unique<CompoundInitExpr>(std::move(inits), loc);
    }
    default:
        return std::make_unique<IntLitExpr>(0, loc);
    }
}

// ── Build function table ──────────────────────────────────────────────────────

FunctionDecl *ConstEvalEngine::lookupFn(const std::string &name) const {
    auto it = fnTable_.find(name);
    return it != fnTable_.end() ? it->second : nullptr;
}

bool ConstEvalEngine::isBudgetExceeded(SourceLocation loc) {
    if (instrBudget_ <= 0) {
        diag_.error(loc, "compile-time evaluation instruction budget exceeded "
                         "(possible infinite loop or too-large computation)");
        return true;
    }
    return false;
}

// =============================================================================
// run() — TU-level const-eval pass
// =============================================================================

void ConstEvalEngine::buildFnTable() {
    for (auto &decl : tu_.decls) {
        if (decl->kind == DeclKind::Function) {
            auto &fn = static_cast<FunctionDecl &>(*decl);
            fnTable_[fn.name] = &fn;
        }
    }
}

bool ConstEvalEngine::run() {
    buildFnTable();

    // Process all top-level declarations
    for (auto &decl : tu_.decls) {
        switch (decl->kind) {
        case DeclKind::StaticAssert: {
            auto &sa = static_cast<StaticAssertDecl &>(*decl);
            if (!sa.cond) break;
            auto val = evalExpr(*sa.cond);
            if (val && !val->toBool()) {
                std::string msg = sa.message.empty()
                    ? "static_assert failed"
                    : "static_assert failed: " + sa.message;
                diag_.error(sa.loc, msg);
            }
            break;
        }
        case DeclKind::GlobalVar: {
            auto &gv = static_cast<GlobalVarDecl &>(*decl);
            if (gv.isConst && gv.init) {
                auto val = evalExpr(*gv.init);
                if (val) {
                    globalConsts_[gv.name] = *val;
                    // Replace the initializer with a folded literal so CodeGen
                    // can emit the constant directly without calling consteval
                    // again — CodeGen::evalConstInit has its own independent
                    // (non-interpreting) constant folder that only handles
                    // literals/negation/global-refs/aggregate *literals*, not
                    // e.g. a call to a consteval function; folding back here
                    // is what makes 'const int arr[3] = make_array();' work
                    // for any kind this engine can produce, not just
                    // Int/Float — including the new Struct/Array kinds.
                    switch (val->kind) {
                    case ConstValue::Int:
                    case ConstValue::Bool:
                        gv.init = std::make_unique<IntLitExpr>(val->toInt(), gv.loc);
                        break;
                    case ConstValue::Float:
                        gv.init = std::make_unique<FloatLitExpr>(val->floatVal, gv.loc);
                        break;
                    case ConstValue::String:
                        gv.init = std::make_unique<StringLitExpr>(val->strVal, gv.loc);
                        break;
                    case ConstValue::Struct:
                    case ConstValue::Array:
                        gv.init = constValueToExpr(*val, gv.loc);
                        break;
                    default: break;
                    }
                }
            }
            break;
        }
        case DeclKind::Function: {
            auto &fn = static_cast<FunctionDecl &>(*decl);
            // Enforce consteval: check body doesn't call non-const functions
            if (fn.isConsteval && fn.body) {
                checkFunctionForConstevalCalls(fn, true);
            }
            // Check non-const functions don't call consteval functions
            if (!fn.isConst && !fn.isConsteval && fn.body) {
                checkFunctionForConstevalCalls(fn, false);
            }
            // Resolve if const statements within function bodies
            if (fn.body) {
                resolveIfConstInStmt(*fn.body);
            }
            break;
        }
        default:
            break;
        }
    }

    return !diag_.hasErrors();
}

// =============================================================================
// resolveArraySizes() — fills in ArrayType::size for deferred size expressions
// (named constants / consteval calls) before Sema runs.
// =============================================================================

bool ConstEvalEngine::resolveTypeArraySizes(const TypePtr &ty, bool allowVLA) {
    if (!ty) return true;
    bool ok = true;
    switch (ty->kind) {
    case TypeKind::Array: {
        auto &at = static_cast<ArrayType &>(*ty);
        // Element first, so nested arrays (e.g. 'T arr[square(2)][3]') resolve
        // inside-out the same way normal type-checking would visit them.
        ok = resolveTypeArraySizes(at.element) && ok;
        if (at.sizeExpr) {
            auto *sizeExpr = static_cast<const Expr *>(at.sizeExpr);
            // Speculative when allowVLA: silence diagnostics for the
            // attempt (not just their effect on the error count — emit()
            // prints to stderr immediately, so discardSince() alone would
            // leave a misleading "not a compile-time constant" message on
            // screen for what turns out to be a perfectly legal VLA).
            size_t mark = diag_.checkpoint();
            bool wasSilent = diag_.isSilent();
            if (allowVLA) diag_.setSilent(true);
            auto val = evalExpr(*sizeExpr);
            if (allowVLA) diag_.setSilent(wasSilent);
            if (!val) {
                // Not foldable to a compile-time constant. If this is a
                // local variable's array type inside 'unsafe{}', treat it
                // as a variable-length array instead of an error: discard
                // whatever diagnostic evalExpr emitted (it doesn't apply —
                // this isn't a constant-expression context after all) and
                // leave sizeExpr set for Sema/CodeGen to check/evaluate at
                // runtime, at the declaration's actual scope.
                if (allowVLA) {
                    diag_.discardSince(mark);
                    at.size = -2;
                    break;
                }
                ok = false;
                break;
            }
            if (val->toInt() < 0) {
                diag_.error(sizeExpr->loc, "array size must be non-negative");
                ok = false;
                break;
            }
            at.size = val->toInt();
            at.sizeExpr = nullptr;
        }
        break;
    }
    case TypeKind::Pointer:
        ok = resolveTypeArraySizes(static_cast<PointerType &>(*ty).base);
        break;
    case TypeKind::Reference:
        ok = resolveTypeArraySizes(static_cast<ReferenceType &>(*ty).base);
        break;
    default:
        break;
    }
    return ok;
}

bool ConstEvalEngine::resolveStmtArraySizes(Stmt &s, bool insideUnsafe) {
    bool ok = true;
    switch (s.kind) {
    case StmtKind::Compound: {
        auto &cs = static_cast<CompoundStmt &>(s);
        for (auto &child : cs.body) ok = resolveStmtArraySizes(*child, insideUnsafe) && ok;
        break;
    }
    case StmtKind::VarDecl: {
        auto &vd = static_cast<VarDeclStmt &>(s);
        // VLA only for a local variable's own array type, and only inside
        // 'unsafe{}' (see the ArrayType::size==-2 comment on
        // resolveTypeArraySizes) — matches the "any C feature not natively
        // modeled must at least work in unsafe{}" rule applied everywhere
        // else in this pass (bitfields, designated inits, etc. don't need
        // it since those are always foldable/well-defined; a VLA's whole
        // point is a runtime-determined size, which is inherently unsafe).
        ok = resolveTypeArraySizes(vd.declType, insideUnsafe) && ok;
        break;
    }
    case StmtKind::If: {
        auto &is = static_cast<IfStmt &>(s);
        if (is.then)  ok = resolveStmtArraySizes(*is.then, insideUnsafe) && ok;
        if (is.else_) ok = resolveStmtArraySizes(*is.else_, insideUnsafe) && ok;
        break;
    }
    case StmtKind::While:
    case StmtKind::DoWhile: {
        auto &ws = static_cast<WhileStmt &>(s);
        if (ws.body) ok = resolveStmtArraySizes(*ws.body, insideUnsafe) && ok;
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<ForStmt &>(s);
        if (fs.init) ok = resolveStmtArraySizes(*fs.init, insideUnsafe) && ok;
        if (fs.body) ok = resolveStmtArraySizes(*fs.body, insideUnsafe) && ok;
        break;
    }
    case StmtKind::Unsafe: {
        auto &us = static_cast<UnsafeStmt &>(s);
        if (us.body) ok = resolveStmtArraySizes(*us.body, /*insideUnsafe=*/true) && ok;
        break;
    }
    case StmtKind::Defer: {
        auto &ds = static_cast<DeferStmt &>(s);
        if (ds.body) ok = resolveStmtArraySizes(*ds.body, insideUnsafe) && ok;
        break;
    }
    case StmtKind::Match: {
        auto &ms = static_cast<MatchStmt &>(s);
        for (auto &arm : ms.arms)
            if (arm.body) ok = resolveStmtArraySizes(*arm.body, insideUnsafe) && ok;
        break;
    }
    default:
        break;
    }
    return ok;
}

bool ConstEvalEngine::resolveArraySizes() {
    buildFnTable();
    bool ok = true;

    for (auto &decl : tu_.decls) {
        switch (decl->kind) {
        case DeclKind::GlobalVar: {
            auto &gv = static_cast<GlobalVarDecl &>(*decl);
            ok = resolveTypeArraySizes(gv.type) && ok;
            break;
        }
        case DeclKind::Struct: {
            auto &sd = static_cast<StructDecl &>(*decl);
            for (auto &f : sd.fields) ok = resolveTypeArraySizes(f.type) && ok;
            break;
        }
        case DeclKind::Function: {
            auto &fn = static_cast<FunctionDecl &>(*decl);
            ok = resolveTypeArraySizes(fn.returnType) && ok;
            for (auto &p : fn.params) ok = resolveTypeArraySizes(p.type) && ok;
            if (fn.body) ok = resolveStmtArraySizes(*fn.body) && ok;
            break;
        }
        default:
            break;
        }
    }

    return ok;
}

// Resolve all IfConstStmt nodes in a statement tree.
void ConstEvalEngine::resolveIfConstInStmt(Stmt &s) {
    switch (s.kind) {
    case StmtKind::IfConst: {
        auto &ics = static_cast<IfConstStmt &>(s);
        if (ics.cond) {
            auto val = evalExpr(*ics.cond);
            if (val) ics.constResult = val->toBool();
        }
        // Recurse into both branches for nested if const
        if (ics.then)  resolveIfConstInStmt(*ics.then);
        if (ics.else_) resolveIfConstInStmt(*ics.else_);
        break;
    }
    case StmtKind::Compound: {
        auto &cs = static_cast<CompoundStmt &>(s);
        for (auto &sub : cs.body) resolveIfConstInStmt(*sub);
        break;
    }
    case StmtKind::If: {
        auto &is = static_cast<IfStmt &>(s);
        if (is.then)  resolveIfConstInStmt(*is.then);
        if (is.else_) resolveIfConstInStmt(*is.else_);
        break;
    }
    case StmtKind::While:
    case StmtKind::DoWhile: {
        auto &ws = static_cast<WhileStmt &>(s);
        if (ws.body) resolveIfConstInStmt(*ws.body);
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<ForStmt &>(s);
        if (fs.init) resolveIfConstInStmt(*fs.init);
        if (fs.body) resolveIfConstInStmt(*fs.body);
        break;
    }
    case StmtKind::Unsafe: {
        auto &us = static_cast<UnsafeStmt &>(s);
        if (us.body) resolveIfConstInStmt(*us.body);
        break;
    }
    default:
        break;
    }
}

// =============================================================================
// consteval enforcement: consteval functions must not be called at runtime
// =============================================================================

void ConstEvalEngine::checkFunctionForConstevalCalls(const FunctionDecl &fn,
                                                      bool inConstContext) {
    if (!fn.body) return;
    checkStmtForConstevalCalls(*fn.body, inConstContext);
}

void ConstEvalEngine::checkStmtForConstevalCalls(const Stmt &s,
                                                  bool inConstContext) {
    switch (s.kind) {
    case StmtKind::Compound: {
        auto &cs = static_cast<const CompoundStmt &>(s);
        for (auto &sub : cs.body) checkStmtForConstevalCalls(*sub, inConstContext);
        break;
    }
    case StmtKind::Expr: {
        auto &es = static_cast<const ExprStmt &>(s);
        if (es.expr) checkExprForConstevalCalls(*es.expr, inConstContext);
        break;
    }
    case StmtKind::If: {
        auto &is = static_cast<const IfStmt &>(s);
        if (is.cond) checkExprForConstevalCalls(*is.cond, inConstContext);
        if (is.then)  checkStmtForConstevalCalls(*is.then, inConstContext);
        if (is.else_) checkStmtForConstevalCalls(*is.else_, inConstContext);
        break;
    }
    case StmtKind::While:
    case StmtKind::DoWhile: {
        auto &ws = static_cast<const WhileStmt &>(s);
        if (ws.cond) checkExprForConstevalCalls(*ws.cond, inConstContext);
        if (ws.body) checkStmtForConstevalCalls(*ws.body, inConstContext);
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<const ForStmt &>(s);
        if (fs.init) checkStmtForConstevalCalls(*fs.init, inConstContext);
        if (fs.cond) checkExprForConstevalCalls(*fs.cond, inConstContext);
        if (fs.incr) checkExprForConstevalCalls(*fs.incr, inConstContext);
        if (fs.body) checkStmtForConstevalCalls(*fs.body, inConstContext);
        break;
    }
    case StmtKind::Return: {
        auto &rs = static_cast<const ReturnStmt &>(s);
        if (rs.value) checkExprForConstevalCalls(*rs.value, inConstContext);
        break;
    }
    case StmtKind::VarDecl: {
        auto &vd = static_cast<const VarDeclStmt &>(s);
        if (vd.init) checkExprForConstevalCalls(*vd.init, inConstContext);
        break;
    }
    case StmtKind::Unsafe: {
        if (inConstContext) {
            diag_.error(s.loc, "unsafe block is not allowed in const/consteval context");
        }
        break;
    }
    case StmtKind::StaticAssert: {
        auto &sa = static_cast<const StaticAssertStmt &>(s);
        if (sa.cond) checkExprForConstevalCalls(*sa.cond, true); // always const ctx
        break;
    }
    default:
        break;
    }
}

void ConstEvalEngine::checkExprForConstevalCalls(const Expr &e,
                                                  bool inConstContext) {
    switch (e.kind) {
    case ExprKind::Call: {
        auto &ce = static_cast<const CallExpr &>(e);
        // Find callee function
        FunctionDecl *callee = nullptr;
        if (ce.callee && ce.callee->kind == ExprKind::Ident) {
            auto &id = static_cast<const IdentExpr &>(*ce.callee);
            callee = lookupFn(id.name);
        }
        if (callee) {
            if (callee->isConsteval && !inConstContext) {
                diag_.error(e.loc,
                    "consteval function '" + callee->name +
                    "' cannot be called in a runtime (non-const) context");
            }
            if (inConstContext && !callee->isConst && !callee->isConsteval
                && !callee->isExtern)
            {
                // Calling a non-const function in const context
                diag_.error(e.loc,
                    "cannot call non-const function '" + callee->name +
                    "' in a compile-time context; mark it 'const' or 'consteval'");
            }
        }
        // Check arguments
        for (auto &arg : ce.args)
            checkExprForConstevalCalls(*arg, inConstContext);
        break;
    }
    case ExprKind::Unary: {
        auto &ue = static_cast<const UnaryExpr &>(e);
        if (ue.operand) checkExprForConstevalCalls(*ue.operand, inConstContext);
        break;
    }
    case ExprKind::Binary: {
        auto &be = static_cast<const BinaryExpr &>(e);
        if (be.left)  checkExprForConstevalCalls(*be.left, inConstContext);
        if (be.right) checkExprForConstevalCalls(*be.right, inConstContext);
        break;
    }
    case ExprKind::Ternary: {
        auto &te = static_cast<const TernaryExpr &>(e);
        if (te.cond)  checkExprForConstevalCalls(*te.cond, inConstContext);
        if (te.then)  checkExprForConstevalCalls(*te.then, inConstContext);
        if (te.else_) checkExprForConstevalCalls(*te.else_, inConstContext);
        break;
    }
    case ExprKind::Assign: {
        if (inConstContext) {
            // Assignment to globals is disallowed in const context
            // Assignment to locals is OK (checked in statement-level evaluation)
        }
        auto &ae = static_cast<const AssignExpr &>(e);
        if (ae.lhs) checkExprForConstevalCalls(*ae.lhs, inConstContext);
        if (ae.rhs) checkExprForConstevalCalls(*ae.rhs, inConstContext);
        break;
    }
    default:
        break;
    }
}

// =============================================================================
// evalExpr — top-level: evaluate in an empty frame (for globals)
// =============================================================================

std::optional<ConstValue> ConstEvalEngine::evalExpr(const Expr &e) {
    Frame frame;
    return evalExprF(e, frame);
}

std::optional<ConstValue> ConstEvalEngine::getGlobalConst(const std::string &name) const {
    auto it = globalConsts_.find(name);
    if (it != globalConsts_.end()) return it->second;
    return {};
}

// =============================================================================
// Expression evaluation
// =============================================================================

std::optional<ConstValue> ConstEvalEngine::evalExprF(const Expr &e, Frame &frame) {
    tickInstr();
    if (isBudgetExceeded(e.loc)) return {};

    switch (e.kind) {
    case ExprKind::IntLit:
        return evalIntLit(static_cast<const IntLitExpr &>(e));
    case ExprKind::FloatLit:
        return evalFloatLit(static_cast<const FloatLitExpr &>(e));
    case ExprKind::BoolLit:
        return evalBoolLit(static_cast<const BoolLitExpr &>(e));
    case ExprKind::StringLit:
        return evalStringLit(static_cast<const StringLitExpr &>(e));
    case ExprKind::NullLit:
        return ConstValue::mkInt(0);
    case ExprKind::CharLit:
        // CharLitExpr reuses IntLitExpr for now (stored same way)
        return ConstValue::mkInt(static_cast<const IntLitExpr &>(e).value);
    case ExprKind::Ident:
        return evalIdent(static_cast<const IdentExpr &>(e), frame);
    case ExprKind::Unary:
        return evalUnary(static_cast<const UnaryExpr &>(e), frame);
    case ExprKind::Binary:
        return evalBinary(static_cast<const BinaryExpr &>(e), frame);
    case ExprKind::Ternary:
        return evalTernary(static_cast<const TernaryExpr &>(e), frame);
    case ExprKind::Call:
        return evalCallExpr(static_cast<const CallExpr &>(e), frame);
    case ExprKind::Cast:
        return evalCast(static_cast<const CastExpr &>(e), frame);
    case ExprKind::Assign:
        return evalAssign(static_cast<const AssignExpr &>(e), frame);
    case ExprKind::Compound:
        return evalCompoundInit(static_cast<const CompoundInitExpr &>(e), frame);
    case ExprKind::Member:
    case ExprKind::Arrow:
        return evalMember(static_cast<const MemberExpr &>(e), frame);
    case ExprKind::Subscript:
        return evalSubscriptConst(static_cast<const SubscriptExpr &>(e), frame);
    case ExprKind::SizeofType: {
        // sizeof(T) — return a fixed size (basic impl)
        auto &ste = static_cast<const SizeofTypeExpr &>(e);
        if (!ste.ofType) return ConstValue::mkInt(0);
        switch (ste.ofType->kind) {
        case TypeKind::Void:                   return ConstValue::mkInt(0);
        case TypeKind::Bool:                   return ConstValue::mkInt(1);
        case TypeKind::Char:
        case TypeKind::Int8:
        case TypeKind::UInt8:                  return ConstValue::mkInt(1);
        case TypeKind::Int16:
        case TypeKind::UInt16:                 return ConstValue::mkInt(2);
        case TypeKind::Int32:
        case TypeKind::UInt32:
        case TypeKind::Float32:                return ConstValue::mkInt(4);
        case TypeKind::Int64:
        case TypeKind::UInt64:
        case TypeKind::Float64:                return ConstValue::mkInt(8);
        case TypeKind::Pointer:
        case TypeKind::Reference:              return ConstValue::mkInt(8);
        default:                               return ConstValue::mkInt(8);
        }
    }
    case ExprKind::AlignofType: {
        auto &ae = static_cast<const AlignofTypeExpr &>(e);
        if (!ae.ofType) return ConstValue::mkInt(0);
        switch (ae.ofType->kind) {
        case TypeKind::Bool:
        case TypeKind::Char:
        case TypeKind::Int8:
        case TypeKind::UInt8:                  return ConstValue::mkInt(1);
        case TypeKind::Int16:
        case TypeKind::UInt16:                 return ConstValue::mkInt(2);
        case TypeKind::Int32:
        case TypeKind::UInt32:
        case TypeKind::Float32:                return ConstValue::mkInt(4);
        case TypeKind::Int64:
        case TypeKind::UInt64:
        case TypeKind::Float64:
        case TypeKind::Pointer:
        case TypeKind::Reference:              return ConstValue::mkInt(8);
        default:                               return ConstValue::mkInt(8);
        }
    }
    case ExprKind::FieldCount: {
        auto &fc = static_cast<const FieldCountExpr &>(e);
        if (fc.ofType && fc.ofType->kind == TypeKind::Struct) {
            auto &st = static_cast<const StructType &>(*fc.ofType);
            return ConstValue::mkInt(static_cast<int64_t>(st.fields.size()));
        }
        return ConstValue::mkInt(0);
    }
    case ExprKind::SizeofPack: {
        auto &sp = static_cast<const SizeofPackExpr &>(e);
        return ConstValue::mkInt(sp.resolvedCount);
    }
    default:
        diag_.error(e.loc,
            "expression is not a valid compile-time constant expression");
        return {};
    }
}

std::optional<ConstValue> ConstEvalEngine::evalIntLit(const IntLitExpr &e) {
    return ConstValue::mkInt(e.value);
}

std::optional<ConstValue> ConstEvalEngine::evalFloatLit(const FloatLitExpr &e) {
    return ConstValue::mkFloat(e.value);
}

std::optional<ConstValue> ConstEvalEngine::evalBoolLit(const BoolLitExpr &e) {
    return ConstValue::mkBool(e.value);
}

std::optional<ConstValue> ConstEvalEngine::evalStringLit(const StringLitExpr &e) {
    return ConstValue::mkString(e.value);
}

std::optional<ConstValue> ConstEvalEngine::evalIdent(const IdentExpr &e, Frame &frame) {
    // 1. Local variable in current frame
    auto it = frame.locals.find(e.name);
    if (it != frame.locals.end()) return it->second;

    // 2. Pre-evaluated global const
    auto git = globalConsts_.find(e.name);
    if (git != globalConsts_.end()) return git->second;

    // 3. Global const variable — evaluate its initializer now
    for (auto &decl : tu_.decls) {
        if (decl->kind == DeclKind::GlobalVar) {
            auto &gv = static_cast<const GlobalVarDecl &>(*decl);
            if (gv.name == e.name && gv.isConst && gv.init) {
                Frame empty;
                auto val = evalExprF(*gv.init, empty);
                if (val) {
                    globalConsts_[e.name] = *val;
                    return val;
                }
            }
        }
    }

    // 4. Enum constant — look for it
    for (auto &decl : tu_.decls) {
        if (decl->kind == DeclKind::Enum) {
            auto &ed = static_cast<const EnumDecl &>(*decl);
            int64_t next = 0;
            for (auto &[name, val] : ed.enumerators) {
                if (name == e.name)
                    return ConstValue::mkInt(val.has_value() ? *val : next);
                next = (val.has_value() ? *val : next) + 1;
            }
        }
    }

    diag_.error(e.loc, "'" + e.name + "' is not a compile-time constant");
    return {};
}

std::optional<ConstValue> ConstEvalEngine::evalUnary(const UnaryExpr &e, Frame &frame) {
    if (!e.operand) return {};

    // sizeof(expr): we only need the type which is already resolved by sema
    if (e.op == UnaryOp::SizeofExpr) {
        if (!e.operand->type) return ConstValue::mkInt(8);
        // Reuse SizeofType logic
        SizeofTypeExpr fake(e.operand->type, e.loc);
        return evalExprF(fake, frame);
    }

    // Inc/dec go through resolveLValueSlot (not the plain evalExprF read
    // below) so the target evaluates exactly once — for e.g. 'arr[j++]++',
    // evaluating the operand both to read it *and* separately to locate
    // its slot would double-apply the index's own side effect. This also
    // generalizes inc/dec to any Member/Subscript-rooted lvalue, not just
    // a bare identifier.
    switch (e.op) {
    case UnaryOp::PreInc: case UnaryOp::PreDec:
    case UnaryOp::PostInc: case UnaryOp::PostDec: {
        ConstValue *slot = resolveLValueSlot(*e.operand, frame);
        if (!slot) return {};
        ConstValue old = *slot;
        bool isDec = (e.op == UnaryOp::PreDec || e.op == UnaryOp::PostDec);
        double delta = isDec ? -1.0 : 1.0;
        *slot = (slot->kind == ConstValue::Float)
            ? ConstValue::mkFloat(slot->floatVal + delta)
            : ConstValue::mkInt(slot->toInt() + (int64_t)delta);
        bool isPost = (e.op == UnaryOp::PostInc || e.op == UnaryOp::PostDec);
        return isPost ? old : *slot;
    }
    default: break;
    }

    auto operand = evalExprF(*e.operand, frame);
    if (!operand) return {};

    switch (e.op) {
    case UnaryOp::Neg:
        if (operand->kind == ConstValue::Float)
            return ConstValue::mkFloat(-operand->floatVal);
        return ConstValue::mkInt(-operand->toInt());
    case UnaryOp::Not:
        return ConstValue::mkBool(!operand->toBool());
    case UnaryOp::BitNot:
        return ConstValue::mkInt(~operand->toInt());
    default:
        diag_.error(e.loc, "unsupported unary operation in const context");
        return {};
    }
}

std::optional<ConstValue> ConstEvalEngine::evalBinary(const BinaryExpr &e, Frame &frame) {
    if (!e.left || !e.right) return {};

    // Short-circuit evaluation for && and ||
    if (e.op == BinaryOp::LogAnd) {
        auto l = evalExprF(*e.left, frame);
        if (!l) return {};
        if (!l->toBool()) return ConstValue::mkBool(false);
        auto r = evalExprF(*e.right, frame);
        if (!r) return {};
        return ConstValue::mkBool(r->toBool());
    }
    if (e.op == BinaryOp::LogOr) {
        auto l = evalExprF(*e.left, frame);
        if (!l) return {};
        if (l->toBool()) return ConstValue::mkBool(true);
        auto r = evalExprF(*e.right, frame);
        if (!r) return {};
        return ConstValue::mkBool(r->toBool());
    }

    auto lv = evalExprF(*e.left, frame);
    auto rv = evalExprF(*e.right, frame);
    if (!lv || !rv) return {};

    // Float arithmetic if either side is float
    bool isFloat = (lv->kind == ConstValue::Float || rv->kind == ConstValue::Float);

    switch (e.op) {
    case BinaryOp::Add:
        return isFloat ? ConstValue::mkFloat(lv->toFloat() + rv->toFloat())
                       : ConstValue::mkInt(lv->toInt() + rv->toInt());
    case BinaryOp::Sub:
        return isFloat ? ConstValue::mkFloat(lv->toFloat() - rv->toFloat())
                       : ConstValue::mkInt(lv->toInt() - rv->toInt());
    case BinaryOp::Mul:
        return isFloat ? ConstValue::mkFloat(lv->toFloat() * rv->toFloat())
                       : ConstValue::mkInt(lv->toInt() * rv->toInt());
    case BinaryOp::Div: {
        if (!isFloat && rv->toInt() == 0) {
            diag_.error(e.loc, "division by zero in compile-time expression");
            return {};
        }
        return isFloat ? ConstValue::mkFloat(lv->toFloat() / rv->toFloat())
                       : ConstValue::mkInt(lv->toInt() / rv->toInt());
    }
    case BinaryOp::Mod: {
        if (rv->toInt() == 0) {
            diag_.error(e.loc, "modulo by zero in compile-time expression");
            return {};
        }
        return ConstValue::mkInt(lv->toInt() % rv->toInt());
    }
    case BinaryOp::BitAnd:  return ConstValue::mkInt(lv->toInt() & rv->toInt());
    case BinaryOp::BitOr:   return ConstValue::mkInt(lv->toInt() | rv->toInt());
    case BinaryOp::BitXor:  return ConstValue::mkInt(lv->toInt() ^ rv->toInt());
    case BinaryOp::Shl:     return ConstValue::mkInt(lv->toInt() << rv->toInt());
    case BinaryOp::Shr:     return ConstValue::mkInt(lv->toInt() >> rv->toInt());
    case BinaryOp::Eq:
        return isFloat ? ConstValue::mkBool(lv->toFloat() == rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() == rv->toInt());
    case BinaryOp::NEq:
        return isFloat ? ConstValue::mkBool(lv->toFloat() != rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() != rv->toInt());
    case BinaryOp::Lt:
        return isFloat ? ConstValue::mkBool(lv->toFloat() < rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() < rv->toInt());
    case BinaryOp::Gt:
        return isFloat ? ConstValue::mkBool(lv->toFloat() > rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() > rv->toInt());
    case BinaryOp::LEq:
        return isFloat ? ConstValue::mkBool(lv->toFloat() <= rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() <= rv->toInt());
    case BinaryOp::GEq:
        return isFloat ? ConstValue::mkBool(lv->toFloat() >= rv->toFloat())
                       : ConstValue::mkBool(lv->toInt() >= rv->toInt());
    case BinaryOp::Comma:
        return rv;  // evaluate both, return right
    default:
        diag_.error(e.loc, "unsupported binary operation in const context");
        return {};
    }
}

std::optional<ConstValue> ConstEvalEngine::evalTernary(const TernaryExpr &e, Frame &frame) {
    if (!e.cond) return {};
    auto cond = evalExprF(*e.cond, frame);
    if (!cond) return {};
    if (cond->toBool()) {
        return e.then ? evalExprF(*e.then, frame) : ConstValue::mkVoid();
    } else {
        return e.else_ ? evalExprF(*e.else_, frame) : ConstValue::mkVoid();
    }
}

std::optional<ConstValue> ConstEvalEngine::evalCallExpr(const CallExpr &e, Frame &frame) {
    if (!e.callee) return {};

    // Resolve callee name
    std::string calleeName;
    if (e.callee->kind == ExprKind::Ident)
        calleeName = static_cast<const IdentExpr &>(*e.callee).name;
    else {
        diag_.error(e.loc, "indirect calls are not supported in const context");
        return {};
    }

    FunctionDecl *fn = lookupFn(calleeName);
    if (!fn) {
        diag_.error(e.loc,
            "function '" + calleeName + "' is not available in const context "
            "(extern functions cannot be called at compile time)");
        return {};
    }
    if (!fn->isConst && !fn->isConsteval) {
        diag_.error(e.loc,
            "cannot call non-const function '" + calleeName +
            "' at compile time; mark it 'const' or 'consteval'");
        return {};
    }

    // Evaluate arguments
    std::vector<ConstValue> args;
    for (auto &arg : e.args) {
        auto v = evalExprF(*arg, frame);
        if (!v) return {};
        args.push_back(*v);
    }

    return evalCall(*fn, std::move(args), e.loc);
}

std::optional<ConstValue> ConstEvalEngine::evalAssign(const AssignExpr &e, Frame &frame) {
    if (!e.lhs || !e.rhs) return {};

    auto rval = evalExprF(*e.rhs, frame);
    if (!rval) return {};

    // Resolves to a mutable slot for any lvalue this engine can track — a
    // bare local, or a Member/Subscript chain rooted at one (e.g.
    // 's.field = x', 'arr[i] = x', 'arr[i].field = x') — not just a plain
    // identifier as before, so aggregate locals can actually be mutated.
    ConstValue *slot = resolveLValueSlot(*e.lhs, frame);
    if (!slot) return {};

    ConstValue newVal;
    switch (e.op) {
    case AssignOp::Assign:    newVal = *rval; break;
    case AssignOp::AddAssign:
        newVal = (slot->kind == ConstValue::Float || rval->kind == ConstValue::Float)
            ? ConstValue::mkFloat(slot->toFloat() + rval->toFloat())
            : ConstValue::mkInt(slot->toInt() + rval->toInt());
        break;
    case AssignOp::SubAssign:
        newVal = ConstValue::mkInt(slot->toInt() - rval->toInt());
        break;
    case AssignOp::MulAssign:
        newVal = ConstValue::mkInt(slot->toInt() * rval->toInt());
        break;
    case AssignOp::DivAssign: {
        int64_t r = rval->toInt();
        if (!r) { diag_.error(e.loc, "division by zero in const context"); return {}; }
        newVal = ConstValue::mkInt(slot->toInt() / r);
        break;
    }
    case AssignOp::ModAssign: {
        int64_t r = rval->toInt();
        if (!r) { diag_.error(e.loc, "modulo by zero in const context"); return {}; }
        newVal = ConstValue::mkInt(slot->toInt() % r);
        break;
    }
    case AssignOp::AndAssign: newVal = ConstValue::mkInt(slot->toInt() & rval->toInt()); break;
    case AssignOp::OrAssign:  newVal = ConstValue::mkInt(slot->toInt() | rval->toInt()); break;
    case AssignOp::XorAssign: newVal = ConstValue::mkInt(slot->toInt() ^ rval->toInt()); break;
    case AssignOp::ShlAssign: newVal = ConstValue::mkInt(slot->toInt() << rval->toInt()); break;
    case AssignOp::ShrAssign: newVal = ConstValue::mkInt(slot->toInt() >> rval->toInt()); break;
    }

    *slot = newVal;
    return newVal;
}

// Unwraps a pointer/reference type to its pointee — used by evalMember and
// resolveLValueSlot so 'p->field' and 'p.field' (p a struct value or a
// pointer/reference to one) resolve the same field lookup either way.
static TypePtr unwrapPointeeType(const TypePtr &ty) {
    if (!ty) return ty;
    if (ty->kind == TypeKind::Pointer)   return static_cast<const PointerType &>(*ty).base;
    if (ty->kind == TypeKind::Reference) return static_cast<const ReferenceType &>(*ty).base;
    return ty;
}

std::optional<ConstValue> ConstEvalEngine::evalCompoundInit(const CompoundInitExpr &e, Frame &frame) {
    if (!e.type) {
        diag_.error(e.loc, "compound initializer has no resolved type in const context");
        return {};
    }

    // Slot count and shape come from the resolved type, not from
    // e.inits.size(), so a partial initializer ('struct S s = {.a = 1};')
    // still produces a full-shaped value with the untouched fields
    // zero-valued — same as CodeGen::evalConstInit's approach.
    ConstValue result;
    size_t slotCount;
    if (e.type->kind == TypeKind::Struct) {
        auto &st = static_cast<const StructType &>(*e.type);
        result.kind = ConstValue::Struct;
        result.elems.reserve(st.fields.size());
        for (auto &f : st.fields) result.elems.push_back(zeroValue(f.type));
        slotCount = st.fields.size();
    } else if (e.type->kind == TypeKind::Array) {
        auto &at = static_cast<const ArrayType &>(*e.type);
        int64_t n = at.size > 0 ? at.size : (int64_t)e.inits.size();
        result.kind = ConstValue::Array;
        result.elems.reserve((size_t)n);
        for (int64_t i = 0; i < n; ++i) result.elems.push_back(zeroValue(at.element));
        slotCount = (size_t)n;
    } else {
        diag_.error(e.loc,
            "compound initializer not supported for type '" + e.type->str() +
            "' in const context");
        return {};
    }

    for (size_t i = 0; i < e.inits.size(); ++i) {
        int64_t slot = e.resolvedSlots.empty() ? (int64_t)i : e.resolvedSlots[i];
        if (slot < 0 || (size_t)slot >= slotCount) {
            diag_.error(e.loc, "designated initializer slot out of range");
            return {};
        }
        auto v = evalExprF(*e.inits[i], frame);
        if (!v) return {};
        result.elems[(size_t)slot] = *v;
    }
    return result;
}

std::optional<ConstValue> ConstEvalEngine::evalMember(const MemberExpr &e, Frame &frame) {
    if (!e.base) return {};
    auto base = evalExprF(*e.base, frame);
    if (!base) return {};
    if (base->kind != ConstValue::Struct) {
        diag_.error(e.loc, "'." + e.field + "' member access on a non-struct compile-time value");
        return {};
    }
    TypePtr baseTy = unwrapPointeeType(e.base->type);
    if (!baseTy || baseTy->kind != TypeKind::Struct) {
        diag_.error(e.loc, "member access requires a struct type");
        return {};
    }
    auto &st = static_cast<const StructType &>(*baseTy);
    std::vector<int> path;
    const FieldDecl *fd = st.findFieldPath(e.field, path);
    if (!fd || path.empty()) {
        diag_.error(e.loc, "unknown field '" + e.field + "' in compile-time struct value");
        return {};
    }
    const ConstValue *cur = &*base;
    for (int idx : path) {
        if (idx < 0 || (size_t)idx >= cur->elems.size()) {
            diag_.error(e.loc, "field index out of range in compile-time struct value");
            return {};
        }
        cur = &cur->elems[(size_t)idx];
    }
    return *cur;
}

std::optional<ConstValue> ConstEvalEngine::evalSubscriptConst(const SubscriptExpr &e, Frame &frame) {
    if (!e.base || !e.index) return {};
    auto base = evalExprF(*e.base, frame);
    if (!base) return {};
    if (base->kind != ConstValue::Array) {
        diag_.error(e.loc, "subscript on a non-array compile-time value");
        return {};
    }
    auto idxVal = evalExprF(*e.index, frame);
    if (!idxVal) return {};
    int64_t idx = idxVal->toInt();
    if (idx < 0 || (size_t)idx >= base->elems.size()) {
        diag_.error(e.loc, "compile-time array index " + std::to_string(idx) + " out of bounds");
        return {};
    }
    return base->elems[(size_t)idx];
}

ConstValue *ConstEvalEngine::resolveLValueSlot(const Expr &e, Frame &frame) {
    switch (e.kind) {
    case ExprKind::Ident: {
        auto &id = static_cast<const IdentExpr &>(e);
        auto it = frame.locals.find(id.name);
        if (it == frame.locals.end()) {
            diag_.error(e.loc,
                "'" + id.name + "' is not an assignable compile-time local variable");
            return nullptr;
        }
        return &it->second;
    }
    case ExprKind::Member:
    case ExprKind::Arrow: {
        auto &me = static_cast<const MemberExpr &>(e);
        if (!me.base) return nullptr;
        ConstValue *base = resolveLValueSlot(*me.base, frame);
        if (!base) return nullptr;
        if (base->kind != ConstValue::Struct) {
            diag_.error(e.loc, "'." + me.field + "' member access on a non-struct compile-time value");
            return nullptr;
        }
        TypePtr baseTy = unwrapPointeeType(me.base->type);
        if (!baseTy || baseTy->kind != TypeKind::Struct) {
            diag_.error(e.loc, "member access requires a struct type");
            return nullptr;
        }
        auto &st = static_cast<const StructType &>(*baseTy);
        std::vector<int> path;
        const FieldDecl *fd = st.findFieldPath(me.field, path);
        if (!fd || path.empty()) {
            diag_.error(e.loc, "unknown field '" + me.field + "' in compile-time struct value");
            return nullptr;
        }
        for (int idx : path) {
            if (idx < 0 || (size_t)idx >= base->elems.size()) {
                diag_.error(e.loc, "field index out of range in compile-time struct value");
                return nullptr;
            }
            base = &base->elems[(size_t)idx];
        }
        return base;
    }
    case ExprKind::Subscript: {
        auto &se = static_cast<const SubscriptExpr &>(e);
        if (!se.base || !se.index) return nullptr;
        ConstValue *base = resolveLValueSlot(*se.base, frame);
        if (!base) return nullptr;
        if (base->kind != ConstValue::Array) {
            diag_.error(e.loc, "subscript on a non-array compile-time value");
            return nullptr;
        }
        auto idxVal = evalExprF(*se.index, frame);
        if (!idxVal) return nullptr;
        int64_t idx = idxVal->toInt();
        if (idx < 0 || (size_t)idx >= base->elems.size()) {
            diag_.error(e.loc, "compile-time array index " + std::to_string(idx) + " out of bounds");
            return nullptr;
        }
        return &base->elems[(size_t)idx];
    }
    default:
        diag_.error(e.loc, "expression is not an assignable compile-time lvalue "
                            "(only locals and struct/array field or index chains "
                            "rooted at one can be mutated at compile time)");
        return nullptr;
    }
}

std::optional<ConstValue> ConstEvalEngine::evalCast(const CastExpr &e, Frame &frame) {
    if (!e.operand) return {};
    auto val = evalExprF(*e.operand, frame);
    if (!val) return {};
    if (!e.targetType) return val;

    // Cast to numeric types based on TypeKind
    switch (e.targetType->kind) {
    case TypeKind::Int8:  case TypeKind::Int16: case TypeKind::Int32:
    case TypeKind::Int64: case TypeKind::UInt8: case TypeKind::UInt16:
    case TypeKind::UInt32: case TypeKind::UInt64:
    case TypeKind::Char:
        return ConstValue::mkInt(val->toInt());
    case TypeKind::Float32:
    case TypeKind::Float64:
        return ConstValue::mkFloat(val->toFloat());
    case TypeKind::Bool:
        return ConstValue::mkBool(val->toBool());
    default:
        return val;
    }
}

// =============================================================================
// evalCall — interpret a function body with given arguments
// =============================================================================

std::optional<ConstValue> ConstEvalEngine::evalCall(const FunctionDecl &fn,
                                                      std::vector<ConstValue> args,
                                                      SourceLocation callLoc) {
    if (!fn.body) {
        diag_.error(callLoc,
            "consteval function '" + fn.name + "' has no body to evaluate");
        return {};
    }

    if (recursionDepth_ >= kMaxRecursion) {
        diag_.error(callLoc,
            "compile-time recursion depth limit (" +
            std::to_string(kMaxRecursion) + ") exceeded in '" + fn.name + "'");
        return {};
    }

    // Build new frame with parameter bindings
    Frame frame;
    for (size_t i = 0; i < fn.params.size() && i < args.size(); ++i)
        frame.locals[fn.params[i].name] = args[i];

    ++recursionDepth_;
    bool ok = evalCompound(*fn.body, frame);
    --recursionDepth_;

    if (!ok && !frame.hasReturn) return {};
    if (frame.hasReturn) return frame.returnVal;
    return ConstValue::mkVoid();
}

// =============================================================================
// Statement evaluation
// =============================================================================

bool ConstEvalEngine::evalStmt(const Stmt &s, Frame &frame) {
    tickInstr();
    if (isBudgetExceeded(s.loc)) return false;
    if (frame.hasReturn || breakFlag_ || continueFlag_) return true;

    switch (s.kind) {
    case StmtKind::Compound:
        return evalCompound(static_cast<const CompoundStmt &>(s), frame);
    case StmtKind::Expr: {
        auto &es = static_cast<const ExprStmt &>(s);
        if (es.expr) evalExprF(*es.expr, frame);
        return !diag_.hasErrors();
    }
    case StmtKind::If:
        return evalIf(static_cast<const IfStmt &>(s), frame);
    case StmtKind::While:
    case StmtKind::DoWhile:
        return evalWhile(static_cast<const WhileStmt &>(s), frame);
    case StmtKind::For:
        return evalFor(static_cast<const ForStmt &>(s), frame);
    case StmtKind::Return:
        return evalReturn(static_cast<const ReturnStmt &>(s), frame);
    case StmtKind::VarDecl:
        return evalVarDecl(static_cast<const VarDeclStmt &>(s), frame);
    case StmtKind::Break:
        breakFlag_ = true;
        return true;
    case StmtKind::Continue:
        continueFlag_ = true;
        return true;
    case StmtKind::StaticAssert: {
        auto &sa = static_cast<const StaticAssertStmt &>(s);
        if (sa.cond) {
            auto v = evalExprF(*sa.cond, frame);
            if (v && !v->toBool()) {
                std::string msg = sa.message.empty()
                    ? "static_assert failed"
                    : "static_assert failed: " + sa.message;
                diag_.error(sa.loc, msg);
            }
        }
        return !diag_.hasErrors();
    }
    case StmtKind::Unsafe:
        diag_.error(s.loc, "unsafe block is not allowed in const/consteval context");
        return false;
    default:
        // Label, Goto — not fully supported in const context
        return true;
    }
}

bool ConstEvalEngine::evalCompound(const CompoundStmt &s, Frame &frame) {
    for (auto &sub : s.body) {
        if (!evalStmt(*sub, frame)) return false;
        if (frame.hasReturn || breakFlag_ || continueFlag_) break;
    }
    return true;
}

bool ConstEvalEngine::evalIf(const IfStmt &s, Frame &frame) {
    if (!s.cond) return true;
    auto cond = evalExprF(*s.cond, frame);
    if (!cond) return false;
    if (cond->toBool()) {
        if (s.then) return evalStmt(*s.then, frame);
    } else {
        if (s.else_) return evalStmt(*s.else_, frame);
    }
    return true;
}

bool ConstEvalEngine::evalWhile(const WhileStmt &s, Frame &frame) {
    int64_t iters = 0;
    bool isDoWhile = s.isDoWhile;
    bool doFirst = isDoWhile;

    while (true) {
        if (!doFirst) {
            if (!s.cond) break;
            auto cond = evalExprF(*s.cond, frame);
            if (!cond || !cond->toBool()) break;
        }
        doFirst = false;

        if (s.body) {
            if (!evalStmt(*s.body, frame)) return false;
        }
        if (frame.hasReturn) break;
        if (breakFlag_) { breakFlag_ = false; break; }
        if (continueFlag_) { continueFlag_ = false; }

        if (++iters >= kLoopLimit) {
            diag_.error(s.loc,
                "compile-time loop iteration limit (" +
                std::to_string(kLoopLimit) + ") exceeded");
            return false;
        }

        if (isDoWhile) {
            if (!s.cond) break;
            auto cond = evalExprF(*s.cond, frame);
            if (!cond || !cond->toBool()) break;
        }
    }
    return !diag_.hasErrors();
}

bool ConstEvalEngine::evalFor(const ForStmt &s, Frame &frame) {
    if (s.init) if (!evalStmt(*s.init, frame)) return false;

    int64_t iters = 0;
    while (true) {
        if (s.cond) {
            auto cond = evalExprF(*s.cond, frame);
            if (!cond || !cond->toBool()) break;
        }
        if (s.body) {
            if (!evalStmt(*s.body, frame)) return false;
        }
        if (frame.hasReturn) break;
        if (breakFlag_) { breakFlag_ = false; break; }
        if (continueFlag_) continueFlag_ = false;

        if (s.incr) evalExprF(*s.incr, frame);

        if (++iters >= kLoopLimit) {
            diag_.error(s.loc,
                "compile-time loop iteration limit (" +
                std::to_string(kLoopLimit) + ") exceeded");
            return false;
        }
    }
    return !diag_.hasErrors();
}

bool ConstEvalEngine::evalReturn(const ReturnStmt &s, Frame &frame) {
    if (s.value) {
        auto val = evalExprF(*s.value, frame);
        if (!val) return false;
        frame.returnVal = *val;
    } else {
        frame.returnVal = ConstValue::mkVoid();
    }
    frame.hasReturn = true;
    return true;
}

bool ConstEvalEngine::evalVarDecl(const VarDeclStmt &s, Frame &frame) {
    ConstValue val;
    if (s.init) {
        auto v = evalExprF(*s.init, frame);
        if (!v) return false;
        val = *v;
    } else {
        // No initializer: zero-construct matching the declared shape (e.g.
        // 'struct Circle c;' still needs a Struct-kind slot with one entry
        // per field, not Void) so a later 'c.radius = 2.0;' has something
        // to mutate into via resolveLValueSlot.
        val = zeroValue(s.resolvedType ? s.resolvedType : s.declType);
    }
    frame.locals[s.name] = val;
    return true;
}

// =============================================================================
// isConstExpr — check without evaluating
// =============================================================================

bool ConstEvalEngine::isConstExpr(const Expr &e) const {
    switch (e.kind) {
    case ExprKind::IntLit:
    case ExprKind::FloatLit:
    case ExprKind::BoolLit:
    case ExprKind::StringLit:
    case ExprKind::NullLit:
    case ExprKind::CharLit:
        return true;
    case ExprKind::Ident: {
        auto &id = static_cast<const IdentExpr &>(e);
        if (globalConsts_.count(id.name)) return true;
        // Check const global
        for (auto &decl : tu_.decls) {
            if (decl->kind == DeclKind::GlobalVar) {
                auto &gv = static_cast<const GlobalVarDecl &>(*decl);
                if (gv.name == id.name && gv.isConst) return true;
            }
            if (decl->kind == DeclKind::Enum) {
                auto &ed = static_cast<const EnumDecl &>(*decl);
                for (auto &[name, _] : ed.enumerators)
                    if (name == id.name) return true;
            }
        }
        // Check enum constants via resolved type
        if (id.resolved && id.resolved->isConst) return true;
        return false;
    }
    case ExprKind::Unary: {
        auto &ue = static_cast<const UnaryExpr &>(e);
        return ue.operand && isConstExpr(*ue.operand);
    }
    case ExprKind::Binary: {
        auto &be = static_cast<const BinaryExpr &>(e);
        return be.left && be.right && isConstExpr(*be.left) && isConstExpr(*be.right);
    }
    case ExprKind::Ternary: {
        auto &te = static_cast<const TernaryExpr &>(e);
        return te.cond && te.then && te.else_ &&
               isConstExpr(*te.cond) && isConstExpr(*te.then) && isConstExpr(*te.else_);
    }
    case ExprKind::Call: {
        auto &ce = static_cast<const CallExpr &>(e);
        if (!ce.callee || ce.callee->kind != ExprKind::Ident) return false;
        auto &id = static_cast<const IdentExpr &>(*ce.callee);
        auto it = fnTable_.find(id.name);
        if (it == fnTable_.end()) return false;
        if (!it->second->isConst && !it->second->isConsteval) return false;
        for (auto &arg : ce.args) if (!isConstExpr(*arg)) return false;
        return true;
    }
    case ExprKind::Cast: {
        auto &ce = static_cast<const CastExpr &>(e);
        return ce.operand && isConstExpr(*ce.operand);
    }
    case ExprKind::SizeofType:
    case ExprKind::AlignofType:
    case ExprKind::FieldCount:
    case ExprKind::SizeofPack:
        return true;  // compile-time reflection is always constant
    default:
        return false;
    }
}

} // namespace safec
