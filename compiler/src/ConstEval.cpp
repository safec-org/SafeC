#include "safec/ConstEval.h"
#include <cmath>
#include <cassert>
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
    }
    return "?";
}

// =============================================================================
// ConstEvalEngine
// =============================================================================

ConstEvalEngine::ConstEvalEngine(TranslationUnit &tu, DiagEngine &diag)
    : tu_(tu), diag_(diag) {}

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

bool ConstEvalEngine::run() {
    // Build function table
    for (auto &decl : tu_.decls) {
        if (decl->kind == DeclKind::Function) {
            auto &fn = static_cast<FunctionDecl &>(*decl);
            fnTable_[fn.name] = &fn;
        }
    }

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
                    // can emit the constant directly without calling consteval again.
                    switch (val->kind) {
                    case ConstValue::Int:
                    case ConstValue::Bool:
                        gv.init = std::make_unique<IntLitExpr>(val->toInt(), gv.loc);
                        break;
                    case ConstValue::Float:
                        gv.init = std::make_unique<FloatLitExpr>(val->floatVal, gv.loc);
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
    case UnaryOp::PreInc: {
        int64_t v = operand->toInt() + 1;
        // Try to write back to local
        if (e.operand->kind == ExprKind::Ident) {
            auto &id = static_cast<const IdentExpr &>(*e.operand);
            frame.locals[id.name] = ConstValue::mkInt(v);
        }
        return ConstValue::mkInt(v);
    }
    case UnaryOp::PreDec: {
        int64_t v = operand->toInt() - 1;
        if (e.operand->kind == ExprKind::Ident) {
            auto &id = static_cast<const IdentExpr &>(*e.operand);
            frame.locals[id.name] = ConstValue::mkInt(v);
        }
        return ConstValue::mkInt(v);
    }
    case UnaryOp::PostInc: {
        int64_t old = operand->toInt();
        if (e.operand->kind == ExprKind::Ident) {
            auto &id = static_cast<const IdentExpr &>(*e.operand);
            frame.locals[id.name] = ConstValue::mkInt(old + 1);
        }
        return ConstValue::mkInt(old);
    }
    case UnaryOp::PostDec: {
        int64_t old = operand->toInt();
        if (e.operand->kind == ExprKind::Ident) {
            auto &id = static_cast<const IdentExpr &>(*e.operand);
            frame.locals[id.name] = ConstValue::mkInt(old - 1);
        }
        return ConstValue::mkInt(old);
    }
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

    // Resolve lhs to a variable name
    auto getLhsName = [&]() -> std::string {
        if (e.lhs->kind == ExprKind::Ident)
            return static_cast<const IdentExpr &>(*e.lhs).name;
        return {};
    };
    std::string lhsName = getLhsName();

    if (lhsName.empty()) {
        diag_.error(e.loc, "complex assignment targets are not supported in const context");
        return {};
    }

    // Get current value for compound assignments
    auto getOld = [&]() -> ConstValue {
        auto it = frame.locals.find(lhsName);
        if (it != frame.locals.end()) return it->second;
        return ConstValue::mkInt(0);
    };

    ConstValue newVal;
    switch (e.op) {
    case AssignOp::Assign:    newVal = *rval; break;
    case AssignOp::AddAssign: {
        auto old = getOld();
        newVal = (old.kind == ConstValue::Float || rval->kind == ConstValue::Float)
            ? ConstValue::mkFloat(old.toFloat() + rval->toFloat())
            : ConstValue::mkInt(old.toInt() + rval->toInt());
        break;
    }
    case AssignOp::SubAssign: {
        auto old = getOld();
        newVal = ConstValue::mkInt(old.toInt() - rval->toInt());
        break;
    }
    case AssignOp::MulAssign: {
        auto old = getOld();
        newVal = ConstValue::mkInt(old.toInt() * rval->toInt());
        break;
    }
    case AssignOp::DivAssign: {
        auto old = getOld();
        int64_t r = rval->toInt();
        if (!r) { diag_.error(e.loc, "division by zero in const context"); return {}; }
        newVal = ConstValue::mkInt(old.toInt() / r);
        break;
    }
    case AssignOp::ModAssign: {
        auto old = getOld();
        int64_t r = rval->toInt();
        if (!r) { diag_.error(e.loc, "modulo by zero in const context"); return {}; }
        newVal = ConstValue::mkInt(old.toInt() % r);
        break;
    }
    case AssignOp::AndAssign:
        newVal = ConstValue::mkInt(getOld().toInt() & rval->toInt()); break;
    case AssignOp::OrAssign:
        newVal = ConstValue::mkInt(getOld().toInt() | rval->toInt()); break;
    case AssignOp::XorAssign:
        newVal = ConstValue::mkInt(getOld().toInt() ^ rval->toInt()); break;
    case AssignOp::ShlAssign:
        newVal = ConstValue::mkInt(getOld().toInt() << rval->toInt()); break;
    case AssignOp::ShrAssign:
        newVal = ConstValue::mkInt(getOld().toInt() >> rval->toInt()); break;
    }

    frame.locals[lhsName] = newVal;
    return newVal;
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
    ConstValue val = ConstValue::mkVoid();
    if (s.init) {
        auto v = evalExprF(*s.init, frame);
        if (!v) return false;
        val = *v;
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
        return true;  // sizeof is always constant
    default:
        return false;
    }
}

} // namespace safec
