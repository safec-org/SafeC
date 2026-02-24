#include "safec/Clone.h"
#include <cassert>

namespace safec {

// ── Type substitution ─────────────────────────────────────────────────────────
TypePtr substituteType(const TypePtr &ty, const TypeSubst &subs) {
    if (!ty) return nullptr;
    switch (ty->kind) {
    case TypeKind::Generic: {
        auto &gt = static_cast<const GenericType &>(*ty);
        auto it = subs.find(gt.name);
        return (it != subs.end()) ? it->second : ty;
    }
    case TypeKind::Pointer: {
        auto &pt = static_cast<const PointerType &>(*ty);
        auto nb = substituteType(pt.base, subs);
        return (nb == pt.base) ? ty : makePointer(nb, pt.isConst);
    }
    case TypeKind::Reference: {
        auto &rt = static_cast<const ReferenceType &>(*ty);
        auto nb = substituteType(rt.base, subs);
        return (nb == rt.base) ? ty
             : makeReference(nb, rt.region, rt.nullable, rt.mut, rt.arenaName);
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(*ty);
        auto ne = substituteType(at.element, subs);
        return (ne == at.element) ? ty : makeArray(ne, at.size);
    }
    case TypeKind::Function: {
        auto &ft = static_cast<const FunctionType &>(*ty);
        auto nr = substituteType(ft.returnType, subs);
        bool changed = (nr != ft.returnType);
        std::vector<TypePtr> np;
        for (auto &p : ft.paramTypes) {
            np.push_back(substituteType(p, subs));
            if (np.back() != p) changed = true;
        }
        return changed ? makeFunction(nr, std::move(np), ft.variadic) : ty;
    }
    default:
        return ty;
    }
}

// ── Expression cloning ────────────────────────────────────────────────────────
// Forward-declare the impl so cloneExpr/cloneStmt can call each other.
static ExprPtr cloneExprImpl(const Expr *e, const TypeSubst &subs);
static StmtPtr cloneStmtImpl(const Stmt *s, const TypeSubst &subs);

ExprPtr cloneExpr(const ExprPtr &ep, const TypeSubst &subs) {
    return cloneExprImpl(ep.get(), subs);
}

StmtPtr cloneStmt(const StmtPtr &sp, const TypeSubst &subs) {
    return cloneStmtImpl(sp.get(), subs);
}

static ExprPtr cloneExprImpl(const Expr *ep, const TypeSubst &subs) {
    if (!ep) return nullptr;
    const Expr &e = *ep;

    // After building the clone, copy base-class fields.
    auto fix = [&](ExprPtr res) -> ExprPtr {
        if (res && e.type) res->type = substituteType(e.type, subs);
        if (res) res->isLValue = e.isLValue;
        return res;
    };

    switch (e.kind) {
    case ExprKind::IntLit:
        return fix(std::make_unique<IntLitExpr>(
            static_cast<const IntLitExpr &>(e).value, e.loc));
    case ExprKind::FloatLit:
        return fix(std::make_unique<FloatLitExpr>(
            static_cast<const FloatLitExpr &>(e).value, e.loc));
    case ExprKind::BoolLit:
        return fix(std::make_unique<BoolLitExpr>(
            static_cast<const BoolLitExpr &>(e).value, e.loc));
    case ExprKind::StringLit:
        return fix(std::make_unique<StringLitExpr>(
            static_cast<const StringLitExpr &>(e).value, e.loc));
    case ExprKind::CharLit:
        return fix(std::make_unique<CharLitExpr>(
            static_cast<const CharLitExpr &>(e).value, e.loc));
    case ExprKind::NullLit:
        return fix(std::make_unique<NullLitExpr>(e.loc));
    case ExprKind::Ident: {
        auto &ie = static_cast<const IdentExpr &>(e);
        // resolved/resolvedFn left null: sema will re-resolve in the clone body
        return fix(std::make_unique<IdentExpr>(ie.name, ie.loc));
    }
    case ExprKind::Unary:
    case ExprKind::AddrOf:
    case ExprKind::Deref: {
        auto &ue = static_cast<const UnaryExpr &>(e);
        auto res = std::make_unique<UnaryExpr>(
            ue.op, cloneExprImpl(ue.operand.get(), subs), ue.loc);
        res->kind = e.kind; // preserve AddrOf / Deref distinction
        return fix(std::move(res));
    }
    case ExprKind::Binary: {
        auto &be = static_cast<const BinaryExpr &>(e);
        return fix(std::make_unique<BinaryExpr>(
            be.op,
            cloneExprImpl(be.left.get(), subs),
            cloneExprImpl(be.right.get(), subs),
            be.loc));
    }
    case ExprKind::Ternary: {
        auto &te = static_cast<const TernaryExpr &>(e);
        return fix(std::make_unique<TernaryExpr>(
            cloneExprImpl(te.cond.get(), subs),
            cloneExprImpl(te.then.get(), subs),
            cloneExprImpl(te.else_.get(), subs),
            te.loc));
    }
    case ExprKind::Call: {
        auto &ce = static_cast<const CallExpr &>(e);
        std::vector<ExprPtr> args;
        for (auto &a : ce.args) args.push_back(cloneExprImpl(a.get(), subs));
        return fix(std::make_unique<CallExpr>(
            cloneExprImpl(ce.callee.get(), subs), std::move(args), ce.loc));
    }
    case ExprKind::Subscript: {
        auto &se = static_cast<const SubscriptExpr &>(e);
        auto res = std::make_unique<SubscriptExpr>(
            cloneExprImpl(se.base.get(), subs),
            cloneExprImpl(se.index.get(), subs),
            se.loc);
        res->boundsCheckOmit = se.boundsCheckOmit;
        return fix(std::move(res));
    }
    case ExprKind::Member:
    case ExprKind::Arrow: {
        auto &me = static_cast<const MemberExpr &>(e);
        return fix(std::make_unique<MemberExpr>(
            cloneExprImpl(me.base.get(), subs), me.field, me.isArrow, me.loc));
    }
    case ExprKind::Cast: {
        auto &ce = static_cast<const CastExpr &>(e);
        return fix(std::make_unique<CastExpr>(
            substituteType(ce.targetType, subs),
            cloneExprImpl(ce.operand.get(), subs),
            ce.loc));
    }
    case ExprKind::SizeofType: {
        auto &se = static_cast<const SizeofTypeExpr &>(e);
        return fix(std::make_unique<SizeofTypeExpr>(
            substituteType(se.ofType, subs), se.loc));
    }
    case ExprKind::Assign: {
        auto &ae = static_cast<const AssignExpr &>(e);
        return fix(std::make_unique<AssignExpr>(
            ae.op,
            cloneExprImpl(ae.lhs.get(), subs),
            cloneExprImpl(ae.rhs.get(), subs),
            ae.loc));
    }
    case ExprKind::Compound: {
        auto &ce = static_cast<const CompoundInitExpr &>(e);
        std::vector<ExprPtr> inits;
        for (auto &i : ce.inits) inits.push_back(cloneExprImpl(i.get(), subs));
        return fix(std::make_unique<CompoundInitExpr>(std::move(inits), ce.loc));
    }
    default:
        return fix(std::make_unique<NullLitExpr>(e.loc));
    }
}

// ── Statement cloning ─────────────────────────────────────────────────────────
static StmtPtr cloneStmtImpl(const Stmt *sp, const TypeSubst &subs) {
    if (!sp) return nullptr;
    const Stmt &s = *sp;

    switch (s.kind) {
    case StmtKind::Compound: {
        auto &cs = static_cast<const CompoundStmt &>(s);
        auto res = std::make_unique<CompoundStmt>(cs.loc);
        for (auto &st : cs.body)
            res->body.push_back(cloneStmtImpl(st.get(), subs));
        return res;
    }
    case StmtKind::Expr: {
        auto &es = static_cast<const ExprStmt &>(s);
        return std::make_unique<ExprStmt>(
            cloneExprImpl(es.expr.get(), subs), es.loc);
    }
    case StmtKind::If: {
        auto &is = static_cast<const IfStmt &>(s);
        return std::make_unique<IfStmt>(
            cloneExprImpl(is.cond.get(), subs),
            cloneStmtImpl(is.then.get(), subs),
            cloneStmtImpl(is.else_.get(), subs),
            is.loc);
    }
    case StmtKind::While:
    case StmtKind::DoWhile: {
        auto &ws = static_cast<const WhileStmt &>(s);
        return std::make_unique<WhileStmt>(
            cloneExprImpl(ws.cond.get(), subs),
            cloneStmtImpl(ws.body.get(), subs),
            ws.isDoWhile, ws.loc);
    }
    case StmtKind::For: {
        auto &fs = static_cast<const ForStmt &>(s);
        return std::make_unique<ForStmt>(
            cloneStmtImpl(fs.init.get(), subs),
            cloneExprImpl(fs.cond.get(), subs),
            cloneExprImpl(fs.incr.get(), subs),
            cloneStmtImpl(fs.body.get(), subs),
            fs.loc);
    }
    case StmtKind::Return: {
        auto &rs = static_cast<const ReturnStmt &>(s);
        return std::make_unique<ReturnStmt>(
            cloneExprImpl(rs.value.get(), subs), rs.loc);
    }
    case StmtKind::Break:
        return std::make_unique<BreakStmt>(s.loc);
    case StmtKind::Continue:
        return std::make_unique<ContinueStmt>(s.loc);
    case StmtKind::Goto: {
        auto &gs = static_cast<const GotoStmt &>(s);
        return std::make_unique<GotoStmt>(gs.label, gs.loc);
    }
    case StmtKind::Label: {
        auto &ls = static_cast<const LabelStmt &>(s);
        return std::make_unique<LabelStmt>(
            ls.label, cloneStmtImpl(ls.body.get(), subs), ls.loc);
    }
    case StmtKind::VarDecl: {
        auto &vs = static_cast<const VarDeclStmt &>(s);
        auto res = std::make_unique<VarDeclStmt>(
            vs.name,
            substituteType(vs.declType, subs),
            cloneExprImpl(vs.init.get(), subs),
            vs.loc);
        res->isConst  = vs.isConst;
        res->isStatic = vs.isStatic;
        return res;
    }
    case StmtKind::Unsafe: {
        auto &us = static_cast<const UnsafeStmt &>(s);
        return std::make_unique<UnsafeStmt>(
            cloneStmtImpl(us.body.get(), subs), us.loc);
    }
    case StmtKind::StaticAssert: {
        auto &sa = static_cast<const StaticAssertStmt &>(s);
        return std::make_unique<StaticAssertStmt>(
            cloneExprImpl(sa.cond.get(), subs), sa.message, sa.loc);
    }
    case StmtKind::IfConst: {
        auto &ic = static_cast<const IfConstStmt &>(s);
        auto res = std::make_unique<IfConstStmt>(
            cloneExprImpl(ic.cond.get(), subs),
            cloneStmtImpl(ic.then.get(), subs),
            cloneStmtImpl(ic.else_.get(), subs),
            ic.loc);
        res->constResult = ic.constResult;
        return res;
    }
    default:
        return std::make_unique<BreakStmt>(s.loc); // safe sentinel
    }
}

// ── Function declaration cloning ──────────────────────────────────────────────
std::unique_ptr<FunctionDecl> cloneFunctionDecl(const FunctionDecl &fn,
                                                  const TypeSubst &subs) {
    auto clone = std::make_unique<FunctionDecl>(fn.name, fn.loc);
    clone->returnType  = substituteType(fn.returnType, subs);
    clone->isConst     = fn.isConst;
    clone->isConsteval = fn.isConsteval;
    clone->isInline    = fn.isInline;
    clone->isExtern    = fn.isExtern;
    clone->isVariadic  = fn.isVariadic;
    // caller clears genericParams and sets mangled name

    for (auto &p : fn.params) {
        ParamDecl pd;
        pd.name = p.name;
        pd.type = substituteType(p.type, subs);
        pd.loc  = p.loc;
        clone->params.push_back(std::move(pd));
    }

    if (fn.body) {
        // Clone the body using the raw-pointer helper (no ownership transfer)
        auto bodyClone = cloneStmtImpl(fn.body.get(), subs);
        clone->body.reset(static_cast<CompoundStmt *>(bodyClone.release()));
    }

    return clone;
}

} // namespace safec
