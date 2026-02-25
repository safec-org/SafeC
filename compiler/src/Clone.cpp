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
    // An undefined struct name inside a function body may represent a generic
    // type parameter (the parser emits it as StructType; resolveGenericNames
    // converts it for param/return types but not for body-level cast targets).
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*ty);
        if (!st.isDefined) {
            auto it = subs.find(st.name);
            if (it != subs.end()) return it->second;
        }
        return ty;
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
    case TypeKind::Tuple: {
        auto &tt = static_cast<const TupleType &>(*ty);
        bool changed = false;
        std::vector<TypePtr> ne;
        for (auto &e : tt.elementTypes) {
            ne.push_back(substituteType(e, subs));
            if (ne.back() != e) changed = true;
        }
        return changed ? makeTuple(std::move(ne)) : ty;
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
        for (auto &a : ce.args) {
            // Pack expansion: if arg is an ident matching a pack param name,
            // expand to multiple args (args0, args1, ...)
            if (a && a->kind == ExprKind::Ident) {
                auto &id = static_cast<const IdentExpr &>(*a);
                std::string countKey = "__pack_count_" + id.name;
                auto pit = subs.find(countKey);
                if (pit != subs.end()) {
                    // pit->second is an Int type whose name encodes the count
                    int count = 0;
                    if (pit->second && pit->second->kind == TypeKind::Struct) {
                        // We stored count in StructType name as a number string
                        auto &st = static_cast<const StructType &>(*pit->second);
                        count = std::atoi(st.name.c_str());
                    }
                    for (int pi = 0; pi < count; ++pi) {
                        args.push_back(std::make_unique<IdentExpr>(
                            id.name + std::to_string(pi), id.loc));
                    }
                    continue;
                }
            }
            args.push_back(cloneExprImpl(a.get(), subs));
        }
        auto cloned = std::make_unique<CallExpr>(
            cloneExprImpl(ce.callee.get(), subs), std::move(args), ce.loc);
        cloned->taggedUnionTag = ce.taggedUnionTag;
        cloned->taggedUnionName = ce.taggedUnionName;
        return fix(std::move(cloned));
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
    case ExprKind::AlignofType: {
        auto &ae = static_cast<const AlignofTypeExpr &>(e);
        return fix(std::make_unique<AlignofTypeExpr>(
            substituteType(ae.ofType, subs), ae.loc));
    }
    case ExprKind::FieldCount: {
        auto &fc = static_cast<const FieldCountExpr &>(e);
        return fix(std::make_unique<FieldCountExpr>(
            substituteType(fc.ofType, subs), fc.loc));
    }
    case ExprKind::SizeofPack: {
        auto &sp = static_cast<const SizeofPackExpr &>(e);
        auto clone = std::make_unique<SizeofPackExpr>(sp.packName, sp.loc);
        // Count pack entries in subs
        int count = 0;
        for (int i = 0; ; ++i) {
            if (subs.count(sp.packName + "__" + std::to_string(i))) ++count;
            else break;
        }
        clone->resolvedCount = count;
        return fix(std::move(clone));
    }
    case ExprKind::TaggedUnionInit: {
        auto &tui = static_cast<const TaggedUnionInitExpr &>(e);
        return fix(std::make_unique<TaggedUnionInitExpr>(
            tui.unionName, tui.variantName, tui.tag,
            cloneExprImpl(tui.payload.get(), subs), tui.loc));
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
    case ExprKind::TupleLit: {
        auto &te = static_cast<const TupleLitExpr &>(e);
        std::vector<ExprPtr> elems;
        for (auto &el : te.elements) elems.push_back(cloneExprImpl(el.get(), subs));
        return fix(std::make_unique<TupleLitExpr>(std::move(elems), te.loc));
    }
    case ExprKind::New: {
        auto &ne = static_cast<const NewExpr &>(e);
        return fix(std::make_unique<NewExpr>(
            ne.regionName, substituteType(ne.allocType, subs), ne.loc));
    }
    case ExprKind::Spawn: {
        auto &se = static_cast<const SpawnExpr &>(e);
        return fix(std::make_unique<SpawnExpr>(
            cloneExprImpl(se.fnExpr.get(), subs),
            cloneExprImpl(se.argExpr.get(), subs), se.loc));
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
    clone->isConst       = fn.isConst;
    clone->isConsteval   = fn.isConsteval;
    clone->isInline      = fn.isInline;
    clone->isExtern      = fn.isExtern;
    clone->isVariadic    = fn.isVariadic;
    clone->isMethod      = fn.isMethod;
    clone->methodOwner   = fn.methodOwner;
    clone->isConstMethod = fn.isConstMethod;
    // caller clears genericParams and sets mangled name

    // Mutable copy of subs for pack expansion metadata
    TypeSubst bodySubs = subs;

    for (auto &p : fn.params) {
        if (p.isPack) {
            // Expand pack parameter into N concrete params
            std::string packTypeName;
            if (p.type && p.type->kind == TypeKind::Generic)
                packTypeName = static_cast<const GenericType &>(*p.type).name;
            int count = 0;
            for (int i = 0; ; ++i) {
                auto it = bodySubs.find(packTypeName + "__" + std::to_string(i));
                if (it == bodySubs.end()) break;
                ParamDecl pd;
                pd.name = p.name + std::to_string(i);
                pd.type = it->second;
                pd.loc  = p.loc;
                clone->params.push_back(std::move(pd));
                ++count;
            }
            // Store pack count for body cloning (CallExpr expansion)
            // Use StructType name to encode the count as a string
            auto countTy = std::make_shared<StructType>(std::to_string(count), false);
            bodySubs["__pack_count_" + p.name] = countTy;
        } else {
            ParamDecl pd;
            pd.name = p.name;
            pd.type = substituteType(p.type, bodySubs);
            pd.loc  = p.loc;
            clone->params.push_back(std::move(pd));
        }
    }

    if (fn.body) {
        // Clone the body using the raw-pointer helper (no ownership transfer)
        auto bodyClone = cloneStmtImpl(fn.body.get(), bodySubs);
        clone->body.reset(static_cast<CompoundStmt *>(bodyClone.release()));
    }

    return clone;
}

} // namespace safec
