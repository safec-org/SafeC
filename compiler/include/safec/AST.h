#pragma once
#include "safec/Diagnostic.h"
#include "safec/Type.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace safec {

// ── Forward declarations ──────────────────────────────────────────────────────
struct Stmt;  struct Expr;  struct Decl;
struct FunctionDecl; struct StructDecl; struct VarDecl;
using StmtPtr = std::unique_ptr<Stmt>;
using ExprPtr = std::unique_ptr<Expr>;
using DeclPtr = std::unique_ptr<Decl>;

// ── Visitor interfaces (forward declared) ─────────────────────────────────────
struct ASTVisitor;

// =============================================================================
// EXPRESSIONS
// =============================================================================

enum class ExprKind {
    IntLit, FloatLit, BoolLit, StringLit, CharLit, NullLit,
    Ident,          // variable / function name
    Unary,          // -x, !x, ~x, &x, *x, ++x, x++, sizeof(x)
    Binary,         // x + y, x == y, ...
    Ternary,        // cond ? a : b
    Call,           // f(args)
    Subscript,      // a[i]
    Member,         // s.field
    Arrow,          // p->field
    Cast,           // (T)expr
    SizeofType,     // sizeof(T)
    Assign,         // x = y, x += y, ...
    Compound,       // initializer: {a, b, c}
    AddrOf,         // address-of: &x  (in safe context → &stack ref)
    Deref,          // dereference: *p
};

enum class UnaryOp {
    Neg, Not, BitNot,
    PreInc, PreDec, PostInc, PostDec,
    SizeofExpr,
};

enum class BinaryOp {
    Add, Sub, Mul, Div, Mod,
    BitAnd, BitOr, BitXor, Shl, Shr,
    LogAnd, LogOr,
    Eq, NEq, Lt, Gt, LEq, GEq,
    Comma,
};

enum class AssignOp {
    Assign,
    AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
    AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign,
};

// ── Base expression ───────────────────────────────────────────────────────────
struct Expr {
    ExprKind       kind;
    SourceLocation loc;
    TypePtr        type;  // resolved by sema; null before type-check
    bool           isLValue = false;

    explicit Expr(ExprKind k, SourceLocation l) : kind(k), loc(l) {}
    virtual ~Expr() = default;
};

// ── Literal expressions ───────────────────────────────────────────────────────
struct IntLitExpr : Expr {
    int64_t value;
    IntLitExpr(int64_t v, SourceLocation l)
        : Expr(ExprKind::IntLit, l), value(v) {}
};

struct FloatLitExpr : Expr {
    double value;
    FloatLitExpr(double v, SourceLocation l)
        : Expr(ExprKind::FloatLit, l), value(v) {}
};

struct BoolLitExpr : Expr {
    bool value;
    BoolLitExpr(bool v, SourceLocation l)
        : Expr(ExprKind::BoolLit, l), value(v) {}
};

struct StringLitExpr : Expr {
    std::string value;
    StringLitExpr(std::string v, SourceLocation l)
        : Expr(ExprKind::StringLit, l), value(std::move(v)) {}
};

struct CharLitExpr : Expr {
    char value;
    CharLitExpr(char v, SourceLocation l)
        : Expr(ExprKind::CharLit, l), value(v) {}
};

struct NullLitExpr : Expr {
    explicit NullLitExpr(SourceLocation l) : Expr(ExprKind::NullLit, l) {}
};

// ── Identifier reference ──────────────────────────────────────────────────────
struct IdentExpr : Expr {
    std::string name;
    VarDecl    *resolved = nullptr;   // filled by name resolution
    FunctionDecl *resolvedFn = nullptr;
    IdentExpr(std::string n, SourceLocation l)
        : Expr(ExprKind::Ident, l), name(std::move(n)) {}
};

// ── Unary expression ──────────────────────────────────────────────────────────
struct UnaryExpr : Expr {
    UnaryOp op;
    ExprPtr operand;
    UnaryExpr(UnaryOp o, ExprPtr e, SourceLocation l)
        : Expr(ExprKind::Unary, l), op(o), operand(std::move(e)) {}
};

// ── Binary expression ─────────────────────────────────────────────────────────
struct BinaryExpr : Expr {
    BinaryOp op;
    ExprPtr  left, right;
    BinaryExpr(BinaryOp o, ExprPtr l, ExprPtr r, SourceLocation loc)
        : Expr(ExprKind::Binary, loc), op(o),
          left(std::move(l)), right(std::move(r)) {}
};

// ── Ternary ───────────────────────────────────────────────────────────────────
struct TernaryExpr : Expr {
    ExprPtr cond, then, else_;
    TernaryExpr(ExprPtr c, ExprPtr t, ExprPtr e, SourceLocation l)
        : Expr(ExprKind::Ternary, l), cond(std::move(c)),
          then(std::move(t)), else_(std::move(e)) {}
};

// ── Function call ─────────────────────────────────────────────────────────────
struct CallExpr : Expr {
    ExprPtr              callee;
    std::vector<ExprPtr> args;
    CallExpr(ExprPtr fn, std::vector<ExprPtr> a, SourceLocation l)
        : Expr(ExprKind::Call, l), callee(std::move(fn)),
          args(std::move(a)) {}
};

// ── Subscript ─────────────────────────────────────────────────────────────────
struct SubscriptExpr : Expr {
    ExprPtr base, index;
    SubscriptExpr(ExprPtr b, ExprPtr i, SourceLocation l)
        : Expr(ExprKind::Subscript, l), base(std::move(b)), index(std::move(i)) {}
};

// ── Member access ─────────────────────────────────────────────────────────────
struct MemberExpr : Expr {
    ExprPtr     base;
    std::string field;
    bool        isArrow;  // true → ptr->field, false → val.field
    MemberExpr(ExprPtr b, std::string f, bool arrow, SourceLocation l)
        : Expr(arrow ? ExprKind::Arrow : ExprKind::Member, l),
          base(std::move(b)), field(std::move(f)), isArrow(arrow) {}
};

// ── Cast ──────────────────────────────────────────────────────────────────────
struct CastExpr : Expr {
    TypePtr targetType;
    ExprPtr operand;
    CastExpr(TypePtr t, ExprPtr e, SourceLocation l)
        : Expr(ExprKind::Cast, l), targetType(std::move(t)),
          operand(std::move(e)) {}
};

// ── sizeof(type) ──────────────────────────────────────────────────────────────
struct SizeofTypeExpr : Expr {
    TypePtr ofType;
    SizeofTypeExpr(TypePtr t, SourceLocation l)
        : Expr(ExprKind::SizeofType, l), ofType(std::move(t)) {}
};

// ── Assignment ────────────────────────────────────────────────────────────────
struct AssignExpr : Expr {
    AssignOp op;
    ExprPtr  lhs, rhs;
    AssignExpr(AssignOp o, ExprPtr l, ExprPtr r, SourceLocation loc)
        : Expr(ExprKind::Assign, loc), op(o),
          lhs(std::move(l)), rhs(std::move(r)) {}
};

// ── Compound initializer {a, b, c} ────────────────────────────────────────────
struct CompoundInitExpr : Expr {
    std::vector<ExprPtr> inits;
    CompoundInitExpr(std::vector<ExprPtr> v, SourceLocation l)
        : Expr(ExprKind::Compound, l), inits(std::move(v)) {}
};

// =============================================================================
// STATEMENTS
// =============================================================================

enum class StmtKind {
    Compound, Expr, If, While, DoWhile, For, Return,
    Break, Continue, Goto, Label,
    VarDecl,       // local variable declaration
    Unsafe,        // unsafe { ... }
    StaticAssert,  // static_assert(cond[, "msg"])
};

struct Stmt {
    StmtKind       kind;
    SourceLocation loc;
    explicit Stmt(StmtKind k, SourceLocation l) : kind(k), loc(l) {}
    virtual ~Stmt() = default;
};

struct CompoundStmt : Stmt {
    std::vector<StmtPtr> body;
    explicit CompoundStmt(SourceLocation l) : Stmt(StmtKind::Compound, l) {}
};

struct ExprStmt : Stmt {
    ExprPtr expr;
    ExprStmt(ExprPtr e, SourceLocation l)
        : Stmt(StmtKind::Expr, l), expr(std::move(e)) {}
};

struct IfStmt : Stmt {
    ExprPtr cond;
    StmtPtr then, else_;
    IfStmt(ExprPtr c, StmtPtr t, StmtPtr e, SourceLocation l)
        : Stmt(StmtKind::If, l), cond(std::move(c)),
          then(std::move(t)), else_(std::move(e)) {}
};

struct WhileStmt : Stmt {
    ExprPtr cond;
    StmtPtr body;
    bool    isDoWhile;
    WhileStmt(ExprPtr c, StmtPtr b, bool doWhile, SourceLocation l)
        : Stmt(doWhile ? StmtKind::DoWhile : StmtKind::While, l),
          cond(std::move(c)), body(std::move(b)), isDoWhile(doWhile) {}
};

struct ForStmt : Stmt {
    StmtPtr init;   // can be VarDeclStmt or ExprStmt
    ExprPtr cond;   // nullable
    ExprPtr incr;   // nullable
    StmtPtr body;
    ForStmt(StmtPtr i, ExprPtr c, ExprPtr inc, StmtPtr b, SourceLocation l)
        : Stmt(StmtKind::For, l), init(std::move(i)), cond(std::move(c)),
          incr(std::move(inc)), body(std::move(b)) {}
};

struct ReturnStmt : Stmt {
    ExprPtr value;  // nullable for void returns
    ReturnStmt(ExprPtr v, SourceLocation l)
        : Stmt(StmtKind::Return, l), value(std::move(v)) {}
};

struct BreakStmt    : Stmt { explicit BreakStmt(SourceLocation l)    : Stmt(StmtKind::Break, l) {} };
struct ContinueStmt : Stmt { explicit ContinueStmt(SourceLocation l) : Stmt(StmtKind::Continue, l) {} };

struct GotoStmt : Stmt {
    std::string label;
    GotoStmt(std::string lbl, SourceLocation l)
        : Stmt(StmtKind::Goto, l), label(std::move(lbl)) {}
};

struct LabelStmt : Stmt {
    std::string label;
    StmtPtr     body;
    LabelStmt(std::string lbl, StmtPtr b, SourceLocation l)
        : Stmt(StmtKind::Label, l), label(std::move(lbl)), body(std::move(b)) {}
};

// ── Local variable declaration ────────────────────────────────────────────────
struct VarDeclStmt : Stmt {
    std::string name;
    TypePtr     declType;   // null → infer (future)
    ExprPtr     init;       // nullable
    bool        isConst    = false;
    bool        isStatic   = false;

    VarDeclStmt(std::string n, TypePtr t, ExprPtr i, SourceLocation l)
        : Stmt(StmtKind::VarDecl, l), name(std::move(n)),
          declType(std::move(t)), init(std::move(i)) {}

    // Filled by sema for codegen
    TypePtr resolvedType;
};

// ── unsafe { ... } ───────────────────────────────────────────────────────────
struct UnsafeStmt : Stmt {
    StmtPtr body;
    explicit UnsafeStmt(StmtPtr b, SourceLocation l)
        : Stmt(StmtKind::Unsafe, l), body(std::move(b)) {}
};

// ── static_assert ─────────────────────────────────────────────────────────────
struct StaticAssertStmt : Stmt {
    ExprPtr     cond;
    std::string message;
    StaticAssertStmt(ExprPtr c, std::string m, SourceLocation l)
        : Stmt(StmtKind::StaticAssert, l), cond(std::move(c)),
          message(std::move(m)) {}
};

// =============================================================================
// TOP-LEVEL DECLARATIONS
// =============================================================================

enum class DeclKind {
    Function, Struct, Enum, Region, GlobalVar, TypeAlias, StaticAssert
};

struct Decl {
    DeclKind       kind;
    SourceLocation loc;
    std::string    name;
    explicit Decl(DeclKind k, std::string n, SourceLocation l)
        : kind(k), loc(l), name(std::move(n)) {}
    virtual ~Decl() = default;
};

// ── Parameter ─────────────────────────────────────────────────────────────────
struct ParamDecl {
    std::string name;
    TypePtr     type;
    SourceLocation loc;
};

// ── Generic parameter: <T: Constraint> ───────────────────────────────────────
struct GenericParam {
    std::string name;
    std::string constraint; // e.g., "Numeric"; empty = unconstrained
};

// ── Function declaration ──────────────────────────────────────────────────────
struct FunctionDecl : Decl {
    TypePtr                  returnType;
    std::vector<ParamDecl>   params;
    std::vector<GenericParam> genericParams;
    std::unique_ptr<CompoundStmt> body;  // null → declaration only
    bool  isConst    = false;  // const fn
    bool  isConsteval = false;
    bool  isInline   = false;
    bool  isExtern   = false;
    bool  isVariadic = false;

    FunctionDecl(std::string n, SourceLocation l)
        : Decl(DeclKind::Function, std::move(n), l) {}

    TypePtr funcType() const;
};

// ── Struct declaration ────────────────────────────────────────────────────────
struct StructDecl : Decl {
    std::vector<FieldDecl> fields;
    bool isUnion   = false;
    bool isTaggedUnion = false;
    std::vector<GenericParam> genericParams;
    std::shared_ptr<StructType> type;   // filled by sema

    StructDecl(std::string n, SourceLocation l)
        : Decl(DeclKind::Struct, std::move(n), l) {}
};

// ── Enum declaration ──────────────────────────────────────────────────────────
struct EnumDecl : Decl {
    std::vector<std::pair<std::string, std::optional<int64_t>>> enumerators;
    std::shared_ptr<EnumType> type;   // filled by sema

    EnumDecl(std::string n, SourceLocation l)
        : Decl(DeclKind::Enum, std::move(n), l) {}
};

// ── Region declaration ────────────────────────────────────────────────────────
struct RegionDecl : Decl {
    int64_t capacity = 0;  // from "capacity: N"

    RegionDecl(std::string n, SourceLocation l)
        : Decl(DeclKind::Region, std::move(n), l) {}
};

// ── Global variable declaration ───────────────────────────────────────────────
struct GlobalVarDecl : Decl {
    TypePtr type;
    ExprPtr init;
    bool    isConst  = false;
    bool    isStatic = false;
    bool    isExtern = false;

    GlobalVarDecl(std::string n, SourceLocation l)
        : Decl(DeclKind::GlobalVar, std::move(n), l) {}
};

// ── Type alias ────────────────────────────────────────────────────────────────
struct TypeAliasDecl : Decl {
    TypePtr aliasedType;
    TypeAliasDecl(std::string n, TypePtr t, SourceLocation l)
        : Decl(DeclKind::TypeAlias, std::move(n), l),
          aliasedType(std::move(t)) {}
};

// ── static_assert at top level ────────────────────────────────────────────────
struct StaticAssertDecl : Decl {
    ExprPtr     cond;
    std::string message;
    StaticAssertDecl(ExprPtr c, std::string m, SourceLocation l)
        : Decl(DeclKind::StaticAssert, "", l),
          cond(std::move(c)), message(std::move(m)) {}
};

// =============================================================================
// TRANSLATION UNIT
// =============================================================================

struct TranslationUnit {
    std::vector<DeclPtr> decls;
    const char          *filename = nullptr;
};

// ── Variable reference (for sema symbol table) ────────────────────────────────
struct VarDecl {
    std::string name;
    TypePtr     type;
    bool        isParam   = false;
    bool        isConst   = false;
    bool        isStatic  = false;
    bool        isGlobal  = false;
    bool        initialized = false;
    // For region safety: if this is a reference, which scope depth it came from
    int         scopeDepth  = 0;
    // Codegen slot
    void       *llvmAlloca  = nullptr;  // llvm::AllocaInst*
};

} // namespace safec
