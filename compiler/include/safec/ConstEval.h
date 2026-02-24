#pragma once
#include "safec/AST.h"
#include "safec/Diagnostic.h"
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>

namespace safec {

// =============================================================================
// ConstValue — the compile-time value representation
// =============================================================================
struct ConstValue {
    enum Kind { Void, Int, Float, Bool, String };

    Kind    kind      = Void;
    int64_t intVal    = 0;
    double  floatVal  = 0.0;
    bool    boolVal   = false;
    std::string strVal;

    static ConstValue mkVoid()              { return {}; }
    static ConstValue mkInt(int64_t v)      { ConstValue c; c.kind = Int;    c.intVal   = v; return c; }
    static ConstValue mkFloat(double v)     { ConstValue c; c.kind = Float;  c.floatVal = v; return c; }
    static ConstValue mkBool(bool v)        { ConstValue c; c.kind = Bool;   c.boolVal  = v; return c; }
    static ConstValue mkString(std::string s) {
        ConstValue c; c.kind = String; c.strVal = std::move(s); return c;
    }

    // Coercions (for use in arithmetic / conditions)
    int64_t     toInt()   const;
    double      toFloat() const;
    bool        toBool()  const;
    std::string typeStr() const;
    std::string repr()    const;

    bool isNumeric() const { return kind == Int || kind == Float || kind == Bool; }
};

// =============================================================================
// ConstEvalEngine — sandboxed compile-time interpreter
// =============================================================================
// Responsibilities:
//   1. Evaluate const/consteval function bodies
//   2. Evaluate static_assert conditions → emit error if false
//   3. Evaluate const global variable initializers
//   4. Enforce that consteval functions are not called in runtime context
//   5. Provide evalExpr() for use by Sema (static_assert, array bounds, enum values)
//
// Safety limits (CONST_EVAL.md §10):
//   - Recursion depth:   kMaxRecursion  = 256
//   - Loop iterations:   kLoopLimit     = 1,000,000
//   - Instruction budget:kInstrBudget   = 10,000,000
class ConstEvalEngine {
public:
    ConstEvalEngine(TranslationUnit &tu, DiagEngine &diag);

    // Run the const-eval pass over the whole TU.
    // - Evaluates all static_assert nodes
    // - Evaluates all const global initializers (stores result back on GlobalVarDecl)
    // - Enforces consteval-only-in-const-context rule
    // Returns false if any error was emitted.
    bool run();

    // Evaluate a single expression in a const context.
    // The expression must have its types resolved (run after sema).
    // Returns nullopt on error (diagnostic already emitted).
    std::optional<ConstValue> evalExpr(const Expr &e);

    // Evaluate a consteval/const function call with given arguments.
    std::optional<ConstValue> evalCall(const FunctionDecl &fn,
                                        std::vector<ConstValue> args,
                                        SourceLocation callLoc);

    // Check whether an expression is a valid constant expression (no side-effects,
    // no runtime-only calls).  Diagnoses if not.
    bool isConstExpr(const Expr &e) const;

    // Retrieve the pre-evaluated constant for a const global variable (if any).
    // Returns nullopt if not pre-evaluated.
    std::optional<ConstValue> getGlobalConst(const std::string &name) const;

private:
    // ── Execution frame ───────────────────────────────────────────────────────
    struct Frame {
        std::unordered_map<std::string, ConstValue> locals;
        bool      hasReturn = false;
        ConstValue returnVal;
    };

    // ── Expression evaluation ─────────────────────────────────────────────────
    std::optional<ConstValue> evalExprF(const Expr &e, Frame &frame);
    std::optional<ConstValue> evalIntLit   (const IntLitExpr   &e);
    std::optional<ConstValue> evalFloatLit (const FloatLitExpr &e);
    std::optional<ConstValue> evalBoolLit  (const BoolLitExpr  &e);
    std::optional<ConstValue> evalStringLit(const StringLitExpr &e);
    std::optional<ConstValue> evalIdent    (const IdentExpr &e, Frame &frame);
    std::optional<ConstValue> evalUnary    (const UnaryExpr &e, Frame &frame);
    std::optional<ConstValue> evalBinary   (const BinaryExpr &e, Frame &frame);
    std::optional<ConstValue> evalTernary  (const TernaryExpr &e, Frame &frame);
    std::optional<ConstValue> evalCallExpr (const CallExpr &e, Frame &frame);
    std::optional<ConstValue> evalCast     (const CastExpr &e, Frame &frame);
    std::optional<ConstValue> evalAssign   (const AssignExpr &e, Frame &frame);

    // ── Statement evaluation ──────────────────────────────────────────────────
    // Returns false if a hard error was hit (budget exceeded etc.)
    bool evalStmt (const Stmt &s, Frame &frame);
    bool evalCompound(const CompoundStmt &s, Frame &frame);
    bool evalIf      (const IfStmt &s, Frame &frame);
    bool evalWhile   (const WhileStmt &s, Frame &frame);
    bool evalFor     (const ForStmt &s, Frame &frame);
    bool evalReturn  (const ReturnStmt &s, Frame &frame);
    bool evalVarDecl (const VarDeclStmt &s, Frame &frame);

    // ── TU-level passes ───────────────────────────────────────────────────────
    void checkFunctionForConstevalCalls(const FunctionDecl &fn, bool inConstContext);
    void checkStmtForConstevalCalls(const Stmt &s, bool inConstContext);
    void checkExprForConstevalCalls(const Expr &e, bool inConstContext);
    void resolveIfConstInStmt(Stmt &s);  // fill constResult on IfConstStmt nodes

    // ── Helpers ───────────────────────────────────────────────────────────────
    FunctionDecl *lookupFn(const std::string &name) const;
    bool          isBudgetExceeded(SourceLocation loc);
    void          tickInstr(int n = 1) { instrBudget_ -= n; }

    // ── Limits ────────────────────────────────────────────────────────────────
    static constexpr int     kMaxRecursion = 256;
    static constexpr int64_t kLoopLimit    = 1'000'000LL;
    static constexpr int64_t kInstrBudget  = 10'000'000LL;

    // ── State ─────────────────────────────────────────────────────────────────
    TranslationUnit &tu_;
    DiagEngine      &diag_;

    std::unordered_map<std::string, FunctionDecl*> fnTable_;
    std::unordered_map<std::string, ConstValue>    globalConsts_; // pre-evaluated global consts

    int     recursionDepth_ = 0;
    int64_t instrBudget_    = kInstrBudget;

    // For loop break/continue (simple flag: break exits innermost loop)
    bool breakFlag_    = false;
    bool continueFlag_ = false;
};

} // namespace safec
