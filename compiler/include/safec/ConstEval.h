#pragma once
#include "safec/AST.h"
#include "safec/Diagnostic.h"
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>

namespace safec {

// =============================================================================
// ConstValue — the compile-time value representation
// =============================================================================
struct ConstValue {
    // Array/Struct: aggregate values, added so consteval/const functions can
    // build, mutate, and return structs and arrays — not just scalars. Both
    // reuse 'elems' (Array: index order: Struct: StructType::fields order,
    // resolved via the AST's already-Sema'd struct type rather than storing
    // a redundant name/shape here). There is deliberately no Pointer kind:
    // compile-time pointers would need a real compile-time memory model
    // (a heap of addressable storage, escape analysis for returned
    // addresses, aliasing) — struct/array *values* passed and returned by
    // value cover everything expressible without that, at a fraction of
    // the complexity.
    enum Kind { Void, Int, Float, Bool, String, Array, Struct };

    Kind    kind      = Void;
    int64_t intVal    = 0;
    double  floatVal  = 0.0;
    bool    boolVal   = false;
    std::string strVal;
    std::vector<ConstValue> elems;  // Array/Struct only

    static ConstValue mkVoid()              { return {}; }
    static ConstValue mkInt(int64_t v)      { ConstValue c; c.kind = Int;    c.intVal   = v; return c; }
    static ConstValue mkFloat(double v)     { ConstValue c; c.kind = Float;  c.floatVal = v; return c; }
    static ConstValue mkBool(bool v)        { ConstValue c; c.kind = Bool;   c.boolVal  = v; return c; }
    static ConstValue mkString(std::string s) {
        ConstValue c; c.kind = String; c.strVal = std::move(s); return c;
    }
    static ConstValue mkArray(std::vector<ConstValue> v) {
        ConstValue c; c.kind = Array; c.elems = std::move(v); return c;
    }
    static ConstValue mkStruct(std::vector<ConstValue> v) {
        ConstValue c; c.kind = Struct; c.elems = std::move(v); return c;
    }

    // Coercions (for use in arithmetic / conditions)
    int64_t     toInt()   const;
    double      toFloat() const;
    bool        toBool()  const;
    std::string typeStr() const;
    std::string repr()    const;

    bool isNumeric() const { return kind == Int || kind == Float || kind == Bool; }
    bool isAggregate() const { return kind == Array || kind == Struct; }
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

    // Resolves deferred array-size expressions (ArrayType::sizeExpr) across
    // every type reachable from the TU — global vars, struct fields, function
    // params/return types, and local var decls in function bodies — so that
    // sizes like 'int arr[square(3)]' (a named constant or consteval function
    // call, not a bare literal) are known before Sema runs. Must run BEFORE
    // Sema/run(), since it only needs the raw parsed AST (function lookup by
    // name, no resolved symbol table). Returns false if any size expression
    // failed to resolve to a constant (diagnostic already emitted).
    bool resolveArraySizes();

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
    std::optional<ConstValue> evalCompoundInit(const CompoundInitExpr &e, Frame &frame);
    std::optional<ConstValue> evalMember   (const MemberExpr &e, Frame &frame);
    std::optional<ConstValue> evalSubscriptConst(const SubscriptExpr &e, Frame &frame);

    // Resolves an lvalue expression (Ident, or a Member/Subscript chain
    // rooted at one) to a mutable pointer into the aggregate value stored
    // in 'frame.locals', for in-place mutation ('s.field = x', 'arr[i]++',
    // etc.). Returns nullptr (diagnostic already emitted) if 'e' isn't an
    // lvalue this engine can track — notably, anything not rooted at a
    // plain local (e.g. a function call's result) isn't, since there's no
    // compile-time address for it to mutate through.
    ConstValue *resolveLValueSlot(const Expr &e, Frame &frame);

    // Default-constructed aggregate/scalar value matching 'ty's shape, used
    // to seed an uninitialized local ('struct Circle c;' with no '= {...}')
    // so a later 'c.radius = 2.0;' has a Struct-kind slot to mutate into,
    // instead of leaving it Void.
    static ConstValue zeroValue(const TypePtr &ty);

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
    void          buildFnTable();

    // ── resolveArraySizes() helpers ───────────────────────────────────────────
    // 'allowVLA': true only for a local variable's declared type while
    // inside an 'unsafe { ... }' block — see resolveStmtArraySizes. When an
    // array-size expression can't be folded to a constant AND allowVLA is
    // set, this is a variable-length array rather than an error: the
    // (still-)diagnostic is discarded, ArrayType::size is set to -2 (a
    // distinct "VLA" marker so CodeGen dynamically allocates it — see
    // genVarDecl), and sizeExpr is *not* cleared, since Sema/CodeGen still
    // need it to check/evaluate the size at its actual (runtime) scope.
    bool resolveTypeArraySizes(const TypePtr &ty, bool allowVLA = false);
    bool resolveStmtArraySizes(Stmt &s, bool insideUnsafe = false);

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
