#pragma once
#include "safec/Diagnostic.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace safec {

// ── Macro definition ──────────────────────────────────────────────────────────
struct MacroDef {
    std::string              name;
    std::string              body;        // expansion text
    std::vector<std::string> params;      // non-empty → function-like
    bool                     isFunctionLike = false;
    bool                     isVariadic     = false;  // last param is __VA_ARGS__
    SourceLocation           loc;
};

// ── Options ───────────────────────────────────────────────────────────────────
struct PreprocOptions {
    bool compatMode     = false;   // --compat-preprocessor: allow fn-like macros etc.
    int  maxIncludeDepth = 64;
    std::vector<std::string>                     includePaths;  // -I dirs
    std::unordered_map<std::string, std::string> cmdlineDefs;   // -D NAME[=VAL]
};

// ── Preprocessor ──────────────────────────────────────────────────────────────
// Performs a text-level pass before lexing:
//   - #include / #pragma once
//   - Object-like #define / #undef
//   - Conditional compilation: #if / #ifdef / #ifndef / #elif / #else / #endif
//   - Macro expansion (object-like; function-like only in compat mode)
//   - __FILE__ / __LINE__ built-ins
//
// In safe mode (default): function-like macros, ## token-pasting, and
// # stringification trigger compile errors.
class Preprocessor {
public:
    Preprocessor(std::string source, std::string filename,
                 DiagEngine &diag, PreprocOptions opts = {});

    // Run preprocessing; returns preprocessed source text.
    // Directive lines are replaced with blank lines to preserve line numbers.
    std::string process();

    // Access the final macro table (useful for diagnostics / tooling).
    const std::unordered_map<std::string, MacroDef>& macros() const { return macros_; }

private:
    // ── Top-level processing ──────────────────────────────────────────────────
    std::string processFile(const std::string &src,
                            const std::string &filename,
                            int depth);

    // Parse and execute one directive line (text after the '#').
    // Returns injected content (for #include) or empty string.
    std::string handleDirective(const std::string &rest,
                                const std::string &filename,
                                unsigned lineNo, int depth);

    // ── Directive handlers ────────────────────────────────────────────────────
    void        handleDefine(const std::string &rest,
                             const std::string &filename, unsigned lineNo);
    void        handleUndef(const std::string &rest);
    void        handleIfdef(const std::string &name, bool negate);
    void        handleIf(const std::string &rest,
                         const std::string &filename, unsigned lineNo);
    void        handleElif(const std::string &rest,
                           const std::string &filename, unsigned lineNo);
    void        handleElse(const std::string &filename, unsigned lineNo);
    void        handleEndif(const std::string &filename, unsigned lineNo);
    std::string handleInclude(const std::string &rest,
                              const std::string &curFile,
                              unsigned lineNo, int depth);
    void        handlePragma(const std::string &rest,
                             const std::string &filename, unsigned lineNo);
    void        handleError(const std::string &rest,
                            const std::string &filename, unsigned lineNo);
    void        handleWarning(const std::string &rest,
                              const std::string &filename, unsigned lineNo);

    // ── Conditional stack ─────────────────────────────────────────────────────
    struct CondState {
        bool active;          // currently emitting this branch?
        bool anyTaken;        // any branch of this #if has been taken?
    };
    std::vector<CondState> condStack_;
    bool isActive() const;  // true when all levels on stack are active

    // ── Macro expansion ───────────────────────────────────────────────────────
    // Expand macros in a code line (skipping string/char literals).
    std::string expandLine(const std::string &line,
                           const std::string &filename, unsigned lineNo);

    // Expand a token stream (text) with recursion guard.
    std::string expandTokens(const std::string &text,
                             std::unordered_set<std::string> &guard,
                             const std::string &filename, unsigned lineNo);

    // Expand a function-like macro call starting at `pos` (after the macro name).
    // On success advances pos past the closing ')'. Returns expanded text.
    std::string expandFunctionLike(const MacroDef &def,
                                   const std::string &text,
                                   size_t &pos,
                                   std::unordered_set<std::string> &guard,
                                   const std::string &filename, unsigned lineNo);

    // ── #if expression evaluator ──────────────────────────────────────────────
    // Full recursive-descent C preprocessor integer expression evaluator.
    int64_t evalCondExpr(const std::string &expr,
                         const std::string &filename, unsigned lineNo);

    // Pre-expand macros in `expr` then call evalCondExprRaw.
    int64_t evalCondExprExpanded(const std::string &expr,
                                 const std::string &filename, unsigned lineNo);

    // Recursive descent on pre-expanded text.
    int64_t parseTernary(const std::string &s, size_t &p,
                         const std::string &f, unsigned ln);
    int64_t parseOr    (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseAnd   (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseBitOr (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseBitXor(const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseBitAnd(const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseEq    (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseRel   (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseShift (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseAdd   (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseMul   (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parseUnary (const std::string &s, size_t &p, const std::string &f, unsigned ln);
    int64_t parsePrimary(const std::string &s, size_t &p, const std::string &f, unsigned ln);

    // ── File resolution ───────────────────────────────────────────────────────
    std::string resolveInclude(const std::string &name,
                               bool isSystem,
                               const std::string &curFile);
    static std::string readFile(const std::string &path);

    // ── Character / string helpers ────────────────────────────────────────────
    static std::string   trim(const std::string &s);
    static bool          isIdentStart(char c);
    static bool          isIdentChar(char c);
    static std::string   readIdentAt(const std::string &s, size_t &pos);
    static void          skipWS(const std::string &s, size_t &pos);

    // ── State ─────────────────────────────────────────────────────────────────
    std::string    source_;
    std::string    filename_;
    DiagEngine    &diag_;
    PreprocOptions opts_;

    std::unordered_map<std::string, MacroDef> macros_;
    std::unordered_set<std::string>           pragmaOnceFiles_;

    // Current file/line for __FILE__ / __LINE__ expansion
    std::string  curFile_;
    unsigned     curLine_ = 1;
};

} // namespace safec
