#pragma once
#include <string>
#include <vector>
#include <cstdio>

namespace safec {

struct SourceLocation {
    unsigned line   = 1;
    unsigned col    = 1;
    const char *file = nullptr;

    SourceLocation() = default;
    SourceLocation(unsigned l, unsigned c, const char *f = nullptr)
        : line(l), col(c), file(f) {}
};

enum class DiagLevel { Note, Warning, Error, Fatal };

struct Diagnostic {
    DiagLevel     level;
    SourceLocation loc;
    std::string   message;
};

class DiagEngine {
public:
    explicit DiagEngine(const char *filename = nullptr)
        : filename_(filename) {}

    void note   (SourceLocation l, std::string msg) { emit(DiagLevel::Note,    l, std::move(msg)); }
    void warn   (SourceLocation l, std::string msg) { emit(DiagLevel::Warning, l, std::move(msg)); }
    void error  (SourceLocation l, std::string msg) { emit(DiagLevel::Error,   l, std::move(msg)); ++errorCount_; }
    void fatal  (SourceLocation l, std::string msg) { emit(DiagLevel::Fatal,   l, std::move(msg)); ++errorCount_; }

    bool hasErrors()  const { return errorCount_ > 0; }
    int  errorCount() const { return errorCount_; }

    const std::vector<Diagnostic> &diagnostics() const { return diags_; }

    void setSilent(bool s) { silent_ = s; }
    bool isSilent() const { return silent_; }
    void reset() { diags_.clear(); errorCount_ = 0; }

    // ── Speculative-parse support ──────────────────────────────────────────
    // Some grammar productions are ambiguous (e.g. '(' could start a cast or
    // a parenthesized expression) and are resolved by trying one parse and
    // backtracking on failure. Backtracking the token stream alone isn't
    // enough — any diagnostics emitted mid-attempt must be discarded too, or
    // an abandoned speculative parse silently poisons the real error count.
    size_t checkpoint() const { return diags_.size(); }
    void discardSince(size_t mark) {
        if (mark >= diags_.size()) return;
        for (size_t i = mark; i < diags_.size(); ++i) {
            if (diags_[i].level == DiagLevel::Error || diags_[i].level == DiagLevel::Fatal)
                --errorCount_;
        }
        diags_.resize(mark);
    }

private:
    void emit(DiagLevel level, SourceLocation loc, std::string msg) {
        if (!silent_) {
            const char *prefix = "";
            switch (level) {
            case DiagLevel::Note:    prefix = "note";    break;
            case DiagLevel::Warning: prefix = "warning"; break;
            case DiagLevel::Error:   prefix = "error";   break;
            case DiagLevel::Fatal:   prefix = "fatal";   break;
            }
            const char *f = loc.file ? loc.file : (filename_ ? filename_ : "<input>");
            fprintf(stderr, "%s:%u:%u: %s: %s\n", f, loc.line, loc.col, prefix, msg.c_str());
        }
        diags_.push_back({level, loc, std::move(msg)});
    }

    const char           *filename_   = nullptr;
    int                   errorCount_ = 0;
    bool                  silent_     = false;
    std::vector<Diagnostic> diags_;
};

} // namespace safec
