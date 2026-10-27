#pragma once
#include <string>
#include <vector>
#include <cstdio>
#include <mutex>
#include <unordered_set>

namespace safec {

// A file name with process lifetime, for SourceLocation::file: locations
// (macro definitions, diagnostics) outlive the strings the preprocessor and
// lexer hold while a file is being read. Safe to call from the parallel
// per-file compiles.
inline const char *internFileName(const std::string &f) {
    static std::mutex mu;
    static std::unordered_set<std::string> names;
    std::lock_guard<std::mutex> lock(mu);
    return names.insert(f).first->c_str();
}

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

    // Every line printed to stderr so far, in order. The incremental cache
    // stores the part printed after preprocessing next to the cached
    // bitcode, so a cache hit can print the same warnings again.
    const std::string &transcript() const { return transcript_; }

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
            char head[64];
            snprintf(head, sizeof(head), ":%u:%u: ", loc.line, loc.col);
            std::string line = std::string(f) + head + prefix + ": " + msg + "\n";
            fputs(line.c_str(), stderr);
            transcript_ += line;
        }
        diags_.push_back({level, loc, std::move(msg)});
    }

    const char           *filename_   = nullptr;
    int                   errorCount_ = 0;
    bool                  silent_     = false;
    std::vector<Diagnostic> diags_;
    std::string           transcript_;
};

} // namespace safec
