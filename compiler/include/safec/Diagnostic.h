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

private:
    void emit(DiagLevel level, SourceLocation loc, std::string msg) {
        const char *prefix = "";
        switch (level) {
        case DiagLevel::Note:    prefix = "note";    break;
        case DiagLevel::Warning: prefix = "warning"; break;
        case DiagLevel::Error:   prefix = "error";   break;
        case DiagLevel::Fatal:   prefix = "fatal";   break;
        }
        const char *f = loc.file ? loc.file : (filename_ ? filename_ : "<input>");
        fprintf(stderr, "%s:%u:%u: %s: %s\n", f, loc.line, loc.col, prefix, msg.c_str());
        diags_.push_back({level, loc, std::move(msg)});
    }

    const char           *filename_   = nullptr;
    int                   errorCount_ = 0;
    std::vector<Diagnostic> diags_;
};

} // namespace safec
