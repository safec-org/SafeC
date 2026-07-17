#include "Analyzer.h"
#include "Builder.h"   // for runCmd (static method)
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>

// popen/pclose are POSIX; MSVC's CRT exposes the identical functions under
// an underscore-prefixed name (a non-standard-extension naming convention,
// not a behavior difference) — this is the only source of platform
// divergence in this file.
#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
#endif

namespace fs = std::filesystem;

namespace safeguard {

Analyzer::Analyzer(std::string projectRoot, Manifest manifest, bool verbose)
    : root_(std::move(projectRoot))
    , manifest_(std::move(manifest))
    , verbose_(verbose)
{}

// ── discovery (mirrors Builder) ───────────────────────────────────────────────

std::string Analyzer::findSafec() const {
    const char* home = std::getenv("SAFEC_HOME");
    if (home) {
        fs::path c = fs::path(home) / "compiler" / "build" / "safec";
        if (fs::exists(c)) return c.string();
    }
    return "safec";
}

std::string Analyzer::findStdDir() const {
    const char* home = std::getenv("SAFEC_HOME");
    if (home) {
        fs::path d = fs::path(home) / "std";
        if (fs::exists(d)) return d.string();
    }
    return "";
}

std::vector<std::string> Analyzer::collectSources(const std::string& srcDir) const {
    std::vector<std::string> srcs;
    if (!fs::exists(srcDir)) return srcs;
    for (auto& e : fs::recursive_directory_iterator(srcDir))
        if (e.is_regular_file() && e.path().extension() == ".sc")
            srcs.push_back(e.path().string());
    std::sort(srcs.begin(), srcs.end());
    return srcs;
}

// ── AST dump ──────────────────────────────────────────────────────────────────

std::string Analyzer::runDumpAst(const std::string& safecBin,
                                   const std::string& srcPath,
                                   const std::vector<std::string>& includeDirs) const {
    // Build command: safec <src> --dump-ast -o /dev/null 2>&1
    std::string cmd = safecBin;
    for (auto& inc : includeDirs)
        cmd += " -I \"" + inc + "\"";
    cmd += " \"" + srcPath + "\" --dump-ast 2>&1";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string out;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}

// ── Lint pass: count unsafe{} blocks and warn if > 3 per file ─────────────────

std::vector<AnalyzerDiag> Analyzer::lintUnsafeBlocks(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(src);
    std::string line;
    int count  = 0;
    // Just count occurrences of "unsafe {" across the whole file.
    while (std::getline(ss, line)) {
        if (line.find("unsafe") != std::string::npos &&
            line.find('{')      != std::string::npos) {
            ++count;
        }
    }
    // Warn if a non-implementation file has > 5 unsafe blocks.
    if (count > 5) {
        AnalyzerDiag d;
        d.line    = 0;
        d.level   = "warning";
        d.code    = "SA001";
        d.message = "file contains " + std::to_string(count) +
                    " unsafe{} blocks — consider refactoring";
        diags.push_back(d);
    }
    return diags;
}

// ── Lint pass: warn on malloc/free without unsafe{} ──────────────────────────

std::vector<AnalyzerDiag> Analyzer::lintNullChecks(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(src);
    std::string line;
    int lineno = 0;
    while (std::getline(ss, line)) {
        ++lineno;
        // Heuristic: alloc() result assigned without null check on same line.
        if (line.find("alloc(") != std::string::npos &&
            line.find("if")     == std::string::npos &&
            line.find("//")     == std::string::npos) {
            AnalyzerDiag d;
            d.line    = lineno;
            d.level   = "note";
            d.code    = "SA002";
            d.message = "result of alloc() should be null-checked";
            diags.push_back(d);
        }
    }
    return diags;
}

// ── Lint pass: empty unsafe{} block ───────────────────────────────────────────

std::vector<AnalyzerDiag> Analyzer::lintEmptyUnsafe(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    // Scan the whole file (not line-by-line) so a block split across lines
    // ('unsafe {\n}\n') is still caught.
    size_t pos = 0;
    int lineno = 1;
    std::vector<size_t> lineStarts = {0};
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i] == '\n') lineStarts.push_back(i + 1);
    auto lineOf = [&](size_t offset) {
        auto it = std::upper_bound(lineStarts.begin(), lineStarts.end(), offset);
        return (int)std::distance(lineStarts.begin(), it);
    };
    while ((pos = src.find("unsafe", pos)) != std::string::npos) {
        size_t after = pos + 6;
        size_t brace = src.find_first_not_of(" \t\r\n", after);
        if (brace != std::string::npos && src[brace] == '{') {
            size_t close = src.find_first_not_of(" \t\r\n", brace + 1);
            if (close != std::string::npos && src[close] == '}') {
                AnalyzerDiag d;
                d.line    = lineOf(pos);
                d.level   = "warning";
                d.code    = "SA004";
                d.message = "empty unsafe{} block — has no effect, safe to remove";
                diags.push_back(d);
            }
        }
        pos = after;
    }
    (void)lineno;
    return diags;
}

// ── Lint pass: TODO/FIXME/XXX markers ─────────────────────────────────────────

std::vector<AnalyzerDiag> Analyzer::lintTodoMarkers(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(src);
    std::string line;
    int lineno = 0;
    static const char* kMarkers[] = {"TODO", "FIXME", "XXX"};
    while (std::getline(ss, line)) {
        ++lineno;
        for (auto* marker : kMarkers) {
            size_t p = line.find(marker);
            if (p == std::string::npos) continue;
            // Only inside a comment — a plain match avoids false positives
            // on identifiers like 'TodoList' by requiring a preceding '//'
            // or the marker being inside a '/* ... */' span is out of
            // scope for a line-based scan, so this covers the common
            // '// TODO: ...' case, which is the overwhelming majority.
            size_t commentPos = line.find("//");
            if (commentPos != std::string::npos && commentPos < p) {
                AnalyzerDiag d;
                d.line    = lineno;
                d.level   = "note";
                d.code    = "SA005";
                d.message = std::string("unresolved ") + marker + " marker";
                diags.push_back(d);
                break;
            }
        }
    }
    return diags;
}

// ── Lint pass: suspicious '=' in an if-condition ─────────────────────────────

std::vector<AnalyzerDiag> Analyzer::lintSuspiciousAssign(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(src);
    std::string line;
    int lineno = 0;
    while (std::getline(ss, line)) {
        ++lineno;
        size_t ifPos = 0;
        while ((ifPos = line.find("if", ifPos)) != std::string::npos) {
            bool wordStart = (ifPos == 0 || (!isalnum((unsigned char)line[ifPos - 1]) &&
                                              line[ifPos - 1] != '_'));
            size_t afterIf = ifPos + 2;
            bool wordEnd = afterIf < line.size() &&
                           !isalnum((unsigned char)line[afterIf]) && line[afterIf] != '_';
            if (!wordStart || !wordEnd) { ifPos += 2; continue; }

            size_t k = afterIf;
            while (k < line.size() && isspace((unsigned char)line[k])) ++k;
            if (k >= line.size() || line[k] != '(') { ifPos = afterIf; continue; }

            int depth = 1;
            size_t j = k + 1;
            while (j < line.size() && depth > 0) {
                if (line[j] == '(') ++depth;
                else if (line[j] == ')') --depth;
                ++j;
            }
            if (depth == 0) {
                std::string cond = line.substr(k + 1, (j - 1) - (k + 1));
                for (size_t c = 0; c < cond.size(); ++c) {
                    if (cond[c] != '=') continue;
                    bool prevEq   = c > 0 && cond[c - 1] == '=';
                    bool nextEq   = c + 1 < cond.size() && cond[c + 1] == '=';
                    bool prevBang = c > 0 && cond[c - 1] == '!';
                    bool prevLt   = c > 0 && cond[c - 1] == '<';
                    bool prevGt   = c > 0 && cond[c - 1] == '>';
                    bool prevCompound = c > 0 && std::string("+-*/%&|^").find(cond[c - 1]) != std::string::npos;
                    if (!prevEq && !nextEq && !prevBang && !prevLt && !prevGt && !prevCompound) {
                        AnalyzerDiag d;
                        d.line    = lineno;
                        d.level   = "warning";
                        d.code    = "SA006";
                        d.message = "assignment ('=') in if-condition — did you mean '=='?";
                        diags.push_back(d);
                        break;
                    }
                }
            }
            ifPos = afterIf;
        }
    }
    return diags;
}

// ── Lint pass: duplicate #include ─────────────────────────────────────────────

std::vector<AnalyzerDiag> Analyzer::lintDuplicateInclude(const std::string& src) const {
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(src);
    std::string line;
    int lineno = 0;
    std::vector<std::pair<std::string, int>> seen;
    while (std::getline(ss, line)) {
        ++lineno;
        size_t h = line.find_first_not_of(" \t");
        if (h == std::string::npos || line[h] != '#') continue;
        if (line.compare(h, 8, "#include") != 0) continue;
        std::string target = line.substr(h);
        for (auto& [prevTarget, prevLine] : seen) {
            if (prevTarget == target) {
                AnalyzerDiag d;
                d.line    = lineno;
                d.level   = "warning";
                d.code    = "SA007";
                d.message = "duplicate include, first seen at line " +
                            std::to_string(prevLine);
                diags.push_back(d);
                break;
            }
        }
        seen.push_back({target, lineno});
    }
    return diags;
}

std::vector<AnalyzerDiag> Analyzer::lintUnusedVars(const std::string& /*src*/,
                                                      const std::string& ast) const {
    // Parse AST text for "unused" warnings emitted by safec --dump-ast.
    std::vector<AnalyzerDiag> diags;
    std::istringstream ss(ast);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("warning") != std::string::npos &&
            line.find("unused")  != std::string::npos) {
            AnalyzerDiag d;
            d.level   = "warning";
            d.code    = "SA003";
            d.message = line;
            diags.push_back(d);
        }
    }
    return diags;
}

// ── Per-file analysis ─────────────────────────────────────────────────────────

AnalysisResult Analyzer::analyzeFile(const std::string& srcPath,
                                      const std::vector<std::string>& includeDirs) const {
    AnalysisResult result;
    result.file = srcPath;

    // Read source text.
    std::ifstream f(srcPath);
    if (!f) {
        AnalyzerDiag d;
        d.line    = 0;
        d.level   = "error";
        d.code    = "SA000";
        d.message = "cannot open source file";
        result.diags.push_back(d);
        return result;
    }
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    // Run AST dump.
    std::string ast = runDumpAst(findSafec(), srcPath, includeDirs);

    // Lint passes.
    auto unsafe_diags    = lintUnsafeBlocks(src);
    auto null_diags      = lintNullChecks(src);
    auto unused_diags    = lintUnusedVars(src, ast);
    auto empty_unsafe     = lintEmptyUnsafe(src);
    auto todo_diags       = lintTodoMarkers(src);
    auto assign_diags     = lintSuspiciousAssign(src);
    auto dup_include_diags = lintDuplicateInclude(src);

    for (auto& d : unsafe_diags)     { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : null_diags)       { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : unused_diags)     { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : empty_unsafe)     { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : todo_diags)       { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : assign_diags)     { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : dup_include_diags){ d.file = srcPath; result.diags.push_back(d); }

    return result;
}

// ── Full project analysis ─────────────────────────────────────────────────────

bool Analyzer::analyze() {
    std::string stdDir = findStdDir();
    std::vector<std::string> incs;
    if (!stdDir.empty()) {
        incs.push_back(stdDir);
        incs.push_back(fs::path(stdDir).parent_path().string());
    }

    fs::path srcDir = fs::path(root_) / "src";
    auto srcs = collectSources(srcDir.string());

    std::vector<AnalysisResult> results;
    for (auto& src : srcs) {
        if (verbose_) std::cout << "safeguard: analyzing " << src << "\n";
        results.push_back(analyzeFile(src, incs));
    }

    printResults(results);
    return errorCount(results) == 0;
}

// ── Reporting ─────────────────────────────────────────────────────────────────

void Analyzer::printResults(const std::vector<AnalysisResult>& results) {
    for (auto& r : results) {
        for (auto& d : r.diags) {
            std::string loc = r.file;
            if (d.line > 0) loc += ":" + std::to_string(d.line);
            std::cout << loc << ": " << d.level << " [" << d.code << "] "
                      << d.message << "\n";
        }
    }
    int errors = errorCount(results);
    int total  = 0;
    for (auto& r : results) total += static_cast<int>(r.diags.size());
    std::cout << "safeguard: analysis complete — "
              << total << " diagnostic(s), " << errors << " error(s)\n";
}

int Analyzer::errorCount(const std::vector<AnalysisResult>& results) {
    int n = 0;
    for (auto& r : results)
        for (auto& d : r.diags)
            if (d.level == "error") ++n;
    return n;
}

} // namespace safeguard
