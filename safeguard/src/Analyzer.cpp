#include "Analyzer.h"
#include "Builder.h"   // for runCmd (static method)
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>

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
    int lineno = 0;
    int count  = 0;
    std::string firstFile;
    // Just count occurrences of "unsafe {" across the whole file.
    while (std::getline(ss, line)) {
        ++lineno;
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
    auto unsafe_diags = lintUnsafeBlocks(src);
    auto null_diags   = lintNullChecks(src);
    auto unused_diags = lintUnusedVars(src, ast);

    for (auto& d : unsafe_diags) { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : null_diags)   { d.file = srcPath; result.diags.push_back(d); }
    for (auto& d : unused_diags) { d.file = srcPath; result.diags.push_back(d); }

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
