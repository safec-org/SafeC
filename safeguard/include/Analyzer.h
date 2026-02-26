#pragma once
#include "Manifest.h"
#include <string>
#include <vector>

namespace safeguard {

// A single diagnostic emitted by the static analyzer.
struct AnalyzerDiag {
    std::string file;
    int         line = 0;
    std::string level;    // "error" | "warning" | "note"
    std::string code;     // e.g. "SA001"
    std::string message;
};

// Result from a single-file analysis pass.
struct AnalysisResult {
    std::string              file;
    std::vector<AnalyzerDiag> diags;
    bool ok() const { return diags.empty(); }
};

// Static analysis driver: invokes `safec --dump-ast` to obtain the AST for each
// source file, then runs built-in lint passes over the JSON output.
class Analyzer {
public:
    explicit Analyzer(std::string projectRoot, Manifest manifest,
                      bool verbose = false);

    // Run all lint passes on every source file in the project.
    // Returns true if no errors (warnings are non-fatal).
    bool analyze();

    // Analyze a single file.
    AnalysisResult analyzeFile(const std::string& srcPath,
                                const std::vector<std::string>& includeDirs) const;

    // Print results in a human-readable format.
    static void printResults(const std::vector<AnalysisResult>& results);

    // Return count of error-level diagnostics across all results.
    static int errorCount(const std::vector<AnalysisResult>& results);

private:
    std::string root_;
    Manifest    manifest_;
    bool        verbose_;

    // Locate safec binary (same logic as Builder).
    std::string findSafec() const;
    std::string findStdDir() const;
    std::vector<std::string> collectSources(const std::string& srcDir) const;

    // Run safec --dump-ast and capture stdout.
    std::string runDumpAst(const std::string& safecBin,
                            const std::string& srcPath,
                            const std::vector<std::string>& includeDirs) const;

    // Lint passes (operate on AST text / raw source lines).
    std::vector<AnalyzerDiag> lintUnusedVars(const std::string& src,
                                               const std::string& ast) const;
    std::vector<AnalyzerDiag> lintUnsafeBlocks(const std::string& src) const;
    std::vector<AnalyzerDiag> lintNullChecks(const std::string& src) const;
};

} // namespace safeguard
