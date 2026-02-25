#pragma once
#include "safec/Diagnostic.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace safec {

/// CHeaderImporter uses clang's AST JSON dump to extract C function and type
/// declarations from system headers and emit SafeC-compatible extern text.
///
/// Integration: called by Preprocessor when #include <header.h> fails to
/// resolve via the normal SafeC include path search.  The returned text is
/// injected directly into the preprocessed source stream, so all names become
/// visible to the SafeC lexer/parser just as in C++.
class CHeaderImporter {
public:
    explicit CHeaderImporter(DiagEngine &diag);

    /// True when a usable clang binary was found at construction time.
    bool available() const { return !clangPath_.empty(); }

    /// Import declarations from the named C header.
    /// @param headerName  e.g. "stdio.h", "stdlib.h", "string.h"
    /// @param includePaths  additional -I search paths forwarded to clang
    /// @return  SafeC extern declaration text, or empty string on failure
    std::string import(const std::string &headerName,
                       const std::vector<std::string> &includePaths = {});

private:
    DiagEngine                     &diag_;
    std::string                     clangPath_;
    std::unordered_set<std::string> imported_; // headers already processed

    static std::string findClang();

    std::string runClangASTDump(const std::string &headerName,
                                const std::vector<std::string> &includePaths);

    std::string buildDeclarations(const std::string &jsonText);

    // Convert a clang qualType string to SafeC-compatible type syntax.
    static std::string cleanType(const std::string &qt);

    // True if the type string contains a function pointer pattern.
    static bool hasFunctionPointer(const std::string &qt);
};

} // namespace safec
