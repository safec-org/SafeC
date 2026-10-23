#pragma once
// --emit-doc-json: the declarations of one or more parsed translation units
// as JSON, for documentation tools (safeguard doc / safec-doc).
//
// Declarations come from the parser (names, kinds, generics, parameters,
// fields, methods, enumerators, namespaces, macros, source locations); the
// doc comment of each is read from its original source file around that
// location (see DocJson.cpp for the comment conventions), and its
// signature is the declaration's own source text, comments removed and
// whitespace collapsed.
#include "safec/AST.h"
#include "safec/Preprocessor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace safec {

struct DocOptions {
    // Directories whose declarations are documented, paths in the output
    // relative to the first root that holds them. Empty: every file, with
    // absolute paths.
    std::vector<std::string> roots;
    // name=dir: declarations under 'dir' are listed (without their doc
    // text) as items of the external crate 'name', so a doc tool can link
    // to them — e.g. std=<SafeC>/std when documenting a library.
    std::vector<std::pair<std::string, std::string>> externs;
    // Also list 'static' (internal-linkage) functions and globals.
    bool includePrivate = false;
};

class DocCollector {
public:
    explicit DocCollector(DocOptions opts);
    ~DocCollector();

    // Records the declarations of one parsed translation unit and the macros
    // its preprocessing defined. Call once per input file.
    void add(const TranslationUnit &tu,
             const std::unordered_map<std::string, MacroDef> &macros,
             const std::string &inputPath);

    // The whole collection as a JSON document.
    std::string json();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace safec
