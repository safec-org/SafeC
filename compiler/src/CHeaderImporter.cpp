#include "safec/CHeaderImporter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <sstream>
#include <unordered_set>

namespace safec {

// =============================================================================
// Construction / clang discovery
// =============================================================================

CHeaderImporter::CHeaderImporter(DiagEngine &diag) : diag_(diag) {
    clangPath_ = findClang();
}

std::string CHeaderImporter::findClang() {
    // Prefer the native system clang (arm64 on Apple Silicon), then Homebrew
    // LLVM, then whatever is on PATH.
    static const char *candidates[] = {
        "/usr/bin/clang",
        "/usr/local/bin/clang",
        "/opt/homebrew/bin/clang",
        "/usr/local/Cellar/llvm/21.1.8_1/bin/clang",
        "clang",
        nullptr
    };
    for (int i = 0; candidates[i]; ++i) {
        std::string p = candidates[i];
        // Entries without '/' are PATH-relative; assume available.
        if (p.find('/') == std::string::npos) return p;
        if (llvm::sys::fs::exists(p)) return p;
    }
    return {};
}

// =============================================================================
// Public entry point
// =============================================================================

std::string CHeaderImporter::import(const std::string &headerName,
                                     const std::vector<std::string> &includePaths) {
    if (clangPath_.empty()) return {};
    if (imported_.count(headerName)) return {};
    imported_.insert(headerName);

    std::string json = runClangASTDump(headerName, includePaths);
    if (json.empty()) return {};

    return buildDeclarations(json);
}

// =============================================================================
// Invoke clang -ast-dump=json
// =============================================================================

std::string CHeaderImporter::runClangASTDump(
        const std::string &headerName,
        const std::vector<std::string> &includePaths) {

    // Write a tiny C file containing just the #include directive.
    llvm::SmallString<128> tmpPath;
    {
        int fd;
        if (llvm::sys::fs::createTemporaryFile("safec_chi", "c", fd, tmpPath))
            return {};
        llvm::raw_fd_ostream tmp(fd, /*shouldClose=*/true);
        tmp << "#include <" << headerName << ">\n";
    }

    // Build the shell command:
    //   clang -x c -fsyntax-only -Xclang -ast-dump=json <tmpfile> 2>/dev/null
    // We pass -I paths so the header can be found if it is non-system.
    std::string cmd;
    cmd += clangPath_;
    cmd += " -x c -fsyntax-only -Xclang -ast-dump=json";
    for (auto &p : includePaths) {
        cmd += " -I'";
        cmd += p;
        cmd += "'";
    }
    cmd += " '";
    cmd += tmpPath.str().str();
    cmd += "' 2>/dev/null";

    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        llvm::sys::fs::remove(tmpPath);
        return {};
    }

    std::string json;
    json.reserve(512 * 1024);
    char buf[8192];
    while (fgets(buf, sizeof(buf), pipe))
        json += buf;
    pclose(pipe);

    llvm::sys::fs::remove(tmpPath);
    return json;
}

// =============================================================================
// Type utilities
// =============================================================================

// Returns true if the type string references a C type that SafeC does not
// (yet) support, so we can skip those declarations gracefully.
static bool hasUnsupportedType(const std::string &qt) {
    // "long double" — SafeC has Float32/Float64 only, no 80/128-bit float.
    if (qt.find("long double") != std::string::npos) return true;
    // Wide character types — not in SafeC's primitive set.
    if (qt.find("wchar_t")  != std::string::npos) return true;
    if (qt.find("__wchar_t") != std::string::npos) return true;
    return false;
}

bool CHeaderImporter::hasFunctionPointer(const std::string &qt) {
    // C function pointers: (*) and ObjC/Clang block types: (^)
    return qt.find("(*") != std::string::npos ||
           qt.find("( *") != std::string::npos ||
           qt.find("(^") != std::string::npos ||
           qt.find("( ^") != std::string::npos;
}

std::string CHeaderImporter::cleanType(const std::string &qt) {
    // Walk token-by-token, dropping qualifiers SafeC doesn't use and
    // normalising spacing around pointer stars.
    std::string out;
    out.reserve(qt.size());
    size_t i = 0, n = qt.size();

    auto isIdentCh  = [](char c){ return isalnum((unsigned char)c) || c == '_'; };
    auto isIdentStart=[](char c){ return isalpha((unsigned char)c) || c == '_'; };

    while (i < n) {
        // Read an identifier-like token
        if (isIdentStart(qt[i])) {
            size_t start = i;
            while (i < n && isIdentCh(qt[i])) ++i;
            std::string tok = qt.substr(start, i - start);

            // Drop these qualifiers / attributes entirely
            if (tok == "__restrict" || tok == "__restrict__" ||
                tok == "restrict"   || tok == "__attribute__" ||
                tok == "_Nonnull"   || tok == "_Nullable"     ||
                tok == "_Null_unspecified" || tok == "__nonnull" ||
                tok == "__wur"      || tok == "__THROW"        ||
                tok == "__THROWNL"  || tok == "__LEAF"         ||
                tok == "__cold"     || tok == "__pure"         ||
                tok == "__noreturn__" || tok == "__extension__"  ||
                tok == "__sort_noescape" || tok == "__bsearch_noescape" ||
                tok == "__noescape" || tok == "__ptrauth_abi_breakage__") {
                // If it's __attribute__, also skip the following ((...)) group
                if (tok == "__attribute__") {
                    while (i < n && (qt[i] == ' ' || qt[i] == '\t')) ++i;
                    if (i < n && qt[i] == '(') {
                        int depth = 0;
                        while (i < n) {
                            if      (qt[i] == '(') ++depth;
                            else if (qt[i] == ')') { --depth; ++i; if (!depth) break; }
                            else ++i;
                        }
                    }
                }
                // Skip trailing spaces
                while (i < n && qt[i] == ' ') ++i;
                continue;
            }

            // Normalise double-underscore C99 keywords to plain C
            if (tok == "__const")            { out += "const"; continue; }
            if (tok == "__volatile__" || tok == "__volatile") { out += "volatile"; continue; }
            if (tok == "__signed__"   || tok == "__signed")   { out += "signed";   continue; }
            if (tok == "__inline"     || tok == "__inline__") { out += "inline";   continue; }

            out += tok;
        } else {
            out += qt[i++];
        }
    }

    // Collapse multiple spaces into one and trim
    std::string norm;
    norm.reserve(out.size());
    bool lastSpace = false;
    for (char c : out) {
        if (c == ' ' || c == '\t') {
            if (!lastSpace && !norm.empty()) { norm += ' '; lastSpace = true; }
        } else {
            norm += c; lastSpace = false;
        }
    }
    while (!norm.empty() && norm.back() == ' ') norm.pop_back();
    return norm;
}

// =============================================================================
// JSON → SafeC declarations
// =============================================================================

// Extract the return type from a C function qualType string.
// e.g. "int (const char *, ...)"  →  "int"
//      "FILE *(const char *)"     →  "FILE *"
static std::string extractReturnType(const std::string &fnQualType) {
    // Find the first '(' that starts the parameter list.
    // The return type is everything before it (trimmed).
    // Note: function pointer return types also contain '(' but are handled
    // elsewhere (hasFunctionPointer guard).
    size_t paren = fnQualType.find('(');
    if (paren == std::string::npos) return fnQualType;
    std::string ret = fnQualType.substr(0, paren);
    // Trim trailing whitespace
    while (!ret.empty() && ret.back() == ' ') ret.pop_back();
    return ret;
}

// Returns the underlying concrete type for a TypedefDecl JSON node.
// We prefer desugaredQualType when present (it resolves chained typedefs to
// the primitive/struct type) but fall back to qualType.
static std::string getTypedefUnderlying(const llvm::json::Object &node) {
    const auto *typeObj = node.getObject("type");
    if (!typeObj) return {};
    if (auto dq = typeObj->getString("desugaredQualType"))
        return dq->str();
    if (auto q = typeObj->getString("qualType"))
        return q->str();
    return {};
}

std::string CHeaderImporter::buildDeclarations(const std::string &jsonText) {
    auto parsed = llvm::json::parse(jsonText);
    if (!parsed) return {};

    const llvm::json::Object *root = parsed->getAsObject();
    if (!root) return {};
    const llvm::json::Array *inner = root->getArray("inner");
    if (!inner) return {};

    // Track what we have already emitted to suppress duplicates.
    std::unordered_set<std::string> emittedNames;
    // Track struct names we have forward-declared.
    std::unordered_set<std::string> forwardedStructs;

    std::ostringstream out;
    out << "// ---- begin C header import ----\n";

    // Note: we intentionally do NOT emit standalone "struct X;" forward
    // declarations because SafeC's parser treats "struct X ;" as a variable
    // declaration and fails with "expected identifier after type".
    // Opaque struct types flow in via "typedef struct X T;" which the parser
    // handles correctly; any direct "struct X *" param types are also accepted
    // because the parser creates an anonymous StructType on sight.
    (void)forwardedStructs; // suppress unused-variable warning

    // ── Pass 1: typedefs (order as they appear in the JSON, which mirrors
    //            source order — so dependency order is preserved) ──────────────
    for (const llvm::json::Value &item : *inner) {
        const llvm::json::Object *node = item.getAsObject();
        if (!node) continue;

        auto kind = node->getString("kind");
        if (!kind || *kind != "TypedefDecl") continue;

        auto nameOpt = node->getString("name");
        if (!nameOpt) continue;
        std::string name = nameOpt->str();

        if (emittedNames.count(name)) continue;

        std::string underlying = getTypedefUnderlying(*node);
        if (underlying.empty()) continue;

        // Skip typedefs whose underlying type is a function pointer / block.
        if (hasFunctionPointer(underlying)) continue;

        // Skip array typedefs (e.g. "unsigned char [16]", "struct T [1]").
        // SafeC typedef syntax does not support array dimensions on the base.
        if (underlying.find('[') != std::string::npos) continue;

        // Skip types that SafeC does not support.
        if (hasUnsupportedType(underlying)) continue;

        // Skip __int128 — not a SafeC primitive type.
        if (underlying.find("__int128") != std::string::npos) continue;

        std::string cleaned = cleanType(underlying);
        if (cleaned.empty()) continue;

        // Circular typedefs (desugaredQualType == name) arise from two cases:
        //   1. Enum typedefs: "typedef enum { ... } foo_t;" → qualType="enum foo_t"
        //      → treat as int (enums are int-sized in C).
        //   2. True circular aliases (e.g. system quirks) → skip.
        if (cleaned == name) {
            const llvm::json::Object *typeObj = node->getObject("type");
            std::string qual;
            if (typeObj) {
                if (auto q = typeObj->getString("qualType")) qual = q->str();
            }
            if (qual.rfind("enum ", 0) == 0) {
                // Emit "typedef int name;" for enum types.
                emittedNames.insert(name);
                out << "typedef int " << name << ";\n";
            }
            continue;
        }

        emittedNames.insert(name);
        out << "typedef " << cleaned << " " << name << ";\n";
    }

    // ── Pass 2: function declarations ─────────────────────────────────────────
    for (const llvm::json::Value &item : *inner) {
        const llvm::json::Object *node = item.getAsObject();
        if (!node) continue;

        auto kind = node->getString("kind");
        if (!kind || *kind != "FunctionDecl") continue;

        auto nameOpt = node->getString("name");
        if (!nameOpt) continue;
        std::string name = nameOpt->str();

        // Skip purely internal symbols.
        if (name.size() >= 2 && name[0] == '_' && name[1] == '_') continue;

        if (emittedNames.count(name)) continue;

        // Get the full function type to extract the return type.
        const llvm::json::Object *typeObj = node->getObject("type");
        if (!typeObj) continue;
        auto fnQualType = typeObj->getString("qualType");
        if (!fnQualType) continue;

        std::string fnTy = fnQualType->str();
        if (hasFunctionPointer(fnTy)) continue;
        if (hasUnsupportedType(fnTy))  continue;

        std::string retType = cleanType(extractReturnType(fnTy));
        if (retType.empty()) continue;

        // Collect parameters from ParmVarDecl children.
        const llvm::json::Array *children = node->getArray("inner");
        std::string params;
        bool firstParam = true;
        bool skipThis   = false;

        if (children) {
            for (const llvm::json::Value &ch : *children) {
                const llvm::json::Object *pnode = ch.getAsObject();
                if (!pnode) continue;
                auto pkind = pnode->getString("kind");
                if (!pkind || *pkind != "ParmVarDecl") continue;

                const llvm::json::Object *ptypeObj = pnode->getObject("type");
                if (!ptypeObj) { skipThis = true; break; }
                auto pq = ptypeObj->getString("qualType");
                if (!pq) { skipThis = true; break; }

                std::string pType = cleanType(pq->str());
                if (hasFunctionPointer(pType)) { skipThis = true; break; }

                if (!firstParam) params += ", ";
                params += pType;

                // Append parameter name if present.
                auto pname = pnode->getString("name");
                if (pname && !pname->empty()) {
                    params += " ";
                    params += pname->str();
                }
                firstParam = false;
            }
        }
        if (skipThis) continue;

        // Check for variadic.
        bool variadic = false;
        if (auto v = node->getBoolean("variadic")) variadic = *v;
        if (variadic) {
            if (!firstParam) params += ", ";
            params += "...";
        }

        emittedNames.insert(name);
        out << "extern " << retType << " " << name << "(" << params << ");\n";
    }

    out << "// ---- end C header import ----\n";
    return out.str();
}

} // namespace safec
