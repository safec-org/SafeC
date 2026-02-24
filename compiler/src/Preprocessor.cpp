#include "safec/Preprocessor.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cassert>
#include <cstring>
#include <cctype>
#include <climits>
#include <algorithm>

namespace safec {
namespace fs = std::filesystem;

// =============================================================================
// Static helpers
// =============================================================================

std::string Preprocessor::trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool Preprocessor::isIdentStart(char c) {
    return std::isalpha((unsigned char)c) || c == '_';
}

bool Preprocessor::isIdentChar(char c) {
    return std::isalnum((unsigned char)c) || c == '_';
}

std::string Preprocessor::readIdentAt(const std::string &s, size_t &pos) {
    size_t start = pos;
    while (pos < s.size() && isIdentChar(s[pos])) ++pos;
    return s.substr(start, pos - start);
}

void Preprocessor::skipWS(const std::string &s, size_t &pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
}

std::string Preprocessor::readFile(const std::string &path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// =============================================================================
// Constructor
// =============================================================================

Preprocessor::Preprocessor(std::string source, std::string filename,
                           DiagEngine &diag, PreprocOptions opts)
    : source_(std::move(source)), filename_(std::move(filename)),
      diag_(diag), opts_(std::move(opts))
{
    // Apply command-line defines (-D NAME or -D NAME=VALUE)
    for (auto &[name, val] : opts_.cmdlineDefs) {
        MacroDef def;
        def.name = name;
        def.body = val;
        macros_[name] = std::move(def);
    }
}

// =============================================================================
// Top-level process
// =============================================================================

std::string Preprocessor::process() {
    return processFile(source_, filename_, 0);
}

bool Preprocessor::isActive() const {
    for (auto &cs : condStack_)
        if (!cs.active) return false;
    return true;
}

std::string Preprocessor::processFile(const std::string &src,
                                       const std::string &filename,
                                       int depth) {
    std::string output;
    output.reserve(src.size());

    // Split source into logical lines (handle line continuation '\')
    std::vector<std::pair<std::string,unsigned>> lines; // (text, originalLineNo)
    {
        std::string cur;
        unsigned lineNo = 1;
        unsigned startLine = 1;
        for (size_t i = 0; i < src.size(); ++i) {
            char c = src[i];
            if (c == '\\' && i + 1 < src.size() && src[i + 1] == '\n') {
                ++i; ++lineNo;
                cur += ' ';
            } else if (c == '\n') {
                lines.push_back({cur, startLine});
                cur.clear();
                ++lineNo;
                startLine = lineNo;
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            lines.push_back({cur, startLine});
    }

    std::string savedFile = curFile_;
    unsigned    savedLine = curLine_;
    curFile_ = filename;

    for (auto &[line, lineNo] : lines) {
        curLine_ = lineNo;

        // Find first non-whitespace
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;

        if (i < line.size() && line[i] == '#') {
            // Directive line
            std::string rest = (i + 1 < line.size()) ? line.substr(i + 1) : std::string{};
            // Skip optional whitespace between # and directive name
            std::string injected = handleDirective(rest, filename, lineNo, depth);
            if (!injected.empty()) {
                // Injected include content: already ends with newlines
                output += injected;
            } else {
                output += '\n'; // blank line to preserve line numbers
            }
        } else if (isActive()) {
            // Active code line: strip single-line comments then expand macros
            // Strip // comments (naive: doesn't handle // inside strings perfectly
            // but the lexer handles multi-line /* */ so we leave those)
            std::string effective = line;
            // Find // outside string literals
            bool inStr  = false;
            bool inChar = false;
            for (size_t k = 0; k < effective.size(); ++k) {
                char c = effective[k];
                if (c == '\\' && (inStr || inChar)) { ++k; continue; }
                if (c == '"'  && !inChar) { inStr  = !inStr;  continue; }
                if (c == '\'' && !inStr)  { inChar = !inChar; continue; }
                if (!inStr && !inChar && c == '/' && k + 1 < effective.size()
                    && effective[k + 1] == '/') {
                    effective = effective.substr(0, k);
                    break;
                }
            }
            output += expandLine(effective, filename, lineNo);
            output += '\n';
        } else {
            output += '\n'; // skipped line
        }
    }

    curFile_ = savedFile;
    curLine_ = savedLine;
    return output;
}

// =============================================================================
// Directive dispatch
// =============================================================================

std::string Preprocessor::handleDirective(const std::string &rest,
                                           const std::string &filename,
                                           unsigned lineNo, int depth) {
    size_t pos = 0;
    skipWS(rest, pos);

    if (pos >= rest.size()) return {}; // bare '#'

    // Read directive name
    std::string dir;
    if (isIdentStart(rest[pos])) {
        dir = readIdentAt(rest, pos);
    } else if (std::isdigit((unsigned char)rest[pos])) {
        // # N linemarker (from another preprocessor) — ignore
        return {};
    } else {
        return {};
    }

    skipWS(rest, pos);
    std::string arg = (pos < rest.size()) ? rest.substr(pos) : std::string{};

    // ── Conditional directives (processed even when inactive) ─────────────────
    if (dir == "if") {
        // Count nesting even if inactive
        if (!isActive()) {
            // Push an "already inactive" state
            condStack_.push_back({false, true});
            return {};
        }
        handleIf(arg, filename, lineNo);
        return {};
    }
    if (dir == "ifdef") {
        if (!isActive()) {
            condStack_.push_back({false, true});
            return {};
        }
        size_t p = 0; skipWS(arg, p);
        std::string name = readIdentAt(arg, p);
        handleIfdef(name, false);
        return {};
    }
    if (dir == "ifndef") {
        if (!isActive()) {
            condStack_.push_back({false, true});
            return {};
        }
        size_t p = 0; skipWS(arg, p);
        std::string name = readIdentAt(arg, p);
        handleIfdef(name, true);
        return {};
    }
    if (dir == "elif") {
        handleElif(arg, filename, lineNo);
        return {};
    }
    if (dir == "else") {
        handleElse(filename, lineNo);
        return {};
    }
    if (dir == "endif") {
        handleEndif(filename, lineNo);
        return {};
    }

    // ── Active-only directives ─────────────────────────────────────────────────
    if (!isActive()) return {};

    if (dir == "define") {
        handleDefine(arg, filename, lineNo);
        return {};
    }
    if (dir == "undef") {
        handleUndef(arg);
        return {};
    }
    if (dir == "include") {
        return handleInclude(arg, filename, lineNo, depth);
    }
    if (dir == "pragma") {
        handlePragma(arg, filename, lineNo);
        return {};
    }
    if (dir == "error") {
        handleError(arg, filename, lineNo);
        return {};
    }
    if (dir == "warning") {
        handleWarning(arg, filename, lineNo);
        return {};
    }
    if (dir == "line") {
        // #line N ["filename"] — update current line tracking
        size_t p = 0; skipWS(arg, p);
        int64_t n = 0;
        while (p < arg.size() && std::isdigit((unsigned char)arg[p]))
            n = n * 10 + (arg[p++] - '0');
        curLine_ = (unsigned)n;
        return {};
    }

    diag_.warn({lineNo, 1, filename.c_str()},
               "unknown preprocessor directive '#" + dir + "'");
    return {};
}

// =============================================================================
// #define
// =============================================================================

void Preprocessor::handleDefine(const std::string &rest,
                                 const std::string &filename,
                                 unsigned lineNo) {
    size_t pos = 0;
    skipWS(rest, pos);
    if (pos >= rest.size() || !isIdentStart(rest[pos])) {
        diag_.error({lineNo, 1, filename.c_str()}, "invalid #define");
        return;
    }
    std::string name = readIdentAt(rest, pos);

    MacroDef def;
    def.name = name;
    def.loc  = {lineNo, 1, filename.c_str()};

    // Function-like if '(' immediately follows name (no space)
    if (pos < rest.size() && rest[pos] == '(') {
        def.isFunctionLike = true;
        if (!opts_.compatMode) {
            diag_.error({lineNo, 1, filename.c_str()},
                "function-like macros are not allowed in safe mode; "
                "use generic<T> functions instead (or pass --compat-preprocessor)");
            return;
        }
        // Parse parameter list
        ++pos; // skip '('
        skipWS(rest, pos);
        while (pos < rest.size() && rest[pos] != ')') {
            if (rest.substr(pos, 3) == "...") {
                def.isVariadic = true;
                pos += 3;
                def.params.push_back("__VA_ARGS__");
                break;
            }
            std::string param = readIdentAt(rest, pos);
            def.params.push_back(param);
            skipWS(rest, pos);
            if (pos < rest.size() && rest[pos] == ',') { ++pos; skipWS(rest, pos); }
        }
        if (pos < rest.size() && rest[pos] == ')') ++pos;
        skipWS(rest, pos);
    } else {
        // Object-like: skip whitespace before body
        skipWS(rest, pos);
    }

    def.body = (pos < rest.size()) ? rest.substr(pos) : std::string{};

    // In safe mode: reject ## (token pasting) and # stringification in body
    if (!opts_.compatMode) {
        for (size_t i = 0; i < def.body.size(); ++i) {
            if (def.body[i] == '#') {
                if (i + 1 < def.body.size() && def.body[i + 1] == '#') {
                    diag_.error({lineNo, 1, filename.c_str()},
                        "token pasting '##' is not allowed in safe mode");
                    return;
                }
                if (!def.isFunctionLike) {
                    // Object-like macros shouldn't have # either
                    diag_.error({lineNo, 1, filename.c_str()},
                        "stringification '#' is not allowed in safe mode");
                    return;
                }
            }
        }
    }

    // Validate object-like: body should be a constant expression
    // (we just accept it and let the parser/sema validate semantics)

    macros_[name] = std::move(def);
}

// =============================================================================
// #undef
// =============================================================================

void Preprocessor::handleUndef(const std::string &rest) {
    size_t pos = 0;
    skipWS(rest, pos);
    std::string name = readIdentAt(rest, pos);
    macros_.erase(name);
}

// =============================================================================
// Conditional compilation
// =============================================================================

void Preprocessor::handleIfdef(const std::string &name, bool negate) {
    bool defined = (macros_.count(name) > 0);
    bool cond    = negate ? !defined : defined;
    condStack_.push_back({cond, cond});
}

void Preprocessor::handleIf(const std::string &rest,
                              const std::string &filename,
                              unsigned lineNo) {
    int64_t val = evalCondExpr(rest, filename, lineNo);
    bool cond = (val != 0);
    condStack_.push_back({cond, cond});
}

void Preprocessor::handleElif(const std::string &rest,
                                const std::string &filename,
                                unsigned lineNo) {
    if (condStack_.empty()) {
        diag_.error({lineNo, 1, filename.c_str()}, "#elif without #if");
        return;
    }
    CondState &top = condStack_.back();
    if (top.anyTaken) {
        // A prior branch was taken: this branch is inactive
        top.active = false;
    } else {
        int64_t val = evalCondExpr(rest, filename, lineNo);
        bool cond   = (val != 0);
        top.active   = cond;
        top.anyTaken = cond;
    }
}

void Preprocessor::handleElse(const std::string &filename, unsigned lineNo) {
    if (condStack_.empty()) {
        diag_.error({lineNo, 1, filename.c_str()}, "#else without #if");
        return;
    }
    CondState &top = condStack_.back();
    top.active = !top.anyTaken;
    // anyTaken stays: if prior branch was taken, else is inactive
}

void Preprocessor::handleEndif(const std::string &filename, unsigned lineNo) {
    if (condStack_.empty()) {
        diag_.error({lineNo, 1, filename.c_str()}, "#endif without #if");
        return;
    }
    condStack_.pop_back();
}

// =============================================================================
// #include
// =============================================================================

std::string Preprocessor::handleInclude(const std::string &rest,
                                         const std::string &curFile,
                                         unsigned lineNo, int depth) {
    if (depth >= opts_.maxIncludeDepth) {
        diag_.error({lineNo, 1, curFile.c_str()},
                    "maximum #include depth exceeded");
        return {};
    }

    // Parse "filename" or <filename>
    size_t pos = 0;
    skipWS(rest, pos);
    if (pos >= rest.size()) {
        diag_.error({lineNo, 1, curFile.c_str()}, "invalid #include");
        return {};
    }

    bool isSystem = false;
    std::string name;
    if (rest[pos] == '"') {
        ++pos;
        while (pos < rest.size() && rest[pos] != '"') name += rest[pos++];
        ++pos; // closing "
    } else if (rest[pos] == '<') {
        isSystem = true;
        ++pos;
        while (pos < rest.size() && rest[pos] != '>') name += rest[pos++];
        ++pos; // closing >
    } else {
        diag_.error({lineNo, 1, curFile.c_str()},
                    "expected '\"filename\"' or '<filename>' after #include");
        return {};
    }

    if (name.empty()) {
        diag_.error({lineNo, 1, curFile.c_str()}, "empty filename in #include");
        return {};
    }

    std::string path = resolveInclude(name, isSystem, curFile);
    if (path.empty()) {
        diag_.error({lineNo, 1, curFile.c_str()},
                    "cannot find include file '" + name + "'");
        return {};
    }

    // Canonicalize for pragma-once tracking
    std::error_code ec;
    std::string canonical = fs::weakly_canonical(path, ec).string();
    if (!ec && pragmaOnceFiles_.count(canonical)) {
        return {}; // already included
    }

    std::string content = readFile(path);
    if (content.empty()) {
        // Empty file or read error — silently skip
        return {};
    }

    // Check for '#pragma once' at the top of included file
    {
        size_t p = 0;
        while (p < content.size() && (content[p]==' '||content[p]=='\t'||content[p]=='\n'||content[p]=='\r'))
            ++p;
        if (p + 12 < content.size() && content.substr(p, 13) == "#pragma once\n") {
            pragmaOnceFiles_.insert(canonical);
        }
    }

    return processFile(content, path, depth + 1);
}

std::string Preprocessor::resolveInclude(const std::string &name,
                                          bool isSystem,
                                          const std::string &curFile) {
    // 1. Relative to current file (for quoted includes)
    if (!isSystem && !curFile.empty()) {
        fs::path p = fs::path(curFile).parent_path() / name;
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    // 2. Search include paths
    for (auto &dir : opts_.includePaths) {
        fs::path p = fs::path(dir) / name;
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    // 3. System include paths (limited set for standalone use)
    if (isSystem) {
        static const char *sysDirs[] = {
            "/usr/include", "/usr/local/include",
            "/usr/local/Cellar/llvm/21.1.8_1/include", nullptr
        };
        for (int i = 0; sysDirs[i]; ++i) {
            fs::path p = fs::path(sysDirs[i]) / name;
            std::error_code ec;
            if (fs::exists(p, ec)) return p.string();
        }
    }
    return {};
}

// =============================================================================
// #pragma
// =============================================================================

void Preprocessor::handlePragma(const std::string &rest,
                                 const std::string &filename,
                                 unsigned lineNo) {
    size_t pos = 0;
    skipWS(rest, pos);
    std::string directive = readIdentAt(rest, pos);
    if (directive == "once") {
        // Mark current file as already included
        std::error_code ec;
        std::string canonical = fs::weakly_canonical(curFile_, ec).string();
        if (!ec) pragmaOnceFiles_.insert(canonical);
    }
    // Other pragmas ignored silently
}

void Preprocessor::handleError(const std::string &rest,
                                const std::string &filename,
                                unsigned lineNo) {
    diag_.error({lineNo, 1, filename.c_str()}, "#error " + trim(rest));
}

void Preprocessor::handleWarning(const std::string &rest,
                                  const std::string &filename,
                                  unsigned lineNo) {
    diag_.warn({lineNo, 1, filename.c_str()}, "#warning " + trim(rest));
}

// =============================================================================
// Macro expansion
// =============================================================================

std::string Preprocessor::expandLine(const std::string &line,
                                      const std::string &filename,
                                      unsigned lineNo) {
    std::unordered_set<std::string> guard;
    return expandTokens(line, guard, filename, lineNo);
}

std::string Preprocessor::expandTokens(const std::string &text,
                                        std::unordered_set<std::string> &guard,
                                        const std::string &filename,
                                        unsigned lineNo) {
    std::string result;
    result.reserve(text.size());
    size_t pos = 0;

    while (pos < text.size()) {
        char c = text[pos];

        // String literal: copy verbatim
        if (c == '"') {
            result += c; ++pos;
            while (pos < text.size()) {
                char sc = text[pos];
                result += sc; ++pos;
                if (sc == '\\' && pos < text.size()) {
                    result += text[pos++]; // escaped char
                } else if (sc == '"') {
                    break;
                }
            }
            continue;
        }
        // Char literal: copy verbatim
        if (c == '\'') {
            result += c; ++pos;
            while (pos < text.size()) {
                char sc = text[pos];
                result += sc; ++pos;
                if (sc == '\\' && pos < text.size()) {
                    result += text[pos++];
                } else if (sc == '\'') {
                    break;
                }
            }
            continue;
        }
        // Identifier: possibly a macro
        if (isIdentStart(c)) {
            size_t start = pos;
            std::string name = readIdentAt(text, pos);

            // Handle __FILE__ and __LINE__ built-ins
            if (name == "__FILE__") {
                result += '"';
                result += curFile_;
                result += '"';
                continue;
            }
            if (name == "__LINE__") {
                result += std::to_string(lineNo);
                continue;
            }
            // Disabled built-ins
            if (name == "__TIME__" || name == "__DATE__") {
                diag_.error({lineNo, 1, filename.c_str()},
                    "__TIME__ and __DATE__ are disabled in SafeC "
                    "(non-deterministic build info)");
                result += "0";
                continue;
            }

            auto it = macros_.find(name);
            if (it == macros_.end() || guard.count(name)) {
                // Not a macro, or recursion guard → output as-is
                result += name;
                continue;
            }

            const MacroDef &def = it->second;

            if (def.isFunctionLike) {
                // Skip whitespace to see if there's a '('
                size_t savedPos = pos;
                while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t'))
                    ++pos;
                if (pos < text.size() && text[pos] == '(') {
                    guard.insert(name);
                    std::string expanded = expandFunctionLike(def, text, pos, guard, filename, lineNo);
                    guard.erase(name);
                    // Re-expand result
                    std::string reexpanded = expandTokens(expanded, guard, filename, lineNo);
                    result += reexpanded;
                } else {
                    // No '(' → treat as non-invocation
                    pos = savedPos;
                    result += name;
                }
            } else {
                // Object-like macro: expand body
                guard.insert(name);
                std::string expanded = expandTokens(def.body, guard, filename, lineNo);
                guard.erase(name);
                result += expanded;
            }
            continue;
        }
        // Everything else: copy as-is
        result += c;
        ++pos;
    }
    return result;
}

std::string Preprocessor::expandFunctionLike(const MacroDef &def,
                                               const std::string &text,
                                               size_t &pos,
                                               std::unordered_set<std::string> &guard,
                                               const std::string &filename,
                                               unsigned lineNo) {
    // pos points at '('
    ++pos; // skip '('

    // Collect arguments (handle nested parens)
    std::vector<std::string> args;
    {
        std::string cur;
        int depth = 0;
        while (pos < text.size()) {
            char c = text[pos++];
            if (c == '(' ) { depth++; cur += c; }
            else if (c == ')') {
                if (depth == 0) {
                    args.push_back(trim(cur));
                    break;
                }
                depth--; cur += c;
            } else if (c == ',' && depth == 0) {
                args.push_back(trim(cur));
                cur.clear();
            } else {
                cur += c;
            }
        }
    }

    // Pre-expand each argument
    std::vector<std::string> expArgs;
    for (auto &a : args) {
        std::unordered_set<std::string> ag;
        expArgs.push_back(expandTokens(a, ag, filename, lineNo));
    }

    // Substitute into body
    std::string body = def.body;
    std::string result;
    size_t p = 0;
    while (p < body.size()) {
        // Token pasting ##
        if (body[p] == '#' && p + 1 < body.size() && body[p+1] == '#') {
            // Already validated in safe mode; in compat mode paste
            if (!result.empty() && result.back() == ' ') result.pop_back();
            p += 2;
            while (p < body.size() && (body[p]==' '||body[p]=='\t')) ++p;
            // Read next token to paste
            if (isIdentStart(body[p])) {
                std::string tok = readIdentAt(body, p);
                auto it = std::find(def.params.begin(), def.params.end(), tok);
                if (it != def.params.end()) {
                    size_t idx = (size_t)(it - def.params.begin());
                    std::string val = (idx < expArgs.size()) ? expArgs[idx] : std::string{};
                    result += val;
                } else {
                    result += tok;
                }
            }
            continue;
        }
        // Stringification #
        if (body[p] == '#' && def.isFunctionLike) {
            ++p;
            while (p < body.size() && (body[p]==' '||body[p]=='\t')) ++p;
            std::string tok = readIdentAt(body, p);
            auto it = std::find(def.params.begin(), def.params.end(), tok);
            if (it != def.params.end()) {
                size_t idx = (size_t)(it - def.params.begin());
                std::string raw = (idx < args.size()) ? args[idx] : std::string{};
                // Stringify: escape special chars
                result += '"';
                for (char c : raw) {
                    if (c == '"' || c == '\\') result += '\\';
                    result += c;
                }
                result += '"';
            }
            continue;
        }
        // Identifier: possibly a parameter
        if (isIdentStart(body[p])) {
            std::string tok = readIdentAt(body, p);
            auto it = std::find(def.params.begin(), def.params.end(), tok);
            if (it != def.params.end()) {
                size_t idx = (size_t)(it - def.params.begin());
                result += (idx < expArgs.size()) ? expArgs[idx] : std::string{};
            } else {
                result += tok;
            }
            continue;
        }
        result += body[p++];
    }

    return result;
}

// =============================================================================
// #if expression evaluator
// =============================================================================

int64_t Preprocessor::evalCondExpr(const std::string &expr,
                                    const std::string &filename,
                                    unsigned lineNo) {
    // First expand macros in the expression, but handle 'defined' specially
    // We do this by expanding everything except the operand of 'defined'
    std::string expanded;
    size_t pos = 0;
    while (pos < expr.size()) {
        skipWS(expr, pos);
        if (pos >= expr.size()) break;

        if (isIdentStart(expr[pos])) {
            size_t start = pos;
            std::string name = readIdentAt(expr, pos);
            if (name == "defined") {
                // Keep 'defined' and its argument unexpanded
                expanded += "defined";
                skipWS(expr, pos);
                if (pos < expr.size() && expr[pos] == '(') {
                    expanded += '('; ++pos;
                    skipWS(expr, pos);
                    std::string macName = readIdentAt(expr, pos);
                    skipWS(expr, pos);
                    if (pos < expr.size() && expr[pos] == ')') ++pos;
                    expanded += macName + ')';
                } else {
                    std::string macName = readIdentAt(expr, pos);
                    expanded += ' '; expanded += macName;
                }
            } else {
                // Expand the identifier as a macro if possible
                auto it = macros_.find(name);
                if (it != macros_.end() && !it->second.isFunctionLike) {
                    std::unordered_set<std::string> guard;
                    guard.insert(name);
                    expanded += expandTokens(it->second.body, guard, filename, lineNo);
                } else {
                    expanded += name;
                }
            }
        } else {
            expanded += expr[pos++];
        }
    }

    // Now evaluate the fully-expanded expression
    size_t p = 0;
    skipWS(expanded, p);
    int64_t result = parseTernary(expanded, p, filename, lineNo);
    return result;
}

// ── Recursive descent evaluator ───────────────────────────────────────────────

int64_t Preprocessor::parseTernary(const std::string &s, size_t &p,
                                    const std::string &f, unsigned ln) {
    int64_t cond = parseOr(s, p, f, ln);
    skipWS(s, p);
    if (p < s.size() && s[p] == '?') {
        ++p;
        int64_t then = parseTernary(s, p, f, ln);
        skipWS(s, p);
        if (p < s.size() && s[p] == ':') ++p;
        int64_t else_ = parseTernary(s, p, f, ln);
        return cond ? then : else_;
    }
    return cond;
}

int64_t Preprocessor::parseOr(const std::string &s, size_t &p,
                                const std::string &f, unsigned ln) {
    int64_t v = parseAnd(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p + 1 < s.size() && s[p] == '|' && s[p+1] == '|') {
            p += 2;
            int64_t r = parseAnd(s, p, f, ln);
            v = (v || r) ? 1 : 0;
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseAnd(const std::string &s, size_t &p,
                                 const std::string &f, unsigned ln) {
    int64_t v = parseBitOr(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p + 1 < s.size() && s[p] == '&' && s[p+1] == '&') {
            p += 2;
            int64_t r = parseBitOr(s, p, f, ln);
            v = (v && r) ? 1 : 0;
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseBitOr(const std::string &s, size_t &p,
                                   const std::string &f, unsigned ln) {
    int64_t v = parseBitXor(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p < s.size() && s[p] == '|' && (p+1 >= s.size() || s[p+1] != '|')) {
            ++p; v |= parseBitXor(s, p, f, ln);
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseBitXor(const std::string &s, size_t &p,
                                    const std::string &f, unsigned ln) {
    int64_t v = parseBitAnd(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p < s.size() && s[p] == '^') {
            ++p; v ^= parseBitAnd(s, p, f, ln);
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseBitAnd(const std::string &s, size_t &p,
                                    const std::string &f, unsigned ln) {
    int64_t v = parseEq(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p < s.size() && s[p] == '&' && (p+1 >= s.size() || s[p+1] != '&')) {
            ++p; v &= parseEq(s, p, f, ln);
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseEq(const std::string &s, size_t &p,
                                const std::string &f, unsigned ln) {
    int64_t v = parseRel(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p + 1 < s.size() && s[p] == '=' && s[p+1] == '=') {
            p += 2; v = (v == parseRel(s, p, f, ln)) ? 1 : 0;
        } else if (p + 1 < s.size() && s[p] == '!' && s[p+1] == '=') {
            p += 2; v = (v != parseRel(s, p, f, ln)) ? 1 : 0;
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseRel(const std::string &s, size_t &p,
                                 const std::string &f, unsigned ln) {
    int64_t v = parseShift(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p + 1 < s.size() && s[p] == '<' && s[p+1] == '=') {
            p += 2; v = (v <= parseShift(s, p, f, ln)) ? 1 : 0;
        } else if (p + 1 < s.size() && s[p] == '>' && s[p+1] == '=') {
            p += 2; v = (v >= parseShift(s, p, f, ln)) ? 1 : 0;
        } else if (p < s.size() && s[p] == '<' && (p+1>=s.size()||s[p+1]!='<')) {
            ++p; v = (v < parseShift(s, p, f, ln)) ? 1 : 0;
        } else if (p < s.size() && s[p] == '>' && (p+1>=s.size()||s[p+1]!='>')) {
            ++p; v = (v > parseShift(s, p, f, ln)) ? 1 : 0;
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseShift(const std::string &s, size_t &p,
                                   const std::string &f, unsigned ln) {
    int64_t v = parseAdd(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p + 1 < s.size() && s[p] == '<' && s[p+1] == '<') {
            p += 2; v <<= parseAdd(s, p, f, ln);
        } else if (p + 1 < s.size() && s[p] == '>' && s[p+1] == '>') {
            p += 2; v >>= parseAdd(s, p, f, ln);
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseAdd(const std::string &s, size_t &p,
                                 const std::string &f, unsigned ln) {
    int64_t v = parseMul(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p < s.size() && s[p] == '+' && (p+1>=s.size()||s[p+1]!='+')) {
            ++p; v += parseMul(s, p, f, ln);
        } else if (p < s.size() && s[p] == '-' && (p+1>=s.size()||s[p+1]!='-')) {
            ++p; v -= parseMul(s, p, f, ln);
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseMul(const std::string &s, size_t &p,
                                 const std::string &f, unsigned ln) {
    int64_t v = parseUnary(s, p, f, ln);
    while (true) {
        skipWS(s, p);
        if (p < s.size() && s[p] == '*') {
            ++p; v *= parseUnary(s, p, f, ln);
        } else if (p < s.size() && s[p] == '/') {
            ++p;
            int64_t r = parseUnary(s, p, f, ln);
            v = r ? v / r : 0;
        } else if (p < s.size() && s[p] == '%') {
            ++p;
            int64_t r = parseUnary(s, p, f, ln);
            v = r ? v % r : 0;
        } else break;
    }
    return v;
}

int64_t Preprocessor::parseUnary(const std::string &s, size_t &p,
                                   const std::string &f, unsigned ln) {
    skipWS(s, p);
    if (p < s.size() && s[p] == '!') { ++p; return !parseUnary(s, p, f, ln) ? 1 : 0; }
    if (p < s.size() && s[p] == '~') { ++p; return ~parseUnary(s, p, f, ln); }
    if (p < s.size() && s[p] == '-') { ++p; return -parseUnary(s, p, f, ln); }
    if (p < s.size() && s[p] == '+') { ++p; return +parseUnary(s, p, f, ln); }
    return parsePrimary(s, p, f, ln);
}

int64_t Preprocessor::parsePrimary(const std::string &s, size_t &p,
                                    const std::string &f, unsigned ln) {
    skipWS(s, p);
    if (p >= s.size()) return 0;

    // Parenthesized expression
    if (s[p] == '(') {
        ++p;
        int64_t v = parseTernary(s, p, f, ln);
        skipWS(s, p);
        if (p < s.size() && s[p] == ')') ++p;
        return v;
    }

    // Integer literal (decimal, hex, octal)
    if (std::isdigit((unsigned char)s[p])) {
        int64_t v = 0;
        if (p + 1 < s.size() && s[p] == '0' && (s[p+1] == 'x' || s[p+1] == 'X')) {
            p += 2;
            while (p < s.size() && std::isxdigit((unsigned char)s[p])) {
                v = v * 16 + (std::isdigit((unsigned char)s[p]) ?
                              s[p] - '0' : std::tolower((unsigned char)s[p]) - 'a' + 10);
                ++p;
            }
        } else if (s[p] == '0') {
            ++p;
            while (p < s.size() && s[p] >= '0' && s[p] <= '7')
                v = v * 8 + (s[p++] - '0');
        } else {
            while (p < s.size() && std::isdigit((unsigned char)s[p]))
                v = v * 10 + (s[p++] - '0');
        }
        // Skip integer suffixes
        while (p < s.size() && (s[p]=='u'||s[p]=='U'||s[p]=='l'||s[p]=='L')) ++p;
        return v;
    }

    // Character literal
    if (s[p] == '\'') {
        ++p;
        int64_t v = 0;
        if (p < s.size() && s[p] == '\\') {
            ++p;
            if (p < s.size()) {
                switch (s[p++]) {
                case 'n':  v = '\n'; break;
                case 't':  v = '\t'; break;
                case 'r':  v = '\r'; break;
                case '0':  v = '\0'; break;
                case '\\': v = '\\'; break;
                case '\'': v = '\''; break;
                default:   v = 0;   break;
                }
            }
        } else if (p < s.size()) {
            v = (unsigned char)s[p++];
        }
        while (p < s.size() && s[p] != '\'') ++p;
        if (p < s.size()) ++p; // closing '
        return v;
    }

    // Identifier: 'defined', or an unexpanded macro name (→ 0)
    if (isIdentStart(s[p])) {
        std::string name = readIdentAt(s, p);
        if (name == "defined") {
            skipWS(s, p);
            bool hasParen = (p < s.size() && s[p] == '(');
            if (hasParen) ++p;
            skipWS(s, p);
            std::string macName = readIdentAt(s, p);
            skipWS(s, p);
            if (hasParen && p < s.size() && s[p] == ')') ++p;
            return macros_.count(macName) ? 1 : 0;
        }
        // Unexpanded identifier → 0 (C standard behaviour)
        return 0;
    }

    return 0;
}

} // namespace safec
