#include "Formatter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace safeguard {

Formatter::Formatter(FormatOptions opts) : opts_(opts) {}

// ── formatSource ─────────────────────────────────────────────────────────────
//
// This is a brace-depth reindenter, not a full AST-aware pretty-printer like
// rustfmt or clang-format — it deliberately never touches:
//   - string/char literal contents
//   - comment text (line or block)
//   - horizontal spacing *within* a line, or line-wrapping/reflow
//
// What it does:
//   - recomputes each line's leading whitespace from '{'/'}' nesting depth
//     (a line starting with '}' is dedented one level before printing, so
//     "} else {" lines up with the 'if' that opened the block)
//   - preprocessor directive lines (first non-space char is '#') are always
//     emitted at column 0, matching this codebase's own convention
//   - trims trailing whitespace from every line
//   - collapses runs of 2+ blank lines down to exactly 1
//   - a line that starts inside a still-open block comment (from a previous
//     line) is left untouched — many block comments use intentional
//     interior alignment ('* foo' style) that reindenting would break
//
// This scope — safe, mechanical, and easy to reason about — was chosen over
// a full reflow formatter because SafeC's own Lexer discards comments
// entirely during tokenization (see Lexer::skipLineComment/skipBlockComment
// in the compiler): there is no lossless AST to round-trip through, so any
// formatter has to work from raw source text directly, and getting
// arbitrary reflow right at that level without accidentally corrupting
// string/comment content is a much larger undertaking than reindentation.
std::string Formatter::formatSource(const std::string& src) const {
    std::vector<std::string> rawLines;
    {
        std::istringstream iss(src);
        std::string line;
        while (std::getline(iss, line)) rawLines.push_back(line);
    }
    bool endsWithNewline = !src.empty() && src.back() == '\n';

    std::string indentUnit = opts_.useTabs ? "\t"
        : std::string((size_t)std::max(0, opts_.indentWidth), ' ');

    std::vector<std::string> out;
    int  depth = 0;
    bool inBlockComment = false;
    int  blankRun = 0;

    for (auto& rawLine : rawLines) {
        // Trim trailing whitespace unconditionally.
        std::string line = rawLine;
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' ||
                                  line.back() == '\r'))
            line.pop_back();

        bool startedInBlockComment = inBlockComment;

        // Leading whitespace, stripped, to inspect the first real character.
        size_t firstNonWs = line.find_first_not_of(" \t");
        std::string content = (firstNonWs == std::string::npos) ? ""
                                                                  : line.substr(firstNonWs);

        if (content.empty()) {
            ++blankRun;
            if (blankRun <= 1) out.push_back("");
            continue;
        }
        blankRun = 0;

        if (startedInBlockComment) {
            // Preserve this line's original indentation verbatim (minus the
            // trailing-whitespace trim already applied) — don't reindent
            // inside an unfinished block comment.
            out.push_back(line);
        } else {
            bool isPreprocessor = content[0] == '#';
            bool isCloser       = content[0] == '}';
            int  effectiveDepth = isCloser ? std::max(0, depth - 1) : depth;
            std::string indent;
            if (!isPreprocessor) {
                for (int i = 0; i < effectiveDepth; ++i) indent += indentUnit;
            }
            out.push_back(indent + content);
        }

        // Scan this line's content (from firstNonWs onward is enough — pure
        // leading whitespace can't contain braces/strings/comments) to
        // update depth/inBlockComment state, character by character.
        const std::string& scan = line;
        size_t i = startedInBlockComment ? 0 : firstNonWs;
        size_t n = scan.size();
        while (i < n) {
            if (inBlockComment) {
                size_t close = scan.find("*/", i);
                if (close == std::string::npos) { i = n; break; }
                inBlockComment = false;
                i = close + 2;
                continue;
            }
            char c = scan[i];
            if (c == '/' && i + 1 < n && scan[i + 1] == '/') {
                break; // rest of line is a line comment
            }
            if (c == '/' && i + 1 < n && scan[i + 1] == '*') {
                size_t close = scan.find("*/", i + 2);
                if (close == std::string::npos) { inBlockComment = true; i = n; break; }
                i = close + 2;
                continue;
            }
            if (c == '"') {
                ++i;
                while (i < n) {
                    if (scan[i] == '\\' && i + 1 < n) { i += 2; continue; }
                    if (scan[i] == '"') { ++i; break; }
                    ++i;
                }
                continue;
            }
            if (c == '\'') {
                ++i;
                while (i < n) {
                    if (scan[i] == '\\' && i + 1 < n) { i += 2; continue; }
                    if (scan[i] == '\'') { ++i; break; }
                    ++i;
                }
                continue;
            }
            if (c == '{') { ++depth; ++i; continue; }
            if (c == '}') { depth = std::max(0, depth - 1); ++i; continue; }
            ++i;
        }
    }

    // Drop a single trailing blank line accumulated by collapsing (matches
    // "file ends with exactly one newline, no trailing blank lines").
    while (!out.empty() && out.back().empty()) out.pop_back();

    std::string result;
    for (size_t i = 0; i < out.size(); ++i) {
        result += out[i];
        if (i + 1 < out.size() || endsWithNewline) result += "\n";
    }
    return result;
}

// ── run ───────────────────────────────────────────────────────────────────────
bool Formatter::run(const std::string& srcDir, bool checkOnly, bool verbose) const {
    if (!fs::exists(srcDir)) return true;

    std::vector<std::string> files;
    for (auto& e : fs::recursive_directory_iterator(srcDir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".sc" || ext == ".h") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    bool anyChanged = false;
    for (auto& path : files) {
        std::ifstream in(path);
        if (!in) continue;
        std::string src((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        std::string formatted = formatSource(src);
        if (formatted == src) continue;

        anyChanged = true;
        if (checkOnly) {
            std::cout << "would reformat: " << path << "\n";
        } else {
            std::ofstream out(path, std::ios::trunc);
            out << formatted;
            if (verbose) std::cout << "formatted: " << path << "\n";
        }
    }

    if (checkOnly) {
        if (!anyChanged) std::cout << "safeguard: all files formatted\n";
        return !anyChanged;
    }
    if (verbose && !anyChanged) std::cout << "safeguard: nothing to format\n";
    return true;
}

} // namespace safeguard
