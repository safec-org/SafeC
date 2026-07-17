#pragma once
#include <string>
#include <vector>

namespace safeguard {

struct FormatOptions {
    int  indentWidth = 4;    // spaces per brace-nesting level
    bool useTabs      = false;
};

// Result of formatting a single file.
struct FormatResult {
    std::string file;
    bool        changed = false;   // formatted text differs from input
    std::string formatted;         // the formatted text
};

// safeguard fmt: a brace-depth-aware reindenter and whitespace normalizer
// for .sc/.h source, in the spirit of Rust's 'cargo fmt' but deliberately
// scoped narrower — see the doc comment on Formatter::formatSource for
// exactly what it does and does not do, and why.
class Formatter {
public:
    explicit Formatter(FormatOptions opts = {});

    // Format a single file's contents. Never touches string/char literal
    // contents or comment text — only leading-whitespace (indentation),
    // trailing whitespace, tab/space normalization, and collapsing runs of
    // blank lines to at most one.
    std::string formatSource(const std::string& src) const;

    // Format every .sc/.h file under srcDir (recursively).
    // checkOnly: don't write files, just report which ones would change
    // (mirrors 'cargo fmt --check' / 'gofmt -l').
    // Returns true if nothing needed formatting (or --check found no
    // diffs); false if any file was reformatted (or, under --check,
    // would be).
    bool run(const std::string& srcDir, bool checkOnly, bool verbose) const;

private:
    FormatOptions opts_;
};

} // namespace safeguard
