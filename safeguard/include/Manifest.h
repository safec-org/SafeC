#pragma once
#include <string>
#include <vector>
#include <optional>

namespace safeguard {

struct Dependency {
    std::string name;
    std::string version; // semver string, e.g. "1.0.0"
    std::string path;    // local path override (empty if registry dep)
};

struct PackageInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    std::string license;
};

struct BuildConfig {
    std::string edition;           // e.g. "2026"
    std::vector<std::string> srcs; // explicit source files (empty = auto-discover)
    std::string output;            // output binary/library name (defaults to package name)
    std::vector<std::string> cflags; // extra clang flags for final link
    std::vector<std::string> libs;    // external libraries: -l<name> at final link
    std::vector<std::string> libDirs; // library search dirs: -L<dir> at final link
    std::string lto;               // "" (off, default), "thin", or "full"
    std::string crateType;         // "bin" (default), "staticlib", or "cdylib"
};

struct Manifest {
    PackageInfo  package;
    BuildConfig  build;
    std::vector<Dependency> dependencies;

    // [features] table, Cargo-style: name -> list of other feature names it
    // turns on when enabled (an "optional dependency" style entry — SafeC
    // has no optional-dependency concept yet, so this only chains other
    // features). Declaration order preserved (matches Package.toml order).
    // A feature named "default" is not special to the parser — it's just
    // an ordinary entry — but resolveFeatures() treats it as the implicit
    // starting set unless the caller opts out.
    std::vector<std::pair<std::string, std::vector<std::string>>> features;
};

// Computes the enabled feature set for a build: starts from the "default"
// feature's list (unless noDefault), adds every name in 'requested', then
// repeatedly follows each enabled feature's own sub-feature list to a fixed
// point (so e.g. fullstack = ["frontend", "backend"] pulls both in when
// "fullstack" is requested or default). Unknown names in 'requested' or in
// a feature's sub-list are kept as plain leaf features (a feature need not
// be declared with its own [features] entry to be turned on — mirrors
// Cargo's "features are just string flags" model, minus optional deps).
// Result order is deterministic (first-enabled order), each name appears
// once.
std::vector<std::string> resolveFeatures(const Manifest& m,
                                          const std::vector<std::string>& requested,
                                          bool noDefault);

class ManifestParser {
public:
    // Parse a Package.toml file.  Throws std::runtime_error on malformed input.
    static Manifest parseFile(const std::string& path);

    // Parse from a string (for testing).
    static Manifest parseString(const std::string& src,
                                 const std::string& sourceName = "<string>");

    // Emit a canonical Package.toml for the given manifest.
    static std::string serialize(const Manifest& m);

private:
    // Internal helpers
    struct Token { std::string key, value; };
    static std::vector<Token> tokenize(const std::string& src,
                                        const std::string& srcName);
    static Manifest buildManifest(const std::vector<Token>& tokens,
                                   const std::string& srcName);
};

} // namespace safeguard
