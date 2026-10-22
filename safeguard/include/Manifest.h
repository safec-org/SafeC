#pragma once
#include <string>
#include <vector>
#include <optional>

namespace safeguard {

// Where a dependency comes from: a git URL in 'version' (cloned under
// deps/<name>), a local directory in 'path', or neither when 'lib' names
// prebuilt library files directly. A directory with a src/ is a SafeC
// source package built to build/deps/lib<name>.a; one without is a
// prebuilt library directory (libraries in lib/, lib64/ or the directory
// itself, headers in include/). 'build' makes the directory a C/C++/ObjC
// project safeguard builds itself instead: "sources" compiles its source
// files into build/deps/<name>/lib<name>.a, "cmake" configures, builds and
// installs it into build/deps/<name>/install.
struct Dependency {
    std::string name;
    std::string version; // git URL to clone
    std::string path;    // local directory (relative to the project root)
    std::vector<std::string> libs;     // 'lib': prebuilt library files, linked as is
                                       // (for build = "cmake": against the install prefix)
    std::vector<std::string> includes; // 'include': header dirs (prebuilt libraries, or
                                       // a built dependency's public headers)
    std::string build;                 // "" (none), "sources" or "cmake"
    std::vector<std::string> sources;  // build = "sources": files, directories or
                                       // '*' patterns, against the dependency's directory
    std::vector<std::string> defines;  // build = "sources": -D for its compiles
    std::vector<std::string> cflags;   // build = "sources": extra compile flags
    std::vector<std::string> cmakeArgs;  // build = "cmake": extra configure arguments
    std::vector<std::string> frameworks; // macOS frameworks its library needs
};

struct PackageInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    std::string license;
};

// [build.macos] / [build.linux] / [build.windows]: settings that apply only
// when building for that OS (the --target's, or the host's), appended to
// [build]'s own.
struct PlatformBuildConfig {
    std::vector<std::string> cflags;
    std::vector<std::string> libs;
    std::vector<std::string> libDirs;
    std::vector<std::string> frameworks;
};

struct BuildConfig {
    std::string edition;           // e.g. "2026"
    std::vector<std::string> srcs; // explicit source files (empty = auto-discover)
    std::string output;            // output binary/library name (defaults to package name)
    std::vector<std::string> cflags; // extra clang flags for final link
    std::vector<std::string> libs;    // external libraries: -l<name> at final link
    std::vector<std::string> libDirs; // library search dirs: -L<dir> at final link
    std::string lto;               // "" (off, default), "thin", or "full"
    std::string blas;              // CPU BLAS for std's sgemm: "" / "auto" (default: AOCL-BLAS on
                                   // AMD, oneMKL on Intel, Accelerate on macOS, if found),
                                   // "aocl", "mkl", "accelerate", "openblas" or "none"
    std::string crateType;         // "bin" (default), "staticlib", or "cdylib"
    std::vector<std::string> frameworks; // macOS frameworks: -framework <name> at final link
    bool objcArc = false;          // compile .m/.mm with -fobjc-arc
    PlatformBuildConfig forMacos, forLinux, forWindows;   // [build.<os>]
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
