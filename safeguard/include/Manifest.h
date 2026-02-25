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
    std::string edition;           // e.g. "2025"
    std::vector<std::string> srcs; // explicit source files (empty = auto-discover)
    std::string output;            // output binary name (defaults to package name)
    std::vector<std::string> cflags; // extra clang flags for final link
};

struct Manifest {
    PackageInfo  package;
    BuildConfig  build;
    std::vector<Dependency> dependencies;
};

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
