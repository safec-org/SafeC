#pragma once
#include <string>

namespace safeguard {

class Project {
public:
    // Create a new project at `path/name` (the directory must not exist).
    // Populates Package.toml, src/main.sc, and a .gitignore. 'lang' picks
    // the entry point: "sc" (SafeC, default), or "c", "cpp", "objc",
    // "objcpp" for a package with no SafeC in it (src/main.c/.cpp/.m/.mm).
    // 'withStd' writes '[build] std = true' for such a package, so it builds
    // and links SafeC's std from the start (a SafeC package always does).
    static void createNew(const std::string& name,
                           const std::string& parentDir = ".",
                           const std::string& lang = "sc",
                           bool withStd = false);

    // Initialise the current directory as a safeguard project (Package.toml
    // + src/main.<ext> if they do not already exist).
    static void initHere(const std::string& name, const std::string& lang = "sc",
                         bool withStd = false);

    // Return the project root by searching upward from `startDir` for a
    // Package.toml file.  Returns "" if not found.
    static std::string findRoot(const std::string& startDir = ".");

private:
    // Instantiate a template file by substituting {{KEY}} placeholders.
    static std::string applyTemplate(const std::string& tmpl,
                                      const std::string& name,
                                      const std::string& version,
                                      const std::string& author);

    // Write `content` to `path`, creating parent directories as needed.
    static void writeFile(const std::string& path, const std::string& content);

    // Create directory (and parents).  No-op if already exists.
    static void mkdirP(const std::string& path);
};

} // namespace safeguard
