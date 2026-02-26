#pragma once
#include "Manifest.h"
#include "Lock.h"
#include <string>
#include <vector>

namespace safeguard {

struct BuildOptions {
    bool   release = false;       // enable optimisations (-O2)
    bool   emitLLVM = false;      // stop after safec --emit-llvm
    bool   verbose = false;       // print every command before running it
    std::string extraFlags;       // appended to every safec invocation
};

class Builder {
public:
    explicit Builder(std::string projectRoot, Manifest manifest,
                     BuildOptions opts = {});

    // Run a full build (.sc → IR → static lib + link).
    // Returns true on success.
    bool build();

    // Build then execute the output binary.
    // Returns the exit code of the child process, or -1 on build failure.
    int run(const std::vector<std::string>& args = {});

    // Delete build artefacts.
    void clean();

    // Return path to the compiled binary (valid after a successful build()).
    std::string outputPath() const;

    // ── Library support ──────────────────────────────────────────────────────

    // Build a static library from all .sc files in `srcDir` (recursively).
    // Outputs to `libOut` (e.g., "build/libfoo.a").
    // Returns true on success.
    bool buildLib(const std::string& srcDir,
                  const std::string& libOut,
                  const std::vector<std::string>& includeDirs = {});

    // Build (or use cached) SafeC standard library.
    // Returns the path to the built .a, or "" on failure.
    std::string ensureStdLib();

    // ── Dependency management ─────────────────────────────────────────────────

    // Fetch and build all dependencies listed in the manifest.
    // Dependencies are cloned under <root>/deps/<name>/ and built to
    // <root>/build/deps/lib<name>.a.
    // Returns true if all deps succeed.
    bool fetchAndBuildDeps();

private:
    std::string   root_;
    Manifest      manifest_;
    BuildOptions  opts_;

    // Locate the safec binary.
    std::string findSafec() const;

    // Locate the SafeC std/ directory (sibling of the compiler dir).
    std::string findStdDir() const;

    // Collect all *.sc source files under srcDir.
    std::vector<std::string> collectSources(const std::string& srcDir) const;

    // Compile one .sc file to a .ll file.  Returns the .ll path, or "".
    std::string compileSrc(const std::string& safecBin,
                            const std::string& srcPath,
                            const std::string& buildDir,
                            const std::vector<std::string>& includeDirs = {}) const;

    // Compile .ll to .o with clang.  Returns .o path or "".
    std::string llToObj(const std::string& llFile,
                         const std::string& buildDir) const;

    // Pack .o files into a static archive with ar.  Returns true on success.
    bool packArchive(const std::vector<std::string>& objFiles,
                      const std::string& archivePath) const;

    // Link all .o files + extra libs with clang.  Returns true on success.
    bool linkFinal(const std::vector<std::string>& objFiles,
                    const std::vector<std::string>& archives,
                    const std::string& output) const;

    // Fork + exec a command, wait for it, return exit code.
    static int runCmd(const std::vector<std::string>& argv, bool verbose);

    // Clone a git repository.  Returns true on success.
    bool gitClone(const std::string& url, const std::string& dest) const;

    // ── Reproducible builds ───────────────────────────────────────────────────

    // Write Package.lock after a successful build.
    void writeLock(const std::vector<std::string>& srcFiles) const;

    // Check Package.lock against current state.
    // Prints warnings for mismatches; returns false if any dep SHA differs.
    bool checkLock() const;
};

} // namespace safeguard
