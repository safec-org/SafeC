#pragma once
#include "Manifest.h"
#include "Lock.h"
#include <string>
#include <vector>

namespace safeguard {

// Source language, dispatched on file extension. SafeC compiles through
// safec (.sc -> .ll -> .o); C and C++ compile straight to .o via clang /
// clang++ — all three still land as ordinary object files, so they link
// together into one binary using SafeC's C ABI compatibility with no
// special glue needed.
enum class SrcLang { SafeC, C, Cpp };

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

    // Fast compile-only pass ('cargo check' equivalent): runs every project
    // source through its front end (safec's Sema for .sc, 'clang -fsyntax-only'
    // for .c/.cpp) far enough to report every error, but never builds the
    // standard library, never assembles an object file, and never links —
    // just the parts needed to know "does this compile," which is normally
    // the majority of a full build's wall time. Returns true if every file
    // is error-free.
    bool check();

    // Build and run every file under tests/ as an independent standalone
    // binary linked against the same stdlib/dependencies as the main
    // project (mirrors Rust's integration-test convention: each file in
    // tests/ is its own program with its own main(), not a set of #[test]
    // functions extracted from one binary — SafeC has no such attribute).
    // A test passes if its binary exits 0. Prints a per-file pass/fail
    // line plus a final summary; returns true iff every test passed (or
    // there is no tests/ directory at all, matching 'no tests' being a
    // trivially-passing state rather than a failure).
    bool test();

    // Build then execute the output binary.
    // Returns the exit code of the child process, or -1 on build failure.
    int run(const std::vector<std::string>& args = {});

    // Delete build artefacts.
    void clean();

    // Return path to the compiled binary (valid after a successful build()).
    std::string outputPath() const;

    // Check Package.lock against current state.
    // Prints warnings for mismatches; returns false if any dep SHA differs.
    bool checkLock() const;

    // ── Library support ──────────────────────────────────────────────────────

    // Build a static library from all .sc/.c/.cpp files in `srcDir`
    // (recursively). Outputs to `libOut` (e.g., "build/libfoo.a").
    // 'compatPreprocessor' passes --compat-preprocessor to safec — needed for
    // the standard library itself, which (unlike ordinary user code) relies
    // on function-like macros in a few headers (e.g. ring buffer static
    // storage declarations).
    // 'tolerateArchMismatch': skip (warn, don't abort) a source file that
    // fails to assemble to real machine code for the current target — used
    // only for the standard library, which ships architecture-specific
    // files (e.g. std/hal/riscv.sc's CSR asm, std/hal/aarch64.sc's MRS/MSR
    // asm) alongside portable ones in the same tree; a native aarch64
    // build has no way to assemble riscv.sc's instructions and was never
    // going to call its functions anyway. A genuine SafeC-level error
    // (parse/type-check failure, reported by safec itself rather than by
    // the downstream assembler) still aborts unconditionally either way —
    // this only tolerates the assembler rejecting instructions safec
    // itself accepted as syntactically valid inline asm text. User project
    // sources always use the strict (non-tolerant) default.
    // 'foreignIncludeDirs': include search dirs used for .c/.cpp files
    // instead of 'includeDirs' — kept separate because 'includeDirs' for
    // the SafeC side typically includes SafeC's own std/ directory, whose
    // headers (e.g. std/stdint.h) use '#define' rather than 'typedef' and
    // corrupt a real C/C++ compiler's own <cstdint>/<vector>/... headers
    // if put on their search path. Defaults to empty (no extra include
    // dirs for foreign files) when not given.
    // Returns true on success.
    bool buildLib(const std::string& srcDir,
                  const std::string& libOut,
                  const std::vector<std::string>& includeDirs = {},
                  bool compatPreprocessor = false,
                  bool tolerateArchMismatch = false,
                  const std::vector<std::string>& foreignIncludeDirs = {});

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

    // Locate a clang/clang++ whose LLVM version matches (or is at least as
    // new as) the one safec was built against. safec's --emit-llvm output
    // can use very recent LLVM IR vocabulary (e.g. a newly added function
    // attribute string on an intrinsic declaration); a different, older
    // clang picked up first on PATH may print such text fine from its own
    // codegen but fail to *parse* it back when assembling .ll -> .o, since
    // the writer and reader must agree on vocabulary. Checked in order:
    // $SAFEC_CLANG/$SAFEC_CLANGXX, common LLVM install prefixes, then a
    // bare PATH lookup as the final fallback (unchanged prior behavior).
    // Use ONLY for assembling safec's own .ll output (llToObj) — a
    // standalone LLVM install's clang++ isn't necessarily configured with
    // this machine's system SDK/libc++ search paths (a real, observed
    // failure: Homebrew clang++ compiling a plain '#include <vector>' file
    // failed to find libc++'s own <stddef.h>), so genuine .c/.cpp source
    // compilation and final linking should use findSystemClang() instead.
    std::string findClang(bool cxx) const;

    // Locate a "normal", properly-configured-for-this-machine clang/clang++
    // — checks $SAFEC_CLANG/$SAFEC_CLANGXX, otherwise a bare PATH lookup.
    // Used for compiling genuine .c/.cpp source (compileForeign) and for
    // the final link (linkFinal): neither involves re-parsing safec's own
    // LLVM IR text, so there's no reason to prefer a specific LLVM
    // install's clang over whatever this system's default already is —
    // doing so risks exactly the opposite problem findClang() exists to
    // avoid (a toolchain that doesn't agree with the system's own SDK/
    // runtime library search paths).
    std::string findSystemClang(bool cxx) const;

    // Locate the SafeC std/ directory (sibling of the compiler dir).
    std::string findStdDir() const;

    // Collect all *.sc source files under srcDir.
    std::vector<std::string> collectSources(const std::string& srcDir) const;

    // Collect *.sc, *.c, *.cc, *.cpp, *.cxx source files under srcDir,
    // sorted together so mixed-language builds are deterministic.
    std::vector<std::string> collectAllSources(const std::string& srcDir) const;

    // Classify a source file by extension. Unrecognized extensions default
    // to SafeC (matches the pre-mixed-language behavior of collectSources,
    // which only ever found .sc files).
    static SrcLang langOf(const std::string& path);

    // Compile one .sc file to a .ll file.  Returns the .ll path, or "".
    // 'suppressOutput': discard the child compiler's own stdout/stderr
    // (unless verbose) — see runCmd's comment; only buildLib's
    // tolerateArchMismatch path sets this.
    std::string compileSrc(const std::string& safecBin,
                            const std::string& srcPath,
                            const std::string& buildDir,
                            const std::vector<std::string>& includeDirs = {},
                            bool compatPreprocessor = false,
                            bool suppressOutput = false) const;

    // Compile .ll to .o with clang.  Returns .o path or "".
    std::string llToObj(const std::string& llFile,
                         const std::string& buildDir,
                         bool suppressOutput = false) const;

    // Compile a .c/.cpp/.cc/.cxx file straight to a .o with clang/clang++.
    // Returns the .o path, or "" on failure.
    std::string compileForeign(const std::string& srcPath,
                                const std::string& buildDir,
                                const std::vector<std::string>& includeDirs,
                                SrcLang lang,
                                bool suppressOutput = false) const;

    // Pack .o files into a static archive with ar.  Returns true on success.
    bool packArchive(const std::vector<std::string>& objFiles,
                      const std::string& archivePath) const;

    // Link all .o files + extra libs. Uses clang++ as the link driver when
    // useCxxDriver is set (any C++ sources were compiled into the build) so
    // the C++ runtime (libc++/libstdc++, exceptions, RTTI) links in; plain
    // clang otherwise. Returns true on success.
    bool linkFinal(const std::vector<std::string>& objFiles,
                    const std::vector<std::string>& archives,
                    const std::string& output,
                    bool useCxxDriver = false) const;

    // Fork + exec a command, wait for it, return exit code.
    // 'suppressOutput': when true and 'verbose' is false, discard the
    // child's stdout/stderr instead of letting it print — see the call
    // site comment in runCmd's definition for why this exists.
    static int runCmd(const std::vector<std::string>& argv, bool verbose,
                       bool suppressOutput = false);

    // Clone a git repository.  Returns true on success.
    bool gitClone(const std::string& url, const std::string& dest) const;

    // ── Reproducible builds ───────────────────────────────────────────────────

    // Write Package.lock after a successful build.
    void writeLock(const std::vector<std::string>& srcFiles) const;
};

} // namespace safeguard
