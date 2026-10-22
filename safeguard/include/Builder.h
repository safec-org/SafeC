#pragma once
#include "Manifest.h"
#include "Lock.h"
#include "StdModules.h"
#include <set>
#include <string>
#include <vector>

namespace safeguard {

// Source language, dispatched on file extension. SafeC compiles through
// safec (.sc -> .ll -> .o); C, C++, Objective-C (.m) and Objective-C++
// (.mm) compile straight to .o via clang / clang++ — all of them still land
// as ordinary object files, so they link together into one binary using
// SafeC's C ABI compatibility with no special glue needed.
enum class SrcLang { SafeC, C, Cpp, ObjC, ObjCpp };
// Compiled with clang++ / linked with the C++ runtime.
inline bool isCxx(SrcLang l) { return l == SrcLang::Cpp || l == SrcLang::ObjCpp; }
// Needs the Objective-C runtime (-lobjc); macOS only.
inline bool isObjC(SrcLang l) { return l == SrcLang::ObjC || l == SrcLang::ObjCpp; }

struct BuildOptions {
    bool   release = false;       // enable optimisations (-O2)
    bool   emitLLVM = false;      // stop after safec --emit-llvm
    bool   verbose = false;       // print every command before running it
    std::string extraFlags;       // appended to every safec invocation

    // Rust-Cargo-style [features] selection. 'features' are explicitly
    // requested feature names (--features a,b); 'noDefaultFeatures' skips
    // the manifest's "default" feature entry (--no-default-features).
    // Builder::build()/check()/test() resolve these against the manifest's
    // [features] table (see Manifest.h's resolveFeatures) and pass each
    // enabled feature to safec as -DSAFEC_FEATURE_<UPPERNAME>=1, so project
    // sources gate code with '#ifdef SAFEC_FEATURE_<NAME>' — reusing
    // safec's existing preprocessor rather than adding new language syntax.
    std::vector<std::string> features;
    bool   noDefaultFeatures = false;

    // --target <alias|triple>: cross-compile (a bare-metal alias from
    // 'safec --list-targets', or any LLVM triple). Empty = host.
    std::string target;

    // Set by 'safeguard bundle': the executable is linked for its app
    // bundle (rpath to the bundled libraries; on Windows the GUI subsystem
    // and the icon/version/manifest resource). 'signIdentity' (--sign)
    // overrides [bundle] codesign_identity.
    bool bundle = false;
    std::string signIdentity;
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

    // Build the bin target for its [bundle] and package it: build/<name>.app
    // on macOS, build/<name>/ (exe, DLLs, resources) on Windows, and a .deb
    // plus build/<name>.AppDir on Linux. Returns true on success.
    bool bundle();

    // Build then execute the output binary.
    // Returns the exit code of the child process, or -1 on build failure.
    int run(const std::vector<std::string>& args = {});

    // Delete build artefacts.
    void clean();

    // Return path to the compiled binary (valid after a successful build()).
    std::string outputPath() const;
    // build/include: the headers a library build exports beside its
    // build/lib<output>.a/.so/.dylib, so build/ can be consumed as a
    // prebuilt dependency (its include/ and its library).
    std::string exportedIncludeDir() const;
    // A source's key for names under build/ and Package.lock (see Builder.cpp).
    std::string sourceKey(const std::string& src) const;
    std::string flatBuildName(const std::string& src) const;
    // Windows cdylib: build/<output>.lib, the import library dependents
    // link against (the DLL's exports are listed in build/<output>.def).
    std::string importLibPath() const;

    // Check Package.lock against current state.
    // Prints warnings for mismatches; returns false if any dep SHA differs.
    bool checkLock() const;

    // ── Library support ──────────────────────────────────────────────────────

    // Build a static library from all .sc/.c/.cpp files in `srcDir`
    // (recursively) — a SafeC dependency built from source. Outputs to
    // `libOut` (e.g., "build/libfoo.a").
    // 'compatPreprocessor' passes --compat-preprocessor to safec.
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
                  const std::vector<std::string>& foreignIncludeDirs = {});

    // Build (or reuse) the parts of SafeC's standard library a build uses:
    // the std modules 'roots' (std headers/units, see stdRoots) reach, and
    // the 'runtime-symbol' units the build's code ('refs': the external
    // symbols of its own and its dependencies' IR, see stdRefs) and those
    // modules' own code refer to, to a fixed point — or all of them with
    // [build] std = "all" — each unit compiled to its own cached object,
    // packed into build/deps/[<target>/]libsafec_std.a. A native build
    // links the prebuilt archive of a release install instead when there is
    // a matching one (see findPrebuiltStd), compiling only what it lacks.
    // Sets stdArchive_ ("" when nothing is to be linked) and the system
    // libraries those modules need (stdLinkFlags_, stdDebDepends_; see
    // StdModules.h). With 'link' false, only works out the modules (into
    // stdUnits_), compiling what that needs, for 'std-modules'. Returns
    // false on failure.
    bool ensureStdLib(const std::set<std::string>& roots, const std::set<std::string>& refs,
                      bool link = true);

    // The std files the given project files, its dependencies' sources and
    // its prebuilt SafeC libraries include.
    std::set<std::string> stdRoots(const std::vector<std::string>& files) const;
    // The external symbols the build's dependencies refer to (the '.refs'
    // beside a SafeC library; see ensureStdLib).
    std::set<std::string> depStdRefs() const;

    // 'safeguard std-modules': what the build (or, given std headers and
    // files in 'args', those) takes from std, and the system libraries.
    bool printStdModules(const std::vector<std::string>& args);

    // 'safeguard std-prebuilt <prefix>': compile every hosted std unit that
    // no feature gates, in release mode, into
    // <prefix>/lib/<host triple>/libsafec_std.a with its .stamp (compiler
    // and target) and .units (each unit's input hash and external symbols)
    // -- the archive a release install ships and native builds link
    // instead of compiling std.
    bool writePrebuiltStd(const std::string& prefix);

    // ── Dependency management ─────────────────────────────────────────────────

    // Fetch and build all dependencies listed in the manifest.
    // Git dependencies are cloned under <root>/deps/<name>/; source
    // packages are built to <root>/build/deps/lib<name>.a, prebuilt
    // libraries are checked to exist (see Dependency in Manifest.h).
    // Returns true if all deps succeed.
    bool fetchAndBuildDeps();

    // What a dependency contributes to a build, once fetched.
    struct DepLayout {
        std::string dir;                          // its directory ("" for lib-only)
        bool fromSource = false;                  // built from <dir>/src
        bool native = false;                      // build = "sources"/"cmake" (buildNativeDep)
        bool cxx = false;                         // brings C++ in: link with clang++
        std::string outDir;                       // native: build/deps/<name>
        std::vector<std::string> safecIncludes;   // include dirs for SafeC sources
        std::vector<std::string> foreignIncludes; // include dirs for C/C++ sources
        std::vector<std::string> linkInputs;      // library files and link flags (rpath,
                                                  // -lobjc, -Wl,-framework,<name>)
        std::vector<std::string> runtimeLibs;     // shared libraries a program loads at
                                                  // run time (.dll copied beside it on
                                                  // Windows; bundled by 'bundle')
        std::string error;                        // why it can't be used, if so
    };
    DepLayout depLayout(const Dependency& dep) const;

private:
    std::string   root_;
    Manifest      manifest_;
    BuildOptions  opts_;

    // The std archive this build links, and what its modules link (set by
    // ensureStdLib).
    std::string stdArchive_;
    std::vector<std::string> stdLinkFlags_;
    std::vector<std::string> stdDebDepends_;
    std::set<std::string> stdUnits_;          // the modules (ensureStdLib)
    bool stdFromSource_ = false;              // never the prebuilt archive (writePrebuiltStd)
    bool writingPrebuiltStd_ = false;         // 'std-prebuilt' (see stdClangFlags)

    // One std unit's object: compiled into objDir (or reused from there)
    // with the symbols its IR refers to kept beside it (<stem>.refs).
    // 'skipped': it doesn't compile for this target (a .skip marker).
    struct StdUnitObj {
        std::string obj;
        bool skipped = false;
        std::set<std::string> refs;
    };
    bool stdUnitObject(const StdIndex& idx, const std::string& unit, const std::string& objDir,
                       bool compile, StdUnitObj& out) const;
    // A hash of everything a std unit's compile reads, hex.
    std::string stdUnitHash(const StdIndex& idx, const std::string& unit) const;
    mutable std::map<std::string, std::string> fileHashes_;   // std file -> hashFile

    // A release install's prebuilt std archive (see writePrebuiltStd).
    struct PrebuiltStd {
        std::string archive;                                  // "" if none
        std::map<std::string, std::string> unitHash;          // unit -> input hash
        std::map<std::string, std::set<std::string>> refs;    // unit -> external symbols
        std::set<std::string> skipped;                        // didn't compile for it
    };
    // The one beside the safec this build uses (<bin>/../lib/<triple>/) or
    // under $SAFEC_HOME/lib/<triple>/, for a native build, if its stamp
    // matches this compiler and target and nothing opts out
    // ([build] std_prebuilt = false, SAFEC_STD_FROM_SOURCE=1).
    PrebuiltStd findPrebuiltStd() const;
    static bool readPrebuiltUnits(const std::string& file, PrebuiltStd& out);
    // The stamp a prebuilt archive for this compiler and host carries.
    std::string prebuiltStdStamp() const;
    // The clang flags every std unit is compiled with, in debug and release
    // builds alike and as the prebuilt std is: -O2, and for macOS the
    // deployment target (kStdMacosMin, or MACOSX_DEPLOYMENT_TARGET when set;
    // always kStdMacosMin for 'std-prebuilt').
    std::vector<std::string> stdClangFlags() const;
    // The oldest macOS a release's binaries and prebuilt std run on.
    static constexpr const char* kStdMacosMin = "11.0";
    // The host as a target triple (aarch64-apple-darwin, x86_64-unknown-
    // linux-gnu, x86_64-pc-windows-msvc, ...): a prebuilt archive's directory.
    static std::string hostTriple();
    // The external symbols of the SafeC sources among 'files', compiled to
    // IR into 'outDir' (for 'std-modules'). False if one doesn't compile.
    bool sourceRefs(const std::vector<std::string>& files, const std::string& outDir,
                    std::set<std::string>& refs);
    // The include directories for SafeC sources: std, the project's
    // include/ and its dependencies'.
    std::vector<std::string> safecIncludeDirs() const;
    // The target, its OS and the enabled features, for std's modules.
    StdPlatform stdPlatform() const;

    // Locate the safec binary.
    std::string findSafec() const;

    // --target support: the clang flags matching opts_.target (from
    // 'safec --print-target-clang-flags'; empty for the host), whether the
    // target has no OS, and the per-target build subdirectory name.
    const std::vector<std::string>& targetClangFlags() const;
    bool targetIsBareMetal() const;
    bool targetIsWasm() const;
    bool isCrossTarget() const { return !opts_.target.empty(); }
    std::string targetTriple() const;
    std::string targetDirName() const { return opts_.target.empty() ? "host" : opts_.target; }
    // "macos", "linux" or "windows" for the --target (the host without
    // one), "" for bare metal or anything else; picks [build.<os>].
    std::string targetOS() const;
    bool targetIsWindows() const { return targetOS() == "windows"; }
    // The C runtime every object and the link use: on Windows (MSVC ABI)
    // the shared one (msvcrt / ucrtbase DLLs, clang's -fms-runtime-lib=dll,
    // /MD), so a cdylib and the program that loads it share one heap and
    // memory may cross the DLL boundary. Empty elsewhere.
    std::vector<std::string> runtimeFlags() const;
    // ... and at the link: clang's driver names libcmt (the static CRT)
    // whatever the objects ask for, so the link swaps it for msvcrt.
    std::vector<std::string> runtimeLinkFlags() const;
    void stdForeignInclude(std::vector<std::string>& foreignIncs, const std::string& stdDir) const;
    const PlatformBuildConfig* platformBuild() const;
    // [build] cflags followed by [build.<os>] cflags.
    std::vector<std::string> compileFlags() const;
    // libc-named functions (malloc, free, ...) the project itself defines —
    // passed to safec as --provides on bare-metal builds so calls to them
    // aren't flagged as needing a hosted runtime.
    const std::string& providedFunctionsArg() const;
    mutable bool providesResolved_ = false;
    mutable std::string providesArg_;
    mutable bool targetFlagsResolved_ = false;
    mutable std::vector<std::string> targetFlags_;

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
    // 'excludeSubdirNames': any path whose components include one of
    // these names (case-sensitive, exact directory-name match — e.g.
    // "wasm" matches ".../std/wasm/..." but not ".../std/webassembly/...")
    // is skipped.
    std::vector<std::string> collectAllSources(
        const std::string& srcDir,
        const std::vector<std::string>& excludeSubdirNames = {}) const;

    // Classify a source file by extension. Unrecognized extensions default
    // to SafeC (matches the pre-mixed-language behavior of collectSources,
    // which only ever found .sc files).
    static SrcLang langOf(const std::string& path);

    // Compile one .sc file to a .ll file.  Returns the .ll path, or "".
    // 'suppressOutput': discard the child compiler's own stdout/stderr
    // (unless verbose) — see runCmd's comment; only buildLib's
    // tolerateArchMismatch path sets this.
    // 'defines': extra -DNAME[=VALUE] flags (used to pass enabled [features]
    // through as SAFEC_FEATURE_<NAME>=1 — see BuildOptions::features).
    std::string compileSrc(const std::string& safecBin,
                            const std::string& srcPath,
                            const std::string& buildDir,
                            const std::vector<std::string>& includeDirs = {},
                            bool compatPreprocessor = false,
                            bool suppressOutput = false,
                            const std::vector<std::string>& defines = {},
                            const std::string& outStem = "") const;

    // ── [features] ────────────────────────────────────────────────────────────

    // Resolves opts_.features/noDefaultFeatures against manifest_.features
    // (see Manifest.h's resolveFeatures) into the final enabled-feature set.
    std::vector<std::string> enabledFeatures() const;

    // enabledFeatures(), reshaped into '-DSAFEC_FEATURE_<UPPERNAME>=1' argv
    // entries ready to append to a safec invocation.
    std::vector<std::string> featureDefines() const;

    // ── .scx (HTML-templating source) ───────────────────────────────────────

    // If 'src' has a .scx extension, transpiles it (see ScxTranspiler.h)
    // into plain SafeC source written under buildDir/scx/ and returns that
    // generated .sc path; otherwise returns 'src' unchanged. Returns "" and
    // prints a "safeguard: scx: ..." error on malformed markup.
    std::string materializeScx(const std::string& src,
                                const std::string& buildDir) const;

    // Compile .ll to .o with clang.  Returns .o path or "". 'stdUnit': a
    // std unit, compiled with stdClangFlags whatever the build's mode.
    std::string llToObj(const std::string& llFile,
                         const std::string& buildDir,
                         bool suppressOutput = false,
                         bool stdUnit = false) const;

    // Compile a .c/.cpp/.cc/.cxx/.m/.mm file straight to a .o with
    // clang/clang++. 'flags' replaces the project's cflags (a built
    // dependency's own defines/cflags). Returns the .o path, or "" on failure.
    std::string compileForeign(const std::string& srcPath,
                                const std::string& buildDir,
                                const std::vector<std::string>& includeDirs,
                                SrcLang lang,
                                bool suppressOutput = false,
                                const std::vector<std::string>* flags = nullptr) const;

    // Pack .o files into a static archive with ar.  Returns true on success.
    bool packArchive(const std::vector<std::string>& objFiles,
                      const std::string& archivePath) const;

    // Link all .o files + extra libs. Uses clang++ as the link driver when
    // useCxxDriver is set (any C++ sources were compiled into the build) so
    // the C++ runtime (libc++/libstdc++, exceptions, RTTI) links in; plain
    // clang otherwise. Also applies manifest_.build.libs/libDirs/lto
    // (-l/-L/-flto). 'shared' produces a dynamic library (-shared) instead
    // of an executable, for build.crate_type = "cdylib". 'objc' links the
    // Objective-C runtime; [build] frameworks are passed on macOS.
    // 'extraFlags' go last (a cdylib's export list, a bundle's link flags).
    // Returns true on success.
    // Link flags for the CPU BLAS std's sgemm picks up (see [build] blas).
    std::vector<std::string> blasLinkFlags() const;
    bool linkFinal(const std::vector<std::string>& objFiles,
                    const std::vector<std::string>& archives,
                    const std::string& output,
                    bool useCxxDriver = false,
                    bool shared = false,
                    bool objc = false,
                    const std::vector<std::string>& extraFlags = {}) const;

    // Fork + exec a command, wait for it, return exit code.
    // 'suppressOutput': when true and 'verbose' is false, discard the
    // child's stdout/stderr instead of letting it print — see the call
    // site comment in runCmd's definition for why this exists.
    // 'stdoutPath': write the child's stdout to that file instead.
    static int runCmd(const std::vector<std::string>& argv, bool verbose,
                       bool suppressOutput = false, const std::string& stdoutPath = "");
    // Run a command, returning its stdout in 'out'; false if it failed.
    static bool captureCmd(const std::vector<std::string>& argv, std::string& out,
                           bool verbose = false);

    // An LLVM tool (llvm-nm, llvm-rc, ...): beside the LLVM clang findClang()
    // picks, beside the clang on PATH, else the bare name.
    std::string findLlvmTool(const std::string& name) const;

    // Windows cdylib: write build/<output>.def exporting the external
    // functions defined in the package's own objects (not std's or a
    // dependency's). Returns the .def path, "" if none, or "!" on error.
    std::string writeExportsDef(const std::vector<std::string>& objFiles) const;

    // Build a 'build = "sources"/"cmake"' dependency into l.outDir, unless
    // it is up to date. Returns true on success.
    bool buildNativeDep(const Dependency& dep, const DepLayout& l) const;

    // What the dependencies contribute beyond their libraries: whether one
    // needs the C++ link driver, and their run-time shared libraries.
    bool depsNeedCxx() const;
    std::vector<std::string> depRuntimeLibs() const;
    // Windows: copy the dependencies' DLLs into 'dir' (beside an exe or
    // DLL just linked there), the way an rpath finds them elsewhere.
    bool copyRuntimeLibs(const std::string& dir) const;

    // ── bundle ───────────────────────────────────────────────────────────────
    // Link flags the executable needs for its bundle (see BuildOptions::bundle).
    bool bundleLinkFlags(std::vector<std::string>& out) const;
    // Windows: write the .rc (icon, version info, application manifest)
    // under build/bundle/ and compile it to a .res; "" on failure.
    std::string windowsResource() const;
    bool bundleMacOS() const;
    bool bundleWindows() const;
    bool bundleLinux() const;

    // A dependency's directory: <root>/deps/<name> for a git URL, 'path'
    // (against the project root) for a local one, "" for a lib-only entry.
    std::string depDir(const Dependency& dep) const;

    // Every dependency's include dirs, and its link inputs (a source
    // package's built archive, a prebuilt one's library files).
    void depIncludes(std::vector<std::string>& safecIncs,
                     std::vector<std::string>& foreignIncs) const;
    std::vector<std::string> depLinkInputs() const;

    // Clone a git repository.  Returns true on success.
    bool gitClone(const std::string& url, const std::string& dest) const;

    // ── Reproducible builds ───────────────────────────────────────────────────

    // Write Package.lock after a successful build.
    void writeLock(const std::vector<std::string>& srcFiles) const;
};

} // namespace safeguard
