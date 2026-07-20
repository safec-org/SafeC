#include "Builder.h"
#include "Lock.h"
#include "ScxTranspiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cctype>
#include <fcntl.h>
#ifdef _WIN32
#include <process.h>   // _spawnvp
#include <io.h>        // _dup, _dup2, _open, _close (stdout/stderr redirection)
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace safeguard {

Builder::Builder(std::string projectRoot, Manifest manifest, BuildOptions opts)
    : root_(std::move(projectRoot))
    , manifest_(std::move(manifest))
    , opts_(std::move(opts))
{}

// ── safec / std discovery ─────────────────────────────────────────────────────

std::string Builder::findSafec() const {
    const char* home = std::getenv("SAFEC_HOME");
    if (home) {
        // 1. $SAFEC_HOME/bin/safec[.exe] — the installed layout install.sh/
        //    install.ps1 produce (SAFEC_HOME = install prefix, e.g. ~/safec).
#ifdef _WIN32
        fs::path installed = fs::path(home) / "bin" / "safec.exe";
#else
        fs::path installed = fs::path(home) / "bin" / "safec";
#endif
        if (fs::exists(installed)) return installed.string();
        // 2. $SAFEC_HOME/compiler/build/safec[.exe] — a from-source checkout
        //    built directly via CMake (SAFEC_HOME = SafeC repo root).
#ifdef _WIN32
        fs::path c = fs::path(home) / "compiler" / "build" / "safec.exe";
#else
        fs::path c = fs::path(home) / "compiler" / "build" / "safec";
#endif
        if (fs::exists(c)) return c.string();
    }
    // 3. PATH fallback
    return "safec";
}

std::string Builder::findClang(bool cxx) const {
    const char* envVar = cxx ? "SAFEC_CLANGXX" : "SAFEC_CLANG";
    if (const char* e = std::getenv(envVar); e && *e) return e;

    const char* name = cxx ? "clang++" : "clang";
    static const char* kPrefixes[] = {
        "/opt/homebrew/opt/llvm/bin",  // Homebrew, Apple Silicon
        "/usr/local/opt/llvm/bin",     // Homebrew, Intel
    };
    for (auto* prefix : kPrefixes) {
        fs::path c = fs::path(prefix) / name;
        if (fs::exists(c)) return c.string();
    }
    return name; // PATH fallback — previous behavior
}

std::string Builder::findSystemClang(bool cxx) const {
    const char* envVar = cxx ? "SAFEC_CLANGXX" : "SAFEC_CLANG";
    if (const char* e = std::getenv(envVar); e && *e) return e;
    return cxx ? "clang++" : "clang";
}

std::string Builder::findStdDir() const {
    // 1. $SAFEC_HOME/std
    const char* home = std::getenv("SAFEC_HOME");
    if (home) {
        fs::path d = fs::path(home) / "std";
        if (fs::exists(d)) return d.string();
    }
    // 2. Sibling of the safec binary
    std::string safecBin = findSafec();
    if (safecBin != "safec") {
        fs::path d = fs::path(safecBin).parent_path().parent_path().parent_path() / "std";
        if (fs::exists(d)) return d.string();
    }
    return "";
}

// ── source discovery ─────────────────────────────────────────────────────────

std::vector<std::string> Builder::collectSources(const std::string& srcDir) const {
    std::vector<std::string> srcs;
    if (!fs::exists(srcDir)) return srcs;
    for (auto& e : fs::recursive_directory_iterator(srcDir)) {
        if (e.is_regular_file() && e.path().extension() == ".sc")
            srcs.push_back(e.path().string());
    }
    std::sort(srcs.begin(), srcs.end());
    return srcs;
}

SrcLang Builder::langOf(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    if (ext == ".c")   return SrcLang::C;
    if (ext == ".cc" || ext == ".cpp" || ext == ".cxx") return SrcLang::Cpp;
    // .scx (scx markup templating) transpiles to plain SafeC before it ever
    // reaches safec (see materializeScx) — from compileSrc's point of view
    // it's just SafeC source, same as .sc.
    return SrcLang::SafeC;
}

std::vector<std::string> Builder::collectAllSources(
    const std::string& srcDir, const std::vector<std::string>& excludeSubdirNames) const {
    std::vector<std::string> srcs;
    if (!fs::exists(srcDir)) return srcs;
    static const char* kExts[] = { ".sc", ".scx", ".c", ".cc", ".cpp", ".cxx" };
    for (auto& e : fs::recursive_directory_iterator(srcDir)) {
        if (!e.is_regular_file()) continue;
        if (!excludeSubdirNames.empty()) {
            bool excluded = false;
            fs::path p = e.path();
            for (const auto& comp : p) {
                std::string name = comp.string();
                if (std::find(excludeSubdirNames.begin(), excludeSubdirNames.end(), name)
                    != excludeSubdirNames.end()) {
                    excluded = true;
                    break;
                }
            }
            if (excluded) continue;
        }
        std::string ext = e.path().extension().string();
        for (auto* k : kExts) {
            if (ext == k) { srcs.push_back(e.path().string()); break; }
        }
    }
    std::sort(srcs.begin(), srcs.end());
    return srcs;
}

// ── [features] ───────────────────────────────────────────────────────────────

std::vector<std::string> Builder::enabledFeatures() const {
    return resolveFeatures(manifest_, opts_.features, opts_.noDefaultFeatures);
}

std::vector<std::string> Builder::featureDefines() const {
    std::vector<std::string> defs;
    for (auto& name : enabledFeatures()) {
        std::string upper;
        upper.reserve(name.size());
        for (char c : name)
            upper += (char)std::toupper((unsigned char)(c == '-' ? '_' : c));
        defs.push_back("-DSAFEC_FEATURE_" + upper + "=1");
    }
    return defs;
}

// ── .scx (HTML-templating source) ───────────────────────────────────────────

std::string Builder::materializeScx(const std::string& src,
                                     const std::string& buildDir) const {
    if (fs::path(src).extension() != ".scx") return src;

    std::ifstream f(src);
    if (!f) {
        std::cerr << "safeguard: cannot open " << src << "\n";
        return "";
    }
    std::string source((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    std::string transpiled;
    try {
        transpiled = safeguard::transpileScx(source, src);
    } catch (std::exception& e) {
        std::cerr << "safeguard: " << e.what() << "\n";
        return "";
    }

    fs::path scxOutDir = fs::path(buildDir) / "scx";
    fs::create_directories(scxOutDir);
    std::string stem;
    try {
        fs::path rel = fs::relative(fs::path(src), root_);
        std::string r = rel.string();
        for (char& c : r)
            if (c == '/' || c == '\\' || c == '.') c = '_';
        stem = r;
    } catch (...) {
        stem = fs::path(src).stem().string();
    }
    fs::path outPath = scxOutDir / (stem + ".sc");
    std::ofstream out(outPath);
    if (!out) {
        std::cerr << "safeguard: cannot write " << outPath << "\n";
        return "";
    }
    out << transpiled;
    out.close();
    return outPath.string();
}

// ── command runner ────────────────────────────────────────────────────────────

int Builder::runCmd(const std::vector<std::string>& argv, bool verbose,
                     bool suppressOutput) {
    if (argv.empty()) return -1;
    if (verbose) {
        for (size_t i = 0; i < argv.size(); ++i)
            std::cout << (i ? " " : "") << argv[i];
        std::cout << "\n";
    }
    // Only used by buildLib's tolerateArchMismatch path: a stdlib file that
    // fails to assemble for the current target (e.g. std/hal/riscv.sc's CSR
    // asm on a non-RISC-V host) is an *expected*, already-summarized-by-the-
    // caller outcome ("safeguard: warning: skipping ..."), not a real
    // failure — letting the child's own multi-line compiler error through
    // on every one of those (there are several across the stdlib) buried
    // genuine failures in noise. Verbose mode still shows everything, since
    // its whole contract is "show me every command and its output."
    bool quiet = suppressOutput && !verbose;
#ifdef _WIN32
    // _spawnvp is the MSVC CRT's argv-based process launcher — like
    // execvp, it searches PATH and takes an argv array directly (no shell
    // involved), so arguments containing spaces/quotes don't need manual
    // shell-quoting the way a system()-based approach would require.
    // _P_WAIT blocks until the child exits and yields its exit code
    // directly, playing the same role fork()+waitpid()+WEXITSTATUS() does
    // on POSIX below.
    int savedOut = -1, savedErr = -1, nullFd = -1;
    if (quiet) {
        fflush(stdout); fflush(stderr);
        savedOut = _dup(1);
        savedErr = _dup(2);
        nullFd = _open("NUL", _O_WRONLY);
        if (nullFd != -1) { _dup2(nullFd, 1); _dup2(nullFd, 2); }
    }
    std::vector<const char*> args;
    for (auto& s : argv) args.push_back(s.c_str());
    args.push_back(nullptr);
    intptr_t rc = _spawnvp(_P_WAIT, args[0], args.data());
    int spawnErrno = errno;
    if (quiet) {
        fflush(stdout); fflush(stderr);
        if (savedOut != -1) { _dup2(savedOut, 1); _close(savedOut); }
        if (savedErr != -1) { _dup2(savedErr, 2); _close(savedErr); }
        if (nullFd != -1) _close(nullFd);
    }
    if (rc == -1) {
        if (!quiet) {
            std::cerr << "safeguard: spawn '" << argv[0] << "' failed: "
                      << strerror(spawnErrno) << "\n";
        }
        return -1;
    }
    return (int)rc;
#else
    int savedOut = -1, savedErr = -1, nullFd = -1;
    if (quiet) {
        fflush(stdout); fflush(stderr);
        savedOut = dup(1);
        savedErr = dup(2);
        nullFd = open("/dev/null", O_WRONLY);
        if (nullFd != -1) { dup2(nullFd, 1); dup2(nullFd, 2); }
    }
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "safeguard: fork failed: " << strerror(errno) << "\n";
        if (quiet) {
            if (savedOut != -1) { dup2(savedOut, 1); close(savedOut); }
            if (savedErr != -1) { dup2(savedErr, 2); close(savedErr); }
            if (nullFd != -1) close(nullFd);
        }
        return -1;
    }
    if (pid == 0) {
        // Child inherits the redirected (or original) fds from the fork —
        // no per-child redirect needed here.
        std::vector<char*> args;
        for (auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
        args.push_back(nullptr);
        execvp(args[0], args.data());
        if (!quiet) {
            std::cerr << "safeguard: exec '" << argv[0] << "' failed: "
                      << strerror(errno) << "\n";
        }
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (quiet) {
        fflush(stdout); fflush(stderr);
        if (savedOut != -1) { dup2(savedOut, 1); close(savedOut); }
        if (savedErr != -1) { dup2(savedErr, 2); close(savedErr); }
        if (nullFd != -1) close(nullFd);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// ── compile one .sc → .ll ─────────────────────────────────────────────────────

std::string Builder::compileSrc(const std::string& safecBin,
                                  const std::string& srcPath,
                                  const std::string& buildDir,
                                  const std::vector<std::string>& includeDirs,
                                  bool compatPreprocessor,
                                  bool suppressOutput,
                                  const std::vector<std::string>& defines) const {
    // Derive a flat output name to avoid collisions across subdirectories
    fs::path src(srcPath);
    // Use relative-path-derived stem: e.g. src/foo/bar.sc → foo_bar.ll
    std::string stem;
    try {
        fs::path rel = fs::relative(src, root_);
        std::string r = rel.string();
        // Replace path separators and dots with underscores
        for (char& c : r)
            if (c == '/' || c == '\\' || c == '.') c = '_';
        stem = r;
    } catch (...) {
        stem = src.stem().string();
    }
    fs::path llFile = fs::path(buildDir) / (stem + ".ll");

    std::vector<std::string> cmd = { safecBin };
    for (auto& inc : includeDirs) {
        cmd.push_back("-I");
        cmd.push_back(inc);
    }
    for (auto& d : defines) cmd.push_back(d);
    cmd.push_back(srcPath);
    cmd.push_back("--emit-llvm");
    if (compatPreprocessor) cmd.push_back("--compat-preprocessor");
    cmd.push_back("-o");
    cmd.push_back(llFile.string());
    if (!opts_.extraFlags.empty())
        cmd.push_back(opts_.extraFlags);

    int rc = runCmd(cmd, opts_.verbose, suppressOutput);
    if (rc != 0) {
        if (!suppressOutput)
            std::cerr << "safeguard: compilation failed for " << srcPath << "\n";
        return "";
    }
    return llFile.string();
}

// ── compile .ll → .o ─────────────────────────────────────────────────────────

std::string Builder::llToObj(const std::string& llFile,
                               const std::string& buildDir,
                               bool suppressOutput) const {
    fs::path obj = fs::path(buildDir) / (fs::path(llFile).stem().string() + ".o");
    std::vector<std::string> cmd = { findClang(false), "-c", llFile, "-o", obj.string() };
    if (opts_.release) cmd.push_back("-O2");
    int rc = runCmd(cmd, opts_.verbose, suppressOutput);
    if (rc != 0) {
        if (!suppressOutput)
            std::cerr << "safeguard: clang -c failed for " << llFile << "\n";
        return "";
    }
    return obj.string();
}

// ── compile .c/.cpp → .o directly ────────────────────────────────────────────

std::string Builder::compileForeign(const std::string& srcPath,
                                     const std::string& buildDir,
                                     const std::vector<std::string>& includeDirs,
                                     SrcLang lang,
                                     bool suppressOutput) const {
    // Flatten to a unique object name the same way compileSrc() does for
    // .sc files, so a C file and a SafeC file with the same basename in
    // different subdirectories don't clobber each other's .o.
    fs::path src(srcPath);
    std::string stem;
    try {
        fs::path rel = fs::relative(src, root_);
        std::string r = rel.string();
        for (char& c : r)
            if (c == '/' || c == '\\' || c == '.') c = '_';
        stem = r;
    } catch (...) {
        stem = src.stem().string();
    }
    fs::path obj = fs::path(buildDir) / (stem + ".o");

    std::string compiler = findSystemClang(lang == SrcLang::Cpp);
    std::vector<std::string> cmd = { compiler, "-c", srcPath, "-o", obj.string() };
    for (auto& inc : includeDirs) {
        cmd.push_back("-I");
        cmd.push_back(inc);
    }
    if (opts_.release) cmd.push_back("-O2");
    for (auto& f : manifest_.build.cflags) cmd.push_back(f);

    int rc = runCmd(cmd, opts_.verbose, suppressOutput);
    if (rc != 0) {
        if (!suppressOutput)
            std::cerr << "safeguard: " << compiler << " -c failed for " << srcPath << "\n";
        return "";
    }
    return obj.string();
}

// ── pack .o files → static archive ───────────────────────────────────────────

bool Builder::packArchive(const std::vector<std::string>& objFiles,
                            const std::string& archivePath) const {
    // ar rcs <archive> <obj1> <obj2> ...
    std::vector<std::string> cmd = { "ar", "rcs", archivePath };
    for (auto& o : objFiles) cmd.push_back(o);
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: ar failed for " << archivePath << "\n";
        return false;
    }
    return true;
}

// ── final link ────────────────────────────────────────────────────────────────

bool Builder::linkFinal(const std::vector<std::string>& objFiles,
                          const std::vector<std::string>& archives,
                          const std::string& output,
                          bool useCxxDriver,
                          bool shared) const {
    std::vector<std::string> cmd = { findSystemClang(useCxxDriver) };
    if (opts_.release) cmd.push_back("-O2");
    if (!manifest_.build.lto.empty())
        cmd.push_back(manifest_.build.lto == "full" ? "-flto" : "-flto=thin");
    if (shared) cmd.push_back("-shared");
    for (auto& o : objFiles) cmd.push_back(o);
    // Archives: pass with -force_load on macOS so unused symbols are kept,
    // or use the portable -Wl,--whole-archive on Linux.
    // For simplicity, just list the .a files directly (works for most cases).
    for (auto& a : archives) cmd.push_back(a);
#ifndef _WIN32
    // math functions live in a separate libm on macOS/Linux; on Windows
    // they're part of the standard CRT that's always linked, and there is
    // no 'm.lib' for clang's MSVC-style link.exe driver to find — passing
    // -lm there fails the link with LNK1181 ("cannot open input file
    // 'm.lib'") instead of being a harmless no-op.
    cmd.push_back("-lm"); // math library for std/math.sc
#endif
    for (auto& d : manifest_.build.libDirs) cmd.push_back("-L" + d);
    for (auto& l : manifest_.build.libs)    cmd.push_back("-l" + l);
    // Extra link flags from manifest
    for (auto& f : manifest_.build.cflags) cmd.push_back(f);
    cmd.push_back("-o");
    cmd.push_back(output);
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: link failed\n";
        return false;
    }
    return true;
}

// ── build a library from a source directory ───────────────────────────────────

bool Builder::buildLib(const std::string& srcDir,
                        const std::string& libOut,
                        const std::vector<std::string>& includeDirs,
                        bool compatPreprocessor,
                        bool tolerateArchMismatch,
                        const std::vector<std::string>& foreignIncludeDirs,
                        const std::vector<std::string>& excludeSubdirNames) {
    std::string safecBin = findSafec();
    std::string buildDir = fs::path(libOut).parent_path().string();
    fs::create_directories(buildDir);

    auto srcs = collectAllSources(srcDir, excludeSubdirNames);
    if (srcs.empty()) return true; // nothing to compile

    std::vector<std::string> objs;
    for (auto& src : srcs) {
        SrcLang lang = langOf(src);
        if (lang == SrcLang::SafeC) {
            std::string realSrc = materializeScx(src, buildDir);
            if (realSrc.empty()) {
                if (tolerateArchMismatch) continue; // scx error already printed
                return false;
            }
            // safec itself can also reject a file purely for being
            // architecture-inapplicable (e.g. std::simd's ARM-only DSP
            // builtins raising a Sema/CodeGen error on a non-ARM target,
            // the same intent as the assembler rejecting riscv.sc's CSR
            // instructions on non-RISC-V — just enforced earlier in the
            // pipeline). Both are tolerated identically here: either way,
            // the practical effect of skipping is the same (the file's
            // functions simply aren't available for this target; anything
            // that actually calls them gets a normal, clear "undefined
            // symbol" at link time instead of a silent wrong answer).
            std::string ll = compileSrc(safecBin, realSrc, buildDir, includeDirs, compatPreprocessor,
                                         /*suppressOutput=*/tolerateArchMismatch);
            if (ll.empty()) {
                if (tolerateArchMismatch) {
                    std::cerr << "safeguard: warning: skipping '" << src
                              << "' — does not compile for the current target\n";
                    continue;
                }
                return false;
            }
            std::string obj = llToObj(ll, buildDir, /*suppressOutput=*/tolerateArchMismatch);
            if (obj.empty()) {
                if (tolerateArchMismatch) {
                    std::cerr << "safeguard: warning: skipping '" << src
                              << "' — does not assemble for the current target "
                                 "(likely architecture-specific inline asm)\n";
                    continue;
                }
                return false;
            }
            objs.push_back(obj);
        } else {
            std::string obj = compileForeign(src, buildDir, foreignIncludeDirs, lang,
                                              /*suppressOutput=*/tolerateArchMismatch);
            if (obj.empty()) {
                if (tolerateArchMismatch) {
                    std::cerr << "safeguard: warning: skipping '" << src
                              << "' — does not compile for the current target\n";
                    continue;
                }
                return false;
            }
            objs.push_back(obj);
        }
    }
    if (objs.empty()) return true; // everything tolerated/skipped
    return packArchive(objs, libOut);
}

// ── build (or reuse) the SafeC standard library ───────────────────────────────

std::string Builder::ensureStdLib() {
    std::string stdDir = findStdDir();
    if (stdDir.empty()) {
        std::cerr << "safeguard: warning: cannot find SafeC std/ directory; "
                     "set SAFEC_HOME to your SafeC install prefix or repository root\n";
        return "";
    }

    fs::path libPath = fs::path(root_) / "build" / "deps" / "libsafec_std.a";

    // Only rebuild if the library is missing.
    if (fs::exists(libPath)) return libPath.string();

    if (opts_.verbose)
        std::cout << "safeguard: building std library from " << stdDir << "\n";

    // The std .sc files include their own headers; pass the std dir's parent
    // so that #include "std/io.h" resolves, and the std dir itself so that
    // #include "io.h" (relative includes inside .sc files) resolves.
    fs::path stdParent = fs::path(stdDir).parent_path();
    std::vector<std::string> incs = { stdDir, stdParent.string() };

    // 'wasm': std/wasm/*.sc's malloc/free/etc. are wasm32-freestanding-only
    // — see collectAllSources's doc comment and std/wasm/wasm_rt.sc's own
    // header comment for the full story (a real, CI-confirmed Windows
    // LNK2005 symbol collision against libucrt otherwise).
    if (!buildLib(stdDir, libPath.string(), incs, /*compatPreprocessor=*/true,
                  /*tolerateArchMismatch=*/true, /*foreignIncludeDirs=*/{},
                  /*excludeSubdirNames=*/{"wasm"}))
        return "";
    return libPath.string();
}

// ── git clone ─────────────────────────────────────────────────────────────────

bool Builder::gitClone(const std::string& url, const std::string& dest) const {
    if (fs::exists(dest)) {
        if (opts_.verbose)
            std::cout << "safeguard: " << dest << " already exists, skipping clone\n";
        return true;
    }
    std::vector<std::string> cmd = { "git", "clone", "--depth=1", url, dest };
    int rc = runCmd(cmd, true); // always print git clone
    return rc == 0;
}

// ── fetch and build all dependencies ─────────────────────────────────────────

bool Builder::fetchAndBuildDeps() {
    if (manifest_.dependencies.empty()) return true;

    fs::path depsDir   = fs::path(root_) / "deps";
    fs::path depLibDir = fs::path(root_) / "build" / "deps";
    fs::create_directories(depsDir);
    fs::create_directories(depLibDir);

    std::string safecBin = findSafec();

    for (auto& dep : manifest_.dependencies) {
        if (dep.name.empty()) continue;

        std::string destDir;
        if (!dep.path.empty()) {
            // Local path dependency
            destDir = dep.path;
            if (!fs::exists(destDir)) {
                std::cerr << "safeguard: local dependency path not found: "
                          << destDir << "\n";
                return false;
            }
        } else {
            // Remote dependency — dep.version holds the GitHub URL
            destDir = (depsDir / dep.name).string();
            if (!dep.version.empty()) {
                if (!gitClone(dep.version, destDir)) {
                    std::cerr << "safeguard: failed to fetch dependency '"
                              << dep.name << "' from " << dep.version << "\n";
                    return false;
                }
            } else {
                std::cerr << "safeguard: dependency '" << dep.name
                          << "' has no version/url field\n";
                return false;
            }
        }

        // Build the dependency
        fs::path libOut = depLibDir / ("lib" + dep.name + ".a");
        if (!fs::exists(libOut)) {
            // The dep's src/ directory
            fs::path depSrc = fs::path(destDir) / "src";
            if (!fs::exists(depSrc)) {
                std::cerr << "safeguard: dependency '" << dep.name
                          << "' has no src/ directory at " << destDir << "\n";
                return false;
            }
            // Include its own include/ or src/ directory
            std::vector<std::string> incs;
            fs::path depInc = fs::path(destDir) / "include";
            if (fs::exists(depInc)) incs.push_back(depInc.string());
            incs.push_back(depSrc.string());
            // Also add std dir (SafeC-side only — see buildLib's
            // foreignIncludeDirs doc comment for why this must not reach
            // a real C/C++ compiler's search path).
            std::string stdDir = findStdDir();
            if (!stdDir.empty()) {
                incs.push_back(stdDir);
                incs.push_back(fs::path(stdDir).parent_path().string());
            }
            std::vector<std::string> foreignIncs;
            if (fs::exists(depInc)) foreignIncs.push_back(depInc.string());

            if (!buildLib(depSrc.string(), libOut.string(), incs,
                          /*compatPreprocessor=*/false, /*tolerateArchMismatch=*/false,
                          foreignIncs)) {
                std::cerr << "safeguard: failed to build dependency '"
                          << dep.name << "'\n";
                return false;
            }
        }
        std::cout << "safeguard: dependency '" << dep.name << "' ready\n";
    }
    return true;
}

// ── public: build ─────────────────────────────────────────────────────────────

std::string Builder::outputPath() const {
    const std::string& type = manifest_.build.crateType;
    if (type == "staticlib") {
        return (fs::path(root_) / "build" /
                ("lib" + manifest_.build.output + ".a")).string();
    }
    if (type == "cdylib") {
#if defined(__APPLE__)
        const char* ext = ".dylib";
#elif defined(_WIN32)
        const char* ext = ".dll";
#else
        const char* ext = ".so";
#endif
        return (fs::path(root_) / "build" /
                ("lib" + manifest_.build.output + ext)).string();
    }
    return (fs::path(root_) / "build" / manifest_.build.output).string();
}

bool Builder::build() {
    std::string safecBin = findSafec();
    std::string buildDir = (fs::path(root_) / "build").string();
    fs::create_directories(buildDir);

    // 0. Verify lock file (aborts if dep SHAs mismatch).
    if (!checkLock()) return false;

    // 1. Fetch + build dependencies
    if (!fetchAndBuildDeps()) return false;

    // 2. Build (or reuse) std library. A missing std/ directory is only a
    // warning (a project using nothing but extern declarations can still
    // link without it), but if std/ was found and building it failed, that's
    // a real error — silently linking without it just produces a wall of
    // confusing "undefined symbol" errors instead.
    std::string stdDir  = findStdDir();
    std::string stdLib  = ensureStdLib();
    if (stdLib.empty() && !stdDir.empty()) {
        std::cerr << "safeguard: failed to build the standard library "
                     "(see compiler errors above) — aborting build\n";
        return false;
    }

    // 3. Determine include paths for user sources
    std::vector<std::string> userIncs;
    if (!stdDir.empty()) {
        userIncs.push_back(stdDir);
        userIncs.push_back(fs::path(stdDir).parent_path().string());
    }
    // Also add include/ from each dependency
    for (auto& dep : manifest_.dependencies) {
        std::string destDir = dep.path.empty()
            ? (fs::path(root_) / "deps" / dep.name).string()
            : dep.path;
        fs::path depInc = fs::path(destDir) / "include";
        if (fs::exists(depInc)) userIncs.push_back(depInc.string());
        fs::path depSrc = fs::path(destDir) / "src";
        if (fs::exists(depSrc)) userIncs.push_back(depSrc.string());
    }

    // 3b. Separate, narrower include list for genuine .c/.cpp files.
    // SafeC's own std/ directory must NOT be on a real C/C++ compiler's
    // search path: std/stdint.h etc. use #define (not typedef) for their
    // typedefs — fine for SafeC's own preprocessing model, but a real
    // #include <cstdint>/<vector>/... pulling those macro names in scope
    // corrupts libc++'s own declarations of the same names (observed: a
    // plain '#include <vector>' file failed to compile with cryptic
    // "expected unqualified-id" errors deep inside libc++ headers once
    // std/stdint.h's '#define uint8_t unsigned char' etc. were in scope).
    // A project's own include/ and each dependency's include/ are still
    // fair game — those are meant to be consumed by any language.
    std::vector<std::string> foreignIncs;
    fs::path projInc = fs::path(root_) / "include";
    if (fs::exists(projInc)) foreignIncs.push_back(projInc.string());
    for (auto& dep : manifest_.dependencies) {
        std::string destDir = dep.path.empty()
            ? (fs::path(root_) / "deps" / dep.name).string()
            : dep.path;
        fs::path depInc = fs::path(destDir) / "include";
        if (fs::exists(depInc)) foreignIncs.push_back(depInc.string());
    }

    // 4. Compile user sources. Mixed SafeC/C/C++ — each file is dispatched
    // by extension and compiled to its own .o (SafeC via safec --emit-llvm
    // + clang -c; C/C++ straight to .o via clang/clang++), never merged
    // into a shared translation unit, so per-file recompilation stays
    // granular the same way it already was for pure-SafeC projects.
    fs::path srcDir = fs::path(root_) / "src";
    std::vector<std::string> srcs;
    if (!manifest_.build.srcs.empty()) {
        for (auto& s : manifest_.build.srcs) {
            fs::path p(s);
            srcs.push_back(p.is_absolute() ? s : (fs::path(root_) / p).string());
        }
    } else {
        srcs = collectAllSources(srcDir.string());
    }
    if (srcs.empty()) {
        std::cerr << "safeguard: no .sc/.c/.cpp source files found under src/\n";
        return false;
    }

    bool hasCpp = false;
    for (auto& src : srcs) if (langOf(src) == SrcLang::Cpp) hasCpp = true;

    std::vector<std::string> defs = featureDefines();
    if (opts_.verbose && !defs.empty()) {
        std::cout << "safeguard: enabled features:";
        for (auto& f : enabledFeatures()) std::cout << " " << f;
        std::cout << "\n";
    }

    if (opts_.emitLLVM) {
        // Only SafeC sources have an LLVM IR stage to stop at; C/C++ files
        // have no equivalent partial output, so skip them entirely here
        // rather than compiling straight to .o for a build that won't link.
        for (auto& src : srcs) {
            if (langOf(src) != SrcLang::SafeC) continue;
            std::string realSrc = materializeScx(src, buildDir);
            if (realSrc.empty()) return false;
            if (compileSrc(safecBin, realSrc, buildDir, userIncs, false, false, defs).empty())
                return false;
        }
        std::cout << "safeguard: LLVM IR written to " << buildDir << "/\n";
        return true;
    }

    // 5. Compile each source straight through to a .o (SafeC via .ll; C/C++
    // directly), collecting object files for the final link.
    std::vector<std::string> objFiles;
    for (auto& src : srcs) {
        SrcLang lang = langOf(src);
        if (lang == SrcLang::SafeC) {
            std::string realSrc = materializeScx(src, buildDir);
            if (realSrc.empty()) return false;
            std::string ll = compileSrc(safecBin, realSrc, buildDir, userIncs, false, false, defs);
            if (ll.empty()) return false;
            std::string obj = llToObj(ll, buildDir);
            if (obj.empty()) return false;
            objFiles.push_back(obj);
        } else {
            std::string obj = compileForeign(src, buildDir, foreignIncs, lang);
            if (obj.empty()) return false;
            objFiles.push_back(obj);
        }
    }

    // 6. Collect archive libraries
    std::vector<std::string> archives;
    if (!stdLib.empty()) archives.push_back(stdLib);
    fs::path depLibDir = fs::path(root_) / "build" / "deps";
    for (auto& dep : manifest_.dependencies) {
        fs::path libOut = depLibDir / ("lib" + dep.name + ".a");
        if (fs::exists(libOut)) archives.push_back(libOut.string());
    }

    // 7. Produce the final artifact. build.crate_type selects what "final"
    // means: a linked executable (the default, "bin"), a dynamic library
    // ("cdylib", linked with -shared), or a static archive ("staticlib",
    // packed with 'ar' — no linker/-l/-L/lto involved, since a .a is just
    // object files bundled for a *later* link to consume, not itself
    // linked). clang++ is used as the link driver whenever any C++ source
    // was compiled in, so the C++ runtime (libc++/libstdc++, exceptions,
    // RTTI) links in — plain clang can link the resulting object files
    // (SafeC's LLVM-emitted .o and C/C++'s clang-emitted .o are all just
    // ordinary object code) but won't pull in libc++ on its own.
    const std::string& crateType = manifest_.build.crateType;
    if (crateType == "staticlib") {
        std::vector<std::string> allObjs = objFiles;
        // Static libs bundle only this project's own objects — archives
        // (std/deps' .a files) are meant to be linked in later by whatever
        // consumes this .a, not re-flattened into it here.
        if (!packArchive(allObjs, outputPath())) return false;
    } else if (crateType == "cdylib") {
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp, /*shared=*/true))
            return false;
    } else {
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp)) return false;
    }

    // 8. Write / refresh Package.lock after a successful build.
    writeLock(srcs);
    return true;
}

// ── public: check ─────────────────────────────────────────────────────────────

bool Builder::check() {
    std::string safecBin = findSafec();
    fs::path checkDir = fs::path(root_) / "build" / "check";
    fs::create_directories(checkDir);

    // No ensureStdLib()/fetchAndBuildDeps() here on purpose — checking only
    // needs the std/ *headers* on the include path for declarations to
    // resolve, not the archived, linkable implementation, which is the
    // part of a full build that actually takes real time.
    std::string stdDir = findStdDir();
    std::vector<std::string> userIncs;
    if (!stdDir.empty()) {
        userIncs.push_back(stdDir);
        userIncs.push_back(fs::path(stdDir).parent_path().string());
    }
    std::vector<std::string> foreignIncs;
    fs::path projInc = fs::path(root_) / "include";
    if (fs::exists(projInc)) foreignIncs.push_back(projInc.string());
    for (auto& dep : manifest_.dependencies) {
        std::string destDir = dep.path.empty()
            ? (fs::path(root_) / "deps" / dep.name).string()
            : dep.path;
        fs::path depInc = fs::path(destDir) / "include";
        if (fs::exists(depInc)) { userIncs.push_back(depInc.string()); foreignIncs.push_back(depInc.string()); }
        fs::path depSrc = fs::path(destDir) / "src";
        if (fs::exists(depSrc)) userIncs.push_back(depSrc.string());
    }

    fs::path srcDir = fs::path(root_) / "src";
    std::vector<std::string> srcs;
    if (!manifest_.build.srcs.empty()) {
        for (auto& s : manifest_.build.srcs) {
            fs::path p(s);
            srcs.push_back(p.is_absolute() ? s : (fs::path(root_) / p).string());
        }
    } else {
        srcs = collectAllSources(srcDir.string());
    }
    if (srcs.empty()) {
        std::cerr << "safeguard: no .sc/.c/.cpp source files found under src/\n";
        return false;
    }

    std::vector<std::string> defs = featureDefines();

    bool ok = true;
    for (auto& src : srcs) {
        SrcLang lang = langOf(src);
        if (lang == SrcLang::SafeC) {
            // --emit-llvm runs the full front end (Preprocess/Lex/Parse/
            // Sema/ConstEval) and stops right after emitting IR — no
            // object-file assembly, no link.
            std::string realSrc = materializeScx(src, checkDir.string());
            if (realSrc.empty()) { ok = false; continue; }
            if (compileSrc(safecBin, realSrc, checkDir.string(), userIncs, false, false, defs).empty())
                ok = false;
        } else {
            std::vector<std::string> cmd = { findSystemClang(lang == SrcLang::Cpp),
                                              "-fsyntax-only", src };
            for (auto& inc : foreignIncs) { cmd.push_back("-I"); cmd.push_back(inc); }
            if (runCmd(cmd, opts_.verbose) != 0) {
                std::cerr << "safeguard: check failed for " << src << "\n";
                ok = false;
            }
        }
    }
    if (ok) std::cout << "safeguard: check passed (" << srcs.size() << " file(s))\n";
    return ok;
}

// ── public: test ──────────────────────────────────────────────────────────────

bool Builder::test() {
    fs::path testsDir = fs::path(root_) / "tests";
    if (!fs::exists(testsDir)) {
        std::cout << "safeguard: no tests/ directory — nothing to run\n";
        return true;
    }

    if (!fetchAndBuildDeps()) return false;
    std::string stdDir = findStdDir();
    std::string stdLib = ensureStdLib();
    if (stdLib.empty() && !stdDir.empty()) {
        std::cerr << "safeguard: failed to build the standard library "
                     "(see compiler errors above) — aborting test run\n";
        return false;
    }

    std::vector<std::string> userIncs;
    if (!stdDir.empty()) {
        userIncs.push_back(stdDir);
        userIncs.push_back(fs::path(stdDir).parent_path().string());
    }
    for (auto& dep : manifest_.dependencies) {
        std::string destDir = dep.path.empty()
            ? (fs::path(root_) / "deps" / dep.name).string()
            : dep.path;
        fs::path depInc = fs::path(destDir) / "include";
        if (fs::exists(depInc)) userIncs.push_back(depInc.string());
        fs::path depSrc = fs::path(destDir) / "src";
        if (fs::exists(depSrc)) userIncs.push_back(depSrc.string());
    }

    std::vector<std::string> archives;
    if (!stdLib.empty()) archives.push_back(stdLib);
    fs::path depLibDir = fs::path(root_) / "build" / "deps";
    for (auto& dep : manifest_.dependencies) {
        fs::path libOut = depLibDir / ("lib" + dep.name + ".a");
        if (fs::exists(libOut)) archives.push_back(libOut.string());
    }

    std::string safecBin = findSafec();
    fs::path buildDir = fs::path(root_) / "build" / "tests";
    fs::create_directories(buildDir);

    auto testSrcs = collectAllSources(testsDir.string());
    if (testSrcs.empty()) {
        std::cout << "safeguard: tests/ has no .sc/.c/.cpp files\n";
        return true;
    }

    std::vector<std::string> defs = featureDefines();

    int passed = 0, failed = 0;
    for (auto& src : testSrcs) {
        std::string name = fs::path(src).stem().string();
        SrcLang lang = langOf(src);
        std::string obj;
        if (lang == SrcLang::SafeC) {
            std::string realSrc = materializeScx(src, buildDir.string());
            // compatPreprocessor=true: tests routinely pull in
            // <std/test/test.h>, whose ASSERT_* convenience macros are
            // function-like — parsing that #define requires compat mode
            // regardless of whether a given test file invokes them.
            std::string ll = realSrc.empty() ? "" :
                compileSrc(safecBin, realSrc, buildDir.string(), userIncs, true, false, defs);
            if (!ll.empty()) obj = llToObj(ll, buildDir.string());
        } else {
            obj = compileForeign(src, buildDir.string(), userIncs, lang);
        }
        if (obj.empty()) {
            std::cout << "test " << name << " ... FAILED (build error)\n";
            ++failed;
            continue;
        }
        std::string testBin = (buildDir / name).string();
        if (!linkFinal({obj}, archives, testBin, lang == SrcLang::Cpp)) {
            std::cout << "test " << name << " ... FAILED (link error)\n";
            ++failed;
            continue;
        }
        int rc = runCmd({testBin}, opts_.verbose);
        if (rc == 0) {
            std::cout << "test " << name << " ... ok\n";
            ++passed;
        } else {
            std::cout << "test " << name << " ... FAILED (exit " << rc << ")\n";
            ++failed;
        }
    }
    std::cout << "safeguard: " << passed << " passed, " << failed << " failed\n";
    return failed == 0;
}

// ── public: run ───────────────────────────────────────────────────────────────

int Builder::run(const std::vector<std::string>& args) {
    if (manifest_.build.crateType != "bin") {
        std::cerr << "safeguard: 'run' requires build.crate_type = \"bin\" "
                     "(this project builds a " << manifest_.build.crateType
                  << " — there is no executable entry point to run)\n";
        return -1;
    }
    if (!build()) return -1;
    std::string out = outputPath();
    std::vector<std::string> cmd = { out };
    for (auto& a : args) cmd.push_back(a);
    return runCmd(cmd, opts_.verbose);
}

// ── public: clean ─────────────────────────────────────────────────────────────

void Builder::clean() {
    fs::path buildDir = fs::path(root_) / "build";
    if (fs::exists(buildDir)) {
        auto n = fs::remove_all(buildDir);
        std::cout << "safeguard: removed " << n << " file(s) in build/\n";
    } else {
        std::cout << "safeguard: build/ does not exist, nothing to clean\n";
    }
    fs::path depsDir = fs::path(root_) / "deps";
    if (fs::exists(depsDir)) {
        auto n = fs::remove_all(depsDir);
        std::cout << "safeguard: removed " << n << " file(s) in deps/\n";
    }
}

// ── Reproducible builds ───────────────────────────────────────────────────────

void Builder::writeLock(const std::vector<std::string>& srcFiles) const {
    LockFile lf;

    // Hash the safec binary.
    std::string safecBin = findSafec();
    if (safecBin != "safec") {
        lf.safec_hash = hashFile(safecBin);
    }

    // Record pinned git SHAs for all dependencies.
    fs::path depsDir = fs::path(root_) / "deps";
    for (auto& dep : manifest_.dependencies) {
        LockedDep ld;
        ld.name = dep.name;
        ld.url  = dep.path.empty() ? dep.version : dep.path;
        std::string depDir = dep.path.empty()
            ? (depsDir / dep.name).string()
            : dep.path;
        if (fs::exists(depDir)) {
            ld.git_sha = gitRevParse(depDir);
        }
        lf.deps.push_back(ld);
    }

    // Hash every source file.
    for (auto& src : srcFiles) {
        std::string h = hashFile(src);
        if (!h.empty()) {
            std::string rel;
            try {
                rel = fs::relative(src, root_).string();
            } catch (...) {
                rel = src;
            }
            lf.sources[rel] = h;
        }
    }

    std::string lockPath = (fs::path(root_) / "Package.lock").string();
    try {
        lockWrite(lf, lockPath);
        if (opts_.verbose)
            std::cout << "safeguard: Package.lock written\n";
    } catch (std::exception& e) {
        std::cerr << "safeguard: warning: could not write lock: " << e.what() << "\n";
    }
}

bool Builder::checkLock() const {
    std::string lockPath = (fs::path(root_) / "Package.lock").string();
    LockFile lf;
    if (!lockRead(lf, lockPath)) {
        // No lock file yet — nothing to check.
        return true;
    }

    bool ok = true;

    // Check safec binary hash.
    std::string safecBin = findSafec();
    if (!lf.safec_hash.empty() && safecBin != "safec") {
        std::string cur = hashFile(safecBin);
        if (!cur.empty() && cur != lf.safec_hash) {
            std::cerr << "safeguard: warning: safec binary has changed since "
                         "Package.lock was written (lock=" << lf.safec_hash
                      << " current=" << cur << ")\n";
            // Binary change: warn only, don't fail build.
        }
    }

    // Check git SHAs for dependencies.
    fs::path depsDir = fs::path(root_) / "deps";
    for (auto& locked : lf.deps) {
        std::string depDir;
        // Find matching dep in manifest.
        for (auto& dep : manifest_.dependencies) {
            if (dep.name == locked.name) {
                depDir = dep.path.empty()
                    ? (depsDir / dep.name).string()
                    : dep.path;
                break;
            }
        }
        if (depDir.empty() || !fs::exists(depDir)) continue;
        std::string cur = gitRevParse(depDir);
        if (!cur.empty() && !locked.git_sha.empty() && cur != locked.git_sha) {
            std::cerr << "safeguard: error: dependency '" << locked.name
                      << "' git SHA mismatch (locked=" << locked.git_sha
                      << " current=" << cur << ")\n"
                      << "  Run 'safeguard fetch' to update, then rebuild.\n";
            ok = false;
        }
    }

    return ok;
}

} // namespace safeguard
