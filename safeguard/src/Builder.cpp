#include "Builder.h"
#include "Lock.h"
#include "ScxTranspiler.h"
#include "StdModules.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <set>
#include <fstream>
#include <cctype>
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
    // 0. $SAFEC — an explicit compiler binary, like SAFEC_CLANG for clang
    //    and SAFEC=... in the tests/stdlib scripts. Without this, pointing
    //    SAFEC_HOME at a checkout whose compiler isn't built (a fresh git
    //    worktree, say) silently fell through to whatever 'safec' is on
    //    PATH — typically an older install, which then rejected dozens of
    //    current std sources and got them dropped from libsafec_std.a as
    //    "does not compile for the current target".
    if (const char* e = std::getenv("SAFEC"); e && *e) return e;
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
        // SAFEC_HOME's std/ will still be used (findStdDir), so a PATH
        // compiler of a different vintage can mismatch it — say so once.
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "safeguard: warning: no compiler under SAFEC_HOME (" << home
                      << "/bin/safec or " << home << "/compiler/build/safec); "
                         "using 'safec' from PATH (set SAFEC to choose one)\n";
        }
    }
    // 3. PATH fallback
    return "safec";
}

std::string Builder::findClang(bool cxx) const {
    const char* envVar = cxx ? "SAFEC_CLANGXX" : "SAFEC_CLANG";
    if (const char* e = std::getenv(envVar); e && *e) return e;

    // The clang safec itself uses (safec --print-clang): the one from the
    // LLVM safec was built with, so it reads safec's IR — a system 'clang'
    // older than that LLVM rejects it. clang++ is taken from beside it.
    // An older safec without the option prints nothing usable; fall through.
    static std::string fromSafec = [this] {
        std::string cmd = "\"" + findSafec() + "\" --print-clang";
#ifdef _WIN32
        FILE* p = _popen(cmd.c_str(), "r");
#else
        FILE* p = popen(cmd.c_str(), "r");
#endif
        std::string out;
        if (p) {
            char line[1024];
            if (fgets(line, sizeof(line), p)) out = line;
#ifdef _WIN32
            _pclose(p);
#else
            pclose(p);
#endif
        }
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        return fs::path(out).is_absolute() && fs::exists(out) ? out : std::string();
    }();
    if (!fromSafec.empty()) {
        if (!cxx) return fromSafec;
        fs::path c(fromSafec);
        std::string name = c.filename().string();
        size_t at = name.find("clang");
        if (at != std::string::npos) {
            fs::path cxxPath = c.parent_path() / (name.substr(0, at) + "clang++" + name.substr(at + 5));
            if (fs::exists(cxxPath)) return cxxPath.string();
        }
    }

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

std::string Builder::findLlvmTool(const std::string& name) const {
#ifdef _WIN32
    std::string file = name + ".exe";
#else
    std::string file = name;
#endif
    for (const std::string& clang : { findClang(false), findSystemClang(false) }) {
        fs::path c(clang);
        if (c.has_parent_path() && fs::exists(c.parent_path() / file))
            return (c.parent_path() / file).string();
    }
    // The clang on PATH: an LLVM install keeps its tools beside it.
    if (const char* path = std::getenv("PATH")) {
#ifdef _WIN32
        const char sep = ';'; const char* clangFile = "clang.exe";
#else
        const char sep = ':'; const char* clangFile = "clang";
#endif
        std::stringstream ss(path);
        std::string dir;
        while (std::getline(ss, dir, sep)) {
            if (dir.empty() || !fs::exists(fs::path(dir) / clangFile)) continue;
            if (fs::exists(fs::path(dir) / file)) return (fs::path(dir) / file).string();
            break;
        }
    }
    return name;
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
    if (ext == ".m")   return SrcLang::ObjC;
    if (ext == ".mm")  return SrcLang::ObjCpp;
    // .scx (scx markup templating) transpiles to plain SafeC before it ever
    // reaches safec (see materializeScx) — from compileSrc's point of view
    // it's just SafeC source, same as .sc.
    return SrcLang::SafeC;
}

std::vector<std::string> Builder::collectAllSources(
    const std::string& srcDir, const std::vector<std::string>& excludeSubdirNames) const {
    std::vector<std::string> srcs;
    if (!fs::exists(srcDir)) return srcs;
    static const char* kExts[] = { ".sc", ".scx", ".c", ".cc", ".cpp", ".cxx", ".m", ".mm" };
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
    std::string stem = flatBuildName(fs::path(src).string());
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

// A source's path relative to the project when there is one (same drive,
// on Windows), else its absolute path: fs::relative across drives (std on
// D:, a project in C:\\...\\Temp) yields nothing usable.
std::string Builder::sourceKey(const std::string& src) const {
    std::error_code ec;
    fs::path rel = fs::relative(src, root_, ec);
    if (!ec && !rel.empty() && !rel.has_root_name() && !rel.has_root_directory())
        return rel.string();
    return fs::absolute(src, ec).lexically_normal().string();
}

// sourceKey() flattened into one file name for build/: separators, dots and
// a drive's ':' become '_' (a ':' on NTFS would name an alternate data
// stream of one file, so every std object ended up in the same place).
std::string Builder::flatBuildName(const std::string& src) const {
    std::string r = sourceKey(src);
    for (char& c : r)
        if (c == '/' || c == '\\' || c == '.' || c == ':') c = '_';
    return r.empty() ? fs::path(src).stem().string() : r;
}

#ifdef _WIN32
// One argument as the child's CommandLineToArgvW will read it back: quoted
// when empty or holding a space, tab or quote; a quote, and the backslashes
// before a quote or the closing quote, escaped with backslashes.
static std::string quoteWindowsArg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string out = "\"";
    size_t slashes = 0;
    for (char c : a) {
        if (c == '\\') { ++slashes; continue; }
        if (c == '"') out.append(slashes * 2 + 1, '\\');
        else out.append(slashes, '\\');
        slashes = 0;
        out += c;
    }
    out.append(slashes * 2, '\\');
    out += '"';
    return out;
}
#endif

int Builder::runCmd(const std::vector<std::string>& argv, bool verbose,
                     bool suppressOutput, const std::string& stdoutPath) {
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
    // What safeguard printed so far comes before the child's own output.
    std::cout.flush();
    fflush(stdout);
#ifdef _WIN32
    // _spawnvp is the MSVC CRT's argv-based process launcher — like
    // execvp, it searches PATH and takes an argv array (no shell involved).
    // It does join that array into one command line *without* quoting,
    // though, and the child splits it again (CommandLineToArgvW rules), so
    // each argument is quoted here: a path under 'Program Files' would
    // otherwise arrive as two arguments.
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
    // 'stdoutPath': the child's stdout goes to that file (see captureCmd).
    int keptOut = -1, outFd = -1;
    if (!stdoutPath.empty()) {
        fflush(stdout);
        keptOut = _dup(1);
        outFd = _open(stdoutPath.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC, 0644);
        if (outFd != -1) _dup2(outFd, 1);
    }
    std::vector<std::string> quoted;
    for (auto& s : argv) quoted.push_back(quoteWindowsArg(s));
    std::vector<const char*> args;
    for (auto& s : quoted) args.push_back(s.c_str());
    args.push_back(nullptr);
    intptr_t rc = _spawnvp(_P_WAIT, argv[0].c_str(), args.data());
    int spawnErrno = errno;
    if (keptOut != -1) {
        fflush(stdout);
        _dup2(keptOut, 1); _close(keptOut);
        if (outFd != -1) _close(outFd);
    }
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
    // 'stdoutPath': the child's stdout goes to that file (see captureCmd).
    int keptOut = -1, outFd = -1;
    if (!stdoutPath.empty()) {
        fflush(stdout);
        keptOut = dup(1);
        outFd = open(stdoutPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (outFd != -1) dup2(outFd, 1);
    }
    pid_t pid = fork();
    auto restoreOut = [&]() {
        if (keptOut == -1) return;
        fflush(stdout);
        dup2(keptOut, 1); close(keptOut);
        if (outFd != -1) close(outFd);
    };
    if (pid < 0) {
        std::cerr << "safeguard: fork failed: " << strerror(errno) << "\n";
        restoreOut();
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
    restoreOut();
    if (quiet) {
        fflush(stdout); fflush(stderr);
        if (savedOut != -1) { dup2(savedOut, 1); close(savedOut); }
        if (savedErr != -1) { dup2(savedErr, 2); close(savedErr); }
        if (nullFd != -1) close(nullFd);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// Run a command and collect its stdout (through a temporary file, so the
// argv-based runCmd above needs no shell quoting). Returns false if it
// couldn't run or exited non-zero.
bool Builder::captureCmd(const std::vector<std::string>& argv, std::string& out, bool verbose) {
    static int seq = 0;
#ifdef _WIN32
    int pid = _getpid();
#else
    int pid = (int)getpid();
#endif
    std::error_code ec;
    fs::path tmp = fs::temp_directory_path(ec) /
                   ("safeguard-" + std::to_string(pid) + "-" + std::to_string(++seq) + ".out");
    int rc = runCmd(argv, verbose, /*suppressOutput=*/false, tmp.string());
    std::ifstream in(tmp);
    std::stringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    in.close();
    fs::remove(tmp, ec);
    return rc == 0;
}

// ── compile one .sc → .ll ─────────────────────────────────────────────────────

std::string Builder::compileSrc(const std::string& safecBin,
                                  const std::string& srcPath,
                                  const std::string& buildDir,
                                  const std::vector<std::string>& includeDirs,
                                  bool compatPreprocessor,
                                  bool suppressOutput,
                                  const std::vector<std::string>& defines,
                                  const std::string& outStem) const {
    // Derive a flat output name to avoid collisions across subdirectories
    fs::path src(srcPath);
    // Use relative-path-derived stem: e.g. src/foo/bar.sc → foo_bar.ll
    // ('outStem' names it instead: std's units, see ensureStdLib).
    std::string stem = outStem.empty() ? flatBuildName(fs::path(src).string()) : outStem;
    fs::path llFile = fs::path(buildDir) / (stem + ".ll");

    std::vector<std::string> cmd = { safecBin };
    for (auto& inc : includeDirs) {
        cmd.push_back("-I");
        cmd.push_back(inc);
    }
    for (auto& d : defines) cmd.push_back(d);
    cmd.push_back(srcPath);
    if (!opts_.target.empty()) cmd.push_back("--target=" + opts_.target);
    if (targetIsBareMetal() && !providedFunctionsArg().empty())
        cmd.push_back(providedFunctionsArg());
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
                               bool suppressOutput,
                               bool stdUnit) const {
    fs::path obj = fs::path(buildDir) / (fs::path(llFile).stem().string() + ".o");
    std::vector<std::string> cmd = { findClang(false), "-c", llFile, "-o", obj.string() };
    for (auto& f : targetClangFlags()) cmd.push_back(f);
    for (auto& f : runtimeFlags()) cmd.push_back(f);
    cmd.push_back("-Wno-override-module");
    if (stdUnit) for (auto& f : stdClangFlags()) cmd.push_back(f);   // std: optimized in debug builds too
    else if (opts_.release) cmd.push_back("-O2");
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
                                     bool suppressOutput,
                                     const std::vector<std::string>* flags) const {
    // Objective-C needs Apple's runtime and frameworks; GNUstep isn't
    // wired up, so say so instead of failing on <Foundation/Foundation.h>.
    if (isObjC(lang) && targetOS() != "macos") {
        if (!suppressOutput)
            std::cerr << "safeguard: " << srcPath << ": Objective-C sources (.m/.mm) build only "
                         "for macOS\n";
        return "";
    }
    // Flatten to a unique object name the same way compileSrc() does for
    // .sc files, so a C file and a SafeC file with the same basename in
    // different subdirectories don't clobber each other's .o.
    fs::path src(srcPath);
    std::string stem = flatBuildName(fs::path(src).string());
    fs::path obj = fs::path(buildDir) / (stem + ".o");

    // A cross target compiles with the LLVM clang and its target flags,
    // like the SafeC objects and the link (linkFinal).
    std::string compiler = isCrossTarget() ? findClang(isCxx(lang)) : findSystemClang(isCxx(lang));
    std::vector<std::string> cmd = { compiler, "-c", srcPath, "-o", obj.string() };
    for (auto& f : targetClangFlags()) cmd.push_back(f);
    for (auto& f : runtimeFlags()) cmd.push_back(f);
    for (auto& inc : includeDirs) {
        cmd.push_back("-I");
        cmd.push_back(inc);
    }
    if (opts_.release) cmd.push_back("-O2");
    // 'flags': a built dependency's own (see buildNativeDep); otherwise the
    // project's [build] cflags, plus -fobjc-arc for ObjC when objc_arc is set.
    if (flags) {
        for (auto& f : *flags) cmd.push_back(f);
    } else {
        for (auto& f : compileFlags()) cmd.push_back(f);
        if (isObjC(lang) && manifest_.build.objcArc) cmd.push_back("-fobjc-arc");
    }

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
    // ar rcs <archive> <obj1> <obj2> ... with LLVM's llvm-ar when it sits
    // beside clang: it indexes ELF, Mach-O and COFF members alike, where the
    // system 'ar' only knows its own platform's (Apple's leaves a cross-built
    // Windows or Linux archive with no symbol table, so nothing links from it).
    std::string ar = findLlvmTool("llvm-ar");
    if (ar == "llvm-ar") ar = "ar";   // not found beside clang: the system one
    std::vector<std::string> cmd = { ar, "rcs", archivePath };
    for (auto& o : objFiles) cmd.push_back(o);
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: ar failed for " << archivePath << "\n";
        return false;
    }
    return true;
}

// ── CPU BLAS ─────────────────────────────────────────────────────────────────
// std/simd/gemm.sc calls cblas_sgemm through a weak reference, so a program
// gets the CPU vendor's BLAS for its matrix products (tensor_matmul,
// conv2d_forward, ...) just by linking it — which a build with gemm.sc in
// it does ('// safeguard: link hosted blas', see ensureStdLib). [build]
// blas picks which:
//   "auto" (default)  AOCL-BLAS on AMD CPUs, oneMKL on Intel, Accelerate on
//                     macOS -- if the library is found; otherwise none
//   "aocl" / "mkl" / "accelerate" / "openblas"   that one (error if missing)
//   "none"            std's built-in SIMD kernel only
// SAFEC_BLAS overrides the manifest's choice and SAFEC_BLAS_LIBS replaces the
// whole set of link flags. Libraries are searched in AOCL_ROOT / MKLROOT /
// OPENBLAS_ROOT (or their lib/, lib_LP64/, lib/intel64/ subdirectories), then
// the default install locations.

namespace {

// Path to lib<name>.so (or the highest-numbered lib<name>.so.N) in dir.
std::string findSharedLib(const fs::path& dir, const std::string& name) {
    std::error_code ec;
    fs::path plain = dir / ("lib" + name + ".so");
    if (fs::exists(plain, ec)) return plain.string();
    std::string best;
    for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::string f = it->path().filename().string();
        std::string pre = "lib" + name + ".so.";
        if (f.rfind(pre, 0) == 0 && (best.empty() || f > fs::path(best).filename().string()))
            best = it->path().string();
    }
    return best;
}

// The first directory (root, root/<sub> for each sub, or a default) holding lib<name>.
std::string findLibDir(const char* envVar, const std::vector<std::string>& subs,
                       const std::vector<std::string>& defaults, const std::string& name) {
    std::vector<fs::path> cands;
    if (const char* r = std::getenv(envVar)) {
        cands.push_back(r);
        for (auto& s : subs) cands.push_back(fs::path(r) / s);
    }
    for (auto& d : defaults) {
        // A trailing '*' component globs one directory level (versioned installs).
        auto star = d.find('*');
        if (star == std::string::npos) { cands.push_back(d); continue; }
        fs::path parent = fs::path(d.substr(0, star)).parent_path();
        std::string rest = d.substr(d.find('/', star) == std::string::npos ? d.size() : d.find('/', star) + 1);
        std::error_code ec;
        std::vector<fs::path> hits;
        for (auto it = fs::directory_iterator(parent, ec); !ec && it != fs::directory_iterator(); it.increment(ec))
            hits.push_back(rest.empty() ? it->path() : it->path() / rest);
        std::sort(hits.rbegin(), hits.rend());   // newest version first
        for (auto& h : hits) cands.push_back(h);
    }
    for (auto& c : cands)
        if (!findSharedLib(c, name).empty()) return c.string();
    return "";
}

std::string cpuVendor() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("vendor_id", 0) == 0) {
            if (line.find("AuthenticAMD") != std::string::npos) return "amd";
            if (line.find("GenuineIntel") != std::string::npos) return "intel";
            return "other";
        }
    return "other";
}

} // namespace

#ifdef __APPLE__
// -U for each of std's weak references into Accelerate (see blasLinkFlags).
static std::vector<std::string> appleWeakUndefined() {
    return { "-Wl,-U,_cblas_sgemm", "-Wl,-U,_vDSP_conv", "-Wl,-U,_vDSP_convD" };
}
#endif

std::vector<std::string> Builder::blasLinkFlags() const {
    std::vector<std::string> out;
    if (targetIsBareMetal() || targetIsWasm()) return out;
#ifdef _WIN32
    return out;   // gemm.sc doesn't use a vendor BLAS on Windows yet
#else
    if (isCrossTarget()) return out;
    if (const char* libs = std::getenv("SAFEC_BLAS_LIBS")) {
        std::istringstream ss(libs);
        std::string f;
        while (ss >> f) out.push_back(f);
        return out;
    }
    std::string mode = manifest_.build.blas.empty() ? "auto" : manifest_.build.blas;
    if (const char* e = std::getenv("SAFEC_BLAS")) mode = e;
    bool automatic = mode == "auto";
#ifdef __APPLE__
    if (mode == "auto" || mode == "accelerate") {
        out = { "-framework", "Accelerate" };
        if (opts_.verbose) std::cout << "safeguard: CPU BLAS: Accelerate\n";
        return out;
    }
    // Mach-O resolves even weak references at link time; without Accelerate,
    // let std's weak references into it (cblas_sgemm in std/simd/gemm.sc,
    // vDSP_conv/vDSP_convD in std/dsp/dsp.sc) stay undefined -- null at run
    // time, so std's own kernels run.
    if (mode == "none") return appleWeakUndefined();
#endif
    if (mode == "none") return out;
    if (automatic) {
        std::string v = cpuVendor();
        mode = v == "amd" ? "aocl" : v == "intel" ? "mkl" : "none";
        if (mode == "none") return out;
    }
    auto rpathLink = [&](const std::string& dir, const std::vector<std::string>& names) -> bool {
        std::vector<std::string> paths;
        for (auto& n : names) {
            std::string p = findSharedLib(dir, n);
            if (p.empty()) return false;
            paths.push_back(p);
        }
        out.push_back("-Wl,-rpath," + dir);
        // Only weak references point at these, which --as-needed wouldn't count.
        out.push_back("-Wl,--push-state,--no-as-needed");
        for (auto& p : paths) out.push_back(p);
        out.push_back("-lgomp");
        out.push_back("-Wl,--pop-state");
        return true;
    };
    std::string dir;
    bool ok = false;
    if (mode == "aocl") {
        dir = findLibDir("AOCL_ROOT", {"lib", "lib_LP64", "gcc/lib_LP64"},
                         {"/opt/AMD/aocl/aocl-linux-gcc-*/gcc/lib_LP64", "/opt/AMD/aocl/*/lib_LP64",
                          "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"}, "blis-mt");
        ok = !dir.empty() && rpathLink(dir, {"blis-mt"});
    } else if (mode == "mkl") {
        dir = findLibDir("MKLROOT", {"lib", "lib/intel64"},
                         {"/opt/intel/oneapi/mkl/latest/lib", "/opt/intel/oneapi/mkl/latest/lib/intel64",
                          "/opt/intel/mkl/lib/intel64", "/usr/lib/x86_64-linux-gnu"}, "mkl_core");
        // GNU OpenMP threading layer: the gcc/clang link line, no libiomp5 needed.
        ok = !dir.empty() && rpathLink(dir, {"mkl_intel_lp64", "mkl_gnu_thread", "mkl_core"});
    } else if (mode == "openblas") {
        dir = findLibDir("OPENBLAS_ROOT", {"lib"}, {"/usr/local/lib", "/usr/lib/x86_64-linux-gnu/openblas-pthread",
                                                    "/usr/lib/x86_64-linux-gnu"}, "openblas");
        ok = !dir.empty() && rpathLink(dir, {"openblas"});
    } else if (mode == "accelerate") {
        std::cerr << "safeguard: warning: build.blas = \"accelerate\" needs macOS; using the built-in kernel\n";
        return {};
    }
#ifdef __APPLE__
    // A BLAS other than Accelerate leaves the vDSP references unresolved.
    for (auto& f : appleWeakUndefined())
        if (f != "-Wl,-U,_cblas_sgemm" || !ok) out.push_back(f);
#endif
    if (!ok) {
        out.clear();
#ifdef __APPLE__
        out = appleWeakUndefined();
#endif
        if (!automatic)
            std::cerr << "safeguard: warning: build.blas = \"" << mode
                      << "\" but the library wasn't found (set AOCL_ROOT / MKLROOT / OPENBLAS_ROOT); "
                         "using the built-in kernel\n";
        else if (opts_.verbose)
            std::cout << "safeguard: CPU BLAS: " << mode << " not found, using std's built-in kernel\n";
        return out;
    }
    if (opts_.verbose) std::cout << "safeguard: CPU BLAS: " << mode << " (" << dir << ")\n";
    return out;
#endif
}

// ── final link ────────────────────────────────────────────────────────────────

bool Builder::linkFinal(const std::vector<std::string>& objFiles,
                          const std::vector<std::string>& archives,
                          const std::string& output,
                          bool useCxxDriver,
                          bool shared,
                          bool objc,
                          const std::vector<std::string>& extraFlags) const {
    // A cross target links with the same LLVM clang that assembled the IR
    // (the system one may not know the target), through lld; bare metal
    // gets no libc/crt — the program's own runtime, linker script ('-T') and
    // libraries come from the manifest's [build] cflags/libs/lib_dirs.
    std::vector<std::string> cmd = { isCrossTarget() ? findClang(useCxxDriver)
                                                     : findSystemClang(useCxxDriver) };
    for (auto& f : targetClangFlags()) cmd.push_back(f);
    for (auto& f : runtimeFlags()) cmd.push_back(f);
    for (auto& f : runtimeLinkFlags()) cmd.push_back(f);
    if (isCrossTarget()) cmd.push_back("-fuse-ld=lld");
    if (targetIsBareMetal()) { cmd.push_back("-nostdlib"); cmd.push_back("-static"); }
    // WebAssembly: a library module the host instantiates — no _start, its
    // exports are the __attribute__((export_name)) functions, unresolved
    // externs become imports.
    if (targetIsWasm())
        for (const char* f : { "-Wl,--no-entry", "-Wl,--allow-undefined" })
            cmd.push_back(f);
    if (opts_.release) cmd.push_back("-O2");
    if (!manifest_.build.lto.empty())
        cmd.push_back(manifest_.build.lto == "full" ? "-flto" : "-flto=thin");
    if (shared) cmd.push_back("-shared");
    for (auto& o : objFiles) cmd.push_back(o);
    // Archives: pass with -force_load on macOS so unused symbols are kept,
    // or use the portable -Wl,--whole-archive on Linux.
    // For simplicity, just list the .a files directly (works for most cases).
    for (auto& a : archives) cmd.push_back(a);
    // The system libraries std's modules in the build call into (see
    // ensureStdLib and the '// safeguard: link' lines in std/): libm for
    // std/math.sc, Cocoa or X11 or user32/gdi32 for std/gui, the CPU BLAS
    // for std/simd/gemm.sc, ... — only when std is linked in at all (a
    // package with no SafeC in it doesn't; see build()).
    bool linksStd = !stdArchive_.empty() &&
                    std::find(archives.begin(), archives.end(), stdArchive_) != archives.end();
    if (linksStd) for (auto& f : stdLinkFlags_) cmd.push_back(f);
    const PlatformBuildConfig* plat = platformBuild();
    for (auto& d : manifest_.build.libDirs) cmd.push_back("-L" + d);
    if (plat) for (auto& d : plat->libDirs) cmd.push_back("-L" + d);
    // The project's own libraries and frameworks, less the ones std's
    // modules already brought (ld64 warns about duplicates).
    auto linked = [&](const std::string& a, const std::string& b) {
        for (size_t i = 0; i < cmd.size(); ++i) {
            if (b.empty() && cmd[i] == a) return true;
            if (!b.empty() && cmd[i] == a && i + 1 < cmd.size() && cmd[i + 1] == b) return true;
        }
        return false;
    };
    for (auto& l : manifest_.build.libs)    if (!linked("-l" + l, "")) cmd.push_back("-l" + l);
    if (plat) for (auto& l : plat->libs)    if (!linked("-l" + l, "")) cmd.push_back("-l" + l);
    // Frameworks are a Mach-O notion: [build] frameworks apply on macOS and
    // are dropped (with a note) elsewhere; [build.macos] ones only reach here
    // on macOS anyway.
    std::vector<std::string> frameworks = manifest_.build.frameworks;
    if (plat) frameworks.insert(frameworks.end(), plat->frameworks.begin(), plat->frameworks.end());
    if (targetOS() == "macos") {
        for (auto& f : frameworks)
            if (!linked("-framework", f)) { cmd.push_back("-framework"); cmd.push_back(f); }
        if (objc && !linked("-lobjc", "")) cmd.push_back("-lobjc");
    } else if (!manifest_.build.frameworks.empty()) {
        static bool noted = false;
        if (!noted) {
            noted = true;
            std::cerr << "safeguard: note: [build] frameworks apply to macOS builds only; ignored here\n";
        }
    }
    // Extra link flags from manifest
    for (auto& f : compileFlags()) cmd.push_back(f);
    for (auto& f : extraFlags) cmd.push_back(f);
    cmd.push_back("-o");
    cmd.push_back(output);
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: link failed\n";
        return false;
    }
    return true;
}

// ── --target ─────────────────────────────────────────────────────────────────

const std::vector<std::string>& Builder::targetClangFlags() const {
    if (targetFlagsResolved_) return targetFlags_;
    targetFlagsResolved_ = true;
    if (opts_.target.empty()) return targetFlags_;
    std::string cmd = "\"" + findSafec() + "\" --print-target-clang-flags --target=" + opts_.target;
#ifdef _WIN32
    FILE* p = _popen(cmd.c_str(), "r");
#else
    FILE* p = popen(cmd.c_str(), "r");
#endif
    if (p) {
        char line[512];
        while (fgets(line, sizeof(line), p)) {
            std::string l(line);
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            if (!l.empty()) targetFlags_.push_back(l);
        }
#ifdef _WIN32
        _pclose(p);
#else
        pclose(p);
#endif
    }
    if (targetFlags_.empty()) targetFlags_.push_back("--target=" + opts_.target);
    return targetFlags_;
}

const std::string& Builder::providedFunctionsArg() const {
    if (providesResolved_) return providesArg_;
    providesResolved_ = true;
    static const char* kNames[] = {
        "malloc", "free", "calloc", "realloc", "printf", "fprintf", "sprintf",
        "snprintf", "puts", "putchar", "getchar", "fopen", "fclose", "fread",
        "fwrite", "exit", "abort", "atexit", nullptr };
    std::set<std::string> found;
    for (const char* dir : { "src", "tests" }) {
        for (auto& f : collectAllSources((fs::path(root_) / dir).string())) {
            std::ifstream in(f);
            std::string line;
            while (std::getline(in, line)) {
                for (const char** n = kNames; *n; ++n) {
                    // A definition: '<type> name(<params>) {' on one line.
                    size_t at = line.find(std::string(*n) + "(");
                    if (at == std::string::npos) continue;
                    if (at > 0 && (std::isalnum((unsigned char)line[at - 1]) || line[at - 1] == '_')) continue;
                    if (line.find("extern") != std::string::npos) continue;
                    size_t close = line.find(')', at);
                    size_t brace = close == std::string::npos ? std::string::npos : line.find('{', close);
                    // '{' with no ';' between ')' and it: a body, not a prototype.
                    if (brace != std::string::npos && at > 0 &&
                        line.substr(close, brace - close).find(';') == std::string::npos)
                        found.insert(*n);
                }
            }
        }
    }
    if (!found.empty()) {
        providesArg_ = "--provides=";
        bool first = true;
        for (auto& n : found) { if (!first) providesArg_ += ","; providesArg_ += n; first = false; }
    }
    return providesArg_;
}

std::string Builder::targetTriple() const {
    for (auto& f : targetClangFlags())
        if (f.rfind("--target=", 0) == 0) return f.substr(9);
    return "";
}

// No-OS triples: '<arch>-unknown-none-elf', '<arch>-none-eabi', '<arch>-unknown-elf',
// and WebAssembly without WASI ('wasm32-unknown-unknown').
bool Builder::targetIsBareMetal() const {
    std::string t = targetTriple();
    if (t.empty()) return false;
    return t.find("-none") != std::string::npos ||
           (t.find("-unknown-elf") != std::string::npos) || targetIsWasm();
}

bool Builder::targetIsWasm() const {
    std::string t = targetTriple();
    return t.rfind("wasm32-", 0) == 0 || t.rfind("wasm64-", 0) == 0;
}

// With [build] std = true, C/C++/ObjC sources reach std's headers as
// <std/...> through std's parent directory -- never std/ itself, whose
// stdint.h & co. would shadow the system's (see build()'s foreignIncs).
void Builder::stdForeignInclude(std::vector<std::string>& foreignIncs, const std::string& stdDir) const {
    if (manifest_.build.useStd == 1 && !stdDir.empty())
        foreignIncs.push_back(fs::path(stdDir).parent_path().string());
}

std::vector<std::string> Builder::runtimeFlags() const {
    if (!targetIsWindows()) return {};
    std::string t = targetTriple();
    // MinGW (-gnu) targets use their own runtime's shared msvcrt already.
    if (t.find("gnu") != std::string::npos || t.find("mingw") != std::string::npos) return {};
    return { "-fms-runtime-lib=dll" };
}

std::vector<std::string> Builder::runtimeLinkFlags() const {
    if (runtimeFlags().empty()) return {};
    // legacy_stdio_definitions: with the DLL CRT, printf & co. are inline
    // functions in the UCRT headers, which SafeC's own 'extern int printf'
    // declarations never see; this library has their out-of-line bodies.
    return { "-Wl,/NODEFAULTLIB:libcmt", "-Wl,/DEFAULTLIB:msvcrt",
             "-Wl,/DEFAULTLIB:legacy_stdio_definitions" };
}

std::string Builder::targetOS() const {
    if (!isCrossTarget()) {
#if defined(_WIN32)
        return "windows";
#elif defined(__APPLE__)
        return "macos";
#elif defined(__linux__)
        return "linux";
#else
        return "";
#endif
    }
    if (targetIsBareMetal()) return "";
    std::string t = targetTriple();
    if (t.find("windows") != std::string::npos) return "windows";
    if (t.find("apple") != std::string::npos || t.find("darwin") != std::string::npos ||
        t.find("macos") != std::string::npos) return "macos";
    if (t.find("linux") != std::string::npos) return "linux";
    return "";
}

const PlatformBuildConfig* Builder::platformBuild() const {
    std::string os = targetOS();
    if (os == "macos")   return &manifest_.build.forMacos;
    if (os == "linux")   return &manifest_.build.forLinux;
    if (os == "windows") return &manifest_.build.forWindows;
    return nullptr;
}

std::vector<std::string> Builder::compileFlags() const {
    std::vector<std::string> out = manifest_.build.cflags;
    if (const PlatformBuildConfig* p = platformBuild())
        out.insert(out.end(), p->cflags.begin(), p->cflags.end());
    return out;
}

// ── build a library from a source directory ───────────────────────────────────

bool Builder::buildLib(const std::string& srcDir,
                        const std::string& libOut,
                        const std::vector<std::string>& includeDirs,
                        bool compatPreprocessor,
                        const std::vector<std::string>& foreignIncludeDirs) {
    std::string safecBin = findSafec();
    std::string buildDir = fs::path(libOut).parent_path().string();
    fs::create_directories(buildDir);

    auto srcs = collectAllSources(srcDir);
    if (srcs.empty()) return true; // nothing to compile

    std::vector<std::string> objs;
    std::set<std::string> refs;
    for (auto& src : srcs) {
        SrcLang lang = langOf(src);
        std::string obj;
        if (lang == SrcLang::SafeC) {
            std::string realSrc = materializeScx(src, buildDir);
            if (realSrc.empty()) return false;
            std::string ll = compileSrc(safecBin, realSrc, buildDir, includeDirs, compatPreprocessor);
            if (ll.empty()) return false;
            for (auto& r : irExternalRefs(ll)) refs.insert(r);
            obj = llToObj(ll, buildDir);
        } else {
            obj = compileForeign(src, buildDir, foreignIncludeDirs, lang);
        }
        if (obj.empty()) return false;
        objs.push_back(obj);
    }
    // The symbols its code refers to, for the program's runtime units
    // (depStdRefs).
    writeRefs(libOut + ".refs", refs);
    return packArchive(objs, libOut);
}

// ── build (or reuse) the SafeC standard library ───────────────────────────────

StdPlatform Builder::stdPlatform() const {
    StdPlatform p;
    p.triple = targetTriple();
    p.bareMetal = targetIsBareMetal();
    p.wasm = targetIsWasm();
    p.os = p.wasm ? "wasm" : p.bareMetal ? "baremetal" : targetOS();
    if (p.os.empty()) p.os = "bsd";   // the remaining hosted ones (kqueue & co.)
    for (auto& f : enabledFeatures()) p.features.insert(f);
    return p;
}

// The std files a build's own files include (see stdRootsOf): 'files' (its
// SafeC sources, and its C/C++/ObjC ones with [build] std = true), every
// SafeC source of a dependency built from source, and what a prebuilt
// SafeC library recorded beside itself (lib<name>.a.std, see build()).
std::set<std::string> Builder::stdRoots(const std::vector<std::string>& files) const {
    std::string stdDir = findStdDir();
    if (stdDir.empty()) return {};
    const StdIndex& idx = StdIndex::load(stdDir);
    std::vector<std::string> scan;
    for (auto& f : files) {
        if (langOf(f) != SrcLang::SafeC && manifest_.build.useStd != 1) continue;
        // .scx: what it transpiles to (which includes <std/scx/scx.h>).
        std::string real = materializeScx(f, (fs::path(root_) / "build").string());
        scan.push_back(real.empty() ? f : real);
    }
    std::vector<std::string> incs, foreign;
    fs::path projInc = fs::path(root_) / "include";
    if (fs::exists(projInc)) incs.push_back(projInc.string());
    depIncludes(incs, foreign);
    std::set<std::string> roots;
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) continue;
        if (l.fromSource)
            for (auto& f : collectAllSources((fs::path(l.dir) / "src").string()))
                if (langOf(f) == SrcLang::SafeC) scan.push_back(f);
        for (auto& lib : l.linkInputs) {
            std::ifstream in(lib + ".std");
            std::string line;
            while (std::getline(in, line))
                if (!line.empty() && idx.files.count(line)) roots.insert(line);
        }
    }
    for (auto& r : stdRootsOf(idx, scan, incs, stdPlatform())) roots.insert(r);
    return roots;
}

// The external symbols a build's dependencies refer to: the '.refs' a
// SafeC library build writes beside it (buildLib for a source dependency,
// build() for a staticlib). A SafeC library from before those existed (a
// '.std' but no '.refs') counts as referring to every runtime symbol, as
// every build then included those units.
std::set<std::string> Builder::depStdRefs() const {
    std::set<std::string> refs;
    std::string stdDir = findStdDir();
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) continue;
        for (auto& lib : l.linkInputs) {
            if (lib.rfind("-", 0) == 0) continue;
            if (readRefs(lib + ".refs", refs)) continue;
            std::error_code ec;
            if ((l.fromSource || fs::exists(lib + ".std", ec)) && !stdDir.empty())
                for (auto& p : StdIndex::load(stdDir).runtimeSymbolPrefixes()) refs.insert(p);
        }
    }
    return refs;
}

std::vector<std::string> Builder::safecIncludeDirs() const {
    std::vector<std::string> incs, foreign;
    std::string stdDir = findStdDir();
    if (!stdDir.empty()) {
        incs.push_back(stdDir);
        incs.push_back(fs::path(stdDir).parent_path().string());
    }
    fs::path projInc = fs::path(root_) / "include";
    if (fs::exists(projInc)) incs.push_back(projInc.string());
    depIncludes(incs, foreign);
    return incs;
}

bool Builder::sourceRefs(const std::vector<std::string>& files, const std::string& outDir,
                         std::set<std::string>& refs) {
    std::vector<std::string> incs = safecIncludeDirs();
    std::vector<std::string> defs = featureDefines();
    fs::create_directories(outDir);
    for (auto& f : files) {
        if (langOf(f) != SrcLang::SafeC) continue;
        std::string real = materializeScx(f, outDir);
        std::string ll = real.empty() ? "" : compileSrc(findSafec(), real, outDir, incs, false, false, defs);
        if (ll.empty()) return false;
        for (auto& r : irExternalRefs(ll)) refs.insert(r);
    }
    return true;
}

// ── hashes ───────────────────────────────────────────────────────────────────

// A std unit's input hash: FNV-1a 64 (as Package.lock's, hashFile) over
// the paths and hashes of the files its compile reads. It tells versions
// apart; it doesn't authenticate them.
namespace {
struct Fnv64 {
    unsigned long long h = 14695981039346656037ULL;
    void add(const std::string& s) {
        for (char c : s) { h ^= (unsigned char)c; h *= 1099511628211ULL; }
        h *= 1099511628211ULL;   // a separator (xor 0)
    }
    std::string hex() const {
        char buf[17];
        std::snprintf(buf, sizeof buf, "%016llx", h);
        return buf;
    }
};
}

std::string Builder::stdUnitHash(const StdIndex& idx, const std::string& unit) const {
    Fnv64 h;
    for (auto& in : idx.compileInputs(unit)) {   // a sorted set: a stable order
        auto it = fileHashes_.find(in);
        if (it == fileHashes_.end())
            it = fileHashes_.emplace(in, hashFile((fs::path(idx.dir) / in).string())).first;
        h.add(in);
        h.add(it->second);
    }
    return h.hex();
}

// ── prebuilt std ─────────────────────────────────────────────────────────────

std::string Builder::hostTriple() {
#if defined(__aarch64__) || defined(_M_ARM64)
    std::string arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    std::string arch = "x86_64";
#else
    std::string arch = "unknown";
#endif
#if defined(__APPLE__)
    return arch + "-apple-darwin";
#elif defined(_WIN32)
    return arch + "-pc-windows-msvc";
#elif defined(__linux__)
    return arch + "-unknown-linux-gnu";
#else
    return arch + "-unknown-unknown";
#endif
}

std::string Builder::prebuiltStdStamp() const {
    // The compiler by content: its version line alone doesn't change between
    // the commits of a release.
    std::string safec = findSafec();
    if (!fs::path(safec).has_parent_path()) {
        // A PATH-resolved 'safec': find the file.
        if (const char* path = std::getenv("PATH")) {
#ifdef _WIN32
            const char sep = ';'; std::string file = safec + ".exe";
#else
            const char sep = ':'; std::string file = safec;
#endif
            std::stringstream ss(path);
            std::string dir;
            while (std::getline(ss, dir, sep))
                if (!dir.empty() && fs::exists(fs::path(dir) / file)) { safec = (fs::path(dir) / file).string(); break; }
        }
    }
    std::string hash = hashFile(safec);   // as Package.lock's
    if (hash.empty()) return "";
    std::string version;
    captureCmd({ safec, "--version" }, version, false);
    while (!version.empty() && (version.back() == '\n' || version.back() == '\r')) version.pop_back();
    std::string s = "format=1\ntriple=" + hostTriple() + "\nsafec=" + hash + "\n";
    std::string rt;
    for (auto& f : runtimeFlags()) rt += (rt.empty() ? "" : " ") + f;
    s += "runtime=" + rt + "\n";
    // Informational; not compared.
    return s + "safec-version=" + version.substr(0, version.find('\n')) + "\n";
}

std::vector<std::string> Builder::stdClangFlags() const {
    std::vector<std::string> f = { "-O2" };
    if (!isCrossTarget() && targetOS() == "macos") {
        // The prebuilt std runs on kStdMacosMin and later, as the release's
        // binaries do (release.yml); std compiled here matches it, or the
        // deployment target the build asks for.
        std::string min = kStdMacosMin;
        const char* e = std::getenv("MACOSX_DEPLOYMENT_TARGET");
        if (!writingPrebuiltStd_ && e && *e) min = e;
        f.push_back("-mmacosx-version-min=" + min);
    }
    return f;
}

bool Builder::readPrebuiltUnits(const std::string& file, PrebuiltStd& out) {
    std::ifstream in(file);
    if (!in) return false;
    std::string line, cur;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string kind, a, b;
        ss >> kind >> a >> b;
        if (kind == "unit" || kind == "skip") {
            out.unitHash[a] = b;
            if (kind == "skip") out.skipped.insert(a);
            cur = a;
            out.refs[cur];
        } else if (kind == "ref" && !cur.empty()) {
            out.refs[cur].insert(a);
        }
    }
    return true;
}

Builder::PrebuiltStd Builder::findPrebuiltStd() const {
    PrebuiltStd none;
    if (stdFromSource_ || isCrossTarget()) return none;   // cross, bare metal, wasm: never
    std::string triple = hostTriple();
    std::vector<fs::path> cands;
    std::string safec = findSafec();
    if (fs::path(safec).has_parent_path())
        cands.push_back(fs::path(safec).parent_path().parent_path() / "lib" / triple);
    if (const char* home = std::getenv("SAFEC_HOME"); home && *home)
        cands.push_back(fs::path(home) / "lib" / triple);
    fs::path dir;
    std::error_code ec;
    for (auto& c : cands)
        if (fs::exists(c / "libsafec_std.a", ec)) { dir = c; break; }
    if (dir.empty()) return none;
    fs::path archive = dir / "libsafec_std.a";
    auto skip = [&](const std::string& why) {
        if (opts_.verbose)
            std::cout << "safeguard: not using the prebuilt std archive " << archive.string() << ": " << why
                      << "; compiling std from source\n";
        return none;
    };
    if (!manifest_.build.stdPrebuilt) return skip("[build] std_prebuilt = false");
    if (const char* e = std::getenv("SAFEC_STD_FROM_SOURCE"); e && *e && std::string(e) != "0")
        return skip("SAFEC_STD_FROM_SOURCE is set");
    if (!opts_.extraFlags.empty()) return skip("extra compiler flags are given");
    std::string have;
    {
        std::ifstream in(dir / "libsafec_std.stamp");
        std::stringstream ss;
        ss << in.rdbuf();
        have = ss.str();
    }
    if (have.empty()) return skip("it has no libsafec_std.stamp");
    std::string want = prebuiltStdStamp();
    // Compare all but the informational safec-version line.
    auto fields = [](const std::string& s) {
        std::map<std::string, std::string> m;
        std::istringstream ss(s);
        std::string l;
        while (std::getline(ss, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            size_t eq = l.find('=');
            if (eq != std::string::npos && l.compare(0, eq, "safec-version") != 0) m[l.substr(0, eq)] = l.substr(eq + 1);
        }
        return m;
    };
    auto h = fields(have), w = fields(want);
    if (h != w) {
        std::string why;
        for (auto& [k, v] : w)
            if (h[k] != v) {
                why = k == "safec" ? "it was built by a different safec (stamp " + h[k] + ", " + safec + " is " + v + ")"
                    : k == "triple" ? "it is for " + h[k] + ", not " + v
                    : "its " + k + " is '" + h[k] + "', this build's '" + v + "'";
                break;
            }
        if (why.empty()) why = "its stamp is of another format";
        return skip(why);
    }
    PrebuiltStd p;
    if (!readPrebuiltUnits((dir / "libsafec_std.units").string(), p)) return skip("it has no libsafec_std.units");
    p.archive = archive.string();
    return p;
}

bool Builder::stdUnitObject(const StdIndex& idx, const std::string& unit, const std::string& objDirS,
                            bool compile, StdUnitObj& out) const {
    fs::path objDir(objDirS);
    std::string stdDir = idx.dir;
    std::string stem = unit;
    for (char& c : stem) if (c == '/' || c == '.') c = '_';
    fs::path obj = objDir / (stem + ".o"), skip = objDir / (stem + ".skip"), refs = objDir / (stem + ".refs");
    std::error_code ec;
    std::string safecBin = findSafec();
    fs::file_time_type newest = fs::file_time_type::min();
    if (fs::exists(safecBin, ec)) newest = fs::last_write_time(safecBin, ec);
    for (auto& in : idx.compileInputs(unit)) {
        auto t = fs::last_write_time(fs::path(stdDir) / in, ec);
        if (!ec && t > newest) newest = t;
    }
    auto fresh = [&](const fs::path& f) {
        return fs::exists(f, ec) && fs::last_write_time(f, ec) >= newest;
    };
    if (fresh(obj) && readRefs(refs, out.refs)) { out.obj = obj.string(); return true; }
    if (fresh(skip)) { out.skipped = true; return true; }
    if (!compile) return true;   // unknown: no object, no symbols
    fs::remove(obj, ec);
    fs::remove(skip, ec);
    // safec itself can also reject a file purely for being
    // architecture-inapplicable (e.g. std::simd's ARM-only DSP
    // builtins raising a Sema/CodeGen error on a non-ARM target, the
    // same intent as the assembler rejecting riscv.sc's CSR
    // instructions on non-RISC-V — just enforced earlier in the
    // pipeline). Both are tolerated identically: the file's functions
    // simply aren't available for this target; anything that actually
    // calls them gets a normal, clear "undefined symbol" at link time.
    // The std .sc files include their own headers; pass the std dir's parent
    // so that #include "std/io.h" resolves, and the std dir itself so that
    // #include "io.h" (relative includes inside .sc files) resolves.
    std::vector<std::string> incs = { stdDir, fs::path(stdDir).parent_path().string() };
    std::string ll = compileSrc(safecBin, (fs::path(stdDir) / unit).string(), objDir.string(), incs,
                                /*compatPreprocessor=*/true, /*suppressOutput=*/true, {}, stem);
    std::string o = ll.empty() ? "" : llToObj(ll, objDir.string(), /*suppressOutput=*/true, /*stdUnit=*/true);
    if (o.empty()) {
        std::cerr << "safeguard: warning: skipping 'std/" << unit << "' — does not "
                  << (ll.empty() ? "compile" : "assemble") << " for the current target\n";
        std::ofstream(skip) << "";
        out.skipped = true;
        return true;
    }
    out.refs = irExternalRefs(ll);
    writeRefs(refs, out.refs);
    out.obj = o;
    return true;
}

bool Builder::ensureStdLib(const std::set<std::string>& roots, const std::set<std::string>& refs, bool link) {
    stdArchive_.clear();
    stdLinkFlags_.clear();
    stdDebDepends_.clear();
    stdUnits_.clear();
    std::string stdDir = findStdDir();
    // A std archive named by SAFEC_STD_ARCHIVE: linked as is -- the linker
    // takes the members the program references. Its system libraries still
    // come from std's annotations when std/ is around (see below).
    const char* envArchive = std::getenv("SAFEC_STD_ARCHIVE");
    std::string given = envArchive ? envArchive : "";
    if (!given.empty() && !fs::exists(given)) {
        std::cerr << "safeguard: SAFEC_STD_ARCHIVE names " << given << ", which doesn't exist\n";
        return false;
    }
    if (stdDir.empty()) {
        if (!given.empty()) { stdArchive_ = given; return true; }
        std::cerr << "safeguard: warning: cannot find SafeC std/ directory; "
                     "set SAFEC_HOME to your SafeC install prefix or repository root\n";
        return true;
    }
    const StdIndex& idx = StdIndex::load(stdDir);
    static bool reported = false;
    if (!reported) {
        reported = true;
        for (auto& p : idx.problems) std::cerr << "safeguard: warning: " << p << "\n";
    }
    StdPlatform plat = stdPlatform();

    // Where the units' objects come from: the prebuilt archive of a release
    // install, as far as it has them unchanged (std/ newer than it is
    // compiled here), or SAFEC_STD_ARCHIVE's archive, or compiled here.
    PrebuiltStd pre;
    if (!given.empty()) {
        // Its units' symbols, if it is a prebuilt one (see writePrebuiltStd).
        readPrebuiltUnits(fs::path(given).replace_extension(".units").string(), pre);
        pre.unitHash.clear();
    } else {
        pre = findPrebuiltStd();
        if (!pre.archive.empty() && opts_.verbose)
            std::cout << "safeguard: linking the prebuilt std archive " << pre.archive << "\n";
    }

    // Each unit compiled here gets its own object under
    // build/deps/[<target>/]std/, reused until the unit or a std header it
    // includes changes, or the compiler does. The stamp records which
    // compiler, std/ and flags built them: switching either (e.g.
    // SAFEC_HOME/SAFEC now naming a different, possibly older safec)
    // rebuilds everything -- an mtime comparison can't see that, and a
    // PATH-resolved 'safec' has no path to stat at all. std is compiled with
    // the same flags in debug and release builds (stdClangFlags), so the two
    // share these objects. A unit that doesn't compile for the target
    // (std/hal/riscv.sc's CSR asm on another ISA, say) leaves a .skip marker
    // instead, so it is reported once, not on every build.
    fs::path depDir = isCrossTarget() ? fs::path(root_) / "build" / "deps" / targetDirName()
                                      : fs::path(root_) / "build" / "deps";
    fs::path objDir = depDir / "std";
    fs::path libPath = depDir / "libsafec_std.a";
    std::error_code ec;
    std::string stamp = "safec=" + findSafec() + "\nstd=" + stdDir + "\n";
    for (auto& f : runtimeFlags()) stamp += "runtime=" + f + "\n";   // a CRT switch rebuilds std
    for (auto& f : stdClangFlags()) stamp += "flag=" + f + "\n";     // (objects from before: -O0)
    {
        std::ifstream in(objDir / "build.stamp");
        std::stringstream prev;
        prev << in.rdbuf();
        if (!in || prev.str() != stamp) fs::remove_all(objDir, ec);
    }
    fs::remove(fs::path(libPath).concat(".stamp"), ec);   // the whole-std archive's, before units
    fs::create_directories(objDir);
    std::ofstream(objDir / "build.stamp") << stamp;

    // A unit: from the prebuilt archive when it has it built from the same
    // inputs (its 'stale' members are replaced), else compiled here.
    std::map<std::string, StdUnitObj> done;
    std::set<std::string> stale;   // prebuilt members compiled here instead
    std::set<std::string> allRefs = refs;
    bool compileHere = given.empty();
    auto process = [&](const std::string& unit) -> bool {
        if (done.count(unit)) return true;
        StdUnitObj u;
        auto h = pre.unitHash.find(unit);
        if (!pre.archive.empty() && h != pre.unitHash.end()) {
            if (h->second == stdUnitHash(idx, unit)) {
                u.skipped = pre.skipped.count(unit) > 0;
                u.refs = pre.refs[unit];
                done[unit] = u;
                allRefs.insert(u.refs.begin(), u.refs.end());
                return true;
            }
            if (opts_.verbose)
                std::cout << "safeguard: std/" << unit << " changed since the prebuilt archive was built; built here instead\n";
            if (!pre.skipped.count(unit)) stale.insert(unit);
        } else if (!given.empty()) {
            auto r = pre.refs.find(unit);
            if (r != pre.refs.end()) u.refs = r->second;
        }
        if (!stdUnitObject(idx, unit, objDir.string(), compileHere, u)) return false;
        allRefs.insert(u.refs.begin(), u.refs.end());
        done[unit] = u;
        return true;
    };

    // The modules the build uses: the roots' closure, and the units the
    // code's references bring in ('runtime-symbol'), which have includes
    // and references of their own -- to a fixed point.
    std::set<std::string> rtUnits, units;
    for (;;) {
        std::set<std::string> r = roots;
        r.insert(rtUnits.begin(), rtUnits.end());
        units = idx.closure(r, plat);
        for (auto& u : units)
            if (!process(u)) return false;
        bool grew = false;
        for (auto& u : idx.runtimeUnits(allRefs, plat))
            if (rtUnits.insert(u).second) grew = true;
        if (!grew) break;
    }
    stdUnits_ = units;
    if (!link) return true;

    // The system libraries the modules call into.
    for (auto& f : idx.linkFlags(units, plat)) {
        if (f != "blas") { stdLinkFlags_.push_back(f); continue; }
        // The [build] blas choice, less what's already there ('-framework
        // Accelerate' from tensor_vdsp.sc, say) -- a framework as a pair.
        std::vector<std::string> blas = blasLinkFlags();
        for (size_t i = 0; i < blas.size(); ++i) {
            bool pair = blas[i] == "-framework" && i + 1 < blas.size();
            bool have = false;
            for (size_t j = 0; j < stdLinkFlags_.size(); ++j)
                if (stdLinkFlags_[j] == blas[i] &&
                    (!pair || (j + 1 < stdLinkFlags_.size() && stdLinkFlags_[j + 1] == blas[i + 1])))
                    have = true;
            if (!have) {
                stdLinkFlags_.push_back(blas[i]);
                if (pair) stdLinkFlags_.push_back(blas[i + 1]);
            }
            if (pair) ++i;
        }
    }
    stdDebDepends_ = idx.debDepends(units, plat);
    if (opts_.verbose) {
        std::cout << "safeguard: std modules (" << units.size() << "):";
        for (auto& u : units) std::cout << " " << u;
        std::cout << "\n";
    }
    if (!given.empty()) { stdArchive_ = given; return true; }
    // [build] std = "all": every unit, as one archive the linker picks from.
    if (manifest_.build.useStd == 2)
        for (auto& u : idx.allUnits(plat))
            if (!process(u)) return false;

    // The objects compiled here.
    std::vector<std::string> objs;
    bool rebuilt = false;
    for (auto& [unit, u] : done) {
        if (u.obj.empty()) continue;
        objs.push_back(u.obj);
        auto t = fs::last_write_time(u.obj, ec);
        if (!ec && fs::exists(libPath) && t > fs::last_write_time(libPath, ec)) rebuilt = true;
    }
    if (!pre.archive.empty() && objs.empty()) {
        // Everything is in the prebuilt archive.
        stdArchive_ = pre.archive;
        return true;
    }

    // One archive of them, repacked when its member list or a member
    // changes. With a prebuilt archive: a copy of it, its stale members
    // dropped and the ones compiled here added -- one archive, so no unit
    // is linked twice, and no archive order to get right.
    if (objs.empty()) return true;
    std::string members;
    if (!pre.archive.empty()) {
        members = "prebuilt=" + pre.archive + " " +
                  std::to_string((long long)fs::last_write_time(pre.archive, ec).time_since_epoch().count()) + "\n";
        for (auto& s : stale) members += "replaces=" + s + "\n";
    }
    for (auto& o : objs) members += o + "\n";
    fs::path membersPath = fs::path(libPath).concat(".members");
    std::stringstream prev;
    {
        std::ifstream in(membersPath);
        prev << in.rdbuf();
    }
    if (rebuilt || prev.str() != members || !fs::exists(libPath, ec)) {
        fs::remove(libPath, ec);   // ar adds to an existing archive
        if (!pre.archive.empty()) {
            fs::copy_file(pre.archive, libPath, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                std::cerr << "safeguard: cannot copy " << pre.archive << ": " << ec.message() << "\n";
                return false;
            }
            if (!stale.empty()) {
                std::string ar = findLlvmTool("llvm-ar");
                if (ar == "llvm-ar") ar = "ar";
                std::vector<std::string> cmd = { ar, "d", libPath.string() };
                for (auto& s : stale) {
                    std::string stem = s;
                    for (char& c : stem) if (c == '/' || c == '.') c = '_';
                    cmd.push_back(stem + ".o");
                }
                if (runCmd(cmd, opts_.verbose) != 0) {
                    std::cerr << "safeguard: ar d failed for " << libPath.string() << "\n";
                    return false;
                }
            }
        }
        if (!packArchive(objs, libPath.string())) return false;
        std::ofstream(membersPath) << members;
    }
    stdArchive_ = libPath.string();
    return true;
}

// ── public: std-prebuilt ─────────────────────────────────────────────────────

bool Builder::writePrebuiltStd(const std::string& prefix) {
    if (isCrossTarget()) {
        std::cerr << "safeguard: std-prebuilt builds for the host only (no --target)\n";
        return false;
    }
    std::string stdDir = findStdDir();
    if (stdDir.empty()) {
        std::cerr << "safeguard: cannot find SafeC std/ directory; set SAFEC_HOME\n";
        return false;
    }
    stdFromSource_ = true;
    writingPrebuiltStd_ = true;
    const StdIndex& idx = StdIndex::load(stdDir);
    StdPlatform plat = stdPlatform();
    plat.features.clear();
    fs::path out = fs::path(prefix) / "lib" / hostTriple();
    fs::create_directories(out);
    fs::path objDir = fs::path(root_) / "build" / "deps" / "std";
    std::error_code ec;
    fs::remove_all(objDir, ec);
    fs::create_directories(objDir);
    std::string stamp = prebuiltStdStamp();
    if (stamp.empty()) {
        std::cerr << "safeguard: cannot read the compiler " << findSafec() << "\n";
        return false;
    }
    std::vector<std::string> objs;
    std::ostringstream units;
    units << "# safeguard prebuilt std: 'unit <std file> <input hash>' (or 'skip ...': it doesn't\n"
             "# compile for this target), then the external symbols its IR refers to, 'ref <name>'\n";
    for (auto& unit : idx.allUnits(plat, /*featureBackends=*/false)) {
        StdUnitObj u;
        if (!stdUnitObject(idx, unit, objDir.string(), true, u)) return false;
        units << (u.skipped ? "skip " : "unit ") << unit << " " << stdUnitHash(idx, unit) << "\n";
        for (auto& r : u.refs) units << "ref " << r << "\n";
        if (!u.obj.empty()) objs.push_back(u.obj);
    }
    fs::path lib = out / "libsafec_std.a";
    fs::remove(lib, ec);
    if (objs.empty() || !packArchive(objs, lib.string())) return false;
    std::ofstream(out / "libsafec_std.units", std::ios::binary) << units.str();
    std::ofstream(out / "libsafec_std.stamp", std::ios::binary) << stamp;
    std::cout << "safeguard: wrote " << lib.string() << " (" << objs.size() << " std units)\n";
    return true;
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

// ── dependency layout ────────────────────────────────────────────────────────

std::string Builder::depDir(const Dependency& dep) const {
    if (!dep.path.empty()) {
        fs::path p(dep.path);
        return (p.is_absolute() ? p : fs::path(root_) / p).lexically_normal().string();
    }
    if (!dep.version.empty()) return (fs::path(root_) / "deps" / dep.name).string();
    return "";
}

// Library files for 'name' in a prebuilt library directory: lib<name> (or
// <name>.lib) in lib/, lib64/ or the directory itself, static first. A
// directory without one by that name contributes every library in the
// first of those that has any (static ones if there are), e.g. an OpenSSL
// install with libssl and libcrypto for a dependency named "openssl".
static std::vector<fs::path> findLibraries(const fs::path& dir, const std::string& name) {
    auto isStatic = [](const fs::path& f) {
        std::string e = f.extension().string();
        return e == ".a" || e == ".lib";
    };
    auto isShared = [](const fs::path& f) {
        std::string n = f.filename().string(), e = f.extension().string();
        return e == ".dylib" || e == ".so" || n.find(".so.") != std::string::npos;
    };
    for (const char* sub : { "lib", "lib64", "" }) {
        fs::path d = *sub ? dir / sub : dir;
        if (!fs::is_directory(d)) continue;
        for (const std::string& f : { "lib" + name + ".a", name + ".lib", "lib" + name + ".lib",
                                      "lib" + name + ".dylib", "lib" + name + ".so" })
            if (fs::exists(d / f)) return { d / f };
        std::vector<fs::path> statics, shareds;
        for (auto& e : fs::directory_iterator(d)) {
            if (!e.is_regular_file() && !e.is_symlink()) continue;
            if (isStatic(e.path())) statics.push_back(e.path());
            else if (isShared(e.path()) && e.path().filename().string().rfind("lib", 0) == 0)
                shareds.push_back(e.path());
        }
        std::sort(statics.begin(), statics.end());
        std::sort(shareds.begin(), shareds.end());
        if (!statics.empty()) return statics;
        if (!shareds.empty()) {
            // libfoo.so.1.2 next to libfoo.so: keep one name per library
            std::vector<fs::path> out;
            std::set<std::string> seen;
            for (auto& f : shareds) {
                std::string stem = f.filename().string();
                stem = stem.substr(0, stem.find('.'));
                if (seen.insert(stem).second) out.push_back(f);
            }
            return out;
        }
    }
    return {};
}

// '*' (any run of characters) and '?' (one) in a file name.
static bool wildcardMatch(const std::string& pat, const std::string& name) {
    size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pat.size() && (pat[p] == '?' || pat[p] == name[n])) { ++p; ++n; }
        else if (p < pat.size() && pat[p] == '*') { star = p++; mark = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++mark; }
        else return false;
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

static bool isNativeSource(const fs::path& f) {
    std::string e = f.extension().string();
    return e == ".c" || e == ".cc" || e == ".cpp" || e == ".cxx" || e == ".m" || e == ".mm";
}

// The source files of a build = "sources" dependency. Each 'sources' entry
// is a file, a directory (searched recursively) or a pattern with '*'/'?'
// in its last component, against the dependency's directory; with none,
// its src/ (recursively) if it has one, else the files directly in it.
// Only C/C++/Objective-C files count.
static std::vector<std::string> nativeSources(const Dependency& dep, const fs::path& dir,
                                              std::string& error) {
    std::set<std::string> out;
    auto addTree = [&](const fs::path& d) {
        for (auto& e : fs::recursive_directory_iterator(d))
            if (e.is_regular_file() && isNativeSource(e.path())) out.insert(e.path().lexically_normal().string());
    };
    if (dep.sources.empty()) {
        if (fs::is_directory(dir / "src")) addTree(dir / "src");
        else
            for (auto& e : fs::directory_iterator(dir))
                if (e.is_regular_file() && isNativeSource(e.path())) out.insert(e.path().lexically_normal().string());
    }
    for (auto& s : dep.sources) {
        fs::path p = fs::path(s).is_absolute() ? fs::path(s) : dir / s;
        std::string last = p.filename().string();
        if (last.find_first_of("*?") != std::string::npos) {
            size_t before = out.size();
            if (fs::is_directory(p.parent_path()))
                for (auto& e : fs::directory_iterator(p.parent_path()))
                    if (e.is_regular_file() && isNativeSource(e.path()) &&
                        wildcardMatch(last, e.path().filename().string()))
                        out.insert(e.path().lexically_normal().string());
            if (out.size() == before) { error = "no C/C++/ObjC file matches " + p.string(); return {}; }
        } else if (fs::is_directory(p)) {
            addTree(p);
        } else if (fs::is_regular_file(p)) {
            if (!isNativeSource(p)) {
                error = p.string() + " is not a C, C++ or Objective-C source (.c/.cc/.cpp/.cxx/.m/.mm)";
                return {};
            }
            out.insert(p.lexically_normal().string());
        } else {
            error = "source not found: " + p.string();
            return {};
        }
    }
    if (out.empty()) error = "no C/C++/ObjC sources in " + dir.string() + " (list them in 'sources')";
    return { out.begin(), out.end() };
}

Builder::DepLayout Builder::depLayout(const Dependency& dep) const {
    DepLayout l;
    l.dir = depDir(dep);
    fs::path base = l.dir.empty() ? fs::path(root_) : fs::path(l.dir);
    auto against = [&](const std::string& s, const fs::path& b) {
        fs::path p(s);
        return (p.is_absolute() ? p : b / p).lexically_normal();
    };
    std::vector<fs::path> libs;
    bool objc = false;
    std::vector<std::string> defaultIncludes;   // used when 'include' isn't given
    if (!dep.build.empty()) {
        // A C/C++/ObjC project safeguard builds itself (buildNativeDep) into
        // build/deps/<name>. Its library may not exist yet: fetchAndBuildDeps
        // builds it, then checks.
        if (!fs::is_directory(l.dir)) { l.error = "directory not found: " + l.dir; return l; }
        l.native = true;
        l.outDir = (fs::path(root_) / "build" / "deps" / dep.name).string();
        if (dep.build == "sources") {
            for (auto& src : nativeSources(dep, l.dir, l.error)) {
                SrcLang lang = langOf(src);
                if (isCxx(lang)) l.cxx = true;
                if (isObjC(lang)) objc = true;
            }
            if (!l.error.empty()) return l;
            libs.push_back(fs::path(l.outDir) / ("lib" + dep.name + ".a"));
            defaultIncludes.push_back(fs::is_directory(fs::path(l.dir) / "include")
                                      ? (fs::path(l.dir) / "include").string() : l.dir);
        } else {
            // What a CMake project compiles can't be told from outside, and
            // clang++ links plain C just as well.
            l.cxx = true;
            fs::path prefix = fs::path(l.outDir) / "install";
            if (!dep.libs.empty()) {
                for (auto& f : dep.libs) libs.push_back(against(f, prefix));
            } else {
                libs = findLibraries(prefix, dep.name);
                // No install() rules: the libraries the build tree holds.
                if (libs.empty() && fs::is_directory(fs::path(l.outDir) / "build"))
                    libs = findLibraries(fs::path(l.outDir) / "build", dep.name);
            }
            if (fs::is_directory(prefix / "include")) defaultIncludes.push_back((prefix / "include").string());
            else if (fs::is_directory(fs::path(l.dir) / "include"))
                defaultIncludes.push_back((fs::path(l.dir) / "include").string());
        }
    } else if (!dep.libs.empty()) {
        for (auto& f : dep.libs) {
            fs::path p = against(f, base);
            if (!fs::exists(p)) { l.error = "library not found: " + p.string(); return l; }
            libs.push_back(p);
        }
    } else if (l.dir.empty()) {
        l.error = "needs a 'version' (git URL), a 'path' or a 'lib'";
        return l;
    } else if (!fs::exists(l.dir)) {
        l.error = "directory not found: " + l.dir;
        return l;
    } else if (fs::is_directory(fs::path(l.dir) / "src")) {
        l.fromSource = true;
        fs::path inc = fs::path(l.dir) / "include";
        if (fs::exists(inc)) { l.safecIncludes.push_back(inc.string()); l.foreignIncludes.push_back(inc.string()); }
        l.safecIncludes.push_back((fs::path(l.dir) / "src").string());
        l.linkInputs.push_back((fs::path(root_) / "build" / "deps" / ("lib" + dep.name + ".a")).string());
        for (auto& src : collectAllSources((fs::path(l.dir) / "src").string())) {
            if (isCxx(langOf(src))) l.cxx = true;
            if (isObjC(langOf(src))) objc = true;
        }
        if (objc && targetOS() == "macos") l.linkInputs.push_back("-lobjc");
        for (auto& f : dep.frameworks)
            if (targetOS() == "macos") l.linkInputs.push_back("-Wl,-framework," + f);
        return l;
    } else {
        libs = findLibraries(l.dir, dep.name);
        if (libs.empty()) {
            l.error = "no src/ and no library (lib" + dep.name + ".a/.so/.dylib or any other) in " +
                      l.dir + ", lib/ or lib64/";
            return l;
        }
    }
    if (!l.dir.empty() && !l.native && fs::is_directory(fs::path(l.dir) / "include"))
        defaultIncludes.push_back((fs::path(l.dir) / "include").string());
    if (!dep.includes.empty()) {
        for (auto& i : dep.includes) {
            fs::path p = against(i, base);
            if (!fs::is_directory(p)) { l.error = "include directory not found: " + p.string(); return l; }
            l.safecIncludes.push_back(p.string());
            l.foreignIncludes.push_back(p.string());
        }
    } else {
        l.safecIncludes = defaultIncludes;
        l.foreignIncludes = defaultIncludes;
    }
    std::set<std::string> rpaths;
    for (auto& f : libs) {
        std::string e = f.extension().string();
        if (targetIsWindows()) {
            // A DLL is linked through its import library (<stem>.lib beside
            // it) and found at run time beside the program: copyRuntimeLibs
            // puts it there. An import library's DLL is <stem>.dll beside
            // it, or in ../bin/ (CMake's install layout).
            fs::path lib = f, dll;
            if (e == ".dll") {
                dll = f;
                lib = fs::path(f).replace_extension(".lib");
                if (!fs::exists(lib)) { l.error = "no import library " + lib.string() + " for " + f.string(); return l; }
            } else if (e == ".lib") {
                for (const fs::path& c : { fs::path(f).replace_extension(".dll"),
                                           f.parent_path().parent_path() / "bin" / fs::path(f).replace_extension(".dll").filename() })
                    if (fs::exists(c)) { dll = c; break; }
            }
            l.linkInputs.push_back(lib.string());
            if (!dll.empty()) l.runtimeLibs.push_back(dll.string());
            continue;
        }
        l.linkInputs.push_back(f.string());
        if (e != ".a" && e != ".lib") {
            l.runtimeLibs.push_back(f.string());
            if (rpaths.insert(f.parent_path().string()).second)
                l.linkInputs.push_back("-Wl,-rpath," + f.parent_path().string());   // find it at run time
        }
    }
    if (targetOS() == "macos") {
        if (objc) l.linkInputs.push_back("-lobjc");
        for (auto& f : dep.frameworks) l.linkInputs.push_back("-Wl,-framework," + f);
    }
    return l;
}

void Builder::depIncludes(std::vector<std::string>& safecIncs,
                          std::vector<std::string>& foreignIncs) const {
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) continue;   // reported by fetchAndBuildDeps
        safecIncs.insert(safecIncs.end(), l.safecIncludes.begin(), l.safecIncludes.end());
        foreignIncs.insert(foreignIncs.end(), l.foreignIncludes.begin(), l.foreignIncludes.end());
    }
}

std::vector<std::string> Builder::depLinkInputs() const {
    std::vector<std::string> out;
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) continue;
        for (auto& f : l.linkInputs)
            if (f.rfind("-", 0) == 0 || fs::exists(f)) out.push_back(f);
    }
    return out;
}

bool Builder::depsNeedCxx() const {
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (l.error.empty() && l.cxx) return true;
    }
    return false;
}

std::vector<std::string> Builder::depRuntimeLibs() const {
    std::vector<std::string> out;
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) continue;
        for (auto& f : l.runtimeLibs)
            if (fs::exists(f) && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    }
    return out;
}

bool Builder::copyRuntimeLibs(const std::string& dir) const {
    if (!targetIsWindows()) return true;   // an rpath finds them (see depLayout)
    for (auto& f : depRuntimeLibs()) {
        fs::path dest = fs::path(dir) / fs::path(f).filename();
        std::error_code ec;
        if (fs::equivalent(f, dest, ec)) continue;
        fs::copy_file(f, dest, fs::copy_options::update_existing, ec);
        if (ec) {
            std::cerr << "safeguard: could not copy " << f << " to " << dir << ": " << ec.message() << "\n";
            return false;
        }
    }
    return true;
}

// ── C/C++/ObjC dependencies built from source ────────────────────────────────

// The newest modification time of the files under 'dir', skipping .git and
// any directory in 'skip' (the project's own build/ and deps/, when the
// dependency directory contains them).
static fs::file_time_type newestFileTime(const fs::path& dir, const std::vector<fs::path>& skip) {
    fs::file_time_type newest = fs::file_time_type::min();
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_directory(ec)) {
            fs::path p = it->path().lexically_normal();
            bool skipped = it->path().filename() == ".git";
            for (auto& s : skip) if (fs::equivalent(p, s, ec)) skipped = true;
            ec.clear();
            if (skipped) it.disable_recursion_pending();
            continue;
        }
        auto t = it->last_write_time(ec);
        if (!ec && t > newest) newest = t;
        ec.clear();
    }
    return newest;
}

bool Builder::buildNativeDep(const Dependency& dep, const DepLayout& l) const {
    fs::path out(l.outDir);
    fs::create_directories(out);
    std::string cc = findSystemClang(false), cxx = findSystemClang(true);

    // Rebuilt when its settings (this stamp) or any file under its
    // directory changed since the last build.
    std::ostringstream cfg;
    cfg << "build=" << dep.build << "\ndir=" << l.dir << "\nrelease=" << opts_.release
        << "\ncc=" << cc << "\ncxx=" << cxx << "\n";
    auto list = [&](const char* key, const std::vector<std::string>& v) {
        cfg << key << "=";
        for (auto& x : v) cfg << x << ";";
        cfg << "\n";
    };
    list("sources", dep.sources);
    list("defines", dep.defines);
    list("cflags", dep.cflags);
    list("include", dep.includes);
    list("cmake_args", dep.cmakeArgs);
    list("lib", dep.libs);
    fs::path stampPath = out / "build.stamp";
    bool sameConfig = false;
    {
        std::ifstream in(stampPath);
        std::stringstream prev;
        prev << in.rdbuf();
        sameConfig = in && prev.str() == cfg.str();
    }
    std::error_code ec;
    if (sameConfig) {
        auto built = fs::last_write_time(stampPath, ec);
        auto newest = newestFileTime(l.dir, { fs::path(root_) / "build", fs::path(root_) / "deps" });
        if (!ec && newest <= built) {
            if (opts_.verbose) std::cout << "safeguard: dependency '" << dep.name << "' is up to date\n";
            return true;
        }
    }
    fs::remove(stampPath, ec);
    std::cout << "safeguard: building dependency '" << dep.name << "' (" << dep.build << ")\n";

    if (dep.build == "sources") {
        std::string err;
        auto srcs = nativeSources(dep, l.dir, err);
        if (!err.empty()) { std::cerr << "safeguard: dependency '" << dep.name << "': " << err << "\n"; return false; }
        std::vector<std::string> flags;
        for (auto& d : dep.defines) flags.push_back("-D" + d);
        // Its objects may end up in a shared library (a cdylib dependent).
        if (!targetIsWindows()) flags.push_back("-fPIC");
        for (auto& f : dep.cflags) flags.push_back(f);
        fs::path objDir = out / "obj";
        fs::remove_all(objDir, ec);
        fs::create_directories(objDir);
        std::vector<std::string> objs;
        for (auto& src : srcs) {
            std::string obj = compileForeign(src, objDir.string(), l.foreignIncludes, langOf(src), false, &flags);
            if (obj.empty()) return false;
            objs.push_back(obj);
        }
        fs::path lib = out / ("lib" + dep.name + ".a");
        fs::remove(lib, ec);
        if (!packArchive(objs, lib.string())) return false;
    } else {
        if (isCrossTarget()) {
            std::cerr << "safeguard: dependency '" << dep.name << "': build = \"cmake\" can't be "
                         "cross-compiled for --target yet; build it with a CMake toolchain file and "
                         "use it as a prebuilt dependency\n";
            return false;
        }
        fs::path buildDir = out / "build", prefix = out / "install";
        // A changed configuration starts from a fresh cache.
        if (!sameConfig) fs::remove_all(buildDir, ec);
        // A non-debug build type on Windows: a Debug one would pull in the
        // debug CRT, which doesn't mix with the release CRT the rest of the
        // program links (the shared one, MultiThreadedDLL — see
        // runtimeFlags()).
        std::string config = opts_.release ? "Release" : targetIsWindows() ? "RelWithDebInfo" : "Debug";
        if (!fs::exists(buildDir / "CMakeCache.txt")) {
            std::vector<std::string> conf = {
                "cmake", "-S", l.dir, "-B", buildDir.string(), "--no-warn-unused-cli",
                "-DCMAKE_BUILD_TYPE=" + config,
                "-DCMAKE_INSTALL_PREFIX=" + prefix.string(),
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DCMAKE_C_COMPILER=" + cc,
                "-DCMAKE_CXX_COMPILER=" + cxx,
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            };
            if (targetIsWindows()) {
                conf.push_back("-DCMAKE_POLICY_DEFAULT_CMP0091=NEW");
                conf.push_back("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL");   // as runtimeFlags()
            }
            // Ninja when there is one: the Visual Studio generators ignore
            // CMAKE_C_COMPILER, so on Windows it's what makes "the same
            // compiler" hold.
            if (runCmd({ "ninja", "--version" }, false, /*suppressOutput=*/true) == 0) {
                conf.push_back("-G");
                conf.push_back("Ninja");
            }
            for (auto& a : dep.cmakeArgs) conf.push_back(a);
            if (runCmd(conf, opts_.verbose) != 0) {
                std::cerr << "safeguard: dependency '" << dep.name << "': cmake configure failed\n";
                fs::remove_all(buildDir, ec);
                return false;
            }
        }
        if (runCmd({ "cmake", "--build", buildDir.string(), "--config", config, "--parallel" }, opts_.verbose) != 0) {
            std::cerr << "safeguard: dependency '" << dep.name << "': cmake build failed\n";
            return false;
        }
        // A fresh prefix, so a header or library the project no longer
        // installs doesn't linger.
        fs::remove_all(prefix, ec);
        if (runCmd({ "cmake", "--install", buildDir.string(), "--config", config }, opts_.verbose) != 0) {
            std::cerr << "safeguard: dependency '" << dep.name << "': cmake --install failed\n";
            return false;
        }
    }
    std::ofstream(stampPath) << cfg.str();
    return true;
}

// ── fetch and build all dependencies ─────────────────────────────────────────

bool Builder::fetchAndBuildDeps() {
    if (manifest_.dependencies.empty()) return true;

    fs::path depLibDir = fs::path(root_) / "build" / "deps";
    fs::create_directories(depLibDir);

    for (auto& dep : manifest_.dependencies) {
        if (dep.name.empty()) continue;

        if (dep.path.empty() && !dep.version.empty()) {
            // Remote dependency — dep.version holds the git URL
            fs::create_directories(fs::path(root_) / "deps");
            if (!gitClone(dep.version, depDir(dep))) {
                std::cerr << "safeguard: failed to fetch dependency '"
                          << dep.name << "' from " << dep.version << "\n";
                return false;
            }
        }
        DepLayout l = depLayout(dep);
        if (!l.error.empty()) {
            std::cerr << "safeguard: dependency '" << dep.name << "': " << l.error << "\n";
            return false;
        }
        if (l.native) {
            if (!buildNativeDep(dep, l)) {
                std::cerr << "safeguard: failed to build dependency '" << dep.name << "'\n";
                return false;
            }
            // Now that it's built, its libraries must be there.
            DepLayout built = depLayout(dep);
            bool any = false;
            for (auto& f : built.linkInputs) if (f.rfind("-", 0) != 0 && fs::exists(f)) any = true;
            if (!built.error.empty() || !any) {
                std::cerr << "safeguard: dependency '" << dep.name << "': "
                          << (!built.error.empty() ? built.error
                              : "built, but no library found in " + built.outDir +
                                "/install (lib/) or its build tree; name it with 'lib'") << "\n";
                return false;
            }
            std::cout << "safeguard: dependency '" << dep.name << "' ready (" << dep.build << ")\n";
            continue;
        }
        if (!l.fromSource) {
            std::cout << "safeguard: dependency '" << dep.name << "' ready (prebuilt)\n";
            continue;
        }

        // Build the dependency
        fs::path libOut = depLibDir / ("lib" + dep.name + ".a");
        if (!fs::exists(libOut) || !fs::exists(fs::path(libOut).concat(".refs"))) {
            fs::path depSrc = fs::path(l.dir) / "src";
            std::vector<std::string> incs = l.safecIncludes;
            // Also add std dir (SafeC-side only — see buildLib's
            // foreignIncludeDirs doc comment for why this must not reach
            // a real C/C++ compiler's search path).
            std::string stdDir = findStdDir();
            if (!stdDir.empty()) {
                incs.push_back(stdDir);
                incs.push_back(fs::path(stdDir).parent_path().string());
            }
            if (!buildLib(depSrc.string(), libOut.string(), incs,
                          /*compatPreprocessor=*/false, l.foreignIncludes)) {
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
        // Windows names a DLL <output>.dll, no "lib" prefix; its import
        // library is <output>.lib beside it (importLibPath).
        if (targetIsWindows())
            return (fs::path(root_) / "build" / (manifest_.build.output + ".dll")).string();
        const char* ext = targetOS() == "macos" ? ".dylib" : ".so";
        return (fs::path(root_) / "build" /
                ("lib" + manifest_.build.output + ext)).string();
    }
    // Windows: the .exe clang would add anyway (a cross link wouldn't), so
    // this names the real file.
    if (targetIsWindows())
        return (fs::path(root_) / "build" / (manifest_.build.output + ".exe")).string();
    return (fs::path(root_) / "build" / manifest_.build.output).string();
}

std::string Builder::exportedIncludeDir() const {
    return (fs::path(root_) / "build" / "include").string();
}

std::string Builder::importLibPath() const {
    return (fs::path(root_) / "build" / (manifest_.build.output + ".lib")).string();
}

// A DLL exports only what it is told to (SafeC has no dllexport), so a
// Windows cdylib exports every external function defined in the package's
// own object files — the same set a .so/.dylib exposes from them — listed
// in a .def file. std's and the dependencies' archives linked into it are
// left out, as are C++-mangled names ('?...': SafeC can't call those) and
// data (importing a variable from a DLL needs __declspec(dllimport) in the
// consumer, which SafeC can't express; export an accessor function).
std::string Builder::writeExportsDef(const std::vector<std::string>& objFiles) const {
    std::vector<std::string> cmd = { findLlvmTool("llvm-nm"), "--defined-only", "--extern-only", "-P" };
    for (auto& o : objFiles) cmd.push_back(o);
    std::string out;
    if (!captureCmd(cmd, out, opts_.verbose)) {
        std::cerr << "safeguard: llvm-nm failed listing the DLL's exports (it ships with LLVM, "
                     "beside clang; put it on PATH)\n";
        return "!";
    }
    // -P lines: "<name> <type> <value> <size>"; "<file>:" headers between objects.
    std::set<std::string> names;
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        std::istringstream ls(line);
        std::string name, type;
        if (!(ls >> name >> type)) continue;
        if (type == "T" && name[0] != '?') names.insert(name);
    }
    if (names.empty()) return "";
    fs::path def = fs::path(root_) / "build" / (manifest_.build.output + ".def");
    std::ofstream f(def);
    f << "EXPORTS\n";
    for (auto& n : names) f << "    " << n << "\n";
    if (!f) {
        std::cerr << "safeguard: cannot write " << def.string() << "\n";
        return "!";
    }
    return def.string();
}

// The public headers of a library build, copied into build/include with
// their paths kept: everything under the project's include/, and the
// headers (.h, and C++'s .hh/.hpp/.hxx) under src/ (both are what a dependent building this package from
// source gets on its include path -- see depLayout). The directory is
// rebuilt each time, so a header deleted from the sources leaves it too.
// A src/ header with the same path as an include/ one is an error: a
// dependent would see one or the other depending on the search order.
static bool exportHeaders(const fs::path& root, const fs::path& dest) {
    std::error_code ec;
    fs::remove_all(dest, ec);
    std::map<std::string, fs::path> files;   // relative path -> source
    auto collect = [&](const fs::path& dir, bool headersOnly) {
        if (!fs::is_directory(dir)) return true;
        for (auto& e : fs::recursive_directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            if (headersOnly && ext != ".h" && ext != ".hh" && ext != ".hpp" && ext != ".hxx") continue;
            std::string rel = fs::relative(e.path(), dir).generic_string();
            auto [it, fresh] = files.emplace(rel, e.path().lexically_normal());
            if (!fresh) {
                std::cerr << "safeguard: header '" << rel << "' is both " << it->second.string()
                          << " and " << e.path().lexically_normal().string() << "; a library exports one header per path\n";
                return false;
            }
        }
        return true;
    };
    if (!collect(root / "include", false) || !collect(root / "src", true)) return false;
    for (auto& [rel, src] : files) {
        fs::path out = dest / rel;
        fs::create_directories(out.parent_path());
        fs::copy_file(src, out, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "safeguard: could not export " << src.string() << ": " << ec.message() << "\n";
            return false;
        }
    }
    return true;
}

bool Builder::build() {
    std::string safecBin = findSafec();
    std::string buildDir = (fs::path(root_) / "build").string();
    fs::create_directories(buildDir);

    // 0. Verify lock file (aborts if dep SHAs mismatch).
    if (!checkLock()) return false;

    // 1. Fetch + build dependencies
    if (!fetchAndBuildDeps()) return false;

    // The sources: [build] srcs, or everything under src/.
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
        std::cerr << "safeguard: no .sc/.c/.cpp/.m/.mm source files found under src/\n";
        return false;
    }

    // 2. Whether std is built (step 6) and linked. A package with no SafeC
    // in it (its own sources or a SafeC source dependency) — a plain C,
    // C++ or Objective-C one — doesn't use std, and neither builds nor
    // links it.
    bool usesStd = std::any_of(srcs.begin(), srcs.end(),
                               [](const std::string& f) { return langOf(f) == SrcLang::SafeC; });
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (l.error.empty() && l.fromSource) usesStd = true;
    }
    // [build] std overrides: a C/C++/ObjC package calling into std sets it.
    if (manifest_.build.useStd >= 0) usesStd = manifest_.build.useStd != 0;
    std::string stdDir  = findStdDir();
    // Only the std modules the sources (and dependencies) include, and the
    // ones their compiled code calls into (ensureStdLib, after step 5).
    std::set<std::string> stdRootSet;
    if (usesStd && !opts_.emitLLVM) stdRootSet = stdRoots(srcs);

    // 3. Determine include paths for user sources
    std::vector<std::string> userIncs;
    if (!stdDir.empty()) {
        userIncs.push_back(stdDir);
        userIncs.push_back(fs::path(stdDir).parent_path().string());
    }
    // Each dependency's include/ (and a source package's src/) — added
    // below together with the foreign list.

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
    if (fs::exists(projInc)) {
        // Both languages: a library's exported headers (see exportHeaders)
        // are included by its own SafeC sources too.
        userIncs.push_back(projInc.string());
        foreignIncs.push_back(projInc.string());
    }
    stdForeignInclude(foreignIncs, stdDir);
    depIncludes(userIncs, foreignIncs);

    // 4. Compile user sources. Mixed SafeC/C/C++/ObjC — each file is
    // dispatched by extension and compiled to its own .o (SafeC via safec
    // --emit-llvm + clang -c; the others straight to .o via clang/clang++),
    // never merged into a shared translation unit, so per-file
    // recompilation stays granular the same way it already was for
    // pure-SafeC projects. C++ anywhere in the link (its own sources or a
    // dependency's) makes clang++ the link driver; Objective-C links the
    // ObjC runtime.
    bool hasCpp = depsNeedCxx(), hasObjC = false;
    for (auto& src : srcs) {
        if (isCxx(langOf(src))) hasCpp = true;
        if (isObjC(langOf(src))) hasObjC = true;
    }

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
    std::set<std::string> refs;   // the external symbols of their IR (see ensureStdLib)
    for (auto& src : srcs) {
        SrcLang lang = langOf(src);
        if (lang == SrcLang::SafeC) {
            std::string realSrc = materializeScx(src, buildDir);
            if (realSrc.empty()) return false;
            std::string ll = compileSrc(safecBin, realSrc, buildDir, userIncs, false, false, defs);
            if (ll.empty()) return false;
            for (auto& r : irExternalRefs(ll)) refs.insert(r);
            std::string obj = llToObj(ll, buildDir);
            if (obj.empty()) return false;
            objFiles.push_back(obj);
        } else {
            std::string obj = compileForeign(src, buildDir, foreignIncs, lang);
            if (obj.empty()) return false;
            objFiles.push_back(obj);
        }
    }

    // 6. Build (or reuse) the std library. A missing std/ directory is only
    // a warning (see ensureStdLib), but if std/ was found and building it
    // failed, that's a real error — silently linking without it just
    // produces a wall of confusing "undefined symbol" errors instead.
    if (usesStd) {
        for (auto& r : depStdRefs()) refs.insert(r);
        if (!ensureStdLib(stdRootSet, refs)) {
            std::cerr << "safeguard: failed to build the standard library "
                         "(see compiler errors above) — aborting build\n";
            return false;
        }
    }
    std::vector<std::string> archives;
    for (auto& f : depLinkInputs()) archives.push_back(f);
    if (!stdArchive_.empty()) archives.push_back(stdArchive_);

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
        if (!exportHeaders(root_, exportedIncludeDir())) return false;
        // The std files it includes, beside it: a program linking this
        // library builds those std modules too (see stdRoots); and the
        // symbols its code refers to, for the runtime units (depStdRefs).
        if (!stdRootSet.empty()) {
            std::ofstream stdList(outputPath() + ".std");
            for (auto& r : stdRootSet) stdList << r << "\n";
        }
        if (usesStd) writeRefs(outputPath() + ".refs", refs);
    } else if (crateType == "cdylib") {
        // Windows: the exports are listed explicitly (writeExportsDef), and
        // the link writes the import library build/<output>.lib beside the
        // DLL.
        std::vector<std::string> extra;
        std::string def;
        if (targetIsWindows()) {
            def = writeExportsDef(objFiles);
            if (def == "!") return false;
            if (def.empty())
                std::cerr << "safeguard: warning: " << outputPath() << " exports no functions, so no "
                             "import library is written for it\n";
            else
                extra.push_back("-Wl,/DEF:" + def);
        } else if (targetOS() == "linux") {
            // A SONAME, so a dependent records "lib<output>.so" (found on its
            // rpath, or beside it in a bundle) rather than this path.
            extra.push_back("-Wl,-soname," + fs::path(outputPath()).filename().string());
        }
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp, /*shared=*/true, hasObjC, extra))
            return false;
        if (!def.empty() && !fs::exists(importLibPath())) {
            std::cerr << "safeguard: the link wrote no import library " << importLibPath() << "\n";
            return false;
        }
        if (!exportHeaders(root_, exportedIncludeDir())) return false;
        if (!copyRuntimeLibs(buildDir)) return false;
    } else {
        std::vector<std::string> extra;
        if (opts_.bundle && !bundleLinkFlags(extra)) return false;
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp, /*shared=*/false, hasObjC, extra))
            return false;
        if (!copyRuntimeLibs(buildDir)) return false;
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
    if (fs::exists(projInc)) {
        // Both languages: a library's exported headers (see exportHeaders)
        // are included by its own SafeC sources too.
        userIncs.push_back(projInc.string());
        foreignIncs.push_back(projInc.string());
    }
    stdForeignInclude(foreignIncs, stdDir);
    depIncludes(userIncs, foreignIncs);

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
        std::cerr << "safeguard: no .sc/.c/.cpp/.m/.mm source files found under src/\n";
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
            if (isObjC(lang) && targetOS() != "macos") {
                std::cerr << "safeguard: " << src << ": Objective-C sources (.m/.mm) build only for macOS\n";
                ok = false;
                continue;
            }
            std::vector<std::string> cmd = { findSystemClang(isCxx(lang)),
                                              "-fsyntax-only", src };
            for (auto& inc : foreignIncs) { cmd.push_back("-I"); cmd.push_back(inc); }
            if (isObjC(lang) && manifest_.build.objcArc) cmd.push_back("-fobjc-arc");
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
    auto testSrcs = collectAllSources(testsDir.string());
    if (testSrcs.empty()) {
        std::cout << "safeguard: tests/ has no .sc/.c/.cpp/.m/.mm files\n";
        return true;
    }
    // std is built and linked when SafeC is involved (see build()).
    bool usesStd = std::any_of(testSrcs.begin(), testSrcs.end(),
                               [](const std::string& f) { return langOf(f) == SrcLang::SafeC; });
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (l.error.empty() && l.fromSource) usesStd = true;
    }
    // [build] std overrides: a C/C++/ObjC package calling into std sets it.
    if (manifest_.build.useStd >= 0) usesStd = manifest_.build.useStd != 0;
    std::string stdDir = findStdDir();

    std::vector<std::string> userIncs;
    if (!stdDir.empty()) {
        userIncs.push_back(stdDir);
        userIncs.push_back(fs::path(stdDir).parent_path().string());
    }
    // C/C++/ObjC tests get the project's include/ and the dependencies'
    // headers, never std/ (see build()'s foreignIncs).
    std::vector<std::string> depForeignIncs;
    fs::path projInc = fs::path(root_) / "include";
    if (fs::exists(projInc)) {
        userIncs.push_back(projInc.string());
        depForeignIncs.push_back(projInc.string());
    }
    stdForeignInclude(depForeignIncs, stdDir);
    depIncludes(userIncs, depForeignIncs);
    bool depsCxx = depsNeedCxx();

    std::string safecBin = findSafec();
    fs::path buildDir = fs::path(root_) / "build" / "tests";
    fs::create_directories(buildDir);
    if (!copyRuntimeLibs(buildDir.string())) return false;

    std::vector<std::string> defs = featureDefines();

    // Compile every test first: the std modules they all include, and the
    // ones their code calls into, go into one archive the tests share.
    std::vector<std::string> testObjs;
    std::set<std::string> refs;
    for (auto& src : testSrcs) {
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
            if (!ll.empty()) {
                for (auto& r : irExternalRefs(ll)) refs.insert(r);
                obj = llToObj(ll, buildDir.string());
            }
        } else {
            obj = compileForeign(src, buildDir.string(), depForeignIncs, lang);
        }
        testObjs.push_back(obj);
    }
    if (usesStd) {
        for (auto& r : depStdRefs()) refs.insert(r);
        if (!ensureStdLib(stdRoots(testSrcs), refs)) {
            std::cerr << "safeguard: failed to build the standard library "
                         "(see compiler errors above) — aborting test run\n";
            return false;
        }
    }
    std::vector<std::string> archives;
    for (auto& f : depLinkInputs()) archives.push_back(f);
    if (!stdArchive_.empty()) archives.push_back(stdArchive_);

    int passed = 0, failed = 0;
    for (size_t ti = 0; ti < testSrcs.size(); ++ti) {
        const std::string& src = testSrcs[ti];
        std::string name = fs::path(src).stem().string();
        SrcLang lang = langOf(src);
        const std::string& obj = testObjs[ti];
        if (obj.empty()) {
            std::cout << "test " << name << " ... FAILED (build error)\n";
            ++failed;
            continue;
        }
        if (isCrossTarget()) {
            // Can't execute a foreign-ISA test binary here: compiling it
            // for the target (Sema, CodeGen and the assembler all ran) is
            // what a cross 'test' run checks.
            std::cout << "test " << name << " ... built for " << opts_.target << " (not run)\n";
            ++passed;
            continue;
        }
        std::string testBin = (buildDir / name).string();
        if (!linkFinal({obj}, archives, testBin, isCxx(lang) || depsCxx, /*shared=*/false, isObjC(lang))) {
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
    if (isCrossTarget()) {
        std::cerr << "safeguard: 'run' can't execute a --target " << opts_.target
                  << " binary on this host — use 'build' and run it on the target\n";
        return -1;
    }
    if (!build()) return -1;
    std::string out = outputPath();
    std::vector<std::string> cmd = { out };
    for (auto& a : args) cmd.push_back(a);
    return runCmd(cmd, opts_.verbose);
}

// ── public: bundle ───────────────────────────────────────────────────────────
// 'safeguard bundle' packages a bin target with its [bundle] settings:
//   macOS    build/<name>.app — Contents/MacOS/<exe>, Info.plist, Resources/
//            (icon, resources), Frameworks/ (the dependencies' dylibs),
//            code-signed (ad-hoc unless an identity is given)
//   Windows  build/<name>/ — <exe>.exe linked for the GUI subsystem with an
//            icon/version/manifest resource, the dependencies' DLLs, the
//            resources: a portable app folder
//   Linux    build/<pkg>_<version>_<arch>.deb (dpkg-deb) and the same tree
//            as build/<name>.AppDir

namespace {

// Width and height of a PNG (from its IHDR chunk); false if it isn't one.
bool pngSize(const fs::path& file, unsigned& w, unsigned& h) {
    std::ifstream in(file, std::ios::binary);
    unsigned char b[24];
    if (!in.read(reinterpret_cast<char*>(b), sizeof b)) return false;
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    if (std::memcmp(b, sig, 8) != 0 || std::memcmp(b + 12, "IHDR", 4) != 0) return false;
    w = (unsigned)b[16] << 24 | (unsigned)b[17] << 16 | (unsigned)b[18] << 8 | b[19];
    h = (unsigned)b[20] << 24 | (unsigned)b[21] << 16 | (unsigned)b[22] << 8 | b[23];
    return true;
}

// Copy a resource (a file, or a directory with its contents) into 'destDir'
// under its own name.
bool copyResource(const fs::path& src, const fs::path& destDir) {
    std::error_code ec;
    if (!fs::exists(src)) {
        std::cerr << "safeguard: bundle resource not found: " << src.string() << "\n";
        return false;
    }
    fs::create_directories(destDir, ec);
    fs::path dest = destDir / src.filename();
    if (fs::is_directory(src))
        fs::copy(src, dest, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    else
        fs::copy_file(src, dest, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "safeguard: could not copy " << src.string() << ": " << ec.message() << "\n";
        return false;
    }
    return true;
}

std::string xmlEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '"') out += "&quot;";
        else out += c;
    }
    return out;
}

// The numeric parts of a version ("1.2.3-beta" -> 1, 2, 3, 0) — Windows'
// version resource and manifest want four numbers.
std::vector<unsigned> versionParts(const std::string& v) {
    std::vector<unsigned> parts;
    size_t i = 0;
    while (parts.size() < 4 && i < v.size() && std::isdigit((unsigned char)v[i])) {
        unsigned n = 0;
        while (i < v.size() && std::isdigit((unsigned char)v[i])) n = n * 10 + (unsigned)(v[i++] - '0');
        parts.push_back(n > 65535 ? 65535 : n);
        if (i < v.size() && v[i] == '.') ++i; else break;
    }
    while (parts.size() < 4) parts.push_back(0);
    return parts;
}

// [bundle] category: a freedesktop main category ("Utility", ...) or a
// macOS "public.app-category.*" type, translated for the other platform
// where there's a counterpart.
const std::pair<const char*, const char*> kCategories[] = {
    { "Utility",     "public.app-category.utilities" },
    { "Development", "public.app-category.developer-tools" },
    { "Game",        "public.app-category.games" },
    { "Graphics",    "public.app-category.graphics-design" },
    { "Education",   "public.app-category.education" },
    { "Office",      "public.app-category.productivity" },
    { "AudioVideo",  "public.app-category.music" },
    { "Audio",       "public.app-category.music" },
    { "Video",       "public.app-category.video" },
    { "Network",     "public.app-category.social-networking" },
    { "Science",     "public.app-category.education" },
    { "Finance",     "public.app-category.finance" },
};

std::string macCategory(const std::string& c) {
    if (c.rfind("public.app-category.", 0) == 0) return c;
    for (auto& [fd, mac] : kCategories) if (c == fd) return mac;
    return "";
}

std::string desktopCategory(const std::string& c) {
    if (c.rfind("public.app-category.", 0) != 0) return c.empty() ? "Utility" : c;
    for (auto& [fd, mac] : kCategories) if (c == mac) return fd;
    return "Utility";
}

// A Debian package name: lowercase letters, digits, '+', '-', '.'.
std::string debName(const std::string& name) {
    std::string out;
    for (char c : name) {
        char l = (char)std::tolower((unsigned char)c);
        out += (std::isalnum((unsigned char)l) || l == '+' || l == '.' || l == '-') ? l : '-';
    }
    return out;
}

std::string trimmed(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    size_t i = 0;
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    return s.substr(i);
}

} // namespace

bool Builder::bundleLinkFlags(std::vector<std::string>& out) const {
    std::string os = targetOS();
    if (os == "macos") {
        // The bundled dylibs (bundleMacOS) are found through this rpath.
        out.push_back("-Wl,-rpath,@executable_path/../Frameworks");
    } else if (os == "linux") {
        // /usr/lib/<pkg>/ in the .deb, and the same place in the AppDir.
        out.push_back("-Wl,-rpath,$ORIGIN/../lib/" + debName(manifest_.package.name));
    } else if (os == "windows") {
        std::string res = windowsResource();
        if (res.empty()) return false;
        out.push_back(res);
        out.push_back("-Wl,/MANIFEST:NO");   // the resource carries the manifest
        if (manifest_.bundle.subsystem == "windows") {
            // No console window; mainCRTStartup still calls a plain main().
            out.push_back("-Wl,/SUBSYSTEM:WINDOWS");
            out.push_back("-Wl,/ENTRY:mainCRTStartup");
        }
    }
    return true;
}

std::string Builder::windowsResource() const {
    const BundleConfig& b = manifest_.bundle;
    fs::path dir = fs::path(root_) / "build" / "bundle";
    fs::create_directories(dir);
    std::vector<unsigned> v = versionParts(b.version);
    std::string v4 = std::to_string(v[0]) + "." + std::to_string(v[1]) + "." +
                     std::to_string(v[2]) + "." + std::to_string(v[3]);

    // The icon: an .ico as is, a .png wrapped in a one-image .ico (Windows
    // reads PNG-compressed icon images, up to 256x256).
    fs::path ico;
    if (!b.icon.empty()) {
        fs::path icon = fs::path(root_) / b.icon;
        std::string ext = icon.extension().string();
        unsigned w = 0, h = 0;
        if (!fs::exists(icon)) {
            std::cerr << "safeguard: bundle icon not found: " << icon.string() << "\n";
            return "";
        }
        if (ext == ".ico") {
            ico = icon;
        } else if (ext == ".png") {
            if (!pngSize(icon, w, h) || w > 256 || h > 256) {
                std::cerr << "safeguard: " << icon.string() << ": a Windows icon needs a PNG of at most "
                             "256x256 (or give an .ico)\n";
                return "";
            }
            std::ifstream in(icon, std::ios::binary);
            std::string png((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            ico = dir / "app.ico";
            std::ofstream out(ico, std::ios::binary);
            auto u16 = [&](unsigned x) { out.put((char)(x & 0xff)); out.put((char)(x >> 8 & 0xff)); };
            auto u32 = [&](unsigned long x) { u16((unsigned)(x & 0xffff)); u16((unsigned)(x >> 16 & 0xffff)); };
            u16(0); u16(1); u16(1);                              // ICONDIR: an icon, one image
            out.put((char)(w == 256 ? 0 : w)); out.put((char)(h == 256 ? 0 : h));
            out.put(0); out.put(0);                              // no palette, reserved
            u16(1); u16(32);                                     // planes, bits per pixel
            u32((unsigned long)png.size()); u32(6 + 16);         // image size, offset
            out << png;
        } else {
            std::cerr << "safeguard: warning: " << icon.string() << ": Windows can't use an .icns icon "
                         "(give a .png or .ico); the executable gets none\n";
        }
    }

    // The application manifest: per-monitor DPI awareness, common controls
    // v6 (themed controls), and no elevation.
    fs::path manifest = dir / "app.manifest";
    std::ofstream(manifest) <<
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">\n"
        "  <assemblyIdentity type=\"win32\" name=\"" << xmlEscape(b.identifier) << "\" version=\"" << v4 << "\"/>\n"
        "  <dependency>\n"
        "    <dependentAssembly>\n"
        "      <assemblyIdentity type=\"win32\" name=\"Microsoft.Windows.Common-Controls\" version=\"6.0.0.0\"\n"
        "                        processorArchitecture=\"*\" publicKeyToken=\"6595b64144ccf1df\" language=\"*\"/>\n"
        "    </dependentAssembly>\n"
        "  </dependency>\n"
        "  <application xmlns=\"urn:schemas-microsoft-com:asm.v3\">\n"
        "    <windowsSettings>\n"
        "      <dpiAware xmlns=\"http://schemas.microsoft.com/SMI/2005/WindowsSettings\">true/pm</dpiAware>\n"
        "      <dpiAwareness xmlns=\"http://schemas.microsoft.com/SMI/2016/WindowsSettings\">PerMonitorV2</dpiAwareness>\n"
        "    </windowsSettings>\n"
        "  </application>\n"
        "  <compatibility xmlns=\"urn:schemas-microsoft-com:compatibility.v1\">\n"
        "    <application>\n"
        "      <supportedOS Id=\"{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}\"/>\n"   // Windows 10/11
        "    </application>\n"
        "  </compatibility>\n"
        "  <trustInfo xmlns=\"urn:schemas-microsoft-com:asm.v3\">\n"
        "    <security>\n"
        "      <requestedPrivileges>\n"
        "        <requestedExecutionLevel level=\"asInvoker\" uiAccess=\"false\"/>\n"
        "      </requestedPrivileges>\n"
        "    </security>\n"
        "  </trustInfo>\n"
        "</assembly>\n";

    // The .rc: numeric constants only (no <windows.h>), so it needs no
    // preprocessor. Strings double their quotes.
    auto rcStr = [](const std::string& s) {
        std::string out = "\"";
        for (char c : s) out += c == '"' ? std::string("\"\"") : std::string(1, c);
        return out + "\"";
    };
    std::string exe = manifest_.build.output + ".exe";
    std::string vc = std::to_string(v[0]) + "," + std::to_string(v[1]) + "," +
                     std::to_string(v[2]) + "," + std::to_string(v[3]);
    fs::path rc = dir / "app.rc";
    {
        std::ofstream f(rc);
        if (!ico.empty()) f << "1 ICON " << rcStr(fs::absolute(ico).generic_string()) << "\n";
        f << "1 24 " << rcStr(fs::absolute(manifest).generic_string()) << "\n"   // RT_MANIFEST
          << "1 VERSIONINFO\n"
          << "FILEVERSION " << vc << "\n"
          << "PRODUCTVERSION " << vc << "\n"
          << "FILEFLAGSMASK 0x3f\nFILEFLAGS 0x0\nFILEOS 0x40004\nFILETYPE 0x1\nFILESUBTYPE 0x0\n"
          << "BEGIN\n"
          << "  BLOCK \"StringFileInfo\"\n  BEGIN\n    BLOCK \"040904b0\"\n    BEGIN\n";
        auto val = [&](const char* k, const std::string& s) {
            if (!s.empty()) f << "      VALUE \"" << k << "\", " << rcStr(s) << "\n";
        };
        val("CompanyName", manifest_.package.author);
        val("FileDescription", manifest_.package.description.empty() ? b.name : manifest_.package.description);
        val("FileVersion", b.version);
        val("InternalName", manifest_.build.output);
        val("LegalCopyright", b.copyright);
        val("OriginalFilename", exe);
        val("ProductName", b.name);
        val("ProductVersion", b.version);
        f << "    END\n  END\n"
          << "  BLOCK \"VarFileInfo\"\n  BEGIN\n    VALUE \"Translation\", 0x409, 1200\n  END\n"
          << "END\n";
    }

    // llvm-rc (beside clang), or the Windows SDK's rc.
    fs::path res = dir / (manifest_.build.output + ".res");
    std::error_code ec;
    fs::remove(res, ec);
    std::string llvmRc = findLlvmTool("llvm-rc");
    int rcCode = runCmd({ llvmRc, "/no-preprocess", "/FO", res.string(), rc.string() }, opts_.verbose,
                        /*suppressOutput=*/llvmRc == "llvm-rc");
    if (rcCode != 0 || !fs::exists(res)) {
        rcCode = runCmd({ "rc", "/nologo", "/fo", res.string(), rc.string() }, opts_.verbose);
        if (rcCode != 0 || !fs::exists(res)) {
            std::cerr << "safeguard: could not compile " << rc.string() << " (needs llvm-rc, which "
                         "ships with LLVM, or the Windows SDK's rc)\n";
            return "";
        }
    }
    return res.string();
}

bool Builder::bundleMacOS() const {
    const BundleConfig& b = manifest_.bundle;
    fs::path build = fs::path(root_) / "build";
    fs::path app = build / (b.name + ".app");
    fs::path contents = app / "Contents";
    std::error_code ec;
    fs::remove_all(app, ec);
    fs::create_directories(contents / "MacOS");
    fs::create_directories(contents / "Resources");
    std::string exeName = manifest_.build.output;
    fs::path exe = contents / "MacOS" / exeName;
    fs::copy_file(outputPath(), exe, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "safeguard: could not copy " << outputPath() << ": " << ec.message() << "\n";
        return false;
    }

    // The icon: an .icns as is; a .png turned into one (sips scales it to
    // each size of an .iconset, iconutil packs it).
    std::string iconFile;
    if (!b.icon.empty()) {
        fs::path icon = fs::path(root_) / b.icon;
        std::string ext = icon.extension().string();
        if (!fs::exists(icon)) {
            std::cerr << "safeguard: bundle icon not found: " << icon.string() << "\n";
            return false;
        }
        iconFile = icon.stem().string() + ".icns";
        fs::path icns = contents / "Resources" / iconFile;
        if (ext == ".icns") {
            fs::copy_file(icon, icns, fs::copy_options::overwrite_existing, ec);
        } else if (ext == ".png") {
            fs::path set = build / "bundle" / (icon.stem().string() + ".iconset");
            fs::remove_all(set, ec);
            fs::create_directories(set);
            for (int size : { 16, 32, 128, 256, 512 }) {
                for (int scale : { 1, 2 }) {
                    std::string px = std::to_string(size * scale);
                    std::string name = "icon_" + std::to_string(size) + "x" + std::to_string(size) +
                                       (scale == 2 ? "@2x" : "") + ".png";
                    if (runCmd({ "sips", "-z", px, px, icon.string(), "--out", (set / name).string() },
                               opts_.verbose, /*suppressOutput=*/true) != 0) {
                        std::cerr << "safeguard: sips could not scale " << icon.string() << "\n";
                        return false;
                    }
                }
            }
            if (runCmd({ "iconutil", "-c", "icns", set.string(), "-o", icns.string() }, opts_.verbose) != 0) {
                std::cerr << "safeguard: iconutil could not make " << icns.string() << "\n";
                return false;
            }
        } else {
            std::cerr << "safeguard: warning: " << icon.string() << ": macOS can't use an .ico icon "
                         "(give a .png or .icns); the bundle gets none\n";
            iconFile.clear();
        }
    }
    for (auto& r : b.resources)
        if (!copyResource(fs::path(root_) / r, contents / "Resources")) return false;

    {
        std::ofstream f(contents / "Info.plist");
        auto key = [&](const char* k, const std::string& v) {
            f << "\t<key>" << k << "</key>\n\t<string>" << xmlEscape(v) << "</string>\n";
        };
        f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
             "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
             "<plist version=\"1.0\">\n<dict>\n";
        key("CFBundleDevelopmentRegion", "en");
        key("CFBundleExecutable", exeName);
        key("CFBundleIdentifier", b.identifier);
        key("CFBundleInfoDictionaryVersion", "6.0");
        key("CFBundleName", b.name);
        key("CFBundleDisplayName", b.name);
        key("CFBundlePackageType", "APPL");
        key("CFBundleShortVersionString", b.version);
        key("CFBundleVersion", b.version);
        if (!iconFile.empty()) key("CFBundleIconFile", iconFile);
        key("LSMinimumSystemVersion", b.minSystemVersion);
        if (!macCategory(b.category).empty()) key("LSApplicationCategoryType", macCategory(b.category));
        if (!b.copyright.empty()) key("NSHumanReadableCopyright", b.copyright);
        f << "\t<key>NSHighResolutionCapable</key>\n\t<true/>\n"
             "\t<key>NSPrincipalClass</key>\n\t<string>NSApplication</string>\n"
             "</dict>\n</plist>\n";
    }
    std::ofstream(contents / "PkgInfo") << "APPL????";

    // The dependencies' dylibs go into Contents/Frameworks; each one's
    // install name becomes @rpath/<file> (in it and in the executable),
    // and the executable's rpath (bundleLinkFlags) points there.
    std::vector<fs::path> frameworks;
    for (auto& lib : depRuntimeLibs()) {
        if (fs::path(lib).extension() != ".dylib") continue;
        fs::create_directories(contents / "Frameworks");
        fs::path dest = contents / "Frameworks" / fs::path(lib).filename();
        fs::copy_file(lib, dest, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "safeguard: could not copy " << lib << ": " << ec.message() << "\n";
            return false;
        }
        std::string out;
        if (!captureCmd({ "otool", "-D", dest.string() }, out)) {
            std::cerr << "safeguard: otool -D failed for " << dest.string() << "\n";
            return false;
        }
        // "<file>:\n<install name>\n"
        std::string id = trimmed(out.substr(out.find('\n') == std::string::npos ? out.size() : out.find('\n') + 1));
        std::string rpathId = "@rpath/" + dest.filename().string();
        if (runCmd({ "install_name_tool", "-id", rpathId, dest.string() }, opts_.verbose) != 0 ||
            (!id.empty() && id != rpathId &&
             runCmd({ "install_name_tool", "-change", id, rpathId, exe.string() }, opts_.verbose) != 0)) {
            std::cerr << "safeguard: install_name_tool failed for " << dest.string() << "\n";
            return false;
        }
        frameworks.push_back(dest);
    }

    // Sign the nested code first, then the bundle (install_name_tool has
    // invalidated the linker's ad-hoc signatures).
    std::string identity = !opts_.signIdentity.empty() ? opts_.signIdentity : b.codesignIdentity;
    if (identity != "none") {
        for (auto& f : frameworks)
            if (runCmd({ "codesign", "--force", "--sign", identity, f.string() }, opts_.verbose) != 0) {
                std::cerr << "safeguard: codesign failed for " << f.string() << "\n";
                return false;
            }
        if (runCmd({ "codesign", "--force", "--sign", identity, app.string() }, opts_.verbose) != 0) {
            std::cerr << "safeguard: codesign failed for " << app.string() << "\n";
            return false;
        }
    }
    std::cout << "safeguard: bundled " << app.lexically_normal().string()
              << (identity == "none" ? " (unsigned)" : identity == "-" ? " (ad-hoc signed)" : " (signed)") << "\n";
    return true;
}

bool Builder::bundleWindows() const {
    const BundleConfig& b = manifest_.bundle;
    fs::path dir = fs::path(root_) / "build" / b.name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    fs::path exe = outputPath();
    fs::copy_file(exe, dir / exe.filename(), fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "safeguard: could not copy " << exe.string() << ": " << ec.message() << "\n";
        return false;
    }
    for (auto& lib : depRuntimeLibs())
        if (fs::path(lib).extension() == ".dll" && !copyResource(lib, dir)) return false;
    for (auto& r : b.resources)
        if (!copyResource(fs::path(root_) / r, dir)) return false;
    std::cout << "safeguard: bundled " << dir.lexically_normal().string() << "\n";
    return true;
}

bool Builder::bundleLinux() const {
    const BundleConfig& b = manifest_.bundle;
    const std::string pkg = debName(manifest_.package.name);
    const std::string exeName = manifest_.build.output;
    fs::path build = fs::path(root_) / "build";
    std::error_code ec;

    // The Debian architecture: dpkg's own for the host, else the target's.
    std::string arch, dpkgArch;
    bool haveDpkg = captureCmd({ "dpkg", "--print-architecture" }, dpkgArch);
    if (isCrossTarget()) {
        std::string t = targetTriple();
        std::string cpu = t.substr(0, t.find('-'));
        arch = cpu == "x86_64" ? "amd64" : cpu == "aarch64" ? "arm64"
             : (cpu.size() == 4 && cpu[0] == 'i' && cpu.substr(2) == "86") ? "i386"
             : cpu.rfind("armv7", 0) == 0 ? "armhf" : cpu;
    } else {
        arch = trimmed(dpkgArch);
    }
    if (arch.empty()) {
#if defined(__x86_64__)
        arch = "amd64";
#elif defined(__aarch64__)
        arch = "arm64";
#else
        arch = "unknown";
#endif
    }

    // The package tree: /usr/bin/<exe>, /usr/lib/<pkg>/ (the dependencies'
    // shared libraries, on the executable's rpath), /usr/share/applications/
    // <id>.desktop, /usr/share/icons/hicolor/<size>/apps/<id>.png and
    // /usr/share/<pkg>/ (resources).
    fs::path stage = build / "deb" / (pkg + "_" + b.version + "_" + arch);
    fs::remove_all(stage, ec);
    fs::path usr = stage / "usr";
    fs::create_directories(usr / "bin");
    fs::copy_file(outputPath(), usr / "bin" / exeName, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "safeguard: could not copy " << outputPath() << ": " << ec.message() << "\n";
        return false;
    }
    fs::permissions(usr / "bin" / exeName, fs::perms(0755), ec);
    for (auto& lib : depRuntimeLibs()) {
        // libfoo.so, and the libfoo.so.N... names beside it (the SONAME the
        // executable asks for is usually one of them), symlinks kept.
        fs::path f(lib);
        std::string stem = f.filename().string();
        stem = stem.substr(0, stem.find(".so") == std::string::npos ? stem.size() : stem.find(".so") + 3);
        fs::path dest = usr / "lib" / pkg;
        fs::create_directories(dest);
        for (auto& e : fs::directory_iterator(f.parent_path(), ec)) {
            std::string n = e.path().filename().string();
            if (n.rfind(stem, 0) != 0) continue;
            fs::remove(dest / n, ec);
            if (e.is_symlink()) fs::copy_symlink(e.path(), dest / n, ec);
            else fs::copy_file(e.path(), dest / n, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                std::cerr << "safeguard: could not copy " << e.path().string() << ": " << ec.message() << "\n";
                return false;
            }
        }
    }
    fs::path iconPng;
    if (!b.icon.empty()) {
        fs::path icon = fs::path(root_) / b.icon;
        unsigned w = 0, h = 0;
        if (!fs::exists(icon)) {
            std::cerr << "safeguard: bundle icon not found: " << icon.string() << "\n";
            return false;
        }
        if (icon.extension() == ".png" && pngSize(icon, w, h)) {
            fs::path dir = usr / "share" / "icons" / "hicolor" / (std::to_string(w) + "x" + std::to_string(h)) / "apps";
            fs::create_directories(dir);
            iconPng = dir / (b.identifier + ".png");
            fs::copy_file(icon, iconPng, fs::copy_options::overwrite_existing, ec);
        } else {
            std::cerr << "safeguard: warning: " << icon.string() << ": the Linux desktop entry needs a .png "
                         "icon; the package gets none\n";
        }
    }
    for (auto& r : b.resources)
        if (!copyResource(fs::path(root_) / r, usr / "share" / pkg)) return false;
    fs::path desktop = usr / "share" / "applications" / (b.identifier + ".desktop");
    fs::create_directories(desktop.parent_path());
    {
        std::ofstream f(desktop);
        f << "[Desktop Entry]\nType=Application\nName=" << b.name << "\nExec=" << exeName << "\n";
        if (!iconPng.empty()) f << "Icon=" << b.identifier << "\n";
        if (!manifest_.package.description.empty()) f << "Comment=" << manifest_.package.description << "\n";
        f << "Categories=" << desktopCategory(b.category) << ";\nTerminal=false\n";
    }

    // The AppDir: the same /usr tree plus AppRun, the desktop entry and the
    // icon at its top (what appimagetool packs into an AppImage).
    fs::path appDir = build / (b.name + ".AppDir");
    fs::remove_all(appDir, ec);
    fs::create_directories(appDir);
    fs::copy(usr, appDir / "usr", fs::copy_options::recursive | fs::copy_options::copy_symlinks, ec);
    fs::create_symlink("usr/bin/" + exeName, appDir / "AppRun", ec);
    fs::copy_file(desktop, appDir / desktop.filename(), ec);
    if (!iconPng.empty()) {
        fs::copy_file(iconPng, appDir / iconPng.filename(), ec);
        fs::create_symlink(iconPng.filename(), appDir / ".DirIcon", ec);
    }
    if (ec) {
        std::cerr << "safeguard: could not assemble " << appDir.string() << ": " << ec.message() << "\n";
        return false;
    }
    std::cout << "safeguard: bundled " << appDir.lexically_normal().string() << "\n";

    // DEBIAN/control, then dpkg-deb.
    if (!haveDpkg) {
        std::cerr << "safeguard: warning: dpkg not found; skipped the .deb (install dpkg to build one)\n";
        return true;
    }
    uintmax_t bytes = 0;
    for (auto& e : fs::recursive_directory_iterator(usr, ec))
        if (e.is_regular_file() && !e.is_symlink()) bytes += e.file_size();
    fs::create_directories(stage / "DEBIAN");
    {
        std::ofstream f(stage / "DEBIAN" / "control");
        f << "Package: " << pkg << "\n"
          << "Version: " << b.version << "\n"
          << "Architecture: " << arch << "\n"
          << "Maintainer: " << (manifest_.package.author.empty() ? "unknown" : manifest_.package.author) << "\n"
          << "Installed-Size: " << (bytes + 1023) / 1024 << "\n";
        // [bundle] depends, then the packages std's modules name for the
        // system libraries they link ('// safeguard: deb', e.g. libx11-6
        // for std/gui's X11 backend).
        std::vector<std::string> depends = b.depends;
        for (auto& d : stdDebDepends_) {
            bool listed = false;
            for (auto& have : depends)
                if (have.substr(0, have.find_first_of(" (|")) == d) listed = true;
            if (!listed) depends.push_back(d);
        }
        if (!depends.empty()) {
            f << "Depends: ";
            for (size_t i = 0; i < depends.size(); ++i) f << (i ? ", " : "") << depends[i];
            f << "\n";
        }
        f << "Section: " << (desktopCategory(b.category) == "Game" ? "games" : "misc") << "\n"
          << "Priority: optional\n"
          << "Description: " << (manifest_.package.description.empty() ? b.name : manifest_.package.description)
          << "\n";
    }
    // dpkg-deb insists on 0755 directories.
    fs::permissions(stage, fs::perms(0755), ec);
    for (auto& e : fs::recursive_directory_iterator(stage, ec)) {
        if (e.is_symlink()) continue;
        if (e.is_directory()) fs::permissions(e.path(), fs::perms(0755), ec);
        else if (e.path() != usr / "bin" / exeName) fs::permissions(e.path(), fs::perms(0644), ec);
    }
    fs::path deb = build / (pkg + "_" + b.version + "_" + arch + ".deb");
    fs::remove(deb, ec);
    if (runCmd({ "dpkg-deb", "--build", "--root-owner-group", stage.string(), deb.string() }, opts_.verbose) != 0) {
        std::cerr << "safeguard: dpkg-deb failed\n";
        return false;
    }
    std::cout << "safeguard: bundled " << deb.lexically_normal().string() << "\n";
    return true;
}

bool Builder::bundle() {
    if (!manifest_.bundle.present) {
        std::cerr << "safeguard: 'bundle' needs a [bundle] table in Package.toml (at least "
                     "identifier = \"org.example." << manifest_.package.name << "\")\n";
        return false;
    }
    if (manifest_.build.crateType != "bin") {
        std::cerr << "safeguard: 'bundle' packages an application: it needs build.crate_type = \"bin\"\n";
        return false;
    }
    std::string os = targetOS();
    if (os.empty()) {
        std::cerr << "safeguard: 'bundle' packages for macOS, Windows or Linux, not --target "
                  << opts_.target << "\n";
        return false;
    }
    opts_.bundle = true;
    if (!build()) return false;
    if (os == "macos") return bundleMacOS();
    if (os == "windows") return bundleWindows();
    return bundleLinux();
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
    for (auto& dep : manifest_.dependencies) {
        LockedDep ld;
        ld.name = dep.name;
        ld.url  = !dep.path.empty() ? dep.path : !dep.version.empty() ? dep.version
                : dep.libs.empty() ? "" : dep.libs.front();
        std::string dir = depDir(dep);
        if (!dir.empty() && fs::exists(dir)) {
            ld.git_sha = gitRevParse(dir);
        }
        lf.deps.push_back(ld);
    }

    // Hash every source file.
    for (auto& src : srcFiles) {
        std::string h = hashFile(src);
        if (!h.empty()) {
            lf.sources[sourceKey(src)] = h;
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
    for (auto& locked : lf.deps) {
        std::string depDir;
        // Find matching dep in manifest.
        for (auto& dep : manifest_.dependencies) {
            if (dep.name == locked.name) {
                depDir = this->depDir(dep);
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

// ── public: std-modules ───────────────────────────────────────────────────────

bool Builder::printStdModules(const std::vector<std::string>& args) {
    std::string stdDir = findStdDir();
    if (stdDir.empty()) {
        std::cerr << "safeguard: cannot find SafeC std/ directory; set SAFEC_HOME\n";
        return false;
    }
    const StdIndex& idx = StdIndex::load(stdDir);
    StdPlatform plat = stdPlatform();
    // Arguments: std files (std/x.h, <std/x.h>) are roots themselves, other
    // files are scanned; none: the project's sources, as 'build' does.
    std::set<std::string> roots;
    std::vector<std::string> files;
    for (auto& a : args) {
        std::string r = idx.resolve(a.size() > 2 && a[0] == '<' ? a.substr(1, a.size() - 2) : a);
        if (!r.empty()) roots.insert(r);
        else if (fs::is_regular_file(a)) files.push_back(fs::absolute(a).string());
        else {
            std::cerr << "safeguard: " << a << ": neither a std file (std/...) nor a source file\n";
            return false;
        }
    }
    if (args.empty()) {
        if (!manifest_.build.srcs.empty()) {
            for (auto& s : manifest_.build.srcs)
                files.push_back(fs::path(s).is_absolute() ? s : (fs::path(root_) / s).string());
        } else {
            files = collectAllSources((fs::path(root_) / "src").string());
        }
    }
    std::set<std::string> refs;
    if (!files.empty() || args.empty()) {
        for (auto& r : stdRoots(files)) roots.insert(r);
        // What their code calls into: compiled to IR as a build does.
        if (!sourceRefs(files, (fs::path(root_) / "build" / "std-modules").string(), refs)) return false;
        if (args.empty())
            for (auto& r : depStdRefs()) refs.insert(r);
    }
    // The modules, as a build works them out (compiling the std units
    // that takes, for the symbols their code refers to).
    if (!ensureStdLib(roots, refs, /*link=*/false)) return false;
    std::set<std::string> units = stdUnits_;
    for (auto& p : idx.problems) std::cout << "problem " << p << "\n";
    for (auto& u : units) std::cout << "unit std/" << u << "\n";
    std::vector<std::string> flags = idx.linkFlags(units, plat);
    for (size_t i = 0; i < flags.size(); ++i) {
        if (flags[i] == "-framework" && i + 1 < flags.size()) {
            std::cout << "link -framework " << flags[++i] << "\n";
            continue;
        }
        if (flags[i] != "blas") { std::cout << "link " << flags[i] << "\n"; continue; }
        std::string line;
        for (auto& b : blasLinkFlags()) line += " " + b;
        std::cout << "link" << (line.empty() ? " (none)" : line) << "   ([build] blas)\n";
    }
    for (auto& d : idx.debDepends(units, plat)) std::cout << "deb " << d << "\n";
    return true;
}

} // namespace safeguard
