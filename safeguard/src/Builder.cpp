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
    // What safeguard printed so far comes before the child's own output.
    std::cout.flush();
    fflush(stdout);
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
                               bool suppressOutput) const {
    fs::path obj = fs::path(buildDir) / (fs::path(llFile).stem().string() + ".o");
    std::vector<std::string> cmd = { findClang(false), "-c", llFile, "-o", obj.string() };
    for (auto& f : targetClangFlags()) cmd.push_back(f);
    cmd.push_back("-Wno-override-module");
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

    std::string compiler = findSystemClang(isCxx(lang));
    std::vector<std::string> cmd = { compiler, "-c", srcPath, "-o", obj.string() };
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

// ── CPU BLAS ─────────────────────────────────────────────────────────────────
// std/simd/gemm.sc calls cblas_sgemm through a weak reference, so a program
// gets the CPU vendor's BLAS for its matrix products (tensor_matmul,
// conv2d_forward, ...) just by linking it. [build] blas picks which:
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
                          bool objc) const {
    // A cross target links with the same LLVM clang that assembled the IR
    // (the system one may not know the target), through lld; bare metal
    // gets no libc/crt — the program's own runtime, linker script ('-T') and
    // libraries come from the manifest's [build] cflags/libs/lib_dirs.
    std::vector<std::string> cmd = { isCrossTarget() ? findClang(useCxxDriver)
                                                     : findSystemClang(useCxxDriver) };
    for (auto& f : targetClangFlags()) cmd.push_back(f);
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
#ifndef _WIN32
    // math functions live in a separate libm on macOS/Linux; on Windows
    // they're part of the standard CRT that's always linked, and there is
    // no 'm.lib' for clang's MSVC-style link.exe driver to find — passing
    // -lm there fails the link with LNK1181 ("cannot open input file
    // 'm.lib'") instead of being a harmless no-op.
    if (!targetIsBareMetal()) cmd.push_back("-lm"); // math library for std/math.sc
#endif
    const PlatformBuildConfig* plat = platformBuild();
    for (auto& d : manifest_.build.libDirs) cmd.push_back("-L" + d);
    if (plat) for (auto& d : plat->libDirs) cmd.push_back("-L" + d);
    for (auto& l : manifest_.build.libs)    cmd.push_back("-l" + l);
    if (plat) for (auto& l : plat->libs)    cmd.push_back("-l" + l);
    // Frameworks are a Mach-O notion: [build] frameworks apply on macOS and
    // are dropped (with a note) elsewhere; [build.macos] ones only reach here
    // on macOS anyway.
    std::vector<std::string> frameworks = manifest_.build.frameworks;
    if (plat) frameworks.insert(frameworks.end(), plat->frameworks.begin(), plat->frameworks.end());
    if (targetOS() == "macos") {
        for (auto& f : frameworks) { cmd.push_back("-framework"); cmd.push_back(f); }
        if (objc) cmd.push_back("-lobjc");
    } else if (!manifest_.build.frameworks.empty()) {
        static bool noted = false;
        if (!noted) {
            noted = true;
            std::cerr << "safeguard: note: [build] frameworks apply to macOS builds only; ignored here\n";
        }
    }
    // std's CPU BLAS (std/simd/gemm.sc) matters only when std is linked in —
    // a package with no SafeC sources at all doesn't link it (see build()).
    bool linksStd = std::any_of(archives.begin(), archives.end(), [](const std::string& a) {
        return fs::path(a).filename() == "libsafec_std.a";
    });
    if (linksStd) for (auto& f : blasLinkFlags()) cmd.push_back(f);
    // Extra link flags from manifest
    for (auto& f : compileFlags()) cmd.push_back(f);
    cmd.push_back("-o");
    cmd.push_back(output);
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: link failed\n";
        return false;
    }
    return true;
}

// ── platform-variant std sources ──────────────────────────────────────────────
// Several std modules ship one implementation per OS API, distinguished by a
// filename suffix (sched/io_nb_{bsd,linux,win32}.sc, sched/reactor_{kqueue,
// epoll,win32}.sc, ipc/uds_{bsd,linux}.sc, ipc/pipe_{posix,win32}.sc,
// gui/gui_{cocoa,x11,win32,fb}.sc — fb being the bare-metal one). They define the same symbols, so only the
// host's variant may go into libsafec_std.a: with all of them archived, the
// linker resolves each symbol from whichever member it meets first
// (alphabetically the BSD one, whose sockaddr layout is wrong on Linux) and
// then fails on the duplicates the next pulled-in member brings along.
static bool stdVariantMatches(const fs::path& src, const std::string& triple, bool bareMetal) {
    std::string stem = src.stem().string();
    auto endsWith = [&](const char* suf) {
        std::string s(suf);
        return stem.size() > s.size() &&
               stem.compare(stem.size() - s.size(), s.size(), s) == 0;
    };
    bool win, mac, linux_, bsd;
    if (triple.empty()) {
#if defined(_WIN32)
        win = true; mac = false; linux_ = false; bsd = false;
#elif defined(__APPLE__)
        win = false; mac = true; linux_ = false; bsd = false;
#elif defined(__linux__)
        win = false; mac = false; linux_ = true; bsd = false;
#else
        win = false; mac = false; linux_ = false; bsd = true;
#endif
    } else {
        win = triple.find("windows") != std::string::npos;
        mac = triple.find("apple") != std::string::npos || triple.find("darwin") != std::string::npos;
        linux_ = triple.find("linux") != std::string::npos;
        bsd = triple.find("bsd") != std::string::npos;
    }
    if (endsWith("_fb")) return bareMetal;   // bare-metal framebuffer backend (see gui_fb.sc)
    if (bareMetal) {
        // No OS API at all: none of the OS-specific backends apply.
        if (endsWith("_linux") || endsWith("_epoll") || endsWith("_bsd") || endsWith("_kqueue") ||
            endsWith("_cocoa") || endsWith("_x11") || endsWith("_win32") || endsWith("_posix"))
            return false;
        return true;
    }
    if (endsWith("_linux") || endsWith("_epoll")) return linux_;
    if (endsWith("_bsd") || endsWith("_kqueue")) return mac || bsd;
    if (endsWith("_cocoa")) return mac;
    if (endsWith("_x11")) return linux_ || bsd;
    if (endsWith("_win32")) return win;
    if (endsWith("_posix")) return !win;
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
        // tolerateArchMismatch is only set for the std library build.
        if (tolerateArchMismatch && !stdVariantMatches(src, targetTriple(), targetIsBareMetal())) continue;
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

    // One std archive per target (host builds keep the historical path).
    fs::path libPath = isCrossTarget()
        ? fs::path(root_) / "build" / "deps" / targetDirName() / "libsafec_std.a"
        : fs::path(root_) / "build" / "deps" / "libsafec_std.a";
    fs::create_directories(libPath.parent_path());

    // Reuse the archive unless a std source or the compiler changed since it
    // was built (an edited std file used to keep linking the old object).
    // The stamp file records which compiler and std/ built it: switching
    // either (e.g. SAFEC_HOME/SAFEC now naming a different, possibly older
    // safec) must rebuild too — an mtime comparison can't see that, and a
    // PATH-resolved 'safec' has no path to stat at all — or an archive
    // missing every file the other compiler rejected keeps being reused
    // with no "skipping" warnings printed.
    fs::path stampPath = libPath;
    stampPath += ".stamp";
    std::string stamp = "safec=" + findSafec() + "\nstd=" + stdDir + "\n";
    if (fs::exists(libPath)) {
        std::error_code ec;
        auto built = fs::last_write_time(libPath, ec);
        bool stale = false;
        for (auto it = fs::recursive_directory_iterator(stdDir, ec);
             !ec && !stale && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            auto ext = it->path().extension();
            if ((ext == ".sc" || ext == ".h") && it->last_write_time(ec) > built) stale = true;
        }
        fs::path safec = findSafec();
        if (!stale && fs::exists(safec, ec) && fs::last_write_time(safec, ec) > built) stale = true;
        if (!stale) {
            std::ifstream in(stampPath);
            std::stringstream prev;
            prev << in.rdbuf();
            if (!in || prev.str() != stamp) stale = true;
        }
        if (!stale) return libPath.string();
        if (opts_.verbose) std::cout << "safeguard: std library is out of date, rebuilding\n";
        fs::remove(libPath, ec);
    }

    if (opts_.verbose)
        std::cout << "safeguard: building std library from " << stdDir << "\n";

    // The std .sc files include their own headers; pass the std dir's parent
    // so that #include "std/io.h" resolves, and the std dir itself so that
    // #include "io.h" (relative includes inside .sc files) resolves.
    fs::path stdParent = fs::path(stdDir).parent_path();
    std::vector<std::string> incs = { stdDir, stdParent.string() };

    // 'wasm': std/wasm/*.sc's malloc/free/etc. are wasm-freestanding-only
    // — see collectAllSources's doc comment and std/wasm/wasm_rt.sc's own
    // header comment for the full story (a real, CI-confirmed Windows
    // LNK2005 symbol collision against libucrt otherwise). A wasm target's
    // archive keeps them: they are its runtime.
    std::vector<std::string> excluded;
    if (!targetIsWasm()) excluded.push_back("wasm");
    if (!buildLib(stdDir, libPath.string(), incs, /*compatPreprocessor=*/true,
                  /*tolerateArchMismatch=*/true, /*foreignIncludeDirs=*/{},
                  /*excludeSubdirNames=*/excluded))
        return "";
    std::ofstream(stampPath) << stamp;
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
        for (const std::string& f : { "lib" + name + ".a", name + ".lib",
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
        l.linkInputs.push_back(f.string());
#ifndef _WIN32
        std::string e = f.extension().string();
        if (e != ".a" && e != ".lib" && rpaths.insert(f.parent_path().string()).second)
            l.linkInputs.push_back("-Wl,-rpath," + f.parent_path().string());   // find it at run time
#endif
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
        // program links (and MultiThreaded, the static CRT, is clang's
        // default there too).
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
                conf.push_back("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded");
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
        if (!fs::exists(libOut)) {
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
                          /*compatPreprocessor=*/false, /*tolerateArchMismatch=*/false,
                          l.foreignIncludes)) {
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

std::string Builder::exportedIncludeDir() const {
    return (fs::path(root_) / "build" / "include").string();
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

    // 2. Build (or reuse) std library. A missing std/ directory is only a
    // warning (a project using nothing but extern declarations can still
    // link without it), but if std/ was found and building it failed, that's
    // a real error — silently linking without it just produces a wall of
    // confusing "undefined symbol" errors instead. A package with no SafeC
    // in it (its own sources or a SafeC source dependency) — a plain C,
    // C++ or Objective-C one — doesn't use std, and neither builds nor
    // links it.
    bool usesStd = std::any_of(srcs.begin(), srcs.end(),
                               [](const std::string& f) { return langOf(f) == SrcLang::SafeC; });
    for (auto& dep : manifest_.dependencies) {
        DepLayout l = depLayout(dep);
        if (l.error.empty() && l.fromSource) usesStd = true;
    }
    std::string stdDir  = findStdDir();
    std::string stdLib  = usesStd ? ensureStdLib() : "";
    if (usesStd && stdLib.empty() && !stdDir.empty()) {
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
    for (auto& f : depLinkInputs()) archives.push_back(f);
    if (!stdLib.empty()) archives.push_back(stdLib);

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
    } else if (crateType == "cdylib") {
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp, /*shared=*/true, hasObjC))
            return false;
        if (!exportHeaders(root_, exportedIncludeDir())) return false;
    } else {
        if (!linkFinal(objFiles, archives, outputPath(), hasCpp, /*shared=*/false, hasObjC))
            return false;
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
    std::string stdDir = findStdDir();
    std::string stdLib = usesStd ? ensureStdLib() : "";
    if (usesStd && stdLib.empty() && !stdDir.empty()) {
        std::cerr << "safeguard: failed to build the standard library "
                     "(see compiler errors above) — aborting test run\n";
        return false;
    }

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
    depIncludes(userIncs, depForeignIncs);
    bool depsCxx = depsNeedCxx();

    std::vector<std::string> archives;
    for (auto& f : depLinkInputs()) archives.push_back(f);
    if (!stdLib.empty()) archives.push_back(stdLib);

    std::string safecBin = findSafec();
    fs::path buildDir = fs::path(root_) / "build" / "tests";
    fs::create_directories(buildDir);

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
            obj = compileForeign(src, buildDir.string(), depForeignIncs, lang);
        }
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

} // namespace safeguard
