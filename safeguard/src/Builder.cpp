#include "Builder.h"
#include "Lock.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace safeguard {

Builder::Builder(std::string projectRoot, Manifest manifest, BuildOptions opts)
    : root_(std::move(projectRoot))
    , manifest_(std::move(manifest))
    , opts_(std::move(opts))
{}

// ── safec / std discovery ─────────────────────────────────────────────────────

std::string Builder::findSafec() const {
    // 1. $SAFEC_HOME/compiler/build/safec  (SAFEC_HOME = SafeC repo root)
    const char* home = std::getenv("SAFEC_HOME");
    if (home) {
        fs::path c = fs::path(home) / "compiler" / "build" / "safec";
        if (fs::exists(c)) return c.string();
    }
    // 2. PATH fallback
    return "safec";
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

// ── command runner ────────────────────────────────────────────────────────────

int Builder::runCmd(const std::vector<std::string>& argv, bool verbose) {
    if (argv.empty()) return -1;
    if (verbose) {
        for (size_t i = 0; i < argv.size(); ++i)
            std::cout << (i ? " " : "") << argv[i];
        std::cout << "\n";
    }
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "safeguard: fork failed: " << strerror(errno) << "\n";
        return -1;
    }
    if (pid == 0) {
        std::vector<char*> args;
        for (auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
        args.push_back(nullptr);
        execvp(args[0], args.data());
        std::cerr << "safeguard: exec '" << argv[0] << "' failed: "
                  << strerror(errno) << "\n";
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// ── compile one .sc → .ll ─────────────────────────────────────────────────────

std::string Builder::compileSrc(const std::string& safecBin,
                                  const std::string& srcPath,
                                  const std::string& buildDir,
                                  const std::vector<std::string>& includeDirs) const {
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
    cmd.push_back(srcPath);
    cmd.push_back("--emit-llvm");
    cmd.push_back("-o");
    cmd.push_back(llFile.string());
    if (!opts_.extraFlags.empty())
        cmd.push_back(opts_.extraFlags);

    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: compilation failed for " << srcPath << "\n";
        return "";
    }
    return llFile.string();
}

// ── compile .ll → .o ─────────────────────────────────────────────────────────

std::string Builder::llToObj(const std::string& llFile,
                               const std::string& buildDir) const {
    fs::path obj = fs::path(buildDir) / (fs::path(llFile).stem().string() + ".o");
    std::vector<std::string> cmd = { "clang", "-c", llFile, "-o", obj.string() };
    if (opts_.release) cmd.push_back("-O2");
    int rc = runCmd(cmd, opts_.verbose);
    if (rc != 0) {
        std::cerr << "safeguard: clang -c failed for " << llFile << "\n";
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
                          const std::string& output) const {
    std::vector<std::string> cmd = { "clang" };
    if (opts_.release) cmd.push_back("-O2");
    for (auto& o : objFiles) cmd.push_back(o);
    // Archives: pass with -force_load on macOS so unused symbols are kept,
    // or use the portable -Wl,--whole-archive on Linux.
    // For simplicity, just list the .a files directly (works for most cases).
    for (auto& a : archives) cmd.push_back(a);
    cmd.push_back("-lm"); // math library for std/math.sc
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
                        const std::vector<std::string>& includeDirs) {
    std::string safecBin = findSafec();
    std::string buildDir = fs::path(libOut).parent_path().string();
    fs::create_directories(buildDir);

    auto srcs = collectSources(srcDir);
    if (srcs.empty()) return true; // nothing to compile

    std::vector<std::string> objs;
    for (auto& src : srcs) {
        std::string ll = compileSrc(safecBin, src, buildDir, includeDirs);
        if (ll.empty()) return false;
        std::string obj = llToObj(ll, buildDir);
        if (obj.empty()) return false;
        objs.push_back(obj);
    }
    return packArchive(objs, libOut);
}

// ── build (or reuse) the SafeC standard library ───────────────────────────────

std::string Builder::ensureStdLib() {
    std::string stdDir = findStdDir();
    if (stdDir.empty()) {
        std::cerr << "safeguard: warning: cannot find SafeC std/ directory; "
                     "set SAFEC_HOME to the SafeC repository root\n";
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

    if (!buildLib(stdDir, libPath.string(), incs)) return "";
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
            // Also add std dir
            std::string stdDir = findStdDir();
            if (!stdDir.empty()) {
                incs.push_back(stdDir);
                incs.push_back(fs::path(stdDir).parent_path().string());
            }

            if (!buildLib(depSrc.string(), libOut.string(), incs)) {
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

    // 2. Build (or reuse) std library
    std::string stdLib  = ensureStdLib();
    std::string stdDir  = findStdDir();

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

    // 4. Compile user sources
    fs::path srcDir = fs::path(root_) / "src";
    std::vector<std::string> srcs;
    if (!manifest_.build.srcs.empty()) {
        srcs = manifest_.build.srcs;
    } else {
        srcs = collectSources(srcDir.string());
    }
    if (srcs.empty()) {
        std::cerr << "safeguard: no .sc source files found under src/\n";
        return false;
    }

    std::vector<std::string> llFiles;
    for (auto& src : srcs) {
        std::string ll = compileSrc(safecBin, src, buildDir, userIncs);
        if (ll.empty()) return false;
        llFiles.push_back(ll);
    }

    if (opts_.emitLLVM) {
        std::cout << "safeguard: LLVM IR written to " << buildDir << "/\n";
        return true;
    }

    // 5. Compile .ll → .o
    std::vector<std::string> objFiles;
    for (auto& ll : llFiles) {
        std::string obj = llToObj(ll, buildDir);
        if (obj.empty()) return false;
        objFiles.push_back(obj);
    }

    // 6. Collect archive libraries
    std::vector<std::string> archives;
    if (!stdLib.empty()) archives.push_back(stdLib);
    fs::path depLibDir = fs::path(root_) / "build" / "deps";
    for (auto& dep : manifest_.dependencies) {
        fs::path libOut = depLibDir / ("lib" + dep.name + ".a");
        if (fs::exists(libOut)) archives.push_back(libOut.string());
    }

    // 7. Link final binary
    if (!linkFinal(objFiles, archives, outputPath())) return false;

    // 8. Write / refresh Package.lock after a successful build.
    writeLock(srcs);
    return true;
}

// ── public: run ───────────────────────────────────────────────────────────────

int Builder::run(const std::vector<std::string>& args) {
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
