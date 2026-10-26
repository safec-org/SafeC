#include "Manifest.h"
#include "Builder.h"
#include "Analyzer.h"
#include "Formatter.h"
#include "Project.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

static void usage() {
    std::cout << R"(Usage: safeguard <command> [options]

Commands:
  new   <name> [--lang L] [--std]  Create a new project directory with a skeleton
  init  [name] [--lang L] [--std]  Initialise the current directory as a project
                L: sc (SafeC, default), c, cpp, objc, objcpp — a package
                with no SafeC in it (src/main.c, .cpp, .m or .mm)
                --std: such a package builds and links SafeC's std too
                ([build] std = true)
  fetch         Clone and build all [dependencies] from Package.toml
  build         Compile all sources in the current project
  check         Fast compile-only pass — reports errors without linking
  std-modules [std/x.h|file...]
                The std sources a build compiles and the system libraries
                it links for them (for the project's sources, or for the
                given std headers and files)
  std-prebuilt <prefix>
                Compile std for this machine, optimized, into one archive
                per std section, <prefix>/lib/<triple>/std/<section>.a
                (with its <section>.units): the prebuilt std a release
                install ships
  test          Build and run every file under tests/ as a standalone binary
  run   [args..]Build and immediately run the output binary
  bundle        Build the bin target and package it per its [bundle] table:
                build/<name>.app (macOS), build/<name>/ (Windows),
                a .deb and build/<name>.AppDir (Linux), and with
                --target wasm32 build/<name>.web/: the module, its JS glue
                and an index.html (no [bundle] table needed)
  doc   [--serve] [--std]
                Generate the API reference (build/doc/): every public
                declaration of include/ and src/ headers with its doc
                comment, per-item pages, a module tree, search, source
                pages and links into std's reference. --serve serves it
                (--port N, default 8000); --std documents SafeC's std
                instead; --out DIR, --private (hidden items too),
                --std-url URL (where std links point), --no-check
  publish [--ci=github|gitlab|gitea|forgejo] [--bump=patch|minor|major]
          [--write] [--tag] [--push]
                Prepare a library for publishing to git: .gitignore,
                README/LICENSE checks, CI that builds and tests on
                Linux/macOS/Windows, attaches prebuilt libraries to a
                release and deploys 'safeguard doc' to Pages on a tag.
                Dry run (prints the plan) unless --write; commits/tags
                only with --tag, pushes only with --push
  clean         Remove the build/ and deps/ directories
  verify-lock   Check Package.lock against current dependency state
  lint          Run static-analysis lint passes on all source files
  format        Reindent and whitespace-normalize all .sc/.h source files

Dependency format in Package.toml:
  [[dependencies]]
  name    = "mylib"
  version = "https://github.com/user/mylib"   # git URL, cloned to deps/mylib
  # or a local directory:
  name    = "mylib"
  path    = "../mylib"
  # or prebuilt library files:
  name    = "mylib"
  lib     = ["vendor/libmylib.a"]             # .a / .so / .dylib / .lib
  include = ["vendor/include"]
  # or a C/C++/ObjC project safeguard builds (into build/deps/<name>):
  name       = "mylib"
  path       = "../mylib"                      # or version = git URL
  build      = "cmake"                         # configure, build, install
  cmake_args = ["-DMYLIB_TESTS=OFF"]
  # or: build = "sources", sources = ["src", "extra/*.c"],
  #     defines = ["NDEBUG"], cflags = ["-fobjc-arc"], include = ["include"]
  frameworks = ["Foundation"]                  # macOS frameworks it needs
  A directory (fetched or local) with src/ is a SafeC package built from
  source; one without is a prebuilt library directory: lib<name> (or every
  library) in lib/, lib64/ or the directory itself, headers in include/
  (on Windows, <name>.lib — a DLL's import library, the DLL beside it or
  in ../bin/ is copied next to the program). SafeC calls C++ only through
  extern "C" functions.

[build] keys for C/C++/Objective-C:
  frameworks = ["Cocoa"]     macOS frameworks to link (ignored elsewhere)
  objc_arc   = true          compile .m/.mm with -fobjc-arc
  std        = true          build and link SafeC's std (default: only
                             when the package has SafeC); false: never;
                             "all": every std module, not only the ones
                             the sources include
  [build.macos] / [build.linux] / [build.windows]
    cflags, libs, lib_dirs, frameworks — added only when building for that OS

[bundle] format (for 'safeguard bundle'):
  [bundle]
  name       = "Hello"                 # default: package name
  identifier = "org.example.hello"     # required
  version    = "1.0.0"                 # default: package version
  icon       = "assets/icon.png"       # .png (any OS), .icns, .ico
  resources  = ["assets/data"]         # copied into the bundle
  category   = "Utility"               # or "public.app-category.*"
  copyright  = "(c) 2026 Example"
  subsystem  = "windows"               # Windows: "windows" (GUI) / "console"
  min_system_version = "11.0"          # macOS LSMinimumSystemVersion
  codesign_identity  = "-"             # macOS: "-" ad-hoc, identity, "none"
  depends    = ["libsqlite3-0"]        # Linux: the .deb's Depends (with
                                       # those std's modules name)
  web_entry   = "start"                # wasm32: the export the page calls
  web_present = "webgl"                # wasm32: "webgl" or "2d" (std/gui)
  web_windows = "frames"               # wasm32: "canvas" (default) / "frames"
  web_page    = "web/index.html"       # wasm32: your page instead of one made
  web_files   = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
                                       # wasm32: files the module can fopen()

Options (build / run / check / test):
  --release     Enable optimisations (-O2)
  --emit-llvm   Stop after emitting LLVM IR (do not link) [build only]
  --verbose     Print every command before executing it
  --features <a,b,c>       Enable the named [features] (comma-separated;
                            repeatable) in addition to the manifest's
                            "default" feature
  --no-default-features    Do not enable the manifest's "default" feature
  --crate-type <staticlib|cdylib|bin>
                           Build this kind of artifact instead of the
                           manifest's [build] crate_type
  --target <alias|triple>  Cross-compile: a bare-metal alias (x86, x86_64,
                            aarch32, aarch64, thumb, thumb2, arm-6m, arm-7m,
                            arm-7em, arm-8m, rv32i, rv32e, rv64i, rv64e,
                            wasm32, wasm64 — see 'safec --list-targets') or
                            any LLVM triple. Bare-metal links use -nostdlib
                            -static; supply a linker script/runtime via
                            [build] cflags/libs. WebAssembly links a library
                            module (no entry; exports are functions marked
                            __attribute__((export_name)), unresolved externs
                            become imports; std/wasm/wasm_rt supplies the
                            runtime).
                            'test' builds without running; 'run' is refused.

Options (bundle):
  --sign <identity>        macOS code-signing identity ("-" = ad-hoc,
                            "none" = unsigned); overrides codesign_identity

[features] format in Package.toml (Rust-Cargo-style; see advanced/safeguard
docs for the full write-up):
  [features]
  default   = ["backend"]           # enabled unless --no-default-features
  frontend  = []
  backend   = []
  fullstack = ["frontend", "backend"]  # enabling one feature can enable others

Each enabled feature 'name' is passed to safec as -DSAFEC_FEATURE_<NAME>=1
(hyphens become underscores), so project sources gate code with:
  #ifdef SAFEC_FEATURE_BACKEND
  ...
  #endif

Options (format):
  --check       Report files that would change, without writing them
                (exit code 1 if any would change)

Environment:
  SAFEC_HOME    Path to the SafeC install prefix (set automatically by
                install.sh/install.ps1), or a from-source repository root
                (e.g. export SAFEC_HOME=/path/to/safec or /path/to/SafeC)

Examples:
  safeguard new hello_world
  safeguard new cfast --lang c
  cd hello_world && safeguard build
  safeguard check
  safeguard test
  safeguard format --check
  safeguard run -- --my-flag
  safeguard build --features frontend --no-default-features
)";
}

// Walk up from cwd to find the project root; throw if not found.
static std::pair<std::string, safeguard::Manifest> requireProject() {
    std::string root = safeguard::Project::findRoot(".");
    if (root.empty())
        throw std::runtime_error(
            "no Package.toml found; run 'safeguard new <name>' or "
            "'safeguard init' first");
    auto manifest = safeguard::ManifestParser::parseFile(
        (fs::path(root) / "Package.toml").string());
    return {root, manifest};
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }

    std::string cmd = argv[1];

    // ── new / init ────────────────────────────────────────────────────────────
    if (cmd == "new" || cmd == "init") {
        std::string name, lang = "sc";
        bool withStd = false;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--std") { withStd = true; continue; }
            if (arg == "--lang") {
                if (i + 1 >= argc) {
                    std::cerr << "safeguard: --lang requires an argument\n";
                    return 1;
                }
                lang = argv[++i];
                continue;
            }
            if (arg.rfind("--lang=", 0) == 0) { lang = arg.substr(7); continue; }
            if (arg.rfind("--", 0) == 0 || !name.empty()) {
                std::cerr << "safeguard: unknown option '" << arg << "'\n";
                return 1;
            }
            name = arg;
        }
        if (name.empty()) {
            if (cmd == "new") {
                std::cerr << "safeguard: 'new' requires a project name\n";
                return 1;
            }
            name = fs::current_path().filename().string();
        }
        try {
            if (cmd == "new") safeguard::Project::createNew(name, ".", lang, withStd);
            else safeguard::Project::initHere(name, lang, withStd);
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ── lint (formerly 'analyze', kept as an alias) ─────────────────────────────
    if (cmd == "lint" || cmd == "analyze") {
        bool verbose = false;
        for (int i = 2; i < argc; ++i)
            if (std::string(argv[i]) == "--verbose") verbose = true;
        try {
            auto [root, manifest] = requireProject();
            safeguard::Analyzer analyzer(root, manifest, verbose);
            return analyzer.analyze() ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── format ────────────────────────────────────────────────────────────────
    if (cmd == "format" || cmd == "fmt") {
        bool checkOnly = false, verbose = false;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--check")   { checkOnly = true; continue; }
            if (arg == "--verbose") { verbose   = true; continue; }
            std::cerr << "safeguard: unknown option '" << arg << "'\n";
            return 1;
        }
        try {
            auto [root, manifest] = requireProject();
            (void)manifest;
            safeguard::Formatter fmt;
            std::string srcDir = (fs::path(root) / "src").string();
            bool ok = fmt.run(srcDir, checkOnly, verbose);
            return ok ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── doc ───────────────────────────────────────────────────────────────────
    if (cmd == "doc") {
        safeguard::DocOptions dopts;
        safeguard::BuildOptions bopts;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&](const char* flag) -> std::string {
                std::string f = flag;
                if (arg.rfind(f + "=", 0) == 0) return arg.substr(f.size() + 1);
                if (i + 1 >= argc) throw std::runtime_error(f + " requires an argument");
                return argv[++i];
            };
            try {
                if (arg == "--std") dopts.std = true;
                else if (arg == "--serve") dopts.serve = true;
                else if (arg == "--private" || arg == "--document-private-items") dopts.includePrivate = true;
                else if (arg == "--no-check") dopts.check = false;
                else if (arg == "--verbose") bopts.verbose = true;
                else if (arg == "--port" || arg.rfind("--port=", 0) == 0) dopts.port = std::atoi(value("--port").c_str());
                else if (arg == "--out" || arg.rfind("--out=", 0) == 0) dopts.out = value("--out");
                else if (arg == "--std-url" || arg.rfind("--std-url=", 0) == 0) dopts.stdUrl = value("--std-url");
                else if (arg == "--source-ref" || arg.rfind("--source-ref=", 0) == 0) dopts.sourceRef = value("--source-ref");
                else if (arg == "--home" || arg.rfind("--home=", 0) == 0) dopts.homeUrl = value("--home");
                else { std::cerr << "safeguard: unknown option '" << arg << "'\n"; return 1; }
            } catch (std::exception& e) {
                std::cerr << "safeguard: " << e.what() << "\n";
                return 1;
            }
        }
        try {
            if (dopts.std) {
                safeguard::Manifest none;
                none.package.name = "std";
                safeguard::Builder builder(fs::current_path().string(), none, bopts);
                return builder.doc(dopts) ? 0 : 1;
            }
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, bopts);
            return builder.doc(dopts) ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── publish ───────────────────────────────────────────────────────────────
    if (cmd == "publish") {
        safeguard::PublishOptions popts;
        safeguard::BuildOptions bopts;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&](const char* flag) -> std::string {
                std::string f = flag;
                if (arg.rfind(f + "=", 0) == 0) return arg.substr(f.size() + 1);
                if (i + 1 >= argc) throw std::runtime_error(f + " requires an argument");
                return argv[++i];
            };
            try {
                if (arg == "--write") popts.write = true;
                else if (arg == "--dry-run") popts.write = false;
                else if (arg == "--tag") popts.tag = true;
                else if (arg == "--push") popts.push = true;
                else if (arg == "--no-verify") popts.verify = false;
                else if (arg == "--verbose") bopts.verbose = true;
                else if (arg == "--ci" || arg.rfind("--ci=", 0) == 0) popts.ci = value("--ci");
                else if (arg == "--bump" || arg.rfind("--bump=", 0) == 0) popts.bump = value("--bump");
                else if (arg == "--remote" || arg.rfind("--remote=", 0) == 0) popts.remote = value("--remote");
                else { std::cerr << "safeguard: unknown option '" << arg << "'\n"; return 1; }
            } catch (std::exception& e) {
                std::cerr << "safeguard: " << e.what() << "\n";
                return 1;
            }
        }
        try {
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, bopts);
            return builder.publish(popts) ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── verify-lock ───────────────────────────────────────────────────────────
    if (cmd == "verify-lock") {
        try {
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, {});
            return builder.checkLock() ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── fetch ─────────────────────────────────────────────────────────────────
    if (cmd == "fetch") {
        safeguard::BuildOptions opts;
        opts.verbose = true;
        try {
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, opts);
            return builder.fetchAndBuildDeps() ? 0 : 1;
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
    }

    // ── build / check / test / run / bundle / clean ───────────────────────────
    if (cmd == "build" || cmd == "check" || cmd == "test" || cmd == "run" || cmd == "bundle" ||
        cmd == "clean" || cmd == "std-modules" || cmd == "std-prebuilt") {
        safeguard::BuildOptions opts;
        std::vector<std::string> runArgs;
        bool pastSep = false; // '--' separator for run args

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (pastSep) { runArgs.push_back(arg); continue; }
            if (arg == "--")        { pastSep = true; continue; }
            if (arg == "--release") { opts.release   = true; continue; }
            if (arg == "--emit-llvm") { opts.emitLLVM = true; continue; }
            if (arg == "--verbose") { opts.verbose   = true; continue; }
            if (arg == "--no-default-features") { opts.noDefaultFeatures = true; continue; }
            if (arg == "--target") {
                if (i + 1 >= argc) {
                    std::cerr << "safeguard: --target requires an argument\n";
                    return 1;
                }
                opts.target = argv[++i];
                continue;
            }
            if (arg.rfind("--target=", 0) == 0) { opts.target = arg.substr(9); continue; }
            if (arg == "--crate-type" || arg.rfind("--crate-type=", 0) == 0) {
                if (arg == "--crate-type") {
                    if (i + 1 >= argc) {
                        std::cerr << "safeguard: --crate-type requires staticlib, cdylib or bin\n";
                        return 1;
                    }
                    opts.crateType = argv[++i];
                } else {
                    opts.crateType = arg.substr(13);
                }
                continue;
            }
            if (arg == "--sign" && cmd == "bundle") {
                if (i + 1 >= argc) {
                    std::cerr << "safeguard: --sign requires an identity (\"-\" for ad-hoc)\n";
                    return 1;
                }
                opts.signIdentity = argv[++i];
                continue;
            }
            if (arg == "--features") {
                if (i + 1 >= argc) {
                    std::cerr << "safeguard: --features requires an argument\n";
                    return 1;
                }
                std::string list = argv[++i];
                size_t start = 0;
                while (start <= list.size()) {
                    size_t comma = list.find(',', start);
                    std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                    if (!item.empty()) opts.features.push_back(item);
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
                continue;
            }
            if (arg.rfind("--features=", 0) == 0) {
                std::string list = arg.substr(11);
                size_t start = 0;
                while (start <= list.size()) {
                    size_t comma = list.find(',', start);
                    std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                    if (!item.empty()) opts.features.push_back(item);
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
                continue;
            }
            if ((cmd == "std-modules" || cmd == "std-prebuilt") && arg.rfind("-", 0) != 0) {
                runArgs.push_back(arg);
                continue;
            }
            std::cerr << "safeguard: unknown option '" << arg << "'\n";
            return 1;
        }

        try {
            if (cmd == "std-prebuilt") {
                if (runArgs.size() != 1) {
                    std::cerr << "safeguard: usage: safeguard std-prebuilt <prefix>\n";
                    return 1;
                }
                // Built in a scratch project of its own, removed afterwards.
                fs::path work = fs::temp_directory_path() /
                                ("safeguard-std-prebuilt-" + std::to_string(std::rand()));
                fs::create_directories(work);
                opts.release = true;
                safeguard::Manifest none;
                safeguard::Builder builder(work.string(), none, opts);
                bool ok = builder.writePrebuiltStd(fs::absolute(runArgs[0]).string());
                std::error_code ec;
                fs::remove_all(work, ec);
                return ok ? 0 : 1;
            }
            if (cmd == "std-modules" && !runArgs.empty() && !fs::exists("Package.toml")) {
                // Outside a project: just the named std files and sources
                // (the std units it compiles cached in a scratch project).
                safeguard::Manifest none;
                fs::path work = fs::temp_directory_path() / "safeguard-std-modules";
                fs::create_directories(work);
                std::vector<std::string> args;
                for (auto& a : runArgs) {
                    bool stdName = a.rfind("std/", 0) == 0 || a.rfind("<", 0) == 0;
                    args.push_back(!stdName && fs::exists(a) ? fs::absolute(a).string() : a);
                }
                safeguard::Builder builder(work.string(), none, opts);
                return builder.printStdModules(args) ? 0 : 1;
            }
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, opts);

            if (cmd == "clean") {
                builder.clean();
                return 0;
            }
            if (cmd == "build") {
                return builder.build() ? 0 : 1;
            }
            if (cmd == "check") {
                return builder.check() ? 0 : 1;
            }
            if (cmd == "std-modules") {
                return builder.printStdModules(runArgs) ? 0 : 1;
            }
            if (cmd == "test") {
                return builder.test() ? 0 : 1;
            }
            if (cmd == "bundle") {
                return builder.bundle() ? 0 : 1;
            }
            if (cmd == "run") {
                int rc = builder.run(runArgs);
                return rc;
            }
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ── unknown ───────────────────────────────────────────────────────────────
    if (cmd == "--help" || cmd == "-h") { usage(); return 0; }
    std::cerr << "safeguard: unknown command '" << cmd << "'\n\n";
    usage();
    return 1;
}
