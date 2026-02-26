#include "Manifest.h"
#include "Builder.h"
#include "Analyzer.h"
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
  new   <name>  Create a new project directory with a skeleton
  init  [name]  Initialise the current directory as a project
  fetch         Clone and build all [dependencies] from Package.toml
  build         Compile all sources in the current project
  run   [args..]Build and immediately run the output binary
  clean         Remove the build/ and deps/ directories
  verify-lock   Check Package.lock against current dependency state
  analyze       Run static analysis lint passes on all source files

Dependency format in Package.toml:
  [[dependencies]]
  name    = "mylib"
  version = "https://github.com/user/mylib"   # GitHub URL
  # or for a local path:
  name    = "mylib"
  path    = "../mylib"

Options (build / run):
  --release     Enable optimisations (-O2)
  --emit-llvm   Stop after emitting LLVM IR (do not link)
  --verbose     Print every command before executing it

Environment:
  SAFEC_HOME    Path to the SafeC repository root
                (e.g. export SAFEC_HOME=/path/to/SafeC)

Examples:
  safeguard new hello_world
  cd hello_world && safeguard build
  safeguard run -- --my-flag
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

    // ── new ───────────────────────────────────────────────────────────────────
    if (cmd == "new") {
        if (argc < 3) {
            std::cerr << "safeguard: 'new' requires a project name\n";
            return 1;
        }
        try {
            safeguard::Project::createNew(argv[2]);
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ── init ──────────────────────────────────────────────────────────────────
    if (cmd == "init") {
        std::string name = (argc >= 3) ? argv[2]
                                       : fs::current_path().filename().string();
        try {
            safeguard::Project::initHere(name);
        } catch (std::exception& e) {
            std::cerr << "safeguard error: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ── analyze ───────────────────────────────────────────────────────────────
    if (cmd == "analyze") {
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

    // ── build / run / clean ───────────────────────────────────────────────────
    if (cmd == "build" || cmd == "run" || cmd == "clean") {
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
            std::cerr << "safeguard: unknown option '" << arg << "'\n";
            return 1;
        }

        try {
            auto [root, manifest] = requireProject();
            safeguard::Builder builder(root, manifest, opts);

            if (cmd == "clean") {
                builder.clean();
                return 0;
            }
            if (cmd == "build") {
                return builder.build() ? 0 : 1;
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
