#include "Project.h"
#include "Manifest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>   // getenv

namespace fs = std::filesystem;

namespace safeguard {

// ── template application ──────────────────────────────────────────────────────

static std::string replaceAll(std::string s,
                               const std::string& from,
                               const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string Project::applyTemplate(const std::string& tmpl,
                                    const std::string& name,
                                    const std::string& version,
                                    const std::string& author) {
    std::string out = tmpl;
    out = replaceAll(out, "{{NAME}}",    name);
    out = replaceAll(out, "{{VERSION}}", version);
    out = replaceAll(out, "{{AUTHOR}}",  author);
    return out;
}

// ── file helpers ──────────────────────────────────────────────────────────────

void Project::mkdirP(const std::string& path) {
    fs::create_directories(path);
}

void Project::writeFile(const std::string& path, const std::string& content) {
    mkdirP(fs::path(path).parent_path().string());
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << content;
}

// ── built-in templates ────────────────────────────────────────────────────────

static const char kPackageTomlTemplate[] =
R"([package]
name        = "{{NAME}}"
version     = "{{VERSION}}"
author      = "{{AUTHOR}}"
description = ""
license     = "MIT"

[build]
edition = "2025"

# To add a GitHub dependency:
# [[dependencies]]
# name    = "mylib"
# version = "https://github.com/user/mylib"
#
# To add a local path dependency:
# [[dependencies]]
# name    = "mylib"
# path    = "../mylib"
)";

static const char kMainScTemplate[] =
R"(// {{NAME}} — entry point
// Uncomment to use the SafeC standard library:
// #include "prelude.h"

extern int printf(const char* fmt, ...);

int main() {
    unsafe { printf("Hello from {{NAME}}!\n"); }
    return 0;
}
)";

static const char kGitignore[] =
"build/\n"
"*.ll\n"
"*.o\n"
"*.a\n";

// ── public API ────────────────────────────────────────────────────────────────

std::string Project::findRoot(const std::string& startDir) {
    fs::path cur = fs::absolute(startDir);
    while (true) {
        if (fs::exists(cur / "Package.toml"))
            return cur.string();
        fs::path parent = cur.parent_path();
        if (parent == cur) break; // filesystem root
        cur = parent;
    }
    return "";
}

void Project::createNew(const std::string& name, const std::string& parentDir) {
    fs::path dest = fs::path(parentDir) / name;
    if (fs::exists(dest))
        throw std::runtime_error("directory already exists: " + dest.string());

    // Determine author from git config or $USER
    std::string author;
    {
        const char* user = std::getenv("USER");
        author = user ? user : "unknown";
    }

    std::string version = "0.1.0";

    // Package.toml
    writeFile((dest / "Package.toml").string(),
              applyTemplate(kPackageTomlTemplate, name, version, author));

    // src/main.sc
    writeFile((dest / "src" / "main.sc").string(),
              applyTemplate(kMainScTemplate, name, version, author));

    // .gitignore
    writeFile((dest / ".gitignore").string(), kGitignore);

    std::cout << "safeguard: created project '" << name
              << "' at " << dest.string() << "\n";
    std::cout << "  Run:  cd " << name << " && safeguard build\n";
}

void Project::initHere(const std::string& name) {
    std::string cwd = fs::current_path().string();

    const char* user = std::getenv("USER");
    std::string author = user ? user : "unknown";
    std::string version = "0.1.0";

    bool wroteManifest = false;
    if (!fs::exists("Package.toml")) {
        writeFile("Package.toml",
                  applyTemplate(kPackageTomlTemplate, name, version, author));
        wroteManifest = true;
    }

    bool wroteSrc = false;
    if (!fs::exists("src/main.sc")) {
        writeFile("src/main.sc",
                  applyTemplate(kMainScTemplate, name, version, author));
        wroteSrc = true;
    }

    if (!fs::exists(".gitignore"))
        writeFile(".gitignore", kGitignore);

    std::cout << "safeguard: initialised project '" << name << "' in " << cwd << "\n";
    if (!wroteManifest) std::cout << "  (Package.toml already existed, skipped)\n";
    if (!wroteSrc)      std::cout << "  (src/main.sc already existed, skipped)\n";
}

} // namespace safeguard
