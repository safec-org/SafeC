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
edition = "2026"

# To add a GitHub dependency:
# [[dependencies]]
# name    = "mylib"
# version = "https://github.com/user/mylib"
#
# To add a local dependency (a SafeC package with src/, or a directory of
# prebuilt libraries in lib/ with headers in include/):
# [[dependencies]]
# name    = "mylib"
# path    = "../mylib"
#
# To link prebuilt library files directly:
# [[dependencies]]
# name    = "mylib"
# lib     = ["vendor/libmylib.a"]
# include = ["vendor/include"]
#
# To build a C/C++ library from source (a CMake project, or plain source
# files with build = "sources") and link it:
# [[dependencies]]
# name    = "mylib"
# path    = "../mylib"
# build   = "cmake"
)";

static const char kMainScTemplate[] =
R"(// {{NAME}} — entry point
// Uncomment to use the SafeC standard library:
// #include <std/prelude.h>

extern int printf(const char* fmt, ...);

int main() {
    unsafe { printf("Hello from {{NAME}}!\n"); }
    return 0;
}
)";

// Entry points for 'new --lang c|cpp|objc|objcpp': a package with no SafeC
// in it, built by clang/clang++ (Objective-C on macOS).
static const char kMainCTemplate[] =
R"(// {{NAME}} — entry point
#include <stdio.h>

int main(void) {
    printf("Hello from {{NAME}}!\n");
    return 0;
}
)";

static const char kMainCppTemplate[] =
R"(// {{NAME}} — entry point
#include <iostream>

int main() {
    std::cout << "Hello from {{NAME}}!\n";
    return 0;
}
)";

static const char kMainObjCTemplate[] =
R"(// {{NAME}} — entry point
#import <Foundation/Foundation.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        NSString* greeting = [NSString stringWithFormat:@"Hello from %s!", "{{NAME}}"];
        printf("%s\n", greeting.UTF8String);
    }
    return 0;
}
)";

// The source file and [build] lines for each --lang.
struct LangTemplate {
    const char* lang;
    const char* file;      // under src/
    const char* source;
    const char* buildKeys; // appended to [build]
};
static const LangTemplate kLangs[] = {
    { "sc",     "main.sc", kMainScTemplate,   "" },
    { "c",      "main.c",  kMainCTemplate,    "" },
    { "cpp",    "main.cpp", kMainCppTemplate, "" },
    { "objc",   "main.m",  kMainObjCTemplate, "frameworks = [\"Foundation\"]\nobjc_arc   = true\n" },
    { "objcpp", "main.mm", kMainObjCTemplate, "frameworks = [\"Foundation\"]\nobjc_arc   = true\n" },
};

static const LangTemplate& langTemplate(const std::string& lang) {
    std::string l = lang == "safec" ? "sc" : lang == "c++" ? "cpp" : lang == "objc++" ? "objcpp" : lang;
    for (auto& t : kLangs) if (l == t.lang) return t;
    throw std::runtime_error("unknown --lang '" + lang + "' (expected sc, c, cpp, objc or objcpp)");
}

static std::string manifestFor(const LangTemplate& t, const std::string& tmpl, bool withStd) {
    std::string m = tmpl;
    std::string anchor = "edition = \"2026\"\n";
    std::string keys = t.buildKeys;
    if (withStd && std::string(t.lang) != "sc") keys += "std        = true\n";
    m.insert(m.find(anchor) + anchor.size(), keys);
    return m;
}

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

void Project::createNew(const std::string& name, const std::string& parentDir,
                        const std::string& lang, bool withStd) {
    const LangTemplate& t = langTemplate(lang);
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
              applyTemplate(manifestFor(t, kPackageTomlTemplate, withStd), name, version, author));

    // src/main.sc (or main.c/.cpp/.m/.mm)
    writeFile((dest / "src" / t.file).string(),
              applyTemplate(t.source, name, version, author));

    // .gitignore
    writeFile((dest / ".gitignore").string(), kGitignore);

    std::cout << "safeguard: created project '" << name
              << "' at " << dest.string() << "\n";
    std::cout << "  Run:  cd " << name << " && safeguard build\n";
}

void Project::initHere(const std::string& name, const std::string& lang, bool withStd) {
    const LangTemplate& t = langTemplate(lang);
    std::string cwd = fs::current_path().string();
    std::string mainFile = std::string("src/") + t.file;

    const char* user = std::getenv("USER");
    std::string author = user ? user : "unknown";
    std::string version = "0.1.0";

    bool wroteManifest = false;
    if (!fs::exists("Package.toml")) {
        writeFile("Package.toml",
                  applyTemplate(manifestFor(t, kPackageTomlTemplate, withStd), name, version, author));
        wroteManifest = true;
    }

    bool wroteSrc = false;
    if (!fs::exists(mainFile)) {
        writeFile(mainFile, applyTemplate(t.source, name, version, author));
        wroteSrc = true;
    }

    if (!fs::exists(".gitignore"))
        writeFile(".gitignore", kGitignore);

    std::cout << "safeguard: initialised project '" << name << "' in " << cwd << "\n";
    if (!wroteManifest) std::cout << "  (Package.toml already existed, skipped)\n";
    if (!wroteSrc)      std::cout << "  (" << mainFile << " already existed, skipped)\n";
}

} // namespace safeguard
