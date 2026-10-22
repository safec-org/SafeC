#include "Manifest.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace safeguard {

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    size_t l = s.find_first_not_of(" \t\r\n");
    if (l == std::string::npos) return "";
    size_t r = s.find_last_not_of(" \t\r\n");
    return s.substr(l, r - l + 1);
}

static std::string stripComment(const std::string& line) {
    // Remove everything after an unquoted '#'
    bool inQ = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') inQ = !inQ;
        if (!inQ && line[i] == '#') return line.substr(0, i);
    }
    return line;
}

static std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// Parse a TOML array value like '["src/main.sc", "src/foo.c"]' into a
// vector of unquoted strings. tok.value has already been through unquote()
// in tokenize(), which only strips a single leading/trailing quote pair, so
// a bracketed array survives here as the literal '[...]' text — split it
// on commas, tolerating (and stripping) per-element quotes ourselves.
static std::vector<std::string> splitList(const std::string& value) {
    std::vector<std::string> out;
    std::string v = trim(value);
    if (v.size() >= 2 && v.front() == '[' && v.back() == ']')
        v = v.substr(1, v.size() - 2);
    std::string cur;
    bool inQ = false;
    for (char c : v) {
        if (c == '"') { inQ = !inQ; continue; }
        if (c == ',' && !inQ) {
            std::string t = trim(cur);
            if (!t.empty()) out.push_back(t);
            cur.clear();
            continue;
        }
        cur += c;
    }
    std::string t = trim(cur);
    if (!t.empty()) out.push_back(t);
    return out;
}

// ── tokenize ─────────────────────────────────────────────────────────────────

// Returns tokens as {section.key, value} pairs.
// Section headers like [package] set a running prefix.
std::vector<ManifestParser::Token> ManifestParser::tokenize(
        const std::string& src, const std::string& srcName) {
    std::vector<Token> tokens;
    std::istringstream ss(src);
    std::string line;
    std::string section;
    int lineno = 0;

    while (std::getline(ss, line)) {
        ++lineno;
        line = trim(stripComment(line));
        if (line.empty()) continue;

        if (line.front() == '[') {
            // Section header: [table] or [[array-of-tables]]
            bool array = line.size() > 1 && line[1] == '[';
            size_t close = line.find(array ? "]]" : "]");
            if (close == std::string::npos)
                throw std::runtime_error(srcName + ":" + std::to_string(lineno) +
                                         ": unterminated section header");
            size_t open = array ? 2 : 1;
            section = trim(line.substr(open, close - open));
            // Emit a section-start marker
            tokens.push_back({"__section__", section});
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error(srcName + ":" + std::to_string(lineno) +
                                     ": expected key = value, got: " + line);

        std::string key   = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        tokens.push_back({section + "." + key, unquote(value)});
    }
    return tokens;
}

// ── buildManifest ─────────────────────────────────────────────────────────────

Manifest ManifestParser::buildManifest(const std::vector<Token>& tokens,
                                        const std::string& srcName) {
    Manifest m;
    m.build.edition = "2026";
    Dependency curDep;
    bool inDep = false;

    auto flushDep = [&]() {
        if (inDep && !curDep.name.empty()) {
            m.dependencies.push_back(curDep);
            curDep = {};
        }
        inDep = false;
    };
    auto boolean = [&](const Token& tok) {
        if (tok.value == "true") return true;
        if (tok.value == "false") return false;
        throw std::runtime_error(srcName + ": " + tok.key + " must be true or false, got \"" +
                                 tok.value + "\"");
    };
    // [build.macos] / [build.linux] / [build.windows]: "build.linux.libs"
    // sets forLinux.libs. Returns false for any other key.
    auto platformKey = [&](const Token& tok) {
        struct { const char* prefix; PlatformBuildConfig* cfg; } kOS[] = {
            { "build.macos.", &m.build.forMacos },
            { "build.linux.", &m.build.forLinux },
            { "build.windows.", &m.build.forWindows },
        };
        for (auto& os : kOS) {
            if (tok.key.rfind(os.prefix, 0) != 0) continue;
            std::string k = tok.key.substr(std::string(os.prefix).size());
            if (k == "cflags")          os.cfg->cflags     = splitList(tok.value);
            else if (k == "libs")       os.cfg->libs       = splitList(tok.value);
            else if (k == "lib_dirs")   os.cfg->libDirs    = splitList(tok.value);
            else if (k == "frameworks") os.cfg->frameworks = splitList(tok.value);
            else
                throw std::runtime_error(srcName + ": unknown key '" + k + "' in [" +
                                         tok.key.substr(0, std::string(os.prefix).size() - 1) +
                                         "] (expected cflags, libs, lib_dirs or frameworks)");
            return true;
        }
        return false;
    };

    for (auto& tok : tokens) {
        if (tok.key == "__section__") {
            flushDep();
            if (tok.value == "dependencies") inDep = true;
            else if (tok.value == "bundle") m.bundle.present = true;
            else if (tok.value.rfind("build.", 0) == 0 && tok.value != "build.macos" &&
                     tok.value != "build.linux" && tok.value != "build.windows")
                throw std::runtime_error(srcName + ": unknown table [" + tok.value + "]; per-platform "
                                         "settings go in [build.macos], [build.linux] or [build.windows]");
            continue;
        }
        if (platformKey(tok)) continue;

        // package.*
        if (tok.key == "package.name")        m.package.name        = tok.value;
        else if (tok.key == "package.version") m.package.version     = tok.value;
        else if (tok.key == "package.author")  m.package.author      = tok.value;
        else if (tok.key == "package.description") m.package.description = tok.value;
        else if (tok.key == "package.license") m.package.license     = tok.value;

        // build.*
        else if (tok.key == "build.edition")  m.build.edition  = tok.value;
        else if (tok.key == "build.output")   m.build.output   = tok.value;
        else if (tok.key == "build.srcs")     m.build.srcs     = splitList(tok.value);
        else if (tok.key == "build.cflags")   m.build.cflags   = splitList(tok.value);
        else if (tok.key == "build.libs")     m.build.libs     = splitList(tok.value);
        else if (tok.key == "build.lib_dirs") m.build.libDirs  = splitList(tok.value);
        else if (tok.key == "build.lto")      m.build.lto      = tok.value;
        else if (tok.key == "build.blas")     m.build.blas     = tok.value;
        else if (tok.key == "build.crate_type") m.build.crateType = tok.value;
        else if (tok.key == "build.frameworks") m.build.frameworks = splitList(tok.value);
        else if (tok.key == "build.objc_arc")   m.build.objcArc  = boolean(tok);
        else if (tok.key == "build.compat_preprocessor") m.build.compatPreprocessor = boolean(tok);
        else if (tok.key == "build.std")        m.build.useStd   = tok.value == "all" ? 2 : boolean(tok) ? 1 : 0;
        else if (tok.key == "build.std_prebuilt") m.build.stdPrebuilt = boolean(tok);

        // bundle.*
        else if (tok.key == "bundle.name")       m.bundle.name       = tok.value;
        else if (tok.key == "bundle.identifier") m.bundle.identifier = tok.value;
        else if (tok.key == "bundle.version")    m.bundle.version    = tok.value;
        else if (tok.key == "bundle.icon")       m.bundle.icon       = tok.value;
        else if (tok.key == "bundle.resources")  m.bundle.resources  = splitList(tok.value);
        else if (tok.key == "bundle.category")   m.bundle.category   = tok.value;
        else if (tok.key == "bundle.copyright")  m.bundle.copyright  = tok.value;
        else if (tok.key == "bundle.subsystem")  m.bundle.subsystem  = tok.value;
        else if (tok.key == "bundle.min_system_version") m.bundle.minSystemVersion = tok.value;
        else if (tok.key == "bundle.codesign_identity")  m.bundle.codesignIdentity = tok.value;
        else if (tok.key == "bundle.depends")    m.bundle.depends    = splitList(tok.value);
        else if (tok.key.rfind("bundle.", 0) == 0)
            throw std::runtime_error(srcName + ": unknown key '" + tok.key.substr(7) + "' in [bundle]");

        // dependencies.*
        else if (tok.key == "dependencies.name")    curDep.name    = tok.value;
        else if (tok.key == "dependencies.version") curDep.version = tok.value;
        else if (tok.key == "dependencies.path")    curDep.path    = tok.value;
        else if (tok.key == "dependencies.lib")     curDep.libs     = splitList(tok.value);
        else if (tok.key == "dependencies.include") curDep.includes = splitList(tok.value);
        else if (tok.key == "dependencies.build")   curDep.build    = tok.value;
        else if (tok.key == "dependencies.sources") curDep.sources  = splitList(tok.value);
        else if (tok.key == "dependencies.defines") curDep.defines  = splitList(tok.value);
        else if (tok.key == "dependencies.cflags")  curDep.cflags   = splitList(tok.value);
        else if (tok.key == "dependencies.cmake_args") curDep.cmakeArgs  = splitList(tok.value);
        else if (tok.key == "dependencies.frameworks") curDep.frameworks = splitList(tok.value);

        // features.* — [features] table: name = ["dep1", "dep2", ...]
        else if (tok.key.rfind("features.", 0) == 0) {
            std::string featName = tok.key.substr(9);
            m.features.push_back({featName, splitList(tok.value)});
        }
    }
    flushDep();

    if (m.package.name.empty())
        throw std::runtime_error(srcName + ": [package] name is required");
    if (m.package.version.empty())
        m.package.version = "0.1.0";
    if (m.build.output.empty())
        m.build.output = m.package.name;
    if (m.build.crateType.empty())
        m.build.crateType = "bin";
    else if (m.build.crateType != "bin" && m.build.crateType != "staticlib" &&
             m.build.crateType != "cdylib")
        throw std::runtime_error(srcName + ": build.crate_type must be "
                                 "\"bin\", \"staticlib\", or \"cdylib\", got \"" +
                                 m.build.crateType + "\"");
    if (!m.build.blas.empty() && m.build.blas != "auto" && m.build.blas != "aocl" &&
        m.build.blas != "mkl" && m.build.blas != "accelerate" && m.build.blas != "openblas" &&
        m.build.blas != "none")
        throw std::runtime_error(srcName + ": build.blas must be \"auto\", \"aocl\", \"mkl\", "
                                 "\"accelerate\", \"openblas\" or \"none\", got \"" + m.build.blas + "\"");
    if (!m.build.lto.empty() && m.build.lto != "thin" && m.build.lto != "full")
        throw std::runtime_error(srcName + ": build.lto must be \"thin\" or "
                                 "\"full\", got \"" + m.build.lto + "\"");

    for (auto& d : m.dependencies) {
        std::string where = srcName + ": dependency '" + d.name + "': ";
        if (!d.build.empty() && d.build != "sources" && d.build != "cmake")
            throw std::runtime_error(where + "build must be \"sources\" or \"cmake\", got \"" + d.build + "\"");
        if (!d.build.empty() && d.path.empty() && d.version.empty())
            throw std::runtime_error(where + "build = \"" + d.build + "\" needs a 'path' or a "
                                     "'version' (git URL) naming the project to build");
        if (d.build != "sources" && (!d.sources.empty() || !d.defines.empty() || !d.cflags.empty()))
            throw std::runtime_error(where + "sources, defines and cflags need build = \"sources\""
                                     + (d.build == "cmake" ? " (pass CMake settings in cmake_args)" : ""));
        if (d.build != "cmake" && !d.cmakeArgs.empty())
            throw std::runtime_error(where + "cmake_args needs build = \"cmake\"");
        if (d.build == "sources" && !d.libs.empty())
            throw std::runtime_error(where + "build = \"sources\" produces its own library; drop 'lib'");
    }

    if (m.bundle.present) {
        if (m.bundle.identifier.empty())
            throw std::runtime_error(srcName + ": [bundle] identifier is required "
                                     "(a reverse-DNS name such as \"org.example." + m.package.name + "\")");
        for (char c : m.bundle.identifier)
            if (!std::isalnum((unsigned char)c) && c != '.' && c != '-')
                throw std::runtime_error(srcName + ": bundle.identifier may only contain letters, "
                                         "digits, '.' and '-', got \"" + m.bundle.identifier + "\"");
        if (m.bundle.name.empty()) m.bundle.name = m.package.name;
        if (m.bundle.version.empty()) m.bundle.version = m.package.version;
        if (m.bundle.version.empty() || !std::isdigit((unsigned char)m.bundle.version[0]))
            throw std::runtime_error(srcName + ": the bundle version must start with a digit "
                                     "(e.g. \"1.2.0\"), got \"" + m.bundle.version + "\"");
        if (m.bundle.subsystem.empty()) m.bundle.subsystem = "windows";
        else if (m.bundle.subsystem != "windows" && m.bundle.subsystem != "console")
            throw std::runtime_error(srcName + ": bundle.subsystem must be \"windows\" or "
                                     "\"console\", got \"" + m.bundle.subsystem + "\"");
        if (m.bundle.minSystemVersion.empty()) m.bundle.minSystemVersion = "11.0";
        if (m.bundle.codesignIdentity.empty()) m.bundle.codesignIdentity = "-";
        if (!m.bundle.icon.empty()) {
            std::string ext = m.bundle.icon.substr(m.bundle.icon.find_last_of('.') == std::string::npos
                                                   ? m.bundle.icon.size() : m.bundle.icon.find_last_of('.'));
            if (ext != ".png" && ext != ".icns" && ext != ".ico")
                throw std::runtime_error(srcName + ": bundle.icon must be a .png, .icns or .ico file, got \"" +
                                         m.bundle.icon + "\"");
        }
    }

    return m;
}

// ── public API ────────────────────────────────────────────────────────────────

Manifest ManifestParser::parseFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return parseString(src, path);
}

Manifest ManifestParser::parseString(const std::string& src,
                                      const std::string& srcName) {
    auto tokens = tokenize(src, srcName);
    return buildManifest(tokens, srcName);
}

std::string ManifestParser::serialize(const Manifest& m) {
    std::ostringstream o;
    o << "[package]\n";
    o << "name        = \"" << m.package.name        << "\"\n";
    o << "version     = \"" << m.package.version     << "\"\n";
    o << "author      = \"" << m.package.author      << "\"\n";
    if (!m.package.description.empty())
        o << "description = \"" << m.package.description << "\"\n";
    if (!m.package.license.empty())
        o << "license     = \"" << m.package.license     << "\"\n";

    o << "\n[build]\n";
    o << "edition = \"" << m.build.edition << "\"\n";
    if (!m.build.output.empty() && m.build.output != m.package.name)
        o << "output  = \"" << m.build.output << "\"\n";
    if (!m.build.srcs.empty()) {
        o << "srcs    = [";
        for (size_t i = 0; i < m.build.srcs.size(); ++i)
            o << (i ? ", " : "") << "\"" << m.build.srcs[i] << "\"";
        o << "]\n";
    }
    if (!m.build.cflags.empty()) {
        o << "cflags  = [";
        for (size_t i = 0; i < m.build.cflags.size(); ++i)
            o << (i ? ", " : "") << "\"" << m.build.cflags[i] << "\"";
        o << "]\n";
    }
    if (!m.build.libs.empty()) {
        o << "libs     = [";
        for (size_t i = 0; i < m.build.libs.size(); ++i)
            o << (i ? ", " : "") << "\"" << m.build.libs[i] << "\"";
        o << "]\n";
    }
    if (!m.build.libDirs.empty()) {
        o << "lib_dirs = [";
        for (size_t i = 0; i < m.build.libDirs.size(); ++i)
            o << (i ? ", " : "") << "\"" << m.build.libDirs[i] << "\"";
        o << "]\n";
    }
    if (!m.build.lto.empty())
        o << "lto        = \"" << m.build.lto << "\"\n";
    if (!m.build.blas.empty())
        o << "blas       = \"" << m.build.blas << "\"\n";
    if (!m.build.crateType.empty() && m.build.crateType != "bin")
        o << "crate_type = \"" << m.build.crateType << "\"\n";
    if (!m.build.frameworks.empty()) {
        o << "frameworks = [";
        for (size_t i = 0; i < m.build.frameworks.size(); ++i)
            o << (i ? ", " : "") << "\"" << m.build.frameworks[i] << "\"";
        o << "]\n";
    }
    if (m.build.objcArc)
        o << "objc_arc   = true\n";
    if (m.build.compatPreprocessor)
        o << "compat_preprocessor = true\n";
    if (m.build.useStd >= 0)
        o << "std        = " << (m.build.useStd == 2 ? "\"all\"" : m.build.useStd ? "true" : "false") << "\n";
    if (!m.build.stdPrebuilt) o << "std_prebuilt = false\n";

    for (auto& d : m.dependencies) {
        o << "\n[[dependencies]]\n";
        o << "name    = \"" << d.name    << "\"\n";
        if (!d.version.empty())
            o << "version = \"" << d.version << "\"\n";
        if (!d.path.empty())
            o << "path    = \"" << d.path    << "\"\n";
        auto list = [&](const char* key, const std::vector<std::string>& v) {
            if (v.empty()) return;
            o << key << "[";
            for (size_t i = 0; i < v.size(); ++i)
                o << (i ? ", " : "") << "\"" << v[i] << "\"";
            o << "]\n";
        };
        list("lib     = ", d.libs);
        list("include = ", d.includes);
        if (!d.build.empty())
            o << "build   = \"" << d.build << "\"\n";
        list("sources = ", d.sources);
        list("defines = ", d.defines);
        list("cflags  = ", d.cflags);
        list("cmake_args = ", d.cmakeArgs);
        list("frameworks = ", d.frameworks);
    }

    if (!m.features.empty()) {
        o << "\n[features]\n";
        for (auto& [name, deps] : m.features) {
            o << name << " = [";
            for (size_t i = 0; i < deps.size(); ++i)
                o << (i ? ", " : "") << "\"" << deps[i] << "\"";
            o << "]\n";
        }
    }

    return o.str();
}

// ── feature resolution ──────────────────────────────────────────────────────

std::vector<std::string> resolveFeatures(const Manifest& m,
                                          const std::vector<std::string>& requested,
                                          bool noDefault) {
    std::vector<std::string> enabled;
    std::vector<std::string> seenNames;

    auto isSeen = [&](const std::string& n) {
        for (auto& s : seenNames) if (s == n) return true;
        return false;
    };
    auto lookup = [&](const std::string& n) -> const std::vector<std::string>* {
        for (auto& [name, deps] : m.features) if (name == n) return &deps;
        return nullptr;
    };

    std::vector<std::string> worklist;
    if (!noDefault) {
        if (auto* def = lookup("default")) worklist = *def;
    }
    for (auto& r : requested) worklist.push_back(r);

    for (size_t i = 0; i < worklist.size(); ++i) {
        const std::string& name = worklist[i];
        if (isSeen(name)) continue;
        seenNames.push_back(name);
        enabled.push_back(name);
        if (auto* subDeps = lookup(name))
            for (auto& d : *subDeps) worklist.push_back(d);
    }
    return enabled;
}

} // namespace safeguard
