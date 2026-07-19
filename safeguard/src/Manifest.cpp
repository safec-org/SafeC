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
            // Section header
            size_t close = line.find(']');
            if (close == std::string::npos)
                throw std::runtime_error(srcName + ":" + std::to_string(lineno) +
                                         ": unterminated section header");
            // Handle [[dependencies]] array-of-table header
            std::string inner = trim(line.substr(1, close - 1));
            // Strip extra [ ] for array-of-table
            if (!inner.empty() && inner.front() == '[' && inner.back() == ']')
                inner = trim(inner.substr(1, inner.size() - 2));
            section = inner;
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

    for (auto& tok : tokens) {
        if (tok.key == "__section__") {
            if (tok.value == "dependencies") {
                flushDep();
                inDep = true;
            } else {
                flushDep();
            }
            continue;
        }

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
        else if (tok.key == "build.crate_type") m.build.crateType = tok.value;

        // dependencies.*
        else if (tok.key == "dependencies.name")    curDep.name    = tok.value;
        else if (tok.key == "dependencies.version") curDep.version = tok.value;
        else if (tok.key == "dependencies.path")    curDep.path    = tok.value;

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
    if (!m.build.lto.empty() && m.build.lto != "thin" && m.build.lto != "full")
        throw std::runtime_error(srcName + ": build.lto must be \"thin\" or "
                                 "\"full\", got \"" + m.build.lto + "\"");

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
    if (!m.build.crateType.empty() && m.build.crateType != "bin")
        o << "crate_type = \"" << m.build.crateType << "\"\n";

    for (auto& d : m.dependencies) {
        o << "\n[[dependencies]]\n";
        o << "name    = \"" << d.name    << "\"\n";
        o << "version = \"" << d.version << "\"\n";
        if (!d.path.empty())
            o << "path    = \"" << d.path    << "\"\n";
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
