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
    m.build.edition = "2025";
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

        // dependencies.*
        else if (tok.key == "dependencies.name")    curDep.name    = tok.value;
        else if (tok.key == "dependencies.version") curDep.version = tok.value;
        else if (tok.key == "dependencies.path")    curDep.path    = tok.value;
    }
    flushDep();

    if (m.package.name.empty())
        throw std::runtime_error(srcName + ": [package] name is required");
    if (m.package.version.empty())
        m.package.version = "0.1.0";
    if (m.build.output.empty())
        m.build.output = m.package.name;

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

    for (auto& d : m.dependencies) {
        o << "\n[[dependencies]]\n";
        o << "name    = \"" << d.name    << "\"\n";
        o << "version = \"" << d.version << "\"\n";
        if (!d.path.empty())
            o << "path    = \"" << d.path    << "\"\n";
    }
    return o.str();
}

} // namespace safeguard
