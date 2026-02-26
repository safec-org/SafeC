#include "Lock.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <array>

namespace safeguard {

// ── FNV-1a 64-bit hash ────────────────────────────────────────────────────────

static uint64_t fnv1a64(const uint8_t* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string toHex16(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx",
             static_cast<unsigned long long>(v));
    return buf;
}

std::string hashFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    uint64_t h = 14695981039346656037ULL;
    char buf[4096];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        auto n = static_cast<size_t>(f.gcount());
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(static_cast<uint8_t>(buf[i]));
            h *= 1099511628211ULL;
        }
    }
    return toHex16(h);
}

// ── Git utilities ─────────────────────────────────────────────────────────────

std::string gitRevParse(const std::string& repoDir) {
    // Run: git -C <repoDir> rev-parse HEAD
    std::string cmd = "git -C \"" + repoDir + "\" rev-parse HEAD 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[64] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, pipe);
    pclose(pipe);
    // Strip trailing newline.
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) --n;
    buf[n] = '\0';
    return std::string(buf, n);
}

// ── Lock file serialisation ───────────────────────────────────────────────────
//
// Format (simple TOML-like, hand-parseable):
//
//   [safec]
//   hash = "abc123..."
//
//   [dep.mylib]
//   url = "https://..."
//   git_sha = "deadbeef..."
//
//   [sources]
//   "src/main.sc" = "hash..."
//

void lockWrite(const LockFile& lf, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("lock: cannot write " + path);

    f << "# SafeC Package.lock — auto-generated, do not edit manually\n\n";

    f << "[safec]\n";
    f << "hash = \"" << lf.safec_hash << "\"\n\n";

    for (auto& dep : lf.deps) {
        f << "[dep." << dep.name << "]\n";
        f << "url = \"" << dep.url << "\"\n";
        f << "git_sha = \"" << dep.git_sha << "\"\n\n";
    }

    if (!lf.sources.empty()) {
        f << "[sources]\n";
        // Sort keys for reproducibility.
        std::vector<std::string> keys;
        for (auto& kv : lf.sources) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (auto& k : keys) {
            f << "\"" << k << "\" = \"" << lf.sources.at(k) << "\"\n";
        }
        f << "\n";
    }
}

bool lockRead(LockFile& lf, const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;

    lf = {};
    std::string section;
    std::string depName;
    std::string line;

    while (std::getline(f, line)) {
        // Strip comment.
        auto cpos = line.find('#');
        if (cpos != std::string::npos) line = line.substr(0, cpos);
        // Trim whitespace.
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
            line.erase(line.begin());
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' ||
                                  line.back() == '\r'))
            line.pop_back();
        if (line.empty()) continue;

        // Section header.
        if (line.front() == '[') {
            section = line.substr(1, line.size() - 2);
            if (section.substr(0, 4) == "dep.") {
                depName = section.substr(4);
                lf.deps.push_back({depName, "", ""});
                section = "dep";
            }
            continue;
        }

        // Key = "value"
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        // Strip leading quotes from key (for sources section).
        if (!key.empty() && key.front() == '"') {
            key = key.substr(1, key.size() - 2);
        }
        std::string val = line.substr(eq + 1);
        while (!val.empty() && val.front() == ' ') val.erase(val.begin());
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if (section == "safec") {
            if (key == "hash") lf.safec_hash = val;
        } else if (section == "dep") {
            if (!lf.deps.empty()) {
                if (key == "url")     lf.deps.back().url = val;
                if (key == "git_sha") lf.deps.back().git_sha = val;
            }
        } else if (section == "sources") {
            lf.sources[key] = val;
        }
    }
    return true;
}

} // namespace safeguard
