#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace safeguard {

// One dependency entry in the lock file.
struct LockedDep {
    std::string name;
    std::string url;       // original URL from Package.toml
    std::string git_sha;   // exact git SHA1 at time of lock
};

// The full lock file (Package.lock).
struct LockFile {
    std::string              safec_hash;  // FNV-1a hex hash of safec binary
    std::vector<LockedDep>   deps;
    // source file path → FNV-1a hex hash
    std::unordered_map<std::string, std::string> sources;
};

// ── File I/O ──────────────────────────────────────────────────────────────────

// Write a LockFile to `path`. Throws std::runtime_error on failure.
void lockWrite(const LockFile& lf, const std::string& path);

// Read a LockFile from `path`. Returns false if file does not exist.
// Throws std::runtime_error on malformed content.
bool lockRead(LockFile& lf, const std::string& path);

// ── Hashing utilities ─────────────────────────────────────────────────────────

// FNV-1a 64-bit hash of the file at `path`.
// Returns the 16-character hex string, or "" on read error.
std::string hashFile(const std::string& path);

// ── Git utilities ─────────────────────────────────────────────────────────────

// Run `git rev-parse HEAD` in `repoDir`.
// Returns the SHA1 string, or "" on error.
std::string gitRevParse(const std::string& repoDir);

} // namespace safeguard
