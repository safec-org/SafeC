#include "StdModules.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;

namespace safeguard {

// ── platform-variant std sources ──────────────────────────────────────────────
// Several std modules ship one implementation per OS API, distinguished by a
// filename suffix (sched/io_nb_{bsd,linux,win32}.sc, sched/reactor_{kqueue,
// epoll,win32}.sc, ipc/uds_{bsd,linux}.sc, ipc/pipe_{posix,win32}.sc,
// gui/gui_{cocoa,x11,win32,fb}.sc — fb being the bare-metal one,
// audio/audio_{alsa_linux,jack_posix,coreaudio_macos,wasapi_win32,
// webaudio_wasm}.sc — wasm being WebAssembly only). They define the same symbols, so only the
// target's variant may go into a build: with all of them archived, the
// linker resolves each symbol from whichever member it meets first
// (alphabetically the BSD one, whose sockaddr layout is wrong on Linux) and
// then fails on the duplicates the next pulled-in member brings along.
bool stdVariantMatches(const fs::path& src, const std::string& triple, bool bareMetal) {
    std::string stem = src.stem().string();
    auto endsWith = [&](const char* suf) {
        std::string s(suf);
        return stem.size() > s.size() &&
               stem.compare(stem.size() - s.size(), s.size(), s) == 0;
    };
    bool win, mac, linux_, bsd;
    if (triple.empty()) {
#if defined(_WIN32)
        win = true; mac = false; linux_ = false; bsd = false;
#elif defined(__APPLE__)
        win = false; mac = true; linux_ = false; bsd = false;
#elif defined(__linux__)
        win = false; mac = false; linux_ = true; bsd = false;
#else
        win = false; mac = false; linux_ = false; bsd = true;
#endif
    } else {
        win = triple.find("windows") != std::string::npos;
        mac = triple.find("apple") != std::string::npos || triple.find("darwin") != std::string::npos;
        linux_ = triple.find("linux") != std::string::npos;
        bsd = triple.find("bsd") != std::string::npos;
    }
    // bare-metal framebuffer backend (see gui_fb.sc); WebAssembly, which
    // safeguard also treats as bare metal, has its own (_wasm: gui_web_wasm.sc)
    if (endsWith("_fb")) return bareMetal && triple.rfind("wasm", 0) != 0;
    // WebAssembly host (browser/Node) backends: JS imports, see audio_webaudio_wasm.sc.
    if (endsWith("_wasm")) return triple.rfind("wasm", 0) == 0;
    if (bareMetal) {
        // No OS API at all: none of the OS-specific backends apply.
        if (endsWith("_linux") || endsWith("_epoll") || endsWith("_bsd") || endsWith("_kqueue") ||
            endsWith("_cocoa") || endsWith("_macos") || endsWith("_x11") || endsWith("_win32") ||
            endsWith("_posix"))
            return false;
        return true;
    }
    if (endsWith("_linux") || endsWith("_epoll")) return linux_;
    if (endsWith("_bsd") || endsWith("_kqueue")) return mac || bsd;
    if (endsWith("_cocoa") || endsWith("_macos")) return mac;
    if (endsWith("_x11")) return linux_ || bsd;
    if (endsWith("_win32")) return win;
    if (endsWith("_posix")) return !win;
    return true;
}

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

std::vector<std::string> words(const std::string& s) {
    std::istringstream ss(s);
    std::vector<std::string> out;
    std::string w;
    while (ss >> w) out.push_back(w);
    return out;
}

bool isIdent(const std::string& s) {
    if (s.empty() || std::isdigit((unsigned char)s[0])) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '_'; });
}

// A condition this can decide later: one (possibly negated) macro test.
// 'defined(X)', 'defined X', 'X', each optionally behind '!'.
bool parseCondition(std::string c, StdIndex::Guard& g) {
    size_t slash = c.find("//");
    if (slash != std::string::npos) c = c.substr(0, slash);
    slash = c.find("/*");
    if (slash != std::string::npos) c = c.substr(0, slash);
    c = trim(c);
    bool neg = false;
    if (!c.empty() && c[0] == '!') { neg = true; c = trim(c.substr(1)); }
    if (c.rfind("defined", 0) == 0) {
        c = trim(c.substr(7));
        if (!c.empty() && c.front() == '(' && c.back() == ')') c = trim(c.substr(1, c.size() - 2));
    }
    if (!isIdent(c)) return false;
    g.macro = c;
    g.defined = !neg;
    return true;
}

// '<std/x.h>' / '<std/x.sc>' in a directive argument: the std-relative path.
std::string directivePath(std::string a) {
    if (a.size() >= 2 && (a.front() == '<' || a.front() == '"')) a = a.substr(1, a.size() - 2);
    if (a.rfind("std/", 0) == 0) a = a.substr(4);
    return a;
}

// ${VAR} and ${VAR:-default} in a link flag.
std::string expandVars(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '$' && i + 1 < s.size() && s[i + 1] == '{') {
            size_t close = s.find('}', i);
            if (close != std::string::npos) {
                std::string body = s.substr(i + 2, close - i - 2), def;
                size_t dash = body.find(":-");
                if (dash != std::string::npos) { def = body.substr(dash + 2); body = body.substr(0, dash); }
                const char* v = std::getenv(body.c_str());
                out += (v && *v) ? std::string(v) : def;
                i = close;
                continue;
            }
        }
        out += s[i];
    }
    return out;
}

std::string featureMacro(const std::string& name) {
    std::string upper = "SAFEC_FEATURE_";
    for (char c : name) upper += (char)std::toupper((unsigned char)(c == '-' ? '_' : c));
    return upper;
}

} // namespace

std::vector<StdIndex::Include> StdIndex::scanIncludes(const fs::path& file) {
    std::vector<Include> out;
    std::ifstream in(file);
    if (!in) return out;
    // One frame per open #if: its decidable guard, if the branch has one.
    struct Frame { bool has; Guard g; };
    std::vector<Frame> stack;
    std::string line;
    bool inComment = false;
    while (std::getline(in, line)) {
        if (inComment) {
            size_t end = line.find("*/");
            if (end == std::string::npos) continue;
            line = line.substr(end + 2);
            inComment = false;
        }
        std::string t = trim(line);
        if (t.rfind("/*", 0) == 0 && t.find("*/") == std::string::npos) { inComment = true; continue; }
        if (t.empty() || t[0] != '#') continue;
        t = trim(t.substr(1));
        size_t sp = 0;
        while (sp < t.size() && std::isalpha((unsigned char)t[sp])) ++sp;
        std::string dir = t.substr(0, sp), rest = trim(t.substr(sp));
        if (dir == "include") {
            if (rest.size() < 2 || (rest[0] != '<' && rest[0] != '"')) continue;
            char close = rest[0] == '<' ? '>' : '"';
            size_t e = rest.find(close, 1);
            if (e == std::string::npos) continue;
            Include inc;
            inc.name = rest.substr(1, e - 1);
            inc.angled = rest[0] == '<';
            for (auto& f : stack) if (f.has) inc.guards.push_back(f.g);
            out.push_back(inc);
        } else if (dir == "if" || dir == "ifdef" || dir == "ifndef") {
            Frame f{false, {}};
            if (dir == "if") f.has = parseCondition(rest, f.g);
            else if (isIdent(trim(rest))) {
                f.has = true;
                f.g.macro = trim(rest);
                f.g.defined = dir == "ifdef";
            }
            stack.push_back(f);
        } else if (dir == "elif" || dir == "elifdef" || dir == "elifndef") {
            if (!stack.empty()) stack.back().has = false;   // "not the earlier ones, and ..."
        } else if (dir == "else") {
            if (!stack.empty()) stack.back().g.defined = !stack.back().g.defined;
        } else if (dir == "endif") {
            if (!stack.empty()) stack.pop_back();
        }
    }
    return out;
}

const StdIndex& StdIndex::load(const std::string& stdDir) {
    static std::map<std::string, std::unique_ptr<StdIndex>> cache;
    auto& slot = cache[stdDir];
    if (slot) return *slot;
    slot = std::make_unique<StdIndex>();
    StdIndex& idx = *slot;
    idx.dir = stdDir;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(stdDir, ec); !ec && it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        std::string ext = it->path().extension().string();
        if (ext != ".sc" && ext != ".h") continue;
        File f;
        f.rel = fs::relative(it->path(), stdDir).generic_string();
        f.header = ext == ".h";
        f.includes = scanIncludes(it->path());
        // The '// safeguard:' directives.
        std::ifstream in(it->path());
        std::string line;
        while (std::getline(in, line)) {
            std::string t = trim(line);
            if (t.rfind("//", 0) != 0) continue;
            t = trim(t.substr(2));
            if (t.rfind("safeguard:", 0) != 0) continue;
            // A '(' starts a remark: '// safeguard: runtime (what for)'.
            auto w = words(t.substr(10, t.find('(') == std::string::npos ? std::string::npos : t.find('(') - 10));
            if (w.empty()) continue;
            std::string kind = w[0];
            std::vector<std::string> args(w.begin() + 1, w.end());
            auto bad = [&](const std::string& why) {
                idx.problems.push_back("std/" + f.rel + ": '// safeguard: " + kind + "': " + why);
            };
            if (kind == "implements" || kind == "sources") {
                if ((kind == "implements") == f.header) { bad(f.header ? "only in a .sc" : "only in a .h"); continue; }
                if (args.empty()) { bad("names no file"); continue; }
                for (auto& a : args) (kind == "implements" ? f.implements : f.sources).push_back(directivePath(a));
            } else if (kind == "feature") {
                if (f.header || args.empty()) { bad("a .sc names its features"); continue; }
                f.features.insert(f.features.end(), args.begin(), args.end());
            } else if (kind == "runtime") {
                if (f.header) { bad("only in a .sc"); continue; }
                f.runtime = true;
                f.runtimeOs = args;
            } else if (kind == "runtime-symbol") {
                if (f.header || args.empty()) { bad("a .sc names its symbol prefixes"); continue; }
                f.runtimeSymbols.insert(f.runtimeSymbols.end(), args.begin(), args.end());
            } else if (kind == "link" || kind == "deb" || kind == "deb-recommends") {
                if (f.header || args.size() < 2) { bad("expects '<os> <items...>' in a .sc"); continue; }
                LinkRule r{args[0], std::vector<std::string>(args.begin() + 1, args.end())};
                (kind == "link" ? f.links : kind == "deb" ? f.debs : f.debRecommends).push_back(r);
            } else {
                bad("unknown directive");
            }
        }
        idx.files[f.rel] = std::move(f);
    }
    // Which header each unit implements: foo_bar_baz.sc -> the first of
    // foo_bar_baz.h, foo_bar.h, foo.h beside it; plus 'implements' and
    // a header's 'sources'.
    for (auto& [rel, f] : idx.files) {
        if (f.header) {
            for (auto& s : f.sources) {
                if (!idx.files.count(s)) idx.problems.push_back("std/" + rel + ": sources std/" + s + ": no such file");
                else idx.units[rel].push_back(s);
            }
            continue;
        }
        bool owned = false;
        fs::path p(rel);
        std::string stem = p.stem().string(), parent = p.parent_path().generic_string();
        while (true) {
            std::string h = (parent.empty() ? "" : parent + "/") + stem + ".h";
            if (idx.files.count(h)) { idx.units[h].push_back(rel); owned = true; break; }
            size_t us = stem.rfind('_');
            if (us == std::string::npos || us == 0) break;
            stem = stem.substr(0, us);
        }
        for (auto& h : f.implements) {
            if (!idx.files.count(h)) { idx.problems.push_back("std/" + rel + ": implements std/" + h + ": no such file"); continue; }
            idx.units[h].push_back(rel);
            owned = true;
        }
        if (!owned && !f.runtime && f.runtimeSymbols.empty())
            idx.problems.push_back("std/" + rel + ": implements no header (name it with '// safeguard: implements <std/...h>')");
    }
    for (auto& [h, us] : idx.units) {
        std::sort(us.begin(), us.end());
        us.erase(std::unique(us.begin(), us.end()), us.end());
    }
    return idx;
}

std::string StdIndex::resolve(const std::string& includeName) const {
    std::string n = includeName;
    if (n.rfind("std/", 0) != 0) return "";
    n = n.substr(4);
    return files.count(n) ? n : "";
}

bool StdIndex::osMatches(const std::string& sel, const StdPlatform& p) {
    if (sel == "any") return true;
    if (sel == "hosted") return p.os != "baremetal" && p.os != "wasm";
    if (sel == "unix") return p.os == "macos" || p.os == "linux" || p.os == "bsd";
    return sel == p.os;
}

bool StdIndex::guardsHold(const Include& inc, const StdPlatform& p, bool userFile, bool stdUnit) {
    // The OS macros safec defines from the target triple (compiler/src/main.cpp).
    std::string t = p.triple;
    if (t.empty()) {
#if defined(__APPLE__)
        t = "apple";
#elif defined(_WIN32)
        t = "windows";
#elif defined(__linux__)
        t = "linux";
#endif
    }
    bool apple = t.find("apple") != std::string::npos || t.find("darwin") != std::string::npos;
    bool win = !apple && (t.find("windows") != std::string::npos || t.find("win32") != std::string::npos);
    bool lin = !apple && !win && t.find("linux") != std::string::npos;
    for (auto& g : inc.guards) {
        int known = -1;   // -1 undecided, 0 undefined, 1 defined
        if (g.macro == "__APPLE__") known = apple;
        else if (g.macro == "_WIN32") known = win;
        else if (g.macro == "__linux__") known = lin;
        else if (g.macro.rfind("SAFEC_FEATURE_", 0) == 0) {
            if (stdUnit) known = 0;
            else if (userFile) {
                known = 0;
                for (auto& f : p.features) if (featureMacro(f) == g.macro) known = 1;
            }
        }
        if (known != -1 && (known == 1) != g.defined) return false;
    }
    return true;
}

bool StdIndex::unitApplies(const File& f, const StdPlatform& p) const {
    if (f.header) return false;
    if (!stdVariantMatches(fs::path(f.rel), p.triple, p.bareMetal)) return false;
    // std/wasm/: the WebAssembly runtime (malloc & co.) — never beside a libc.
    if (f.rel.rfind("wasm/", 0) == 0 && !p.wasm) return false;
    if (!f.features.empty()) {
        bool any = false;
        for (auto& want : f.features)
            for (auto& have : p.features)
                if (featureMacro(want) == featureMacro(have)) any = true;
        if (!any) return false;
    }
    return true;
}

std::set<std::string> StdIndex::closure(const std::set<std::string>& roots, const StdPlatform& p) const {
    std::set<std::string> seen, out;
    std::vector<std::string> work(roots.begin(), roots.end());
    for (auto& [rel, f] : files) {
        if (!f.runtime) continue;
        bool os = f.runtimeOs.empty();
        for (auto& s : f.runtimeOs) if (osMatches(s, p)) os = true;
        if (os) work.push_back(rel);
    }
    while (!work.empty()) {
        std::string rel = work.back();
        work.pop_back();
        if (!seen.insert(rel).second) continue;
        auto it = files.find(rel);
        if (it == files.end()) continue;
        const File& f = it->second;
        if (!f.header) {
            if (!unitApplies(f, p)) continue;
            out.insert(rel);
        }
        for (auto& inc : f.includes) {
            if (!guardsHold(inc, p, false, !f.header)) continue;
            std::string r = resolve(inc.name);
            if (!r.empty()) work.push_back(r);
        }
        if (f.header) {
            auto u = units.find(rel);
            if (u != units.end()) for (auto& s : u->second) work.push_back(s);
        }
    }
    return out;
}

std::set<std::string> StdIndex::runtimeUnits(const std::set<std::string>& refs, const StdPlatform& p) const {
    std::set<std::string> out;
    for (auto& [rel, f] : files) {
        if (f.header || f.runtimeSymbols.empty() || !unitApplies(f, p)) continue;
        for (auto& pre : f.runtimeSymbols) {
            auto it = refs.lower_bound(pre);
            if (it != refs.end() && it->compare(0, pre.size(), pre) == 0) { out.insert(rel); break; }
        }
    }
    return out;
}

std::set<std::string> StdIndex::runtimeSymbolPrefixes() const {
    std::set<std::string> out;
    for (auto& [rel, f] : files) out.insert(f.runtimeSymbols.begin(), f.runtimeSymbols.end());
    return out;
}

std::set<std::string> StdIndex::allUnits(const StdPlatform& p, bool featureBackends) const {
    StdPlatform any = p;
    std::set<std::string> out;
    for (auto& [rel, f] : files) {
        if (f.header) continue;
        any.features.clear();
        if (featureBackends)
            for (auto& w : f.features) any.features.insert(w);   // every backend, as before
        if (unitApplies(f, any)) out.insert(rel);
    }
    return out;
}

std::set<std::string> irExternalRefs(const fs::path& llFile) {
    std::set<std::string> out;
    std::ifstream in(llFile);
    std::string line;
    while (std::getline(in, line)) {
        // 'declare <type> @name(' and '@name = external global ...'.
        bool decl = line.rfind("declare ", 0) == 0;
        if (!decl && !(line.size() > 1 && line[0] == '@' && line.find(" = external ") != std::string::npos))
            continue;
        size_t at = line.find('@');
        if (at == std::string::npos) continue;
        std::string name;
        if (at + 1 < line.size() && line[at + 1] == '"') {
            size_t close = line.find('"', at + 2);
            if (close == std::string::npos) continue;
            name = line.substr(at + 2, close - at - 2);
        } else {
            size_t e = at + 1;
            while (e < line.size() && (std::isalnum((unsigned char)line[e]) || line[e] == '_' ||
                                       line[e] == '.' || line[e] == '$'))
                ++e;
            name = line.substr(at + 1, e - at - 1);
        }
        if (!name.empty() && name.rfind("llvm.", 0) != 0) out.insert(name);
    }
    return out;
}

bool readRefs(const fs::path& file, std::set<std::string>& out) {
    std::ifstream in(file);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.insert(line);
    }
    return true;
}

void writeRefs(const fs::path& file, const std::set<std::string>& refs) {
    std::ofstream out(file, std::ios::binary);
    for (auto& r : refs) out << r << "\n";
}

std::vector<std::string> StdIndex::linkFlags(const std::set<std::string>& unitSet, const StdPlatform& p) const {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (auto& u : unitSet) {
        auto it = files.find(u);
        if (it == files.end()) continue;
        for (auto& r : it->second.links) {
            if (!osMatches(r.os, p)) continue;
            for (size_t i = 0; i < r.flags.size(); ++i) {
                std::string flag = expandVars(r.flags[i]);
                if (flag == "-framework" && i + 1 < r.flags.size()) {
                    std::string name = r.flags[++i];
                    if (seen.insert("-framework " + name).second) { out.push_back("-framework"); out.push_back(name); }
                    continue;
                }
                if (flag.rfind("-L", 0) == 0) {
                    std::error_code ec;
                    if (!fs::is_directory(flag.substr(2), ec)) continue;   // not installed here
                    if (seen.insert(flag).second) {
                        out.push_back(flag);
                        if (p.os != "windows") out.push_back("-Wl,-rpath," + flag.substr(2));
                    }
                    continue;
                }
                if (seen.insert(flag).second) out.push_back(flag);
            }
        }
    }
    return out;
}

std::vector<std::string> StdIndex::debDepends(const std::set<std::string>& unitSet, const StdPlatform& p) const {
    std::vector<std::string> out;
    for (auto& u : unitSet) {
        auto it = files.find(u);
        if (it == files.end()) continue;
        for (auto& r : it->second.debs)
            if (osMatches(r.os, p))
                for (auto& d : r.flags)
                    if (std::find(out.begin(), out.end(), d) == out.end()) out.push_back(d);
    }
    return out;
}

std::vector<std::string> StdIndex::debRecommends(const std::set<std::string>& unitSet, const StdPlatform& p) const {
    std::vector<std::string> out;
    for (auto& u : unitSet) {
        auto it = files.find(u);
        if (it == files.end()) continue;
        for (auto& r : it->second.debRecommends)
            if (osMatches(r.os, p))
                for (auto& d : r.flags)
                    if (std::find(out.begin(), out.end(), d) == out.end()) out.push_back(d);
    }
    return out;
}

std::set<std::string> StdIndex::compileInputs(const std::string& unit) const {
    std::set<std::string> seen;
    std::vector<std::string> work{unit};
    while (!work.empty()) {
        std::string rel = work.back();
        work.pop_back();
        if (!seen.insert(rel).second) continue;
        auto it = files.find(rel);
        if (it == files.end()) continue;
        for (auto& inc : it->second.includes) {
            std::string r = resolve(inc.name);
            if (!r.empty()) work.push_back(r);
        }
    }
    return seen;
}

std::set<std::string> stdRootsOf(const StdIndex& idx, const std::vector<std::string>& files,
                                 const std::vector<std::string>& includeDirs, const StdPlatform& p) {
    std::set<std::string> roots;
    std::set<std::string> seen;
    std::vector<fs::path> work;
    for (auto& f : files) work.push_back(fs::path(f));
    std::error_code ec;
    fs::path stdDir = fs::weakly_canonical(idx.dir, ec);
    auto underStd = [&](const fs::path& f) {
        fs::path c = fs::weakly_canonical(f, ec);
        auto r = c.lexically_relative(stdDir);
        return !r.empty() && r.native()[0] != '.';
    };
    while (!work.empty()) {
        fs::path f = work.back();
        work.pop_back();
        if (!seen.insert(fs::weakly_canonical(f, ec).string()).second) continue;
        for (auto& inc : StdIndex::scanIncludes(f)) {
            if (!StdIndex::guardsHold(inc, p, true, false)) continue;
            std::string r = idx.resolve(inc.name);
            if (!r.empty()) { roots.insert(r); continue; }
            // The project's own (or a dependency's) header: scan it too.
            std::vector<fs::path> cands;
            if (!inc.angled) cands.push_back(f.parent_path() / inc.name);
            for (auto& d : includeDirs) cands.push_back(fs::path(d) / inc.name);
            for (auto& c : cands) {
                if (fs::is_regular_file(c, ec) && !underStd(c)) { work.push_back(c); break; }
            }
            // A bare <io.h> is not std's: std/ itself is not on the include
            // path (only its parent, for <std/...>), so it is a system header.
        }
    }
    return roots;
}

} // namespace safeguard
