// 'safeguard doc': a project's (or std's) API reference.
//
//   1. safec --emit-doc-json over the package's headers (include/, src/)
//      and SafeC sources (src/): every declaration, its doc comment,
//      signature and location (std's items too, as links);
//   2. safec-doc (safeguard/doctool, written in SafeC) turns that into
//      build/doc/: per-item and per-module pages, source pages, a search
//      index the wasm client searches, and checks every link.
//
// safec-doc ships prebuilt in a release (bin/safec-doc, its assets in
// share/safec/doctool/theme); without one (a from-source checkout) it is
// built on first use from its SafeC sources — $SAFEC_HOME/safeguard/doctool
// or <prefix>/share/safec/doctool — into a per-user cache keyed by those
// sources and the compiler, with this same safeguard.
#include "Builder.h"
#include "Lock.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#else
#include <climits>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace safeguard {

namespace {

#ifdef _WIN32
const char* kExe = ".exe";
#else
const char* kExe = "";
#endif

const char* kDefaultStdDocs = "https://safec-org.github.io/safec-docs/api/";

// This safeguard's own executable.
fs::path selfExe() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return fs::path(std::wstring(buf, n));
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    uint32_t size = sizeof buf;
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::error_code ec;
        fs::path p = fs::canonical(buf, ec);
        return ec ? fs::path(buf) : p;
    }
#else
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p;
#endif
    return fs::path("safeguard");
}

std::string jsonStr(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
            else out += (char)c;
        }
    }
    return out + "\"";
}

std::string slash(const fs::path& p) { return p.generic_string(); }

// A response-file argument: quoted, '/'-separated.
std::string rspArg(const std::string& a) {
    std::string out = "\"";
    for (char c : a) {
        if (c == '\\') out += '/';
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out + "\"";
}

fs::path cacheRoot() {
    if (const char* e = std::getenv("SAFEGUARD_CACHE"); e && *e) return e;
#ifdef _WIN32
    if (const char* e = std::getenv("LOCALAPPDATA"); e && *e) return fs::path(e) / "safeguard";
#endif
    if (const char* e = std::getenv("XDG_CACHE_HOME"); e && *e) return fs::path(e) / "safeguard";
    if (const char* e = std::getenv("HOME"); e && *e) return fs::path(e) / ".cache" / "safeguard";
    return fs::temp_directory_path() / "safeguard-cache";
}

// Every file under 'dir' (sorted, relative), skipping build/ and deps/.
std::vector<fs::path> treeFiles(const fs::path& dir) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::string name = it->path().filename().string();
        if (it->is_directory() && (name == "build" || name == "deps")) { it.disable_recursion_pending(); continue; }
        if (it->is_regular_file()) {
            std::string ext = it->path().extension().string();
            if (name == "doc.wasm" || name == "safec_dom.js") continue;   // build outputs in theme/
            out.push_back(fs::relative(it->path(), dir));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string readFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

// ── locating / building safec-doc ────────────────────────────────────────────

bool Builder::findDocTool(std::string& binary, std::string& assets) {
    // 1. Explicit.
    if (const char* e = std::getenv("SAFEC_DOC"); e && *e) {
        binary = e;
        const char* a = std::getenv("SAFEC_DOC_ASSETS");
        assets = a && *a ? a : (fs::path(e).parent_path().parent_path() / "share" / "safec" / "doctool" / "theme").string();
        return true;
    }
    // 2. Prebuilt beside safeguard (a release install), or under SAFEC_HOME.
    std::vector<fs::path> bins = { selfExe().parent_path() };
    if (const char* h = std::getenv("SAFEC_HOME"); h && *h) bins.push_back(fs::path(h) / "bin");
    for (auto& b : bins) {
        fs::path bin = b / (std::string("safec-doc") + kExe);
        fs::path theme = b.parent_path() / "share" / "safec" / "doctool" / "theme";
        if (fs::exists(bin) && fs::exists(theme / "style.css")) {
            binary = bin.string();
            assets = theme.string();
            return true;
        }
    }
    // 3. Built from its sources, cached.
    std::vector<fs::path> srcs;
    if (const char* h = std::getenv("SAFEC_HOME"); h && *h) {
        srcs.push_back(fs::path(h) / "safeguard" / "doctool");
        srcs.push_back(fs::path(h) / "share" / "safec" / "doctool");
    }
    srcs.push_back(selfExe().parent_path().parent_path() / "doctool");                       // safeguard/build/..
    srcs.push_back(selfExe().parent_path().parent_path() / "share" / "safec" / "doctool");   // <prefix>/bin/..
    fs::path src;
    for (auto& s : srcs) if (fs::exists(s / "src" / "main.sc") && fs::exists(s / "Package.toml")) { src = s; break; }
    if (src.empty()) {
        std::cerr << "safeguard: no safec-doc found: neither a prebuilt one (bin/safec-doc beside safeguard) "
                     "nor its sources ($SAFEC_HOME/safeguard/doctool)\n";
        return false;
    }
    // The cache key: the sources, the compiler and this safeguard.
    std::string key;
    for (auto& f : treeFiles(src)) key += slash(f) + ":" + hashFile((src / f).string()) + ";";
    std::string safec = findSafec();
    key += "safec:" + hashFile(safec) + ";safeguard:" + hashFile(selfExe().string());
    std::string stdDir = findStdDir();
    if (!stdDir.empty()) key += ";dom:" + hashFile((fs::path(stdDir) / "wasm" / "safec_dom.js").string());
    std::string tmp = (cacheRoot() / "key.tmp").string();
    fs::create_directories(cacheRoot());
    { std::ofstream(tmp, std::ios::binary) << key; }
    std::string id = hashFile(tmp);
    fs::remove(tmp);
    fs::path dir = cacheRoot() / "safec-doc" / id;
    fs::path bin = dir / "build" / (std::string("safec-doc") + kExe);
    fs::path theme = dir / "assets";
    if (fs::exists(bin) && fs::exists(theme / "style.css")) {
        binary = bin.string();
        assets = theme.string();
        return true;
    }
    std::cout << "safeguard: building safec-doc from " << src.string() << " (once; cached in "
              << dir.string() << ")\n";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    for (auto& f : treeFiles(src)) {
        fs::create_directories((dir / f).parent_path());
        fs::copy_file(src / f, dir / f, fs::copy_options::overwrite_existing, ec);
    }
    std::string self = selfExe().string();
    // Runs 'argv' with 'cwd' as the working directory.
    auto runIn = [&](const fs::path& cwd, const std::vector<std::string>& argv, bool verbose, const std::string& logFile) {
        std::error_code e2;
        fs::path prev = fs::current_path(e2);
        fs::current_path(cwd, e2);
        if (e2) return -1;
        int r = runCmd(argv, verbose, /*suppressOutput=*/false, logFile);
        fs::current_path(prev, e2);
        return r;
    };
    fs::path log = dir / "build.log";
    if (runIn(dir, { self, "build", "--release" }, opts_.verbose, log.string()) != 0 || !fs::exists(bin)) {
        std::cerr << "safeguard: building safec-doc failed:\n" << readFile(log);
        return false;
    }
    fs::create_directories(theme);
    for (auto& e : fs::directory_iterator(dir / "theme")) fs::copy_file(e.path(), theme / e.path().filename(), fs::copy_options::overwrite_existing, ec);
    // The search client (wasm32): optional — without wasm-ld the pages
    // still work, only search is missing.
    fs::path clientLog = dir / "client.log";
    fs::path wasm = dir / "client" / "build" / "safec-doc-client";
    if (runIn(dir / "client", { self, "build", "--release", "--target", "wasm32" }, opts_.verbose, clientLog.string()) == 0 &&
        fs::exists(wasm)) {
        fs::copy_file(wasm, theme / "doc.wasm", fs::copy_options::overwrite_existing, ec);
    } else {
        std::cerr << "safeguard: warning: the search client (wasm32) did not build, so the reference has no "
                     "search (needs clang's wasm-ld; see " << clientLog.string() << ")\n";
    }
    if (!stdDir.empty())
        fs::copy_file(fs::path(stdDir) / "wasm" / "safec_dom.js", theme / "safec_dom.js", fs::copy_options::overwrite_existing, ec);
    binary = bin.string();
    assets = theme.string();
    return true;
}

// ── safeguard doc ────────────────────────────────────────────────────────────

bool Builder::doc(const DocOptions& o) {
    std::string safec = findSafec();
    std::string stdDir = findStdDir();
    if (stdDir.empty()) {
        std::cerr << "safeguard: cannot find SafeC std/ directory; set SAFEC_HOME\n";
        return false;
    }
    stdDir = fs::canonical(stdDir).string();
    fs::path root = o.std ? fs::path(stdDir) : fs::path(root_);
    fs::path out = o.out.empty() ? (o.std ? fs::current_path() / "build" / "doc" : fs::path(root_) / "build" / "doc")
                                 : fs::absolute(o.out);
    fs::path work = (o.std ? fs::current_path() / "build" : fs::path(root_) / "build") / "doc-json";
    fs::create_directories(work);
    std::string name = o.std ? "std" : manifest_.package.name;

    // 1. The inputs: headers first (they are the API), then sources.
    std::vector<std::string> files;
    auto add = [&](const fs::path& dir, bool sources) {
        if (!fs::exists(dir)) return;
        for (auto& rel : treeFiles(dir)) {
            std::string ext = rel.extension().string();
            if (ext == ".h" || (sources && ext == ".sc")) files.push_back((dir / rel).string());
        }
    };
    if (o.std) {
        add(root, true);
    } else {
        add(root / "include", false);
        add(root / "src", true);
    }
    if (files.empty()) {
        std::cerr << "safeguard: nothing to document (no .h or .sc files under "
                  << (o.std ? root.string() : (root / "include").string() + " or " + (root / "src").string()) << ")\n";
        return false;
    }

    std::vector<std::string> args = { "--emit-doc-json", "--doc-root", root.string() };
    if (!o.std) { args.push_back("--doc-extern"); args.push_back("std=" + stdDir); }
    if (o.includePrivate) args.push_back("--doc-private");
    if (o.std || manifest_.build.compatPreprocessor) args.push_back("--compat-preprocessor");
    std::vector<std::string> incs;
    if (o.std) {
        incs = { stdDir, fs::path(stdDir).parent_path().string() };
    } else {
        incs = safecIncludeDirs();
        incs.push_back((root / "src").string());
        for (auto& d : featureDefines()) args.push_back(d);
    }
    for (auto& i : incs) { args.push_back("-I"); args.push_back(i); }
    for (auto& f : files) args.push_back(f);
    fs::path json = work / (name + ".json");
    args.push_back("-o");
    args.push_back(json.string());
    fs::path rsp = work / "emit-doc-json.rsp";
    {
        std::ofstream r(rsp, std::ios::binary);
        for (auto& a : args) r << rspArg(a) << "\n";
    }
    std::cout << "safeguard: extracting declarations from " << files.size() << " file(s)\n";
    fs::path emitLog = work / "emit-doc-json.log";
    int rc = runCmd({ safec, "@" + slash(rsp) }, opts_.verbose, false, emitLog.string());
    std::string emitOut = readFile(emitLog);
    if (rc != 0 || !fs::exists(json)) {
        std::cerr << emitOut << "safeguard: safec --emit-doc-json failed\n";
        return false;
    }
    // Inputs that did not parse are reported, not fatal.
    if (emitOut.find("warning:") != std::string::npos) {
        std::istringstream in(emitOut);
        std::string line;
        while (std::getline(in, line))
            if (line.find("warning:") != std::string::npos) std::cerr << "safeguard: " << line << "\n";
    }

    // 2. The doc tool.
    std::string tool, assets;
    if (!findDocTool(tool, assets)) return false;

    // 3. Its configuration.
    std::string version, desc, repo, readme, sourceUrl, sourcePrefix, includePrefix;
    if (o.std) {
        std::string v;
        captureCmd({ safec, "--version" }, v, false);
        v.erase(0, v.find(' ') == std::string::npos ? 0 : v.find(' ') + 1);
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
        version = v;
        desc = "The SafeC standard library";
        repo = "https://github.com/safec-org/SafeC";
        std::string ref = o.sourceRef.empty() ? "main" : o.sourceRef;
        sourceUrl = repo + "/blob/" + ref + "/{file}";
        sourcePrefix = "std/";
        includePrefix = "std/";
    } else {
        version = manifest_.package.version;
        desc = manifest_.package.description;
        repo = manifest_.package.repository;
        fs::path rd = root / (manifest_.package.readme.empty() ? "README.md" : manifest_.package.readme);
        if (fs::exists(rd)) readme = rd.string();
        // GitHub / Gitea / Forgejo: <repo>/blob/<tag>/<file>; GitLab: /-/blob/.
        if (!repo.empty()) {
            std::string r = repo;
            if (r.size() > 4 && r.compare(r.size() - 4, 4, ".git") == 0) r.resize(r.size() - 4);
            while (!r.empty() && r.back() == '/') r.pop_back();
            std::string tag = manifest_.publish.tagPrefix + version;
            bool gitlab = r.find("gitlab") != std::string::npos || manifest_.publish.ci == "gitlab";
            if (r.rfind("http", 0) == 0)
                sourceUrl = r + (gitlab ? "/-/blob/" : "/blob/") + tag + "/{file}";
            repo = r;
        }
    }
    std::string stdUrl = o.stdUrl;
    if (stdUrl.empty()) if (const char* e = std::getenv("SAFEC_STD_DOCS_URL"); e && *e) stdUrl = e;
    if (stdUrl.empty()) stdUrl = kDefaultStdDocs;

    fs::path cfg = work / "config.json";
    {
        std::ofstream c(cfg, std::ios::binary);
        c << "{\n"
          << "  \"input\": " << jsonStr(slash(json)) << ",\n"
          << "  \"out\": " << jsonStr(slash(out)) << ",\n"
          << "  \"assets\": " << jsonStr(slash(assets)) << ",\n"
          << "  \"root\": " << jsonStr(slash(root)) << ",\n"
          << "  \"name\": " << jsonStr(name) << ",\n"
          << "  \"version\": " << jsonStr(version) << ",\n"
          << "  \"description\": " << jsonStr(desc) << ",\n"
          << "  \"repository\": " << jsonStr(repo) << ",\n"
          << "  \"readme\": " << jsonStr(slash(readme)) << ",\n"
          << "  \"source_url\": " << jsonStr(sourceUrl) << ",\n"
          << "  \"source_prefix\": " << jsonStr(sourcePrefix) << ",\n"
          << "  \"include_prefix\": " << jsonStr(includePrefix) << ",\n"
          << "  \"private\": " << (o.includePrivate ? "true" : "false") << ",\n"
          << "  \"externs\": {" << (o.std ? "" : "\"std\": " + jsonStr(stdUrl)) << "}\n"
          << "}\n";
    }

    // 4. Generate and check.
    if (runCmd({ tool, "build", cfg.string() }, opts_.verbose) != 0) {
        std::cerr << "safeguard: safec-doc failed\n";
        return false;
    }
    if (o.check && runCmd({ tool, "check", out.string() }, opts_.verbose) != 0) {
        std::cerr << "safeguard: the generated reference has broken links\n";
        return false;
    }
    std::cout << "safeguard: documentation in " << (out / "index.html").string() << "\n";
    if (o.serve) {
        std::cout << "safeguard: serving " << out.string() << " at http://localhost:" << o.port
                  << "/ (Ctrl-C to stop)\n";
        return runCmd({ tool, "serve", out.string(), "--port", std::to_string(o.port) }, opts_.verbose) == 0;
    }
    return true;
}

} // namespace safeguard
