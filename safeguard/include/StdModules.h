#pragma once
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace safeguard {

// ── std modules ──────────────────────────────────────────────────────────────
// A build compiles and links only the parts of SafeC's std it uses: the std
// headers its sources include, every std header those include, and the std
// sources ("units", std/**/*.sc) implementing them -- whose own includes
// count too, to a fixed point. Which units implement a header:
//
//   - by default, dir/foo.h is implemented by dir/foo.sc and by every
//     dir/foo_<suffix>.sc that has no header dir/foo_<suffix>.h of its own
//     (the longest existing header prefix wins: gui/gui_cocoa.sc implements
//     gui/gui.h, ml/tensor_blas.sc implements ml/tensor_blas.h);
//   - and explicitly, by directives in comments of the form
//     '// safeguard: <directive> <args...>' anywhere in a std file:
//
//       implements <std/x/y.h> ...   (a .sc) also implements these headers
//       sources <std/x/y.sc> ...     (a .h)  also implemented by these units
//       feature <name> ...           (a .sc) a unit only for builds with one
//                                    of these [features] enabled (a GPU
//                                    backend behind its feature)
//       runtime [<os>...]            (a .sc) always in the build (for these
//                                    OSes): it defines functions safec emits
//                                    calls to without any #include
//       link <os> <flag>...          (a .sc) system libraries the unit calls
//                                    into, for builds for <os>
//       deb <package>...             (a .sc) the Debian packages providing
//                                    them, added to a .deb's Depends
//
// Text from a '(' on is a remark. <os> is macos, linux, windows, bsd, unix (macos, linux and bsd), hosted
// (any OS), baremetal, wasm or any. A link flag is -l<name>, -framework
// <Name>, -L<dir> (${VAR} and ${VAR:-default} expanded; kept, with an rpath,
// only if the directory exists) or 'blas', the CPU BLAS [build] blas picks.
//
// Includes are followed whatever #if they sit under, except where the
// condition is one this can decide: a single (possibly negated)
// defined(X) / X / #ifdef X / #ifndef X of an OS macro (__APPLE__, _WIN32,
// __linux__, set from the target triple as safec does) or, in a project's
// own files, of a SAFEC_FEATURE_<NAME> macro. std's own units are compiled
// without feature macros; in std headers they're left undecided.

// What a std module set is computed for.
struct StdPlatform {
    std::string triple;     // the target triple ("" for the host)
    std::string os;         // macos, linux, windows, bsd, baremetal or wasm
    bool bareMetal = false;
    bool wasm = false;
    std::set<std::string> features;   // enabled [features], as named in the manifest
};

// Whether a platform-variant std unit (foo_linux.sc, gui_cocoa.sc, ...) is
// the one for this target (see its definition).
bool stdVariantMatches(const std::filesystem::path& src, const std::string& triple, bool bareMetal);

class StdIndex {
public:
    // The index of the std/ tree at stdDir (read once per process).
    static const StdIndex& load(const std::string& stdDir);

    struct Guard {
        std::string macro;   // the macro the condition tests
        bool defined;        // true: the include needs it defined
    };
    struct Include {
        std::string name;    // as written between <> or ""
        bool angled = false;
        std::vector<Guard> guards;   // decidable enclosing conditions
    };
    struct LinkRule {
        std::string os;
        std::vector<std::string> flags;
    };
    struct File {
        std::string rel;                     // std-relative path, '/'-separated
        bool header = false;
        std::vector<Include> includes;
        std::vector<std::string> implements; // headers ('implements')
        std::vector<std::string> sources;    // units ('sources')
        std::vector<std::string> features;
        bool runtime = false;
        std::vector<std::string> runtimeOs;  // empty: any
        std::vector<LinkRule> links;
        std::vector<LinkRule> debs;
    };

    std::string dir;                         // the std/ directory
    std::map<std::string, File> files;       // by rel
    std::map<std::string, std::vector<std::string>> units;  // header rel -> its units
    std::vector<std::string> problems;       // malformed directives, orphan units

    // The std file an include names ("std/io.h" for <std/io.h>), or "".
    std::string resolve(const std::string& includeName) const;

    // The units a build with these roots (std headers or units, rel) needs.
    std::set<std::string> closure(const std::set<std::string>& roots, const StdPlatform& p) const;
    // Every unit for this platform (variants and the wasm runtime
    // permitting): [build] std = "all".
    std::set<std::string> allUnits(const StdPlatform& p) const;
    // Whether a unit joins builds for p at all (variant, feature, wasm/).
    bool unitApplies(const File& f, const StdPlatform& p) const;
    // The system link flags and .deb packages of these units, in order,
    // deduplicated ('blas' is left as is for the caller to expand).
    std::vector<std::string> linkFlags(const std::set<std::string>& units, const StdPlatform& p) const;
    std::vector<std::string> debDepends(const std::set<std::string>& units, const StdPlatform& p) const;
    // Every std file a unit's compile reads (itself and the headers it
    // includes, transitively, under any condition): its rebuild inputs.
    std::set<std::string> compileInputs(const std::string& unit) const;

    // Whether an OS selector covers p.
    static bool osMatches(const std::string& sel, const StdPlatform& p);
    // Whether an include's guards can all hold ('features': whether the
    // file is a project's own, whose SAFEC_FEATURE_* macros are known).
    static bool guardsHold(const Include& inc, const StdPlatform& p, bool userFile, bool stdUnit);
    // The #includes of a source or header file.
    static std::vector<Include> scanIncludes(const std::filesystem::path& file);
};

// The std files (rel) a project's own files include: 'files' are scanned,
// and so is every non-std file they include, found beside the including
// file or in 'includeDirs' (the project's and its dependencies' include
// directories). An include that resolves to nothing is ignored.
std::set<std::string> stdRootsOf(const StdIndex& idx, const std::vector<std::string>& files,
                                 const std::vector<std::string>& includeDirs, const StdPlatform& p);

} // namespace safeguard
