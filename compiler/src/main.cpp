#include "safec/Preprocessor.h"
#include "safec/Diagnostic.h"
#include "safec/Lexer.h"
#include "safec/Parser.h"
#include "safec/Sema.h"
#include "safec/ConstEval.h"
#include "safec/CodeGen.h"
#include <cstdint>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Support/Program.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#ifndef SAFEC_VERSION
#define SAFEC_VERSION "unknown"
#endif

// ── CLI options ───────────────────────────────────────────────────────────────
static llvm::cl::opt<std::string>
    InputFile(llvm::cl::Positional, llvm::cl::desc("<input.sc>"),
              llvm::cl::Required);

static llvm::cl::opt<std::string>
    OutputFile("o", llvm::cl::desc("Output file"),
               llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    EmitLLVM("emit-llvm", llvm::cl::desc("Emit LLVM IR instead of bitcode"));

static llvm::cl::opt<bool>
    DumpAST("dump-ast", llvm::cl::desc("Dump AST only, skip codegen"));

static llvm::cl::opt<bool>
    DumpPP("dump-pp", llvm::cl::desc("Dump preprocessed source and exit"));

static llvm::cl::opt<bool>
    NoSema("no-sema", llvm::cl::desc("Skip semantic analysis (parse only)"));

static llvm::cl::opt<bool>
    NoConstEval("no-consteval", llvm::cl::desc("Skip const-eval pass"));

static llvm::cl::opt<bool>
    CompatPreprocessor("compat-preprocessor",
        llvm::cl::desc("Enable full C preprocessor (function-like macros, "
                       "token pasting, stringification)"));

static llvm::cl::opt<bool>
    NoImportCHeaders("no-import-c-headers",
        llvm::cl::desc("Disable automatic C header import via clang"));

static llvm::cl::opt<bool>
    Verbose("v", llvm::cl::desc("Verbose output"));

static llvm::cl::opt<bool>
    NoIncremental("no-incremental", llvm::cl::desc("Disable file-level bitcode cache"));

static llvm::cl::opt<std::string>
    CacheDir("cache-dir", llvm::cl::desc("Bitcode cache directory"),
             llvm::cl::init(".safec_cache"));

static llvm::cl::opt<bool>
    ClearCache("clear-cache", llvm::cl::desc("Delete all cached .bc files and exit"));

static llvm::cl::opt<std::string>
    DebugInfoLevel("g",
        llvm::cl::desc("Debug info level: none (default), lines, full"),
        llvm::cl::init("none"));

static llvm::cl::opt<bool>
    Freestanding("freestanding",
        llvm::cl::desc("Freestanding mode (no hosted runtime, no C header import)"));

// Attached-value form (-O2, -Os, -Oz, bare -O = -O2, default O0) rather than
// a separate '-opt-level <n>' flag, matching clang/rustc's '-O' convention
// that release-mode invocations already expect.
static llvm::cl::opt<std::string>
    OptLevel("O", llvm::cl::desc("Optimization level: 0 (default), 1, 2, 3, s, z"),
             llvm::cl::value_desc("level"), llvm::cl::Prefix, llvm::cl::init("0"),
             llvm::cl::ValueOptional);

static llvm::cl::opt<std::string>
    TargetTriple("target",
        llvm::cl::desc("Cross-target LLVM triple (e.g. x86_64-unknown-linux-gnu, "
                       "aarch64-unknown-linux-gnu, riscv64-unknown-elf, "
                       "wasm32-unknown-unknown, spirv64-unknown-unknown). "
                       "Empty (default) uses the host triple, same as before "
                       "this flag existed."),
        llvm::cl::init(""));

// -I <dir> include paths
static llvm::cl::list<std::string>
    IncludePaths("I", llvm::cl::desc("Add include search directory"),
                 llvm::cl::value_desc("dir"), llvm::cl::Prefix);

// -D NAME[=VALUE] command-line defines
static llvm::cl::list<std::string>
    CmdlineDefs("D", llvm::cl::desc("Define preprocessor macro"),
                llvm::cl::value_desc("NAME[=VALUE]"), llvm::cl::Prefix);

// Note: --version is handled by a manual argv pre-scan in main(), not a
// cl::opt — LLVM's CommandLine library already registers its own built-in
// '-version' option as part of ParseCommandLineOptions() (initCommonOptions()
// in LLVM's CommandLine.cpp), so declaring a second one here aborts with
// "Option 'version' registered more than once!".

// ── Link-driver flags (only take effect with --emit-bin) ──────────────────────
// safec itself only ever produced LLVM IR/bitcode; it never invoked a linker.
// --emit-bin adds a genuine "compile straight to a native binary/library"
// mode by shelling out to the system clang for assembly + linking, the same
// way safeguard's Builder already does per-file — these flags configure that
// step. Without --emit-bin they're accepted but have no effect (a warning is
// printed) since there's no link step for them to feed into.
static llvm::cl::opt<bool>
    EmitBin("emit-bin",
        llvm::cl::desc("Assemble and link a native binary/library directly to "
                       "the output path (invokes the system clang) instead of "
                       "emitting LLVM IR/bitcode"));

static llvm::cl::list<std::string>
    LinkLibs("l", llvm::cl::desc("Link against library <name> (-lname, like clang/gcc)"),
             llvm::cl::value_desc("name"), llvm::cl::Prefix);

static llvm::cl::list<std::string>
    LinkLibDirs("L", llvm::cl::desc("Add library search directory for -l"),
                llvm::cl::value_desc("dir"), llvm::cl::Prefix);

static llvm::cl::opt<bool>
    Shared("shared",
        llvm::cl::desc("With --emit-bin, produce a shared/dynamic library "
                       "instead of an executable"));

static llvm::cl::opt<bool>
    StaticLib("static-lib",
        llvm::cl::desc("With --emit-bin, produce a static library (.a, via "
                       "'ar') instead of an executable"));

static llvm::cl::opt<std::string>
    LtoMode("lto",
        llvm::cl::desc("With --emit-bin, enable LTO: 'thin' (default when "
                       "bare --lto is given) or 'full'"),
        llvm::cl::value_desc("thin|full"), llvm::cl::ValueOptional, llvm::cl::init(""));

static llvm::cl::opt<bool>
    ReleaseProfile("release",
        llvm::cl::desc("Release profile: optimize for speed (-O2) unless -O "
                       "is given explicitly"));

// ── FNV-1a 64-bit hash ────────────────────────────────────────────────────────
static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// A cache key of just the preprocessed source text is stale the moment the
// compiler itself is rebuilt: unchanged source silently gets a previous
// binary's cached bitcode forever, even across semantic/codegen fixes. Fold
// in an identifier for the running executable (its mtime + size) so a
// rebuilt compiler always misses old entries instead of silently reusing
// possibly-incorrect output.
static std::string compilerIdentity(const char *argv0) {
    std::string exe = llvm::sys::fs::getMainExecutable(argv0, (void *)&compilerIdentity);
    llvm::sys::fs::file_status st;
    if (exe.empty() || llvm::sys::fs::status(exe, st))
        return {}; // best-effort — an empty identity just means no extra protection
    auto mtime = st.getLastModificationTime().time_since_epoch().count();
    return exe + "|" + std::to_string(mtime) + "|" + std::to_string(st.getSize());
}

// CLI flags that change CodeGen output without changing it via ppSrc's text
// (TargetTriple/DebugInfoLevel are applied after preprocessing, at CodeGen
// construction) must be folded into the cache key too — otherwise compiling
// the same file for two different '--target' triples (exactly what
// multi-target support exists for) silently returns the first target's
// stale cached bitcode for the second, with no error at all. Freestanding
// is already reflected in ppSrc indirectly (it flips a preprocessor macro
// that #ifdef-branches code), but is included explicitly anyway rather than
// relying on that being true of every current and future freestanding-gated
// file.
static std::string cacheKeyFlags() {
    return "target=" + TargetTriple.getValue() +
           "|dbg=" + DebugInfoLevel.getValue() +
           "|freestanding=" + (Freestanding ? "1" : "0") +
           "|opt=" + OptLevel.getValue() +
           "|release=" + (ReleaseProfile ? "1" : "0");
}

// Bare '-O' (no attached digit) means "optimize, pick a sensible default" —
// matches gcc/clang treating a bare '-O' as '-O1'/'-O2'-ish rather than a
// parse error. Every other value maps 1:1 to LLVM's canned levels.
static llvm::OptimizationLevel resolveOptLevel() {
    std::string lvl = OptLevel.getValue();
    // --release means "optimize for speed" but must not override an -O the
    // user gave explicitly (getNumOccurrences() distinguishes "-O was never
    // passed" from "-O0 was passed on purpose").
    if (ReleaseProfile && OptLevel.getNumOccurrences() == 0) lvl = "2";
    if (lvl.empty() || lvl == "2") return llvm::OptimizationLevel::O2;
    if (lvl == "0") return llvm::OptimizationLevel::O0;
    if (lvl == "1") return llvm::OptimizationLevel::O1;
    if (lvl == "3") return llvm::OptimizationLevel::O3;
    if (lvl == "s") return llvm::OptimizationLevel::Os;
    if (lvl == "z") return llvm::OptimizationLevel::Oz;
    fprintf(stderr, "error: invalid -O level '%s' (expected 0,1,2,3,s,z)\n", lvl.c_str());
    exit(1);
}

static const char *optLevelName(llvm::OptimizationLevel level) {
    if (level == llvm::OptimizationLevel::O1) return "1";
    if (level == llvm::OptimizationLevel::O2) return "2";
    if (level == llvm::OptimizationLevel::O3) return "3";
    if (level == llvm::OptimizationLevel::Os) return "s";
    if (level == llvm::OptimizationLevel::Oz) return "z";
    return "0";
}

// Runs LLVM's standard per-module optimization pipeline in place. Reuses
// 'tm' (the CodeGen's TargetMachine, when cross-targeting was requested via
// --target) so target-aware analyses like TargetTransformInfo see the real
// target instead of defaulting to host-CPU cost assumptions.
static void optimizeModule(llvm::Module &mod, llvm::TargetMachine *tm,
                            llvm::OptimizationLevel level) {
    llvm::LoopAnalysisManager     lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager    cgam;
    llvm::ModuleAnalysisManager   mam;

    llvm::PassBuilder pb(tm);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm =
        (level == llvm::OptimizationLevel::O0)
            ? pb.buildO0DefaultPipeline(level)
            : pb.buildPerModuleDefaultPipeline(level);
    mpm.run(mod, mam);
}

// ── Link driver: --emit-bin ────────────────────────────────────────────────────
// safec's own pipeline stops at LLVM IR/bitcode; producing a real binary or
// library means handing that IR to the system clang (for assembly + linking)
// or 'ar' (for a static archive), the same tools safeguard's Builder already
// shells out to per-file. SAFEC_CLANG/SAFEC_AR override the discovered path,
// mirroring safeguard's SAFEC_CLANG/SAFEC_CLANGXX env vars.
static std::string findTool(const char *envVar, const char *name) {
    if (const char *e = std::getenv(envVar); e && *e) return e;
    static const char *kPrefixes[] = {
        "/opt/homebrew/opt/llvm/bin",  // Homebrew, Apple Silicon
        "/usr/local/opt/llvm/bin",     // Homebrew, Intel
    };
    for (auto *prefix : kPrefixes) {
        llvm::SmallString<256> p(prefix);
        llvm::sys::path::append(p, name);
        if (llvm::sys::fs::exists(p)) return std::string(p);
    }
    if (auto found = llvm::sys::findProgramByName(name)) return *found;
    return name; // let ExecuteAndWait report "not found" if this fails too
}

// Runs 'exe' with 'args' (args[0] is conventionally the program name, as
// argv[] expects), waiting for completion. Returns the child's exit code, or
// -1 if the process could not be started at all.
static int runTool(const std::string &exe, const std::vector<std::string> &args,
                    bool verbose) {
    std::vector<llvm::StringRef> refs;
    refs.reserve(args.size());
    for (auto &a : args) refs.push_back(a);
    if (verbose) {
        fprintf(stderr, "[safec]");
        for (auto &a : args) fprintf(stderr, " %s", a.c_str());
        fprintf(stderr, "\n");
    }
    std::string errMsg;
    int rc = llvm::sys::ExecuteAndWait(exe, refs, /*Env=*/std::nullopt, /*Redirects=*/{},
                                        /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
    if (!errMsg.empty())
        fprintf(stderr, "error: failed to run '%s': %s\n", exe.c_str(), errMsg.c_str());
    return rc;
}

// Assembles+links (or archives) 'llPath' into 'outputPath'. Returns true on
// success. Consults LinkLibs/LinkLibDirs/Shared/StaticLib/LtoMode/TargetTriple
// — see their cl::opt declarations above for what each means.
static bool linkBinary(const std::string &llPath, const std::string &outputPath,
                        llvm::OptimizationLevel optLevel, bool verbose) {
    std::string optFlag = std::string("-O") + optLevelName(optLevel);

    std::string ltoFlag;
    if (LtoMode.getNumOccurrences() > 0) {
        std::string mode = LtoMode.getValue();
        if (mode.empty() || mode == "thin") ltoFlag = "-flto=thin";
        else if (mode == "full") ltoFlag = "-flto";
        else {
            fprintf(stderr, "error: invalid --lto mode '%s' (expected 'thin' or 'full')\n",
                    mode.c_str());
            return false;
        }
    }

    if (StaticLib) {
        // A static library is just an archive — 'ar' does the packaging, no
        // linker/-l/-L/LTO involved (those are link-time concepts; a .a is
        // just object files bundled together for a *later* link to consume).
        std::string clang = findTool("SAFEC_CLANG", "clang");
        llvm::SmallString<256> objPath(llPath);
        llvm::sys::path::replace_extension(objPath, "o");
        std::vector<std::string> compileArgs = { clang, "-c", llPath, "-o", std::string(objPath) };
        if (!TargetTriple.getValue().empty())
            compileArgs.push_back("--target=" + TargetTriple.getValue());
        compileArgs.push_back(optFlag);
        if (runTool(clang, compileArgs, verbose) != 0) return false;

        std::string ar = findTool("SAFEC_AR", "ar");
        std::vector<std::string> arArgs = { ar, "rcs", outputPath, std::string(objPath) };
        return runTool(ar, arArgs, verbose) == 0;
    }

    std::string clang = findTool("SAFEC_CLANG", "clang");
    std::vector<std::string> args = { clang, llPath, "-o", outputPath, optFlag };
    if (!TargetTriple.getValue().empty())
        args.push_back("--target=" + TargetTriple.getValue());
    if (!ltoFlag.empty()) args.push_back(ltoFlag);
    if (Shared) args.push_back("-shared");
    for (auto &dir : LinkLibDirs) args.push_back("-L" + dir);
    for (auto &lib : LinkLibs)    args.push_back("-l" + lib);
    return runTool(clang, args, verbose) == 0;
}

// ── Read file ─────────────────────────────────────────────────────────────────
static std::string readFile(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "error: cannot open file '%s'\n", path.c_str());
        exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── AST dump (simple text representation) ────────────────────────────────────
static void dumpDecl(safec::Decl &d, int indent = 0) {
    std::string pad(indent * 2, ' ');
    switch (d.kind) {
    case safec::DeclKind::Function: {
        auto &fn = static_cast<safec::FunctionDecl &>(d);
        fprintf(stdout, "%s%s%sFunction '%s' -> %s%s\n",
                pad.c_str(),
                fn.isConsteval ? "consteval " : (fn.isConst ? "const " : ""),
                fn.isExtern    ? "extern " : "",
                fn.name.c_str(),
                fn.returnType ? fn.returnType->str().c_str() : "?",
                fn.body ? " {...}" : " (decl)");
        break;
    }
    case safec::DeclKind::Struct: {
        auto &sd = static_cast<safec::StructDecl &>(d);
        fprintf(stdout, "%s%s '%s' {\n",
                pad.c_str(), sd.isUnion ? "Union" : "Struct", sd.name.c_str());
        for (auto &f : sd.fields)
            fprintf(stdout, "%s  %s: %s\n",
                    pad.c_str(), f.name.c_str(),
                    f.type ? f.type->str().c_str() : "?");
        fprintf(stdout, "%s}\n", pad.c_str());
        break;
    }
    case safec::DeclKind::Enum: {
        auto &ed = static_cast<safec::EnumDecl &>(d);
        fprintf(stdout, "%sEnum '%s'\n", pad.c_str(), ed.name.c_str());
        break;
    }
    case safec::DeclKind::Region: {
        auto &rd = static_cast<safec::RegionDecl &>(d);
        fprintf(stdout, "%sRegion '%s' { capacity: %lld }\n",
                pad.c_str(), rd.name.c_str(), (long long)rd.capacity);
        break;
    }
    case safec::DeclKind::GlobalVar: {
        auto &gv = static_cast<safec::GlobalVarDecl &>(d);
        fprintf(stdout, "%s%sGlobal '%s': %s\n",
                pad.c_str(), gv.isConst ? "const " : "",
                gv.name.c_str(),
                gv.type ? gv.type->str().c_str() : "?");
        break;
    }
    case safec::DeclKind::StaticAssert: {
        fprintf(stdout, "%sstatic_assert(...)\n", pad.c_str());
        break;
    }
    default:
        fprintf(stdout, "%s<decl kind=%d>\n", pad.c_str(), (int)d.kind);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    llvm::InitLLVM X(argc, argv);

    // --version is handled by scanning argv directly, before
    // ParseCommandLineOptions: InputFile is a cl::Required positional, so
    // the normal parse path exits with a "missing input file" error before
    // ShowVersion could ever be checked if no .sc file was also given (the
    // common case: 'safec --version' alone).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            fprintf(stdout, "safec %s\n", SAFEC_VERSION);
            return 0;
        }
    }

    // All targets, not just native: --target lets a single safec binary
    // cross-generate for any ISA this LLVM was built with (this build has
    // X86, AArch64, RISCV, WebAssembly and SPIRV, among others — see
    // `llvm-config --targets-built`), which std::simd needs since its
    // whole point is genuinely different per-ISA vector codegen, not just
    // whatever the host happens to be.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm::cl::ParseCommandLineOptions(argc, argv, "SafeC compiler\n");

    if (!EmitBin) {
        if (!LinkLibs.empty() || !LinkLibDirs.empty() || Shared || StaticLib ||
            LtoMode.getNumOccurrences() > 0) {
            fprintf(stderr, "warning: -l/-L/--shared/--static-lib/--lto have no "
                            "effect without --emit-bin\n");
        }
    } else if (OutputFile == "-") {
        fprintf(stderr, "error: --emit-bin requires an explicit -o <path> "
                        "(stdout is not a valid binary output)\n");
        return 1;
    }

    // ── --clear-cache early exit ───────────────────────────────────────────────
    if (ClearCache) {
        std::error_code EC;
        llvm::sys::fs::directory_iterator it(CacheDir, EC), end;
        for (; !EC && it != end; it.increment(EC)) {
            llvm::StringRef p = it->path();
            if (p.ends_with(".bc"))
                llvm::sys::fs::remove(p);
        }
        if (Verbose) fprintf(stderr, "[safec] Cache cleared: %s\n", CacheDir.c_str());
        return 0;
    }

    // ── 1. Read source ─────────────────────────────────────────────────────────
    std::string src   = readFile(InputFile);
    const char *fname = InputFile.c_str();

    safec::DiagEngine diag(fname);

    // ── 2. Preprocess ─────────────────────────────────────────────────────────
    if (Verbose) fprintf(stderr, "[safec] Preprocessing %s ...\n", fname);

    safec::PreprocOptions ppOpts;
    ppOpts.compatMode     = CompatPreprocessor;
    ppOpts.importCHeaders = !NoImportCHeaders;
    ppOpts.freestanding   = Freestanding;
    if (Freestanding) {
        ppOpts.importCHeaders = false;
        ppOpts.cmdlineDefs["__SAFEC_FREESTANDING__"] = "1";
    }
    for (auto &d : IncludePaths) ppOpts.includePaths.push_back(d);
    for (auto &d : CmdlineDefs) {
        auto eq = d.find('=');
        if (eq == std::string::npos) {
            ppOpts.cmdlineDefs[d] = "1";
        } else {
            ppOpts.cmdlineDefs[d.substr(0, eq)] = d.substr(eq + 1);
        }
    }

    safec::Preprocessor preproc(src, fname, diag, ppOpts);
    std::string ppSrc = preproc.process();

    if (diag.hasErrors()) {
        fprintf(stderr, "Preprocessor failed with %d error(s)\n", diag.errorCount());
        return 1;
    }
    if (Verbose) fprintf(stderr, "[safec] Preprocessing done (%zu macros defined)\n",
                         preproc.macros().size());

    if (DumpPP) {
        fprintf(stdout, "%s", ppSrc.c_str());
        return 0;
    }

    // ── 2b. Incremental cache check ───────────────────────────────────────────
    // Skipped entirely for --dump-ast: that mode inspects the parse tree
    // itself, which the cache (populated from --emit-llvm/object-file runs)
    // has no representation of — honoring a cache hit here silently printed
    // disassembled bitcode from a previous invocation instead of an AST dump.
    // Also skipped for --emit-bin: the cache-hit path below only knows how
    // to replay a cached module as IR text or bitcode, not run it through
    // the link driver, so honoring a hit here would silently skip linking
    // and leave OutputFile as raw bitcode instead of a real binary/library.
    if (!NoIncremental && !DumpAST && !EmitBin) {
        uint64_t hash = fnv1a64(ppSrc + "|" + compilerIdentity(argv[0]) + "|" + cacheKeyFlags());
        char hexBuf[17];
        snprintf(hexBuf, sizeof(hexBuf), "%016llx", (unsigned long long)hash);
        llvm::SmallString<256> cachePath(CacheDir.getValue());
        llvm::sys::path::append(cachePath,
            std::string(hexBuf) + "_" +
            llvm::sys::path::filename(InputFile.getValue()).str() + ".bc");

        auto mbOrErr = llvm::MemoryBuffer::getFile(cachePath);
        if (mbOrErr) {
            if (Verbose)
                fprintf(stderr, "[safec] Cache hit: %s\n", cachePath.c_str());
            llvm::LLVMContext ctx2;
            auto modOrErr = llvm::parseBitcodeFile(**mbOrErr, ctx2);
            if (modOrErr) {
                auto &cachedMod = *modOrErr;
                if (OutputFile == "-") {
                    cachedMod->print(llvm::outs(), nullptr);
                } else if (EmitLLVM) {
                    std::error_code EC2;
                    llvm::raw_fd_ostream out(OutputFile.getValue(), EC2,
                                            llvm::sys::fs::OF_Text);
                    cachedMod->print(out, nullptr);
                } else {
                    std::error_code EC2;
                    std::string outPath = (OutputFile == "-")
                        ? "out.bc" : OutputFile.getValue();
                    llvm::raw_fd_ostream out(outPath, EC2,
                                            llvm::sys::fs::OF_None);
                    llvm::WriteBitcodeToFile(*cachedMod, out);
                }
                return 0;
            } else {
                llvm::consumeError(modOrErr.takeError());
                // Fall through to full compile
            }
        }
    }

    // ── 3. Lex ────────────────────────────────────────────────────────────────
    if (Verbose) fprintf(stderr, "[safec] Lexing ...\n");
    safec::Lexer lexer(ppSrc, fname, diag);
    auto tokens = lexer.lexAll();
    if (diag.hasErrors()) {
        fprintf(stderr, "Lexer failed with %d error(s)\n", diag.errorCount());
        return 1;
    }
    if (Verbose) fprintf(stderr, "[safec] Lexed %zu tokens\n", tokens.size());

    // ── 4. Parse ──────────────────────────────────────────────────────────────
    safec::Parser parser(std::move(tokens), diag);
    auto tu = parser.parseTranslationUnit();
    if (!tu) {
        fprintf(stderr, "Parse failed\n"); return 1;
    }
    tu->filename = fname;

    if (diag.hasErrors()) {
        fprintf(stderr, "Parser failed with %d error(s)\n", diag.errorCount());
        return 1;
    }
    if (Verbose) fprintf(stderr, "[safec] Parsed %zu top-level declarations\n",
                         tu->decls.size());

    // ── 5. AST dump ───────────────────────────────────────────────────────────
    if (DumpAST) {
        fprintf(stdout, "=== AST: %s ===\n", fname);
        for (auto &d : tu->decls) dumpDecl(*d);
        return 0;
    }

    // ── 5b. Resolve deferred array-size expressions ──────────────────────────
    // Array sizes that weren't plain literals (named constants, consteval
    // function calls like 'square(3)') were left unresolved by the parser —
    // fold them now, before Sema, since Sema/CodeGen assume ArrayType::size
    // is already known.
    {
        safec::ConstEvalEngine arraySizeCE(*tu, diag);
        bool ok = arraySizeCE.resolveArraySizes();
        if (!ok || diag.hasErrors()) {
            fprintf(stderr, "Array-size resolution failed with %d error(s)\n",
                    diag.errorCount());
            return 1;
        }
    }

    // ── 6. Semantic analysis ──────────────────────────────────────────────────
    if (!NoSema) {
        if (Verbose) fprintf(stderr, "[safec] Running semantic analysis ...\n");
        safec::Sema sema(*tu, diag);
        if (Freestanding) sema.setFreestanding(true);
        bool ok = sema.run();
        if (!ok || diag.hasErrors()) {
            fprintf(stderr, "Semantic analysis failed with %d error(s)\n",
                    diag.errorCount());
            return 1;
        }
        if (Verbose) fprintf(stderr, "[safec] Semantic analysis passed\n");
    }

    // ── 7. Const-eval pass ────────────────────────────────────────────────────
    if (!NoConstEval && !NoSema) {
        if (Verbose) fprintf(stderr, "[safec] Running const-eval pass ...\n");
        safec::ConstEvalEngine ce(*tu, diag);
        bool ok = ce.run();
        if (!ok || diag.hasErrors()) {
            fprintf(stderr, "Const-eval pass failed with %d error(s)\n",
                    diag.errorCount());
            return 1;
        }
        if (Verbose) fprintf(stderr, "[safec] Const-eval pass passed\n");
    }

    // ── 8. Code generation ────────────────────────────────────────────────────
    llvm::LLVMContext ctx;
    safec::DebugLevel dbgLevel = safec::DebugLevel::None;
    if (DebugInfoLevel == "lines") dbgLevel = safec::DebugLevel::Lines;
    else if (DebugInfoLevel == "full") dbgLevel = safec::DebugLevel::Full;

    safec::CodeGen cg(ctx, InputFile, diag, dbgLevel, TargetTriple);
    if (Freestanding) cg.setFreestanding(true);
    auto mod = cg.generate(*tu);

    if (diag.hasErrors()) {
        fprintf(stderr, "Code generation failed with %d error(s)\n",
                diag.errorCount());
        return 1;
    }

    // ── 8a. Optimize ──────────────────────────────────────────────────────────
    // Runs before the cache write below so cached bitcode already reflects
    // the requested -O level (cacheKeyFlags() folds 'opt=' into the cache
    // key, so a different -O next run is a cache miss rather than silently
    // reusing this level's output).
    llvm::OptimizationLevel optLevel = resolveOptLevel();
    if (optLevel != llvm::OptimizationLevel::O0) {
        if (Verbose) fprintf(stderr, "[safec] Running optimization pipeline (-O%s) ...\n",
                             optLevelName(optLevel));
        optimizeModule(*mod, cg.getTargetMachine(), optLevel);
    }

    // ── 8b. Write to incremental cache ───────────────────────────────────────
    if (!NoIncremental) {
        llvm::sys::fs::create_directories(CacheDir.getValue());
        uint64_t hash = fnv1a64(ppSrc + "|" + compilerIdentity(argv[0]) + "|" + cacheKeyFlags());
        char hexBuf[17];
        snprintf(hexBuf, sizeof(hexBuf), "%016llx", (unsigned long long)hash);
        llvm::SmallString<256> cachePath(CacheDir.getValue());
        llvm::sys::path::append(cachePath,
            std::string(hexBuf) + "_" +
            llvm::sys::path::filename(InputFile.getValue()).str() + ".bc");
        std::error_code EC;
        llvm::raw_fd_ostream cacheOut(cachePath, EC, llvm::sys::fs::OF_None);
        if (!EC) llvm::WriteBitcodeToFile(*mod, cacheOut);
        if (Verbose) fprintf(stderr, "[safec] Cache written: %s\n", cachePath.c_str());
    }

    // ── 9. Output ─────────────────────────────────────────────────────────────
    if (EmitBin) {
        llvm::SmallString<256> tmpLL;
        std::error_code tmpEC = llvm::sys::fs::createTemporaryFile("safec", "ll", tmpLL);
        if (tmpEC) {
            fprintf(stderr, "error: cannot create temporary file: %s\n",
                    tmpEC.message().c_str());
            return 1;
        }
        {
            std::error_code EC;
            llvm::raw_fd_ostream out(tmpLL, EC, llvm::sys::fs::OF_Text);
            if (EC) {
                fprintf(stderr, "error: cannot write temporary IR file '%s': %s\n",
                        tmpLL.c_str(), EC.message().c_str());
                return 1;
            }
            mod->print(out, nullptr);
        }
        bool ok = linkBinary(std::string(tmpLL), OutputFile.getValue(), optLevel, Verbose);
        llvm::sys::fs::remove(tmpLL);
        if (!ok) {
            fprintf(stderr, "error: linking failed for '%s'\n", OutputFile.c_str());
            return 1;
        }
        if (Verbose) fprintf(stderr, "[safec] Wrote %s: %s\n",
                             StaticLib ? "static library" : Shared ? "shared library" : "binary",
                             OutputFile.c_str());
        return 0;
    }

    if (OutputFile == "-") {
        mod->print(llvm::outs(), nullptr);
    } else if (EmitLLVM) {
        std::error_code EC;
        llvm::raw_fd_ostream out(OutputFile, EC, llvm::sys::fs::OF_Text);
        if (EC) {
            fprintf(stderr, "error: cannot open output file '%s': %s\n",
                    OutputFile.c_str(), EC.message().c_str());
            return 1;
        }
        mod->print(out, nullptr);
    } else {
        std::error_code EC;
        std::string outPath = (OutputFile == "-") ? "out.bc" : OutputFile.getValue();
        llvm::raw_fd_ostream out(outPath, EC, llvm::sys::fs::OF_None);
        if (EC) {
            fprintf(stderr, "error: cannot open output file '%s': %s\n",
                    outPath.c_str(), EC.message().c_str());
            return 1;
        }
        llvm::WriteBitcodeToFile(*mod, out);
        if (Verbose) fprintf(stderr, "[safec] Wrote bitcode to %s\n", outPath.c_str());
    }

    return 0;
}
