#include "safec/Preprocessor.h"
#include "safec/Diagnostic.h"
#include "safec/Lexer.h"
#include "safec/Parser.h"
#include "safec/Sema.h"
#include "safec/ConstEval.h"
#include "safec/CodeGen.h"

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

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

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

// -I <dir> include paths
static llvm::cl::list<std::string>
    IncludePaths("I", llvm::cl::desc("Add include search directory"),
                 llvm::cl::value_desc("dir"), llvm::cl::Prefix);

// -D NAME[=VALUE] command-line defines
static llvm::cl::list<std::string>
    CmdlineDefs("D", llvm::cl::desc("Define preprocessor macro"),
                llvm::cl::value_desc("NAME[=VALUE]"), llvm::cl::Prefix);

// ── FNV-1a 64-bit hash ────────────────────────────────────────────────────────
static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
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
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    llvm::cl::ParseCommandLineOptions(argc, argv, "SafeC compiler\n");

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
    if (!NoIncremental) {
        uint64_t hash = fnv1a64(ppSrc);
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

    safec::CodeGen cg(ctx, InputFile, diag, dbgLevel);
    if (Freestanding) cg.setFreestanding(true);
    auto mod = cg.generate(*tu);

    if (diag.hasErrors()) {
        fprintf(stderr, "Code generation failed with %d error(s)\n",
                diag.errorCount());
        return 1;
    }

    // ── 8b. Write to incremental cache ───────────────────────────────────────
    if (!NoIncremental) {
        llvm::sys::fs::create_directories(CacheDir.getValue());
        uint64_t hash = fnv1a64(ppSrc);
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
