#include "safec/Diagnostic.h"
#include "safec/Lexer.h"
#include "safec/Parser.h"
#include "safec/Sema.h"
#include "safec/CodeGen.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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
    NoSema("no-sema", llvm::cl::desc("Skip semantic analysis (parse only)"));

static llvm::cl::opt<bool>
    Verbose("v", llvm::cl::desc("Verbose output"));

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
        fprintf(stdout, "%sFunction '%s' -> %s%s\n",
                pad.c_str(), fn.name.c_str(),
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
        fprintf(stdout, "%sGlobal '%s': %s\n",
                pad.c_str(), gv.name.c_str(),
                gv.type ? gv.type->str().c_str() : "?");
        break;
    }
    default:
        fprintf(stdout, "%s<decl kind=%d>\n", pad.c_str(), (int)d.kind);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    llvm::InitLLVM X(argc, argv);
    // Only initialize the x86 native target (what this LLVM build ships with).
    // We only emit LLVM IR / bitcode, so we don't need all targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    llvm::cl::ParseCommandLineOptions(argc, argv, "SafeC compiler\n");

    // ── 1. Read source ─────────────────────────────────────────────────────────
    std::string src    = readFile(InputFile);
    const char *fname  = InputFile.c_str();

    safec::DiagEngine diag(fname);

    if (Verbose) fprintf(stderr, "[safec] Lexing %s ...\n", fname);

    // ── 2. Lex ────────────────────────────────────────────────────────────────
    safec::Lexer lexer(src, fname, diag);
    auto tokens = lexer.lexAll();
    if (diag.hasErrors()) {
        fprintf(stderr, "Lexer failed with %d error(s)\n", diag.errorCount());
        return 1;
    }
    if (Verbose) fprintf(stderr, "[safec] Parsed %zu tokens\n", tokens.size());

    // ── 3. Parse ──────────────────────────────────────────────────────────────
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

    // ── 4. AST dump ───────────────────────────────────────────────────────────
    if (DumpAST) {
        fprintf(stdout, "=== AST: %s ===\n", fname);
        for (auto &d : tu->decls) dumpDecl(*d);
        return 0;
    }

    // ── 5. Semantic analysis ──────────────────────────────────────────────────
    if (!NoSema) {
        if (Verbose) fprintf(stderr, "[safec] Running semantic analysis ...\n");
        safec::Sema sema(*tu, diag);
        bool ok = sema.run();
        if (!ok || diag.hasErrors()) {
            fprintf(stderr, "Semantic analysis failed with %d error(s)\n",
                    diag.errorCount());
            return 1;
        }
        if (Verbose) fprintf(stderr, "[safec] Semantic analysis passed\n");
    }

    // ── 6. Code generation ────────────────────────────────────────────────────
    llvm::LLVMContext ctx;
    safec::CodeGen cg(ctx, InputFile, diag);
    auto mod = cg.generate(*tu);

    if (diag.hasErrors()) {
        fprintf(stderr, "Code generation failed with %d error(s)\n",
                diag.errorCount());
        return 1;
    }

    // ── 7. Output ─────────────────────────────────────────────────────────────
    if (OutputFile == "-") {
        // Default: emit LLVM IR to stdout
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
        // Emit LLVM bitcode
        std::error_code EC;
        std::string outPath = OutputFile == "-" ? "out.bc" : OutputFile.getValue();
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
