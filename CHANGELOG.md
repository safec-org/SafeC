# Changelog

## Unreleased

### Self-hosted compiler (selfhost/)

- **scc**, a SafeC compiler written in SafeC, is in the new top-level
  `selfhost/` directory. It compiles SafeC straight to a WebAssembly binary,
  with no LLVM and no linker. It has its own lexer, preprocessor, parser,
  semantic analysis with safec's safety checks and diagnostics, and wasm
  backend, plus a small runtime written in SafeC. `selfhost/SUBSET.md`
  lists the supported language subset.
- scc builds with the C++ `safec`, both natively (`selfhost/build.sh`) and
  as WebAssembly (`selfhost/build-web.sh`). It also compiles itself:
  `selfhost/bootstrap.sh` checks that the self-compiled compiler reproduces
  itself byte for byte.
- There is a playground API for running SafeC in the browser:
  `selfhost/web/scc.js` provides `compile(source)` and
  `run(wasm, { stdout, stdin })`, with time and memory limits in a Web
  Worker. The contract is in `selfhost/web/README.md`.
- A differential test harness, `selfhost/tests/difftest.py`, compiles the
  book and reference programs, a corpus and negative tests with both
  compilers and compares the results. It also runs `compiler/tests`
  compile-only. CI runs all of it on Linux, in the new `selfhost` job.
