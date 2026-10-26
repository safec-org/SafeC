# Changelog

All notable changes to SafeC — the compiler, the standard library,
safeguard and the install scripts. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are the
release tags.

## [Unreleased]

### Added — `safeguard doc` and `safeguard publish`

- `safeguard doc` generates a package's API reference into `build/doc/`, like
  rustdoc: per-item pages (structs, unions, enums, traits, functions,
  macros, constants, type aliases) with highlighted declarations, Markdown
  doc comments, intra-doc links (`[Vec::push]`), code examples and source
  links; a module tree (one module per header); highlighted source pages;
  client-side search (a SafeC program compiled to wasm32 on `std/wasm/dom`);
  light and dark themes; links into std's published reference; a link check
  of the result. `--serve` serves it over `std/http`, `--std` documents std
  itself.
- `safec --emit-doc-json`: the declarations of the inputs — kinds, names,
  namespaces, generics and bounds, parameters, fields, methods, enumerators,
  macros, signatures and locations — with their doc comments, as JSON, from
  the real front end (parse only). `--doc-root` / `--doc-extern` /
  `--doc-private` select what is listed.
- Doc comment conventions: `///`, `/** */`, trailing `///<`, `//!` module
  docs, and std's plain `//` comments above or beside a declaration;
  `@hidden`. See the SafeC docs' *Doc Comments* page.
- `safeguard/doctool`: **safec-doc**, the generator `safeguard doc` runs, a
  SafeC program on `std/serial/json`, `std/serial/markdown` (render hooks),
  scx templates, `std/dir`, `std/path`, `std/http` and `std/wasm/dom`.
  Releases ship it prebuilt (`bin/safec-doc`, `share/safec/doctool/`);
  from a source checkout it is built on first use and cached.
- `safeguard publish` prepares a library's git repository for publishing:
  `git init`, `.gitignore`, README/LICENSE/metadata checks, `--bump`
  (semver), and CI for GitHub Actions (default), GitLab CI, Gitea or Forgejo
  Actions (`--ci=`) that builds and tests on Linux/macOS/Windows and, on a
  tag, attaches prebuilt static and shared libraries per OS/arch to a
  release and deploys `safeguard doc` to the repository's Pages. A dry run
  unless `--write`; `--tag` commits and tags, `--push` pushes.
- `Package.toml`: `[package]` `repository`, `homepage`, `documentation`,
  `readme`, `keywords`; a `[publish]` table (`ci`, `branch`, `tag_prefix`,
  `pages`, `release`).
- `safeguard build --crate-type staticlib|cdylib|bin` overrides
  `[build] crate_type`.
- `examples/doc_demo`: a documented example library.

### Added — ownership checks (`Drop`) and thread safety (`Send`/`Sync`)

- **`Drop`**: a type with a `void drop()` method, or one holding such a
  value, must be dropped or moved exactly once on every path (`return`,
  `break`, `continue`, `goto`, `match`, `defer`, loops). Leaks, use after
  drop or move, double drops, overwriting a live value, and dropping or
  moving a value out of borrowed memory are compile errors, with a note at
  the declaration. Checked per generic instantiation; panicking paths are
  exempt. No run-time cost: nothing is inserted.
- `consume` methods (`int close() consume;`) destroy their receiver like
  `drop()`; destructors must take apart every owned field. `drop(x)` (drop
  glue for any type), `needs_drop(T)`, conditional `void drop() if (cond);`,
  and `forget(x)` (only in `unsafe`). `std/ownership.h`: `std::replace`,
  `std::swap`.
- A reference a method returns into its receiver borrows from it: using it
  after the receiver is dropped or moved is an error (`MutexGuard::get()`,
  `Vec::at()`).
- **`Send` / `Sync`**, derived from each type's structure, overridable with
  `unsafe impl Send/Sync;` and `impl !Send/!Sync;`, usable as generic
  bounds. `spawn` takes only static data that is Sync (or synchronizes its
  own writes); `spawn_scoped` lends a local until it is joined (no access
  meanwhile for a mutable lend; the local must outlive the join);
  `chan_send` needs a Send value and moves it.
- `std/thread.h`: `thread_spawn` / `SpawnedThread<T, R>` (moves its
  argument into the thread, `join()` returns the result), `Mutex<T>` /
  `MutexGuard<T>` (the value only through the guard, unlocked by its
  `drop()`), `RwLock<T>`. `std/atomic.h`: `AtomicInt`, `AtomicLong`.
- `Result::unwrap()` / `unwrap_err()` (consume).

### Changed — ownership

- The standard library's owning types implement `Drop`: collections,
  `File`, directories, sockets, TLS, HTTP, threads, sync primitives, GUI,
  audio/MIDI, async (futures, tasks, wakers, executors, channels), ML
  containers. `free()`/`close()`/`destroy()` stay as `consume` aliases.
  `Vec::clear()`/`drop()`/`set()` drop the elements they discard.
- `ThreadPool::spawn`/`block_on`, `Chan<T>`, `AsyncChannel<T>` and
  `Oneshot<T>` require Send values; GUI windows are `!Send`.
- Dropping a spawned, unfinished `AsyncTask` panics instead of leaving its
  executor to poll freed memory.

### Fixed — found by the ownership checks

- Leaks, double frees and use-after-frees across std, tests and examples,
  among them: `TlsStream::client()`/`TlsAcceptor::new()` failures kept
  their backend state; a failed client-certificate CertificateVerify, the
  GUI SVG renderer (a use after free of a by-value `String`), widget and
  PNG loaders, static-file and router helpers, URL/header/query removal,
  `Audio::open` (a value used after being moved), LLM and LoRA/JiT ML
  paths (double frees), and safec-doc's link checker.

### Changed

- `safeguard test` in a library package (`staticlib`/`cdylib`) builds the
  library and links it into each test, with `src/` on the include path.
