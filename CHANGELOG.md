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

### Changed

- `safeguard test` in a library package (`staticlib`/`cdylib`) builds the
  library and links it into each test, with `src/` on the include path.
