// 'safeguard publish': prepare a package's git repository for publishing.
//
// Publishing a SafeC library means a git repository others depend on by
// URL ([[dependencies]] version = "<git URL>"), prebuilt libraries attached
// to its releases, and its API reference ('safeguard doc') on the
// repository's own Pages site. This command sets that up:
//
//   - a git repository (git init when there is none);
//   - .gitignore entries for build output, dependency caches, editor and OS
//     files;
//   - README / LICENSE / Package.toml metadata checks (a README and, for an
//     MIT license, a LICENSE are written when missing);
//   - an optional version bump (semver, in Package.toml);
//   - CI for the chosen provider: build + test on Linux, macOS and Windows;
//     on a tag, prebuilt libraries per OS/arch (static and shared for a
//     library) attached to a release, and the docs deployed to Pages;
//   - with --tag, a commit of those files and an annotated tag; with --push,
//     pushing the branch and the tag.
//
// Nothing is changed without --write (a dry run prints the plan); nothing
// is pushed without --push.
#include "Builder.h"
#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <filesystem>
#include <map>
#include <set>
#include <tuple>

namespace fs = std::filesystem;

namespace safeguard {

namespace {

std::string readText(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string trimmed(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) ++a;
    return s.substr(a);
}

std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// ── semver ───────────────────────────────────────────────────────────────────

struct SemVer {
    unsigned long long major = 0, minor = 0, patch = 0;
    std::string pre, build;
    std::string str() const {
        std::string s = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
        if (!pre.empty()) s += "-" + pre;
        if (!build.empty()) s += "+" + build;
        return s;
    }
};

bool parseSemVer(const std::string& v, SemVer& out) {
    static const std::regex re(R"(^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$)");
    std::smatch m;
    if (!std::regex_match(v, m, re)) return false;
    out.major = std::stoull(m[1]);
    out.minor = std::stoull(m[2]);
    out.patch = std::stoull(m[3]);
    out.pre = m[4];
    out.build = m[5];
    return true;
}

// <0, 0, >0 by semver precedence (build metadata ignored; a pre-release
// sorts before its release, pre-release identifiers compared as strings).
int semverCompare(const SemVer& a, const SemVer& b) {
    auto t = [](const SemVer& s) { return std::make_tuple(s.major, s.minor, s.patch); };
    if (t(a) != t(b)) return t(a) < t(b) ? -1 : 1;
    if (a.pre == b.pre) return 0;
    if (a.pre.empty()) return 1;
    if (b.pre.empty()) return -1;
    return a.pre < b.pre ? -1 : 1;
}

} // namespace

// The next version for --bump: "major", "minor", "patch" (a pre-release
// "1.2.0-rc.1" bumps to its release "1.2.0" on patch) or an explicit
// version. Empty on error (with 'err' set).
std::string publishBumpVersion(const std::string& current, const std::string& bump, std::string& err) {
    SemVer v;
    if (!parseSemVer(current, v)) { err = "version \"" + current + "\" in Package.toml is not semver (MAJOR.MINOR.PATCH)"; return ""; }
    SemVer n = v;
    n.build.clear();
    if (bump == "major") { n.major++; n.minor = 0; n.patch = 0; n.pre.clear(); }
    else if (bump == "minor") { n.minor++; n.patch = 0; n.pre.clear(); }
    else if (bump == "patch") { if (n.pre.empty()) n.patch++; n.pre.clear(); }
    else {
        SemVer e;
        if (!parseSemVer(bump, e)) { err = "--bump takes major, minor, patch or a semver version, got \"" + bump + "\""; return ""; }
        if (semverCompare(e, v) <= 0) { err = "--bump " + bump + " is not newer than " + current; return ""; }
        n = e;
    }
    return n.str();
}

// The .gitignore lines a SafeC package wants.
const std::vector<std::string>& publishGitignoreEntries() {
    static const std::vector<std::string> kEntries = {
        // build output and caches
        "build/", "deps/", ".safec_cache/", "dist/", "public/",
        // editors
        ".vscode/", ".idea/", "*.swp", "*.swo", "*~", ".#*",
        // OS junk
        ".DS_Store", "Thumbs.db", "desktop.ini",
    };
    return kEntries;
}

namespace {

// ── CI templates ─────────────────────────────────────────────────────────────
// @NAME@ @OUTPUT@ @KIND@ (lib|bin) @BRANCH@ @TAG_PREFIX@ @SAFEC_VERSION@

const char* kInstallScript = R"SH(#!/usr/bin/env bash
# Installs clang (LLVM @LLVM@) and a SafeC release on a CI runner: Linux
# (apt.llvm.org), macOS (Homebrew) or Windows (Git Bash; LLVM's official
# archive and install.ps1). Generated by 'safeguard publish'.
#
#   source .ci/install-safec.sh   # exports SAFEC_HOME, PATH, SAFEC_CLANG
#
# On GitHub / Gitea / Forgejo Actions the variables also go to
# $GITHUB_ENV / $GITHUB_PATH, so later steps see them.
# SAFEC_VERSION pins the SafeC release (a tag; empty = the newest).
_safec_ci_install() {
    set -e
    local llvm="${LLVM_VERSION:-@LLVM@}"
    local prefix="${SAFEC_PREFIX:-$HOME/safec}"
    local version="${SAFEC_VERSION:-@SAFEC_VERSION@}"
    local llvm_bin clang
    case "$(uname -s)" in
        Linux*)
            if ! command -v "clang-$llvm" >/dev/null 2>&1; then
                local sudo=""; [ "$(id -u)" != 0 ] && sudo="sudo"
                $sudo apt-get update -qq
                $sudo apt-get install -y -qq wget curl gnupg lsb-release software-properties-common xz-utils git
                wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
                $sudo bash /tmp/llvm.sh "$llvm"
                $sudo apt-get install -y -qq "clang-$llvm" "lld-$llvm"
            fi
            llvm_bin="/usr/lib/llvm-$llvm/bin"; clang="clang-$llvm" ;;
        Darwin*)
            brew list "llvm@$llvm" >/dev/null 2>&1 || brew install "llvm@$llvm" || brew install llvm
            llvm_bin="$(brew --prefix "llvm@$llvm" 2>/dev/null || brew --prefix llvm)/bin"; clang="$llvm_bin/clang" ;;
        MINGW*|MSYS*|CYGWIN*)
            local sdk="/c/llvm-$llvm"
            if [ ! -x "$sdk/bin/clang.exe" ]; then
                local pkg="@LLVM_PKG@"
                curl -fsSL -o /tmp/llvm.tar.xz "https://github.com/llvm/llvm-project/releases/download/llvmorg-$pkg/clang+llvm-$pkg-x86_64-pc-windows-msvc.tar.xz"
                mkdir -p "$sdk" && tar -xf /tmp/llvm.tar.xz -C "$sdk" --strip-components=1
            fi
            llvm_bin="$sdk/bin"; clang="$llvm_bin/clang.exe"
            curl -fsSL -o /tmp/install.ps1 https://raw.githubusercontent.com/safec-org/SafeC/main/install.ps1
            local winprefix; winprefix="$(cygpath -w "$prefix")"
            PATH="$llvm_bin:$PATH" powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w /tmp/install.ps1)" \
                -Prefix "$winprefix" -SkipEnv ${version:+-Version "$version"}
            ;;
    esac
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) ;;
        *) PATH="$llvm_bin:$PATH" bash -c "curl -fsSL https://raw.githubusercontent.com/safec-org/SafeC/main/install.sh | bash -s -- --prefix='$prefix' --skip-env ${version:+--version=$version}" ;;
    esac
    export SAFEC_HOME="$prefix"
    export PATH="$prefix/bin:$llvm_bin:$PATH"
    export SAFEC_CLANG="$clang"
    export SAFEC_CLANGXX="${clang%clang*}clang++${clang##*clang}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        { echo "SAFEC_HOME=$SAFEC_HOME"; echo "SAFEC_CLANG=$SAFEC_CLANG"; echo "SAFEC_CLANGXX=$SAFEC_CLANGXX"; } >> "$GITHUB_ENV"
        { echo "$prefix/bin"; echo "$llvm_bin"; } >> "$GITHUB_PATH"
    fi
    safec --version
    set +e
}
_safec_ci_install
)SH";

const char* kPackageScript = R"SH(#!/usr/bin/env bash
# Builds this package's release archive for one target into dist/:
#   @NAME@-<version>-<triple>/{lib,include}  (a library: static and shared)
#   @NAME@-<version>-<triple>/bin             (a program)
# plus Package.toml, README and LICENSE, as <dir>.tar.gz (.zip on Windows)
# and its .sha256. The lib/ + include/ layout is what safeguard takes as a
# prebuilt dependency directory ([[dependencies]] path = "<extracted dir>").
# Generated by 'safeguard publish'.
#
#   bash .ci/package.sh <target-triple>
set -euo pipefail
triple="${1:?usage: package.sh <target-triple>}"
version="$(sed -n 's/^version *= *"\(.*\)".*/\1/p' Package.toml | head -1)"
pkg="@NAME@-$version-$triple"
out="dist/$pkg"
rm -rf "$out" && mkdir -p "$out"
case "$triple" in *windows*) win=1 ;; *) win=0 ;; esac
if [ "@KIND@" = lib ]; then
    mkdir -p "$out/lib" "$out/include"
    safeguard build --release --crate-type=staticlib
    cp "build/lib@OUTPUT@.a" "$out/lib/"
    if [ -d build/include ]; then cp -R build/include/. "$out/include/"; fi
    safeguard build --release --crate-type=cdylib
    case "$triple" in
        *windows*) cp "build/@OUTPUT@.dll" "build/@OUTPUT@.lib" "$out/lib/" ;;
        *apple*)   cp "build/lib@OUTPUT@.dylib" "$out/lib/" ;;
        *)         cp "build/lib@OUTPUT@.so" "$out/lib/" ;;
    esac
else
    mkdir -p "$out/bin"
    safeguard build --release
    if [ "$win" = 1 ]; then cp "build/@OUTPUT@.exe" "$out/bin/"; else cp "build/@OUTPUT@" "$out/bin/"; fi
fi
cp Package.toml "$out/"
for f in README* LICENSE* COPYING* NOTICE*; do [ -f "$f" ] && cp "$f" "$out/"; done
cd dist
if [ "$win" = 1 ] && command -v 7z >/dev/null 2>&1; then
    7z a -bd "$pkg.zip" "$pkg" >/dev/null && archive="$pkg.zip"
else
    tar -czf "$pkg.tar.gz" "$pkg" && archive="$pkg.tar.gz"
fi
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$archive" > "$archive.sha256"
else shasum -a 256 "$archive" > "$archive.sha256"; fi
rm -rf "$pkg"
echo "dist/$archive"
)SH";

const char* kGiteaReleaseScript = R"SH(#!/usr/bin/env bash
# Creates (or reuses) the release for the current tag on a Gitea / Forgejo
# server and uploads every file in dist/ to it, over the server's API
# (GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_REF_NAME are set by its
# Actions runner; RELEASE_TOKEN needs write access to the repository).
# Generated by 'safeguard publish'.
set -euo pipefail
api="${GITHUB_SERVER_URL%/}/api/v1/repos/${GITHUB_REPOSITORY}"
tag="${GITHUB_REF_NAME:?not a tag build}"
auth="Authorization: token ${RELEASE_TOKEN:?set RELEASE_TOKEN (a secret with repository write access)}"
prerelease=false; case "$tag" in *-*) prerelease=true ;; esac
id="$(curl -fsS -H "$auth" "$api/releases/tags/$tag" 2>/dev/null | sed -n 's/^{"id":\([0-9]*\).*/\1/p')"
if [ -z "$id" ]; then
    id="$(curl -fsS -X POST -H "$auth" -H 'Content-Type: application/json' \
        -d "{\"tag_name\":\"$tag\",\"name\":\"$tag\",\"prerelease\":$prerelease}" \
        "$api/releases" | sed -n 's/^{"id":\([0-9]*\).*/\1/p')"
fi
[ -n "$id" ] || { echo "cannot create the release for $tag" >&2; exit 1; }
for f in dist/*; do
    [ -f "$f" ] || continue
    curl -fsS -X POST -H "$auth" -F "attachment=@$f" "$api/releases/$id/assets?name=$(basename "$f")" >/dev/null
    echo "uploaded $(basename "$f")"
done
)SH";

// ── GitHub Actions ───────────────────────────────────────────────────────────

const char* kGithubCi = R"YML(# Generated by 'safeguard publish': build and test on Linux, macOS and
# Windows on every push to @BRANCH@ and every pull request.
name: CI

on:
  push:
    branches: [@BRANCH@]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

env:
  LLVM_VERSION: "@LLVM@"
  # The SafeC release to build with (a tag; empty = the newest).
  SAFEC_VERSION: "@SAFEC_VERSION@"

jobs:
  test:
    name: test (${{ matrix.triple }})
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, triple: x86_64-unknown-linux-gnu }
          - { os: macos-14, triple: aarch64-apple-darwin }
          - { os: windows-latest, triple: x86_64-pc-windows-msvc }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM ${{ env.LLVM_VERSION }} and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: Build
        shell: bash
        run: safeguard build --release
      - name: Test
        shell: bash
        run: safeguard test
)YML";

const char* kGithubRelease = R"YML(# Generated by 'safeguard publish': on a version tag (@TAG_PREFIX@1.2.3), build
# prebuilt @KIND_TEXT@ for every OS/arch and attach them to a GitHub Release,
# and deploy the API reference ('safeguard doc') to this repository's GitHub
# Pages site (Settings -> Pages -> Source: "GitHub Actions").
name: Release

on:
  push:
    tags: ['@TAG_PREFIX@*']
  workflow_dispatch:

permissions:
  contents: read

env:
  LLVM_VERSION: "@LLVM@"
  SAFEC_VERSION: "@SAFEC_VERSION@"

jobs:
  build:
    name: build (${{ matrix.triple }})
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, triple: x86_64-unknown-linux-gnu }
          - { os: macos-14, triple: aarch64-apple-darwin }
          - { os: windows-latest, triple: x86_64-pc-windows-msvc }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM ${{ env.LLVM_VERSION }} and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: Test
        shell: bash
        run: safeguard test
      - name: Package
        shell: bash
        run: bash .ci/package.sh "${{ matrix.triple }}"
      - uses: actions/upload-artifact@v4
        with:
          name: dist-${{ matrix.triple }}
          path: dist/*
          if-no-files-found: error

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM ${{ env.LLVM_VERSION }} and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: API reference
        run: safeguard doc
      - uses: actions/upload-pages-artifact@v3
        with:
          path: build/doc

  release:
    if: startsWith(github.ref, 'refs/tags/')
    needs: build
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: dist-*
          merge-multiple: true
          path: dist
      - uses: softprops/action-gh-release@v2
        with:
          files: dist/*
          generate_release_notes: true
          prerelease: ${{ contains(github.ref_name, '-') }}

  pages:
    if: startsWith(github.ref, 'refs/tags/')
    needs: docs
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
)YML";

// ── GitLab CI ────────────────────────────────────────────────────────────────

const char* kGitlabCi = R"YML(# Generated by 'safeguard publish': build and test on Linux, macOS and
# Windows; on a version tag (@TAG_PREFIX@1.2.3), prebuilt @KIND_TEXT@ for every
# OS/arch in the project's generic package registry, linked from a GitLab
# Release, and the API reference ('safeguard doc') on GitLab Pages.
#
# The macOS and Windows jobs run on GitLab.com's hosted runners (the
# saas-macos-* and saas-windows-* tags; macOS runners need a Premium or
# Ultimate plan). On a self-managed instance, register runners with those
# tags or remove the jobs.

stages: [test, package, release, deploy]

variables:
  LLVM_VERSION: "@LLVM@"
  SAFEC_VERSION: "@SAFEC_VERSION@"

.linux:
  image: ubuntu:24.04
  tags: [saas-linux-small-amd64]
  before_script:
    - apt-get update -qq && apt-get install -y -qq curl ca-certificates git
    - source .ci/install-safec.sh

.macos:
  image: macos-14-xcode-15
  tags: [saas-macos-medium-m1]
  before_script:
    - source .ci/install-safec.sh

.windows:
  tags: [saas-windows-medium-amd64]

test:linux:
  extends: .linux
  stage: test
  script:
    - safeguard build --release
    - safeguard test

test:macos:
  extends: .macos
  stage: test
  script:
    - safeguard build --release
    - safeguard test

test:windows:
  extends: .windows
  stage: test
  script:
    - '& "C:\Program Files\Git\bin\bash.exe" -lc "source .ci/install-safec.sh && safeguard build --release && safeguard test"'

package:linux:
  extends: .linux
  stage: package
  rules:
    - if: $CI_COMMIT_TAG
  script:
    - bash .ci/package.sh x86_64-unknown-linux-gnu
  artifacts:
    paths: [dist/]

package:macos:
  extends: .macos
  stage: package
  rules:
    - if: $CI_COMMIT_TAG
  script:
    - bash .ci/package.sh aarch64-apple-darwin
  artifacts:
    paths: [dist/]

package:windows:
  extends: .windows
  stage: package
  rules:
    - if: $CI_COMMIT_TAG
  script:
    - '& "C:\Program Files\Git\bin\bash.exe" -lc "source .ci/install-safec.sh && bash .ci/package.sh x86_64-pc-windows-msvc"'
  artifacts:
    paths: [dist/]

release:
  stage: release
  image: registry.gitlab.com/gitlab-org/release-cli:latest
  rules:
    - if: $CI_COMMIT_TAG
  needs: [package:linux, package:macos, package:windows]
  script:
    - apk add --no-cache curl
    - |
      base="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/generic/@NAME@/${CI_COMMIT_TAG}"
      links=""
      for f in dist/*; do
        n="$(basename "$f")"
        curl -fsS --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file "$f" "$base/$n"
        links="$links --assets-link {\"name\":\"$n\",\"url\":\"$base/$n\",\"link_type\":\"package\"}"
      done
      eval release-cli create --name "$CI_COMMIT_TAG" --tag-name "$CI_COMMIT_TAG" $links

pages:
  extends: .linux
  stage: deploy
  rules:
    - if: $CI_COMMIT_TAG
  script:
    - safeguard doc --out public
  artifacts:
    paths: [public]
)YML";

// ── Gitea / Forgejo Actions ──────────────────────────────────────────────────
// Same workflow syntax as GitHub Actions; @RUNNER@ is the runner label.

const char* kGiteaCi = R"YML(# Generated by 'safeguard publish': build and test on every push to
# @BRANCH@ and every pull request (@PROVIDER@ Actions).
#
# @PROVIDER@ runners are self-hosted: the Linux job runs on any runner with
# the '@RUNNER@' label (act_runner's default). macOS and Windows need host
# runners of your own registered with the 'macos' / 'windows' labels; once
# you have them, uncomment their matrix entries (a job no runner serves
# waits forever).
name: CI

on:
  push:
    branches: [@BRANCH@]
  pull_request:
  workflow_dispatch:

env:
  LLVM_VERSION: "@LLVM@"
  SAFEC_VERSION: "@SAFEC_VERSION@"

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { runner: @RUNNER@, triple: x86_64-unknown-linux-gnu }
          # - { runner: macos, triple: aarch64-apple-darwin }
          # - { runner: windows, triple: x86_64-pc-windows-msvc }
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: Build
        shell: bash
        run: safeguard build --release
      - name: Test
        shell: bash
        run: safeguard test
)YML";

const char* kGiteaRelease = R"YML(# Generated by 'safeguard publish': on a version tag (@TAG_PREFIX@1.2.3), build
# prebuilt @KIND_TEXT@ and attach them to a @PROVIDER@ release, and publish the API
# reference ('safeguard doc') to the 'pages' branch.
#
# Needs a repository secret RELEASE_TOKEN: an access token with write
# access to this repository (releases and the 'pages' branch).
# @PROVIDER@ has no Pages service of its own@PAGES_NOTE@
# macOS / Windows builds need host runners labeled 'macos' / 'windows'
# (see ci.yml); uncomment their matrix entries when you have them.
name: Release

on:
  push:
    tags: ['@TAG_PREFIX@*']

env:
  LLVM_VERSION: "@LLVM@"
  SAFEC_VERSION: "@SAFEC_VERSION@"

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { runner: @RUNNER@, triple: x86_64-unknown-linux-gnu }
          # - { runner: macos, triple: aarch64-apple-darwin }
          # - { runner: windows, triple: x86_64-pc-windows-msvc }
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: Test
        shell: bash
        run: safeguard test
      - name: Package
        shell: bash
        run: bash .ci/package.sh "${{ matrix.triple }}"
      - name: Attach to the release
        shell: bash
        env:
          RELEASE_TOKEN: ${{ secrets.RELEASE_TOKEN }}
        run: bash .ci/upload-release.sh

  docs:
    runs-on: @RUNNER@
    steps:
      - uses: actions/checkout@v4
      - name: Install LLVM and SafeC
        shell: bash
        run: source .ci/install-safec.sh
      - name: API reference
        run: safeguard doc
      - name: Publish to the 'pages' branch
        env:
          RELEASE_TOKEN: ${{ secrets.RELEASE_TOKEN }}
        run: |
          cd build/doc
          git init -q -b pages
          git config user.name "safeguard publish"
          git config user.email "ci@localhost"
          git add -A
          git commit -qm "API reference for ${GITHUB_REF_NAME}"
          host="${GITHUB_SERVER_URL#*://}"
          git push -f "https://ci:${RELEASE_TOKEN}@${host}/${GITHUB_REPOSITORY}.git" pages
)YML";

std::string fill(std::string t, const std::map<std::string, std::string>& vars) {
    for (auto& [k, v] : vars) t = replaceAll(t, "@" + k + "@", v);
    return t;
}

// The SafeC release this machine runs (install.sh's .release), if known.
std::string installedSafecRelease() {
    if (const char* h = std::getenv("SAFEC_HOME"); h && *h) {
        fs::path f = fs::path(h) / ".release";
        if (fs::exists(f)) return trimmed(readText(f));
    }
    return "";
}

// https URL of a git remote ("git@github.com:u/r.git" -> "https://github.com/u/r").
std::string remoteToHttps(std::string url) {
    url = trimmed(url);
    if (url.rfind("git@", 0) == 0) {
        auto colon = url.find(':');
        if (colon != std::string::npos) url = "https://" + url.substr(4, colon - 4) + "/" + url.substr(colon + 1);
    } else if (url.rfind("ssh://git@", 0) == 0) {
        url = "https://" + url.substr(10);
    }
    if (url.size() > 4 && url.compare(url.size() - 4, 4, ".git") == 0) url.resize(url.size() - 4);
    return url;
}

// Replaces (or adds) 'key = "value"' in Package.toml's [package] table.
std::string setPackageKey(const std::string& toml, const std::string& key, const std::string& value) {
    std::istringstream in(toml);
    std::string line, out, section;
    bool done = false, inPackage = false;
    std::regex keyRe("^\\s*" + key + "\\s*=");
    while (std::getline(in, line)) {
        std::string t = trimmed(line);
        if (!t.empty() && t[0] == '[') {
            if (inPackage && !done) { out += key + " = \"" + value + "\"\n"; done = true; }
            inPackage = t == "[package]";
        } else if (inPackage && !done && std::regex_search(line, keyRe)) {
            auto eq = line.find('=');
            std::string comment;
            auto hash = line.find('#', line.find('"', eq) == std::string::npos ? eq : line.rfind('"') + 1);
            if (hash != std::string::npos) comment = "  " + line.substr(hash);
            out += line.substr(0, eq + 1) + " \"" + value + "\"" + comment + "\n";
            done = true;
            continue;
        }
        out += line + "\n";
    }
    if (!done) out += (inPackage ? "" : "\n[package]\n") + key + " = \"" + value + "\"\n";
    return out;
}

} // namespace

// ── safeguard publish ────────────────────────────────────────────────────────

bool Builder::publish(const PublishOptions& o) {
    fs::path root = root_;
    bool write = o.write || o.tag || o.push;
    std::string ci = !o.ci.empty() ? o.ci : !manifest_.publish.ci.empty() ? manifest_.publish.ci : "github";
    if (ci != "github" && ci != "gitlab" && ci != "gitea" && ci != "forgejo") {
        std::cerr << "safeguard: --ci must be github, gitlab, gitea or forgejo, got '" << ci << "'\n";
        return false;
    }
    std::string branch = manifest_.publish.branch.empty() ? "main" : manifest_.publish.branch;
    const std::string& tagPrefix = manifest_.publish.tagPrefix;
    bool isLib = manifest_.build.crateType != "bin";
    int problems = 0, warnings = 0;
    std::vector<std::string> changed;          // files written (relative), for the commit
    auto say = [&](const std::string& what) { std::cout << (write ? "  " : "  [dry run] ") << what << "\n"; };
    auto warn = [&](const std::string& what) { std::cout << "  warning: " << what << "\n"; ++warnings; };
    auto bad = [&](const std::string& what) { std::cout << "  error: " << what << "\n"; ++problems; };
    auto git = [&](std::vector<std::string> args, std::string* out = nullptr) {
        args.insert(args.begin(), { "git", "-C", root.string() });
        std::string tmp;
        bool ok = captureCmd(args, out ? *out : tmp, opts_.verbose);
        if (out) *out = trimmed(*out);
        return ok;
    };
    auto writeFile = [&](const std::string& rel, const std::string& content, bool exec = false) {
        fs::path p = root / rel;
        bool exists = fs::exists(p);
        if (exists && readText(p) == content) { say(rel + " (up to date)"); return; }
        say(std::string(exists ? "update " : "write  ") + rel);
        if (!write) return;
        fs::create_directories(p.parent_path());
        std::ofstream(p, std::ios::binary) << content;
        if (exec) {
            std::error_code ec;
            fs::permissions(p, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                            fs::perm_options::add, ec);
        }
        changed.push_back(rel);
    };

    std::cout << "safeguard publish: " << manifest_.package.name << " " << manifest_.package.version
              << " (" << (isLib ? "library" : "program") << "), CI: " << ci
              << (write ? "" : " — dry run, nothing is changed (--write applies it)") << "\n";

    // 1. git
    // Inside a work tree: a .git (directory, or file for a worktree) here
    // or above.
    bool isRepo = false;
    for (fs::path p = fs::absolute(root); !p.empty(); p = p.parent_path()) {
        if (fs::exists(p / ".git")) { isRepo = true; break; }
        if (p == p.parent_path()) break;
    }
    if (!isRepo) {
        say("git init -b " + branch);
        if (write && !git({ "init", "-q", "-b", branch })) { bad("git init failed"); return false; }
        isRepo = write;
    }

    // --tag: the tag must name the sources as committed, so the tree must be
    // clean before this writes anything (its own files are committed below).
    if (o.tag && isRepo) {
        std::string head, status;
        if (!git({ "rev-parse", "--verify", "-q", "HEAD" }, &head) || head.empty()) {
            std::cout << "  error: the repository has no commits yet; commit the sources first "
                         "(git add -A && git commit), then run --tag again\n";
            return false;
        }
        captureCmd({ "git", "-C", root.string(), "status", "--porcelain", "--untracked-files=no" }, status, opts_.verbose);
        if (!trimmed(status).empty()) {
            std::cout << "  error: uncommitted changes; commit or stash them first:\n" << status;
            return false;
        }
    } else if (o.tag) {
        std::cout << "  error: --tag needs a git repository with the sources committed\n";
        return false;
    }

    // 2. .gitignore
    {
        fs::path gi = root / ".gitignore";
        std::string text = fs::exists(gi) ? readText(gi) : "";
        std::set<std::string> have;
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) have.insert(trimmed(line));
        std::vector<std::string> missing;
        for (auto& e : publishGitignoreEntries())
            if (!have.count(e) && !have.count("/" + e)) missing.push_back(e);
        if (!missing.empty()) {
            std::string add = text.empty() ? "" : (text.back() == '\n' ? "\n" : "\n\n");
            add += "# SafeC build output and caches, editor and OS files (safeguard publish)\n";
            for (auto& m : missing) add += m + "\n";
            writeFile(".gitignore", text + add);
        } else {
            say(".gitignore (up to date)");
        }
    }

    // 3. Package.toml metadata, README, LICENSE
    std::string toml = readText(root / "Package.toml");
    std::string newToml = toml;
    SemVer sv;
    if (!parseSemVer(manifest_.package.version, sv))
        bad("version \"" + manifest_.package.version + "\" is not semver (MAJOR.MINOR.PATCH[-pre])");
    if (manifest_.package.description.empty()) warn("Package.toml has no description");
    if (manifest_.package.license.empty()) warn("Package.toml has no license (e.g. license = \"MIT\")");
    std::string repo = manifest_.package.repository;
    if (repo.empty()) {
        std::string remote;
        if (isRepo && git({ "remote", "get-url", o.remote }, &remote) && !remote.empty()) {
            repo = remoteToHttps(remote);
            say("set repository = \"" + repo + "\" in Package.toml (from remote " + o.remote + ")");
            newToml = setPackageKey(newToml, "repository", repo);
        } else {
            warn("Package.toml has no repository and there is no '" + o.remote +
                 "' remote: add one (git remote add " + o.remote + " <url>) so dependents can fetch it");
        }
    }
    std::string version = manifest_.package.version;
    if (!o.bump.empty()) {
        std::string err;
        std::string next = publishBumpVersion(version, o.bump, err);
        if (next.empty()) { bad(err); }
        else {
            say("version " + version + " -> " + next);
            newToml = setPackageKey(newToml, "version", next);
            version = next;
        }
    }
    if (newToml != toml) writeFile("Package.toml", newToml);

    bool hasReadme = false;
    for (auto& n : { "README.md", "README", "README.txt", "readme.md" }) if (fs::exists(root / n)) hasReadme = true;
    if (!hasReadme) {
        std::string docs = manifest_.package.documentation;
        std::string readme = "# " + manifest_.package.name + "\n\n" +
            (manifest_.package.description.empty() ? "A SafeC " + std::string(isLib ? "library" : "program") + "."
                                                   : manifest_.package.description) + "\n\n";
        if (isLib) {
            readme += "## Use it\n\nIn your `Package.toml`:\n\n```toml\n[[dependencies]]\nname    = \"" +
                      manifest_.package.name + "\"\nversion = \"" + (repo.empty() ? "<git URL>" : repo) +
                      "\"\n```\n\nPrebuilt static and shared libraries for Linux, macOS and Windows are "
                      "attached to each release.\n\n";
        }
        readme += "## Documentation\n\n";
        readme += docs.empty() ? "Run `safeguard doc --serve` for the API reference.\n"
                               : "The API reference is at " + docs + ".\n";
        readme += "\n## Build\n\n```sh\nsafeguard build\nsafeguard test\n```\n";
        if (!manifest_.package.license.empty()) readme += "\n## License\n\n" + manifest_.package.license + "\n";
        writeFile("README.md", readme);
    }
    bool hasLicense = false;
    for (auto& e : fs::directory_iterator(root)) {
        std::string n = e.path().filename().string();
        std::string u = n;
        std::transform(u.begin(), u.end(), u.begin(), ::toupper);
        if (u.rfind("LICENSE", 0) == 0 || u.rfind("LICENCE", 0) == 0 || u.rfind("COPYING", 0) == 0) hasLicense = true;
    }
    if (!hasLicense) {
        if (manifest_.package.license == "MIT") {
            std::time_t now = std::time(nullptr);
            int year = std::localtime(&now)->tm_year + 1900;
            std::string holder = manifest_.package.author.empty() ? "the " + manifest_.package.name + " authors"
                                                                  : manifest_.package.author;
            writeFile("LICENSE",
                "MIT License\n\nCopyright (c) " + std::to_string(year) + " " + holder + "\n\n"
                "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
                "of this software and associated documentation files (the \"Software\"), to deal\n"
                "in the Software without restriction, including without limitation the rights\n"
                "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
                "copies of the Software, and to permit persons to whom the Software is\n"
                "furnished to do so, subject to the following conditions:\n\n"
                "The above copyright notice and this permission notice shall be included in all\n"
                "copies or substantial portions of the Software.\n\n"
                "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
                "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
                "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
                "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
                "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
                "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
                "SOFTWARE.\n");
        } else {
            warn("no LICENSE file" + (manifest_.package.license.empty() ? std::string()
                 : " (Package.toml says " + manifest_.package.license + "; add its text as LICENSE)"));
        }
    }

    // 4. CI
    std::map<std::string, std::string> vars = {
        { "NAME", manifest_.package.name },
        { "OUTPUT", manifest_.build.output.empty() ? manifest_.package.name : manifest_.build.output },
        { "KIND", isLib ? "lib" : "bin" },
        { "KIND_TEXT", isLib ? "libraries (static and shared)" : "binaries" },
        { "BRANCH", branch },
        { "TAG_PREFIX", tagPrefix },
        { "SAFEC_VERSION", installedSafecRelease() },
        { "LLVM", "21" },
        { "LLVM_PKG", "21.1.8" },
    };
    writeFile(".ci/install-safec.sh", fill(kInstallScript, vars), true);
    writeFile(".ci/package.sh", fill(kPackageScript, vars), true);
    if (ci == "github") {
        writeFile(".github/workflows/ci.yml", fill(kGithubCi, vars));
        if (manifest_.publish.release || manifest_.publish.pages)
            writeFile(".github/workflows/release.yml", fill(kGithubRelease, vars));
    } else if (ci == "gitlab") {
        writeFile(".gitlab-ci.yml", fill(kGitlabCi, vars));
    } else {
        bool forgejo = ci == "forgejo";
        vars["PROVIDER"] = forgejo ? "Forgejo" : "Gitea";
        vars["RUNNER"] = forgejo ? "docker" : "ubuntu-latest";
        vars["PAGES_NOTE"] = forgejo
            ? ";\n# Codeberg (Forgejo) serves a repository's 'pages' branch at\n# https://<user>.codeberg.page/<repo>/ (Codeberg Pages); elsewhere, serve\n# the branch with any static host."
            : ";\n# serve the 'pages' branch with a static host (git-pages, a web server\n# pulling the branch, Codeberg Pages on a mirror).";
        std::string dir = forgejo ? ".forgejo/workflows/" : ".gitea/workflows/";
        writeFile(".ci/upload-release.sh", fill(kGiteaReleaseScript, vars), true);
        writeFile(dir + "ci.yml", fill(kGiteaCi, vars));
        writeFile(dir + "release.yml", fill(kGiteaRelease, vars));
    }

    if (problems) {
        std::cout << "safeguard publish: " << problems << " problem(s) to fix first\n";
        return false;
    }

    // 5. tag / push
    std::string tag = tagPrefix + version;
    if (o.tag) {
        std::string existing;
        git({ "tag", "--list", tag }, &existing);
        if (!existing.empty()) { std::cout << "  error: tag " << tag << " already exists\n"; return false; }
        if (o.verify) {
            std::cout << "  running the tests before tagging (--no-verify skips)\n";
            if (!test()) { std::cout << "  error: tests failed; not tagging\n"; return false; }
        }
        std::vector<std::string> add = { "add", "--" };
        for (auto& c : changed) add.push_back(c);
        if (!changed.empty() && !git(add)) { std::cout << "  error: git add failed\n"; return false; }
        std::string staged;
        git({ "diff", "--cached", "--name-only" }, &staged);
        if (!staged.empty()) {
            say("git commit -m \"Release " + manifest_.package.name + " " + version + "\"");
            if (!git({ "commit", "-q", "-m", "Release " + manifest_.package.name + " " + version })) {
                std::cout << "  error: git commit failed (is user.name/user.email set?)\n";
                return false;
            }
        }
        say("git tag -a " + tag);
        if (!git({ "tag", "-a", tag, "-m", manifest_.package.name + " " + version })) {
            std::cout << "  error: git tag failed (no commit yet?)\n";
            return false;
        }
    } else {
        std::cout << "  next: commit, then 'safeguard publish --tag' tags " << tag << "\n";
    }
    if (o.push) {
        std::string cur;
        git({ "rev-parse", "--abbrev-ref", "HEAD" }, &cur);
        say("git push " + o.remote + " " + cur);
        if (!git({ "push", o.remote, cur })) { std::cout << "  error: git push failed\n"; return false; }
        std::string tagged;
        git({ "tag", "--list", tag }, &tagged);
        if (!tagged.empty()) {
            say("git push " + o.remote + " " + tag);
            if (!git({ "push", o.remote, tag })) { std::cout << "  error: pushing the tag failed\n"; return false; }
        }
    } else {
        std::cout << "  nothing pushed (--push pushes the branch and the tag)\n";
    }
    if (ci == "github" && (manifest_.publish.pages))
        std::cout << "  note: enable Pages for the repository (Settings -> Pages -> Source: GitHub Actions)\n";
    std::cout << "safeguard publish: " << (write ? "done" : "dry run done") << ", " << warnings << " warning(s)\n";
    return true;
}

} // namespace safeguard
