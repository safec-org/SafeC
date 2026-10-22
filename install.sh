#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
#  SafeC Install Script — macOS & Linux
#
#  Downloads a prebuilt safec/safeguard/sc-lsp release from GitHub Releases
#  and installs it — no CMake and no build needed. safec and safeguard do
#  call clang at run time (to compile the LLVM IR safec emits, link, and
#  import C headers), so this script also finds a clang >= 21 and points
#  SAFEC_CLANG at it, or says how to install one.
#
#  Usage:  bash install.sh [OPTIONS]
#
#  Options:
#    --prefix=<path>   Install directory (default: ~/safec)
#    --version=<tag>   Release tag to install (default: newest, including pre-releases)
#    --std=<sections>  Prebuilt std sections to install (default: all);
#                      --std=+<sections> adds sections to an existing install
#    --skip-env        Skip shell environment configuration
#    --help            Show this help message
# ──────────────────────────────────────────────────────────────────────────────

REPO="safec-org/SafeC"
# The release binaries are built against LLVM 21 (LLVM_VERSION in
# .github/workflows/release.yml); a clang older than that can't read the IR
# safec emits ("Unknown attribute kind").
MIN_CLANG=21

if [[ -t 1 ]]; then
    BOLD="\033[1m"; RED="\033[1;31m"; GREEN="\033[1;32m"
    YELLOW="\033[1;33m"; BLUE="\033[1;34m"; RESET="\033[0m"
else
    BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi
info()  { echo -e "${BLUE}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*"; }
die()   { error "$@"; exit 1; }

PREFIX=""
VERSION=""
SKIP_ENV=false
STD="all"

show_help() {
    cat <<'EOF'
SafeC Install Script — macOS & Linux

Usage:  bash install.sh [OPTIONS]

Options:
  --prefix=<path>   Install directory (default: ~/safec)
  --version=<tag>   Release tag to install, e.g. v1.0.0-beta-4
                    (default: the newest release, including pre-releases)
  --std=<sections>  The prebuilt std sections to install, comma-separated:
                    core (std/*.sc), collections, gui, ml, crypto, net, http,
                    dsp, fs, sync, alloc, simd, ... (std's top-level
                    directories), or 'all' (the default). std's sources are
                    always installed: a build that needs a section that
                    isn't installed compiles its units from source.
  --std=+<sections> Add prebuilt std sections to an existing install (from
                    the release it was installed from; nothing else changes)
  --skip-env        Skip shell environment configuration
  --help            Show this help message

Examples:
  curl -fsSL https://raw.githubusercontent.com/safec-org/SafeC/main/install.sh | bash
  bash install.sh --prefix=/opt/safec
  bash install.sh --version=v1.0.0-beta-4
  bash install.sh --std=core,collections,gui
  bash install.sh --std=+ml
EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --prefix=*)  PREFIX="${arg#*=}" ;;
        --version=*) VERSION="${arg#*=}" ;;
        --std=*)     STD="${arg#*=}" ;;
        --skip-env)  SKIP_ENV=true ;;
        --help|-h)   show_help ;;
        *) die "Unknown option: $arg (use --help for usage)" ;;
    esac
done
PREFIX="${PREFIX:-${HOME}/safec}"

# --std: 'all', a list of sections, or '+' and sections to add.
ADD_STD=false
if [[ "$STD" == +* ]]; then
    ADD_STD=true
    STD="${STD#+}"
    [[ -n "$STD" && "$STD" != all ]] || die "--std=+ takes the sections to add, e.g. --std=+ml,gui"
fi
[[ -n "$STD" ]] || die "--std needs 'all' or a list of sections, e.g. --std=core,collections"
[[ "$STD" =~ ^[a-z0-9_,]+$ ]] || die "--std: sections are comma-separated names, e.g. --std=core,collections"
if $ADD_STD; then
    # The release the install came from, so the sections match its safec.
    [[ -x "${PREFIX}/bin/safec" ]] || die "No SafeC install in ${PREFIX} to add std sections to (use --prefix)"
    if [[ -z "$VERSION" ]]; then
        [[ -f "${PREFIX}/.release" ]] || die "${PREFIX} doesn't record its release; give it with --version=<tag>"
        VERSION="$(head -1 "${PREFIX}/.release")"
    fi
fi
VERSION="${VERSION:-latest}"

# ── Platform detection ────────────────────────────────────────────────────────
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
    Darwin) plat_os="macos" ;;
    Linux)  plat_os="linux" ;;
    *)      die "Unsupported platform: $os (only macOS and Linux prebuilt releases are published — see https://github.com/${REPO})" ;;
esac
case "$arch" in
    arm64|aarch64) plat_arch="arm64" ;;
    x86_64|amd64)  plat_arch="x86_64" ;;
    *)             die "Unsupported architecture: $arch" ;;
esac

ASSET="safec-${plat_os}-${plat_arch}.tar.gz"
# Published assets today: safec-macos-arm64.tar.gz, safec-linux-x86_64.tar.gz.
# A combination outside that set (e.g. macos-x86_64, linux-arm64) has no
# published release binary yet — fail clearly instead of a confusing 404.
if [[ "$ASSET" != "safec-macos-arm64.tar.gz" && "$ASSET" != "safec-linux-x86_64.tar.gz" ]]; then
    die "No prebuilt release for ${plat_os}/${plat_arch} yet — see https://github.com/${REPO}/releases"
fi
info "Platform: ${BOLD}${plat_os}/${plat_arch}${RESET} → ${ASSET}"

# The release binaries run on macOS 11 or newer, and on Linux with glibc
# 2.25 or newer (release.yml builds and checks them so).
if [[ "$plat_os" == macos ]]; then
    macos_ver="$(sw_vers -productVersion 2>/dev/null || echo 0)"
    if [[ "${macos_ver%%.*}" -lt 11 ]]; then
        die "SafeC's release binaries need macOS 11 or newer (this is ${macos_ver})"
    fi
else
    glibc_ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
    if [[ -z "$glibc_ver" ]]; then
        warn "Couldn't find glibc's version (musl?); SafeC's Linux binaries need glibc 2.25 or newer"
    else
        IFS=. read -r glibc_major glibc_minor _ <<< "$glibc_ver"
        if (( glibc_major < 2 || (glibc_major == 2 && ${glibc_minor:-0} < 25) )); then
            die "SafeC's Linux binaries need glibc 2.25 or newer (this system has ${glibc_ver})"
        fi
    fi
fi

command -v curl &>/dev/null || die "curl not found — required to download the release"
command -v tar  &>/dev/null || die "tar not found — required to extract the release"

# ── Resolve download URL ──────────────────────────────────────────────────────
if [[ "$VERSION" == "latest" ]]; then
    # GitHub's /releases/latest endpoint explicitly excludes prereleases —
    # it 404s outright when every published release is a prerelease (e.g.
    # this project's "1.0.0-beta"). /releases (the list endpoint) includes
    # prereleases and returns them newest-first, so pull the newest match
    # from that instead; the DOWNLOAD_URL pipeline below already takes the
    # first match regardless of which endpoint this came from.
    API_URL="https://api.github.com/repos/${REPO}/releases"
else
    API_URL="https://api.github.com/repos/${REPO}/releases/tags/${VERSION}"
fi

info "Resolving release (${VERSION})..."
DOWNLOAD_URL="$(curl -fsSL "$API_URL" | grep -o "\"browser_download_url\": *\"[^\"]*${ASSET}\"" | sed -E 's/.*"(https:[^"]+)"/\1/' | head -1)"
[[ -n "$DOWNLOAD_URL" ]] || die "Could not find asset '${ASSET}' in release '${VERSION}' — check https://github.com/${REPO}/releases"
ok "Found: ${DOWNLOAD_URL}"

# ── Download + extract ────────────────────────────────────────────────────────
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

info "Downloading..."
curl -fsSL -o "${TMPDIR}/${ASSET}" "$DOWNLOAD_URL"
ok "Downloaded $(du -h "${TMPDIR}/${ASSET}" | cut -f1)"

tar xzf "${TMPDIR}/${ASSET}" -C "$TMPDIR"
EXTRACTED_DIR="$(find "$TMPDIR" -maxdepth 1 -type d -name 'safec-*')"
[[ -n "$EXTRACTED_DIR" ]] || die "Unexpected archive layout"
# The release's tag, from its download URL (.../releases/download/<tag>/...).
RELEASE_TAG="$(echo "$DOWNLOAD_URL" | sed -nE 's|.*/releases/download/([^/]+)/.*|\1|p')"

# The prebuilt std: lib/<triple>/std/<section>.a and .units, one pair per
# section. The ones --std names (all by default) are installed.
STD_SRC="$(find "${EXTRACTED_DIR}/lib" -mindepth 2 -maxdepth 2 -type d -name std 2>/dev/null | head -1)"
SECTIONS=()
if [[ -n "$STD_SRC" ]]; then
    AVAILABLE="$(cd "$STD_SRC" && ls -- *.units 2>/dev/null | sed 's/\.units$//' | tr '\n' ' ')"
    if [[ "$STD" == all ]]; then
        read -r -a SECTIONS <<< "$AVAILABLE"
    else
        IFS=',' read -r -a SECTIONS <<< "$STD"
        for s in ${SECTIONS[@]+"${SECTIONS[@]}"}; do
            [[ -f "${STD_SRC}/${s}.units" ]] || die "No std section '${s}' in this release (sections: ${AVAILABLE})"
        done
    fi
elif [[ "$STD" != all ]] || $ADD_STD; then
    die "This release has no sectioned prebuilt std (lib/<triple>/std/); install it without --std"
fi
install_sections() {   # <destination lib/<triple>/std>
    mkdir -p "$1"
    for s in ${SECTIONS[@]+"${SECTIONS[@]}"}; do
        cp "${STD_SRC}/${s}.units" "$1/"
        if [[ -f "${STD_SRC}/${s}.a" ]]; then cp "${STD_SRC}/${s}.a" "$1/"; fi
    done
}

if $ADD_STD; then
    STD_DST="${PREFIX}/${STD_SRC#"${EXTRACTED_DIR}/"}"
    install_sections "$STD_DST"
    ok "Added std sections to ${STD_DST}: ${SECTIONS[*]-}"
    INSTALLED="$(cd "$STD_DST" && ls -- *.units | sed 's/\.units$//' | tr '\n' ' ')"
    info "Installed std sections: ${INSTALLED}"
    exit 0
fi

info "Extracting to ${BOLD}${PREFIX}${RESET}..."
mkdir -p "$PREFIX"
# Everything but the prebuilt std, then the sections chosen; sections left
# from an earlier install (another release's) are removed first.
if [[ -n "$STD_SRC" ]]; then
    STD_DST="${PREFIX}/${STD_SRC#"${EXTRACTED_DIR}/"}"
    rm -rf "$STD_DST"
    mv "$STD_SRC" "${TMPDIR}/prebuilt-std"
    STD_SRC="${TMPDIR}/prebuilt-std"
fi
cp -R "${EXTRACTED_DIR}/." "$PREFIX/"
if [[ -n "$STD_SRC" ]]; then install_sections "$STD_DST"; fi
if [[ -n "$RELEASE_TAG" ]]; then echo "$RELEASE_TAG" > "${PREFIX}/.release"; fi
chmod +x "${PREFIX}/bin/"* 2>/dev/null || true
ok "Installed to ${PREFIX}"
if [[ -n "$STD_SRC" ]]; then
    if [[ "$STD" == all ]]; then info "Prebuilt std: all ${#SECTIONS[@]} sections"
    else info "Prebuilt std sections: ${SECTIONS[*]-} (add more with: bash install.sh --prefix=${PREFIX} --std=+<sections>)"; fi
fi

# ── Check the installed binaries start ────────────────────────────────────────
if ! version_out="$("${PREFIX}/bin/safec" --version 2>&1)"; then
    warn "Installed safec doesn't start:"
    echo "  ${version_out}" | head -3
    if [[ "$version_out" == *libLLVM* ]]; then
        warn "This release links LLVM dynamically; install LLVM ${MIN_CLANG} (see below) or a newer SafeC release."
    fi
else
    ok "${version_out}"
fi

# ── Find a clang >= MIN_CLANG ─────────────────────────────────────────────────
clang_major() { "$1" --version 2>/dev/null | sed -nE 's/.*clang version ([0-9]+).*/\1/p' | head -1; }
CLANG_PATH=""
for c in clang clang-23 clang-22 clang-21 \
         /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang \
         /usr/lib/llvm-23/bin/clang /usr/lib/llvm-22/bin/clang /usr/lib/llvm-21/bin/clang; do
    p="$(command -v "$c" 2>/dev/null || true)"
    [[ -n "$p" ]] || continue
    m="$(clang_major "$p")"
    if [[ -n "$m" && "$m" -ge "$MIN_CLANG" ]]; then CLANG_PATH="$p"; break; fi
done
SET_CLANG=false
if [[ -n "$CLANG_PATH" ]]; then
    ok "clang: ${CLANG_PATH} ($("$CLANG_PATH" --version | head -1))"
    # Plain 'clang' on PATH is what safec/safeguard try first; only set
    # SAFEC_CLANG when the right one is under another name or location.
    if [[ "$(command -v clang 2>/dev/null || true)" != "$CLANG_PATH" ]]; then SET_CLANG=true; fi
else
    found="$(command -v clang 2>/dev/null || true)"
    if [[ -n "$found" ]]; then
        warn "clang at ${found} is $("$found" --version | head -1 | sed -E 's/.*(clang version [0-9.]+).*/\1/'); SafeC needs clang ${MIN_CLANG} or newer."
    else
        warn "No clang found; SafeC needs clang ${MIN_CLANG} or newer to build programs."
    fi
    case "$plat_os" in
        macos) echo "  brew install llvm" ;;
        linux) echo "  Debian/Ubuntu:  wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh ${MIN_CLANG}"
               echo "  Other distros:  https://github.com/llvm/llvm-project/releases (clang+llvm-${MIN_CLANG}.x)" ;;
    esac
    echo "  Then re-run this script, or set SAFEC_CLANG to that clang's path."
fi

# ── Shell environment ─────────────────────────────────────────────────────────
if [[ "$SKIP_ENV" != true ]]; then
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    bin_dir="${PREFIX}/bin"

    clangxx="${CLANG_PATH%clang*}clang++${CLANG_PATH##*clang}"
    case "$shell_name" in
        fish)
            rc_file="${HOME}/.config/fish/config.fish"
            env_block=$'\n# SafeC\nset -gx SAFEC_HOME "'"${PREFIX}"$'"\nfish_add_path "'"${bin_dir}"$'"\n'
            $SET_CLANG && env_block+=$'set -gx SAFEC_CLANG "'"${CLANG_PATH}"$'"\nset -gx SAFEC_CLANGXX "'"${clangxx}"$'"\n'
            ;;
        zsh|*)
            if [[ "$shell_name" == zsh ]]; then rc_file="${HOME}/.zshrc"; else rc_file="${HOME}/.bashrc"; fi
            env_block=$'\n# SafeC\nexport SAFEC_HOME="'"${PREFIX}"$'"\nexport PATH="'"${bin_dir}"$':$PATH"\n'
            $SET_CLANG && env_block+=$'export SAFEC_CLANG="'"${CLANG_PATH}"$'"\nexport SAFEC_CLANGXX="'"${clangxx}"$'"\n'
            ;;
    esac

    if [[ -f "$rc_file" ]] && grep -q "SAFEC_HOME" "$rc_file" 2>/dev/null; then
        if $SET_CLANG && ! grep -q "SAFEC_CLANG" "$rc_file"; then
            # Earlier install, clang added since: append just those lines.
            if [[ "$shell_name" == fish ]]; then
                printf 'set -gx SAFEC_CLANG "%s"\nset -gx SAFEC_CLANGXX "%s"\n' "$CLANG_PATH" "$clangxx" >> "$rc_file"
            else
                printf 'export SAFEC_CLANG="%s"\nexport SAFEC_CLANGXX="%s"\n' "$CLANG_PATH" "$clangxx" >> "$rc_file"
            fi
            ok "Added SAFEC_CLANG to ${rc_file}"
        else
            info "Shell env already configured in ${rc_file} — leaving as-is"
        fi
    else
        echo "$env_block" >> "$rc_file"
        ok "Added SAFEC_HOME/PATH to ${rc_file} — restart your shell or 'source ${rc_file}'"
    fi
else
    info "Skipping shell setup (--skip-env). Add manually:"
    echo "  export SAFEC_HOME=\"${PREFIX}\""
    echo "  export PATH=\"${PREFIX}/bin:\$PATH\""
    if $SET_CLANG; then
        echo "  export SAFEC_CLANG=\"${CLANG_PATH}\""
        echo "  export SAFEC_CLANGXX=\"${CLANG_PATH%clang*}clang++${CLANG_PATH##*clang}\""
    fi
fi

echo ""
echo -e "${BOLD}SafeC installed.${RESET}"
echo -e "  ${BOLD}SAFEC_HOME${RESET}  ${PREFIX}"
echo -e "  ${BOLD}safec${RESET}       ${PREFIX}/bin/safec"
echo -e "  ${BOLD}safeguard${RESET}   ${PREFIX}/bin/safeguard"
echo -e "  ${BOLD}sc-lsp${RESET}      ${PREFIX}/bin/sc-lsp"
echo ""
echo "Try it:"
echo "  source ${rc_file:-your shell rc file} (or restart your shell)"
echo "  safeguard new hello && cd hello && safeguard run"
