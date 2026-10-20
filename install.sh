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
VERSION="latest"
SKIP_ENV=false

show_help() {
    cat <<'EOF'
SafeC Install Script — macOS & Linux

Usage:  bash install.sh [OPTIONS]

Options:
  --prefix=<path>   Install directory (default: ~/safec)
  --version=<tag>   Release tag to install, e.g. v1.0.0-beta-4
                    (default: the newest release, including pre-releases)
  --skip-env        Skip shell environment configuration
  --help            Show this help message

Examples:
  curl -fsSL https://raw.githubusercontent.com/safec-org/SafeC/main/install.sh | bash
  bash install.sh --prefix=/opt/safec
  bash install.sh --version=v1.0.0-beta-4
EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --prefix=*)  PREFIX="${arg#*=}" ;;
        --version=*) VERSION="${arg#*=}" ;;
        --skip-env)  SKIP_ENV=true ;;
        --help|-h)   show_help ;;
        *) die "Unknown option: $arg (use --help for usage)" ;;
    esac
done
PREFIX="${PREFIX:-${HOME}/safec}"

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

info "Extracting to ${BOLD}${PREFIX}${RESET}..."
mkdir -p "$PREFIX"
tar xzf "${TMPDIR}/${ASSET}" -C "$TMPDIR"
EXTRACTED_DIR="$(find "$TMPDIR" -maxdepth 1 -type d -name 'safec-*')"
[[ -n "$EXTRACTED_DIR" ]] || die "Unexpected archive layout"
cp -R "${EXTRACTED_DIR}/." "$PREFIX/"
chmod +x "${PREFIX}/bin/"* 2>/dev/null || true
ok "Installed to ${PREFIX}"

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
