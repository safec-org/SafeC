#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
#  SafeC Install Script — macOS & Linux
#
#  Downloads a prebuilt safec/safeguard/sc-lsp release from GitHub Releases
#  and installs it — no LLVM, no CMake, no compiler needed on the machine
#  you're installing to.
#
#  Usage:  bash install.sh [OPTIONS]
#
#  Options:
#    --prefix=<path>   Install directory (default: ~/safec)
#    --version=<tag>   Release tag to install (default: latest)
#    --skip-env        Skip shell environment configuration
#    --help            Show this help message
# ──────────────────────────────────────────────────────────────────────────────

REPO="safec-org/SafeC"

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
  --version=<tag>   Release tag to install, e.g. v0.2.0 (default: latest)
  --skip-env        Skip shell environment configuration
  --help            Show this help message

Examples:
  curl -fsSL https://raw.githubusercontent.com/safec-org/SafeC/main/install.sh | bash
  bash install.sh --prefix=/opt/safec
  bash install.sh --version=v0.2.0
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

# ── Shell environment ─────────────────────────────────────────────────────────
if [[ "$SKIP_ENV" != true ]]; then
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    bin_dir="${PREFIX}/bin"

    case "$shell_name" in
        fish)
            rc_file="${HOME}/.config/fish/config.fish"
            env_block=$'\n# SafeC\nset -gx SAFEC_HOME "'"${PREFIX}"$'"\nfish_add_path "'"${bin_dir}"$'"\n'
            ;;
        zsh)
            rc_file="${HOME}/.zshrc"
            env_block=$'\n# SafeC\nexport SAFEC_HOME="'"${PREFIX}"$'"\nexport PATH="'"${bin_dir}"$':$PATH"\n'
            ;;
        *)
            rc_file="${HOME}/.bashrc"
            env_block=$'\n# SafeC\nexport SAFEC_HOME="'"${PREFIX}"$'"\nexport PATH="'"${bin_dir}"$':$PATH"\n'
            ;;
    esac

    if [[ -f "$rc_file" ]] && grep -q "SAFEC_HOME" "$rc_file" 2>/dev/null; then
        info "Shell env already configured in ${rc_file} — leaving as-is"
    else
        echo "$env_block" >> "$rc_file"
        ok "Added SAFEC_HOME/PATH to ${rc_file} — restart your shell or 'source ${rc_file}'"
    fi
else
    info "Skipping shell setup (--skip-env). Add manually:"
    echo "  export SAFEC_HOME=\"${PREFIX}\""
    echo "  export PATH=\"${PREFIX}/bin:\$PATH\""
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
