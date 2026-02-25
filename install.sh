#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
#  SafeC Install Script — macOS & Linux
#  Usage:  bash install.sh [OPTIONS]
#
#  Options:
#    --llvm-dir=<path>   Path to LLVM cmake directory (auto-detected if omitted)
#    --jobs=N            Parallel build jobs (default: nproc)
#    --skip-safeguard    Skip building the safeguard package manager
#    --skip-env          Skip shell environment configuration
#    --help              Show this help message
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Color helpers ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD="\033[1m"
    RED="\033[1;31m"
    GREEN="\033[1;32m"
    YELLOW="\033[1;33m"
    BLUE="\033[1;34m"
    RESET="\033[0m"
else
    BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

info()  { echo -e "${BLUE}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*"; }
die()   { error "$@"; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
LLVM_DIR=""
JOBS=""
SKIP_SAFEGUARD=false
SKIP_ENV=false

show_help() {
    cat <<'EOF'
SafeC Install Script — macOS & Linux

Usage:  bash install.sh [OPTIONS]

Options:
  --llvm-dir=<path>   Path to LLVM cmake directory (auto-detected if omitted)
  --jobs=N            Parallel build jobs (default: nproc / sysctl hw.ncpu)
  --skip-safeguard    Skip building the safeguard package manager
  --skip-env          Skip shell environment configuration
  --help              Show this help message

Examples:
  bash install.sh
  bash install.sh --llvm-dir=/usr/lib/llvm-17/lib/cmake/llvm --jobs=8
  bash install.sh --skip-safeguard --skip-env
EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --llvm-dir=*) LLVM_DIR="${arg#*=}" ;;
        --jobs=*)     JOBS="${arg#*=}" ;;
        --skip-safeguard) SKIP_SAFEGUARD=true ;;
        --skip-env)       SKIP_ENV=true ;;
        --help|-h)        show_help ;;
        *) die "Unknown option: $arg (use --help for usage)" ;;
    esac
done

# ── Platform detection ────────────────────────────────────────────────────────
PLATFORM=""
DISTRO=""

detect_platform() {
    local os
    os="$(uname -s)"
    case "$os" in
        Darwin) PLATFORM="macos" ;;
        Linux)  PLATFORM="linux" ;;
        *)      die "Unsupported platform: $os" ;;
    esac

    if [[ "$PLATFORM" == "linux" ]]; then
        if [[ -f /etc/os-release ]]; then
            # shellcheck source=/dev/null
            . /etc/os-release
            case "${ID:-}" in
                ubuntu|debian|pop|linuxmint|elementary) DISTRO="debian" ;;
                fedora|rhel|centos|rocky|alma)          DISTRO="fedora" ;;
                arch|manjaro|endeavouros)               DISTRO="arch" ;;
                opensuse*|sles)                         DISTRO="suse" ;;
                *)                                      DISTRO="unknown" ;;
            esac
        elif command -v apt-get &>/dev/null; then
            DISTRO="debian"
        elif command -v dnf &>/dev/null; then
            DISTRO="fedora"
        elif command -v pacman &>/dev/null; then
            DISTRO="arch"
        else
            DISTRO="unknown"
        fi
    fi
}

detect_platform
info "Platform: ${BOLD}${PLATFORM}${RESET}${DISTRO:+ ($DISTRO)}"

# ── Default jobs ──────────────────────────────────────────────────────────────
if [[ -z "$JOBS" ]]; then
    if [[ "$PLATFORM" == "macos" ]]; then
        JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    else
        JOBS="$(nproc 2>/dev/null || echo 4)"
    fi
fi
info "Build parallelism: ${BOLD}${JOBS}${RESET} jobs"

# ── Prerequisite checks ──────────────────────────────────────────────────────
check_command() {
    if ! command -v "$1" &>/dev/null; then
        return 1
    fi
    return 0
}

check_cmake_version() {
    local ver
    ver="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    local major minor
    major="${ver%%.*}"
    minor="${ver#*.}"
    if (( major < 3 || (major == 3 && minor < 20) )); then
        die "CMake 3.20+ required (found ${ver}). Please upgrade cmake."
    fi
    ok "CMake ${ver}"
}

check_cxx_compiler() {
    if check_command c++ || check_command g++ || check_command clang++; then
        local cxx
        cxx="$(command -v c++ 2>/dev/null || command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null)"
        ok "C++ compiler: ${cxx}"
    else
        die "No C++ compiler found. Install g++ or clang++."
    fi
}

info "Checking prerequisites..."

if ! check_command cmake; then
    die "CMake not found. Install cmake first (brew install cmake / apt install cmake)."
fi
check_cmake_version
check_cxx_compiler

# ── LLVM detection ────────────────────────────────────────────────────────────
find_llvm_cmake_dir() {
    # User-supplied path
    if [[ -n "$LLVM_DIR" ]]; then
        if [[ -f "${LLVM_DIR}/LLVMConfig.cmake" ]]; then
            ok "LLVM (user-supplied): ${LLVM_DIR}"
            return 0
        else
            die "LLVMConfig.cmake not found in ${LLVM_DIR}"
        fi
    fi

    info "Auto-detecting LLVM installation..."

    # macOS: Homebrew
    if [[ "$PLATFORM" == "macos" ]] && check_command brew; then
        local brew_prefix
        brew_prefix="$(brew --prefix llvm 2>/dev/null || true)"
        if [[ -n "$brew_prefix" && -d "$brew_prefix" ]]; then
            local candidate="${brew_prefix}/lib/cmake/llvm"
            if [[ -f "${candidate}/LLVMConfig.cmake" ]]; then
                LLVM_DIR="$candidate"
                ok "LLVM (Homebrew): ${LLVM_DIR}"
                return 0
            fi
        fi
        # Homebrew opt symlinks can be stale — check Cellar directly
        local cellar_dir
        cellar_dir="$(brew --cellar 2>/dev/null || true)"
        if [[ -n "$cellar_dir" && -d "${cellar_dir}/llvm" ]]; then
            local latest
            latest="$(ls -1d "${cellar_dir}/llvm/"*/ 2>/dev/null | sort -V | tail -1)"
            if [[ -n "$latest" ]]; then
                local candidate="${latest}lib/cmake/llvm"
                if [[ -f "${candidate}/LLVMConfig.cmake" ]]; then
                    LLVM_DIR="$candidate"
                    ok "LLVM (Homebrew Cellar): ${LLVM_DIR}"
                    return 0
                fi
            fi
        fi
    fi

    # Linux: versioned paths
    if [[ "$PLATFORM" == "linux" ]]; then
        local search_dirs=()
        for v in 21 20 19 18 17; do
            search_dirs+=("/usr/lib/llvm-${v}/lib/cmake/llvm")           # Debian
            search_dirs+=("/usr/lib64/llvm${v}/lib/cmake/llvm")          # Fedora
            search_dirs+=("/usr/lib/llvm/${v}/lib/cmake/llvm")           # Arch
        done
        search_dirs+=("/usr/lib/cmake/llvm")                             # Generic
        search_dirs+=("/usr/local/lib/cmake/llvm")

        for d in "${search_dirs[@]}"; do
            if [[ -f "${d}/LLVMConfig.cmake" ]]; then
                LLVM_DIR="$d"
                ok "LLVM: ${LLVM_DIR}"
                return 0
            fi
        done
    fi

    # Generic fallback paths (both platforms)
    local fallbacks=(
        "/usr/local/lib/cmake/llvm"
        "/opt/homebrew/opt/llvm/lib/cmake/llvm"
    )
    for d in "${fallbacks[@]}"; do
        if [[ -f "${d}/LLVMConfig.cmake" ]]; then
            LLVM_DIR="$d"
            ok "LLVM: ${LLVM_DIR}"
            return 0
        fi
    done

    return 1
}

offer_install_llvm() {
    echo ""
    warn "LLVM development libraries not found."
    echo ""

    local install_cmd=""
    case "${PLATFORM}:${DISTRO}" in
        macos:*)    install_cmd="brew install llvm" ;;
        linux:debian) install_cmd="sudo apt-get install -y llvm-17-dev" ;;
        linux:fedora) install_cmd="sudo dnf install -y llvm17-devel" ;;
        linux:arch)   install_cmd="sudo pacman -S --noconfirm llvm" ;;
        linux:suse)   install_cmd="sudo zypper install -y llvm-devel" ;;
        *)
            die "Could not determine how to install LLVM on this system. Use --llvm-dir=<path>."
            ;;
    esac

    echo -e "  Suggested command: ${BOLD}${install_cmd}${RESET}"
    echo ""
    read -rp "Install LLVM now? [y/N] " response
    if [[ "${response,,}" == "y" || "${response,,}" == "yes" ]]; then
        info "Running: ${install_cmd}"
        eval "$install_cmd"
        # Re-detect after install
        if find_llvm_cmake_dir; then
            return 0
        else
            die "LLVM installed but cmake directory not found. Use --llvm-dir=<path>."
        fi
    else
        die "LLVM is required to build SafeC. Use --llvm-dir=<path> or install LLVM manually."
    fi
}

if ! find_llvm_cmake_dir; then
    offer_install_llvm
fi

# ── Build compiler ────────────────────────────────────────────────────────────
echo ""
info "Building SafeC compiler..."

COMPILER_DIR="${SCRIPT_DIR}/compiler"
if [[ ! -d "$COMPILER_DIR" ]]; then
    die "compiler/ directory not found at ${COMPILER_DIR}"
fi

cd "$COMPILER_DIR"

cmake -S . -B build -DLLVM_DIR="${LLVM_DIR}" 2>&1 | tail -5
info "Running cmake --build build -j ${JOBS} ..."
cmake --build build -j "${JOBS}" 2>&1

SAFEC_BIN=""
if [[ -f "build/safec" ]]; then
    SAFEC_BIN="${COMPILER_DIR}/build/safec"
elif [[ -f "build/Release/safec" ]]; then
    SAFEC_BIN="${COMPILER_DIR}/build/Release/safec"
fi

if [[ -z "$SAFEC_BIN" || ! -x "$SAFEC_BIN" ]]; then
    die "Compiler build failed — safec binary not found."
fi
ok "Compiler built: ${SAFEC_BIN}"

# ── Build safeguard ──────────────────────────────────────────────────────────
SAFEGUARD_DIR="${SCRIPT_DIR}/safeguard"
SAFEGUARD_BIN=""

if [[ "$SKIP_SAFEGUARD" == true ]]; then
    info "Skipping safeguard build (--skip-safeguard)"
elif [[ ! -d "$SAFEGUARD_DIR" ]]; then
    warn "safeguard/ directory not found — skipping."
else
    echo ""
    info "Building safeguard package manager..."
    cd "$SAFEGUARD_DIR"

    if cmake -S . -B build 2>&1 | tail -3 && \
       cmake --build build -j "${JOBS}" 2>&1; then
        if [[ -f "build/safeguard" ]]; then
            SAFEGUARD_BIN="${SAFEGUARD_DIR}/build/safeguard"
        elif [[ -f "build/Release/safeguard" ]]; then
            SAFEGUARD_BIN="${SAFEGUARD_DIR}/build/Release/safeguard"
        fi
        if [[ -n "$SAFEGUARD_BIN" ]]; then
            ok "Safeguard built: ${SAFEGUARD_BIN}"
        else
            warn "Safeguard build appeared to succeed but binary not found."
        fi
    else
        warn "Safeguard build failed (non-fatal). You can retry later or use --skip-safeguard."
    fi
fi

cd "$SCRIPT_DIR"

# ── Environment setup ────────────────────────────────────────────────────────
MARKER="# SafeC environment"

detect_shell_profile() {
    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    case "$shell_name" in
        zsh)  echo "${HOME}/.zshrc" ;;
        fish) echo "${HOME}/.config/fish/config.fish" ;;
        *)    echo "${HOME}/.bashrc" ;;
    esac
}

setup_env() {
    if [[ "$SKIP_ENV" == true ]]; then
        info "Skipping environment setup (--skip-env)"
        return
    fi

    echo ""
    info "Configuring shell environment..."

    local profile
    profile="$(detect_shell_profile)"
    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"

    local bin_dir="${COMPILER_DIR}/build"
    local safec_home="${SCRIPT_DIR}"

    # Check if already configured
    if [[ -f "$profile" ]] && grep -qF "$MARKER" "$profile" 2>/dev/null; then
        ok "Environment already configured in ${profile}"
        return
    fi

    echo ""
    info "Will add the following to ${BOLD}${profile}${RESET}:"

    if [[ "$shell_name" == "fish" ]]; then
        local env_block
        env_block=$(cat <<ENVEOF
${MARKER}
set -gx SAFEC_HOME "${safec_home}"
fish_add_path "${bin_dir}"
ENVEOF
)
        if [[ -n "$SAFEGUARD_BIN" ]]; then
            env_block+=$'\n'"fish_add_path \"$(dirname "$SAFEGUARD_BIN")\""
        fi
    else
        local env_block
        env_block=$(cat <<ENVEOF
${MARKER}
export SAFEC_HOME="${safec_home}"
export PATH="${bin_dir}:\$PATH"
ENVEOF
)
        if [[ -n "$SAFEGUARD_BIN" ]]; then
            local sg_dir
            sg_dir="$(dirname "$SAFEGUARD_BIN")"
            if [[ "$sg_dir" != "$bin_dir" ]]; then
                env_block+=$'\n'"export PATH=\"${sg_dir}:\$PATH\""
            fi
        fi
    fi

    echo "$env_block"
    echo ""
    read -rp "Apply these changes? [Y/n] " response
    if [[ "${response,,}" == "n" || "${response,,}" == "no" ]]; then
        warn "Skipped environment setup. Add the above lines manually."
        return
    fi

    echo "" >> "$profile"
    echo "$env_block" >> "$profile"
    ok "Updated ${profile}"
    info "Run ${BOLD}source ${profile}${RESET} or open a new terminal to apply."
}

setup_env

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
info "Verifying installation..."

VERIFY_OK=true

# Check safec binary
if "${SAFEC_BIN}" --help &>/dev/null; then
    ok "safec --help works"
else
    warn "safec --help returned non-zero (may be expected if --help isn't implemented)"
fi

# Compile hello.sc
HELLO_SC="${COMPILER_DIR}/examples/hello.sc"
if [[ -f "$HELLO_SC" ]]; then
    TMPLL="$(mktemp /tmp/safec_test_XXXXXX.ll)"
    if "${SAFEC_BIN}" "$HELLO_SC" --emit-llvm -o "$TMPLL" 2>/dev/null; then
        ok "Compiled examples/hello.sc to LLVM IR"
    else
        warn "Failed to compile examples/hello.sc (non-fatal)"
        VERIFY_OK=false
    fi
    rm -f "$TMPLL"
else
    warn "examples/hello.sc not found — skipping compile test"
fi

# Count std headers
STD_DIR="${SCRIPT_DIR}/std"
if [[ -d "$STD_DIR" ]]; then
    local_count="$(find "$STD_DIR" -name '*.h' -o -name '*.sc' | wc -l | tr -d ' ')"
    ok "Standard library: ${local_count} files in std/"
else
    warn "std/ directory not found"
fi

# Check safeguard
if [[ -n "$SAFEGUARD_BIN" ]]; then
    if "${SAFEGUARD_BIN}" --help &>/dev/null || "${SAFEGUARD_BIN}" help &>/dev/null; then
        ok "safeguard works"
    else
        warn "safeguard returned non-zero"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  SafeC installation complete!${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}SAFEC_HOME${RESET}  ${SCRIPT_DIR}"
echo -e "  ${BOLD}safec${RESET}       ${SAFEC_BIN}"
[[ -n "$SAFEGUARD_BIN" ]] && echo -e "  ${BOLD}safeguard${RESET}   ${SAFEGUARD_BIN}"
echo ""
echo -e "  ${BOLD}Quick start:${RESET}"
echo "    safec examples/hello.sc --emit-llvm -o hello.ll"
echo "    clang hello.ll -o hello"
echo "    ./hello"
echo ""
if [[ "$SKIP_ENV" != true ]]; then
    echo -e "  ${YELLOW}Restart your terminal or run:${RESET}"
    echo "    source $(detect_shell_profile)"
    echo ""
fi
