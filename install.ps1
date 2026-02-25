#Requires -Version 5.1
<#
.SYNOPSIS
    SafeC Install Script — Windows PowerShell

.DESCRIPTION
    Automates building the SafeC compiler and safeguard package manager.
    Auto-detects LLVM, configures environment variables, and verifies the install.

.PARAMETER LlvmDir
    Path to the LLVM cmake directory (auto-detected if omitted).

.PARAMETER Jobs
    Number of parallel build jobs (default: processor count).

.PARAMETER SkipSafeguard
    Skip building the safeguard package manager.

.PARAMETER SkipEnv
    Skip environment variable configuration.

.PARAMETER UseNinja
    Use the Ninja build generator instead of the default.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -LlvmDir "C:\Program Files\LLVM\lib\cmake\llvm" -Jobs 8
    .\install.ps1 -SkipSafeguard -UseNinja
#>

param(
    [string]$LlvmDir = "",
    [int]$Jobs = 0,
    [switch]$SkipSafeguard,
    [switch]$SkipEnv,
    [switch]$UseNinja
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ── Output helpers ────────────────────────────────────────────────────────────
function Write-Info  { param([string]$Msg) Write-Host "[info]  " -ForegroundColor Cyan -NoNewline; Write-Host $Msg }
function Write-Ok    { param([string]$Msg) Write-Host "[ok]    " -ForegroundColor Green -NoNewline; Write-Host $Msg }
function Write-Warn  { param([string]$Msg) Write-Host "[warn]  " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
function Write-Err   { param([string]$Msg) Write-Host "[error] " -ForegroundColor Red -NoNewline; Write-Host $Msg }

function Exit-Fatal {
    param([string]$Msg)
    Write-Err $Msg
    exit 1
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host " ╔════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host " ║     SafeC Installer — Windows PowerShell      ║" -ForegroundColor Cyan
Write-Host " ╚════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Default jobs ──────────────────────────────────────────────────────────────
if ($Jobs -le 0) {
    $Jobs = [Environment]::ProcessorCount
    if ($Jobs -le 0) { $Jobs = 4 }
}
Write-Info "Build parallelism: $Jobs jobs"

# ── Prerequisite checks ──────────────────────────────────────────────────────
Write-Info "Checking prerequisites..."

# CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Exit-Fatal "CMake not found. Install from https://cmake.org/download/ or: winget install Kitware.CMake"
}
$cmakeVer = (cmake --version | Select-Object -First 1) -replace '.*?(\d+\.\d+\.\d+).*', '$1'
Write-Ok "CMake $cmakeVer"

# C++ compiler
$cxxFound = $false
foreach ($cxx in @("cl", "clang++", "g++")) {
    if (Get-Command $cxx -ErrorAction SilentlyContinue) {
        Write-Ok "C++ compiler: $cxx"
        $cxxFound = $true
        break
    }
}
if (-not $cxxFound) {
    Exit-Fatal "No C++ compiler found. Install Visual Studio Build Tools, LLVM, or MinGW."
}

# Build system
$generator = ""
if ($UseNinja) {
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        Exit-Fatal "-UseNinja specified but ninja not found. Install via: choco install ninja"
    }
    $generator = "-G Ninja"
    Write-Ok "Build system: Ninja"
} else {
    if (Get-Command msbuild -ErrorAction SilentlyContinue) {
        Write-Ok "Build system: MSBuild"
    } elseif (Get-Command ninja -ErrorAction SilentlyContinue) {
        $generator = "-G Ninja"
        Write-Ok "Build system: Ninja (auto-detected)"
    } else {
        Write-Warn "No MSBuild or Ninja found — cmake will use its default generator."
    }
}

# ── LLVM detection ───────────────────────────────────────────────────────────
function Find-LlvmDir {
    # User-supplied
    if ($LlvmDir -ne "") {
        if (Test-Path (Join-Path $LlvmDir "LLVMConfig.cmake")) {
            Write-Ok "LLVM (user-supplied): $LlvmDir"
            return $LlvmDir
        } else {
            Exit-Fatal "LLVMConfig.cmake not found in $LlvmDir"
        }
    }

    Write-Info "Auto-detecting LLVM installation..."

    # vcpkg
    if ($env:VCPKG_ROOT) {
        $try = Join-Path $env:VCPKG_ROOT "installed\x64-windows\share\llvm"
        if (Test-Path (Join-Path $try "LLVMConfig.cmake")) {
            Write-Ok "LLVM (vcpkg): $try"
            return $try
        }
    }

    # Standard install paths
    $searchPaths = @(
        "C:\Program Files\LLVM\lib\cmake\llvm",
        "C:\Program Files (x86)\LLVM\lib\cmake\llvm"
    )
    foreach ($p in $searchPaths) {
        if (Test-Path (Join-Path $p "LLVMConfig.cmake")) {
            Write-Ok "LLVM (system): $p"
            return $p
        }
    }

    # Chocolatey
    if ($env:ChocolateyInstall) {
        $chocoLibs = Get-ChildItem (Join-Path $env:ChocolateyInstall "lib") -Directory -Filter "llvm*" -ErrorAction SilentlyContinue
        foreach ($d in $chocoLibs) {
            $try = Join-Path $d.FullName "lib\cmake\llvm"
            if (Test-Path (Join-Path $try "LLVMConfig.cmake")) {
                Write-Ok "LLVM (Chocolatey): $try"
                return $try
            }
        }
    }

    # Scoop
    $scoopPath = Join-Path $env:USERPROFILE "scoop\apps\llvm\current\lib\cmake\llvm"
    if (Test-Path (Join-Path $scoopPath "LLVMConfig.cmake")) {
        Write-Ok "LLVM (Scoop): $scoopPath"
        return $scoopPath
    }

    return $null
}

$detectedLlvm = Find-LlvmDir

if (-not $detectedLlvm) {
    Write-Host ""
    Write-Warn "LLVM development libraries not found."
    Write-Host ""
    Write-Host "  You can install LLVM using one of:"
    Write-Host "    winget install LLVM.LLVM     (winget)"
    Write-Host "    choco install llvm           (Chocolatey)"
    Write-Host "    scoop install llvm           (Scoop)"
    Write-Host "    Download from https://github.com/llvm/llvm-project/releases"
    Write-Host ""

    $response = Read-Host "Install LLVM via winget now? [y/N]"
    if ($response -match "^[yY]") {
        Write-Info "Running: winget install LLVM.LLVM"
        winget install LLVM.LLVM
        $detectedLlvm = Find-LlvmDir
        if (-not $detectedLlvm) {
            Exit-Fatal "LLVM installed but cmake directory not found. Re-run with -LlvmDir."
        }
    } else {
        Exit-Fatal "LLVM is required to build SafeC. Use -LlvmDir or install LLVM manually."
    }
}

$LlvmDir = $detectedLlvm

# ── Build compiler ───────────────────────────────────────────────────────────
Write-Host ""
Write-Info "Building SafeC compiler..."

$compilerDir = Join-Path $ScriptDir "compiler"
if (-not (Test-Path (Join-Path $compilerDir "CMakeLists.txt"))) {
    Exit-Fatal "compiler\CMakeLists.txt not found at $compilerDir"
}

Push-Location $compilerDir
try {
    $cmakeArgs = @("-S", ".", "-B", "build", "-DLLVM_DIR=$LlvmDir")
    if ($generator) { $cmakeArgs = @("-S", ".", "-B", "build") + ($generator -split " ") + @("-DLLVM_DIR=$LlvmDir") }

    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Exit-Fatal "CMake configure failed." }

    & cmake --build build --config Release -j $Jobs
    if ($LASTEXITCODE -ne 0) { Exit-Fatal "Compiler build failed." }

    # Find binary
    $safecBin = $null
    foreach ($candidate in @(
        (Join-Path $compilerDir "build\safec.exe"),
        (Join-Path $compilerDir "build\Release\safec.exe"),
        (Join-Path $compilerDir "build\Debug\safec.exe")
    )) {
        if (Test-Path $candidate) {
            $safecBin = $candidate
            break
        }
    }

    if (-not $safecBin) {
        Exit-Fatal "Compiler build appeared to succeed but safec.exe not found."
    }
    Write-Ok "Compiler built: $safecBin"
} finally {
    Pop-Location
}

# ── Build safeguard ──────────────────────────────────────────────────────────
$safeguardBin = $null
$safeguardDir = Join-Path $ScriptDir "safeguard"

if ($SkipSafeguard) {
    Write-Info "Skipping safeguard build (-SkipSafeguard)"
} elseif (-not (Test-Path (Join-Path $safeguardDir "CMakeLists.txt"))) {
    Write-Warn "safeguard\ directory not found — skipping."
} else {
    Write-Host ""
    Write-Info "Building safeguard package manager..."
    Write-Warn "Note: safeguard uses POSIX APIs and may not compile with MSVC."

    Push-Location $safeguardDir
    try {
        $sgArgs = @("-S", ".", "-B", "build")
        if ($generator) { $sgArgs += ($generator -split " ") }

        & cmake @sgArgs 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Safeguard cmake configure failed (non-fatal)."
            Write-Warn "Safeguard uses POSIX APIs (sys/wait.h) — try WSL or -SkipSafeguard."
        } else {
            & cmake --build build --config Release -j $Jobs 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Safeguard build failed (non-fatal)."
                Write-Warn "Safeguard uses POSIX APIs — consider using WSL or -SkipSafeguard."
            } else {
                foreach ($candidate in @(
                    (Join-Path $safeguardDir "build\safeguard.exe"),
                    (Join-Path $safeguardDir "build\Release\safeguard.exe")
                )) {
                    if (Test-Path $candidate) {
                        $safeguardBin = $candidate
                        break
                    }
                }
                if ($safeguardBin) {
                    Write-Ok "Safeguard built: $safeguardBin"
                } else {
                    Write-Warn "Safeguard build succeeded but binary not found."
                }
            }
        }
    } finally {
        Pop-Location
    }
}

# ── Environment setup ────────────────────────────────────────────────────────
if ($SkipEnv) {
    Write-Info "Skipping environment setup (-SkipEnv)"
} else {
    Write-Host ""
    Write-Info "Configuring environment variables..."

    $safecBinDir = Split-Path -Parent $safecBin

    # Check current SAFEC_HOME
    $currentHome = [System.Environment]::GetEnvironmentVariable("SAFEC_HOME", "User")
    if ($currentHome -eq $ScriptDir) {
        Write-Ok "SAFEC_HOME already set correctly."
    } else {
        Write-Host ""
        Write-Host "  SAFEC_HOME = $ScriptDir"
        Write-Host "  PATH      += $safecBinDir"
        if ($safeguardBin) {
            $sgDir = Split-Path -Parent $safeguardBin
            if ($sgDir -ne $safecBinDir) {
                Write-Host "  PATH      += $sgDir"
            }
        }
        Write-Host ""

        $response = Read-Host "Apply environment changes? [Y/n]"
        if ($response -match "^[nN]") {
            Write-Warn "Skipped. Set SAFEC_HOME and PATH manually."
        } else {
            # Set SAFEC_HOME
            [System.Environment]::SetEnvironmentVariable("SAFEC_HOME", $ScriptDir, "User")
            $env:SAFEC_HOME = $ScriptDir
            Write-Ok "Set SAFEC_HOME = $ScriptDir"

            # Update PATH (duplicate check)
            $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
            if (-not $userPath) { $userPath = "" }

            $pathsToAdd = @($safecBinDir)
            if ($safeguardBin) {
                $sgDir = Split-Path -Parent $safeguardBin
                if ($sgDir -ne $safecBinDir) {
                    $pathsToAdd += $sgDir
                }
            }

            foreach ($dir in $pathsToAdd) {
                $pathEntries = $userPath -split ";"
                if ($pathEntries -notcontains $dir) {
                    if ($userPath -ne "" -and -not $userPath.EndsWith(";")) {
                        $userPath += ";"
                    }
                    $userPath += $dir
                    Write-Ok "Added $dir to user PATH."
                } else {
                    Write-Ok "PATH already contains $dir"
                }
            }

            [System.Environment]::SetEnvironmentVariable("Path", $userPath, "User")
            # Update current session
            $env:Path = "$userPath;$($env:Path)"

            Write-Ok "Environment configured. Restart your terminal to apply."
        }
    }
}

# ── Verification ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Info "Verifying installation..."

try {
    & $safecBin --help 2>$null | Out-Null
    Write-Ok "safec --help works"
} catch {
    Write-Warn "safec --help returned non-zero"
}

# Compile hello.sc
$helloSc = Join-Path $ScriptDir "compiler\examples\hello.sc"
if (Test-Path $helloSc) {
    $tmpLL = Join-Path $env:TEMP "safec_test_$([System.IO.Path]::GetRandomFileName()).ll"
    try {
        & $safecBin $helloSc --emit-llvm -o $tmpLL 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Compiled examples\hello.sc to LLVM IR"
        } else {
            Write-Warn "Failed to compile examples\hello.sc (non-fatal)"
        }
    } catch {
        Write-Warn "Failed to compile examples\hello.sc (non-fatal)"
    }
    Remove-Item $tmpLL -Force -ErrorAction SilentlyContinue
} else {
    Write-Warn "examples\hello.sc not found — skipping compile test"
}

# Count std files
$stdDir = Join-Path $ScriptDir "std"
if (Test-Path $stdDir) {
    $fileCount = (Get-ChildItem $stdDir -Recurse -Include "*.h","*.sc" | Measure-Object).Count
    Write-Ok "Standard library: $fileCount files in std\"
} else {
    Write-Warn "std\ directory not found"
}

if ($safeguardBin) {
    try {
        & $safeguardBin help 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "safeguard works"
        } else {
            Write-Warn "safeguard returned non-zero"
        }
    } catch {
        Write-Warn "safeguard returned non-zero"
    }
}

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host " ════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "   SafeC installation complete!" -ForegroundColor Green
Write-Host " ════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  SAFEC_HOME   $ScriptDir"
Write-Host "  safec        $safecBin"
if ($safeguardBin) {
    Write-Host "  safeguard    $safeguardBin"
}
Write-Host ""
Write-Host "  Quick start:"
Write-Host "    safec examples\hello.sc --emit-llvm -o hello.ll"
Write-Host "    clang hello.ll -o hello.exe"
Write-Host "    .\hello.exe"
Write-Host ""
if (-not $SkipEnv) {
    Write-Host "  Restart your terminal to pick up PATH changes." -ForegroundColor Yellow
    Write-Host ""
}
