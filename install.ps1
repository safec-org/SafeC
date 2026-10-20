#Requires -Version 5.1
<#
.SYNOPSIS
    SafeC Install Script — Windows

.DESCRIPTION
    Downloads a prebuilt safec/safeguard/sc-lsp release from GitHub Releases
    and installs it — no CMake and no build needed. safec and safeguard do
    call clang at run time (to compile the LLVM IR safec emits, link, and
    import C headers), so this script also finds a clang >= 21 and points
    SAFEC_CLANG at it, or says how to install one.

.PARAMETER Prefix
    Install directory. Default: $HOME\safec

.PARAMETER Version
    Release tag to install, e.g. v1.0.0-beta-4. Default: the newest release,
    including pre-releases.

.PARAMETER SkipEnv
    Skip PATH/SAFEC_HOME environment-variable setup.

.EXAMPLE
    irm https://raw.githubusercontent.com/safec-org/SafeC/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Prefix C:\safec -Version v1.0.0-beta-4
#>
param(
    [string]$Prefix = "$HOME\safec",
    [string]$Version = "latest",
    [switch]$SkipEnv
)

$ErrorActionPreference = "Stop"
$Repo = "safec-org/SafeC"
# The release binaries are built against LLVM 21 (see
# .github/workflows/release.yml); an older clang can't read safec's IR.
$MinClang = 21

function Info($msg)  { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Die($msg)   { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }

# Only x86_64 is published today (see .github/workflows/release.yml) — ARM64
# Windows would need its own build job before this script could support it.
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "AMD64") {
    Die "No prebuilt release for Windows/$arch yet (only x86_64 is published) — see https://github.com/$Repo/releases"
}
$asset = "safec-windows-x86_64.zip"
Info "Platform: windows/x86_64 -> $asset"

# ── Resolve download URL ──────────────────────────────────────────────────────
# GitHub's /releases/latest skips pre-releases (and 404s when every release
# is one, as with 1.0.0-beta-N), so "latest" reads the release list instead,
# which is newest-first and includes them, and takes the first with our asset.
if ($Version -eq "latest") {
    $apiUrl = "https://api.github.com/repos/$Repo/releases"
} else {
    $apiUrl = "https://api.github.com/repos/$Repo/releases/tags/$Version"
}

Info "Resolving release ($Version)..."
try {
    $releases = @(Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "safec-install-script" })
} catch {
    Die "Failed to query GitHub Releases API: $_"
}
$assetInfo = $releases | ForEach-Object { $_.assets } | Where-Object { $_.name -eq $asset } | Select-Object -First 1
if (-not $assetInfo) {
    Die "Could not find asset '$asset' in release '$Version' — check https://github.com/$Repo/releases"
}
$downloadUrl = $assetInfo.browser_download_url
Ok "Found: $downloadUrl"

# ── Download + extract ────────────────────────────────────────────────────────
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmpDir | Out-Null
$zipPath = Join-Path $tmpDir $asset

try {
    Info "Downloading..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
    Ok "Downloaded $([math]::Round((Get-Item $zipPath).Length / 1MB, 1)) MB"

    Info "Extracting to $Prefix..."
    New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

    $extractedDir = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "safec-*" } | Select-Object -First 1
    if (-not $extractedDir) { Die "Unexpected archive layout" }
    Copy-Item -Path (Join-Path $extractedDir.FullName "*") -Destination $Prefix -Recurse -Force
    Ok "Installed to $Prefix"
} finally {
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
}

# ── Check the installed binary starts ─────────────────────────────────────────
$binDir = Join-Path $Prefix "bin"
$safecExe = Join-Path $binDir "safec.exe"
try {
    $ver = & $safecExe --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$ver" }
    Ok "$ver"
} catch {
    Warn "Installed safec doesn't start: $_"
}

# ── Find a clang >= $MinClang ──────────────────────────────────────────────────
function Get-ClangMajor($path) {
    try {
        $out = & $path --version 2>$null | Select-Object -First 1
        if ($out -match 'clang version (\d+)') { return [int]$Matches[1] }
    } catch {}
    return $null
}
$clangPath = $null
$candidates = @()
$onPath = Get-Command clang -ErrorAction SilentlyContinue
if ($onPath) { $candidates += $onPath.Source }
$candidates += @("$env:ProgramFiles\LLVM\bin\clang.exe", "${env:ProgramFiles(x86)}\LLVM\bin\clang.exe")
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
        $m = Get-ClangMajor $c
        if ($m -and $m -ge $MinClang) { $clangPath = $c; break }
    }
}
$setClang = $false
if ($clangPath) {
    Ok "clang: $clangPath"
    if (-not $onPath -or $onPath.Source -ne $clangPath) { $setClang = $true }
} else {
    if ($onPath) {
        Warn "clang at $($onPath.Source) is version $(Get-ClangMajor $onPath.Source); SafeC needs clang $MinClang or newer."
    } else {
        Warn "No clang found; SafeC needs clang $MinClang or newer to build programs."
    }
    Write-Host "  winget install LLVM.LLVM   (or https://github.com/llvm/llvm-project/releases)"
    Write-Host "  Then re-run this script, or set SAFEC_CLANG to that clang.exe."
}

# ── Environment setup ──────────────────────────────────────────────────────────
if (-not $SkipEnv) {
    [System.Environment]::SetEnvironmentVariable("SAFEC_HOME", $Prefix, "User")
    if ($setClang) {
        $clangxx = Join-Path (Split-Path $clangPath) "clang++.exe"
        [System.Environment]::SetEnvironmentVariable("SAFEC_CLANG", $clangPath, "User")
        [System.Environment]::SetEnvironmentVariable("SAFEC_CLANGXX", $clangxx, "User")
        Ok "Set SAFEC_CLANG=$clangPath"
    }
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$binDir*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$binDir", "User")
        Ok "Added SAFEC_HOME and $binDir to your User PATH (restart your terminal to pick them up)"
    } else {
        Info "PATH already contains $binDir"
    }
} else {
    Info "Skipping environment setup (-SkipEnv). Add manually:"
    Write-Host "  `$env:SAFEC_HOME = `"$Prefix`""
    Write-Host "  `$env:Path = `"$binDir;`$env:Path`""
    if ($setClang) { Write-Host "  `$env:SAFEC_CLANG = `"$clangPath`"" }
}

Write-Host ""
Write-Host "SafeC installed." -ForegroundColor White
Write-Host "  SAFEC_HOME  $Prefix"
Write-Host "  safec       $binDir\safec.exe"
Write-Host "  safeguard   $binDir\safeguard.exe"
Write-Host "  sc-lsp      $binDir\sc-lsp.exe"
Write-Host ""
Write-Host "Open a new terminal, then try:"
Write-Host "  safeguard new hello; cd hello; safeguard run"
