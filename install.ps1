#Requires -Version 5.1
<#
.SYNOPSIS
    SafeC Install Script — Windows

.DESCRIPTION
    Downloads a prebuilt safec/safeguard/sc-lsp release from GitHub Releases
    and installs it — no LLVM, no Visual Studio, no compiler needed on the
    machine you're installing to.

.PARAMETER Prefix
    Install directory. Default: $HOME\safec

.PARAMETER Version
    Release tag to install, e.g. v0.2.0. Default: latest

.PARAMETER SkipEnv
    Skip PATH/SAFEC_HOME environment-variable setup.

.EXAMPLE
    irm https://raw.githubusercontent.com/safec-org/SafeC/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Prefix C:\safec -Version v0.2.0
#>
param(
    [string]$Prefix = "$HOME\safec",
    [string]$Version = "latest",
    [switch]$SkipEnv
)

$ErrorActionPreference = "Stop"
$Repo = "safec-org/SafeC"

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
if ($Version -eq "latest") {
    $apiUrl = "https://api.github.com/repos/$Repo/releases/latest"
} else {
    $apiUrl = "https://api.github.com/repos/$Repo/releases/tags/$Version"
}

Info "Resolving release ($Version)..."
try {
    $release = Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "safec-install-script" }
} catch {
    Die "Failed to query GitHub Releases API: $_"
}
$assetInfo = $release.assets | Where-Object { $_.name -eq $asset } | Select-Object -First 1
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

# ── Environment setup ──────────────────────────────────────────────────────────
$binDir = Join-Path $Prefix "bin"
if (-not $SkipEnv) {
    [System.Environment]::SetEnvironmentVariable("SAFEC_HOME", $Prefix, "User")
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
