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

.PARAMETER Std
    The prebuilt std sections to install, comma-separated: core (std/*.sc),
    collections, gui, ml, crypto, net, http, dsp, fs, sync, async, alloc, simd, ...
    (std's top-level directories), or "all" (the default). std's sources are
    always installed: a build that needs a section that isn't installed
    compiles its units from source. "+ml,gui" adds sections to an existing
    install, from the release it was installed from; nothing else changes.

.PARAMETER SkipEnv
    Skip PATH/SAFEC_HOME environment-variable setup.

.EXAMPLE
    irm https://raw.githubusercontent.com/safec-org/SafeC/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Prefix C:\safec -Version v1.0.0-beta-4

.EXAMPLE
    .\install.ps1 -Std core,collections,gui

.EXAMPLE
    .\install.ps1 -Std +ml
#>
param(
    [string]$Prefix = "$HOME\safec",
    [string]$Version = "",
    [string]$Std = "all",
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

# -Std: "all", a list of sections, or "+" and sections to add.
$addStd = $Std.StartsWith("+")
$stdList = $Std.TrimStart("+")
if ($stdList -notmatch '^[a-z0-9_,]+$') { Die "-Std: 'all' or comma-separated sections, e.g. -Std core,collections" }
if ($addStd) {
    if ($stdList -eq "all") { Die "-Std +<sections> takes the sections to add, e.g. -Std +ml,gui" }
    if (-not (Test-Path (Join-Path $Prefix "bin\safec.exe"))) { Die "No SafeC install in $Prefix to add std sections to (use -Prefix)" }
    if (-not $Version) {
        $releaseFile = Join-Path $Prefix ".release"
        if (-not (Test-Path $releaseFile)) { Die "$Prefix doesn't record its release; give it with -Version <tag>" }
        $Version = (Get-Content $releaseFile | Select-Object -First 1).Trim()
    }
}
if (-not $Version) { $Version = "latest" }

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
$release = $releases | Where-Object { $_.assets | Where-Object { $_.name -eq $asset } } | Select-Object -First 1
$assetInfo = if ($release) { $release.assets | Where-Object { $_.name -eq $asset } | Select-Object -First 1 }
if (-not $assetInfo) {
    Die "Could not find asset '$asset' in release '$Version' — check https://github.com/$Repo/releases"
}
$releaseTag = $release.tag_name
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

    Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force
    $extractedDir = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "safec-*" } | Select-Object -First 1
    if (-not $extractedDir) { Die "Unexpected archive layout" }

    # The prebuilt std: lib\<triple>\std\<section>.a and .units, one pair
    # per section. The ones -Std names (all by default) are installed.
    $stdSrc = Get-ChildItem -Path (Join-Path $extractedDir.FullName "lib") -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "std" } | Where-Object { Test-Path $_ } | Select-Object -First 1
    $sections = @()
    if ($stdSrc) {
        $available = @(Get-ChildItem -Path $stdSrc -Filter "*.units" | ForEach-Object { $_.BaseName })
        if ($stdList -eq "all") { $sections = $available }
        else {
            $sections = @($stdList.Split(",") | Where-Object { $_ })
            foreach ($s in $sections) {
                if ($available -notcontains $s) { Die "No std section '$s' in this release (sections: $($available -join ' '))" }
            }
        }
        $stdRel = $stdSrc.Substring($extractedDir.FullName.Length).TrimStart("\", "/")
        $stdDst = Join-Path $Prefix $stdRel
    } elseif ($stdList -ne "all" -or $addStd) {
        Die "This release has no sectioned prebuilt std (lib\<triple>\std\); install it without -Std"
    }
    function Install-Sections {
        New-Item -ItemType Directory -Force -Path $stdDst | Out-Null
        foreach ($s in $sections) {
            Copy-Item -Path (Join-Path $stdSrc "$s.units") -Destination $stdDst -Force
            $a = Join-Path $stdSrc "$s.a"
            if (Test-Path $a) { Copy-Item -Path $a -Destination $stdDst -Force }
        }
    }

    if ($addStd) {
        Install-Sections
        Ok "Added std sections to ${stdDst}: $($sections -join ' ')"
        Info "Installed std sections: $((Get-ChildItem -Path $stdDst -Filter '*.units' | ForEach-Object { $_.BaseName }) -join ' ')"
        exit 0
    }

    Info "Extracting to $Prefix..."
    New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
    # Everything but the prebuilt std, then the sections chosen; sections
    # left from an earlier install (another release's) are removed first.
    if ($stdSrc) {
        if (Test-Path $stdDst) { Remove-Item -Recurse -Force $stdDst }
        $moved = Join-Path $tmpDir "prebuilt-std"
        Move-Item -Path $stdSrc -Destination $moved
        $stdSrc = $moved
    }
    Copy-Item -Path (Join-Path $extractedDir.FullName "*") -Destination $Prefix -Recurse -Force
    if ($stdSrc) { Install-Sections }
    if ($releaseTag) { Set-Content -Path (Join-Path $Prefix ".release") -Value $releaseTag }
    Ok "Installed to $Prefix"
    if ($stdSrc) {
        if ($stdList -eq "all") { Info "Prebuilt std: all $($sections.Count) sections" }
        else { Info "Prebuilt std sections: $($sections -join ' ') (add more with: .\install.ps1 -Prefix $Prefix -Std +<sections>)" }
    }
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
