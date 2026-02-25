@echo off
setlocal EnableDelayedExpansion

:: ─────────────────────────────────────────────────────────────────────────────
::  SafeC Install Script — Windows (cmd.exe)
::  Usage:  install.bat [OPTIONS]
::
::  Options:
::    --llvm-dir=<path>   Path to LLVM cmake directory (auto-detected if omitted)
::    --jobs=N            Parallel build jobs (default: %NUMBER_OF_PROCESSORS%)
::    --skip-safeguard    Skip building the safeguard package manager
::    --skip-env          Skip environment variable configuration
::    --ninja             Use Ninja generator instead of default (MSBuild)
::    --help              Show this help message
:: ─────────────────────────────────────────────────────────────────────────────

set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "LLVM_DIR="
set "JOBS=%NUMBER_OF_PROCESSORS%"
set "SKIP_SAFEGUARD=0"
set "SKIP_ENV=0"
set "USE_NINJA=0"

:: ── Parse arguments ──────────────────────────────────────────────────────────
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--skip-safeguard" (
    set "SKIP_SAFEGUARD=1"
    shift
    goto :parse_args
)
if "%~1"=="--skip-env" (
    set "SKIP_ENV=1"
    shift
    goto :parse_args
)
if "%~1"=="--ninja" (
    set "USE_NINJA=1"
    shift
    goto :parse_args
)

:: Handle --key=value style arguments
set "_ARG=%~1"
echo !_ARG! | findstr /b /c:"--llvm-dir=" >nul 2>&1
if !errorlevel!==0 (
    set "LLVM_DIR=!_ARG:--llvm-dir=!"
    shift
    goto :parse_args
)
echo !_ARG! | findstr /b /c:"--jobs=" >nul 2>&1
if !errorlevel!==0 (
    set "JOBS=!_ARG:--jobs=!"
    shift
    goto :parse_args
)

echo [error] Unknown option: %~1 (use --help for usage)
exit /b 1

:args_done

echo.
echo  ╔════════════════════════════════════════════════╗
echo  ║       SafeC Installer — Windows (cmd)         ║
echo  ╚════════════════════════════════════════════════╝
echo.

:: ── Prerequisite checks ─────────────────────────────────────────────────────
echo [info]  Checking prerequisites...

:: Check cmake
where cmake >nul 2>&1
if !errorlevel! neq 0 (
    echo [error] CMake not found. Install from https://cmake.org/download/
    exit /b 1
)
for /f "tokens=3" %%v in ('cmake --version 2^>nul ^| findstr /r "cmake version"') do (
    echo [ok]    CMake %%v
)

:: Check C++ compiler
set "CXX_FOUND=0"
where cl.exe >nul 2>&1 && set "CXX_FOUND=1" && echo [ok]    C++ compiler: cl.exe (MSVC)
if !CXX_FOUND!==0 (
    where clang++ >nul 2>&1 && set "CXX_FOUND=1" && echo [ok]    C++ compiler: clang++
)
if !CXX_FOUND!==0 (
    where g++ >nul 2>&1 && set "CXX_FOUND=1" && echo [ok]    C++ compiler: g++
)
if !CXX_FOUND!==0 (
    echo [error] No C++ compiler found. Install Visual Studio Build Tools, LLVM, or MinGW.
    exit /b 1
)

:: Check build system
set "GENERATOR="
if !USE_NINJA!==1 (
    where ninja >nul 2>&1
    if !errorlevel! neq 0 (
        echo [error] --ninja specified but ninja not found. Install via: choco install ninja
        exit /b 1
    )
    set "GENERATOR=-G Ninja"
    echo [ok]    Build system: Ninja
) else (
    where msbuild >nul 2>&1
    if !errorlevel!==0 (
        echo [ok]    Build system: MSBuild
    ) else (
        where ninja >nul 2>&1
        if !errorlevel!==0 (
            set "GENERATOR=-G Ninja"
            echo [ok]    Build system: Ninja (auto-detected)
        ) else (
            echo [warn]  No MSBuild or Ninja found — cmake will use its default generator.
        )
    )
)

:: ── LLVM detection ──────────────────────────────────────────────────────────
if not "!LLVM_DIR!"=="" (
    if exist "!LLVM_DIR!\LLVMConfig.cmake" (
        echo [ok]    LLVM (user-supplied^): !LLVM_DIR!
        goto :llvm_found
    ) else (
        echo [error] LLVMConfig.cmake not found in !LLVM_DIR!
        exit /b 1
    )
)

echo [info]  Auto-detecting LLVM installation...

:: Check VCPKG
if defined VCPKG_ROOT (
    set "_TRY=!VCPKG_ROOT!\installed\x64-windows\share\llvm"
    if exist "!_TRY!\LLVMConfig.cmake" (
        set "LLVM_DIR=!_TRY!"
        echo [ok]    LLVM (vcpkg^): !LLVM_DIR!
        goto :llvm_found
    )
)

:: Check standard LLVM install paths
for %%p in (
    "C:\Program Files\LLVM\lib\cmake\llvm"
    "C:\Program Files (x86)\LLVM\lib\cmake\llvm"
) do (
    if exist "%%~p\LLVMConfig.cmake" (
        set "LLVM_DIR=%%~p"
        echo [ok]    LLVM (system^): !LLVM_DIR!
        goto :llvm_found
    )
)

:: Check Chocolatey
if defined ChocolateyInstall (
    for /d %%d in ("!ChocolateyInstall!\lib\llvm*") do (
        set "_TRY=%%d\lib\cmake\llvm"
        if exist "!_TRY!\LLVMConfig.cmake" (
            set "LLVM_DIR=!_TRY!"
            echo [ok]    LLVM (Chocolatey^): !LLVM_DIR!
            goto :llvm_found
        )
    )
)

:: Check Scoop
if exist "%USERPROFILE%\scoop\apps\llvm\current\lib\cmake\llvm\LLVMConfig.cmake" (
    set "LLVM_DIR=%USERPROFILE%\scoop\apps\llvm\current\lib\cmake\llvm"
    echo [ok]    LLVM (Scoop^): !LLVM_DIR!
    goto :llvm_found
)

echo.
echo [warn]  LLVM development libraries not found.
echo.
echo   You can install LLVM using one of:
echo     choco install llvm           (Chocolatey)
echo     scoop install llvm           (Scoop)
echo     winget install LLVM.LLVM     (winget)
echo     Download from https://github.com/llvm/llvm-project/releases
echo.
echo   Then re-run:  install.bat --llvm-dir="C:\Program Files\LLVM\lib\cmake\llvm"
echo.

set /p "_INSTALL=Install LLVM via winget now? [y/N] "
if /i "!_INSTALL!"=="y" (
    echo [info]  Running: winget install LLVM.LLVM
    winget install LLVM.LLVM
    if exist "C:\Program Files\LLVM\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=C:\Program Files\LLVM\lib\cmake\llvm"
        echo [ok]    LLVM installed: !LLVM_DIR!
        goto :llvm_found
    ) else (
        echo [error] LLVM installed but cmake dir not found. Re-run with --llvm-dir=...
        exit /b 1
    )
) else (
    echo [error] LLVM is required to build SafeC.
    exit /b 1
)

:llvm_found

:: ── Build compiler ──────────────────────────────────────────────────────────
echo.
echo [info]  Building SafeC compiler...

set "COMPILER_DIR=%SCRIPT_DIR%\compiler"
if not exist "%COMPILER_DIR%\CMakeLists.txt" (
    echo [error] compiler\CMakeLists.txt not found at %COMPILER_DIR%
    exit /b 1
)

pushd "%COMPILER_DIR%"

cmake -S . -B build !GENERATOR! -DLLVM_DIR="!LLVM_DIR!"
if !errorlevel! neq 0 (
    echo [error] CMake configure failed.
    popd
    exit /b 1
)

cmake --build build --config Release -j !JOBS!
if !errorlevel! neq 0 (
    echo [error] Compiler build failed.
    popd
    exit /b 1
)

:: Find binary — check multiple locations (multi-config generators)
set "SAFEC_BIN="
if exist "build\safec.exe" set "SAFEC_BIN=%COMPILER_DIR%\build\safec.exe"
if exist "build\Release\safec.exe" set "SAFEC_BIN=%COMPILER_DIR%\build\Release\safec.exe"
if exist "build\Debug\safec.exe" (
    if "!SAFEC_BIN!"=="" set "SAFEC_BIN=%COMPILER_DIR%\build\Debug\safec.exe"
)

if "!SAFEC_BIN!"=="" (
    echo [error] Compiler build appeared to succeed but safec.exe not found.
    popd
    exit /b 1
)

echo [ok]    Compiler built: !SAFEC_BIN!
popd

:: ── Build safeguard ──────────────────────────────────────────────────────────
set "SAFEGUARD_BIN="
set "SAFEGUARD_DIR=%SCRIPT_DIR%\safeguard"

if !SKIP_SAFEGUARD!==1 (
    echo [info]  Skipping safeguard build (--skip-safeguard^)
    goto :safeguard_done
)
if not exist "%SAFEGUARD_DIR%\CMakeLists.txt" (
    echo [warn]  safeguard\ directory not found — skipping.
    goto :safeguard_done
)

echo.
echo [info]  Building safeguard package manager...
echo [warn]  Note: safeguard uses POSIX APIs and may not compile with MSVC.

pushd "%SAFEGUARD_DIR%"
cmake -S . -B build !GENERATOR! 2>nul
if !errorlevel! neq 0 (
    echo [warn]  Safeguard cmake configure failed (non-fatal^).
    echo [warn]  Safeguard uses POSIX APIs (sys/wait.h^) — try WSL or --skip-safeguard.
    popd
    goto :safeguard_done
)
cmake --build build --config Release -j !JOBS! 2>nul
if !errorlevel! neq 0 (
    echo [warn]  Safeguard build failed (non-fatal^).
    echo [warn]  Safeguard uses POSIX APIs — consider using WSL or --skip-safeguard.
    popd
    goto :safeguard_done
)

if exist "build\safeguard.exe" set "SAFEGUARD_BIN=%SAFEGUARD_DIR%\build\safeguard.exe"
if exist "build\Release\safeguard.exe" set "SAFEGUARD_BIN=%SAFEGUARD_DIR%\build\Release\safeguard.exe"

if not "!SAFEGUARD_BIN!"=="" (
    echo [ok]    Safeguard built: !SAFEGUARD_BIN!
) else (
    echo [warn]  Safeguard build succeeded but binary not found.
)
popd

:safeguard_done

:: ── Environment setup ────────────────────────────────────────────────────────
if !SKIP_ENV!==1 (
    echo [info]  Skipping environment setup (--skip-env^)
    goto :env_done
)

echo.
echo [info]  Configuring environment variables...

:: Get the directory containing safec.exe
for %%F in ("!SAFEC_BIN!") do set "SAFEC_BIN_DIR=%%~dpF"
:: Remove trailing backslash
if "!SAFEC_BIN_DIR:~-1!"=="\" set "SAFEC_BIN_DIR=!SAFEC_BIN_DIR:~0,-1!"

:: Check if SAFEC_HOME already set
set "_CURRENT_HOME="
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v SAFEC_HOME 2^>nul') do set "_CURRENT_HOME=%%b"
if not "!_CURRENT_HOME!"=="" (
    if "!_CURRENT_HOME!"=="%SCRIPT_DIR%" (
        echo [ok]    SAFEC_HOME already set correctly.
        goto :check_path
    )
)

echo.
echo   SAFEC_HOME = %SCRIPT_DIR%
echo   PATH += !SAFEC_BIN_DIR!

:: Warn about setx 1024-char limit
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "_USER_PATH=%%b"
set "_PATH_LEN=0"
if defined _USER_PATH (
    set "_TMP=!_USER_PATH!"
    :count_loop
    if not "!_TMP!"=="" (
        set "_TMP=!_TMP:~1!"
        set /a "_PATH_LEN+=1"
        goto :count_loop
    )
)
if !_PATH_LEN! gtr 900 (
    echo.
    echo [warn]  Your user PATH is !_PATH_LEN! chars. setx has a 1024-char limit.
    echo [warn]  Consider using the PowerShell installer (install.ps1^) which has no limit.
)

echo.
set /p "_APPLY=Apply environment changes? [Y/n] "
if /i "!_APPLY!"=="n" (
    echo [warn]  Skipped. Set SAFEC_HOME and PATH manually.
    goto :env_done
)

setx SAFEC_HOME "%SCRIPT_DIR%" >nul 2>&1
if !errorlevel! neq 0 (
    echo [warn]  Failed to set SAFEC_HOME via setx.
)

:: Set for current session too
set "SAFEC_HOME=%SCRIPT_DIR%"

:check_path
:: Check if safec bin dir is already in PATH
echo !PATH! | findstr /i /c:"!SAFEC_BIN_DIR!" >nul 2>&1
if !errorlevel!==0 (
    echo [ok]    PATH already contains !SAFEC_BIN_DIR!
) else (
    :: Read current user PATH from registry and append
    set "_NEW_PATH="
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "_NEW_PATH=%%b"
    if "!_NEW_PATH!"=="" (
        setx PATH "!SAFEC_BIN_DIR!" >nul 2>&1
    ) else (
        setx PATH "!_NEW_PATH!;!SAFEC_BIN_DIR!" >nul 2>&1
    )
    :: Update current session
    set "PATH=!PATH!;!SAFEC_BIN_DIR!"
    echo [ok]    Added !SAFEC_BIN_DIR! to user PATH.
)

:: Add safeguard to PATH if built and in different dir
if not "!SAFEGUARD_BIN!"=="" (
    for %%F in ("!SAFEGUARD_BIN!") do set "_SG_DIR=%%~dpF"
    if "!_SG_DIR:~-1!"=="\" set "_SG_DIR=!_SG_DIR:~0,-1!"
    if not "!_SG_DIR!"=="!SAFEC_BIN_DIR!" (
        echo !PATH! | findstr /i /c:"!_SG_DIR!" >nul 2>&1
        if !errorlevel! neq 0 (
            for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "_CUR=%%b"
            setx PATH "!_CUR!;!_SG_DIR!" >nul 2>&1
            set "PATH=!PATH!;!_SG_DIR!"
            echo [ok]    Added !_SG_DIR! to user PATH.
        )
    )
)

echo [ok]    Environment configured. Restart your terminal to apply.

:env_done

:: ── Verification ─────────────────────────────────────────────────────────────
echo.
echo [info]  Verifying installation...

"!SAFEC_BIN!" --help >nul 2>&1
if !errorlevel!==0 (
    echo [ok]    safec --help works
) else (
    echo [warn]  safec --help returned non-zero
)

:: Compile hello.sc
set "HELLO_SC=%SCRIPT_DIR%\compiler\examples\hello.sc"
if exist "!HELLO_SC!" (
    set "TMPLL=%TEMP%\safec_test_%RANDOM%.ll"
    "!SAFEC_BIN!" "!HELLO_SC!" --emit-llvm -o "!TMPLL!" >nul 2>&1
    if !errorlevel!==0 (
        echo [ok]    Compiled examples\hello.sc to LLVM IR
    ) else (
        echo [warn]  Failed to compile examples\hello.sc (non-fatal^)
    )
    del "!TMPLL!" >nul 2>&1
) else (
    echo [warn]  examples\hello.sc not found — skipping compile test
)

:: Count std files
set "STD_DIR=%SCRIPT_DIR%\std"
if exist "!STD_DIR!" (
    set "_COUNT=0"
    for /r "!STD_DIR!" %%f in (*.h *.sc) do set /a "_COUNT+=1"
    echo [ok]    Standard library: !_COUNT! files in std\
) else (
    echo [warn]  std\ directory not found
)

if not "!SAFEGUARD_BIN!"=="" (
    "!SAFEGUARD_BIN!" help >nul 2>&1
    if !errorlevel!==0 (
        echo [ok]    safeguard works
    ) else (
        echo [warn]  safeguard returned non-zero
    )
)

:: ── Summary ──────────────────────────────────────────────────────────────────
echo.
echo  ════════════════════════════════════════════════════════════════
echo    SafeC installation complete!
echo  ════════════════════════════════════════════════════════════════
echo.
echo   SAFEC_HOME   %SCRIPT_DIR%
echo   safec        !SAFEC_BIN!
if not "!SAFEGUARD_BIN!"=="" echo   safeguard    !SAFEGUARD_BIN!
echo.
echo   Quick start:
echo     safec examples\hello.sc --emit-llvm -o hello.ll
echo     clang hello.ll -o hello.exe
echo     hello.exe
echo.
if not !SKIP_ENV!==1 (
    echo   Restart your terminal to pick up PATH changes.
    echo.
)

endlocal
exit /b 0

:: ── Help ─────────────────────────────────────────────────────────────────────
:show_help
echo.
echo  SafeC Install Script — Windows (cmd.exe)
echo.
echo  Usage:  install.bat [OPTIONS]
echo.
echo  Options:
echo    --llvm-dir=^<path^>   Path to LLVM cmake directory (auto-detected if omitted)
echo    --jobs=N            Parallel build jobs (default: %%NUMBER_OF_PROCESSORS%%^)
echo    --skip-safeguard    Skip building the safeguard package manager
echo    --skip-env          Skip environment variable configuration
echo    --ninja             Use Ninja generator instead of default
echo    --help              Show this help message
echo.
echo  Examples:
echo    install.bat
echo    install.bat --llvm-dir="C:\Program Files\LLVM\lib\cmake\llvm" --jobs=8
echo    install.bat --skip-safeguard --ninja
echo.
exit /b 0
