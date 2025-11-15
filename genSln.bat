@echo off
setlocal

echo ========================================
echo    Digital-Twin Build System
echo ========================================

set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"
set "SCRIPTS_DIR=%ROOT_DIR%scripts"

echo Root: %ROOT_DIR%
echo Build: %BUILD_DIR%

REM Step 1: Fetch dependencies
echo.
echo [1/2] Fetching dependencies...
python "%SCRIPTS_DIR%\fetch_dependencies.py"
if errorlevel 1 (
    echo ERROR: Failed to fetch dependencies
    echo.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

REM Step 2: Build with VS2022
echo.
echo [2/2] Building with Visual Studio 2022...

REM Create build directory if it doesn't exist
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
    echo Created build directory
)

cd /d "%BUILD_DIR%"

echo Generating Visual Studio 2022 project...
cmake -G "Visual Studio 17 2022" -A x64 "%ROOT_DIR%"
if errorlevel 1 (
    echo.
    echo ERROR: CMake configuration failed
    echo.
    echo Troubleshooting:
    echo - Make sure Visual Studio 2022 is installed
    echo - Make sure CMake is in PATH
    echo - Try running: python scripts\fetch_dependencies.py --clean
    pause
    exit /b 1
)

echo.
echo ========================================
echo    SOLUTION GENERATED SUCCESSFULLY!
echo ========================================
echo.
pause