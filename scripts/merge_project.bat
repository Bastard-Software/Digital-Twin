@echo off
setlocal enabledelayedexpansion

REM Resolve project root (one level above this script)
pushd "%~dp0.."
set "ROOT=%CD%"
popd

set "OUTPUT=%ROOT%\merged.txt"

if exist "%OUTPUT%" del "%OUTPUT%"

echo Merging project files into %OUTPUT%...
echo.

for /r "%ROOT%" %%f in (*) do (
    call :ProcessFile "%%f"
)

echo.
echo Done. Output: %OUTPUT%
goto :eof

:ProcessFile
set "FILEPATH=%~1"
set "EXT=%~x1"

REM Skip excluded directories (string substitution avoids backslash/quote issues)
set "CHK=%FILEPATH:\build\=X%"
if not "%CHK%"=="%FILEPATH%" exit /b
set "CHK=%FILEPATH:\external\=X%"
if not "%CHK%"=="%FILEPATH%" exit /b
set "CHK=%FILEPATH:\.git\=X%"
if not "%CHK%"=="%FILEPATH%" exit /b
set "CHK=%FILEPATH:\.claude\=X%"
if not "%CHK%"=="%FILEPATH%" exit /b

REM Skip the output file and this script
set "CHK=%FILEPATH:\merged.txt=X%"
if not "%CHK%"=="%FILEPATH%" exit /b
set "CHK=%FILEPATH:\merge_project.bat=X%"
if not "%CHK%"=="%FILEPATH%" exit /b

REM Skip binary and compiled files
if /i "%EXT%"==".spv"   exit /b
if /i "%EXT%"==".ttf"   exit /b
if /i "%EXT%"==".otf"   exit /b
if /i "%EXT%"==".exe"   exit /b
if /i "%EXT%"==".dll"   exit /b
if /i "%EXT%"==".lib"   exit /b
if /i "%EXT%"==".obj"   exit /b
if /i "%EXT%"==".pdb"   exit /b
if /i "%EXT%"==".ilk"   exit /b
if /i "%EXT%"==".png"   exit /b
if /i "%EXT%"==".jpg"   exit /b
if /i "%EXT%"==".jpeg"  exit /b
if /i "%EXT%"==".ico"   exit /b
if /i "%EXT%"==".bin"   exit /b

echo Including: %FILEPATH%
echo // ============================================================ >> "%OUTPUT%"
echo // %FILEPATH% >> "%OUTPUT%"
echo // ============================================================ >> "%OUTPUT%"
type "%FILEPATH%" >> "%OUTPUT%"
echo. >> "%OUTPUT%"
echo. >> "%OUTPUT%"

exit /b
