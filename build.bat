@echo off
echo Building Digital-Twin for Visual Studio...

mkdir build
cd build

echo Generating Visual Studio project...
cmake .. -G "Visual Studio 17 2022" -A x64

echo.
echo Project generated in build/ folder
echo You can now open Digital-Twin.sln in Visual Studio
echo.
pause