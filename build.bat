@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

:: ── Tool paths ──────────────────────────────────────────────────────────────
set MSVC_BIN=E:\VIS STUD\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
set SDK_BIN=E:\Windows Kits\10\bin\10.0.26100.0\x64
set SDK_LIB_UM=E:\Windows Kits\10\Lib\10.0.26100.0\um\x64
set SDK_LIB_UCRT=E:\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64
set MSVC_LIB=E:\VIS STUD\VC\Tools\MSVC\14.51.36231\lib\x64
set MSVC_INC=E:\VIS STUD\VC\Tools\MSVC\14.51.36231\include
set SDK_INC_UCRT=E:\Windows Kits\10\Include\10.0.26100.0\ucrt
set SDK_INC_UM=E:\Windows Kits\10\Include\10.0.26100.0\um
set SDK_INC_SHARED=E:\Windows Kits\10\Include\10.0.26100.0\shared

set PATH=%MSVC_BIN%;%SDK_BIN%;%PATH%
set LIB=%SDK_LIB_UM%;%SDK_LIB_UCRT%;%MSVC_LIB%
set INCLUDE=%MSVC_INC%;%SDK_INC_UCRT%;%SDK_INC_UM%;%SDK_INC_SHARED%

echo === CUDA Adam — Windows Build ===
echo.

:: ── Step 1: Generate noise ──────────────────────────────────────────────────
echo === Step 1: Generate noise ===
cd /d "%SCRIPT_DIR%\reference"
python gen_noise.py
if errorlevel 1 goto :error

:: ── Step 2: Brian2 reference (optional) ─────────────────────────────────────
echo.
echo === Step 2: Brian2 reference ===
python -c "import brian2" 2>nul
if not errorlevel 1 (
    python run_brian2_ref.py
    if errorlevel 1 goto :error
) else (
    echo   [SKIP] brian2 not installed - using existing ref files
)

:: ── Step 2b: Connectivity ───────────────────────────────────────────────────
echo.
echo === Step 2b: Generate connectivity ===
python gen_connectivity.py
if errorlevel 1 goto :error

:: ── Step 2c: Brian2 syn reference (optional) ────────────────────────────────
echo.
echo === Step 2c: Brian2 syn reference ===
python -c "import brian2" 2>nul
if not errorlevel 1 (
    python run_brian2_syn_ref.py
    if errorlevel 1 goto :error
) else (
    echo   [SKIP] brian2 not installed
)

:: ── Step 3: Build CUDA ──────────────────────────────────────────────────────
echo.
echo === Step 3: Build CUDA ===
cd /d "%SCRIPT_DIR%\cuda"

if exist build (
    echo Cleaning old build...
    rmdir /s /q build
)
mkdir build
cd build

:: Detect GPU compute capability (e.g. "8.6" -> "86")
set CUDA_ARCH=86
for /f "usebackq tokens=*" %%i in (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul`) do (
    set CAP=%%i
    set CUDA_ARCH=!CAP:.=!
    goto :arch_done
)
:arch_done
echo Using CUDA arch: !CUDA_ARCH!

cmake .. -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=!CUDA_ARCH! ^
  -DCMAKE_CUDA_HOST_COMPILER="E:\VIS STUD\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe" ^
  -DCMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO"
if errorlevel 1 goto :error

cmake --build . --parallel
if errorlevel 1 goto :error

:: ── Step 4: Run simulation ──────────────────────────────────────────────────
echo.
echo === Step 4: Run CUDA simulation ===
cd /d "%SCRIPT_DIR%"
if not exist output mkdir output

cuda\build\sim_runner.exe reference output
if errorlevel 1 goto :error

:: ── Step 5: Python Euler reference ─────────────────────────────────────────
echo.
echo === Step 5: Python Euler reference ===
cd /d "%SCRIPT_DIR%\reference"
python run_python_ref.py
if errorlevel 1 goto :error

:: ── Step 6: Validate ────────────────────────────────────────────────────────
echo.
echo === Step 6: Validate ===
cd /d "%SCRIPT_DIR%"
python validation\compare_traces.py reference output
if errorlevel 1 goto :error

echo.
echo === Done! ===
goto :end

:error
echo.
echo ERROR: step failed
exit /b 1

:end
endlocal
