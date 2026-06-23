@echo off
call "E:\VIS STUD\VC\Auxiliary\Build\vcvarsall.bat" x64 >> E:\Adam\build_log.txt 2>&1
if errorlevel 1 (echo vcvarsall failed >> E:\Adam\build_log.txt & exit /b 1)

cd /d "E:\Adam\Новая папка\cuda_adam\cuda\build"

cmake .. -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 >> E:\Adam\build_log.txt 2>&1
if errorlevel 1 (echo cmake configure failed >> E:\Adam\build_log.txt & exit /b 1)

cmake --build . --parallel >> E:\Adam\build_log.txt 2>&1
if errorlevel 1 (echo cmake build failed >> E:\Adam\build_log.txt & exit /b 1)

echo BUILD OK >> E:\Adam\build_log.txt
