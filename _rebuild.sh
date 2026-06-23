#!/usr/bin/env bash
# Incremental rebuild of the CUDA engine (Ninja recompiles only changed .cu files).
# Paths per BUILD_WINDOWS.md (non-standard, junctioned to D:/E:).
set -e
MSVC_BIN="/e/VIS STUD/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64"
SDK_BIN="/e/Windows Kits/10/bin/10.0.26100.0/x64"
SDK_LIB_UM="E:/Windows Kits/10/Lib/10.0.26100.0/um/x64"
SDK_LIB_UCRT="E:/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64"
MSVC_LIB="E:/VIS STUD/VC/Tools/MSVC/14.51.36231/lib/x64"
cd "$(dirname "$0")/cuda/build"
PATH="$MSVC_BIN:$SDK_BIN:$PATH" \
LIB="$SDK_LIB_UM;$SDK_LIB_UCRT;$MSVC_LIB" \
cmake --build . --parallel
