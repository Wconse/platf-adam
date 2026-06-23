#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Generate noise ==="
cd reference
python3 gen_noise.py

echo ""
echo "=== Step 2: Generate Brian2 reference ==="
python3 run_brian2_ref.py

echo ""
echo "=== Step 2b: Generate connectivity ==="
cd "$SCRIPT_DIR/reference"
python3 gen_connectivity.py

echo ""
echo "=== Step 2c: Brian2 reference WITH synapses ==="
python3 run_brian2_syn_ref.py

echo ""
echo "=== Step 3: Build CUDA ==="
cd "$SCRIPT_DIR/cuda"
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')" 2>/dev/null || cmake ..
make -j$(nproc)

echo ""
echo "=== Step 4: Run CUDA simulation ==="
mkdir -p "$SCRIPT_DIR/output"
./sim_runner "$SCRIPT_DIR/reference" "$SCRIPT_DIR/output"

echo ""
echo "=== Step 5: Python Euler reference (same noise as CUDA) ==="
cd "$SCRIPT_DIR/reference"
python3 run_python_ref.py

echo ""
echo "=== Step 6: Validate ==="
cd "$SCRIPT_DIR"
python3 validation/compare_traces.py reference output

echo ""
echo "=== Done! ==="
