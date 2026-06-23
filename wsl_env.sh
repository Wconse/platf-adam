#!/bin/bash
# Toolchain for ADAM CUDA project. No conda activate (it breaks sourced scripts here).
# Use absolute binary paths instead.
export PROJ=/home/adam/prenatal_morpho/1
export CUDA_ADAM=$PROJ/cuda_adam
export CONDA_ENV=/home/adam/miniconda3/envs/prenatal_morpho
export PY=$CONDA_ENV/bin/python
export CMAKE=$CONDA_ENV/bin/cmake
export NVCC=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$CONDA_ENV/bin:$PATH
