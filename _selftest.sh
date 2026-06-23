#!/bin/bash
source /home/adam/prenatal_morpho/1/cuda_adam/wsl_env.sh
echo "PROJ=[$PROJ]"
echo "PY=[$PY]"
"$PY" --version
"$NVCC" --version | tail -1
