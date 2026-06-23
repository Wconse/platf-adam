#!/bin/bash
# Complete test suite M0–M8 + Brian2 validation.
# Usage: cd cuda_adam && bash tests/run_all.sh
set -e

PROJ=/home/adam/prenatal_morpho/1
CUDA_ADAM=$PROJ/cuda_adam
PY=/home/adam/miniconda3/envs/prenatal_morpho/bin/python
BUILD=$CUDA_ADAM/cuda/build
REF=$CUDA_ADAM/reference
BUNDLE=$REF/multiregion
OUT=/tmp/cuda_adam_test_output

mkdir -p $OUT/m0 $OUT/m1 $OUT/m3 $OUT/m5 $OUT/m6 $OUT/m7

echo "=============================="
echo " Building all CUDA targets"
echo "=============================="
(cd $CUDA_ADAM/cuda/build && make -j4 -s)

echo ""
echo "=============================="
echo " CUDA simulation runs"
echo "=============================="
echo "--- M0 (no synapses) ---"
$BUILD/sim_runner $REF $OUT/m0 --no-synapses

echo "--- M1/M2 (synapses + NMDA) ---"
$BUILD/sim_runner $REF $OUT/m1

echo "--- M3-M5 multi-region ---"
$BUILD/sim_multiregion $BUNDLE $OUT/m5

echo "--- M6 (save state) ---"
$BUILD/sim_multiregion $BUNDLE $OUT/m6 --save-state

echo "--- M7 (timed) ---"
START=$(date +%s)
$BUILD/sim_multiregion $BUNDLE $OUT/m7
END=$(date +%s)
WALL=$((END - START))
echo "Wall time: ${WALL}s"

echo ""
echo "=============================="
echo " Unit tests"
echo "=============================="
echo "--- T2.1-T2.2: M2 unit ---"
$PY $CUDA_ADAM/tests/test_m2_unit.py

echo "--- T1.1-T1.4: M1 micro ---"
$PY $CUDA_ADAM/tests/test_m1_microtests.py

echo "--- T4.1-T4.3: M4 STDP ---"
$PY $CUDA_ADAM/tests/test_m4_stdp.py

echo "--- D1-D7: DevStateVector ---"
$PY $CUDA_ADAM/tests/test_dev_state.py

echo ""
echo "=============================="
echo " Statistical tests"
echo "=============================="
echo "--- T1.5-T1.8: M1 statistical ---"
$PY $CUDA_ADAM/tests/test_m1_statistical.py $OUT/m1 $OUT/m0

echo "--- T2.3-T2.6: M2 statistical ---"
$PY $CUDA_ADAM/tests/test_m2_statistical.py $OUT/m1 $OUT/m0

echo "--- T3.1,T3.3,T3.4,T3.7: M3 integrity ---"
$PY $CUDA_ADAM/tests/test_m3_integrity.py $BUNDLE $OUT/m5 $OUT/m0

echo "--- T5.1,T5.3,T5.4: M5 afferents ---"
$PY $CUDA_ADAM/tests/test_m5_afferents.py $BUNDLE $OUT/m5 $OUT/m5 $OUT/m0

echo "--- T6.1-T6.4: M6 checkpoint ---"
$PY $CUDA_ADAM/tests/test_m6_checkpoint.py $BUNDLE $OUT/m6 $OUT/m0

echo "--- T7.1-T7.5: M7 perf ---"
$PY $CUDA_ADAM/tests/test_m7_perf.py $BUNDLE $OUT/m7 $OUT/m0 $WALL

echo ""
echo "=============================="
echo " Brian2 validation"
echo "=============================="
BRIAN2_JSON=$BUNDLE/brian2_mr_region_rates.json
if [ -f "$BRIAN2_JSON" ]; then
  echo "--- V1-V5: Brian2 comparison ---"
  $PY $CUDA_ADAM/tests/test_validation_vs_brian2.py $BUNDLE $OUT/m5 $BRIAN2_JSON
else
  echo "SKIP: brian2_mr_region_rates.json not found (run run_brian2_multiregion_ref.py first)"
fi

echo ""
echo "=============================="
echo " Slow-loop tests (M8, ~3 min)"
echo "=============================="
mkdir -p /tmp/cuda_m8_ckpt_all
echo "--- M8 basic slow-loop ---"
cd $CUDA_ADAM && $PY tests/test_m8_slowloop.py $BUNDLE $BUILD/sim_multiregion /tmp/cuda_m8_ckpt_all

echo "--- M8 with DevStateVector ---"
mkdir -p /tmp/cuda_m8_devstate_all
cd $CUDA_ADAM && $PY tests/test_m8_with_devstate.py $BUNDLE $BUILD/sim_multiregion /tmp/cuda_m8_devstate_all

echo ""
echo "=============================="
echo " ALL TESTS COMPLETE"
echo "=============================="
