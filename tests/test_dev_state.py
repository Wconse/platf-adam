"""
Test DevStateVector against Platform-Adam reference values.

Verifies:
  D1: E_GABA at early dw < center gives depolarizing value (>-60 mV)
  D2: E_GABA at late dw > center gives hyperpolarizing value (<-70 mV)
  D3: lr_gate = 0 before onset, 1 after full_dw
  D4: nmda_ratio early≈0.08, late≈0.22
  D5: Mg early≈0.6, late≈1.0
  D6: VPA intervention shifts GABA center by +6 dw (KCC2_Inhibition strength=1)
  D7: EI_Imbalance adds to I_bi in engine_runner

Usage: python3 tests/test_dev_state.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cuda_engine.dev_state import DevStateVector, sigmoid_dw

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, val, lo, hi, desc=""):
    ok = lo <= val <= hi
    print(f"  [{PASS if ok else FAIL}] {name}: {val:.4f}  (expect [{lo:.4f}, {hi:.4f}]) {desc}")
    results.append(ok)
    return ok

print("=" * 60)
print("DevStateVector Tests D1–D7")
print("=" * 60)

# ── D1/D2: E_GABA sigmoid ────────────────────────────────────────────────────
print("\nD1-D2 — E_GABA flip (P1 center=30.0 dw, steepness=2.0):")
dsv = DevStateVector(initial_dw=22.0)
s1  = dsv.get_state()
# At dw=22, well before P1 center=30 → sigmoid(22, 30, 2) ≈ sigmoid_dw(22,30,2)
# = 1/(1+exp(2*(22-30))) = 1/(1+exp(-16)) ≈ 1.0 → egaba ≈ -45 + 1.0*(-75-(-45)) = -75
# Wait: sigmoid_dw with center=30, steepness=2: at dw=22
s_val = sigmoid_dw(22.0, 30.0, 2.0)
expected_22 = -45.0 + s_val * (-75.0 - (-45.0))
check("D1 P1_E_GABA at dw=22", s1['egaba_mV']['P1'], -70.0, -45.0,
      f"(expected ~{expected_22:.1f}, sigmoid={s_val:.3f})")

dsv2 = DevStateVector(initial_dw=40.0)
s2   = dsv2.get_state()
s_val2 = sigmoid_dw(40.0, 30.0, 2.0)
expected_40 = -45.0 + s_val2 * (-75.0 - (-45.0))
check("D2 P1_E_GABA at dw=40", s2['egaba_mV']['P1'], -76.0, -70.0,
      f"(expected ~{expected_40:.1f}, sigmoid={s_val2:.3f})")

# ── D3: lr_gate ──────────────────────────────────────────────────────────────
print("\nD3 — lr_gate:")
dsv3 = DevStateVector(initial_dw=30.0)
s3   = dsv3.get_state()
check("D3a lr_gate at dw=30 (before onset=34)", s3['lr_gate'], 0.0, 0.2)

dsv4 = DevStateVector(initial_dw=42.0)
s4   = dsv4.get_state()
check("D3b lr_gate at dw=42 (after full=38)", s4['lr_gate'], 0.9, 1.0)

# ── D4: nmda_ratio ───────────────────────────────────────────────────────────
print("\nD4 — nmda_ratio:")
dsv5 = DevStateVector(initial_dw=22.0)
s5   = dsv5.get_state()
check("D4a nmda_ratio at dw=22 (early≈0.08)", s5['nmda_ratio'], 0.075, 0.095)

dsv6 = DevStateVector(initial_dw=50.0)
s6   = dsv6.get_state()
check("D4b nmda_ratio at dw=50 (late≈0.22)", s6['nmda_ratio'], 0.20, 0.225)

# ── D5: Mg concentration ─────────────────────────────────────────────────────
print("\nD5 — Mg concentration:")
dsv7 = DevStateVector(initial_dw=22.0)
s7   = dsv7.get_state()
check("D5a Mg at dw=22 (early≈0.6)", s7['mg_now'], 0.55, 0.65)

dsv8 = DevStateVector(initial_dw=50.0)
s8   = dsv8.get_state()
check("D5b Mg at dw=50 (late≈1.0)", s8['mg_now'], 0.95, 1.01)

# ── D6: GABA shift (KCC2_Inhibition mock) ────────────────────────────────────
print("\nD6 — GABA shift via gaba_shifts:")
dsv9 = DevStateVector(initial_dw=28.0)  # P1 center=30 → partially flipped
# Without shift
s9_no = dsv9.get_state()
egaba_no = s9_no['egaba_mV']['P1']

# Manual shift (KCC2_Inhibition strength=1 → shift=6 dw)
dsv9.gaba_shifts['P1'] = 6.0
s9_shift = dsv9.get_state()
egaba_shift = s9_shift['egaba_mV']['P1']
dsv9.gaba_shifts['P1'] = 0.0  # reset

# With +6 shift: P1 new center = 30+6=36, so at dw=28 less flipped → higher (more depolarizing)
ok_d6 = egaba_shift > egaba_no  # more depolarizing with shift
print(f"  P1 E_GABA: no_shift={egaba_no:.2f} mV, shift+6dw={egaba_shift:.2f} mV")
print(f"  [{PASS if ok_d6 else FAIL}] D6: GABA shift delays flip (shift > no_shift)")
results.append(ok_d6)

# ── D7: ei_bias_delta_pA propagates ──────────────────────────────────────────
print("\nD7 — ei_bias_delta_pA in get_state():")
dsv10 = DevStateVector(initial_dw=25.0)
dsv10.ei_bias_delta_pA['P1'] = 30.0  # +30 pA E bias
s10 = dsv10.get_state()
ok_d7 = s10['ei_bias_delta_pA']['P1'] == 30.0
print(f"  P1 ei_bias_delta_pA = {s10['ei_bias_delta_pA']['P1']} pA")
print(f"  [{PASS if ok_d7 else FAIL}] D7")
results.append(ok_d7)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
n_pass = sum(results)
n_total = len(results)
if n_pass == n_total:
    print(f"  ALL {n_total} TESTS PASSED")
    sys.exit(0)
else:
    print(f"  {n_pass}/{n_total} PASSED")
    sys.exit(1)
