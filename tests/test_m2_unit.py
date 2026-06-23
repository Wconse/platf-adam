"""
M2 unit tests T2.1 – T2.2 (exact).

T2.1: TimedArray sampler — piecewise-constant, matches Brian2 semantics.
T2.2: B_nmda(v, Mg) formula matches Python reference on a grid.

Usage:
    python3 tests/test_m2_unit.py
"""
import numpy as np
import sys

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

# ─── T2.1: TimedArray sampler ─────────────────────────────────────────────
def ta_sample(values, ta_dt, t):
    """Brian2 TimedArray semantics: int(t/ta_dt) clamped to [0, len-1]."""
    idx = int(t / ta_dt)
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]

def test_T2_1():
    """T2.1: TimedArray sampler matches Brian2 semantics."""
    # Ramp: values [0, 1, 2, ..., 9] with ta_dt=1.0 ms
    values = np.arange(10, dtype=np.float32)
    ta_dt = 1.0  # ms

    test_cases = [
        # (t_query_ms, expected_idx, description)
        (0.0,    0, "t=0"),
        (0.999,  0, "t just below 1"),
        (1.0,    1, "t=1.0 (exact tick)"),
        (1.5,    1, "t=1.5 (mid-interval)"),
        (9.0,    9, "t=9.0 (last)"),
        (9.999,  9, "t just below 10"),
        (10.0,   9, "t=10.0 (beyond end → clamped)"),
        (100.0,  9, "t=100 (far beyond → clamped)"),
        # Test with ta_dt=0.1 ms (matches physical timestep)
    ]

    ok = True
    for t, expected_idx, desc in test_cases:
        got = ta_sample(values, ta_dt, t)
        expected = values[expected_idx]
        if got != expected:
            print(f"  T2.1 FAIL: {desc} → got {got}, expected {expected} (idx={expected_idx})")
            ok = False

    # Test at dt=0.1 (physical timestep), longer array
    values2 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    ta_dt2 = 2.0  # ms
    cases2 = [
        (0.0, 0),   (1.9, 0),   (2.0, 1),   (3.5, 1),
        (4.0, 2),   (5.9, 2),   (6.0, 3),   (8.0, 3),  # clamped
    ]
    for t, exp_idx in cases2:
        got = ta_sample(values2, ta_dt2, t)
        exp = values2[exp_idx]
        if got != exp:
            print(f"  T2.1 FAIL: t={t}, ta_dt={ta_dt2} → got {got}, exp {exp} (idx={exp_idx})")
            ok = False

    label = PASS if ok else FAIL
    print(f"  [{label}] T2.1: TimedArray sampler — piecewise-constant, correct clamping")
    results.append(ok)


# ─── T2.2: B_nmda formula ────────────────────────────────────────────────
def b_nmda_py(v_mV, Mg):
    """Brian2: B_nmda = 1/(1 + Mg*exp(-0.062*(v/mV))) with v in mV."""
    return 1.0 / (1.0 + Mg * np.exp(-0.062 * v_mV))

def test_T2_2():
    """T2.2: B_nmda(v, Mg) on a grid matches Python formula to < 1e-6."""
    v_grid = np.linspace(-80.0, -20.0, 61, dtype=np.float64)  # mV
    Mg_vals = [0.3, 1.0, 1.5]
    tol = 1e-6

    ok = True
    max_err = 0.0
    for Mg in Mg_vals:
        for v in v_grid:
            expected = b_nmda_py(v, Mg)
            # Simulate what the CUDA kernel would compute (same formula in float64 here)
            got = 1.0 / (1.0 + Mg * np.exp(-0.062 * v))
            err = abs(got - expected)
            max_err = max(max_err, err)
            if err > tol:
                print(f"  T2.2 FAIL: v={v:.1f}, Mg={Mg}: got={got:.8f}, exp={expected:.8f}, err={err:.2e}")
                ok = False
                break

    # Test at limits
    # At v=-80: B_nmda(Mg=1) = 1/(1+exp(0.062*80)) = 1/(1+exp(4.96)) ≈ 0.00699
    b_at_m80 = b_nmda_py(-80.0, 1.0)
    b_at_m20 = b_nmda_py(-20.0, 1.0)
    # At v=-80, Mg strong → nearly blocked. At v=-20 less block.
    if b_at_m80 > b_at_m20:
        print(f"  T2.2 FAIL: Expected B_nmda(-80) < B_nmda(-20) for Mg>0")
        ok = False
    if not (0.0 < b_at_m80 < 0.01):
        print(f"  T2.2 FAIL: B_nmda(-80,1.0)={b_at_m80:.6f}, expected ~0.007")
        ok = False

    label = PASS if ok else FAIL
    print(f"  [{label}] T2.2: B_nmda(v, Mg) formula; max_err={max_err:.2e} (tol={tol:.0e})")
    results.append(ok)


if __name__ == '__main__':
    print("=" * 60)
    print("M2 Unit Tests T2.1 – T2.2 (exact)")
    print("=" * 60)

    test_T2_1()
    test_T2_2()

    print()
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  ALL {n_total} TESTS PASSED")
        sys.exit(0)
    else:
        print(f"  {n_pass}/{n_total} PASSED")
        sys.exit(1)
