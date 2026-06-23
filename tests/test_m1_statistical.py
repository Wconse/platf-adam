"""
M1 statistical tests T1.5 – T1.8.

T1.5: Python-Euler-syn vs CUDA-M1 mean rates ≈ (same noise → near-exact match expected)
T1.6: Per-neuron rate correlation and mean |Δrate|
T1.7: Total spike counts within ±5%
T1.8: M0 regression (no-synapse CUDA rates vs recorded M0 baseline)

Usage:
    python3 tests/test_m1_statistical.py <cuda_m1_dir> <cuda_m0_dir>
"""
import numpy as np
import sys
import os

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

N_E = 80
N_I = 20
T_SIM_S = 10.0  # seconds

# Recorded M0 baselines (from verified M0 run)
M0_E_RATE = 0.330
M0_I_RATE = 8.805
M0_TOL = 0.05  # Hz

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cuda_m1_dir> <cuda_m0_dir>")
        sys.exit(1)

    cuda_m1_dir = sys.argv[1]
    cuda_m0_dir = sys.argv[2]
    ref_dir = os.path.join(os.path.dirname(__file__), '..', 'reference')

    results = []

    # Load all rate arrays
    cuda_m1_rates = np.load(os.path.join(cuda_m1_dir, 'cuda_rates.npy'))
    cuda_m0_rates = np.load(os.path.join(cuda_m0_dir, 'cuda_rates.npy'))
    py_syn_rates  = np.load(os.path.join(ref_dir, 'python_euler_syn_rates.npy'))
    syn_ref_rates = np.load(os.path.join(ref_dir, 'syn_ref_rates.npy'))  # Brian2 (different noise)

    cuda_m1_spikes = np.load(os.path.join(cuda_m1_dir, 'cuda_spikes.npy'))
    py_syn_spikes  = np.load(os.path.join(ref_dir, 'python_euler_syn_spikes.npy'))

    print("=" * 60)
    print("M1 Statistical Tests T1.5 – T1.8")
    print("=" * 60)

    # ── T1.5: Mean rates — CUDA vs Python-Euler (same noise) ──────────
    cuda_mean_e = cuda_m1_rates[:N_E].mean()
    cuda_mean_i = cuda_m1_rates[N_E:].mean()
    py_mean_e   = py_syn_rates[:N_E].mean()
    py_mean_i   = py_syn_rates[N_E:].mean()
    brian_mean_e = syn_ref_rates[:N_E].mean()
    brian_mean_i = syn_ref_rates[N_E:].mean()

    print(f"\nT1.5 — Mean rates (same-noise Euler vs CUDA M1):")
    print(f"       {'Source':>18} {'E rate':>8} {'I rate':>8}")
    print(f"  Python Euler syn:  {py_mean_e:>8.3f}  {py_mean_i:>8.3f} Hz")
    print(f"  CUDA M1:           {cuda_mean_e:>8.3f}  {cuda_mean_i:>8.3f} Hz")
    print(f"  Brian2 ref (Δnoise): {brian_mean_e:>8.3f}  {brian_mean_i:>8.3f} Hz  (different RNG)")

    # Same-noise diff should be tiny (< 0.01 Hz — only float32 accumulation)
    diff_e = abs(cuda_mean_e - py_mean_e)
    diff_i = abs(cuda_mean_i - py_mean_i)
    T15_tol = 0.05  # Hz for same-noise comparison
    ok_15 = (diff_e < T15_tol) and (diff_i < T15_tol)
    print(f"  |Δrate E|={diff_e:.4f} Hz, |Δrate I|={diff_i:.4f} Hz  (tol={T15_tol} Hz)")
    print(f"  [{PASS if ok_15 else FAIL}] T1.5")
    results.append(('T1.5', ok_15))

    # ── T1.6: Per-neuron correlation and mean |Δrate| ─────────────────
    corr_e = np.corrcoef(cuda_m1_rates[:N_E], py_syn_rates[:N_E])[0,1]
    corr_i = np.corrcoef(cuda_m1_rates[N_E:], py_syn_rates[N_E:])[0,1]
    delta_rate = np.abs(cuda_m1_rates - py_syn_rates)
    mean_delta = delta_rate.mean()
    max_delta  = delta_rate.max()

    print(f"\nT1.6 — Per-neuron rates (CUDA vs Python Euler, same noise):")
    print(f"  Pearson corr E: {corr_e:.4f}  (req > 0.99 for same noise)")
    print(f"  Pearson corr I: {corr_i:.4f}")
    print(f"  Mean |Δrate|:   {mean_delta:.4f} Hz  (req < 0.1 Hz)")
    print(f"  Max  |Δrate|:   {max_delta:.4f} Hz  (req < 0.5 Hz)")

    ok_16 = (corr_e > 0.99) and (corr_i > 0.99) and (mean_delta < 0.1) and (max_delta < 0.5)
    print(f"  [{PASS if ok_16 else FAIL}] T1.6")
    results.append(('T1.6', ok_16))

    # ── T1.7: Total spike counts within ±5% ──────────────────────────
    # CUDA vs Python Euler syn (same noise)
    def count_spikes(spikes_arr):
        if spikes_arr.shape[1] == 0:
            return 0, 0
        idx = spikes_arr[0].astype(int)
        se = int((idx < N_E).sum())
        si = int((idx >= N_E).sum())
        return se, si

    cuda_se, cuda_si = count_spikes(cuda_m1_spikes)
    py_se, py_si     = count_spikes(py_syn_spikes)

    print(f"\nT1.7 — Spike counts (CUDA vs Python Euler, same noise):")
    print(f"  CUDA:   E={cuda_se}, I={cuda_si}, total={cuda_se+cuda_si}")
    print(f"  Py-Euler: E={py_se}, I={py_si}, total={py_se+py_si}")

    def within_5pct(a, b):
        if b == 0: return a == 0
        return abs(a - b) / b <= 0.05

    ok_17_e = within_5pct(cuda_se, py_se)
    ok_17_i = within_5pct(cuda_si, py_si)
    ok_17 = ok_17_e and ok_17_i

    print(f"  E diff: {abs(cuda_se-py_se)} spikes ({abs(cuda_se-py_se)/max(py_se,1)*100:.1f}%)")
    print(f"  I diff: {abs(cuda_si-py_si)} spikes ({abs(cuda_si-py_si)/max(py_si,1)*100:.1f}%)")
    print(f"  [{PASS if ok_17 else FAIL}] T1.7")
    results.append(('T1.7', ok_17))

    # ── T1.8: M0 regression ──────────────────────────────────────────
    m0_e = cuda_m0_rates[:N_E].mean()
    m0_i = cuda_m0_rates[N_E:].mean()
    ok_18_e = abs(m0_e - M0_E_RATE) < M0_TOL
    ok_18_i = abs(m0_i - M0_I_RATE) < M0_TOL
    ok_18 = ok_18_e and ok_18_i

    print(f"\nT1.8 — M0 regression (no-synapse CUDA):")
    print(f"  CUDA M0 E rate: {m0_e:.3f} Hz  (expected {M0_E_RATE}±{M0_TOL})")
    print(f"  CUDA M0 I rate: {m0_i:.3f} Hz  (expected {M0_I_RATE}±{M0_TOL})")
    print(f"  [{PASS if ok_18 else FAIL}] T1.8")
    results.append(('T1.8', ok_18))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"  SUMMARY: {n_pass}/{n_total} passed")
    for name, ok in results:
        print(f"    [{PASS if ok else FAIL}] {name}")

    if n_pass == n_total:
        print("\n  ALL STATISTICAL TESTS PASSED")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
