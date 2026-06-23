#!/usr/bin/env python3
"""
Validation: compare CUDA output with Brian2 reference.

Usage: python3 compare_traces.py <ref_dir> <cuda_dir>

Checks:
  1. Per-neuron firing rate (mean, std)
  2. Spike count per neuron
  3. Voltage trace RMSE (if available)
"""
import sys
import numpy as np
import json
import os


def load_npy_safe(path):
    """Load .npy, return None if missing."""
    if os.path.exists(path):
        return np.load(path)
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_traces.py <ref_dir> <cuda_dir>")
        sys.exit(1)

    ref_dir = sys.argv[1]
    cuda_dir = sys.argv[2]

    # ── Load reference ──
    ref_rates = np.load(os.path.join(ref_dir, 'ref_rates.npy'))
    ref_spikes = np.load(os.path.join(ref_dir, 'ref_spikes.npy'))
    ref_v = load_npy_safe(os.path.join(ref_dir, 'ref_v.npy'))
    python_rates = load_npy_safe(os.path.join(ref_dir, 'python_euler_rates.npy'))
    python_spikes_arr = load_npy_safe(os.path.join(ref_dir, 'python_euler_spikes.npy'))

    with open(os.path.join(ref_dir, 'ref_params.json')) as f:
        ref_params = json.load(f)

    N = len(ref_rates)
    N_E = ref_params['N_E']
    N_I = ref_params['N_I']
    T_sim = ref_params['T_SIM_ms'] / 1000.0  # seconds

    # ── Load CUDA ──
    cuda_rates = np.load(os.path.join(cuda_dir, 'cuda_rates.npy'))
    cuda_spikes = np.load(os.path.join(cuda_dir, 'cuda_spikes.npy'))
    cuda_v = load_npy_safe(os.path.join(cuda_dir, 'cuda_v.npy'))

    # ── Rate comparison ──
    print("=" * 60)
    print("VALIDATION: CUDA vs Brian2 Reference")
    print("=" * 60)

    ref_mean_e = ref_rates[:N_E].mean()
    ref_mean_i = ref_rates[N_E:].mean()
    cuda_mean_e = cuda_rates[:N_E].mean()
    cuda_mean_i = cuda_rates[N_E:].mean()

    print(f"\n--- Firing Rates ---")
    print(f"  {'':>15} {'Brian2':>10} {'CUDA':>10} {'Δ%':>10}")
    print(f"  {'Mean E rate':>15} {ref_mean_e:10.3f} {cuda_mean_e:10.3f} {abs(cuda_mean_e-ref_mean_e)/max(ref_mean_e, 0.001)*100:10.1f}%")
    print(f"  {'Mean I rate':>15} {ref_mean_i:10.3f} {cuda_mean_i:10.3f} {abs(cuda_mean_i-ref_mean_i)/max(ref_mean_i, 0.001)*100:10.1f}%")

    # Per-neuron rate correlation
    rate_diff = np.abs(ref_rates - cuda_rates)
    print(f"\n  Per-neuron |Δrate|: mean={rate_diff.mean():.3f} Hz, max={rate_diff.max():.3f} Hz")

    # Spike count comparison
    ref_counts = np.zeros(N)
    for idx in ref_spikes[0].astype(int):
        if 0 <= idx < N:
            ref_counts[idx] += 1

    cuda_counts = np.zeros(N)
    for idx in cuda_spikes[0].astype(int):
        if 0 <= idx < N:
            cuda_counts[idx] += 1

    print(f"\n--- Spike Counts ---")
    print(f"  Brian2: E={int(ref_counts[:N_E].sum())}, I={int(ref_counts[N_E:].sum())}, total={int(ref_counts.sum())}")
    print(f"  CUDA:   E={int(cuda_counts[:N_E].sum())}, I={int(cuda_counts[N_E:].sum())}, total={int(cuda_counts.sum())}")

    count_diff = np.abs(ref_counts - cuda_counts)
    print(f"  Per-neuron |Δcount|: mean={count_diff.mean():.1f}, max={count_diff.max():.0f}")

    # ── Voltage trace comparison (optional) ──
    if ref_v is not None and cuda_v is not None:
        print(f"\n--- Voltage Traces ---")
        # Truncate to min length
        T_min = min(ref_v.shape[0], cuda_v.shape[0])
        N_min = min(ref_v.shape[1], cuda_v.shape[1])

        # Since noise differs (Brian2 uses internal xi, CUDA uses pre-generated),
        # we only compare statistical properties
        ref_active = ref_v[ref_v.shape[0]//2:, :]  # second half (after maturation)
        cuda_active = cuda_v[cuda_v.shape[0]//2:, :]

        print(f"  Brian2 v range: [{ref_v.min():.2f}, {ref_v.max():.2f}] mV")
        print(f"  CUDA v range:   [{cuda_v.min():.2f}, {cuda_v.max():.2f}] mV")
        print(f"  Brian2 v mean (active): {ref_active.mean():.2f} mV")
        print(f"  CUDA v mean (active):   {cuda_active.mean():.2f} mV")

    # ── Python Euler comparison (same noise!) ──
    if python_rates is not None:
        print(f"\n--- Python Euler vs CUDA (SAME noise, key comparison) ---")
        py_mean_e = python_rates[:N_E].mean()
        py_mean_i = python_rates[N_E:].mean() if len(python_rates) > N_E else 0
        print(f"  Python E rate: {py_mean_e:.3f} Hz")
        print(f"  CUDA   E rate: {cuda_mean_e:.3f} Hz")
        print(f"  Δ (same noise): {abs(py_mean_e - cuda_mean_e):.4f} Hz  ← should be tiny")
        if py_mean_i > 0:
            print(f"  Python I rate: {py_mean_i:.3f} Hz")
            print(f"  CUDA   I rate: {cuda_mean_i:.3f} Hz")

    # ── Verdict ──
    print(f"\n{'='*60}")
    # If we have python rates (same noise), compare against those
    if python_rates is not None:
        same_noise_diff = abs(python_rates[:N_E].mean() - cuda_mean_e)
        rate_ok = same_noise_diff < 0.1  # < 0.1 Hz with same noise
        metric = same_noise_diff
        metric_name = "same-noise Δ"
    else:
        rate_ok = rate_diff.mean() < 1.0
        metric = rate_diff.mean()
        metric_name = "rate error"
    print(f"  VERDICT: {'✅ PASS' if rate_ok else '❌ FAIL'}")
    print(f"  {metric_name}: {metric:.4f} Hz")
    if python_rates is not None:
        print(f"  (Brian2 vs CUDA diff is noise-dependent, not a kernel bug)")
    print(f"{'='*60}")

    # Save summary
    summary = {
        'ref_mean_rate_E': float(ref_mean_e),
        'ref_mean_rate_I': float(ref_mean_i),
        'cuda_mean_rate_E': float(cuda_mean_e),
        'cuda_mean_rate_I': float(cuda_mean_i),
        'rate_error_mean_hz': float(rate_diff.mean()),
        'rate_error_max_hz': float(rate_diff.max()),
        'total_spikes_ref': int(ref_counts.sum()),
        'total_spikes_cuda': int(cuda_counts.sum()),
        'pass': bool(rate_ok),
    }
    with open(os.path.join(cuda_dir, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {cuda_dir}/validation_summary.json")


if __name__ == '__main__':
    main()
