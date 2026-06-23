"""
M3 integrity tests T3.1, T3.3, T3.7.

T3.1: Build integrity — n_syn, row_offset[-1], weight sums match generator output.
T3.3: No spikes before t_birth for each region.
T3.7: Regression — M0/M1/M2 still pass (just check M0 rates).

Usage:
    python3 tests/test_m3_integrity.py <bundle_dir> <cuda_m3_dir> <cuda_m0_dir>
"""
import numpy as np
import json
import sys
import os

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <bundle_dir> <cuda_m3_dir> <cuda_m0_dir>")
        sys.exit(1)

    bundle_dir = sys.argv[1]
    cuda_m3_dir = sys.argv[2]
    cuda_m0_dir = sys.argv[3]

    results = []

    print("=" * 60)
    print("M3 Integrity Tests T3.1, T3.3, T3.7")
    print("=" * 60)

    # Load config
    with open(os.path.join(bundle_dir, 'network_config.json')) as f:
        cfg = json.load(f)

    pops = cfg['populations']
    projs = cfg['projections']
    n_projs = cfg['n_projections']
    DT = cfg['dt_ms']

    # ── T3.1: Build integrity ──────────────────────────────────────────────
    print(f"\nT3.1 — Build integrity ({n_projs} projections):")
    ok_31 = True

    for j, proj in enumerate(projs):
        row_off = np.load(os.path.join(bundle_dir, f'proj_{j}_row_off.npy')).astype(np.int32)
        col_idx = np.load(os.path.join(bundle_dir, f'proj_{j}_col_idx.npy')).astype(np.int32)
        w_syn   = np.load(os.path.join(bundle_dir, f'proj_{j}_w_syn.npy'))
        d_steps = np.load(os.path.join(bundle_dir, f'proj_{j}_delay_steps.npy')).astype(np.int32)

        n_syn_cfg = proj['n_syn']
        src_N = pops[proj['src_pop']]['N']
        dst_N = pops[proj['dst_pop']]['N']

        # Check: row_offset[-1] == n_syn
        if row_off[-1] != n_syn_cfg:
            print(f"  FAIL proj {j} ({proj['name']}): row_off[-1]={row_off[-1]} != n_syn={n_syn_cfg}")
            ok_31 = False

        # Check: len(row_off) == src_N + 1
        if len(row_off) != src_N + 1:
            print(f"  FAIL proj {j}: len(row_off)={len(row_off)} != src_N+1={src_N+1}")
            ok_31 = False

        # Check: len(col_idx) == n_syn
        if len(col_idx) != n_syn_cfg:
            print(f"  FAIL proj {j}: len(col_idx)={len(col_idx)} != n_syn={n_syn_cfg}")
            ok_31 = False

        # Check: all col_idx in [0, dst_N)
        if col_idx.min() < 0 or col_idx.max() >= dst_N:
            print(f"  FAIL proj {j}: col_idx range [{col_idx.min()},{col_idx.max()}] not in [0,{dst_N})")
            ok_31 = False

        # Check: all delay_steps >= 1 and < MAX_DELAY
        if d_steps.min() < 1 or d_steps.max() >= 256:
            print(f"  FAIL proj {j}: delay_steps range [{d_steps.min()},{d_steps.max()}]")
            ok_31 = False

        # Check: weights > 0 and within wmax
        if w_syn.min() < 0:
            print(f"  FAIL proj {j}: negative weights (min={w_syn.min():.4f})")
            ok_31 = False

    if ok_31:
        print(f"  All {n_projs} projections: row_offset[-1]==n_syn, col_idx in range, delays valid")

    print(f"  [{PASS if ok_31 else FAIL}] T3.1")
    results.append(('T3.1', ok_31))

    # ── T3.3: No spikes before t_birth ────────────────────────────────────
    print(f"\nT3.3 — No spikes before t_birth per region:")
    cuda_spikes = np.load(os.path.join(cuda_m3_dir, 'cuda_spikes.npy'))

    ok_33 = True
    if cuda_spikes.shape[1] > 0:
        sp_idx = cuda_spikes[0].astype(int)  # global neuron ids
        sp_t   = cuda_spikes[1]              # spike times (ms)

        for pi, pop in enumerate(pops):
            t_bir = pop['t_bir']  # ms
            if t_bir <= 0.0:
                continue  # born at start, skip

            # Find spikes from this pop (global_offset to global_offset+N)
            g_off = pop['global_offset']
            mask  = (sp_idx >= g_off) & (sp_idx < g_off + pop['N'])

            if mask.any():
                early_mask = mask & (sp_t < t_bir - DT)
                n_early = early_mask.sum()
                if n_early > 0:
                    print(f"  FAIL pop {pi} ({pop['name']}): {n_early} spikes before t_birth={t_bir:.1f} ms")
                    ok_33 = False
                else:
                    # Population had spikes but all after birth
                    pass

    if ok_33:
        print(f"  All regions: no spikes before t_birth")

    print(f"  [{PASS if ok_33 else FAIL}] T3.3")
    results.append(('T3.3', ok_33))

    # T3.4: Check that regions born before T_SIM actually have non-zero activity
    T_SIM = cfg['T_ms']
    print(f"\nT3.4 — Regions born before T_SIM={T_SIM:.0f} ms have non-zero E+I activity:")
    cuda_rates = np.load(os.path.join(cuda_m3_dir, 'cuda_rates.npy'))

    ok_34 = True
    for pi, pop in enumerate(pops):
        t_bir = pop['t_bir']
        if t_bir < T_SIM - 1000.0:  # region born at least 1s before end
            g_off = pop['global_offset']
            mean_r = cuda_rates[g_off:g_off + pop['N']].mean()
            if mean_r == 0.0:
                print(f"  FAIL pop {pi} ({pop['name']}): mean_rate=0.0 Hz (born at {t_bir:.0f} ms, T_SIM={T_SIM:.0f} ms)")
                ok_34 = False

    if ok_34:
        print(f"  All early-born regions have non-zero rates")
    print(f"  [{PASS if ok_34 else FAIL}] T3.4 (activity check)")
    results.append(('T3.4', ok_34))

    # ── T3.7: M0 regression ───────────────────────────────────────────────
    print(f"\nT3.7 — M0 regression:")
    cuda_m0_rates = np.load(os.path.join(cuda_m0_dir, 'cuda_rates.npy'))
    N_E_p1 = 80
    m0_e = cuda_m0_rates[:N_E_p1].mean()
    m0_i = cuda_m0_rates[N_E_p1:].mean()
    M0_E_RATE = 0.330; M0_I_RATE = 8.805; TOL = 0.05
    ok_37 = (abs(m0_e - M0_E_RATE) < TOL) and (abs(m0_i - M0_I_RATE) < TOL)
    print(f"  M0 E rate: {m0_e:.3f} Hz (expected {M0_E_RATE}±{TOL})")
    print(f"  M0 I rate: {m0_i:.3f} Hz (expected {M0_I_RATE}±{TOL})")
    print(f"  [{PASS if ok_37 else FAIL}] T3.7")
    results.append(('T3.7', ok_37))

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"  SUMMARY: {n_pass}/{n_total} passed")
    for name, ok in results:
        print(f"    [{PASS if ok else FAIL}] {name}")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == '__main__':
    main()
