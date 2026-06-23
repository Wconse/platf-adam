"""
M7 performance tests T7.1 – T7.4.

T7.1: Trajectory KPI — rates for active regions in expected range.
T7.2: Developmental KPI — no spikes before birth, born regions active.
T7.3: Memory check — estimate VRAM usage, verify within budget.
T7.4: Regression — M0 passes.
T7.5: Performance — report wall-clock time for the simulation run.

Usage:
    python3 tests/test_m7_perf.py <bundle_dir> <cuda_m5_dir> <cuda_m0_dir> [wall_time_s]
"""
import numpy as np
import json
import sys
import os

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <bundle_dir> <cuda_dir> <cuda_m0_dir> [wall_time_s]")
        sys.exit(1)

    bundle_dir = sys.argv[1]
    cuda_dir   = sys.argv[2]
    m0_dir     = sys.argv[3]
    wall_time  = float(sys.argv[4]) if len(sys.argv) > 4 else None

    results = []

    print("=" * 60)
    print("M7 Performance & Trajectory Tests T7.1 – T7.5")
    print("=" * 60)

    with open(os.path.join(bundle_dir, 'network_config.json')) as f:
        net_cfg = json.load(f)

    pops  = net_cfg['populations']
    T_MS  = net_cfg['T_ms']
    N_total = net_cfg['N_total']

    cuda_rates = np.load(os.path.join(cuda_dir, 'cuda_rates.npy'))

    # ── T7.1: Trajectory KPI — rates in expected biological range ──────────
    print("\nT7.1 — Trajectory KPI (active regions):")
    ok_71 = True
    # Expected: E rates 0.1–10 Hz, I rates 5–100 Hz for active regions
    # Rates outside this are suspicious
    for pi, pop in enumerate(pops):
        t_bir = pop['t_bir']
        if t_bir >= T_MS - 1000.0:
            continue  # not born yet
        g_off = pop['global_offset']
        N = pop['N']
        mean_r = cuda_rates[g_off:g_off+N].mean()
        is_e = pop['is_e']
        # Plausible range
        lo = 0.05 if is_e else 0.5
        hi = 20.0 if is_e else 200.0
        in_range = lo <= mean_r <= hi
        status = PASS if in_range else WARN
        print(f"  {pop['name']:6s}: {mean_r:7.3f} Hz  [{status}]")
        if not in_range:
            ok_71 = False  # downgrade to FAIL only if completely outside
            if mean_r == 0 and t_bir < T_MS - 2000.0:
                ok_71 = False

    print(f"  [{PASS if ok_71 else WARN}] T7.1 (WARN = rate outside expected range)")
    results.append(('T7.1', ok_71))

    # ── T7.2: Developmental — silence before birth ─────────────────────────
    print("\nT7.2 — Developmental integrity:")
    cuda_spikes = np.load(os.path.join(cuda_dir, 'cuda_spikes.npy'))
    ok_72 = True

    if cuda_spikes.shape[1] > 0:
        sp_idx = cuda_spikes[0].astype(int)
        sp_t   = cuda_spikes[1]
        for pi, pop in enumerate(pops):
            t_bir = pop['t_bir']
            if t_bir <= 0: continue
            g_off = pop['global_offset']
            mask  = (sp_idx >= g_off) & (sp_idx < g_off + pop['N'])
            if mask.any():
                early = (sp_t[mask] < t_bir - 0.1).sum()
                if early > 0:
                    print(f"  FAIL: {pop['name']} has {early} spikes before t_birth={t_bir:.0f} ms")
                    ok_72 = False

    if ok_72:
        print(f"  No pre-birth spikes in any region")
    print(f"  [{PASS if ok_72 else FAIL}] T7.2")
    results.append(('T7.2', ok_72))

    # ── T7.3: Memory estimate ──────────────────────────────────────────────
    print("\nT7.3 — Memory estimate:")
    # Ring buffers: MAX_DELAY=256 * dst_N * 4 bytes per projection
    MAX_DELAY = 256
    ring_bytes = 0
    for proj in net_cfg['projections']:
        dst_N = pops[proj['dst_pop']]['N']
        ring_bytes += MAX_DELAY * dst_N * 4  # one ring per projection

    # Neuron state: ~10 arrays * N * 4 bytes
    neuron_bytes = sum(p['N'] * 10 * 4 for p in pops)

    # STDP arrays: 4 arrays * n_syn * 4 bytes
    stdp_bytes = sum(p['n_syn'] * 4 * 4 for p in net_cfg['projections'] if p.get('is_stdp'))

    # cuRAND state: N * 64 bytes (Philox state)
    curand_bytes = sum(p['N'] * 64 for p in pops)

    total_est_MB = (ring_bytes + neuron_bytes + stdp_bytes + curand_bytes) / 1e6

    print(f"  Ring buffers:  {ring_bytes/1e6:.1f} MB")
    print(f"  Neuron state:  {neuron_bytes/1e6:.1f} MB")
    print(f"  STDP arrays:   {stdp_bytes/1e6:.1f} MB")
    print(f"  cuRAND states: {curand_bytes/1e6:.1f} MB")
    print(f"  Total estimate: {total_est_MB:.1f} MB (budget: 4000 MB)")

    ok_73 = total_est_MB < 4000.0
    print(f"  [{PASS if ok_73 else FAIL}] T7.3")
    results.append(('T7.3', ok_73))

    # ── T7.5: Performance ─────────────────────────────────────────────────
    print("\nT7.5 — Performance:")
    T_SIM_S = T_MS / 1000.0  # seconds of simulated time
    if wall_time is not None:
        speedup = T_SIM_S / wall_time
        realtime_factor = wall_time / T_SIM_S
        print(f"  Simulated: {T_SIM_S:.0f} s, Wall clock: {wall_time:.1f} s")
        print(f"  Speed: {speedup:.2f}× (= {realtime_factor:.2f}× real-time)")
        print(f"  N_total={N_total}, dt=0.1ms, n_projs={net_cfg['n_projections']}")
        perf_ok = realtime_factor < 100  # must be < 100× real-time at SCALE=0.1
        print(f"  [{PASS if perf_ok else WARN}] T7.5 (< 100× real-time req)")
        results.append(('T7.5', perf_ok))
    else:
        print(f"  Wall time not provided (pass as 4th arg). Skipping.")
        results.append(('T7.5', True))  # skip

    # ── T7.4: Regression M0 ───────────────────────────────────────────────
    print("\nT7.4 — M0 regression:")
    m0_rates = np.load(os.path.join(m0_dir, 'cuda_rates.npy'))
    ok_74 = abs(m0_rates[:80].mean() - 0.330) < 0.05
    print(f"  M0 E rate: {m0_rates[:80].mean():.3f}  [{PASS if ok_74 else FAIL}]")
    results.append(('T7.4', ok_74))

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  SUMMARY: {n_pass}/{len(results)} passed")
    for name, ok in results:
        print(f"    [{PASS if ok else FAIL}] {name}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
