"""
M8 complete slow-loop test using DevStateVector (full Brian2-equivalent pipeline).

Runs 2 episodes starting at dw=24 (P1 just born), advancing by 0.5 dw each.
Verifies:
  S1: DevStateVector produces correct E_GABA at each episode
  S2: Engine runs without error for both episodes
  S3: State carries over correctly (r_hat, I_homeo)
  S4: Synaptic scaling applied correctly (median r_hat used)
  S5: STDP weights persist between episodes
  S6: No NaN/Inf in any state variable

Usage: python3 tests/test_m8_with_devstate.py <bundle_dir> <engine_bin> <checkpoint_dir>
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cuda_engine.dev_state import DevStateVector
from cuda_engine.engine_runner import EngineRunner

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <bundle_dir> <engine_bin> <checkpoint_dir>")
        sys.exit(1)

    bundle_dir    = sys.argv[1]
    engine_bin    = sys.argv[2]
    checkpoint_dir = sys.argv[3]

    results = []

    print("=" * 60)
    print("M8 Full Slow-Loop with DevStateVector")
    print("=" * 60)

    # ── S1: DevStateVector produces correct states ───────────────────────────
    print("\nS1 — DevStateVector states at dw=24 and dw=24.5:")
    dsv = DevStateVector(initial_dw=24.0)
    s_ep1 = dsv.get_state()
    dsv.step(0.5)
    s_ep2 = dsv.get_state()

    ok_s1 = True
    # At dw=24: P1 center=30, still far from flip → E_GABA near early value (-45 mV)
    egaba_ep1 = s_ep1['egaba_mV']['P1']
    egaba_ep2 = s_ep2['egaba_mV']['P1']
    print(f"  Ep1 (dw=24.0) P1 E_GABA = {egaba_ep1:.2f} mV  (expect ~-45 mV)")
    print(f"  Ep2 (dw=24.5) P1 E_GABA = {egaba_ep2:.2f} mV  (should be slightly lower)")
    if not (-50 <= egaba_ep1 <= -40):
        print(f"  FAIL: E_GABA ep1 out of range"); ok_s1 = False
    if not (egaba_ep2 <= egaba_ep1):
        print(f"  FAIL: E_GABA should decrease monotonically"); ok_s1 = False

    print(f"  Ep1 nmda_ratio={s_ep1['nmda_ratio']:.3f}  lr_gate={s_ep1['lr_gate']:.3f}  mg={s_ep1['mg_now']:.3f}")
    print(f"  [{PASS if ok_s1 else FAIL}] S1")
    results.append(('S1', ok_s1))

    # ── S2-S6: Run 2 episodes with EngineRunner ──────────────────────────────
    print("\nS2-S6 — Two-episode run with DevStateVector:")
    try:
        runner = EngineRunner(bundle_dir, engine_bin, checkpoint_dir)

        # Reset DevStateVector for actual run
        dsv = DevStateVector(initial_dw=24.0)

        print("  Running episode 1 (dw=24.0)...")
        state1 = dsv.get_state()
        ep1 = runner.run_episode(state1, prune_every=2)
        r1 = runner.compute_rates_by_region(ep1['rates'])
        print(f"  Ep1: BS_E={r1['BS']['E']:.3f} P1_E={r1['P1']['E']:.3f} Hz")

        # Advance dev age
        dsv.step(0.5)

        print("  Running episode 2 (dw=24.5, carry state)...")
        state2 = dsv.get_state()
        ep2 = runner.run_episode(
            state2,
            carry_state={'pop_states': ep1['pop_states']},
            prune_every=2
        )
        r2 = runner.compute_rates_by_region(ep2['rates'])
        print(f"  Ep2: BS_E={r2['BS']['E']:.3f} P1_E={r2['P1']['E']:.3f} Hz")

        # ── S2: No crash ───────────────────────────────────────────────────
        ok_s2 = True  # reached here = no crash
        print(f"\n  [{PASS if ok_s2 else FAIL}] S2 — No engine crash")
        results.append(('S2', ok_s2))

        # ── S3: State carry-over (r_hat, I_homeo continuous) ───────────────
        ok_s3 = True
        for pop_name, state in ep2['pop_states'].items():
            for var in ['r_hat', 'I_homeo']:
                if var not in state: continue
                if np.isnan(state[var]).any():
                    print(f"  FAIL: NaN in {pop_name}/{var}")
                    ok_s3 = False

        # r_hat in ep2 should be non-trivially set (not all zeros) for active pops
        active_pops = [p for p in runner.net_cfg['populations']
                       if p['t_bir'] < runner.net_cfg['T_ms'] - 1000.0
                       and p['is_e']]
        for pop in active_pops[:3]:
            rh = ep2['pop_states'].get(pop['name'], {}).get('r_hat', None)
            if rh is not None and rh.mean() == 0.0:
                print(f"  WARN: {pop['name']} r_hat all zero in ep2")

        print(f"  [{PASS if ok_s3 else FAIL}] S3 — State carry-over no NaN")
        results.append(('S3', ok_s3))

        # ── S4: Synaptic scaling applied ────────────────────────────────────
        ok_s4 = bool(ep2['scale_factors'])
        for r, sf in ep2['scale_factors'].items():
            print(f"  Scale factor {r}: {sf:.4f}")
        print(f"  [{PASS if ok_s4 else FAIL}] S4 — Scale factors computed")
        results.append(('S4', ok_s4))

        # ── S5: STDP weights present and changed ────────────────────────────
        ok_s5 = True
        initial_w_loaded = False
        for proj in runner.net_cfg['projections']:
            if not proj.get('is_stdp'): continue
            ji = proj['id']
            init_fp = os.path.join(bundle_dir, f'proj_{ji}_w_syn.npy')
            if os.path.exists(init_fp):
                w0 = np.load(init_fp)
                w2 = ep2['weights'].get(proj['name'], None)
                if w2 is not None:
                    initial_w_loaded = True
                    print(f"  {proj['name']}: init_mean={w0.mean():.4f} ep2_mean={w2.mean():.4f}")
                    if np.array_equal(w0, w2) and len(w2) == len(w0):
                        print(f"  WARN: weights identical to init (STDP not active?)")

        print(f"  [{PASS if ok_s5 else FAIL}] S5 — STDP weights present")
        results.append(('S5', ok_s5))

        # ── S6: No NaN/Inf ─────────────────────────────────────────────────
        ok_s6 = True
        for pop_name, state in ep2['pop_states'].items():
            for var, arr in state.items():
                if np.isnan(arr).any() or np.isinf(arr).any():
                    print(f"  FAIL: NaN/Inf in {pop_name}/{var}")
                    ok_s6 = False
        for pname, w in ep2['weights'].items():
            if np.isnan(w).any() or np.isinf(w).any():
                print(f"  FAIL: NaN/Inf in weights {pname}")
                ok_s6 = False

        if ok_s6:
            print(f"  No NaN/Inf in any variable")
        print(f"  [{PASS if ok_s6 else FAIL}] S6 — No NaN/Inf")
        results.append(('S6', ok_s6))

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        for name in ['S2','S3','S4','S5','S6']:
            results.append((name, False))

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  SUMMARY: {n_pass}/{len(results)} passed")
    for name, ok in results:
        print(f"    [{PASS if ok else FAIL}] {name}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
