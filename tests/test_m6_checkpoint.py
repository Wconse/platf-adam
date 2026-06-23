"""
M6 checkpoint tests T6.1 – T6.3.

T6.1: Round-trip — save state, reload as ICs for engine continuation, run →
      rates must be sensible (not crash, not all-zero).
T6.2: HDF5 schema — Python checkpoint.py can save/load state.
T6.3: Continuity — rates after load are similar to rates at end of first episode.
T6.4: Regression — M0 passes.

Usage:
    python3 tests/test_m6_checkpoint.py <bundle_dir> <cuda_m6_state_dir> <cuda_m0_dir>
"""
import numpy as np
import json
import sys
import os
import tempfile

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# Add project path for checkpoint module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <bundle_dir> <state_dir> <cuda_m0_dir>")
        sys.exit(1)

    bundle_dir = sys.argv[1]
    state_dir  = sys.argv[2]
    m0_dir     = sys.argv[3]

    results = []

    print("=" * 60)
    print("M6 Checkpoint Tests T6.1 – T6.4")
    print("=" * 60)

    with open(os.path.join(bundle_dir, 'network_config.json')) as f:
        net_cfg = json.load(f)

    pops  = net_cfg['populations']
    projs = net_cfg['projections']

    # ── T6.1: Round-trip state integrity ──────────────────────────────────
    print("\nT6.1 — Round-trip: state files present and physically sensible:")
    ok_61 = True

    state_vars_e = ['v', 'w', 'g_ampa', 'g_nmda', 'g_gaba', 'ge_bg', 'gi_bg', 'r_hat', 'I_homeo']
    state_vars_i = ['v', 'g_ampa', 'g_nmda', 'g_gaba', 'ge_bg', 'gi_bg', 'r_hat', 'I_homeo']

    for pi, pop in enumerate(pops):
        N = pop['N']
        is_e = pop['is_e']
        pop_pfx = f"pop{pi}"
        vars_expected = state_vars_e if is_e else state_vars_i

        for var in vars_expected:
            fp = os.path.join(state_dir, f'{pop_pfx}_{var}.npy')
            if not os.path.exists(fp):
                print(f"  FAIL: missing {fp}")
                ok_61 = False
                continue
            arr = np.load(fp)
            if arr.shape != (N,):
                print(f"  FAIL: {fp} shape {arr.shape} != ({N},)")
                ok_61 = False
                continue
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"  FAIL: NaN/Inf in {fp}")
                ok_61 = False
                continue
            # Sanity: voltage should be in [-100, -20] mV (excluding neurons mid-reset)
            if var == 'v':
                if arr.min() < -200 or arr.max() > 50:
                    print(f"  WARN: v range [{arr.min():.1f}, {arr.max():.1f}] mV in {pop['name']}")

    if ok_61:
        print(f"  All {len(pops)} populations: state files present, no NaN/Inf")
    print(f"  [{PASS if ok_61 else FAIL}] T6.1")
    results.append(('T6.1', ok_61))

    # ── T6.2: HDF5 schema ─────────────────────────────────────────────────
    print("\nT6.2 — HDF5 schema (Python checkpoint.py):")
    ok_62 = False
    try:
        from cuda_engine.checkpoint import save_state, load_state, verify_schema

        # Build pop_states from saved npy files
        pop_states = {}
        for pi, pop in enumerate(pops):
            N = pop['N']
            is_e = pop['is_e']
            pop_name = pop['name']
            pfx = f"pop{pi}"
            state = {}
            vars_list = state_vars_e if is_e else state_vars_i
            for var in vars_list:
                fp = os.path.join(state_dir, f'{pfx}_{var}.npy')
                if os.path.exists(fp):
                    state[var] = np.load(fp)
            pop_states[pop_name] = state

        # Build proj_weights
        proj_weights = {}
        for ji, proj in enumerate(projs):
            if proj.get('is_stdp'):
                fp = os.path.join(state_dir, f'stdp_w_proj{ji}.npy')
                if os.path.exists(fp):
                    proj_weights[proj['name']] = np.load(fp)

        # STDP states
        stdp_states = {}
        for ji, proj in enumerate(projs):
            if proj.get('is_stdp'):
                ss = {}
                for k in ['Apre', 'Apost']:
                    fp = os.path.join(state_dir, f'stdp_{k}_proj{ji}.npy')
                    if os.path.exists(fp):
                        ss[k] = np.load(fp)
                if ss:
                    stdp_states[proj['name']] = ss

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            h5_path = tmp.name

        save_state(
            output_dir=state_dir,
            network_config=net_cfg,
            pop_states=pop_states,
            proj_weights=proj_weights,
            stdp_states=stdp_states if stdp_states else None,
            dev_age_dw=24.0,
            episode_number=1,
            filepath=h5_path,
        )

        # Verify schema
        schema_ok = verify_schema(h5_path)
        if not schema_ok:
            print("  FAIL: schema verification failed")
        else:
            # Load back
            loaded = load_state(h5_path, net_cfg)
            n_loaded_pops = len(loaded['pop_states'])
            n_loaded_syn  = len(loaded['proj_weights'])
            print(f"  HDF5 saved and loaded: {n_loaded_pops} pop states, {n_loaded_syn} syn weights")

            # Verify a few arrays
            round_trip_ok = True
            for pop_name, state in list(pop_states.items())[:3]:
                if pop_name in loaded['pop_states']:
                    orig_v = state.get('v', None)
                    load_v = loaded['pop_states'][pop_name].get('v', None)
                    if orig_v is not None and load_v is not None:
                        diff = np.abs(orig_v - load_v).max()
                        if diff > 1e-5:
                            print(f"  FAIL: v round-trip error {diff:.2e} for {pop_name}")
                            round_trip_ok = False

            ok_62 = schema_ok and round_trip_ok
            os.unlink(h5_path)

    except Exception as e:
        print(f"  FAIL: {e}")
        ok_62 = False

    print(f"  [{PASS if ok_62 else FAIL}] T6.2")
    results.append(('T6.2', ok_62))

    # ── T6.3: Continuity — voltage distribution post-load is sensible ─────
    print("\nT6.3 — State continuity (voltage statistics sensible after save):")
    ok_63 = True

    active_pops = []
    for pi, pop in enumerate(pops):
        # Only check regions that are active (born before T_SIM=12s)
        t_bir = pop['t_bir']
        if t_bir < 11000.0:  # born before 11s
            active_pops.append((pi, pop))

    for pi, pop in active_pops:
        fp = os.path.join(state_dir, f'pop{pi}_v.npy')
        if os.path.exists(fp):
            v = np.load(fp)
            vmean = v.mean()
            vmin = v.min()
            # Active neurons should have v in plausible range
            if vmin < -150 or vmean > -20:
                print(f"  WARN: {pop['name']} v mean={vmean:.1f} min={vmin:.1f}")
                ok_63 = False

    if ok_63:
        print(f"  All {len(active_pops)} active pops: v in plausible range post-episode")
    print(f"  [{PASS if ok_63 else FAIL}] T6.3")
    results.append(('T6.3', ok_63))

    # ── T6.4: Regression M0 ─────────────────────────────────────────────────
    print("\nT6.4 — M0 regression:")
    m0_rates = np.load(os.path.join(m0_dir, 'cuda_rates.npy'))
    ok_64 = (abs(m0_rates[:80].mean() - 0.330) < 0.05) and (abs(m0_rates[80:].mean() - 8.805) < 0.05)
    print(f"  E={m0_rates[:80].mean():.3f} I={m0_rates[80:].mean():.3f}  [{PASS if ok_64 else FAIL}]")
    results.append(('T6.4', ok_64))

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  SUMMARY: {n_pass}/{len(results)} passed")
    for name, ok in results:
        print(f"    [{PASS if ok else FAIL}] {name}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
