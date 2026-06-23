"""
adam.py — Clean ADAM training loop. Designed for Phase-1..4 from scratch.

One responsibility: run episodes, advance developmental state, collect metrics.
No legacy. No hardcoded paths. No monolithic CLI.

Usage:
    python adam.py                        # baseline run, dw 20->52, 1 seed
    python adam.py --dw-end 35 --seeds 3  # short run, 3 seeds
    python adam.py --phase1               # enable Phase-1 chloride dynamics
    python adam.py --tag vpa --vpa 0.35   # VPA intervention
"""
import argparse, json, time, shutil
from pathlib import Path
import numpy as np

ROOT   = Path(__file__).resolve().parent
ENGINE = ROOT / 'cuda' / 'build' / 'sim_multiregion.exe'
BUNDLE = ROOT / 'reference' / 'multiregion_v2'

# ── Imports from cleaned modules ─────────────────────────────────────────────
import sys
sys.path.insert(0, str(ROOT))
from cuda_engine.dev_state import DevStateVector, ChlorideState
from cuda_engine.discontinuity import detect_bursts, population_rate


# ── Engine runner (minimal, no legacy) ───────────────────────────────────────

def run_episode(bundle_dir: Path, engine: Path, ep_dir: Path,
                dev_state: dict, pop_states: dict | None,
                disable_homeo: bool, disable_scaling: bool,
                seed_noise: int, seed_env: int) -> dict:
    """
    Run one engine episode. Returns dict with rates, net_activity, pop_states,
    region_rates_E, region_rates_I.

    Writes ta_egaba from dev_state['egaba_mV'] so ChlorideState E_GABA is used
    when Phase-1 is active.
    """
    import subprocess
    ep_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(bundle_dir / 'network_config.json'))
    T   = cfg['T_steps']
    DT  = cfg['dt_ms']

    # ── Write per-episode overrides ───────────────────────────────────────────
    # E_GABA (from dev_state — Phase-1 chloride or classic sigmoid)
    for region, egaba_val in dev_state.get('egaba_mV', {}).items():
        np.save(ep_dir / f'ta_egaba_{region}.npy',
                np.full(T, float(egaba_val), dtype=np.float32))

    np.save(ep_dir / 'ta_lr_gate.npy',
            np.full(T, float(dev_state.get('lr_gate', 0.5)), dtype=np.float32))
    np.save(ep_dir / 'ta_mg.npy',
            np.full(T, float(dev_state.get('mg_now', 1.0)), dtype=np.float32))
    np.save(ep_dir / 'ta_nmda_ratio.npy',
            np.full(T, float(dev_state.get('nmda_ratio', 0.08)), dtype=np.float32))
    np.save(ep_dir / 'ta_stress.npy',
            np.full(T, 0.5, dtype=np.float32))

    # ── Homeostasis disable patch ─────────────────────────────────────────────
    ep_cfg = json.loads(json.dumps(cfg))   # deep copy
    if disable_homeo:
        for pop in ep_cfg['populations']:
            pop['I_cl'] = 0.0
            pop['I_scl'] = 0.0
    if disable_scaling:
        pass   # scaling handled post-run

    # ── ICs (carry from previous episode or bundle defaults) ─────────────────
    for pi, pop in enumerate(cfg['populations']):
        for var in ['v', 'ge', 'gi', 'w']:
            src = bundle_dir / f'pop_{pi}_{var}.npy'
            dst = ep_dir / f'pop_{pi}_{var}.npy'
            if pop_states and pop['name'] in pop_states:
                ps = pop_states[pop['name']]
                if var == 'ge' and 'ge_bg' in ps:
                    np.save(dst, ps['ge_bg'].astype(np.float32))
                    continue
                if var == 'gi' and 'gi_bg' in ps:
                    np.save(dst, ps['gi_bg'].astype(np.float32))
                    continue
                if var in ps:
                    arr = ps[var]
                    if var in ('g_ampa', 'g_nmda', 'g_gaba'):
                        arr = np.zeros_like(arr)
                    np.save(dst, arr.astype(np.float32))
                    continue
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # ── Build episode bundle (symlinks + overrides) ───────────────────────────
    ebundle = ep_dir / 'bundle'
    ebundle.mkdir(exist_ok=True)
    for f in bundle_dir.iterdir():
        if f.is_file():
            link = ebundle / f.name
            if not link.exists():
                try:   link.symlink_to(f.resolve())
                except: shutil.copy2(f, link)
    # Write patched config
    json.dump(ep_cfg, open(ep_dir / 'network_config.json', 'w'))
    for src in ep_dir.iterdir():
        if src.is_file():
            dst = ebundle / src.name
            if dst.exists(): dst.unlink()
            shutil.copy2(src, dst)

    # ── Run engine ────────────────────────────────────────────────────────────
    out_dir = ep_dir / 'output'
    out_dir.mkdir(exist_ok=True)
    result = subprocess.run([str(engine), str(ebundle), str(out_dir), '--save-state'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Engine failed:\n{result.stderr[-2000:]}")

    # ── Read outputs ──────────────────────────────────────────────────────────
    rates       = np.load(out_dir / 'cuda_rates.npy').astype(np.float64)
    net_path    = out_dir / 'net_activity.npy'
    net_activity = np.load(net_path).astype(np.float64) if net_path.exists() else None

    # Per-pop carry state
    new_pop_states = {}
    for pop in cfg['populations']:
        pi = pop['id']; name = pop['name']
        state = {}
        for var in ['v', 'ge_bg', 'gi_bg', 'g_ampa', 'g_nmda', 'g_gaba',
                    'I_homeo', 'r_hat', 'w']:
            fp = out_dir / f'pop{pi}_{var}.npy'
            if fp.exists():
                state[var] = np.load(fp)
        new_pop_states[name] = state

    # Per-region rates
    region_rates_E: dict = {}
    region_rates_I: dict = {}
    for pop in cfg['populations']:
        r = pop['region']; g = pop['global_offset']; N = pop['N']
        hz = float(rates[g:g+N].mean())
        if pop['is_e']:
            region_rates_E[r] = hz
        else:
            region_rates_I.setdefault(r, []).append(hz)
    region_rates_I = {r: float(np.mean(v)) for r, v in region_rates_I.items()}

    return {
        'rates':          rates,
        'net_activity':   net_activity,
        'pop_states':     new_pop_states,
        'region_rates_E': region_rates_E,
        'region_rates_I': region_rates_I,
        'T_s':            T * DT / 1000.0,
        'N_total':        sum(p['N'] for p in cfg['populations']),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(out: dict, dw: float) -> dict:
    """Compute per-episode scalar metrics from engine output."""
    net = out['net_activity']
    N   = out['N_total']
    T_s = out['T_s']
    mean_hz = float(net.sum() / (len(net) * 0.0001)) / N if net is not None else 0.0

    n_bursts = 0; median_ibi = None
    if net is not None and mean_hz > 0:
        R, valid, _ = population_rate(net, N, bin_ms=5.0, dt_ms=0.1)
        burst_list, binfo = detect_bursts(R, valid, on_factor=3.0, off_factor=1.5, dmin_ms=50.0)
        n_bursts = len(burst_list)
        if len(burst_list) > 1:
            ibis = [(burst_list[i+1][0] - burst_list[i][1]) * 5.0 / 1000.0
                    for i in range(len(burst_list) - 1)]
            median_ibi = float(np.median(ibis))

    return {
        'dw':          dw,
        'mean_hz':     mean_hz,
        'n_bursts':    n_bursts,
        'median_ibi':  median_ibi,
        'region_hz_E': out['region_rates_E'],
        'region_hz_I': out['region_rates_I'],
    }


# ── Main training loop ────────────────────────────────────────────────────────

def run(tag: str, dw_start: float, dw_end: float, dw_step: float,
        seed: int, phase1: bool, vpa_factor: float,
        disable_homeo: bool, out_root: Path):

    out_dir = out_root / tag / f'seed{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg     = json.load(open(BUNDLE / 'network_config.json'))
    ep_s    = cfg['T_steps'] * cfg['dt_ms'] / 1000.0

    dsv = DevStateVector(initial_dw=dw_start)
    dsv.gaba_shifts = {r: 0. for r in dsv.regions}
    dsv.ei_bias_delta_pA = {r: 0. for r in dsv.regions}
    dsv.lr_gate_multiplier = 1.0

    cl_state = ChlorideState(dev_age=dw_start) if phase1 else None

    pop_states = None
    history    = []

    dw = dw_start
    ep = 0
    while dw < dw_end:
        t0 = time.time()
        dsv.dev_age = dw

        # Get developmental state
        state = dsv.get_state()

        # Phase-1: override E_GABA with chloride-derived values
        if cl_state is not None:
            state['egaba_mV'] = cl_state.get_egaba()

        # VPA: shift GABA flip center
        if vpa_factor < 1.0:
            dsv.kcc2_rate_factor = vpa_factor
            dsv.use_kcc2_mechanistic = True

        ep_dir = out_dir / f'ep{ep:04d}'
        out = run_episode(
            bundle_dir=BUNDLE, engine=ENGINE, ep_dir=ep_dir,
            dev_state=state, pop_states=pop_states,
            disable_homeo=disable_homeo, disable_scaling=True,
            seed_noise=137 + seed, seed_env=137 + seed)

        # Update chloride state from this episode's inhibitory activity
        if cl_state is not None:
            cl_state.update(out['region_rates_I'], ep_s, dw)

        metrics = compute_metrics(out, dw)
        history.append(metrics)
        pop_states = out['pop_states']

        eg_sp = state['egaba_mV'].get('SP', 0)
        eg_p1 = state['egaba_mV'].get('P1', 0)
        print(f"  ep{ep:03d} dw={dw:.1f}  {metrics['mean_hz']:.3f}Hz "
              f" bursts={metrics['n_bursts']} IBI={metrics['median_ibi']}"
              f"  E_GABA SP={eg_sp:.0f} P1={eg_p1:.0f}  [{time.time()-t0:.0f}s]")

        dw += dw_step
        ep += 1

    # Save history
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    if cl_state is not None:
        json.dump({'cl_i': cl_state.get_cl_i(), 'egaba': cl_state.get_egaba()},
                  open(out_dir / 'chloride_final.json', 'w'), indent=2)

    print(f"\n  Done. {ep} episodes. History saved to {out_dir}/history.json")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='ADAM training loop')
    p.add_argument('--tag',          default='baseline')
    p.add_argument('--dw-start',     type=float, default=20.0)
    p.add_argument('--dw-end',       type=float, default=52.0)
    p.add_argument('--dw-step',      type=float, default=0.75)
    p.add_argument('--seeds',        type=int,   default=1)
    p.add_argument('--phase1',       action='store_true',
                   help='Enable Phase-1 chloride dynamics (emergent E_GABA)')
    p.add_argument('--no-homeo',     action='store_true')
    p.add_argument('--vpa',          type=float, default=1.0,
                   help='KCC2 rate factor for VPA (1.0=baseline, 0.35=VPA)')
    p.add_argument('--out',          default='output/runs')
    args = p.parse_args()

    out_root = ROOT / args.out
    print(f"ADAM  tag={args.tag}  dw {args.dw_start}→{args.dw_end} "
          f" step={args.dw_step}  seeds={args.seeds}"
          f"  phase1={args.phase1}  vpa={args.vpa}")
    print(f"  bundle: {BUNDLE}")
    print(f"  engine: {ENGINE}\n")

    for seed in range(args.seeds):
        print(f"--- Seed {seed} ---")
        run(tag=args.tag, dw_start=args.dw_start, dw_end=args.dw_end,
            dw_step=args.dw_step, seed=seed, phase1=args.phase1,
            vpa_factor=args.vpa, disable_homeo=args.no_homeo,
            out_root=out_root)


if __name__ == '__main__':
    main()
