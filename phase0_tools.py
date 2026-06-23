"""
phase0_tools.py — Grounded-emergence instruments (audit harness + perturbation runner).

Two tools:
  fingerprint  — run a short baseline, record the full emergence "fingerprint"
                 (rates, E/I, bursts, synchrony, criticality) to a JSON reference.
                 Every later conversion is compared against this frozen reference.

  perturb      — Emergence Test 3 (robustness): sweep a parameter +/-X% and report
                 whether a target metric survives. Knife-edge = coincidence, not emergence.

Usage:
  python phase0_tools.py fingerprint --episodes 3 --tag ref_baseline
  python phase0_tools.py perturb --param ge0 --pct 15 --metric criticality.branching_ratio
"""
import sys, os, json, time, argparse, copy
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from cuda_engine.dev_state import DevStateVector
from cuda_engine.engine_runner import EngineRunner
from cuda_engine.discontinuity import analyze_discontinuity

ENGINE = ROOT / 'cuda' / 'build' / 'sim_multiregion.exe'
BUNDLE = ROOT / 'reference' / 'multiregion'


def _run_baseline(n_episodes, start_dw=20.0, delta_dw=1.0, seed_noise=137,
                  bundle=None, ckpt_dir=None):
    """Run a short baseline trajectory, return list of per-episode dicts."""
    bundle = Path(bundle) if bundle else BUNDLE
    ckpt = Path(ckpt_dir) if ckpt_dir else (ROOT / 'output' / '_phase0_tmp' / 'ckpt')
    ckpt.mkdir(parents=True, exist_ok=True)

    runner = EngineRunner(bundle_dir=str(bundle), engine_bin=str(ENGINE),
                          checkpoint_dir=str(ckpt), seed_noise=seed_noise,
                          seed_env=seed_noise)
    dsv = DevStateVector(initial_dw=start_dw)
    carry = None
    records = []
    for ep in range(n_episodes):
        dw = start_dw + ep * delta_dw
        dsv.dev_age = dw
        dsv.gaba_shifts = {r: 0.0 for r in dsv.regions}
        dsv.ei_bias_delta_pA = {r: 0.0 for r in dsv.regions}
        dsv.lr_gate_multiplier = 1.0
        state = dsv.get_state()
        out = runner.run_episode(dev_state=state, carry_state=carry, prune_every=99)
        carry = {'pop_states': out['pop_states']}
        rbr = runner.compute_rates_by_region(out['rates'])
        records.append({
            'dw': dw,
            'rates_by_region': rbr,
            'metrics': out.get('metrics', {}),
        })
        dsv.step(delta_dw)
    return records


def cmd_fingerprint(args):
    t0 = time.time()
    records = _run_baseline(args.episodes, seed_noise=args.seed)
    # Aggregate into a compact fingerprint (mean over episodes of key scalars)
    mr_vals, syncs = [], []
    for r in records:
        m = r['metrics']
        c = m.get('criticality') or {}
        mr = c.get('mr') or {}
        if mr.get('m') == mr.get('m'):  # not NaN
            mr_vals.append(mr.get('m'))
        if m.get('synchrony') is not None:
            syncs.append(m['synchrony'])

    fp = {
        'episodes': args.episodes,
        'seed': args.seed,
        'mean_mr_branching_ratio': float(np.mean(mr_vals)) if mr_vals else None,
        'mean_synchrony':          float(np.mean(syncs))    if syncs   else None,
        'per_episode': records,
    }
    out_dir = ROOT / 'output' / 'phase0_fingerprints'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fingerprint_{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(fp, f, indent=2)
    print(f"\n=== FINGERPRINT: {args.tag} ===")
    print(f"  MR branching ratio (m): {fp['mean_mr_branching_ratio']}")
    print(f"  synchrony:              {fp['mean_synchrony']}")
    print(f"  saved: {out_path}  ({time.time()-t0:.0f}s)")


def _get_nested(d, dotted):
    cur = d
    for k in dotted.split('.'):
        if cur is None: return None
        cur = cur.get(k)
    return cur


def cmd_perturb(args):
    """Run baseline at nominal, -pct, +pct of a global noise param; report metric."""
    # We perturb by editing the bundle's network_config.json param across all pops,
    # writing a temp bundle for each variant. Only safe global params allowed.
    import shutil
    base_cfg_path = BUNDLE / 'network_config.json'
    with open(base_cfg_path) as f:
        base_cfg = json.load(f)

    pct = args.pct / 100.0
    variants = {'nominal': 1.0, f'-{args.pct}%': 1.0 - pct, f'+{args.pct}%': 1.0 + pct}
    results = {}

    for label, factor in variants.items():
        # Build a temp bundle with the scaled parameter
        tmp_bundle = ROOT / 'output' / '_phase0_tmp' / f'bundle_{label.replace("%","pct").replace("+","p").replace("-","m")}'
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle)
        tmp_bundle.mkdir(parents=True)
        for fp in BUNDLE.iterdir():
            if fp.is_file():
                try:
                    (tmp_bundle / fp.name).symlink_to(fp.resolve())
                except Exception:
                    shutil.copy2(str(fp), str(tmp_bundle / fp.name))
        # Override config with scaled param
        cfg = copy.deepcopy(base_cfg)
        for pop in cfg['populations']:
            if args.param in pop:
                pop[args.param] = pop[args.param] * factor
        cfg_path = tmp_bundle / 'network_config.json'
        if cfg_path.exists() or cfg_path.is_symlink():
            cfg_path.unlink()
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f)

        recs = _run_baseline(args.episodes, seed_noise=args.seed, bundle=tmp_bundle)
        vals = [_get_nested(r['metrics'], args.metric) for r in recs]
        vals = [v for v in vals if v is not None and v == v]
        results[label] = float(np.mean(vals)) if vals else None
        print(f"  {label:8s} ({args.param}×{factor:.2f}): {args.metric} = {results[label]}")

    # Robustness verdict
    nom = results.get('nominal')
    print(f"\n=== PERTURBATION: {args.param} ±{args.pct}%  →  {args.metric} ===")
    if nom:
        spread = [v for v in results.values() if v is not None]
        rng = max(spread) - min(spread)
        rel = rng / abs(nom) if nom else float('inf')
        verdict = 'ROBUST (survives)' if rel < 0.3 else 'FRAGILE (knife-edge)'
        print(f"  nominal={nom:.4f}  range={rng:.4f}  relative={rel:.1%}  → {verdict}")
    else:
        print("  nominal metric unavailable")


def cmd_discontinuity(args):
    """Concatenate N episodes at a FIXED dw into one long net_activity trace,
    then measure burst / IBI structure (operational def in VALIDATION_TARGETS.md).

    A guard window after each episode boundary is excluded (g_* reset to 0 there).
    """
    t0 = time.time()
    ckpt = ROOT / 'output' / '_phase0_tmp' / 'ckpt_disc'
    ckpt.mkdir(parents=True, exist_ok=True)
    runner = EngineRunner(bundle_dir=str(BUNDLE), engine_bin=str(ENGINE),
                          checkpoint_dir=str(ckpt), seed_noise=args.seed,
                          seed_env=args.seed)
    N_total = int(sum(p['N'] for p in runner.net_cfg['populations']))
    T = int(runner.net_cfg['T_steps'])
    DT_MS = float(runner.net_cfg['dt_ms'])
    ep_s = T * DT_MS / 1000.0
    print(f"  N_total={N_total} neurons, episode={ep_s:.1f}s ({T} steps), "
          f"dw={args.dw}, episodes={args.episodes} -> {args.episodes*ep_s:.0f}s trace")

    dsv = DevStateVector(initial_dw=args.dw)
    dsv.dev_age = args.dw                           # FIXED dw (stationary)
    dsv.gaba_shifts = {r: 0.0 for r in dsv.regions}
    dsv.ei_bias_delta_pA = {r: 0.0 for r in dsv.regions}
    dsv.lr_gate_multiplier = 1.0
    state = dsv.get_state()
    # Experimental knob: override E_GABA for ALL regions (mechanism #1 test).
    # Natural dw-20 value is ~-45 mV (above VT=-50 => depolarizing/excitatory).
    # Forcing below VT (e.g. -60) makes GABA hyperpolarizing/inhibitory.
    if args.egaba is not None:
        state['egaba_mV'] = {r: float(args.egaba) for r in state.get('egaba_mV', {})}

    mech = (f"egaba={args.egaba if args.egaba is not None else 'natural'} "
            f"homeo={'OFF' if args.no_homeo else 'on'} "
            f"afferents={'OFF' if args.no_afferents else 'on'} "
            f"ou={'OFF' if args.no_ou else 'on'} "
            f"bg={'OFF' if args.no_bg else 'on'} "
            f"scaling={'OFF' if args.no_scaling else 'on'}")
    print(f"  mechanisms: {mech}")

    if args.single:
        # SINGLE long continuous run: one episode of episodes*ep_s seconds, NO
        # boundaries → no g-reset artifact and no re-birth. Populations are born once
        # (elapsed_abs_s=0, natural t_bir within the run). This is the cleanest trace.
        T_long = args.episodes * T
        runner.net_cfg['T_steps'] = T_long
        print(f"  SINGLE long run: {T_long} steps = {T_long*DT_MS/1000:.0f}s, no boundaries")
        ep_dir = ROOT / 'output' / '_phase0_tmp' / 'disc_single'
        out = runner.run_episode(dev_state=state, episode_dir=str(ep_dir),
                                 carry_state=None, prune_every=999,
                                 disable_scaling=args.no_scaling, elapsed_abs_s=0.0,
                                 disable_homeo=args.no_homeo,
                                 gate_afferents=args.no_afferents,
                                 disable_ou=args.no_ou, disable_bg=args.no_bg)
        na_path = Path(ep_dir) / 'output' / 'net_activity.npy'
        if not na_path.exists():
            print("  single run: net_activity.npy missing — abort"); return
        full = np.load(na_path)
        boundaries = []
        print(f"  single run done: spikes={int(full.sum())} ({time.time()-t0:.0f}s)")
    else:
        carry = None
        traces, boundaries, cum = [], [], 0
        for ep in range(args.episodes):
            ep_dir = ROOT / 'output' / '_phase0_tmp' / f'disc_ep{ep}'
            # Advance the absolute sim clock so populations are born ONCE (at their true
            # t_bir) instead of being re-born every fixed-dw episode (8.7 s artifact).
            out = runner.run_episode(dev_state=state, episode_dir=str(ep_dir),
                                     carry_state=carry, prune_every=999,
                                     disable_scaling=args.no_scaling,
                                     elapsed_abs_s=ep * ep_s,
                                     disable_homeo=args.no_homeo,
                                     gate_afferents=args.no_afferents,
                                     disable_ou=args.no_ou, disable_bg=args.no_bg)
            carry = {'pop_states': out['pop_states']}
            na_path = Path(ep_dir) / 'output' / 'net_activity.npy'
            if not na_path.exists():
                print(f"  ep{ep}: net_activity.npy missing — abort"); return
            na = np.load(na_path)
            boundaries.append(cum)                  # guard cold-start & each boundary
            traces.append(na)
            cum += len(na)
            print(f"  ep{ep}: dw={args.dw} spikes={int(na.sum())} "
                  f"({time.time()-t0:.0f}s elapsed)")
        full = np.concatenate(traces)

    # Head-discard: regions are born ONCE now (at their absolute t_bir). Drop the
    # initial transient up to the last scheduled birth (+ settle) so the IBI is
    # measured on the fully-grown, steady-state network only.
    birth_abs_s = {r: ((bd - 20.0) * 168.0) / runner.DEV
                   for r, bd in runner.region_birth_dw.items()}
    last_birth_s = max(birth_abs_s.values())
    if args.head_discard is not None:
        head_s = float(args.head_discard)
    else:
        head_s = min(last_birth_s + 5.0, 0.5 * args.episodes * ep_s)
    print(f"  region births (abs s): "
          f"{ {r: round(v,1) for r,v in sorted(birth_abs_s.items(), key=lambda x:x[1])} }")
    print(f"  head-discard: first {head_s:.1f}s (last birth at {last_birth_s:.1f}s)")
    res = analyze_discontinuity(full, n_neurons=N_total,
                                boundary_steps=boundaries, guard_ms=args.guard,
                                head_discard_ms=head_s * 1000.0)

    out_dir = ROOT / 'output' / 'phase0_discontinuity'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw concatenated trace so it can be re-analysed (threshold sweeps etc.)
    # without re-running the engine. Boundaries saved alongside.
    np.save(out_dir / f'trace_dw{args.dw}_{args.tag}.npy', full.astype(np.int32))
    np.save(out_dir / f'boundaries_dw{args.dw}_{args.tag}.npy', np.array(boundaries))
    out_path = out_dir / f'ibi_dw{args.dw}_{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump({'dw': args.dw, 'seed': args.seed, 'episodes': args.episodes,
                   'N_total': N_total,
                   'mechanisms': {'egaba': args.egaba, 'homeo_off': args.no_homeo,
                                  'afferents_off': args.no_afferents,
                                  'ou_off': args.no_ou, 'bg_off': args.no_bg,
                                  'scaling_off': args.no_scaling, 'single': args.single},
                   'head_discard_s': head_s, 'result': res}, f, indent=2)

    print(f"\n=== DISCONTINUITY @ dw{args.dw} (tag={args.tag}) ===")
    print(f"  bursts detected:  {res['n_bursts']}")
    print(f"  median IBI:       {res['median_ibi_s']} s  (IQR {res['iqr_ibi_s']} s)")
    print(f"  median burst dur: {res['median_burst_dur_s']} s")
    print(f"  mean rate:        {res['mean_rate_hz']} Hz/neuron")
    print(f"  thresholds:       {res['thresholds']}")
    m = res['median_ibi_s']
    if m != m:   # NaN
        verdict = "QUASI-CONTINUOUS (no separable bursts) — baseline as predicted"
    elif m < 0.3:
        verdict = "QUASI-CONTINUOUS (median IBI < 0.3 s) — fails tracé discontinu"
    elif m >= 3.0:
        verdict = "DISCONTINUOUS (median IBI >= 3 s) — matches tracé discontinu"
    else:
        verdict = f"PARTIAL (0.3 s < IBI < 3 s) — regime shifted, not yet discontinu"
    print(f"  VERDICT (pre-registered): {verdict}")
    print(f"  saved: {out_path}  ({time.time()-t0:.0f}s total)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    fp = sub.add_parser('fingerprint')
    fp.add_argument('--episodes', type=int, default=3)
    fp.add_argument('--seed', type=int, default=137)
    fp.add_argument('--tag', type=str, default='ref')
    fp.set_defaults(func=cmd_fingerprint)

    pt = sub.add_parser('perturb')
    pt.add_argument('--param', type=str, required=True, help='per-pop param, e.g. ge0')
    pt.add_argument('--pct', type=float, default=15.0)
    pt.add_argument('--metric', type=str, default='criticality.mr.m')
    pt.add_argument('--episodes', type=int, default=2)
    pt.add_argument('--seed', type=int, default=137)
    pt.set_defaults(func=cmd_perturb)

    dc = sub.add_parser('discontinuity')
    dc.add_argument('--dw', type=float, default=20.0)
    dc.add_argument('--episodes', type=int, default=10, help='N episodes concatenated')
    dc.add_argument('--seed', type=int, default=137)
    dc.add_argument('--guard', type=float, default=100.0, help='ms discarded after each boundary')
    dc.add_argument('--head-discard', type=float, default=None,
                    help='seconds discarded at trace start (birth transient); auto if unset')
    dc.add_argument('--tag', type=str, default='baseline')
    dc.add_argument('--egaba', type=float, default=None,
                    help='override E_GABA (mV) all regions; mechanism #1 test (e.g. -60)')
    dc.add_argument('--no-scaling', action='store_true',
                    help='disable between-episode synaptic scaling (homeostasis control)')
    dc.add_argument('--single', action='store_true',
                    help='one long continuous run (no boundaries/g-reset/re-birth) instead of concatenation')
    dc.add_argument('--no-homeo', action='store_true',
                    help='disable within-episode I_homeo current (mechanism #3 control)')
    dc.add_argument('--no-afferents', action='store_true',
                    help='zero external afferent drive (mechanism #2 control)')
    dc.add_argument('--no-ou', action='store_true',
                    help='zero correlated OU background conductance (mechanism #4 control)')
    dc.add_argument('--no-bg', action='store_true',
                    help='zero engine tonic background (ge0/gi0 DC + sig_e/sig_i noise) — isolates recurrent vs imposed floor')
    dc.set_defaults(func=cmd_discontinuity)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
