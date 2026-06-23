"""
run_trajectory.py — Full developmental trajectory runner (CUDA engine, Windows native)

Runs progressive trajectory from start_dw to end_dw using sim_multiregion.exe.
Modes:
  baseline — normal development
  vpa      — Valproate 100uM (KCC2_Inhibition, exposure 24-40 DW)
  asd      — ASD moderate severity=0.6 (EI imbalance, LR deficit, GABA delay)

Usage:
  python run_trajectory.py [--mode baseline] [--start 20.0] [--end 52.0] [--delta 1.0]

Output:
  output/trajectory_<mode>/
    trajectory.json      — full KPI log
    rates_ep<N>.npy      — raw rates per episode
  cuda_adam/PROGRESS.md  — appended at completion
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT    = Path(__file__).parent
BUNDLE  = ROOT / 'reference' / 'multiregion'
ENGINE  = ROOT / 'cuda' / 'build' / 'sim_multiregion.exe'
sys.path.insert(0, str(ROOT))

from cuda_engine.dev_state import DevStateVector
from cuda_engine.engine_runner import EngineRunner

# ─── VPA shifts (from valproate_100uM.yaml) ──────────────────────────────────
VPA_GABA_SHIFTS = {
    'BS': 3.5, 'TH': 3.5, 'SP': 4.0,
    'P1': 5.0, 'P2': 4.5, 'A1': 6.0, 'A2': 5.5, 'M1': 4.0,
}
VPA_START_DW = 24.0
VPA_END_DW   = 40.0

# ─── ASD moderate severity=0.6 ───────────────────────────────────────────────
ASD_SEVERITY  = 0.6
ASD_GABA_REGIONS = ['P1', 'P2', 'A1', 'A2']
ASD_EI_REGIONS   = ['A1', 'A2', 'P2']
ASD_GABA_SHIFT   = ASD_SEVERITY * 5.0          # +3.0 DW
ASD_EI_GAIN_PA   = ASD_SEVERITY * 20.0         # +12 pA to E bias
ASD_LR_MULT      = max(0.0, 1.0 - 0.4 * ASD_SEVERITY)  # 0.76


# VPA via KCC2 mechanistic model: inhibits KCC2 upregulation during exposure
VPA_KCC2_RATE_FACTOR = 0.35   # ~65% inhibition of KCC2 upregulation


def apply_intervention(dsv: DevStateVector, mode: str, dev_age: float):
    """Directly set DevStateVector shift/kcc2 fields based on mode."""
    regions = dsv.regions
    # Reset all
    dsv.gaba_shifts      = {r: 0.0 for r in regions}
    dsv.ei_bias_delta_pA = {r: 0.0 for r in regions}
    dsv.lr_gate_multiplier = 1.0
    dsv.kcc2_rate_factor = 1.0  # normal KCC2 upregulation

    if mode == 'baseline':
        pass

    elif mode == 'vpa':
        if VPA_START_DW <= dev_age <= VPA_END_DW:
            for r in regions:
                dsv.gaba_shifts[r] = float(VPA_GABA_SHIFTS.get(r, 4.5))

    elif mode == 'vpa_kcc2':
        # Mechanistic VPA: slows KCC2 upregulation during exposure
        if VPA_START_DW <= dev_age <= VPA_END_DW:
            dsv.kcc2_rate_factor = VPA_KCC2_RATE_FACTOR

    elif mode == 'asd':
        for r in ASD_GABA_REGIONS:
            if r in regions:
                dsv.gaba_shifts[r] = ASD_GABA_SHIFT
        for r in ASD_EI_REGIONS:
            if r in regions:
                dsv.ei_bias_delta_pA[r] = ASD_EI_GAIN_PA
        dsv.lr_gate_multiplier = ASD_LR_MULT


def _print_metrics(m: dict):
    if not m:
        return
    # E/I ratio
    ei = m.get('ei_ratio', {})
    if ei:
        parts = [f"{r}={v:.2f}" for r, v in sorted(ei.items())]
        print(f"    EI-ratio: {' '.join(parts)}")
    # Burst (only regions with at least 1 burst)
    burst = {r: b for r, b in m.get('burst', {}).items() if b.get('count', 0) > 0}
    if burst:
        parts = [f"{r}:{b['count']}x{b['peak_rate_hz']:.0f}Hz({b['fraction']*100:.0f}%)"
                 for r, b in sorted(burst.items())]
        print(f"    Bursts:   {' '.join(parts)}")
    sync = m.get('synchrony')
    if sync is not None:
        print(f"    Sync:     {sync:.4f}")
    crit = m.get('criticality')
    if crit:
        mr = crit.get('mr') or {}
        mval = mr.get('m')
        if mval is not None and mval == mval:  # not NaN
            r2 = mr.get('mr_quality_r2', float('nan'))
            print(f"    Critical: m={mval:.4f} ({mr.get('mr_regime','?')}, "
                  f"fit_R2={r2:.2f})  [robust MR-estimator]")


def fmt_rates(r: dict) -> str:
    parts = []
    for region in ['BS', 'TH', 'SP', 'P1', 'P2', 'A1', 'A2', 'M1']:
        if region in r:
            e   = r[region].get('E', 0.0)
            pv  = r[region].get('I', 0.0)
            sst = r[region].get('SST')
            s = f"{region} E={e:.2f} PV={pv:.1f}"
            if sst is not None:
                s += f" SST={sst:.1f}"
            parts.append(s)
    return '  '.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',   default='baseline', choices=['baseline', 'vpa', 'vpa_kcc2', 'asd'])
    p.add_argument('--start',  type=float, default=20.0)
    p.add_argument('--end',    type=float, default=52.0)
    p.add_argument('--delta',  type=float, default=1.0)
    p.add_argument('--bundle', type=str, default=None,
                   help='Override bundle dir (default: reference/multiregion)')
    p.add_argument('--tag',    type=str, default='',
                   help='Extra tag appended to output directory name')
    args = p.parse_args()

    mode     = args.mode
    start_dw = args.start
    end_dw   = args.end
    delta_dw = args.delta
    tag      = f'_{args.tag}' if args.tag else ''

    bundle_dir = Path(args.bundle) if args.bundle else ROOT / 'reference' / 'multiregion'

    n_episodes = max(1, round((end_dw - start_dw) / delta_dw))

    out_dir = ROOT / 'output' / f'trajectory_{mode}{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"  ADAM TRAJECTORY  mode={mode.upper()} | dw {start_dw} -> {end_dw} | "
          f"delta_dw={delta_dw} | {n_episodes} episodes")
    print(f"  Bundle: {bundle_dir}")
    print(f"  Engine: {ENGINE}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    runner = EngineRunner(
        bundle_dir=str(bundle_dir),
        engine_bin=str(ENGINE),
        checkpoint_dir=str(ckpt_dir),
    )

    use_kcc2_mech = mode == 'vpa_kcc2'
    dsv = DevStateVector(initial_dw=start_dw, use_kcc2_mechanistic=use_kcc2_mech)

    trajectory = []
    carry_state = None
    t_run_start = time.time()

    # Checkpoint DW values for detailed KPI logging
    CHECKPOINT_DWS = {28, 32, 36, 40, 44, 52}

    for ep in range(n_episodes):
        current_dw = start_dw + ep * delta_dw
        dsv.dev_age = current_dw

        apply_intervention(dsv, mode, current_dw)
        dev_state = dsv.get_state()

        egaba_p1 = dev_state['egaba_mV'].get('P1', -999)
        lr = dev_state['lr_gate']
        mg = dev_state['mg_now']
        nr = dev_state['nmda_ratio']

        print(f"\n[Ep {ep+1:3d}/{n_episodes}] dw={current_dw:.1f}  "
              f"E_GABA(P1)={egaba_p1:.1f}mV  lr_gate={lr:.3f}  Mg={mg:.2f}mM  nmda_r={nr:.3f}")

        if mode == 'vpa' and VPA_START_DW <= current_dw <= VPA_END_DW:
            print(f"  [VPA active] gaba_shift P1=+{dsv.gaba_shifts.get('P1',0):.1f}DW")
        if mode == 'vpa_kcc2' and VPA_START_DW <= current_dw <= VPA_END_DW:
            kcc2_p1 = dsv.kcc2_level.get('P1', 0)
            print(f"  [VPA-KCC2] rate_factor={dsv.kcc2_rate_factor:.2f}  KCC2(P1)={kcc2_p1:.3f}")
        if mode == 'asd':
            print(f"  [ASD] gaba_shift P1=+{dsv.gaba_shifts.get('P1',0):.1f}DW  "
                  f"ei_bias A1={dsv.ei_bias_delta_pA.get('A1',0):.1f}pA  "
                  f"lr_mult={dsv.lr_gate_multiplier:.2f}")

        t_ep = time.time()

        ep_out = runner.run_episode(
            dev_state=dev_state,
            carry_state=carry_state,
            prune_every=2,
        )

        wall = time.time() - t_ep
        rates_by_region = runner.compute_rates_by_region(ep_out['rates'])
        m = ep_out.get('metrics', {})

        print(f"  Wall: {wall:.1f}s | " + fmt_rates(rates_by_region))
        _print_metrics(m)

        # Save raw rates
        np.save(out_dir / f'rates_ep{ep:04d}_dw{current_dw:.1f}.npy', ep_out['rates'])

        # Build episode record
        rec = {
            'episode':    ep,
            'dw':         current_dw,
            'wall_s':     round(wall, 1),
            'egaba_mV':   {r: round(v, 2) for r, v in dev_state['egaba_mV'].items()},
            'lr_gate':    round(lr, 4),
            'mg_now':     round(mg, 4),
            'nmda_ratio': round(nr, 4),
            'rates':      {r: {t: round(v, 3) for t, v in tv.items()}
                           for r, tv in rates_by_region.items()},
            'n_pruned':   ep_out.get('n_pruned', 0),
            'scale_factors': {r: round(v, 4) for r, v in ep_out.get('scale_factors', {}).items()},
            'metrics':    ep_out.get('metrics', {}),
            'kcc2_level': {r: round(v, 4) for r, v in dev_state.get('kcc2_level', {}).items()},
        }
        trajectory.append(rec)

        # Prepare carry state for next episode
        carry_state = {'pop_states': ep_out['pop_states']}

        # Print full KPI at checkpoint DWs
        if current_dw in CHECKPOINT_DWS:
            print(f"\n  ★ CHECKPOINT dw={current_dw:.0f} — full KPI:")
            for region, rtypes in sorted(rates_by_region.items()):
                e = rtypes.get('E', 0.0)
                i = rtypes.get('I', 0.0)
                print(f"    {region:3s}: E={e:6.3f} Hz  I={i:6.3f} Hz")

        # Phase 3 (ledger J3): update activity-dependent lr_gate boost from this episode's
        # cross-region coactivation (P2/A2 correlation → faster LR growth).
        region_E = ep_result.get('region_rates_E', {})
        if hasattr(dsv, 'update_lr_gate'):
            _T = runner.net_cfg['T_steps']; _DT = runner.net_cfg['dt_ms']
            dsv.update_lr_gate(region_E, episode_duration_s=_T * _DT / 1000.0)

        # Advance DevStateVector
        dsv.step(delta_dw)

        # Flush trajectory log
        with open(out_dir / 'trajectory.json', 'w') as f:
            json.dump({'mode': mode, 'episodes': trajectory}, f, indent=2)

    total_wall = time.time() - t_run_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  TRAJECTORY COMPLETE — {mode.upper()}")
    print(f"  Total wall time: {total_wall/60:.1f} min")
    print(f"  {n_episodes} episodes × {total_wall/n_episodes:.1f}s avg")
    print("=" * 70)

    print("\n  KPI summary at checkpoint DWs:")
    hdr = f"  {'DW':>4}  {'BS_E':>6} {'BS_I':>6}  {'TH_E':>6} {'TH_I':>6}  " \
          f"{'P1_E':>6} {'P1_I':>6}  {'A1_E':>6} {'A1_I':>6}  " \
          f"{'E_GABA_P1':>10}  {'lr_gate':>7}"
    print(hdr)
    for rec in trajectory:
        if rec['dw'] in CHECKPOINT_DWS or rec['episode'] == 0 or rec['episode'] == n_episodes-1:
            r = rec['rates']
            bs_e  = r.get('BS', {}).get('E', 0.0)
            bs_i  = r.get('BS', {}).get('I', 0.0)
            th_e  = r.get('TH', {}).get('E', 0.0)
            th_i  = r.get('TH', {}).get('I', 0.0)
            p1_e  = r.get('P1', {}).get('E', 0.0)
            p1_i  = r.get('P1', {}).get('I', 0.0)
            a1_e  = r.get('A1', {}).get('E', 0.0)
            a1_i  = r.get('A1', {}).get('I', 0.0)
            eg_p1 = rec['egaba_mV'].get('P1', 0.0)
            lr    = rec['lr_gate']
            print(f"  {rec['dw']:>4.0f}  {bs_e:>6.3f} {bs_i:>6.2f}  "
                  f"{th_e:>6.3f} {th_i:>6.2f}  "
                  f"{p1_e:>6.3f} {p1_i:>6.2f}  "
                  f"{a1_e:>6.3f} {a1_i:>6.2f}  "
                  f"{eg_p1:>10.2f}  {lr:>7.4f}")

    # ── Append to PROGRESS.md ─────────────────────────────────────────────────
    progress_path = ROOT / 'PROGRESS.md'
    with open(progress_path, 'a') as f:
        f.write(f"\n---\n\n### Autonomous Session -- Trajectory {mode.upper()} "
                f"(dw {start_dw:.0f}->{end_dw:.0f}, delta={delta_dw})\n\n")
        f.write(f"**Wall time:** {total_wall/60:.1f} min | "
                f"**Episodes:** {n_episodes} | **Avg:** {total_wall/n_episodes:.1f}s/ep\n\n")
        f.write("| DW | BS_E | BS_I | TH_E | TH_I | P1_E | P1_I | A1_E | A1_I | "
                "E_GABA_P1 | lr_gate |\n")
        f.write("|-----|------|------|------|------|------|------|------|------|"
                "-----------|--------|\n")
        for rec in trajectory:
            if rec['dw'] in CHECKPOINT_DWS or rec['episode'] == 0 or rec['episode'] == n_episodes-1:
                r    = rec['rates']
                eg   = rec['egaba_mV'].get('P1', 0.0)
                lr   = rec['lr_gate']
                f.write(f"| {rec['dw']:.0f} "
                        f"| {r.get('BS',{}).get('E',0):.3f} "
                        f"| {r.get('BS',{}).get('I',0):.2f} "
                        f"| {r.get('TH',{}).get('E',0):.3f} "
                        f"| {r.get('TH',{}).get('I',0):.2f} "
                        f"| {r.get('P1',{}).get('E',0):.3f} "
                        f"| {r.get('P1',{}).get('I',0):.2f} "
                        f"| {r.get('A1',{}).get('E',0):.3f} "
                        f"| {r.get('A1',{}).get('I',0):.2f} "
                        f"| {eg:.2f} "
                        f"| {lr:.4f} |\n")
    print(f"\n  Results appended to PROGRESS.md")
    print(f"  Trajectory JSON: {out_dir / 'trajectory.json'}")

if __name__ == '__main__':
    main()
