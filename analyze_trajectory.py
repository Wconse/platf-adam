#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_trajectory.py — measure the developmental dynamical trajectory.

EMERGENCE TEST (no mechanism imposed): does the network self-organize the
transition from synchronized bursting (dw20, depolarizing GABA) to asynchronous-
irregular near-critical activity (dw34+, hyperpolarizing GABA), driven ONLY by the
E_GABA ramp that dev_state applies? Pre-registered target (VALIDATION_TARGETS.md):
tracé discontinu / bursting predominates < 28 wk PMA, continuous activity by ~35 wk.

For each episode we report, from the validated criticality instrument:
  - MR branching ratio m (+R2 reliability), stationarity (drift)
  - frac_empty (sparse=bursting, dense=continuous)
  - synchrony proxy (Fano of 5ms population rate; high=synchronized bursts)
  - population-burst count + median IBI
We then check whether sparsity/synchrony DECLINE as E_GABA flips — i.e. the
bursting→continuous transition emerges, and near which dw.

Usage: python analyze_trajectory.py [tag]
"""
import numpy as np, glob, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_engine.criticality import compute_criticality

TAG = sys.argv[1] if len(sys.argv) > 1 else 'traj_gabaflip'
root = f'output/runs/{TAG}/seed0'
hist = json.load(open(f'{root}/history.json'))

print(f"Developmental trajectory: {TAG}")
print(f"{'dw':>5} {'E_GABA':>7} {'SP_E':>6} {'SP_I':>6} {'mHz':>6} {'m':>7} {'R2':>5} "
      f"{'drift':>6} {'fracE':>6} {'Fano':>6} {'nburst':>6} {'medIBI':>7} {'regime':>14}")

rows = []
for ep_idx, h in enumerate(hist):
    dw = h['dw']
    epdir = f'{root}/ep{ep_idx:04d}/output/net_activity.npy'
    fs = glob.glob(epdir)
    if not fs:
        continue
    net = np.load(fs[0]).astype(np.float64)
    c = compute_criticality(net)
    mr = c.get('mr', {}); st = c.get('stationarity', {})
    # synchrony: Fano of 5ms population rate
    b = 50; nb = len(net) // b
    R = net[:nb*b].reshape(nb, b).sum(1)
    fano = R.var() / max(R.mean(), 1e-6)
    # population bursts (10ms bins, 25% peak threshold)
    bw = 100; nb2 = len(net)//bw; R2 = net[:nb2*bw].reshape(nb2, bw).sum(1)
    pk = R2.max(); above = R2 > 0.25*pk
    onsets = np.where(np.diff(above.astype(int)) == 1)[0]
    ibis = np.diff(onsets)*10/1000.0 if len(onsets) > 1 else np.array([])
    med_ibi = float(np.median(ibis)) if len(ibis) else float('nan')
    egaba_sp = h.get('region_hz_E') and None
    # E_GABA isn't in history; recompute label from dw via SP firing context
    spe = h['region_hz_E'].get('SP', 0); spi = h['region_hz_I'].get('SP', 0)
    reg = mr.get('mr_regime', '?')[:14]
    print(f"{dw:5.1f} {'':>7} {spe:6.2f} {spi:6.2f} {h['mean_hz']:6.3f} "
          f"{str(mr.get('m','nan')):>7} {str(mr.get('mr_quality_r2','nan')):>5} "
          f"{str(st.get('drift_ratio','nan')):>6} {str(c.get('frac_empty_bins','nan')):>6} "
          f"{fano:6.1f} {len(onsets):6d} {med_ibi:7.2f} {reg:>14}")
    rows.append(dict(dw=dw, frac_empty=c.get('frac_empty_bins', np.nan), fano=fano,
                     m=mr.get('m', np.nan), spe=spe, spi=spi, med_ibi=med_ibi))

# ── Emergence verdict: does bursting->continuous transition emerge? ──
print("\nTRANSITION ANALYSIS (bursting=sparse+high-Fano; continuous=dense+low-Fano):")
if len(rows) >= 3:
    fe = np.array([r['frac_empty'] if r['frac_empty'] is not None else np.nan for r in rows], float)
    fa = np.array([r['fano'] for r in rows], float)
    dws = np.array([r['dw'] for r in rows], float)
    early = rows[0]; late = rows[-1]
    print(f"  dw{early['dw']:.0f}: frac_empty={early['frac_empty']} Fano={early['fano']:.1f}  "
          f"-> dw{late['dw']:.0f}: frac_empty={late['frac_empty']} Fano={late['fano']:.1f}")
    # find dw where Fano drops below half its early value (transition midpoint)
    if np.isfinite(fa[0]) and fa[0] > 0:
        half = fa[0] * 0.5
        below = np.where(fa < half)[0]
        if len(below):
            print(f"  Fano halves at dw~{dws[below[0]]:.1f} (pre-registered target: ~dw28-34)")
        else:
            print(f"  Fano never halves — bursting persists (transition NOT observed)")
    print("  CAUTION: connectivity is hand-tuned (confound). A real emergence claim")
    print("  needs this to hold across seeds and not depend on the specific tuning.")
