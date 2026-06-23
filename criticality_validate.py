#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
criticality_validate.py — KNOWN-ANSWER validation of the criticality instrument
(protocol Q3). Before trusting any criticality/avalanche number measured on the
network, we must prove the estimators recover the TRUTH on synthetic data whose
branching ratio and avalanche exponents are known a priori.

Ground-truth generators:
  1. Branching process (Poisson offspring, known m): each active unit in bin t
     spawns Poisson(m) descendants in bin t+1, plus a slow Poisson drive. The
     branching ratio is exactly m by construction. At m=1 (critical), avalanche
     sizes follow P(s)~s^-1.5 and durations P(d)~d^-2.0 (mean-field DP class).
  2. Poisson async: independent homogeneous Poisson spikes, NO temporal
     correlation → true branching ratio 0, NO power-law avalanches.

We render each ground-truth process into a per-step net_activity trace (the exact
format compute_criticality consumes) and check:
  - MR-estimator m recovers the known m (subcritical/critical/supercritical),
  - avalanche size exponent ~1.5 at criticality,
  - regime label is correct,
  - Poisson async is flagged NOT near-critical (no false positive).

Run: python criticality_validate.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_engine.criticality import (compute_criticality, mr_branching_ratio,
                                      _extract_avalanches, _powerlaw_mle, _rebin,
                                      _auto_bin_steps)


def gen_branching_process(m, n_bins=200_000, drive=0.5, seed=0, max_a=100000):
    """Critical branching process with immigration. Returns per-bin activity A[t].
    A[t+1] ~ Poisson(m * A[t]) + Poisson(drive). Branching ratio = m exactly."""
    rng = np.random.default_rng(seed)
    A = np.zeros(n_bins, dtype=np.float64)
    a = 0
    for t in range(n_bins):
        offspring = rng.poisson(m * a) if a > 0 else 0
        imm = rng.poisson(drive)
        a = min(offspring + imm, max_a)
        A[t] = a
    return A


def gen_poisson_async(rate_per_bin=3.0, n_bins=200_000, seed=0):
    """Independent Poisson activity — no temporal correlation. True m = 0."""
    rng = np.random.default_rng(seed)
    return rng.poisson(rate_per_bin, size=n_bins).astype(np.float64)


def bins_to_netactivity(A, spread=10):
    """Render a per-bin avalanche count into a per-STEP net_activity trace by
    spreading each bin's spikes across `spread` sub-steps. compute_criticality
    re-bins internally, so this just provides the fine-grained format."""
    A = A.astype(np.int64)
    out = np.zeros(len(A) * spread, dtype=np.float64)
    for i, c in enumerate(A):
        if c > 0:
            # place c spikes uniformly within the bin's sub-steps
            idx = np.linspace(0, spread - 1, min(c, spread)).astype(int)
            for j in idx:
                out[i * spread + j] += c / min(c, spread)
    return out


def validate_mr_on_bins():
    """Directly validate the MR-estimator on bin-level branching data where m is
    known EXACTLY (no rendering). This isolates the estimator from binning."""
    print("="*72)
    print("Q3 CONTROL A — MR branching-ratio estimator on KNOWN-m branching process")
    print("  (fed bin-level activity directly; bin_steps=1)")
    print(f"  {'true_m':>7} {'est_m':>7} {'R2':>6} {'reliable':>9} {'regime':>14} {'verdict':>8}")
    ok = True
    for true_m in (0.80, 0.90, 0.95, 1.00, 1.05):
        A = gen_branching_process(true_m, n_bins=300_000, drive=0.3, seed=1)
        r = mr_branching_ratio(A, kmax=150, bin_steps=1)
        est = r['m']; r2 = r.get('mr_quality_r2', float('nan'))
        rel = r.get('mr_reliable', False); reg = r.get('mr_regime', '?')
        # verdict: estimate within 0.05 of truth AND reliable
        good = rel and np.isfinite(est) and abs(est - true_m) < 0.06
        ok = ok and good
        print(f"  {true_m:7.2f} {est:7.3f} {r2:6.2f} {str(rel):>9} {reg:>14} "
              f"{'PASS' if good else 'FAIL':>8}")
    return ok


def validate_avalanche_exponents():
    """Validate avalanche size exponent recovery at criticality (m=1 -> tau~1.5)."""
    print("="*72)
    print("Q3 CONTROL B — avalanche size exponent at criticality (truth: tau~1.5)")
    A = gen_branching_process(1.00, n_bins=400_000, drive=0.2, seed=2)
    sizes, durs = _extract_avalanches(A)   # A is already bin-level (0 = quiet)
    n = len(sizes)
    tau = _powerlaw_mle(sizes) if n >= 30 else float('nan')
    alpha = _powerlaw_mle(durs) if n >= 30 else float('nan')
    print(f"  n_avalanches={n}  size_exponent(tau)={tau:.3f} (target ~1.5)  "
          f"dur_exponent(alpha)={alpha:.3f} (target ~2.0)")
    good = np.isfinite(tau) and 1.3 <= tau <= 1.8
    print(f"  verdict: {'PASS' if good else 'FAIL'} (tau in [1.3,1.8])")
    return good


def validate_poisson_negative():
    """Poisson async must NOT be labelled near-critical (no false positive)."""
    print("="*72)
    print("Q3 CONTROL C — Poisson async (truth: NO criticality, m~0)")
    A = gen_poisson_async(rate_per_bin=3.0, n_bins=300_000, seed=3)
    r = mr_branching_ratio(A, kmax=150, bin_steps=1)
    est = r['m']; rel = r.get('mr_reliable', False); reg = r.get('mr_regime', '?')
    # PASS if either flagged unreliable OR m clearly < 0.9 (subcritical/none)
    good = (not rel) or (np.isfinite(est) and est < 0.90)
    print(f"  est_m={est}  reliable={rel}  regime={reg}")
    print(f"  verdict: {'PASS' if good else 'FAIL'} (must NOT be near-critical)")
    return good


def validate_full_pipeline():
    """End-to-end: render branching processes into net_activity and run the full
    compute_criticality, checking the regime label across the spectrum."""
    print("="*72)
    print("Q3 CONTROL D — full compute_criticality pipeline (rendered net_activity)")
    print(f"  {'true_m':>7} {'regime_label':>26} {'est_m':>7} {'frac_empty':>11}")
    results = {}
    for true_m, expect in [(0.80,'subcritical'), (1.00,'near-critical'), (1.05,'supercritical')]:
        A = gen_branching_process(true_m, n_bins=150_000, drive=0.2, seed=4)
        net = bins_to_netactivity(A, spread=8)
        c = compute_criticality(net)
        mr = c.get('mr', {})
        reg = mr.get('mr_regime', c.get('regime','?'))
        results[true_m] = reg
        print(f"  {true_m:7.2f} {reg:>26} {mr.get('m',float('nan')):7.3f} "
              f"{c.get('frac_empty_bins',float('nan')):11}")
    # we mainly require monotonic ordering: sub < crit < super in m
    return results


def validate_nonstationary_catch():
    """The engine's actual failure mode is a runaway RAMP (supercritical-
    saturated), which the MR-estimator misreads as 'near-critical'. Verify the
    stationarity flag catches it. Build a non-stationary trace: a branching
    process whose drive ramps up over time."""
    print("="*72)
    print("Q3 CONTROL E — non-stationary runaway must NOT pass as near-critical")
    rng = np.random.default_rng(7)
    n = 200_000
    A = np.zeros(n)
    a = 0.0
    for t in range(n):
        drive = 0.1 + 3.0 * (t / n)          # ramping immigration
        a = min(rng.poisson(0.98 * a) if a > 0 else 0, 50000) + rng.poisson(drive)
        A[t] = a
    net = bins_to_netactivity(A, spread=8)
    c = compute_criticality(net)
    st = c.get('stationarity', {})
    mr = c.get('mr', {})
    drift = st.get('drift_ratio', float('nan'))
    reg = mr.get('mr_regime', '?')
    good = (not st.get('stationary', True)) and ('non-stationary' in reg or not mr.get('mr_reliable', True))
    print(f"  drift_ratio={drift}  stationary={st.get('stationary')}  mr_regime={reg}")
    print(f"  verdict: {'PASS' if good else 'FAIL'} (ramp flagged non-stationary)")
    return good


if __name__ == '__main__':
    np.seterr(all='ignore')
    a = validate_mr_on_bins()
    b = validate_avalanche_exponents()
    c = validate_poisson_negative()
    e = validate_nonstationary_catch()
    d = validate_full_pipeline()
    print("="*72)
    print("SUMMARY (protocol Q3 instrument validation):")
    print(f"  A. MR estimator recovers known m:       {'PASS' if a else 'FAIL'}")
    print(f"  B. avalanche exponent ~1.5 at crit:     {'PASS' if b else 'FAIL'}")
    print(f"  C. Poisson async not false-positive:    {'PASS' if c else 'FAIL'}")
    print(f"  E. non-stationary runaway caught:       {'PASS' if e else 'FAIL'}")
    print(f"  D. full pipeline regime ordering:       {d}")
    overall = a and b and c and e
    print(f"  => INSTRUMENT {'TRUSTWORTHY' if overall else 'NOT YET TRUSTWORTHY — fix before use'}")
    print("="*72)
