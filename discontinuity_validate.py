#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discontinuity_validate.py — KNOWN-ANSWER validation of the IBI/burst instrument
(protocol Q3 + Q4), beyond the basic self-test in discontinuity.py.

IBI is now the LOAD-BEARING validation target (literature: 6–30 s by PMA; Cherian
Table 1: 27–28 wk = 10–30 s). Before trusting any model IBI we must prove the
detector:
  (A) recovers a KNOWN IBI across the literature range (2–30 s),
  (B) is ROBUST — recovered IBI does not swing with the arbitrary bin size /
      threshold factors (Q4); if it does, the number is an artifact,
  (C) recovers the MEDIAN of a JITTERED (non-periodic) IBI distribution (real
      bursts are irregular),
  (D) still works on REALISTIC burst shapes (fast rise + exponential decay, not
      square pulses),
  (E) recovers burst DURATION (hundreds of ms target),
  (F) does NOT report spurious seconds-scale bursts on dense/async or
      non-stationary (drifting) traces — the regime that produced bursts=0 in the
      engine (that must be a CORRECT negative, not a detector failure).

Run: python discontinuity_validate.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_engine.discontinuity import analyze_discontinuity, population_rate, detect_bursts, ibi_stats

DT = 0.1
N = 15000


def synth(ibi_s, burst_s=0.3, in_burst_hz=25.0, base_hz=0.1, total_s=180.0,
          jitter_frac=0.0, shape='square', rng=None):
    """Per-step net_activity with known burst structure.
    shape='square': flat in-burst rate. shape='realistic': fast rise + exp decay.
    jitter_frac: IBIs drawn uniformly from ibi_s*(1±jitter_frac)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_steps = int(total_s * 1000.0 / DT)
    dt_s = DT / 1000.0
    trace = rng.poisson(N * base_hz * dt_s, size=n_steps).astype(np.float64)
    burst_steps = int(burst_s * 1000.0 / DT)
    true_ibis = []
    t = burst_steps
    while t + burst_steps < n_steps:
        if shape == 'square':
            lam = np.full(burst_steps, N * in_burst_hz * dt_s)
        else:  # realistic: rise over 20% then exponential decay
            tt = np.arange(burst_steps)
            rise = np.clip(tt / (0.2 * burst_steps), 0, 1)
            decay = np.exp(-(tt - 0.2 * burst_steps).clip(0) / (0.4 * burst_steps))
            env = rise * decay
            lam = N * in_burst_hz * dt_s * env / max(env.max(), 1e-9)
        trace[t:t + burst_steps] = rng.poisson(lam)
        # next burst after IBI (jittered)
        this_ibi = ibi_s * (1.0 + (rng.random() * 2 - 1) * jitter_frac)
        true_ibis.append(this_ibi)
        t += burst_steps + int(this_ibi * 1000.0 / DT)
    return trace, np.array(true_ibis[:-1]) if len(true_ibis) > 1 else np.array(true_ibis)


def synth_dense_drift(total_s=40.0, start_hz=2.0, end_hz=40.0, rng=None):
    """Dense, NON-STATIONARY async trace (no discrete bursts) — the engine failure
    mode. Detector MUST report ~0 discrete seconds-scale bursts (correct negative)."""
    if rng is None:
        rng = np.random.default_rng(1)
    n_steps = int(total_s * 1000.0 / DT)
    dt_s = DT / 1000.0
    rate = np.linspace(start_hz, end_hz, n_steps)
    return rng.poisson(N * rate * dt_s).astype(np.float64)


def measure(trace, **kw):
    return analyze_discontinuity(trace, n_neurons=N, **kw)


if __name__ == '__main__':
    np.seterr(all='ignore')
    print("="*74)
    print("A. IBI RECOVERY across literature range (2–30 s), square bursts")
    print(f"  {'true_IBI':>8} {'det_IBI':>8} {'n_burst':>8} {'burst_dur':>9} {'err%':>6} {'verdict':>7}")
    okA = True
    for ibi in (2.0, 5.0, 10.0, 20.0, 30.0):
        tr, ti = synth(ibi, total_s=max(180, ibi*12), rng=np.random.default_rng(int(ibi)))
        r = measure(tr)
        det = r['median_ibi_s']; err = abs(det - ibi)/ibi*100 if det==det else 999
        good = err < 12
        okA = okA and good
        print(f"  {ibi:8.1f} {det:8.2f} {r['n_bursts']:8d} {r['median_burst_dur_s']:9.3f} "
              f"{err:6.1f} {'PASS' if good else 'FAIL':>7}")

    print("="*74)
    print("B. ROBUSTNESS (Q4): fixed true IBI=10s, vary bin_ms and on_factor")
    print(f"  {'bin_ms':>7} {'on_f':>5} {'det_IBI':>8} {'n_burst':>8}")
    tr, _ = synth(10.0, total_s=180, rng=np.random.default_rng(7))
    detsB = []
    for bm in (2.0, 5.0, 10.0, 20.0):
        for onf in (2.5, 3.0, 4.0):
            R, valid, _ = population_rate(tr, N, bin_ms=bm)
            bursts, info = detect_bursts(R, valid, bin_ms=bm, on_factor=onf)
            st = ibi_stats(bursts, bin_ms=bm)
            detsB.append(st['median_ibi_s'])
            print(f"  {bm:7.1f} {onf:5.1f} {st['median_ibi_s']:8.2f} {st['n_bursts']:8d}")
    detsB = [d for d in detsB if d == d]
    spread = (max(detsB)-min(detsB))/np.median(detsB)*100 if detsB else 999
    okB = spread < 20
    print(f"  -> spread across params = {spread:.1f}%  {'PASS (robust)' if okB else 'FAIL (artifact)'}")

    print("="*74)
    print("C. JITTERED IBI (non-periodic): recover the MEDIAN")
    okC = True
    for ibi, jit in ((10.0, 0.4), (5.0, 0.5)):
        tr, ti = synth(ibi, jitter_frac=jit, total_s=300, rng=np.random.default_rng(int(ibi*3)))
        r = measure(tr)
        true_med = float(np.median(ti)) if len(ti) else ibi
        det = r['median_ibi_s']; err = abs(det-true_med)/true_med*100 if det==det else 999
        good = err < 15; okC = okC and good
        print(f"  IBI~{ibi}±{int(jit*100)}%: true_median={true_med:.2f} det={det:.2f} "
              f"err={err:.1f}% {'PASS' if good else 'FAIL'}")

    print("="*74)
    print("D. REALISTIC burst shape (rise+exp decay), true IBI=10s")
    tr, ti = synth(10.0, shape='realistic', total_s=180, rng=np.random.default_rng(9))
    r = measure(tr)
    det = r['median_ibi_s']; errD = abs(det-10.0)/10.0*100 if det==det else 999
    okD = errD < 15
    print(f"  det_IBI={det:.2f} (true 10) err={errD:.1f}% n_bursts={r['n_bursts']} "
          f"burst_dur={r['median_burst_dur_s']:.3f}  {'PASS' if okD else 'FAIL'}")

    print("="*74)
    print("E. BURST DURATION recovery (target hundreds of ms)")
    okE = True
    for bd in (0.2, 0.5, 1.0):
        tr, _ = synth(8.0, burst_s=bd, total_s=160, rng=np.random.default_rng(int(bd*100)))
        r = measure(tr)
        det = r['median_burst_dur_s']; err = abs(det-bd)/bd*100 if det==det else 999
        good = err < 40; okE = okE and good   # edges fuzzy → looser gate
        print(f"  true_dur={bd}s det={det:.3f}s err={err:.1f}% {'PASS' if good else 'FAIL'}")

    print("="*74)
    print("F. NEGATIVE controls: dense-async drift + continuous must give ~0 bursts")
    trF = synth_dense_drift(rng=np.random.default_rng(11))
    rF = measure(trF)
    okF1 = rF['n_bursts'] <= 2
    print(f"  dense-drift (ramp 2->40Hz): n_bursts={rF['n_bursts']} med_IBI={rF['median_ibi_s']} "
          f"{'PASS (no spurious)' if okF1 else 'FAIL (spurious bursts)'}")
    trG, _ = synth(0.0, base_hz=8.0, total_s=60, rng=np.random.default_rng(12))  # ~continuous
    # continuous: make a flat trace
    trC = np.random.default_rng(13).poisson(N*8.0*DT/1000.0, size=int(60*1000/DT)).astype(float)
    rC = measure(trC)
    okF2 = rC['n_bursts'] <= 1
    print(f"  continuous 8Hz: n_bursts={rC['n_bursts']} {'PASS' if okF2 else 'FAIL'}")

    print("="*74)
    allok = okA and okB and okC and okD and okE and okF1 and okF2
    print("SUMMARY:")
    print(f"  A IBI recovery 2–30s:        {'PASS' if okA else 'FAIL'}")
    print(f"  B robustness to bin/thresh:  {'PASS' if okB else 'FAIL'}")
    print(f"  C jittered-IBI median:       {'PASS' if okC else 'FAIL'}")
    print(f"  D realistic burst shape:     {'PASS' if okD else 'FAIL'}")
    print(f"  E burst-duration recovery:   {'PASS' if okE else 'FAIL'}")
    print(f"  F negative controls:         {'PASS' if (okF1 and okF2) else 'FAIL'}")
    print(f"  => IBI INSTRUMENT {'TRUSTWORTHY' if allok else 'NEEDS FIX before use'}")
    print("="*74)
