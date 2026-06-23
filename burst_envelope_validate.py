#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""burst_envelope_validate.py — KNOWN-ANSWER control for the bouffée instrument.

The bouffée envelope must recover the ENVELOPE duration (1–5 s) and inter-bouffée
interval (10–30 s), NOT the within-bouffée sub-cycle duration (~40 ms). Tests:
  A. recover bouffée DURATION across 1–5 s (with sub-cycle structure inside),
  B. recover inter-bouffée IBI across 10–30 s,
  C. ROBUSTNESS (Q4): recovered dur+IBI stable across bin_ms and gap_merge_ms,
  D. the instrument must NOT collapse to the sub-cycle duration (~40 ms),
  E. negative controls: continuous-async → 0 bouffées; one long burst → 1 bouffée.

Run: python burst_envelope_validate.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_engine.burst_envelope import analyze_bouffees, population_rate, detect_bouffees

DT = 0.1
N = 15000


def synth(bouffee_s, ibi_s, n_bouffees=8, cycle_ms=140.0, sub_burst_ms=40.0,
          in_cycle_hz=30.0, base_hz=0.05, rng=None):
    """Bouffées of duration bouffee_s, each containing sub-bursts every cycle_ms
    (oscillation cycles), separated by ibi_s of near-silence. True envelope dur =
    bouffee_s, true inter-bouffée = ibi_s; sub-cycle dur = sub_burst_ms (~40 ms)."""
    if rng is None:
        rng = np.random.default_rng(0)
    dt_s = DT / 1000.0
    total_s = (bouffee_s + ibi_s) * (n_bouffees + 1)
    n_steps = int(total_s * 1000.0 / DT)
    trace = rng.poisson(N * base_hz * dt_s, size=n_steps).astype(np.float64)
    cyc = int(cycle_ms / DT); sub = int(sub_burst_ms / DT)
    t = int(ibi_s * 1000.0 / DT)
    bouffee_steps = int(bouffee_s * 1000.0 / DT)
    for _ in range(n_bouffees):
        # fill the bouffée window with sub-bursts every cycle
        u = t
        while u < t + bouffee_steps and u + sub < n_steps:
            trace[u:u + sub] = rng.poisson(N * in_cycle_hz * dt_s, size=sub)
            u += cyc
        t += bouffee_steps + int(ibi_s * 1000.0 / DT)
        if t >= n_steps:
            break
    return trace


if __name__ == '__main__':
    np.seterr(all='ignore')
    print("=" * 74)
    print("A+B. Recover bouffée DURATION (1–5 s) and inter-bouffée IBI (10–30 s)")
    print(f"  {'trueDur':>8}{'detDur':>8}{'trueIBI':>8}{'detIBI':>8}{'nB':>4}{'durErr%':>8}{'ibiErr%':>8} verdict")
    okAB = True
    for dur, ibi in ((1.0, 12.0), (2.0, 20.0), (3.0, 20.0), (5.0, 30.0)):
        tr = synth(dur, ibi, rng=np.random.default_rng(int(dur * 10 + ibi)))
        r = analyze_bouffees(tr, N, bin_ms=50, gap_merge_ms=1000, min_bouffee_ms=300)
        dd = r['median_bouffee_dur_s']; di = r['median_inter_bouffee_s']
        de = abs(dd - dur) / dur * 100 if dd == dd else 999
        ie = abs(di - ibi) / ibi * 100 if di == di else 999
        good = de < 25 and ie < 20
        okAB = okAB and good
        print(f"  {dur:8.1f}{dd:8.2f}{ibi:8.1f}{di:8.2f}{r['n_bouffees']:4d}{de:8.1f}{ie:8.1f}  {'PASS' if good else 'FAIL'}")

    print("=" * 74)
    print("C. ROBUSTNESS (Q4): true dur=2s ibi=20s, vary bin_ms and gap_merge_ms")
    tr = synth(2.0, 20.0, rng=np.random.default_rng(7))
    dds, dis = [], []
    print(f"  {'bin_ms':>7}{'gapMrg':>8}{'detDur':>8}{'detIBI':>8}{'nB':>4}")
    for bm in (25.0, 50.0, 100.0):
        for gm in (500.0, 1000.0, 2000.0):
            r = analyze_bouffees(tr, N, bin_ms=bm, gap_merge_ms=gm, min_bouffee_ms=300)
            dds.append(r['median_bouffee_dur_s']); dis.append(r['median_inter_bouffee_s'])
            print(f"  {bm:7.0f}{gm:8.0f}{r['median_bouffee_dur_s']:8.2f}{r['median_inter_bouffee_s']:8.2f}{r['n_bouffees']:4d}")
    dds = [x for x in dds if x == x]; dis = [x for x in dis if x == x]
    dspread = (max(dds) - min(dds)) / np.median(dds) * 100 if dds else 999
    ispread = (max(dis) - min(dis)) / np.median(dis) * 100 if dis else 999
    okC = dspread < 30 and ispread < 20
    print(f"  -> dur spread={dspread:.1f}%  IBI spread={ispread:.1f}%  {'PASS' if okC else 'FAIL'}")

    print("=" * 74)
    print("D. Must NOT collapse to sub-cycle (~0.04 s): detDur must be >> 0.04 s")
    tr = synth(2.0, 20.0, rng=np.random.default_rng(3))
    r = analyze_bouffees(tr, N, bin_ms=50, gap_merge_ms=1000)
    okD = r['median_bouffee_dur_s'] > 0.5  # envelope, not the 0.04 s cycle
    print(f"  detDur={r['median_bouffee_dur_s']:.2f}s (sub-cycle=0.04s)  {'PASS' if okD else 'FAIL'}")

    print("=" * 74)
    print("E. Negative controls")
    cont = np.random.default_rng(5).poisson(N * 8.0 * DT / 1000.0, size=int(120 * 1000 / DT)).astype(float)
    rc = analyze_bouffees(cont, N, bin_ms=50, gap_merge_ms=1000)
    okE1 = rc['n_bouffees'] <= 1
    print(f"  continuous-async 8Hz: n_bouffees={rc['n_bouffees']}  {'PASS' if okE1 else 'FAIL'}")
    one = synth(3.0, 20.0, n_bouffees=1, rng=np.random.default_rng(9))
    ro = analyze_bouffees(one, N, bin_ms=50, gap_merge_ms=1000)
    okE2 = ro['n_bouffees'] == 1 and abs(ro['median_bouffee_dur_s'] - 3.0) < 0.8
    print(f"  single 3s bouffée: n={ro['n_bouffees']} dur={ro['median_bouffee_dur_s']:.2f}  {'PASS' if okE2 else 'FAIL'}")

    print("=" * 74)
    allok = okAB and okC and okD and okE1 and okE2
    print(f"SUMMARY: A+B dur/IBI={'PASS' if okAB else 'FAIL'}  C robust={'PASS' if okC else 'FAIL'}  "
          f"D not-subcycle={'PASS' if okD else 'FAIL'}  E neg={'PASS' if (okE1 and okE2) else 'FAIL'}")
    print(f"  => BOUFFÉE INSTRUMENT {'TRUSTWORTHY' if allok else 'NEEDS FIX'}")
    sys.exit(0 if allok else 1)
