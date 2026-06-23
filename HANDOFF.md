# HANDOFF — operational notes + current state (for next session)

## RUN/COMMAND GOTCHAS (learned the hard way — read before running anything)

### Paths
- **bash `/tmp` ↔ Windows `E:/TEMP`.** The engine prints "Saved to E:/TEMP/...".
  In bash you can `ls /tmp/x`, but **Python `np.load` needs the Windows path `E:/TEMP/x`**,
  NOT `/tmp/x` (FileNotFoundError otherwise). This bites every time.
- Project dir has Cyrillic: `E:\Adam\Новая папка\cuda_adam`. Works, but mind encoding.
- Background task outputs land in `E:\TEMP\claude\...\tasks\<id>.output`.

### Encoding
- Console is **cp1251** (Windows-RU). Python printing non-ASCII (→, é, "tracé") → `UnicodeEncodeError`.
  **Prefix every python run with `PYTHONIOENCODING=utf-8`.** (adam.py has a `→` that crashes without it.)

### Permissions / Bash classifier
- The Bash safety classifier (a proxy model) sometimes goes down → "auto mode cannot
  determine safety" errors that block ALL non-trivial commands. Just retry; or user
  toggles permission mode off "auto" (Shift+Tab); or runs via `!`.
- **`rm -rf` is denied** by the permission classifier. Don't use it — `gen_bundle.py`
  overwrites all standard files on a clean full run, so just regenerate.
- **WebSearch is BLOCKED** by the proxy (freemodel.dev) — "Unofficial Claude Code clone".
  WebFetch works but needs exact URLs. For literature, ask the USER to run search and
  paste URLs/answers (see LITERATURE_PROVENANCE.md for the query list pattern).

### Engine build
- Rebuild: **`bash _rebuild.sh`** (incremental Ninja, ~1–2 min). Only needed for **C++**
  changes (cuda/*.cu, *.cuh). Config/param changes need NO rebuild.
- Non-standard toolchain paths (MSVC 14.51.36231 in E:/VIS STUD, SDK in E:/Windows Kits,
  CUDA 13.3). All in BUILD_WINDOWS.md. nvcc is on PATH.
- Binary: `cuda/build/sim_multiregion.exe`.

### Running the engine
- **Direct (preferred for full scale):** `./cuda/build/sim_multiregion.exe <bundle> <outdir>`.
  Fast, works at SCALE=1.0.
- **adam.py:** `PYTHONIOENCODING=utf-8 python adam.py --no-homeo --dw-start 20 --dw-end 20.5 ...`
  ⚠ **adam.py COPIES the whole bundle per episode → CHOKES at SCALE=1.0** (39.5M synapses,
  Windows symlink fails → slow copy / fail). Use the DIRECT engine for full scale.
  TODO: fix adam.py to symlink invariant files + override only small ta_*.npy.
- To get **no-homeo when running the engine DIRECTLY** (adam's --no-homeo patch is bypassed):
  set **`ADAM_NO_HOMEO=1`** at gen_bundle time (bakes I_cl=I_scl=0 into network_config).
- **"Total spikes: 0" in engine stdout is a SPARSE-SPIKE-LOG ARTIFACT.** Real activity is in
  `cuda_rates.npy` (per-neuron mean) and `net_activity.npy` (global per-step). Don't be fooled.
- A partial/interrupted gen at high SCALE → mixed-size files → engine **SEGFAULT**. Fix:
  regenerate cleanly (full gen overwrites consistently).

### Bundle generation
- `python reference/gen_bundle.py <T_MS> <SCALE> <suffix>` e.g. `30000 1.0 multiregion_v2`.
  **adam.py reads `reference/multiregion_v2`** (that suffix).
- Spatial gen is grid-efficient now (SCALE=1.0 in seconds).
- Env knobs (all read by gen_bundle): `ADAM_IMMATURE`(def 1), `ADAM_NO_HOMEO`,
  `ADAM_SPATIAL`(def 1), `ADAM_IMM_BI/IMM_BG/IMM_BI_I`, `ADAM_GGAP/GAP_DEG`,
  `ADAM_REC_GAIN/PEE_GAIN/PEI_GAIN/WEI_GAIN/WIE_GAIN/IBI_GAIN/GI0_SCALE`,
  `ADAM_RTGT_E/RTGT_I/TAU_HO_MS`, `ADAM_SIG_EE/EI/IE/TOPO/GAP`.

### Runtimes (rough)
- SCALE=0.1 (14.3k neurons): ~60–90 s per 30 s episode.
- SCALE=1.0 (143k neurons): ~74–250 s. Run in background (`nohup ... &` or run_in_background),
  poll with `sleep` + `tail`.

### Measurement (use the VALIDATED instruments — both pass known-answer controls)
- IBI/bursts: `from cuda_engine.discontinuity import analyze_discontinuity`
  `analyze_discontinuity(net, N, head_discard_ms=8000)`. **Valid domain: burst peak/median
  contrast ≥3×** (below → abstains, 0 bursts, NOT a false number). `head_discard_ms` drops
  the startup transient. Validator: `python discontinuity_validate.py`.
- Criticality: `from cuda_engine.criticality import compute_criticality`; MR-m reliable when
  R²>0.9; `stationarity` flag catches runaway; measure on 2nd half/third (after transient).
  Validator: `python criticality_validate.py`.
- **net_activity is GLOBAL** — diluted by unborn regions (at dw20 only BS+SP born, TH@dw22,
  P1@dw24...). Check per-region from `cuda_rates.npy` + `network_config.json` offsets.

## KEY FILES
- `output/_phase0_tmp/OBSERVATIONS.md` — full journal, findings 1–32 (the scientific record).
- `LITERATURE_PROVENANCE.md` — parameter provenance ledger (PARTS A–E, confidence A/B/C/F).
- `VALIDATION_TARGETS.md` — pre-registered held-out targets.
- `cl_physics_sandbox.py` — numpy sandbox (validated physics: chloride, sAHP, gap, iSTDP).
- `criticality_validate.py`, `discontinuity_validate.py` — instrument known-answer controls.
- `reference/gen_bundle.py` — bundle generator (canonical). `reference/spatial.py` — spatial kernels.

## CURRENT SCIENTIFIC STATE — UPDATED 2026-06-21 (findings #33–45, ledger Part F)
ROOT BLOCKER SOLVED. The depol-GABA "interneuron runaway" was mis-diagnosed: it was actually
BACKGROUND-driven tonic firing (adult-scaled bias/bg currents on immature gL=0.5nS → cells
slammed suprathreshold) + STD-less recurrence. Fixes (all grounded, mostly config):
  • Immature current scaling (#33): a/b/I_bi/ge0 ×(gL_imm/gL_adult≈0.05) — physical, removed knobs.
  • Sparsify I→I (off) + E→I (×0.15) in immature (#32): PV interneurons postnatal (ledger D6).
  • STD/Levina on recurrent E→E (#40, ledger F3): self-organizes criticality, BOUNDS runaway
    (static synapses → 350 Hz; STD → m≈0.95 across REC 6–12). Engine STP already = Tsodyks–Markram.
  • Ca-AHP pause pacemaker = AdEx w, τw≈8 s GROUNDED (#41, ledger F9: mEC developing-cortex AHP
    ~17.5 s / 2–3). pump-AHP REJECTED (#F1/F8: matures postnatally, wrong trend).
  • I_CAN plateau (#42–44, ledger F9/F6): ENGINE ADDED (g_can/tau_can/can_inc/E_can; off by
    default) → burst envelope 1–3 s (was 0.1 s). bouffée 1.1–3.0 s, inter-bouffée 17–23 s.
  • Homeostasis (#45): rate-clamp now WORKS on top of STD (tau_ho≫IBI) → self-selects operating
    point across REC 4–8 (m≈1 emergent). Last "hand" (REC) largely removed.
HEADLINE (reference/grounded_ican, 180 s, SCALE=0.1): tracé discontinu reproduced on grounded
params — bouffée 1–3 s, IBI 17–23 s, m≈0.99 near-critical, frac_empty 0.87–0.99. NO curve-fit.

THREE VALIDATED INSTRUMENTS (all pass known-answer controls):
  cuda_engine/discontinuity.py (5 ms population spikes / sub-cycle IBI),
  cuda_engine/burst_envelope.py (BOUFFÉE envelope dur + inter-bouffée IBI — NEW, robust 1.2%),
  cuda_engine/criticality.py (MR branching m, subsampling-invariant). Run reference/characterize.py.
ENGINE: per-pop full-res output pop_activity.npy [n_pops×T_steps] (NEW); I_CAN current (NEW).

## NEXT STEPS (refinements, not blockers)
1. Tighten homeostatic convergence (#45): lower r_tgt or implement TRUE synaptic-scaling (adjust
   recurrent w toward criticality, not bias current) — engine work. Removes REC fully.
2. SCALE=1.0 confirmation of the grounded headline (currently SCALE=0.1).
3. F5 (ledger): is I→E depol-GABA igniting or SHUNTING (Kirmse 2015)? measure ignite-vs-suppress.
4. Verify mEC 17.5/24 s number (#F9) against primary source.
## PHASE 3 FALSIFICATION v2 IN PROGRESS (bzdau98qd)
v1 (bvo3docrl) RETRACTED: geometric proxy timing-blind (Q1) + wrong silence target (Q7, OBS #63).
v2 fixes: Pearson pop_activity traces (Q3 known-answer passed: sync=0.97, unsync=-0.20);
          ADAM_P1P2_GAIN=0 (specific P1→P2 decoupling, not global afferent kill).
Expected: at dw27+ baseline shows Pearson>0 (A2 wakes via lr_gate), decouple shows Pearson≈0.

## CURRENT HEADLINE: grounded_v4 (Phase 3 lr_gate corrected)
grounded_v4 = grounded_v3 + corrected lr_gate (onset dw20, not dw34, Horgos 2020).
ta_lr_gate at dw20 = 0.029 (was 0.5 hardcoded — wrong by 14 weeks). All regions TRACÉ ✓.
dev_state.py: new lr_gate sigmoid (center_dw=27, steepness=0.5) + activity boost (update_lr_gate()).
SCALE=1.0 validated: P1/P2/A1/SP TRACÉ ✓ (contrast 10k-30k×). TH/M1 marginal (finite-size, open).

KEY ENV KNOBS (headline v3 — grounded_v3/v4, fully biology-anchored dw20 config):
  ADAM_NO_HOMEO=1, ADAM_TAUW_E=8000, ADAM_REC_GAIN=6, ADAM_PEE_GAIN=4,
  ADAM_EE_STP=dep_slow, ADAM_EE_TAUD=5000, ADAM_GCAN=2, ADAM_TAU_CAN=1200, ADAM_CAN_INC=0.3,
  ADAM_TH_P1_GAIN=0    ← waiting period (OBS #53, ledger H4, confidence A),
  ADAM_IBI_GAIN=3      ← SP intrinsic activation (OBS #54/56, confidence C — calibrated, ↓ from 4),
  ADAM_SP_BG_SCALE=0.27 ← GROUNDED: R_in=1342 MΩ (Hanganu 2002); gap-junction-driven SP
                          baseline (Cerebral Cortex 2019). NOT a free parameter (ledger G8).
  TH→SP weight=0.42 nS in HIERARCHICAL (Hanganu 2002, wmax=2.1, ledger G7.1).
  Result (grounded_v3, 180s): SP bouf=5 IBI=31.6s m=0.99 ✓; P1 IBI=20.0s m=0.99 ✓ unchanged.
    Confirmed harmless (P1 bouffée IDENTICAL with/without TH_P1 direct, Q1-Q10 passed).
    Default in gen_bundle is 1.0 — must set explicitly for dw20 snapshots.
  Homeostasis: ADAM_NO_HOMEO=0 + ADAM_RTGT_E=0.5 + ADAM_TAU_HO_MS=150000.
  SCALE=1.0: add ADAM_FIXED_OUTDEG=20 (otherwise P2/A1/A2 go continuous, OBS #50).

## SCALE=1.0 VALIDATION (OBS #61 — grounded_v3_s10)
grounded_v3 config at SCALE=1.0 (N=143k, FIXED_OUTDEG=20): P1/P2/A1/SP all in TRACÉ regime
(real contrast 9600-30000×, p50=0, bouffées present). Previous failure (finding #52: P2/A1 continuous)
is gone. TH_E and M1_E show "marginal" only due to few bouffées in 90s (R²<0.4, not reliable) —
NOT continuous. 180s run in progress (grounded_v3_s10_long) for proper statistics.
KEY ENV (SCALE=1.0): same as grounded_v3 + ADAM_FIXED_OUTDEG=20.

## FUNDAMENTAL VALIDATION COMPLETED (OBS #46–51)
All critical risks before Phase 2 closed:
  - Single-cell operating point: physics self-consistent; I_CAN cascade expected (W terminates burst)
  - F5: I→E NOT needed for bursting; mechanism = E→E + STD-SOC (Kirmse confirmed)
  - Spatial structure: global bursts at dw20 = correct biology; local domains emerge in Phase 2
  - SCALE=1.0 without FIXED_OUTDEG: P2/A1/A2 continuous (out_deg=200 too strong)
  - SCALE=1.0 WITH FIXED_OUTDEG=20: headline structure restored ✓ (OBS #51)
  - BS_E silent: expected — gap loading + low bias; not a bug for dw20 biology
  - TH bursty from T-channel + AUD afferents (not from BS) — correct for dw20

## PHASE 2 PLAN (pending literature closure)
Phase 2 = activity-dependent synaptic pruning. Before implementing, need literature on:
  1. STDP rule shape at dw20 (depol-GABA modifies STDP?)
  2. Fraction + timescale of prenatal pruning (dw20→40)  
  3. Local vs inter-regional pruning priority

Current STDP: already on P1→P2 (proj 46) and P2→A1 (proj 47). Both weights growing (+8-12%)
in headline run. Phase 2 starts by analyzing these weight distributions vs distance.
Key addition needed: per-neuron sparse spike recording for spatial correlation analysis.
  (pop_activity.npy = per-pop aggregate only, cannot resolve which cells fire in burst)

## INSTRUMENTS (all validated + known-answer controls pass)
  cuda_engine/discontinuity.py  — 5ms population spikes / IBI
  cuda_engine/burst_envelope.py — bouffée envelope dur + inter-bouffée IBI (NEW, Q3/Q4 passed)
  cuda_engine/criticality.py    — MR branching ratio m (shuffle-surrogate validated)
  reference/characterize.py     — unified analysis (all 3 instruments + bouffée printed)
  reference/op_point_check.py   — NEW: single-cell operating point validator
  reference/burst_probe.py      — quick burst regime screener (not validated, use for mode-check only)
