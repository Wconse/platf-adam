# Pre-Registered Validation Targets (from literature)

> Written BEFORE fixing the network, so that any later "match" is a real test and not
> a story fitted after the fact. Each target has a source and a quantitative range.
> The model's developmental week (dw) is treated as ≈ gestational/post-menstrual week.

---

## TARGET 1 — Early cortex is DISCONTINUOUS (tracé discontinu)

**Claim:** at dw ~20–28 the network should fire in bursts separated by SECONDS of
near-silence, not quasi-continuous activity.

| Quantity | Literature value | dw window |
|---|---|---|
| Inter-burst interval (mean) | ~6 s | 27–29 wk |
| Inter-burst interval (mean) | ~4–5 s | 30–34 wk |
| Inter-burst interval (max) | up to 44–87 s | 23–26 wk |
| IBI shortens with age | <15 s @ 32wk, <10 s @ 34wk | trend |
| Burst voltage vs interburst | 50–300 µV vs <25 µV | <28 wk |

Sources (ranked by comparability to THIS model — see note):
- **PRIMARY** — Spontaneous activity in human fetal cortex in vitro (Moore et al.,
  J Neurosci 2011): https://www.jneurosci.org/content/31/7/2391
  Direct cellular/population recording, no skull, closest analog to net_activity.
- SECONDARY (scale caveat) — Normal EEG of premature infants 24–30 wk (André et al.):
  https://pubmed.ncbi.nlm.nih.gov/18063233/
- SECONDARY — Neonatal EEG, StatPearls: https://www.ncbi.nlm.nih.gov/books/NBK536953/
- SECONDARY — Burst detection in very/extremely premature infants:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5461890/

> **Source-comparability note (round-4 correction).** Scalp EEG IBI is the gap between
> macroscopic synchronous population bursts visible THROUGH the skull — it requires
> thousands of neurons firing together. It is NOT directly comparable to a pause in the
> model's summed `net_activity`. The in-vitro fetal-cortex paper (Moore 2011) is the
> correct primary analog; EEG values are a soft secondary reference only.

### Operational definition (model side) — pre-registered BEFORE any run

To make the comparison valid, "burst" and "IBI" in the model are defined on a
**population-rate trace**, not on per-step total silence:

1. Build a population-rate trace `R(t)` = `net_activity` re-binned to **5 ms bins**
   (50 steps), converted to Hz per neuron: `R = bin_spikes / (N_neurons · 5ms)`.
2. **Burst** = a contiguous run where `R(t) ≥ θ_on` for at least `D_min = 50 ms`,
   with `θ_on = 3 × median(R)` (median taken over the whole trace). A burst ends
   when `R(t) < θ_off = 1.5 × median(R)`.
3. **IBI** = gap (in seconds) between the end of one burst and the start of the next.
4. Report **median IBI** and its IQR across all bursts in the run.

These thresholds (3×/1.5× median, 50 ms) are fixed NOW and must not be tuned after
seeing results. (Sensitivity to ±20% on each threshold is itself a required check —
ADVERSARIAL PROTOCOL Q4.)

### Falsification criterion (a number, pre-registered)

> At dw-20, in the baseline network, **median IBI < ~0.3 s** (quasi-continuous).
> The discontinuity hypothesis PASSES only if, after the mechanistic change,
> **median IBI ≥ 3 s at dw-20** and the trace shows clear burst/silence alternation.
> A result of 0.3–3 s is PARTIAL (regime shifted but not to tracé discontinu).
> The maturational trend must also hold: median IBI must *shorten* as dw → 32–34.

**MODEL STATUS (round-4 correction): the earlier "max pause = 362 ms vs seconds"
comparison is WITHDRAWN** — it compared 0.1 ms total-network silence against scalp
EEG IBI (incomparable units/scale, ADVERSARIAL Q6). The model's IBI under the
operational definition above has NOT yet been measured. No regime claim is justified
until it is.

---

## TARGET 2 — Delta brushes: emergence, peak, decline

**Claim:** delta brushes (spindle-burst events; thalamo-subplate origin) appear ~dw 24–26,
peak ~dw 31–33, fade by ~dw 35. Delta band ~0.7–2 Hz.

Sources:
- Neonatal/Pediatric EEG review: https://neupsykey.com/neonatal-and-pediatric-electroencephalography-2/
- Delta brushes reflect subplate neuron activity (sensory/spontaneous evoked).

**MODEL STATUS: not yet testable** — requires an LFP/field proxy + spectral analysis
(not yet built). T-channels (TH burst machinery) are in place as a substrate.

---

## TARGET 3 — Sparse, asynchronous single-cell firing early; correlated later

**Claim:** earliest stage = asynchronous sparse low-rate single-cell firing; later stage
= correlated network bursts (electrical synapses → chemical synapses), giving plateau
assemblies, delta waves, spindle bursts, early gamma oscillations.

Sources:
- Spontaneous activity in human fetal cortex in vitro (Moore et al., J Neurosci 2011):
  https://www.jneurosci.org/content/31/7/2391
- Developing barrel cortex activity patterns (preterm somatosensory parallel).

**Caveat:** these papers report sparse/low-frequency qualitatively; precise single-neuron
Hz for fetal cortex in vivo is not well established. Do NOT invent a number — treat
single-neuron rate as a soft constraint (low, sparse), and use the IBI/discontinuity
metrics (Target 1) as the hard, quantitative test.

---

## What this means for the program

1. **The hard, falsifiable baseline test is Target 1 (discontinuity / IBI in seconds).**
   It is quantitative, sourced, and the model currently fails it clearly.

2. **Likely cause (hypothesis to test):** the constant Poisson afferent drive
   (AUD/VEST/TACT/GSW) prevents seconds-long silences. Real early cortex is
   discontinuous because activity is largely self-generated, not externally driven.
   Test: reduce/gate afferent drive at early dw and measure whether IBI lengthens
   toward seconds.

3. **Criticality (the original Phase-0 metric) is downstream of this.** Measuring
   branching ratio on quasi-continuous activity was measuring the wrong regime.
   Fix the dynamical regime first; criticality becomes meaningful only afterward.

4. **Methodological note:** episodes are 12 s. If IBI ~6 s, only ~2 bursts fit per
   episode — too few to estimate IBI statistics. Discontinuity validation may need
   longer episodes or dedicated long runs at fixed early dw.

---

## Mechanism analysis (on paper, BEFORE any experiment)

> Round-4 lesson: I originally named ONE cause (afferent drive). Verified against the
> engine, there are at least FOUR mechanisms keeping dw-20 activity quasi-continuous.
> Turning off only afferents would likely leave IBI short and give a misleading "fail."
> Each mechanism gets a separate prediction so the experiment is interpretable.

Verified model facts (from config + main_multiregion.cu):
- `E_GABA(dw20) ≈ −45 mV`, spike threshold `VT = −50 mV`, rest `EL = −70 mV`.
  → GABA reversal is ABOVE threshold ⇒ GABA is depolarizing/excitatory at dw-20.
- Afferent Poisson pops fire continuously from a rate table ("afferents always fire").
- `I_homeo` homeostatic current pulls each pop toward a target rate.
- Correlated OU background noise is injected into `ge_bg` every step.

| # | Mechanism | Why it opposes long silences | Predicted effect on IBI if removed/gated at dw-20 |
|---|---|---|---|
| 1 | **GABA depolarizing** (E_GABA −45 > VT −50) | every inhibitory synapse adds excitation; thousands of them = large internal tonus | **Largest single effect.** Forcing E_GABA below VT (e.g. −60) should let the network fall silent between self-generated bursts. Expect IBI to lengthen most here. |
| 2 | **Afferent Poisson drive** (AUD/VEST/TACT/GSW) | constant external excitation floor | Moderate. Gating afferents alone should lengthen IBI somewhat, but GABA-excitation may keep it short. |
| 3 | **Homeostatic clamp** (`I_homeo`) | actively injects current when rate drops below target → fills silences | Moderate. With homeostasis off, silences can persist; but this is a *clamp artifact*, not biology — must be controlled, not "the answer." |
| 4 | **Correlated OU background** (`ge_bg`) | shared excitatory noise floor | Small. Reducing amplitude should have minor effect on IBI vs. #1–#3. |

**Predicted ordering of contribution:** #1 (GABA) ≳ #2 (afferents) > #3 (homeostasis,
artifactual) > #4 (noise). The experiment must test these **one at a time** (factorial
if feasible), not all at once, or the result cannot be attributed.

**Why this matters for falsification:** if I gate only afferents (#2) and IBI stays
<0.3 s, that does NOT falsify the discontinuity hypothesis — it confirms #1 dominates.
The hypothesis is only fairly tested when GABA-excitation is also addressed.

---

## Pre-registered prediction (the actual Test-2 commitment)

> When the activity-supporting mechanisms are addressed in order of predicted strength
> (#1 GABA-excitation first, then #2 afferent drive), the network's **median IBI at
> dw-20** (operational definition above) should lengthen from <0.3 s toward ≥3 s,
> entering the tracé-discontinu regime, AND should shorten again as dw increases toward
> 32–34 (E_GABA matures past VT on its own via the GABA-flip curve). This is unprogrammed:
> nothing in the model imposes an IBI value — it must emerge from the interplay of GABA
> reversal, drive, and recurrent dynamics.

This is the first genuinely falsifiable, literature-grounded prediction in the project.

---

## RESULT (mechanism #1 experiment) — UNTESTED (experiment invalid: per-episode artifact)

> Wording correction (round-6): the hypothesis is **not** "NOT supported" — that would
> imply a valid test returned a negative. The test was **invalid** (a per-episode
> artifact dominated the trace), so the hypothesis is simply **UNTESTED**. Calling an
> invalid experiment a "negative result" is itself a methodological error.

Ran, with the validated IBI instrument (`discontinuity.py`, passed known-answer control):
- **Baseline** (E_GABA −45, natural dw-20): mean 3.00 Hz/neuron.
- **E_GABA −60** (forced inhibitory): mean 2.83 Hz/neuron.
- **E_GABA −60, homeostasis OFF**: bit-identical to homeostasis-ON.

At the pre-registered threshold (3.0× median) E_GABA −60 appeared "discontinuous"
(3 bursts, IBI ~41 s) vs baseline "continuous" (1 burst). **This was an artifact.**

Two controls demolished it:
1. **Threshold sensitivity (ADVERSARIAL Q4):** burst count went 10 → 3 → 0 as the
   threshold moved 2.0× → 3.0× → 3.6×. The "41 s" was threshold noise on n=3 bursts.
2. **Matched threshold sweep (Q6):** at on=2.0×, baseline and E_GABA −60 are
   **identical** — both 10 bursts, IBI ~8.7 s, same CV (0.58), same p95/max.
   E_GABA −60 vs −45 makes **no difference** to burst/IBI structure.

**Why both show "10 bursts ~9 s apart":** every burst starts at a FIXED phase
(measured 8.5–8.9 s, mean ~8.7 s) within each 12 s episode — exactly one burst per
episode, in both conditions. This is a **per-episode replay artifact**, and the
*primary* driver (round-6 diagnosis) is the **population-birth schedule**, not the
g-reset originally blamed.

Root-cause diagnosis (verified against config `t_bir` + measured burst phase):
- At fixed dw-20 every episode re-runs the same birth schedule. Three populations are
  born *inside* the 12 s window: **TH @ 3.73 s, P1 @ 7.47 s, M1 @ 7.47 s** (P2/A1/A2
  are born after 12 s, so they never live in a dw-20 episode). The cortical pair
  P1+M1 switching on at 7.47 s drives a synchronized discharge ~1.2 s later → burst
  at **~8.7 s**, deterministically, every episode. Measured burst phase (8.5–8.9 s)
  matches `t_bir(P1/M1) + ramp` and is identical across E_GABA −45 / −60.
- Two *other* per-episode resets co-occur but are not the main rhythm: (a) boundary
  reset of g_ampa/g_nmda/g_gaba → 0 acts on ms→slow-recovery scale (secondary, cannot
  be excluded without a discriminating run); (b) STP states (u, x) are not carried
  across the boundary, but their τ (400–600 ms) is far too fast to produce an 8.7 s
  rhythm — excluded on timescale.
- The per-episode *environment* (different each episode) is NOT the cause; no afferent
  event sits at 8.7 s. The 100 ms guard window is far too short to remove it.

**Conclusions (honest):**
- Mechanism-#1 hypothesis (depolarizing GABA causes the quasi-continuous regime) is
  **UNTESTED** — the per-episode artifact dominated, so this run carries no information
  about E_GABA's effect on the IBI regime.
- The episode-concatenation method is **invalid for spontaneous IBI** as currently
  built: each episode replays the population-birth schedule (and resets conductances),
  injecting a periodic synchronized burst locked to birth time. A valid measurement
  needs EITHER a continuous long run at fixed dw with no re-birth / no g-reset
  (CUDA-graph re-capture), OR a discriminating control that moves/removes the births
  and confirms the burst phase tracks them.
- No discontinuity claim is justified. The baseline regime question is still OPEN and
  must be re-measured after the per-episode artifact is removed.

**Not yet checked (Q9):** the discriminating experiment — shift or suppress the P1/M1
birth and verify the burst phase moves with it (would confirm birth as primary) vs.
stays at ~8.7 s (would re-implicate g-reset). This is the next code/run step.

> The protocol worked. A result that looked like the project's first breakthrough was
> caught as an artifact by the threshold and matched-control checks BEFORE it was claimed.

---

## FIX + VALID RE-MEASUREMENT (round-7)

### Root cause confirmed and fixed (the births, not the g-reset)

The 8.7 s artifact was traced to `engine_runner.run_episode`: it rewrote each
population's `t_bir` from `current_dw`. At a FIXED dw (concatenation), `current_dw`
never advances, so `start_abs_s` was always 0 and the same populations were
**re-born at the same in-episode phase every episode** (P1, M1 @ 7.47 s → burst
@ ~8.7 s). Proof, per-episode `t_bir` after the fix:

| episode (elapsed) | born during episode | P1/M1 t_bir |
|---|---|---|
| ep0 (0 s)  | TH, P1, M1 | 7467 ms (born) |
| ep1 (12 s) | P2, A1     | **0 ms (already alive — no re-birth)** |
| ep2 (24 s) | A2         | 0 ms (already alive) |

**Fix:** `run_episode(..., elapsed_abs_s=)` schedules births on the true cumulative
sim clock, so each population is born exactly once. (`engine_runner.py`, §3.)

**Verification on the actual output (Q9 closed):** autocorrelation of the post-fix
rate trace at lag 8.7 s = **−0.17** (no periodicity) — the birth rhythm is gone.
A residual peak remains at the 12 s *episode* period (r ≈ +0.77): the secondary
**g-reset** boundary artifact. It modulates the continuous rate but creates **no
discrete bursts**. Eliminated entirely by the `--single` long-run mode (one
continuous run, no boundaries).

### Valid baseline (fix + fixed weights `--no-scaling`, 10×12 s, head-discard 36.7 s)

| condition | mean rate (Hz/neuron) | n_bursts | Q4 (threshold ±20%) | verdict |
|---|---|---|---|---|
| baseline E_GABA −45 | 14.5 | 0 | 0 bursts at 2.0/2.4/3.0/3.6× (stable) | **quasi-continuous** |
| E_GABA −60 | 13.0 | 0 | 0 bursts at all thresholds | quasi-continuous |

The rate sits in a TIGHT band (~14–18 Hz, IQR ≈ 3). The "0 bursts" verdict is now
**robust to threshold** (contrast the artifact run, where it swung 10→3→0). The
E_GABA override is verified working (isolated test: all 8 regions −45→−60) and the
14.5→13.0 Hz drop confirms it acts — but it does **not** flip the regime.

### What this establishes (and the remaining confound)

- **Mechanism #1 alone is insufficient:** forcing GABA hyperpolarizing (−60) leaves
  the dw-20 network quasi-continuous. Depolarizing GABA is NOT the sole cause of
  continuity. (This is now a *valid* test — the artifact is gone — unlike round-6.)
- **Open confound (hostile-reviewer Q10):** the tight rate band is the signature of
  the within-episode homeostatic current `I_homeo` (mechanism #3) pinning the rate to
  `r_tgt`. `--no-scaling` disabled only *between-episode* weight scaling; `I_homeo`
  was still active. The mechanism-#1 test in isolation is therefore still partly
  masked by #3.
- **Next (running):** pre-registered factorial with single clean long runs —
  toggling `I_homeo` (`--no-homeo`), afferent drive (`--no-afferents`), and E_GABA,
  to find which mechanism(s) actually gate the discontinuity, in the predicted order
  #1 ≳ #2 > #3 > #4.

> Status of Target 1: baseline regime is now VALIDLY measured (quasi-continuous at
> dw-20, robust). Whether the model *can* enter tracé discontinu when the
> activity-supporting mechanisms are removed is the open question the factorial tests.

---

## COMPLETE FACTORIAL + ATTRIBUTION (round-8) — Target 1 RESOLVED (negative)

All runs: single continuous long run (`--single`, 10×12 s = 120 s, no boundaries,
no re-birth), fixed weights (`--no-scaling`), head-discard 36.7 s (last birth + settle),
IBI instrument validated on known-answer synthetic data. Rate = Hz per neuron.

| # | E_GABA | homeo | aff | OU(py) | bg(ge0) | mean Hz | p0 Hz | frac<0.1Hz | longest silence | bursts @2.0/2.4/3.0/3.6× | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noaff (≈baseline*) | −45 | on | off | on | on | 16.40 | 10.29 | 0.000 | 0 s | 0/0/0/0 | **clamped continuous** |
| nohomeo | −45 | off | on | on | on | 0.922 | 0.224 | 0.000 | 0 s | 128/58/25/8 (fragile) | OU-fluctuating |
| nohomeo_noaff | −45 | off | off | on | on | 0.913 | 0.237 | 0.000 | 0 s | 133/63/25/9 (fragile) | OU-fluctuating |
| full_m60 | −60 | off | off | on | on | 0.882 | 0.237 | 0.000 | 0 s | 116/54/14/5 (fragile) | OU-fluctuating |
| bare | −45 | off | off | off | on | 0.568 | 0.251 | 0.000 | 0 s | 0/0/0/0 | smooth continuous |
| bare_m60 | −60 | off | off | off | on | 0.566 | 0.251 | 0.000 | 0 s | 0/0/0/0 | smooth continuous |
| **truebare** | −45 | off | off | off | **off** | **0.000** | 0 | **1.000** | **entire run** | — (no activity) | **SILENT (0 spikes)** |

\* baseline (all mechanisms on) ≈ noaff because the I_homeo clamp dominates the rate
regardless of afferents.

### The decisive observation (threshold-free)

**In every driven condition the network never goes silent** (`frac<0.1 Hz = 0.000`,
`longest_silence = 0 s`, rate floor p0 ≥ 0.22 Hz). The *only* way to reach silence is
to zero the engine's tonic background `ge0` — and then the network is **permanently**
silent (0 spikes in 120 s), not discontinuous. There is no parameter regime between
"continuously driven" and "completely dead." Tracé discontinu (bursts separated by
SECONDS of self-organized silence) does **not** occur anywhere in the factorial.

### Mechanism attribution (each isolated by a direct control)

- **Tonic background `ge0` (engine OU, NOT the python correlated layer) = the sole
  source of the activity floor.** `ge0` is a DC excitatory background conductance
  (1.5–20 nS/pop; `main_multiregion.cu` `ou_noise_curand`, pulls `ge_bg → ge0` every
  step). With it zeroed (`--no-bg`) → **0 spikes**. This is a 5th activity-supporting
  mechanism that the round-4 list missed.
  - **`ge0` is biology, not an artifact (correction, round-8 review).** A tonic
    background conductance models real phenomena: spontaneous quantal transmitter
    release (miniature EPSCs), tonic glial/ambient glutamate, and input from axons not
    explicitly wired. Such a floor SHOULD exist. The legitimate scientific question is
    therefore **calibration, not existence**: is 1.5–20 nS/pop the right magnitude for
    dw-20, or is it (likely) too high by ~10× vs. a biologically plausible ~0.1–1 nS?
    An over-strong `ge0` would impose continuity and mask any structured drive. This is
    a parameter-calibration question to settle against literature BEFORE concluding the
    architecture is wrong. (Earlier framing of `ge0` as a "hardcoded artifact to
    remove" was incorrect.)
- **The recurrent network has no spontaneous activity without input.** truebare = 0
  spikes proves *only* that — there is no self-ignition from rest. It does **NOT**
  prove subcriticality. (Correction, round-8 review: an earlier draft wrote
  "SUBCRITICAL" here — that was wrong. Zero spikes from rest holds for sub-, critical,
  AND super-critical networks alike, because none has an internal source of the first
  spike. Criticality is a statement about what happens AFTER a perturbation: pulse the
  network, then watch — decay ⇒ subcritical, sustain ⇒ critical, grow ⇒ supercritical.
  That perturbation test has NOT been run. The branching-ratio / avalanche regime at
  dw-20 is therefore still **OPEN**, not "subcritical.")
- **I_homeo (mech #3) sets the rate LEVEL, not the regime.** On → 16.4 Hz; off →
  0.57 Hz floor. Robust 17× collapse. Exactly the pre-registered "clamp artifact."
- **Correlated OU (mech #4) is the sole source of the *fluctuations*** a median-relative
  threshold mislabels as "bursts." Remove it (bare) → ultra-tight band (p0 0.25 →
  p100 0.94 Hz), **0 bursts at every threshold**. The apparent "IBI ≈ 2.3 s" in the
  nohomeo conditions was an OU+threshold artifact (Q4: swings 0.42→14.3 s, 13–34×).
- **E_GABA (mech #1) does not affect the regime *at this tonic drive level*:** bare
  0.568 vs bare_m60 0.566 Hz — identical. **Scope caveat (round-8 review):** this is
  context-dependent, not absolute. At a low tonically-driven rate (~0.57 Hz) GABA
  synapses open rarely (only when an E-cell spikes), so the GABA reversal potential has
  little leverage. Under HIGH activity or a structured burst drive — where GABA
  synapses open frequently — E_GABA −45 (depolarizing) vs −60 (hyperpolarizing) could
  matter substantially. The correct statement is *"E_GABA is irrelevant at the current
  ge0-tonic drive level,"* not *"E_GABA is irrelevant."* Its role must be re-tested once
  a structured/higher-activity regime exists.
- **Afferent drive (mech #2) is irrelevant to the regime:** removing it changes the
  rate by <1% (homeo-off) or nothing (homeo-on clamp compensates).

### Verdict on the pre-registered prediction

> The round-4/round-7 Test-2 prediction — "addressing #1 (GABA) then #2 (afferents)
> lengthens median IBI from <0.3 s toward ≥3 s, entering tracé discontinu" — is
> **FALSIFIED.** The predicted contribution ordering (#1 ≳ #2 > #3 > #4) is also wrong:
> the real ordering is **ge0-tonic (regime floor) > #3 I_homeo (rate level) > #4 OU
> (fluctuations) ≫ #1 E_GABA ≈ #2 afferents (no regime effect)**. GABA reversal, the
> predicted dominant lever, has zero effect on the regime.

### Why the model produces continuous activity — and the Subplate (round-8 review)

The dw-20 network is tonically driven by a continuous `ge0` background and (in the
realistic config) rate-clamped by `I_homeo`. Continuous input ⇒ continuous output.
Reaching tracé discontinu requires the drive to the cortical plate to be itself
**structured / bursty**, alternating activity with silence.

**In real biology that structured drive has a specific source: the SUBPLATE.** Subplate
neurons are the transient early scaffold that relays thalamic input to, and generates
spontaneous burst activity in, the cortical plate before thalamocortical synapses
mature. Subplate-driven spontaneous bursts ARE the substrate of tracé discontinu at
this age. **The model already contains a Subplate region (SP, born dw-20).** So the
correct question is not "what new structured-drive mechanism should we add?" but
**"why does the existing SP not perform its biological function?"**

Direct inspection of SP in the model (config wiring + a homeo-off diagnostic run)
reveals TWO concrete gaps:

1. **SP has no efferent projection to the cortical plate (architectural).** Of SP's
   7 projections, ALL are internal (SP_E→SP_E/SP_I/SP_SST). There is no SP→P1 (or
   SP→any cortical region) synapse — `n_syn = 0`, not merely weak. The cortical input
   chain is TH→P1→P2→A1→A2 with SP wired to nothing outside itself. SP also *receives*
   nothing external (no thalamic, no afferent input): it is an isolated recurrent
   island driven only by `ge0`+OU. In biology the canonical early circuit is
   TH→SP→cortical-plate; here SP is bypassed (TH→P1 directly).
2. **SP_E does not burst (dynamical).** In the diagnostic run (homeo off, so no clamp),
   SP_E fires at **0.27 Hz, CV 0.25** (range 0.11–0.52 Hz over 150 ms bins) — a smooth
   tonic floor set by `ge0`, *flatter* than the cortical regions, with no burst/silence
   structure. (SP_I interneurons do fluctuate, CV 0.78, but they have no efferent to
   relay it anywhere.) So even if SP→cortex wiring were added today, SP currently has
   no structured signal to deliver.

**Corrected next step (a diagnosis of an existing mechanism, not a new addition):**
make the Subplate do its job, in order —
   (a) **Calibrate `ge0` down** to a biologically plausible tonic level (~0.1–1 nS vs
       the current 1.5–20 nS) so it no longer floors/masks structured activity (this is
       the Problem-2 calibration question and a prerequisite — an over-strong tonic
       floor would hide any SP burst even after wiring it up).
   (b) **Get SP_E to generate spontaneous bursts** — first run the perturbation /
       branching-ratio test (below) to learn SP's intrinsic regime, then tune SP
       recurrent gain (and/or add the missing TH→SP input) toward a burst-generating
       regime, with avalanche statistics as the metric.
   (c) **Wire SP_E → cortical plate (TH→SP→P1)** so the structured subplate drive can
       reach and pace the cortex.
   (d) Re-measure cortical IBI and test whether tracé discontinu now emerges — and
       whether E_GABA (re-tested under this higher, structured activity) now matters.

### Open / not yet checked (Q9 — explicit)

- **Criticality is UNTESTED.** The branching-ratio / avalanche regime (sub/critical/
  super) needs a perturbation test (pulse → decay/sustain/grow), not the `ge0=0` run.
  This is the corrected Problem-1 item and a Phase-0 instrument to build.
- **`ge0` magnitude is uncalibrated** against a biological tonic-conductance target
  (Problem 2). Settle the number from literature before further regime claims.
- Whether the network/SP behaviour changes at **later dw** (28–34) as E_GABA matures
  and wiring grows — only dw-20 tested.
- Per-region silence vs the whole-network trace (secondary; whole-network is the EEG
  analog).

> The ADVERSARIAL PROTOCOL + round-8 review did real work: rejected a fragile
> "IBI ≈ 2.3 s" (13–34× threshold swing); caught that `--no-ou` left the engine's tonic
> `ge0` running the whole time (preventing a false "recurrent self-excitation" claim);
> corrected an unjustified "subcritical" label (truebare=0 only shows no self-ignition);
> reframed `ge0` from "artifact" to "biological tonic conductance, possibly
> miscalibrated"; scoped the E_GABA-null result to the current drive level; and — the
> key one — redirected the next step from "add structured drive" to "fix the Subplate,
> which already exists to provide it but is both unwired to cortex and non-bursting."

---

## ROUND-9: Criticality answered via validated numpy mirror (Phase-0 instrument DONE)

The Problem-1 perturbation test was finally executed — not through the engine (its
maturation gate `mat=0` at step 0 wipes any seed to V_reset before spike check), but
through a **transparent numpy reimplementation of the engine's exact fused AdEx +
synapse kernels** for the isolated SP microcircuit (`output/_phase0_tmp/sp_sim.py`).

### Instrument validated against the engine itself (Q3, properly):
- Sub-threshold per-neuron v matches CUDA to **rms 1.6e-5 mV** (float32 round-off).
- Under heavy deterministic spiking (85–222 Hz) per-pop rates match to **≤0.2%**.
- Single-neuron AdEx integrator converged (dt 0.1 == 0.005); rheobase ~200 pA sane.
- SP is fully isolated (no projections in/out) so SP-only sim == SP-in-full-engine.

### RESULT [confirmed; impulse AND sustained; adversarial-protocol-passed]:
**At the reference operating point, SP cannot self-propagate excitation — E→E
recurrence is functionally dead.** Pop-resolved spike counts (the key control that
overturned an initial wrong "m≈0.33 subcritical" reading — that m was measuring
feedforward E→PV, not E→E) show **zero excitatory descendants** under:
- impulse seed up to 25% of SP_E (only PV fires, one feedforward generation, dies ~6 ms);
- sustained current drive up to **90% of SP_E at 4× rheobase for 100 ms** (recruits PV
  reliably, 0 non-driven E; post-drive persistence 2.6 ms ≈ one synaptic delay, no
  reverberation);
- E→E weight scaling up to **×8**; depol-GABA E_GABA −45/−55/−75 (no effect).

### Mechanism (direct membrane recording):
1. A single/sustained E→E volley is **sub-threshold for the AdEx upstroke** — a
   non-driven E neuron's membrane reaches only ≈−54 mV vs VT −50 (soft) / V_spike −30.
2. Heavy **background shunt**: tonic ge0+gi0 ≈ 28 nS vs gL 10 nS.
3. **Sparse S_SP_EE** (4000/800² ≈ 0.6% → ~5 presynaptic partners/neuron): even maximal
   drive lands ~5 synaptic hits (units of nS) against the 28 nS shunt.
4. PV is recruited directly (LIF, V_spike −50) and adds inhibition — but inhibition is
   NOT the primary cause: zeroing all inhibitory weights (`kill_inh`) changes nothing.

### Consequences for the roadmap:
- **Criticality question CLOSED for SP**: not sub/critical/super in the branching sense
  — the recurrent E network is below the propagation threshold entirely. The earlier
  "subcritical" label is now justified *mechanistically*, not inferred from `ge0=0`.
- This is the dynamical half of "why doesn't SP do its job?": SP **cannot burst** at
  reference params, independent of the architectural fact that it has no efferents to
  cortex. Both must be fixed for subplate-driven cortical pacing.
- **E_GABA-null result strengthened**: irrelevant to regime even under structured
  sustained drive (the E→E path, not GABA polarity, is the bottleneck).
- Phase-1 calibration target identified: to make SP burst, must lower the background
  shunt and/or raise E→E coupling (density or weight) until non-driven E ignites —
  to be set against a biological tonic-conductance + subplate-burst target.

### Discarded probes (documented, not silently dropped):
- MR-estimator on driven traces: reads the 20 ms colored-OU timescale, not branching.
- Engine carry_state perturbation: blocked by maturation gate.
- `kill_gi0_bg` probe: zeroing background gi0 destabilizes settle (e_desc=−200 artifact)
  — no conclusion drawn from it.
- Fixed a latent bug: `sp_sim.run()` reset the step counter each call (harmless for
  silent-baseline impulse tests; would break multi-phase runs) → persistent counter.

Full chronological journal: `output/_phase0_tmp/OBSERVATIONS.md`.

---

## ROUND-9 RETRACTION (colleague review) — supersedes the ROUND-9 section above

The ROUND-9 section above declared the SP result "confirmed" and Phase-0 "done".
**Both are RETRACTED.** Reviewer raised three valid points; I verified all in the
bundle generator code. Net: the E→E observation is real but UN-interpretable until
the bundle is fixed, and Phase-0 is NOT closed.

### (1) E_GABA = −75 mV for SP at dw-20 is a GENERATOR DEFECT, not SP physiology
- `reference/gen_multiregion.py:61`: `E_GABA_CONST = -75.0  # mV, simplified constant for M3`
- `reference/gen_multiregion_scale.py:56` + `:414-415`: same hardcode, looped over ALL regions.
- Model spec demands the opposite: `Present.md:123` `E_GABA: −45 mV (dw 20) → −75 (dw 34+)`,
  `:132` `E_GABA = −45 + kcc2·(−75−(−45))`. Depolarizing GABA at dw-20 is the project's
  central biology.
- A working ramp exists (`reference/gen_ta_time_varying.py:26-28`, −65→−80) and is
  reported working for P1 (`Present.md:242` GABA-flip dw≈30, VPA experiments) — it was
  simply NOT wired into the multiregion bundle.
- Consequence: the "E_GABA irrelevant" result tested the right thing but in a network
  where NO region has depolarizing GABA — cannot be generalized.

### (2) SP unconnected to the hierarchy is a TOPOLOGY DEFECT, not design
- Hierarchy in both generators (`gen_multiregion.py:324`, `gen_multiregion_scale.py:371`):
  TH→P1→P2→A1→A2, BYPASSING SP. No TH→SP, no SP→P1.
- Spec (`Present.md:22`): `Brainstem → Thalamus → Subplate`; `:42` SP = "transient
  scaffold layer". The intended TH→SP→P1 relay is not in the topology.
- Consequence: even if SP burst, P1 would never see it. "Why doesn't SP burst" analyzes
  an isolated island with no functional role in the system.

### (3) MY OWN ERROR: "SP receives no external input" was WRONG
- `aff_config.json` has `S_GSW_SP: GSW→SP_E` (1024 syn, src_N=32), `aff_rate_GSW = 1.0 Hz`
  constant. SP receives the GSW sensory afferent. I checked only network_config
  `projections`, not `aff_config`.
- Consequence: the numpy mirror (no afferents) is validated ONLY for an afferent-silent
  window — never explicitly controlled in Q3b. The sustained analysis ignored real GSW input.

### (4) Protocol Q5 failed honestly
- "SP should burst" is a direction, not a quantitative target (burst rate/duration/IBI).
  Phase closed with an unpassed STOP question. Retracted.

### Corrected status & roadmap impact
- **Phase-0 NOT closed.** Real Phase-0 deliverables still missing: avalanche/criticality
  metric (withdrawn — MR confound), emergence-audit harness + CI, baseline fingerprint
  across seeds, quantitative pre-registered targets (only qualitative exist).
- What IS solid: the numpy mirror (validated to 1e-5 mV / ≤0.2% in the afferent-silent
  regime) and the bare observation that E→E does not propagate in the isolated SP
  microcircuit at reference params.
- **SP work moves from a Phase-0 RESULT to a Phase-1 FIX task:** before asking "does SP
  burst and pace cortex", the generator must be corrected — wire the E_GABA ramp (−45 at
  dw-20) and add TH→SP→P1 — then regenerate the bundle. That is a model change (awaiting
  user decision), not a Phase-0 measurement.

Full chronological journal incl. retraction header: `output/_phase0_tmp/OBSERVATIONS.md`.

---

## ROUND-10: New canonical bundle (gen_bundle.py) — baseline regime measurement

### Bundle changes (gen_bundle.py replacing all gen_multiregion*.py)
- E_GABA: **-45 mV at dw-20** via dev_state sigmoid (correct spec; old bundle had -75).
- TH→SP→P1 relay: **NOW WIRED** (S_TH_SP_E/I + S_SP_P1_E/I). First time in the project.
- ge0 calibrated to 0.65x old values (mean tonic conductance only; OU noise amplitude
  unchanged). Biological rationale: old 10-20 nS was ~10x too high vs mEPSC/tonic glu.
- 24 pops, 68 projs, 14,300 neurons at SCALE=0.1.

### Baseline regime (dw-20, no-homeo, 120s trace, pre-registered 3.0x/1.5x threshold)
- mean rate: 0.231 Hz/neuron (vs 0.92 Hz in old bundle; 4x lower — ge0 calibration works)
- bursts at 3.0x/1.5x: 31 over 120s, median IBI = 1.1 s, max IBI = 12.1 s
- **VERDICT (pre-registered): PARTIAL (IBI 0.3-3s) — regime shifted toward discontinu but not there yet**

### Q4 (adversarial threshold stability) — FAILS
Threshold sweep: **121 / 31 / 0 bursts** at on_factor 2.4 / 3.0 / 3.6x.
This means the "bursts" are OU-fluctuation peaks above an arbitrary threshold, not
genuine burst/silence transitions. The 12s max IBI is real silence, but flanking
bursts are not robustly separated from background. Same root issue as old bundle.

### What Q4 failure means for the program
Reducing ge0 further would eliminate all activity (0 bursts at scale ≤0.5). The
network cannot produce *intrinsic* burst/silence separation from OU noise alone —
it needs a mechanism that either:
(a) Creates synchronous burst initiation (structured subplate drive via TH→SP→P1, now wired)
(b) Creates post-burst silence (e.g. SFA, chloride depletion from depolarising GABA)

Both (a) and (b) require dynamics that emerge under Phase-1 (E_GABA from chloride).
The current baseline is the correct starting point for Phase-1: active enough to generate
GABA chloride loading, quiet enough that the flip will have visible effect.

### Phase-0 gate decision
Phase-0 baseline is AS GOOD AS IT CAN GET without Phase-1 mechanisms:
- ge0 calibrated to biologically plausible range
- E_GABA correct (-45 depolarising at dw-20)
- TH→SP→P1 relay wired
- Q4 fails because the required mechanism (chloride/SFA) is Phase-1

**RECOMMENDATION: proceed to Phase-1. The Q4 failure is the scientific motivation for
Phase-1, not a blocker. The plan explicitly states: "Criticality is downstream of the
dynamical regime — fix the regime first; criticality becomes meaningful only after."**
The regime is as fixed as it can be without Phase-1.

---

## ROUND-10 CORRECTION (round-10 review) — honest reformulation

The round-10 section above incorrectly declared "Phase-0 closed." It is NOT. Corrected.

### Phase-0 actual status (honest)
Phase-0 produced real findings that changed the project's understanding, but did NOT
complete its original instrumentation plan. The original plan had 5 items:

| Item | Status |
|---|---|
| Working criticality metric | WITHDRAWN (3 attempts: avalanche, MR-on-driven, perturbation — all blocked) |
| Emergence-audit harness multi-seed CI | Written, never run with CI |
| Perturbation runner | numpy mirror only, not the planned harness |
| Baseline fingerprint across seeds | Not done |
| Quantitative pre-registered targets MET | Partial — qualitative only; Q4 fails on IBI |

Phase-0 instead produced: ge0 calibration (10x overestimate found), E_GABA fix
(-45 at dw-20 now correct), TH→SP→P1 wired, gen_bundle.py canonical generator.
These are real findings that justify moving forward — but the original plan items are
not complete.

### Why Q4 failure is a STOP, not a "motivation for Phase-1"
Q4 says: if the result changes >20% when instrument parameters vary ±20%, the result
is an artifact of parameter choice, not signal. The IBI metric varies 121x/31x/0x
across on_factor 2.4/3.0/3.6x. This means the IBI result is UNSTABLE and cannot
be cited as a measurement. Reframing this as "the scientific motivation for Phase-1"
was a protocol violation caught by round-10 review.

### Correct framing: parallel advance, not Phase-0 closure
Phase-0 is OPEN in its criticality/IBI instrument work. Phase-1 begins in parallel
because Phase-1 (chloride dynamics) directly addresses the mechanism that would fix
Q4 (burst/silence separation from chloride depletion post-burst). Once Phase-1
mechanisms are in place, re-running the IBI instrument on Phase-1 network will
constitute the real Phase-0 instrument closure.

### New bundle verification (round-10 review item 2)
Verified in code (NOT just taken on faith):
- `ta_egaba_*.npy`: all -45.0 mV confirmed in files (not just generator claim)
- gen_bundle.py writes E_GABA via `egaba_at_dw(r, DW_START)` → dev_state sigmoid →
  -45.0 at dw=20. Old generator used `E_GABA_CONST=-75.0` hardcoded bypassing sigmoid.
- New projection weights (TH→SP, SP→P1): lognormal from HIERARCHICAL table, same
  method as all other hierarchical projections. No primary biological calibration —
  analogous to TH→P1 (p=0.003, median=0.20) which is itself uncalibrated. Honest gap.
- N changed: 15,158 → 14,300 neurons. SST split counted differently vs old generator.
  This is a DIFFERENT NETWORK — old Phase-0 results do not transfer without re-running.
- numpy mirror (sp_sim.py) NOT re-validated for new bundle. Valid for old bundle only.

### Open quantitative gap (round-10 review item 4)
"Active enough for chloride loading" has no cited number. Biological reference needed:
minimum GABA-driven chloride flux relative to KCC2 efflux capacity for activity-
dependent E_GABA to differ between neurons. This must be established BEFORE claiming
0.23 Hz is a valid starting point for Phase-1. Flagged as Phase-1 precondition.

================================================================================
## PRE-REGISTERED TARGETS (literature, 2026-06-20) — written BEFORE measurement
## Source: prenatal/preterm cortex literature review. These are the held-out
## quantitative targets for the emergence tests (protocol Q5). DO NOT tune to fit.
================================================================================

| Feature                  | Target value                          | Window / note                  | Source |
|--------------------------|---------------------------------------|--------------------------------|--------|
| GDP / burst duration     | hundreds of ms (~0.2–2 s)             | whole early period             | GDP review (Ben-Ari) |
| Inter-burst interval IBI | several seconds; DECREASES with age   | longest before 28 wk PMA       | preterm EEG / SAT |
| Continuity onset         | brief continuous activity first ~25wk | tracé alternant by ~35 wk      | preterm EEG |
| Tracé discontinu regime  | predominates < 28 wk PMA              | replaced by continuous ~35wk+  | preterm EEG |
| Delta brushes            | appear 28–30, peak 32–35, gone 38–42  | wk PMA                         | delta brush lit |
| sAHP kinetics            | max ~1 s post-onset; decay seconds; needs >300 ms depol; V½ ≈ −40 mV | sets IBI period | sAHP (KCa3.1) |
| Subplate firing          | > 40 Hz                               | pacemaker, gap-junction syncytium | subplate amplifier |
| E_GABA flip              | −45 mV (dw20) → −75 mV (dw34+)         | KCC2 up / NKCC1 down           | chloride switch |
| Programmed cell death    | up to 50% neurons; E/I in sync        | activity (bursts) → survival   | apoptosis lit |
| Pause mechanism          | synaptic depression + GABA-B + sAHP   | NOT chloride depletion         | SAT refractoriness / GDP-term |
| Synchronization          | gap junctions (subplate + pyramidal)  | transient, gone with maturation | gap-junction lit |

### Emergence-test rule (per protocol):
- Mean firing rate is NEVER a result (it is r_tgt). Only STRUCTURE counts.
- Validation = emergent IBI/burst-duration/avalanche-stats matching the ABOVE,
  measured WITHOUT fitting to them, across a ROBUST parameter basin.
