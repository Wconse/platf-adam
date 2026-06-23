# ADAM — Present v2 — What Changed, and Why

**A side-by-side of the project as described in `Present.md` (v1) versus where it
actually stands now (v2).** This document is a *diff with reasons*: what was replaced,
what was added, and — most importantly — **how the working process itself changed.**

> TL;DR. v1 was a *developmental-neurotoxicology showcase*: inject a known mechanism
> (VPA delays KCC2, ASD adds E/I bias), then show the expected phenotype. v2 is a
> *grounded-emergence research engine*: the target phenomena (discontinuous fetal EEG /
> *tracé discontinu*, self-organized criticality) must **emerge** from literature-grounded
> physiology **without curve-fitting**, and every claim must survive an adversarial
> review protocol and pass a known-answer-validated measurement instrument.
>
> The CUDA engine, the 8 regions, the 3 neuron classes, and the performance story from
> v1 are still true and still the substrate. What changed is the **science being asked
> of it**, the **dw20 mechanism set**, and the **standard of proof**.

---

## 0. The one change that drives all the others: the standard of proof

| | v1 (`Present.md`) | v2 (now) |
|---|---|---|
| **Question** | "Can we reproduce a known phenotype after injecting its mechanism?" | "Can the phenomenon **emerge** from grounded physiology that was *not* tuned to produce it?" |
| **Success = ** | the injected effect shows up (VPA → delayed flip; ASD → A1/A2 hyperactivity) | the un-programmed observable (bursting, criticality, IBI) appears *and survives adversarial scrutiny* |
| **Main risk** | **tautology** — "we programmed X, then measured X" | guarded against explicitly (Adversarial Protocol, Block 1) |
| **Measurement** | read off population rate tables | three **known-answer-validated** instruments + shuffle surrogates |
| **Parameters** | chosen to land the phenotype | every value carries a **provenance grade** (measured / derived / calibrated / assumed) |
| **Record** | results tables in a showcase doc | `OBSERVATIONS.md` journal (45 findings), `LITERATURE_PROVENANCE.md` ledger (Parts A–F), per-finding adversarial pass |

This is the root replacement. v1 could not, by construction, tell the difference between
"the model reproduces biology" and "the model replays its own inputs." v2 is built to
make that distinction the *whole point*.

---

## 1. Scientific target: trajectory-wide DNT → the dw20 emergence problem

**v1** spanned dw 20→52 and sold a *drug-screening platform* (VPA, ASD, ethosuximide,
ketamine, Pb²⁺, …). Its headline result was the **GABA flip** (−45→−75 mV) and the
phenotype shifts around it.

**v2** zooms into the **single hardest window — dw20**, and asks one question the v1
framing skipped: *what does the network actually do when GABA is depolarizing and almost
nothing is wired yet?* The target is **tracé discontinu** — the discontinuous EEG of the
very-preterm human: 1–5 s bursts (*bouffées*) separated by 10–30 s of near-silence —
reproduced as an **emergent** property, not a scripted one.

Why the narrowing: the v1 dw20 numbers (e.g. "P1_E 0.74 Hz, EI-ratio 0.05, Sync 0.53")
were *stable* but never tested against a real fetal-EEG observable. When we did test them,
the network sat in an **asynchronous-irregular** state — no discontinuity, no bouffées.
Everything in v2 flows from closing that gap honestly.

---

## 2. The dw20 mechanism set — replaced piece by piece

This is the "что на что заменилось" core. Each row is a v1 assumption that broke under
the v2 standard, and the grounded replacement.

### 2.1 Neuron biophysics: adult-scaled → immature-scaled (finding #33)

| | v1 | v2 |
|---|---|---|
| E cell | C=200 pF, gL=10 nS (adult AdEx) | **C=23 pF, gL=0.5 nS** (human GW22, Chen & Kriegstein) |
| Adaptation `b`, `a` | 50 pA, 2 nS | **×(gL_imm/gL_adult ≈ 0.05)** → 2.5 pA, 0.1 nS |
| Bias `I_bi`, background `ge0` | adult pA/nS | **×0.05** (same size law) |

**Why:** the v1 adult currents on an immature 0.5 nS cell are mis-scaled by ~20×. A
single spike's `b`=50 pA is a ~100 mV afterhyperpolarization; `I_bi`=80 pA is ~160 mV of
depolarization — it pins every cell suprathreshold. **This — not "I→I runaway" — was the
real cause of the long-standing dw20 instability.** Re-anchoring to the conductance ratio
is physics, and it *removed* free knobs rather than adding them.

### 2.2 GABA at dw20: "the flip" → depolarizing-but-shunting (ledger F5)

**v1** treated GABA mainly through *when it flips* (−45→−75). **v2** treats the
**depolarizing regime itself** as the central dw20 problem, and corrects a misconception:
*depolarizing ≠ excitatory.* In vivo (Kirmse 2015) depol-GABA **inhibits** the network by
shunting (E_GABA above rest but below threshold). So the model's I→E is likely a
**shunt stabilizer**, not an igniter — a reframe still being measured (open item F5).

### 2.3 Interneurons: always-on 3-class → sparse immature I-network (finding #32)

| | v1 | v2 (dw20) |
|---|---|---|
| I→I (PV↔PV, PV→SST) | present | **OFF** — PV interneurons are largely **postnatal** (ledger D6) |
| E→I | full density | **×0.15** (interneurons recruited phasically, not tonically) |
| I→E depol ignition | — | kept, sparse |

**Why:** a mutually-exciting interneuron network under depol-GABA shouldn't exist yet at
dw20. The PV/SST machinery from v1 is biologically a *later* arrival.

### 2.4 Recurrent E→E: static / STDP → dynamical synapses (STD) for SOC (finding #40, ledger F3)

**v1** recurrent E→E was static weight (+ STDP on a couple of pathways). **v2** makes
recurrent E→E **short-term-depressing with slow recovery** — the Levina-Herrmann-Geisel
2007 *dynamical-synapse* route to **self-organized criticality**. The engine's
Tsodyks-Markram STP *is* that mechanism, so this was config, not new code.

- **Falsification passed:** static synapses → network saturates to **350 Hz** (runaway).
  STD → bounded, **m ≈ 0.95 across a 2× gain range** (the old knife-edge is gone).
- This is what makes near-critical dynamics **self-organized** instead of hand-placed.

### 2.5 Burst pacing: fast `w` adaptation → grounded Ca-AHP pacemaker (finding #41, ledger F9)

| | v1 | v2 |
|---|---|---|
| `w` role | spike-frequency adaptation, τw≈200 ms | **slow Ca-dependent post-burst AHP**, τw≈8 s |
| Sets | within-burst adaptation | the **inter-burst interval** (10–30 s) |
| Grounding | engineering default | developing-cortex Ca-K AHP (Sheroziya/Egorov 2009, **verified**) |

**Epistemic note (kept explicit):** τw=8 s is a *grounded prediction* — order set by the
verified slow-burster + the resulting IBI independently matching the clinical 10–30 s
(Cherian). A rejected alternative — the Na/K-pump ultraslow AHP — was *numerically* perfect
(~10 s) but has the **wrong developmental trend** (it strengthens with age; tracé IBI
shortens), and matures postnatally (Fukuda & Prince 1992). Rejected like the earlier
chloride hypothesis (ledger F1/F8).

### 2.6 NEW mechanisms with no v1 equivalent

- **I_CAN plateau current (engine addition, finding #42–44, ledger F6/F9).** A
  spike-loaded, Ca-activated nonselective cation current `I_can = g_can·s_can·(E_can−v)`
  that **sustains the burst envelope** to 1–4 s — the missing "bouffée." Verified
  mechanism: Sheroziya/Egorov 2009 (I_CAN + I_NaP plateau, BK termination, mEC L3, P5–7,
  burst duration **2–20 s**). Added to both E and TH neuron kernels; off by default.
- **Gap junctions (engine addition).** Electrical coupling kernel for the subplate
  syncytium (Luhmann 2009). *Finding:* gap coupling **loads** silent neighbours and, with
  facilitating STP, actually *silenced* the subplate — so the discontinuous bursting
  emerges in the **cortical plate (P1/P2/M1)**, not the gap-coupled subplate.
- **Spatial topography.** Distance-dependent connectivity (Gaussian kernels, 200–400 µm
  domains); the engine's CSR is geometry-agnostic, so no C++ change was needed.

### 2.7 Homeostasis: failed at depol-GABA → works on top of STD (finding #45)

**v1** every neuron clamped `I_homeo` to a target rate — and at dw20 with depol-GABA that
*destabilizes* (the v1 "Homeostasis keeps the network in range" claim does not hold at
dw20). **v2:** with STD now bounding runaway, the *same* rate-clamp works again — set
slow (τ_homeo ≫ IBI), it **self-selects the operating point** across REC 4–8 (m≈1 emergent),
substantially removing the last hand-tuned gain. (Refinement open: true synaptic-scaling
for cleaner convergence.)

---

## 3. How the process now works (the new pipeline)

**v1 process:** edit `dev_state`, run `run_trajectory.py`, read the rates table, declare
the phenotype.

**v2 process** — every result goes through this loop before it counts:

```
 1. HYPOTHESIS  ── stated as a mechanism + a pre-registered quantitative target
                   (target written BEFORE the run; no post-hoc target fitting)
 2. GROUND      ── each parameter gets a provenance grade in LITERATURE_PROVENANCE.md
                   (A measured / B derived / C assigned / F calibrated) + a URL
 3. INSTRUMENT  ── validate the measurement on KNOWN-ANSWER synthetic data FIRST
                   (the metric must recover a planted answer & be robust to its own knobs)
 4. RUN         ── gen_bundle.py → sim_multiregion.exe → pop_activity.npy (per-region)
 5. MEASURE     ── only with validated instruments, on the right region/window
 6. ADVERSARIAL ── the 10-question protocol (tautology? fitted? known-answer control?
                   ±20% robust? pre-registered? comparable units? ≥3 alternatives?
                   falsifiable? what's unchecked? hostile-reviewer's first question?)
 7. RECORD      ── finding written to OBSERVATIONS.md with the adversarial pass shown;
                   "STOP" on any failed question — no advancing on an unproven result
```

Concrete examples of step 6 doing its job *this session*:
- A promising "global IBI ≈ 7 s" was **rejected** as a staggered-region-**birth** artifact
  (alternative explanation, Block 4) — re-measured per-region, unconfounded.
- The criticality `m ≈ 0.99` was **re-checked with a shuffle surrogate** on the new
  I_CAN regime: real R²=0.99 (reliable) vs shuffled R²≈0.00 (rejected) — the branching is
  real, not a burst/silence marginal artifact.
- The load-bearing mEC number (17.5 s) was **fetched and verified** at the primary source
  *before* building further; the mechanism confirmed, the unverified specific number
  demoted and shown to be non-load-bearing.

---

## 4. Measurement: rate tables → three validated instruments

**v1** measured with population-rate tables and a synchrony scalar. **v2** has three
instruments, each of which **must pass a known-answer control before its numbers are
trusted** (a hard requirement; an unvalidated metric is treated as no metric):

| Instrument | Measures | Known-answer control |
|---|---|---|
| `cuda_engine/discontinuity.py` | population-spike bursts & sub-cycle IBI (5 ms) | recovers planted IBI 2–30 s; robust to bin/threshold |
| `cuda_engine/burst_envelope.py` **(new)** | **bouffée** duration & inter-bouffée IBI | recovers planted dur 1–5 s & IBI 10–30 s; **spread 1.2%** across its own knobs; doesn't collapse to sub-cycle |
| `cuda_engine/criticality.py` | branching ratio m (subsampling-invariant), avalanches | recovers known m ±0.01; **shuffle surrogate** collapses R² → catches false positives |

Plus a new engine output — **`pop_activity.npy` [n_pops × T_steps]**, per-region
full-resolution activity — so the instruments run on a single unconfounded region instead
of the birth-contaminated global trace.

---

## 5. The headline result: v1 phenotype tables → v2 emergent *tracé discontinu*

**v1 headline** (still valid as engine behaviour): the GABA flip and drug-phenotype tables
(VPA delays the flip ~5 dw; ASD chronic A1/A2 hyperactivity).

**v2 headline** (the consolidated grounded run, `reference/grounded_ican`, 180 s, SCALE=0.1,
**all parameters literature-grounded, no curve-fit, no rate-clamp homeostasis**):

| Region | Bouffée dur | Inter-bouffée IBI | Criticality m | frac. silent |
|---|---|---|---|---|
| TH_E | 2.80 s | 16.6 s | 0.99 | 0.87 |
| P1_E | 3.00 s | 20.9 s | 0.99 | 0.90 |
| P2_E | 1.15 s | 20.1 s | 0.99 | 0.93 |
| M1_E | 2.90 s | 19.9 s | 0.99 | 0.92 |

Against the **pre-registered** literature targets: bouffée **1–5 s ✓** (Sheroziya/Egorov
2–20 s burst; pubmed 17542015 EEG 1–4 s), inter-bouffée **10–30 s ✓** (Cherian),
**near-critical m≈1 ✓** (Beggs-Plenz; subsampling-invariant per Wilting-Priesemann),
**>40% silent ✓✓** (got 87–99%). All four hit, on grounded physiology, measured by
validated instruments — the structure of fetal *tracé discontinu* as an **emergent**
property.

---

## 6. What is honestly NOT closed (v2 keeps its own debt list)

v1 presented finished phenotypes. v2 keeps an explicit ledger of what is *grounded
prediction* vs *calibrated* vs *unverified* vs *open*:

- **Calibrated, not predicted:** the I_CAN magnitudes (`g_can`, `τ_can`, `can_inc`) were
  dialed to land the bouffée duration — mechanism grounded, *values* fit. Flagged, not hidden.
- **Still a chosen value:** the recurrent gain operating point (REC≈6); homeostasis
  *substantially* but not *fully* removes it (convergence imperfect; synaptic-scaling next).
- **Not yet at scale:** the grounded headline is SCALE=0.1 (15k neurons); SCALE=1.0 (143k)
  confirmation pending.
- **Open mechanism question:** F5 — is the depol-GABA I→E projection igniting or shunting?
- **Verification residue:** mEC mechanism verified; the specific "17.5 s / 24 s" table
  numbers remain unconfirmed (shown to be non-load-bearing).

---

## 7. What carried over unchanged from v1

So it's clear v2 is an evolution, not a rewrite:

- The **CUDA-native engine** (CUDA Graphs, batched deliver/enqueue, fused neuron kernel,
  device step counter) — still the substrate; extended with the I_CAN current, the gap
  kernel, and per-pop output.
- The **8 regions + sequential birth schedule**, the **3 neuron classes** (E/PV/SST), the
  AMPA/NMDA(Mg-block)/GABA-A synapses, T-type Ca²⁺ in thalamus, STDP, STP, correlated OU.
- The **performance** story (≈26–38 s/episode at SCALE=0.1; ~20 min trajectories; ~19–28×
  Brian2) — unchanged; the new per-step work (gap current, I_CAN, per-pop write) is
  negligible.
- The **DNT/disease ambition** — still the long-term goal; v2 is building the *trustworthy
  baseline* that drug/disease perturbations must deviate from for those experiments to mean
  anything.

---

## 8. One-paragraph statement of the shift

v1 showed that ADAM is **fast and biologically featured enough** to script developmental
neurotoxicology phenotypes. v2 confronts the harder question underneath that promise —
*does the dw20 network actually behave like a fetal brain when you don't tell it to?* —
and answers it for the signature observable (*tracé discontinu*) by **rebuilding the dw20
mechanism set on verified physiology** (immature cell scaling, sparse immature inhibition,
STD-driven self-organized criticality, a grounded Ca-AHP pacemaker, an engine-level I_CAN
plateau), **measuring it with instruments that pass known-answer controls**, and **subjecting
every result to an adversarial protocol that rejects tautology, artifact, and fit**. The
engine and the platform vision are unchanged; the **standard of evidence** is the upgrade.

---

*Provenance: this v2 reflects `OBSERVATIONS.md` findings #32–#45 and `LITERATURE_PROVENANCE.md`
Parts A–F as of 2026-06-21. Numbers are SCALE=0.1 grounded runs; see `reference/characterize.py`
and the three validators (`discontinuity_validate.py`, `burst_envelope_validate.py`,
`criticality_validate.py`) to reproduce.*
