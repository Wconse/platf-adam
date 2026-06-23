# Grounded Emergence — A Development Plan for ADAM

> **Thesis:** Dial in what genes dial in. Let emerge what activity grows.
>
> The goal is not "maximize emergence." Maximal emergence is chaos, and chaos is
> useless. The goal is to make the model's split between *scheduled* and *emergent*
> match the real brain's own split between *genetically programmed* and
> *activity-dependent* development — and then validate that what emerges
> spontaneously resembles what emerges spontaneously in biology.
>
> **The discipline (non-negotiable):** one conversion at a time. Each conversion is
> only "done" when it passes the three emergence tests (below). No new conversion
> starts until the previous one is validated or rolled back. This is what keeps us
> from drowning in beautiful, meaningless dynamics.

---

## The Three Emergence Tests

An emergent property only *counts* — scientifically and for this project — if it passes all three:

1. **Unprogrammed.** The property was not written down anywhere. We imposed a *mechanism*, not the *phenomenon*. (If we drew the curve, it doesn't count.)
2. **Matches held-out data.** It resembles a real biological phenomenon we did **not** tune for, ideally quantitatively.
3. **Robust basin, not knife-edge.** Small parameter changes don't destroy it. Real emergence is a *basin of attraction*, not a single lucky setting. If the property vanishes when you nudge a parameter 10%, it's a coincidence, not emergence.

Every phase in Part II ends by checking these three. Test 3 is the most often skipped and the most important — it's what separates a genuine finding from a fitted artifact.

---

# PART I — EXPLANATORY

## 1. What "grounded emergence" means precisely

A model component can be written in one of two ways:

- **Phenomenological (dial-in):** we write the *outcome* directly.
  *Example:* `E_GABA(dw) = sigmoid(...)` — we draw the GABA-flip curve.
- **Mechanistic (generative):** we write the *underlying process*, and the outcome
  *emerges*. *Example:* model intracellular chloride; let it accumulate from GABA
  input and be pumped out by KCC2; `E_GABA` then *falls out* of the chloride balance.

Grounded emergence is the systematic conversion of components from the first kind to
the second kind — **but only where the real brain also lets that outcome emerge from
activity**, and **only where we can check the result against reality.**

## 2. The brain's own division of labor

Developmental neuroscience has a well-established dichotomy (Katz & Shatz 1996;
the "protomap vs protocortex" debate):

| Activity-INDEPENDENT (genetic clock) | Activity-DEPENDENT (emerges from activity) |
|---|---|
| Neuron birth and migration timing | Fine synaptic refinement |
| Broad areal identity (which region is what) | Topographic map sharpening (tonotopy, retinotopy) |
| Receptor subunit expression schedule (NR2B→NR2A) | Ocular dominance / columnar structure |
| Gross GABA-switch program (KCC2 *capacity* ramps up) | Exact chloride setpoint / `E_GABA` fine value |
| Coarse initial wiring | E/I balance tuning |
| | Critical-period opening and closing |
| | Which weak synapses survive ("use it or lose it") |

**The principle for ADAM:** put each model component on the *same side* of this line
as biology puts it. Things on the left **stay dialed in** — and that is correct, not
a cop-out. Things on the right are **candidates for conversion to emergent.**

## 3. Inventory: ADAM's current dial-ins, classified

| Component | Currently | Biology says it should be | Action |
|---|---|---|---|
| Neuron birth times (`t_bir`) | scheduled | genetic | **keep scheduled** ✓ |
| `Mg`, `nmda_ratio` curves | scheduled | genetic (subunit switch) | **keep scheduled** ✓ |
| Region identity / which pops exist | scheduled | genetic | **keep scheduled** ✓ |
| KCC2 *capacity* ramp | scheduled | genetic | **keep scheduled** (the ramp) |
| `E_GABA` exact value | dialed (sigmoid/target) | **activity-dependent** | **CONVERT** → chloride dynamics |
| Long-range connectivity (`lr_gate` timer) | scheduled timer | **activity-dependent** | **CONVERT** → activity-grown |
| Synaptic pruning (weight threshold) | weight-threshold | **activity-dependent** | **CONVERT** → "use it or lose it" |
| Firing rates (homeostasis clamp) | clamped to `r_tgt` | **emergent**, homeostasis only stabilizes slowly | **CONVERT** → soft homeostasis |
| Topography / spatial maps | absent (random) | activity-dependent | **DEFER** (needs spatial rewrite) |

The four CONVERT rows are the program. The DEFER row is honest scope-limiting:
adding space is a near-total architecture change and is out of scope for now.

## 4. What we let emerge, what stays scheduled — and why

**Stays scheduled (and that's biologically correct):**
- Neuron birth, region identity, receptor-subunit timing, the *capacity* of the KCC2
  pump to ramp up. These are gene-expression clocks in reality. Modelling them as
  activity-emergent would be *wrong*, not more realistic.

**Converted to emergent:**
- **`E_GABA`:** real GABA reversal depends on the chloride load a neuron has actually
  experienced. Two neurons of the same age can have different `E_GABA` if one got more
  GABA input. We'll model chloride; `E_GABA` emerges.
- **Long-range wiring:** real long-range axons find and stabilize targets by correlated
  activity ("cells that fire together wire together," extended to wiring itself).
  We'll let long-range synapses grow/stabilize from correlation; connectivity emerges.
- **Pruning:** real synapses compete; inactive ones are eliminated. We'll prune by
  *activity*, not by a static weight threshold.
- **Firing rates:** real rates are set by the balance of E, I, and inputs; homeostasis
  is a *slow* stabilizer, not a clamp. We'll soften homeostasis; rates emerge.

## 5. What each conversion buys (scientifically) + its validation target

| Conversion | Scientific payoff | Validation target (held-out) |
|---|---|---|
| Emergent `E_GABA` | GABA-flip timing becomes *predicted* from activity, not drawn. Activity changes flip timing — a real, documented effect. | Activity-dependence of GABA reversal in slice physiology (direction + rough magnitude). |
| Activity-dependent pruning | Connectivity becomes *self-sculpted*; pruning rate tied to activity. | Developmental synapse-density curve shape (over-production then pruning). |
| Activity-grown long-range wiring | The flagship result: **maps/connectivity that wire themselves.** | Correlated regions become more strongly connected than uncorrelated ones (no such bias was imposed). |
| Soft homeostasis (emergent rates) | Removes the tautology. Rates become a *result* you can be surprised by. | Developmental E/I-ratio trajectory; self-correction of perturbations. |
| **(cross-cutting) Criticality** | If the constrained network self-organizes near criticality, avalanche sizes follow a power law **we never imposed.** | Neuronal avalanche power law (Beggs & Plenz 2003: size exponent ≈ −1.5). **This is the cleanest quantitative emergence validation available.** |

Criticality is special: it's not a conversion, it's the *measurement that tells us
whether our emergent dynamics are brain-like at all.* It's built first (Phase 0) and
checked after every conversion.

## 6. The meaning of success

Success is **not** "the network does complex things." Success is: *we removed a
dial-in, replaced it with a biological mechanism, and the phenomenon re-appeared on
its own AND matched data we didn't tune for AND survived parameter perturbation.*

If even one conversion achieves this — e.g., the GABA-flip emerges from chloride
dynamics and its activity-dependence matches slice data — that is a genuine,
publishable, non-tautological result. That single result is worth more than every
mechanism added so far.

---

# PART II — ENGINEERING

## Per-phase protocol (the loop, applied to every phase)

```
1. State the biological constraint (what real mechanism are we encoding?)
2. State what should emerge (the phenomenon we are NOT allowed to write directly)
3. Implement the mechanism (new state variable + update rule)
4. STABILITY GATE: run baseline (dw 20→52). Does it stay alive and bounded?
                   If runaway/dead → tune the CONSTRAINT (not the outcome) or roll back.
5. EMERGENCE TEST 1: confirm we did not write the phenomenon anywhere.
6. EMERGENCE TEST 2: measure the phenomenon, compare to held-out biological data.
7. EMERGENCE TEST 3: perturb parameters ±10–20%. Does the phenomenon survive?
8. If all pass → freeze, document, move to next phase. Else → diagnose, iterate, or roll back.
```

Rollback is a first-class outcome, not a failure. A clean rollback with a documented
reason ("chloride dynamics destabilize without a leak term we can't justify") is a
real scientific result.

---

## Phase 0 — STATUS: ~40% (instruments built, criticality metric DOES NOT WORK yet)

**Honest status after two rounds of external review. Both criticality estimators
FAILED control tests on surrogate data — withdrawn. But one real, estimator-independent
finding emerged about the NETWORK, which is more valuable.**

### Criticality measurement: FAILED (do not use yet)
- Avalanche analysis (exponent, σ): unreliable — activity too dense (only 18% empty
  bins with homeostasis), both metrics drift with arbitrary bin size. Withdrawn.
- MR-estimator (my implementation): **failed surrogate control** — returns m≈1 for
  pure Poisson (true m=0) and does not recover known AR branching ratios. Withdrawn.
- A reliability gate now forces `reliable=False` when fit R²<0.9, so it can no longer
  emit a false number. But there is currently NO working criticality metric.
- **Correct fix (deferred):** use the published `mrestimator` package
  (Spitzner et al. 2021) instead of a hand-rolled fit — it handles drive correction,
  lag selection, and offset properly. Requires install + sign-off, not an autonomous
  rush. Two rushed attempts already failed; the lesson is to stop.

### REAL finding (estimator-independent, robust): homeostasis overheats the network
Disabling homeostasis (I_scl=0) changed the network qualitatively:

| | Homeostasis ON | Homeostasis OFF |
|---|---|---|
| total spikes (12s) | 310,169 | 34,643 (9× fewer) |
| empty timestep fraction | 18% | **76%** |

Without homeostasis the dw-20 network is SPARSE and burst-pause-like (76% silent) —
biologically correct for early prenatal cortex (spindle bursts, delta brushes with long
silent intervals). WITH homeostasis it is overheated into near-continuous firing.
**This means the homeostatic clamp is pushing the network into a biologically
implausible regime at early dw.** This is a network-calibration problem, surfaced by
Phase-0 instrumentation — exactly the kind of thing the instruments exist to catch.

### What Phase 0 still owes (per review, honestly incomplete)
- [ ] A working criticality metric (mrestimator package)
- [ ] Emergence-audit harness actually RUN with multi-seed CI (written, not validated)
- [ ] Perturbation runner actually RUN (written, not validated)
- [ ] Quantitative pre-registered targets (number + window + effect size + source)
- [ ] Resolve the homeostasis-overheating / activity-density question (NETWORK issue)

### Next step (per reviewer, and I agree): fix the NETWORK, not the instrument
The density finding says the dw-20 baseline may be mis-calibrated (too hot). Before any
emergence conversion, understand why homeostasis overheats it and whether the sparse
no-homeostasis regime is the biologically correct baseline. This is about the
correctness of the network itself — upstream of every later phase.

### UPDATE (round-3 review → went to the library): found the real baseline defect
Quantitative literature targets are now recorded in `VALIDATION_TARGETS.md`. The key
finding: real early cortex (dw 20–28) is DISCONTINUOUS (tracé discontinu) with
inter-burst intervals of SECONDS. The model's max pause is ~362 ms — it is in the wrong
dynamical regime (quasi-continuous) in BOTH homeostasis on/off variants.

- This is a sharper, sourced, falsifiable defect than "too hot."
- It explains why criticality wouldn't measure: wrong regime entirely.
- Hypothesis (pre-registered): constant Poisson afferent drive prevents seconds-long
  silences; gating it at early dw should lengthen IBI toward seconds.
- Caveat: 12 s episodes fit only ~2 bursts at IBI~6 s → IBI validation needs longer
  runs. This is a methodological decision to confirm before testing.

**The reviewer was right three rounds running. The discipline that worked: stop, go to
the literature, pre-register a number, THEN touch the network.**

---

## Phase 0 — original spec (superseded by status above)

**Built:** network activity trace in the engine (`net_activity.npy`), criticality
module, fingerprint harness, perturbation runner.

**Lesson (this is the real result of Phase 0):** the first criticality estimator —
classic avalanche analysis (size exponent, σ) — turned out to be **unreliable on this
network's data.** Activity is too dense (only ~18% of timesteps are empty), so the
"avalanche = run of non-empty bins" definition degenerates and both the exponent
(1.37–2.05) and σ (0.77–0.97) drift with the arbitrary bin choice. An early claim of
"σ=0.87 / exp=1.29 = emergent criticality" was **withdrawn** — it was a number from an
uncalibrated instrument. (Caught before it entered any results. This is exactly why
Phase 0 builds the ruler first.)

**Fix:** replaced the fragile avalanche estimator with the **MR-estimator**
(Wilting & Priesemann 2018), which infers branching ratio `m` from autocorrelation
decay and needs no avalanche separation — built for dense/subsampled data. Result:
`m ≈ 0.999` (near-critical, slightly subcritical), **stable across bin sizes**
(0.9986–0.9992), stable across episodes. Fit R² ≈ 0.6–0.85 (decay not perfectly
single-exponential — flagged honestly). The criticality module now self-reports
`reliable: False` when activity is too dense for avalanche analysis.

**Primary criticality metric going forward: MR `m`, not avalanche exponents.**

**NEXT (before Phase 1), per review discipline — must be done WITH the user, not autonomously:**
1. Pre-register quantitative targets from literature (e.g. mature networks m→1;
   early networks slightly subcritical; **prediction: VPA should push m UP toward
   supercritical in the dw 28–36 window** — this is unprogrammed and testable = Test 2).
2. Run Test 3 (perturbation robustness of `m`) only after targets are written down.

---

## Phase 0 — Instruments first (no behavior change) [original spec below]

**Why first:** you cannot pursue grounded emergence without the instrument that
*measures* groundedness. Build the ruler before reshaping the thing.

**Build:**
1. **Avalanche / criticality metrics** (`metrics.py`): detect network events, measure
   avalanche size & duration distributions, fit power-law exponent, compute branching
   ratio σ (σ≈1 ⇒ critical). Target reference: size exponent ≈ −1.5, σ ≈ 1.
2. **Emergence-audit harness:** a script that runs the current baseline and records the
   full "fingerprint" — rates, E/I ratio, burst stats, avalanche exponents, synchrony,
   per-region — so every later conversion is compared against a frozen reference.
3. **Perturbation runner:** wraps a trajectory in a ±X% parameter sweep, for Test 3.

**Risk:** none (measurement only). **Effort:** small. **Gate:** baseline fingerprint
recorded and reproducible across seeds (with CI).

---

## Phase 1 — Emergent `E_GABA` from chloride dynamics

**Biological constraint:** intracellular chloride `[Cl⁻]ᵢ` rises when GABA-A channels
pass chloride inward and is pumped out by KCC2 at a *capacity* that ramps genetically.
`E_GABA` is then the Nernst potential of chloride — a *consequence*, not an input.

**What must emerge:** the GABA-flip itself, and its dependence on activity.

**Implementation sketch:**
- New per-neuron state `cl_i` (intracellular chloride).
- Update each step:
  ```
  d[cl_i]/dt = α · I_GABA_flux        (GABA activity loads chloride)
             − KCC2_cap(dw) · ([cl_i] − cl_rest)   (pump removes it)
  E_GABA = (RT/F) · ln([cl_o]/[cl_i])   (Nernst; or linearized)
  ```
- `KCC2_cap(dw)` is the **genetic ramp** — stays scheduled (correct). What's new is
  that the *actual* `E_GABA` now depends on how much GABA the neuron received.
- VPA = lower `KCC2_cap` ramp (mechanistic, as before, but now even more grounded).

**Validation (Test 2):** in a high-activity vs low-activity sub-population at the same
age, `E_GABA` should differ — and flip later in low-activity neurons. This matches the
documented activity-dependence of GABA reversal.

**Risk:** MEDIUM. Chloride loops can oscillate. Mitigate with the `cl_rest` leak and
conservative `α`. **Self-contained:** YES (per-neuron variable).
**Why first conversion:** flagship phenomenon, self-contained, moderate risk.

---

## Phase 2 — Activity-dependent pruning ("use it or lose it")

**Biological constraint:** synapses that are rarely co-active with their target are
eliminated; active ones are stabilized. Pruning is driven by *correlation*, not by a
static weight threshold.

**What must emerge:** the surviving connectivity pattern and the over-production →
pruning density curve.

**Implementation sketch:**
- Per-synapse slow "usefulness" trace `c_syn` that integrates pre-post coincidence
  (cheap: reuse STDP traces where available, add a slow eligibility elsewhere).
- In the slow loop, prune synapses whose `c_syn` is low — *not* whose weight is low.
- Keep total-synapse bookkeeping to plot the density curve.

**Validation (Test 2):** synapse-density-over-development should show the
characteristic overshoot-then-prune shape; survivors should be the co-active ones.

**Risk:** LOW (slow-loop, localized). **Why second:** low-risk win that also makes
connectivity start to self-sculpt, paving the way for Phase 3.

---

## Phase 3 — Activity-grown long-range wiring (replaces `lr_gate` timer)

**Biological constraint:** long-range cortico-cortical axons stabilize on targets with
correlated activity. Connectivity *grows*, it isn't switched on by a clock.

**What must emerge:** the strength and pattern of inter-region connectivity, and its
developmental timing.

**Implementation sketch:**
- Replace the scalar `lr_gate` with per-long-range-synapse weights that strengthen via
  a slow correlation rule (a slow-timescale STDP/Hebbian term on long-range projections),
  gated by a *genetic permissive window* (axons can't grow before they arrive — that
  arrival is genetic and stays scheduled).
- Net effect: long-range coupling *emerges* during the permissive window, at a rate set
  by inter-region correlation — strong where regions are correlated, weak where not.

**Validation (Test 2):** correlated region pairs end up more strongly wired than
uncorrelated pairs, **though no such bias was imposed.** This is the flagship
"wires itself" result. Also: emergent connectivity onset should land near the old
dw 34–38 window *without being told to.*

**Risk:** MEDIUM. Could under-wire (regions stay disconnected) or runaway. The genetic
permissive window bounds it. **Why third:** the highest-payoff emergence result, but it
needs sensible activity (Phases 1–2 done) to have meaningful correlations to learn from.

---

## Phase 4 — Soft homeostasis (emergent firing rates)

**Biological constraint:** homeostasis is a *slow* (hours) stabilizer that nudges
excitability toward a setpoint; it does **not** clamp instantaneous rate. Rates are set
by E/I balance and inputs.

**What must emerge:** the firing rates themselves, and the developmental E/I trajectory.

**Implementation sketch:**
- Lengthen homeostatic time constants drastically and cap its authority (limit how much
  `I_homeo` can shift), so it stabilizes drift over an episode but doesn't pin rates.
- Rates now settle from the network. Report them as *results*, with CI across seeds.

**Validation (Test 2):** the developmental E/I-ratio trajectory should be sensible and
*self-correct* after a transient perturbation (rather than being forced). The avalanche
exponent (Phase 0 instrument) should sit near criticality *without* being tuned to.

**Risk:** HIGH. This removes the safety net; the network may runaway or fall silent.
**Why last:** it's the deepest dial-in and the scariest. Do it only after the
instruments (Phase 0) and the other mechanisms give the network enough internal
structure to self-stabilize. This is also the conversion that, if it works, most fully
removes the tautology critique.

---

## Deferred (honest scope limit)

- **Spatial topography / maps (Conversion D):** requires giving neurons positions and
  distance-dependent connectivity — a near-total architecture rewrite. Genuinely
  valuable (it's the one gap that blocks migration-type phenomena) but out of scope for
  this program. Documented so it isn't silently ignored.
- **Glia, neuromodulators (DA/5-HT/ACh):** real and important for development, but each
  is its own research program. Not part of grounded-emergence v1.

---

## Risk register

| Risk | Where | Mitigation |
|---|---|---|
| Network runaway/silence after removing a clamp | Phases 1, 4 | Stability gate before any other test; genetic constraints bound dynamics; rollback is allowed |
| Beautiful-but-meaningless dynamics | all | Test 2 (held-out data) is mandatory; no result counts without it |
| Coincidental "emergence" (knife-edge) | all | Test 3 (perturbation/robustness) is mandatory |
| Doing too much at once | all | One conversion per phase; hard gate between phases |
| Mistaking dial-in for discovery | all | Test 1; the audit harness flags anything we wrote directly |
| Parameter explosion | all | Each mechanism's parameters must be literature-grounded or explicitly flagged as free |

---

## Ordering summary

```
Phase 0  Instruments        (no risk)        ← build the ruler
Phase 1  Emergent E_GABA     (medium risk)    ← flagship, self-contained
Phase 2  Activity pruning    (low risk)       ← low-risk win, sculpts connectivity
Phase 3  Activity wiring     (medium risk)    ← flagship emergence: "wires itself"
Phase 4  Soft homeostasis    (high risk)      ← deepest; kills the tautology
Deferred Spatial topography  (out of scope)   ← honest limit
```

Each arrow is a gate. We do not pass a gate until the three emergence tests pass or we
record a clean rollback. The whole program can stop after any phase and still have
produced a real result — that is by design.
