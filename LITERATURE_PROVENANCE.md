# Literature Provenance & Debt Ledger

**Purpose.** For every model parameter or validation target that claims biological
grounding, this document states explicitly: (1) what the model uses, (2) what the
literature *actually measures*, (3) the proxy substituted when no direct measurement
exists, (4) the inference chain converting proxy → model scalar, and (5) confidence.

**Why this exists (methodological).** Anatomical/physiological work almost never
provides the scalar a model needs directly (e.g. "perisomatic inhibitory conductance
as % of adult at gestational week W"). It provides *proxy markers* — immunoreactivity
onset, cell density by age, receptor-subunit switches, transporter expression — each
*correlated* with the process but not a direct measurement of synaptic conductance.
Turning a proxy into the model's curve is **an additional inferential step with its
own uncertainty**. If that step is not stated, a pre-registration looks like "found
in the literature" when it is actually "inferred from a proxy." This ledger makes
each such step explicit so no result silently inherits hidden inference.

**Confidence key.**
- **A (anchored):** a number we directly fetched/verified this session from a primary
  source; used close to as-measured.
- **B (proxy-inferred):** literature measures a *correlated proxy*; the model scalar
  is an inference from it. Flagged as assumption.
- **C (cross-species / cross-scale translated):** measurement exists only in another
  species (usually rodent) or another scale; translated to human gestational weeks /
  to our subsampled network. Compounded uncertainty.
- **F (free / pragmatic):** not grounded in a specific measurement; a modelling choice.
  Must be flagged in any emergence claim and ideally varied for robustness (Test 3).

> **Source-verification caveat.** Web *search* was blocked this session; anchors marked
> [FETCHED] were retrieved via direct WebFetch from the cited PMC/Wikipedia pages.
> Items marked [TRAINING] rest on background knowledge and MUST be verified against the
> primary source before any external/publication use. This caveat is itself a debt.

================================================================================
## PART A — VERIFIED ANCHORS (direct fetch this session)
================================================================================

| # | Fact | Value | Source | Conf |
|---|------|-------|--------|------|
| A1 | Subplate gap-junction dye-coupling | ~9 neurons/cell | Luhmann/Kilb/Hanganu-Opatz 2009, Front Neuroanat 3:19 [FETCHED] | A |
| A2 | Subplate gap-junction coupling conductance | ~1.2 nS | same | A |
| A3 | Subplate firing | >40 Hz; osc. bursts ~20 Hz | same | A |
| A4 | Subplate organization | columnar gap-junction syncytium | same | A |
| A5 | Intra-subplate glutamate STP | FACILITATES at 10–40 Hz | same | A |
| A6 | Min GABAergic density for synchrony | ~2 GABAergic SP cells/mm² | Voigt 2001 (via A1) | A |
| A7 | Cortical minicolumn | 30–50 µm, 80–120 neurons | Wikipedia/Mountcastle [FETCHED] | A |
| A8 | sAHP kinetics | max ~1 s, decay several s, needs >300 ms depol, V½≈−40 mV | KCa3.1 sAHP lit [FETCHED] | A |
| A9 | GDP duration / IBI | hundreds of ms / seconds | Ben-Ari, GDP [FETCHED] | A |
| A10 | GDP end | disappear as KCC2 matures, GABA→hyperpol | GDP [FETCHED] | A |
| A11 | Rat neocortex KCC2 | reaches adult ~P15; NKCC1 high neonatally | Cl-homeostasis review [FETCHED] | A (rodent) |
| A12 | Human KCC2/NKCC1 cortical timing | **NOT directly available** (only WNK3 mRNA) | same | — |
| A13 | Preterm EEG tracé discontinu | predominates <28 wk PMA; continuous ~35 wk | preterm EEG [FETCHED] | A |
| A14 | Delta brushes | appear 28–30, peak 32–35, gone 38–42 wk PMA | delta-brush lit [FETCHED] | A |
| A15 | Retinal-wave refractory period | 40–60 s | Retinal waves [FETCHED] | A |

================================================================================
## PART B — MODEL PARAMETERS: provenance & debt
================================================================================

### B1. E_GABA developmental ramp (−45 → −75 mV; per-region flip timing)
- **Model uses:** sigmoid per region, centers SP≈26.5, P1≈30, …, all hyperpol by dw34
  (dev_state.DEFAULT_GABA_FLIP).
- **Directly measured:** ECl / E_GABA reversal via gramicidin-perforated patch — in
  **rodent** slices (A11). Human cortical E_GABA timing per gestational week: **not
  available** (A12).
- **Proxy:** KCC2/NKCC1 expression ratio (rodent) → E_GABA polarity.
- **Inference chain:** rodent transporter expression → rodent E_GABA timing (Pxx) →
  **cross-species translation** to human gestational weeks → per-region offsets we
  *assigned* (subplate before cortical plate). Each arrow adds uncertainty.
- **Assumption flags:** (i) rodent→human time map (see B12); (ii) the per-region
  *ordering/offsets* (SP earlier than P1) are biologically motivated (subplate matures
  first, A4) but the exact dw centers are **not measured** — they are placed to span
  the dw20→34 window. (iii) the −45/−75 endpoints come from Nernst at "high"/"low"
  [Cl]i, themselves rodent-anchored.
- **Confidence:** C (cross-species + proxy). The *shape* (depol→hyperpol over dw20–34)
  is well-grounded; the *exact per-region timing* is an assumption.

### B2. Inhibition-strength maturation (NEW: weak@dw20 → strong@dw34)  ⚠ load-bearing
- **Model would use (proposed):** a scalar ramp of perisomatic/PV inhibitory weight
  (and/or interneuron recruitment) from ~weak at dw20 to adult by ~dw34.
- **Directly measured:** **NONE** gives "inhibitory synaptic conductance as % of adult
  at gestational week W." This scalar does not exist in the literature.
- **Available proxies (correlated, NOT the conductance):**
  - PV immunoreactivity onset (human cortex: PV+ neurons largely **perinatal/postnatal**;
    very sparse before ~term) [TRAINING — verify];
  - PV+ cell density vs age (rises through late fetal → infancy) [TRAINING];
  - GABA-A subunit switch α2→α1 (confers faster IPSCs; **largely postnatal** in human)
    [TRAINING];
  - perisomatic (basket) synapse formation onset [TRAINING];
  - the chloride/E_GABA switch itself (B1) — partly the same process.
- **Inference chain:** choose ONE proxy (recommend: a composite "perisomatic inhibition
  index" rising with PV density + α1 fraction) → map its time course onto dw20–34 →
  declare model inhibitory weight ∝ that index. This is **2–3 inferential steps** beyond
  any measured conductance, plus rodent→human translation.
- **Assumption flags (MUST be stated in any pre-registration):**
  (i) which proxy is the stand-in (we have NOT yet measured this — it is a CHOICE);
  (ii) linearity/sigmoid shape of the ramp is assumed, not measured;
  (iii) "weak inhibition → synchronized bursting" is the *model's* dynamical prediction
       (finding #28), not a literature datum — do not conflate.
- **Confidence:** B→C.
- **RESOLUTION (2026-06-20):** debt closed by REJECTING the parameter. Examining the
  proxies revealed they point to a *different mechanism*: PV/perisomatic inhibition in
  human cortex matures mostly **post-term** (PV immunoreactivity low prenatally, rises
  after ~term; α2→α1 switch largely postnatal) [TRAINING — verify]. So within our window
  (dw20–34) perisomatic inhibition is immature THROUGHOUT — there is no measured
  "strength ramp" in-window to ride on. What IS in-window is the GABA **polarity** flip
  (B1, KCC2). Biologically, at dw20 inhibition is not "weak" — it is **depolarizing
  (excitatory)**; the same interneurons become inhibitory as E_GABA flips. Therefore the
  bursting→continuous transition should be driven by **B1 (polarity)**, NOT by a separate
  inhibition-strength ramp. Finding #28 ("weak inhibition → synchrony") is reinterpreted:
  the correct biological knob is GABA polarity (already modelled), not an added strength
  curve. **Action: do NOT add the B2 ramp. Test the B1-driven transition instead.**
  This is the literature discipline steering away from a free curve dressed as biology.

### B3. sAHP time constant (tauw_E = 1500 ms)
- **Model uses:** AdEx adaptation tauw = 1.5 s as the IBI pacemaker.
- **Directly measured:** sAHP decay "several seconds", max ~1 s post-onset, V½≈−40 mV
  (A8) — but in **mature** pyramidal neurons, and as a *current/voltage* timecourse,
  not as the AdEx `w` time constant.
- **Proxy/inference:** AdEx `w` is a *mathematical analog* of sAHP (single exponential).
  Mapping the multi-component biological sAHP (Ca→KCa3.1 cascade) onto one tau is a
  **model reduction**. 1.5 s sits inside the measured 1–5 s band.
- **Assumption flags:** (i) single-exponential reduction; (ii) developmental value of
  sAHP in immature neurons is **not** the mature value used here (immature sAHP likely
  weaker/slower — unmeasured). (iii) IBI∝tauw is the model's relation (finding #6).
- **Confidence:** B. Order-of-magnitude grounded; exact value is a reduction.

### B4. NMDA/AMPA ratio developmental curve (early 0.50 → late 0.15)
- **Model uses:** DEFAULT_NMDA early=0.5, late=0.15, center dw34.
- **Directly measured:** NMDA:AMPA ratio at synapses *is* measured (whole-cell, evoked
  EPSC at +40/−70 mV); "silent synapses" (NMDA-only) are well-documented; ratio
  **declines** with maturation — but quantified mostly in **rodent** hippocampus/cortex
  [TRAINING].
- **Proxy/inference:** the *direction* (high→low) is solid; the specific 0.50→0.15 values
  and the dw34 center are **assigned**, not read off a human curve. Cross-species.
- **Assumption flags:** values and timing are chosen within the qualitative trend.
- **Confidence:** B→C. Direction A-grade, magnitudes/timing C-grade.

### B5. Gap junctions (g_gap = 1.2 nS, degree ~9, local syncytium)
- **Model uses:** 1.2 nS, ~9 coupled neighbours, spatially local (BS/SP).
- **Directly measured:** A1/A2 — this is our **most directly grounded** parameter
  (dye-coupling count and coupling conductance measured in subplate).
- **Inference chain:** minimal — we use the measured numbers. Remaining assumption:
  these are **rodent** subplate values applied to our human-timed model (cross-species),
  and the *spatial range* σ_gap=0.05 (sheet units) is a modelling choice (F) calibrated
  to give ~9 neighbours, not measured.
- **Confidence:** A for (conductance, degree); F for (spatial σ).

### B6. Spatial connectivity (σ_EE=0.08, local p≈15%, σ_IE broader)
- **Model uses:** Gaussian distance kernels; local E→E ~15% within ~column.
- **Directly measured:** cortical local connection probability ~10–20% within ~100 µm
  (Holmgren 2003; Perin & Markram 2011) — **juvenile rodent**, paired recordings.
- **Proxy/inference:** applied to fetal human subplate/cortex (cross-species + cross-age:
  fetal connectivity is sparser/immature than juvenile). The *sheet size* (1 mm) and σ
  values are **modelling choices** (F) calibrated so the sheet spans several columns.
  σ_IE>σ_EE ("blanket inhibition") is from Fino/Yuste 2011 [TRAINING] — juvenile.
- **Confidence:** B for local p (direction/magnitude); F for σ and sheet scale.

### B7. Conduction velocity (0.10 mm/ms = 0.1 m/s)
- **Model uses:** distance→delay at 0.1 m/s.
- **Directly measured:** immature/unmyelinated cortical axon CV ~0.1–0.3 m/s [TRAINING].
- **Confidence:** B. Order-of-magnitude; exact value within the immature range.

### B8. STP (depression default; subplate should facilitate)
- **Model uses:** STP_DEP (U=0.5, tau_d=400 ms) on most E→E; STP_FAC on E→SST.
- **Directly measured:** A5 — intra-subplate glutamate **facilitates** at 10–40 Hz.
  → **KNOWN MISMATCH:** subplate E→E currently uses depression; literature says
  facilitation. Logged as an open correction (Open Debts).
- **Confidence:** A for the *fact*; model currently INCONSISTENT with it.

### B9. Background tonic conductances (ge0/gi0, GE0_SCALE=0.65)
- **Model uses:** scaled tonic ge0/gi0 (~2–13 nS) representing ambient transmitter.
- **Directly measured:** tonic GABA/glutamate conductance in developing cortex exists
  (~0.1–2 nS tonic excitatory; tonic GABA via δ-subunit) [TRAINING], but our values are
  a **calibration** (GE0_SCALE) acknowledged in-code as "a starting point, not settled."
- **Confidence:** F (pragmatic calibration), self-flagged.

### B10. Region neuron counts, E:I ratio (4:1), SST fraction (30%)
- **Directly measured:** neocortex ~80:20 E:I; SST ~30% of interneurons [TRAINING].
- **Confidence:** B (standard ratios); absolute counts are subsampled (F, see B11).

### B11. Network SCALE / subsampling
- **Model uses:** SCALE multiplies all counts; full scale 143k neurons (still ≪ real).
- **Issue:** subsampling reduces per-neuron K; van Albada et al. 2015 prescribes weight
  compensation. We currently rely on full SCALE rather than compensation. The mapping
  from subsampled dynamics to biological dynamics is an **open methodological assumption**.
- **Confidence:** F/C. Flagged.

### B12. Cross-species developmental time map (DEV=90; gestational week ↔ sim time)
- **Model uses:** 1 dw ≈ 168/90 h of sim time; ALL timing claims (B1–B4) ride on the
  identification of "dw" with human gestational week.
- **Directly measured:** species developmental-time translation exists (Workman et al.
  2013, "translating time") [TRAINING], but mapping rodent transporter/PV data onto human
  gestational weeks is itself a model.
- **Confidence:** C. This is the **single most load-bearing translation** — every
  developmental-timing target inherits its uncertainty.

================================================================================
## PART C — VALIDATION TARGETS (held-out): provenance
================================================================================
(These must be INDEPENDENT of model parameters to count as validation — Test 2.)

| Target | Value | Source | Conf | Independent? |
|--------|-------|--------|------|--------------|
| IBI (tracé discontinu) | seconds; ↓ with age; longest <28 wk | A13 | A | yes (held-out) |
| Burst/GDP duration | hundreds of ms (0.2–2 s) | A9 | A | yes |
| Continuity onset | continuous by ~35 wk PMA | A13 | A | yes |
| Delta brushes | 28–35 wk peak | A14 | A | yes |
| Subplate firing | >40 Hz | A3 | A | yes |
| Programmed cell death | up to 50% | apoptosis lit [FETCHED earlier] | A | yes |
| Avalanche size exponent | τ≈1.5 (criticality) | Beggs-Plenz [TRAINING] | B | yes |
| Branching ratio | m≈1 (slightly <1) | Priesemann [TRAINING] | A | yes |

**Caveat on independence:** the E_GABA ramp (B1) and the IBI/continuity targets (A13)
both reflect the *same* underlying KCC2 maturation. If we tune B1 to hit A13, that is
NOT independent validation — it is circular. The honest test is: does the IBI/continuity
*trajectory shape* emerge with B1 set from its own (transporter) proxy, WITHOUT tuning
to A13? Must be checked.

================================================================================
## OPEN DEBTS (must close before the corresponding experiment counts)
================================================================================
1. **B2 inhibition-maturation ramp — CLOSED by rejection (2026-06-20).** The proxies
   (PV density, α2→α1) mature mostly post-term, i.e. OUT of the dw20–34 window. The
   in-window driver is the GABA polarity flip (B1), not an inhibition-strength ramp.
   Decision: do NOT add the B2 ramp; test the B1-driven bursting→continuous transition.
2. **B8 subplate STP mismatch — CLOSED (2026-06-20).** Subplate (BS/SP) intra-E now
   uses STP_FAC (facilitating) per A5; other regions keep STP_DEP. Fixed in gen_bundle.
3. **B12 time-map** governs all timing; verify Workman-type translation explicitly.
4. **[FETCHED]→primary** for all [TRAINING] items before any external use.
5. **Independence of B1 vs A13** (avoid circular validation of the GABA-flip transition).

================================================================================
## PART D — VERIFIED UPDATES (search package + self-fetch, 2026-06-20)
================================================================================
**Units:** PCW = post-conception weeks; GA/PMA (clinical) = PCW + ~2 wk. Below in PCW
unless noted. Our model "dw" is treated as ≈ PCW (assumption — flag in any claim).

### D1. IBI vs age — VERIFIED, concrete [FETCHED: NCBI Bookshelf NBK390356]  ⚠ model gap
- **24–29 wk CA: IBI = 6–12 s** (amplitude <2 µV). Tracé discontinu ~30–34/35 wk;
  IBIs progressively shorten; continuous by ~44 wk CA.
- ACNS Guideline-16: max IBI <30 wk ≈ 35 s; 34–36 wk ≈ 10 s.
- **MODEL GAP:** our network produced IBI ~0.1–3 s — TOO SHORT by ~3–10×. Real early
  IBI is 6–12 s (up to 35 s). => pause mechanism must be SLOWER than our sAHP τw=1.5 s.
  Cross-check: spindle bursts ~5/min => IBI~12 s (D3) — consistent with 6–12 s.
- Confidence: A (held-out validation target, now quantitative).

### D2. Spatial extent — VERIFIED LOCAL + RICHER PICTURE [PMC4806652 + Yang et al. 2009]
- Spindle burst synchronizes a LOCAL network **200–400 µm diameter** (Ca-imaging
  100–200 µm) ≈ cortical column; >90% span SEVERAL adjacent columns (not global, not
  single). **GABAergic synapses provide the SPATIAL CONFINEMENT to a column.**
  (PMC4806652; confidence A, rodent P0–P7, S1)
- **5–10 APs/neuron/spindle burst**; duration **0.5–3 s**; period **~10 s** (~5/min) —
  CONFIRMS our bouffée 1–3 s and IBI 10–30 s targets are in the right biological range.
- ★ KEY (Yang et al. 2009, J Neurosci, DOI 10.1523/JNEUROSCI.5646-08.2009): THREE co-existing
  patterns in neonatal S1 cortex in vivo (P0–P7 rat):
    1. Spindle bursts: LOCAL, do NOT propagate, domain 200–400 µm. ← OUR TARGET
    2. Gamma oscillations: LOCAL, do NOT propagate.
    3. Long oscillations: PROPAGATE at 25–30 µm/s (~0.025-0.030 mm/s), domain 600–800 µm.
  This is a KEY REFRAME: spindle-burst-like activity (our tracé target) is NON-PROPAGATING
  by definition. All discussion of "wave propagation speed" and σ_EE-vs-wave-velocity is
  IRRELEVANT FOR THE SPINDLE-BURST REGIME. Propagation speed only matters for Long
  Oscillations — a DIFFERENT phenomenon.
- GABA spatial confinement: PMC4806652 states GABAergic synapses confine bursts to
  200-400µm domain. At dw20, GABA is DEPOLARIZING — but Kirmse 2015 (ledger F5) shows
  depol-GABA CAN still provide shunting inhibition in vivo. F5 finding (removing I→E doesn't
  abolish bursting) now reinterpreted: I→E may provide SPATIAL CONFINEMENT (shunting limits
  domain spread), not ignition. Domain SIZE may enlarge if I→E is removed — NOT TESTED.
- SPATIAL OPEN QUESTION: our 1mm sheet (N=1200, SCALE=0.1) fires globally. Per Yang et al.,
  it should show 2-5 independent non-propagating spindle-burst domains. This means σ_EE is
  too long-range for the DOMAIN SIZE reason (200-400µm domains in 1mm sheet need σ≈0.1-0.2
  sheet-units = 100-200µm). But this constraint is grounded in DOMAIN ANATOMY not wave speed.
  Anatomically grounding σ_EE requires measuring LATERAL SPREAD of local axon collaterals in
  dw20 fetal cortex (currently not found; confidence F for current σ_EE=0.08).
- CONFOUNDING NOTE: "200-400µm" also appears in an optogenetic PV-stimulation paper as the
  EXPERIMENT ZONE SIZE (not a natural scale), and as calcium microdomain 50-100µm with ~4-min
  period (entirely different observable). Cite D2/PMC4806652 for spindle-burst domain
  SPECIFICALLY — do not conflate.
- Confidence: A (rodent in vivo P0-P7 S1); human homolog via delta brushes (A14).

### D3. Immature neuron SIZE — VERIFIED, HIGH-IMPACT [Chen & Kriegstein GW22, via search pkg]
- Human GW22 (=PCW20) deep-layer pyramidal: **R_in ~1.1 GΩ, C ~23 pF, fired only 1–2 APs.**
  Rodent subplate: RMP ~−55 mV, R_in ~1–1.2 GΩ (A-anchor PMC2766272).
- **MODEL MISMATCH (major):** we use adult C_E=200 pF, gL_E=10 nS (R_in=100 MΩ). Immature
  is C~23 pF, gL~0.9 nS (R_in~1.1 GΩ) — ~9× smaller C, ~11× smaller gL. τ_m similar
  (~20–25 ms) BUT a 1 nS synapse moves an immature cell ~6× more than our adult cell.
  => This LARGELY EXPLAINS why we kept needing REC_GAIN~6–8 to get activity: we used
  adult-sized neurons, so biological weights were ~6–10× too weak. Using immature sizes
  should make biological weights work WITHOUT artificial scaling. HIGH-VALUE correction.
- "fired only 1–2 APs": immature neurons are NOT repetitive firers (few Na channels) —
  our AdEx repetitive firing is too mature (deeper change, noted not yet actioned).
- Confidence: B+ (specific cited GW22 measurement; verify exact PDF before external use).

### D4. E_GABA timing + regional gradient — VERIFIED direction [Bayatti 2008, Sedmak 2016, Chen&Kriegstein]
- **Regional gradient CONFIRMED:** KCC2 appears in subplate ~16 PCW while cortical plate
  NOT yet KCC2+ at that time (Bayatti 2008); many subplate AND cortical-plate neurons
  KCC2-ir by ~25 PCW (Sedmak 2016). => our SP-before-P1 ordering is grounded.
- **GABA still DEPOLARIZING at GW22 (=PCW20)** in human cortex (Chen & Kriegstein,
  gramicidin patch). Perinatal human cortex Cl- transport "as immature as rat" (Dzhala
  2005) => functional switch NOT adult by birth (protracted, partly postnatal).
- **ASSUMPTION/TENSION (flag):** KCC2 protein appears 16–25 PCW but FUNCTIONAL E_GABA
  switch lags expression and is protracted. Our model flips E_GABA to −75 by dw34 — this
  may be EARLY for the full functional switch. Honest test: set timing from KCC2-expression
  proxy (appearance 16–25 PCW), do NOT tune to the EEG continuity transition (circular).
- Confidence: B (direction/gradient A; exact functional-switch weeks C, proxy-inferred).

### D5. Cross-species time map — VERIFIED [Workman 2013, translatingtime.org]
- ln(PC_days) = const + slope·eventscore; rat const=2.31/slope=1.705/gest=21d,
  human const=3.167/slope=3.72/gest=270d. Rough equivalents (search-pkg calc):
  rat P0 ≈ human PCW~17; P7 ≈ PCW~32; P10 ≈ PCW~39 (term); P15 ≈ PCW~55 (POSTNATAL).
- => rodent KCC2-adult (~P15) ≈ human ~4 mo postnatal — consistent with D4 (switch late).
- Confidence: B+ (model + tool; our DEV=90 is a SEPARATE sim-time compression, B12).

### D6. PV / perisomatic inhibition — VERIFIED mostly POSTNATAL [Cercor 6:620; primate PFC PMC2882199]
- PV-IR linked to mature fast-spiking/perisomatic inhibition is predominantly POSTNATAL
  (PV+ basket/axo-axonic near-absent at birth, rise over first months; not found <39–40
  wk in preterm hippocampus). Transient early PV in V1 Cajal-Retzius ~20 wk (lost by term).
  Primate α2→α1 switch is PROTRACTED (not a narrow window).
- => CONFIRMS B2 rejection: no in-window (dw20–34) inhibition-STRENGTH ramp to ride on;
  in-window driver is GABA POLARITY (D4). Inhibition's role in-window is depol→hyper flip
  + spatial confinement (D2), not strength increase.
- Confidence: A (direction); specific weeks B.

### REMAINING NUMBER-GAPS (search pkg could not close; honest):
- NMDA:AMPA exact early/late values + silent-synapse fractions (only ranges P3–P12 rat).
- Fetal human E→E p(connect) at earliest ages (only juvenile rodent anchors).
- Avalanche τ and branching m for DEVELOPING cultures/slices specifically.
- SAT/delta-brush frequency/duration/amplitude vs PMA as a table.
- Tonic conductance (δ-GABA/glutamate) in nS; cortical-wave propagation speed mm/s.

================================================================================
## PART E — FURTHER VERIFIED CLOSURES (search package 2, 2026-06-20)
================================================================================

### E1. IBI-vs-age TABLE — VERIFIED [Cherian et al. PMC2811985 Table 1; neonatal EEG atlas]  ⚠⚠
- **27–28 wk CA: discontinuity 10–30 s** (SAT high-amp +++); 29–30: 8–15 s; 31–33: 6–10 s;
  34–35: 2–6 s; then tracé discontinu fades. Quiet <25 µV (often <10); high-amp SAT >100 µV.
- Longest-acceptable single IBI: 26w 46s, 27w 36s, 28w 27s, 31–33w 20s, 34–36w 10s, 37–40w 6s.
- **REVISED MODEL TARGET:** IBI is even LONGER than the 6–12 s in D1 — up to **10–30 s at
  27–28 wk**, tens of seconds at the earliest ages. Our model IBI (0.1–3 s) is ~5–30× too
  short.
- **REOPENS IBI MECHANISM (honest correction to finding #6):** sAHP (1–5 s) sets only
  SHORT IBIs (1–2 s). A 10–30 s IBI needs a SLOWER variable (τ ~10–30 s): slow synaptic-
  resource depletion (Tabak/Rinzel developing-network model), slow K+/adenosine, or slow
  ionic recovery (cf. retinal-wave refractory 40–60 s, A15; gap/hemichannel events tens of
  s, PMC4169969). sAHP gives the right SHAPE but wrong TIMESCALE. The IBI pacemaker is a
  SECONDS-TO-TENS-OF-SECONDS process, not the 1.5 s sAHP. Note: this is NOT the refuted
  chloride-efflux-quench (that was within-burst sign/magnitude); it is the separate
  question of what slow variable gates re-ignition. Open mechanism, now well-targeted.
- Confidence: A (held-out validation table).

### E2. Avalanche exponent in DEVELOPING networks — VERIFIED [Beggs&Plenz; Stewart&Plenz]
- Avalanche size distribution power law τ ≈ **1.5** (α=−3/2). Stewart & Plenz: this is
  PRESERVED across organotypic "development in a dish" despite large mean-activity changes.
- => τ≈1.5 is a legitimate DEVELOPING-network criticality target (validates our instrument's
  expected value; our Q3 control B used 1.3–1.8 gate — consistent).
- Confidence: A (in-vitro developing cortex).

### E3. Tonic conductance — VERIFIED scale [Sebe 2010 newborn mouse; Li 2017 rat]
- Tonic GABA current (exogenous 10 µM GABA): P3 ~196 pA, P9 ~37 pA → ~2.8 nS (P3),
  ~0.53 nS (P9) IF E_GABA≈0/Vhold=−70 (scale-check, not basal). Order: **~0.5–3 nS** tonic.
- **MODEL CHECK:** our gi0 background (~11.7 nS scaled) may be HIGH vs ~0.5–3 nS tonic.
  Flag: revisit background conductance magnitude (B9 was self-flagged F/pragmatic).
- Confidence: B (exogenous-GABA scale; basal likely lower).

### E4. NMDA:AMPA / silent synapses — PARTIAL [thalamus ontogeny; rat P3–P12 cortex]
- AMPA component rises fast in early postnatal dev; silent (NMDA-only) synapses P3–P12
  (rat). Exact cortical NMDA:AMPA ratio values NOT pinned. Our 0.5→0.15 remains direction-
  grounded (A), magnitudes assigned (C). Unchanged.

### E5. E_GABA flip-timing risk — VERIFIED reasoning [Chen&Kriegstein GW22; clinical IBI norms]
- Flip set too EARLY → network too inhibitory → loses the preterm discrete-high-amp-burst-
  with-long-pause pattern. Cannot prove "flip at exactly GA X" in human. KEEP flip as a
  HYPOTHESIS; validate on INDEPENDENT targets (KCC2/NKCC1 expression points, gramicidin
  E_GABA points, EEG metrics NOT used for tuning). Avoid circular EEG tuning.
- Confidence: reasoning A; exact timing remains C.

### STILL OPEN (honestly):
- Fetal human E→E paired-recording p(connect): NO direct source found. Proxy = earliest
  postnatal rodent paired recordings (P0–P7 L4/5) via translatingtime. Documented gap.
- Exact NMDA:AMPA cortical ratio values; basal (not exogenous) tonic conductance in nS.

================================================================================
# PART F — IBI pacing + self-organized criticality (debts from 2026-06-21 session)
================================================================================
Context: model produces emergent discontinuous bursting + near-critical m≈0.95 at dw20
(OBSERVATIONS #36–39), BUT (Q4) the IBI timescale was set by a FREE adaptation τw, and the
critical recurrent gain was placed BY HAND (sharp transition at gain≈18, no self-organization).
This part grounds the two missing mechanisms. User ran the literature searches (web blocked).

## F1. Na/K-pump ultraslow AHP as IBI pacemaker — REJECTED as PRIMARY dw20 mechanism [important]
- NUMBERS (verified by user against primary source Avesar et al. 2013, J Neurophysiol):
  AP trains evoke a long AHP ~20 s; decay τ ≈ 8.7±0.8 s (L5 pyramidal), 11.2±2.3 s (CA1);
  ouabain/0-K+/TTX-sensitive → genuinely Na/K-pump + spike dependent. Numerically PERFECT for
  IBI 10–30 s. Network analogues: minute-long usAHP in motor nets (Zhang & Sillar; Picton).
- REJECTION REASON (two independent grounds):
  (1) DEVELOPMENTAL TIMING: pump-AHP MATURES POSTNATALLY — Avesar Discussion states Na-pump
      expression rises at postnatal weeks 4–6 (mouse); all sodium-dependent AHP data is from
      tissue ≥~2 wk postnatal (their "young" group was P12–14). dw20 human ≪ P12–14 mouse
      (translatingtime), so at dw20 the pump-AHP is plausibly WEAKER, not equal to 8.7–11.2 s.
  (2) WRONG TREND DIRECTION: pump-AHP STRENGTHENS with maturation → would make the pause LONGER
      with age. Observed tracé-discontinu IBI SHORTENS with age (24–26 wk longest → continuous
      by term). So the pump's own ontogenetic trend runs OPPOSITE to the data.
- VERDICT: not the primary dw20 IBI-setter. May persist as a minor CONTRIBUTOR only if a
  maturing excitability/connectivity term dominates the trend (Q7 alternative, unproven). Treat
  any pump-AHP term at dw20 as an ASSUMPTION with reversed-trend caveat.
- Confidence: A (the numbers + the maturation caveat); rejection reasoning A.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3735883/ ; pubmed 22405867 ; PMC5547253

## F2. Short-term synaptic depression (STD) recovery τ in developing synapse — VERIFIED [B4]
- Recovery from depression: ~9 s at P7/8, ~3 s at P11/12 (mouse auditory brainstem, developing).
  Order: SECONDS (3–9 s), faster toward maturity. NOT 10–30 s alone.
- MODEL USE: STD is the grounded "fast brake" (burst termination + first seconds of refractory).
  IBI 10–30 s then requires NETWORK stretching of this τ (model showed IBI stretches ~1.5–2.7×
  beyond the slow time constant, OBS #38–39) and/or an additional slower process.
- CAVEAT: data is postnatal (P7–12); STD presence at dw20 human is a (mild) extrapolation —
  STD is a near-universal early synaptic property, lower risk than the pump.
- Confidence: B (τ values, developing synapse but not cortex; dw20 extrapolation flagged).

## F3. Dynamical synapses (STD) → self-organized criticality — MECHANISM [Levina 2007, A1]
- Levina, Herrmann, Geisel 2007 Nat. Phys.: USE-dependent synaptic depression + SLOW resource
  recovery self-organizes the network to criticality across a RANGE of mean coupling (no fine-
  tuning). Resource J: dJ/dt = (1/τ_J)(α/u − J) − u·J·δ(t−t_sp); recovery τ_J = τ·ν·N (slow,
  timescale-separated from spikes). Negative feedback: big burst → more depletion → subcritical;
  quiet → resource recovers → approaches critical. THIS is the SOC mechanism that should fix the
  hand-set-gain knife-edge (OBS #39).
- ENGINE STATUS: already implements Tsodyks–Markram STP = Levina dynamical synapse
  (main_multiregion.cu:211–229: x=1+(x−1)exp(−Δt/τ_d), eff_w=w·U·x, x−=U·x). NO C++ NEEDED —
  set recurrent E→E to depressing with slow τ_d.
- Confidence: mechanism A (theory); applicability to THIS heterogeneous AdEx multiregion net is
  a HYPOTHESIS to test (does m≈1 emerge across a range of static REC_GAIN?).
- URL: materias.df.uba.ar/dnla2019c1/files/2019/06/LevinaHerrmannGeisel...NaturePhysics.2007.pdf

## F4. Homeostatic recovery of criticality in vivo — VERIFIED timescale [Ma/Turrigiano 2019, A2]
- Ma, Turrigiano, Wessel, Hengen 2019 Neuron: cortex homeostatically RESTORES criticality after
  perturbation (monocular deprivation). Branching ratio m drops immediately, RESTORED by ~48 h;
  shape-collapse error recovers ~48–72 h. So self-organization to criticality is REAL but SLOW
  (~2 days). In sim-time (DEV=90: 1 day≈267 ms) 48 h ≈ 533 ms — fast enough to observe.
- Distinction: this is the SLOW (synaptic-scaling) SOC; Levina F3 is the FAST (STD) SOC. For the
  prenatal window the STD route (F3) is more practical and earlier-developing.
- Confidence: A (in vivo, juvenile rodent). URL: Ma_Hengen_Neuron2019supplemental.pdf (squarespace)

## F5. Depolarizing GABA is NOT necessarily excitatory in vivo — REFRAME [A3, Kirmse 2015]
- Kirmse et al. 2015 (neonatal mouse neocortex P3–4, in vivo): GABA depolarizes immature neurons
  but INHIBITS network activity (shunting; E_GABA above rest yet below spike threshold → clamps V).
  "Depolarizing ≠ excitatory" at the network level (also Valeeva 2016, Oh 2016).
- MODEL IMPLICATION: our I→E depol-GABA (E_GABA=−45) may act as a SHUNT STABILIZER, not an
  ignition booster. To CHECK: measure whether removing/strengthening I→E ignites vs suppresses
  bursts. If shunting, iSTDP-style inhibitory homeostasis can still regulate gain via conductance
  even without a hyperpolarizing PSP sign (regime-dependent — flag).
- Confidence: A (in vivo neonatal); applicability to dw20 human + our E_GABA value = B.
- URL: pubmed 26177896 ; PMC7053538 ; onlinelibrary.wiley.com/doi/10.1111/cns.12353

## F6. Burst ENVELOPE 1–5 s is NMDA-dependent — VERIFIED [C1]
- Developing mouse cortex spontaneous synchronized events: AMPA component ~1 s; NMDA component
  EXTENDS the burst to ~4 s and carries most of the spikes (Roles of glutamate/GABA receptors…
  pubmed 17542015). Our model burst ~100 ms is too short → missing sustained NMDA reverberation.
- MODEL USE: to reach 1–5 s envelopes, recurrent NMDA must sustain the burst (plateau / slow
  effective τ), not just single-synapse τ_nmda=100 ms. dev_state already makes NMDA-dominant early.
- Confidence: A (developing cortex). URL: pubmed 17542015

## F7. Criticality + intermittent bursting are COMPATIBLE — VERIFIED [C2, Zierenberg 2018]
- Zierenberg, Wilting, Priesemann 2018: homeostatic excitability + external-input level yields
  CONTINUOUS near-critical dynamics (intermediate input, in-vivo-like) OR INTERMITTENT BURSTS
  (very weak input, culture-like) within the SAME self-organizing model. So our co-occurring
  near-critical-m + discrete bursts is a KNOWN regime, not a contradiction.
- Confidence: A (theory+sim). URL: arxiv.org/abs/1807.01479

## F-SYNTHESIS — grounded unified plan (replaces hand-set gain + free τw)
HYPOTHESIS: recurrent E→E SHORT-TERM DEPRESSION with slow resource recovery (Levina F3) does
DOUBLE DUTY — (i) self-organizes effective coupling to criticality (debt 1; fixes REC_GAIN knife-
edge) AND (ii) provides network refractoriness that, stretched by recurrence, paces IBI (debt 2).
Grounded τ_d range = 3–9 s (F2, measured). No 20-s free current; pump-AHP rejected (F1).
TESTS (falsifiable): (a) does m≈1 emerge across a RANGE of static REC_GAIN with STD on? (SOC);
(b) does IBI land 10–30 s from τ_d∈[3,9]s + network stretch, WITHOUT a free slow current?;
(c) STD OFF (static synapses) → criticality returns to knife-edge + IBI collapses (falsification).
Separate: F5 check (is I→E igniting or shunting?); F6 (NMDA to lengthen burst envelope to 1–5 s).

================================================================================
# PART F — ROUND 2 additions (user lit search #2): grounds the IBI slow variable
================================================================================
## F8. Pump-AHP rejection STRENGTHENED — primary source for late maturation [Fukuda&Prince 1992]
- Fukuda & Prince 1992 (Dev Brain Res; the ref Avesar cites): PGH ratio (electrogenic Na-pump
  activity proxy) "very low" at P7–11, significantly increases P7–11 → P21–25 → P35–39 →
  "pump activity develops gradually over the first 5 weeks of life." Direct confirmation: the
  pump is WEAK in the early window and STRENGTHENS with age → wrong trend for tracé IBI (which
  shortens with age). F1 rejection of pump-AHP as PRIMARY dw20 mechanism stands, now primary-sourced.
- Confidence: A. URL: (Fukuda & Prince 1992, Dev Brain Res — to fetch exact PMID).

## F9. ★ GROUNDED intrinsic bursting mechanism: I_CAN/I_NaP + Ca-K [VERIFIED primary source]
- ★ VERIFIED 2026-06-21 by direct WebFetch of the PRIMARY SOURCE:
  Sheroziya, von Bohlen und Halbach, Unsicker, Egorov 2009, J Neurosci 29(39):12131-44,
  PMID 19793971, DOI 10.1523/JNEUROSCI.1333-09.2009 — "Spontaneous bursting activity in the
  developing entorhinal cortex." Confirmed from the abstract:
    • mEC LAYER III principal neurons, rat P5–7 / P8–10 / P11–13 (developmental).
    • PROLONGED BURST DURATION = 2–20 s [VERIFIED NUMBER].
    • MECHANISM: prolonged burst/plateau = I_CAN (Ca-sensitive nonselective cation) + I_NaP
      (persistent Na); blocked by BAPTA/Cd2+/flufenamic-acid/TTX/nifedipine/riluzole. Burst
      TERMINATION = Ca2+-activated K (BK) currents (Fig 8). Ca-dependence proportions 53/81/29%.
  → This is EXACTLY the F6/F9 mechanism: I_CAN plateau (our engine I_CAN) + Ca-K termination
    (our AdEx w). The MECHANISM is now primary-source verified, not inferred.
- ✗ NOT VERIFIED (corrected): the specific "AHP duration 17.5 s" and "IBI 24 s" numbers (earlier
  search-reported as Table 2) are NOT in the abstract — UNCONFIRMED. Do NOT treat as data.
- ★ MODEL GROUNDING (revised, after verification):
    • BURST/bouffée duration: anchored to the VERIFIED 2–20 s (not the unverified 17.5 s). Our
      model bouffée 1–3 s (OBS #44) sits at the LOW end of 2–20 s — consistent, mildly conservative.
    • τw (AdEx w = Ca-K postburst AHP): grounded by (a) ORDER — verified slow-burster has seconds-
      scale Ca-dependent dynamics; (b) the resulting model IBI 17–23 s independently matches the
      VERIFIED clinical tracé IBI 10–30 s (Cherian, Part D). τw=8 s does NOT depend on the
      unverified 17.5 s number. Trend (shorten τw → shorter IBI; bursts shorten P5→P13 per the
      paper's age groups) is consistent.
- Confidence: MECHANISM A (primary-source verified); burst-duration 2–20 s A (verified);
  τw value B (order + IBI-match, not a single measured τ); cross-region mEC→neocortex flagged.

## F10. STD can be MUCH slower than τd=3–9 s after strong/global bursts — multi-timescale [B-update]
- (i) V1 STD has a SLOW component recovering ~10 s (alongside fast 100s-ms) [classic V1 STD work].
- (ii) Review "Short-Term Presynaptic Plasticity": after SUSTAINED high-frequency activation,
  recovery from depression is "many tens of seconds" (vs seconds for paired-pulse). So a single-
  pool Tsodyks–Markram τd=3–9 s is a baseline; EFFECTIVE refractoriness after a GLOBAL burst can
  be far longer → justifies IBI > τd without inventing a current.
- (iii) Neonatal CA3 interictal bursts: amplitudes not fully recovered until 30–40 s post-event
  (epileptiform regime, but the slow replenishment timescale is the relevant datum).
- MODEL USE: explains OBS #40 "IBI can exceed τd". Optionally a 2nd slow STD pool, but F9 (Ca-AHP)
  is the cleaner single grounded slow variable. Confidence: B (mechanism A, exact τ regime-dependent).

## F11. WHY IBI is gain-dependent — needs near-threshold regime [Gritsun 2011, regime anchor]
- Gritsun, le Feber, Rutten 2011 (Biol Cybern): realistic culture IBIs arise in a WEAKLY-
  synchronized, NOISE-DRIVEN regime with WEIGHT HETEROGENEITY; "pacemaker-driven" networks give
  too-REGULAR IBIs set by the fastest pacemaker. Directly supports OBS #40: our tracé-like (long,
  IRREGULAR) IBI lives NEAR the critical threshold (REC≈6), while deep-supercritical (REC=12) gives
  metronomic ~3 s. ⇒ robust tracé IBI needs the network to SELF-SIT near threshold (slow homeostasis
  / synaptic scaling / intrinsic plasticity over STD), not a fixed deep gain.
- Confidence: A (modeling, culture-grounded). URL: Gritsun et al. 2011, Biol Cybern (to fetch PMID).

## F-SYNTHESIS v2 (grounded, supersedes v1 for debt 2)
- DEBT 2 (IBI): the grounded slow IBI pacemaker is the Ca-dependent postburst AHP of developing
  cortex (F9), modeled by AdEx w with τw ≈ 6–9 s (= mEC AHP duration 17.5 s / 2–3). This is NOT a
  free τw; IBI 10–24 s is a prediction from it, with the correct age trend (shorten τw → shorter IBI).
  pump-AHP rejected (F1/F8). STD slow recovery (F10) is a secondary contributor.
- DEBT 1 (SOC): STD/Levina (F3) — CONFIRMED (OBS #40: static→350 Hz runaway; STD bounds m≈0.95 across
  REC 6–12). 
- REGIME (F11): operate NEAR threshold (REC≈6) for irregular tracé-like IBI; a slow homeostasis should
  self-hold the net there (next step, Task #17 — synaptic scaling, NOT rate-clamp current).
- BURST ENVELOPE (F6/F9): I_CAN/I_Nap plateau lengthens burst to 1–4 s (engine work; deferred).
- NEXT RUN: τw=8000 (mEC Ca-AHP) + EE_STP=dep_slow (STD-SOC) + REC≈6 (near threshold), long, validated
  IBI+criticality → establish the grounded headline. Then trend test (mature τw → IBI shortens).

================================================================================
# PART H — Second literature verification round (2026-06-21): spatial structure, subplate,
#           STDP, Phase 2 redesign
================================================================================
Context: user fetched primary sources closing Q1-Q5 from the formal question list above.
All entries are from PRIMARY sources as specified; model implications derived by us.

## H1. ★ Spindle bursts DO NOT PROPAGATE — fundamental reframe of the spatial question
- Yang et al. 2009 (J Neurosci 29(43):13496–13504, DOI 10.1523/JNEUROSCI.5646-08.2009):
  Three co-existing oscillatory patterns in neonatal rat S1 in vivo (P0–P7):
    (1) Spindle bursts (0.5–3 s, ~5/min): LOCAL, NON-PROPAGATING, domain 200–400 µm.
    (2) Gamma oscillations: LOCAL, NON-PROPAGATING.
    (3) Long oscillations (~25–30 µm/s, domain 600–800 µm): PROPAGATING.
- ★ CRITICAL REFRAME: our tracé discontinu / bouffée target ≈ spindle-burst regime.
  Spindle bursts BY DEFINITION do not propagate. Therefore:
    - Discussion of "wave propagation speed" (G4: 18.2 mm/s vs 0.45 mm/s retinal) was
      applied to the WRONG PHENOMENON. Propagation speed matters for Long Oscillations
      (pattern 3), NOT for our target.
    - σ_EE should NOT be tuned to match any propagation velocity (that would be (a) wrong
      target and (b) circular/fitting as user correctly identified). It should be grounded
      in ANATOMICAL lateral axon collateral spread (still unverified — F confidence for now).
    - Propagation speed of Long Oscillations (25-30 µm/s = 0.025-0.030 mm/s) is ~600-700×
      slower than our model's emergent speed (~18.2 mm/s) — relevant IF we ever target this
      pattern, irrelevant for spindle bursts.
- Spindle burst numbers (D2 consistent + extended): 0.5–3 s duration, 8–30 Hz internal
  frequency, ~5/min period (IBI ~12 s), 200–400 µm domain. Our bouffée (1–3 s, IBI 10–30 s)
  IS in the right biological range for spindle-burst-like activity.
- Confidence: A (primary in vivo rodent). URL: DOI 10.1523/JNEUROSCI.5646-08.2009

## H2. σ_EE grounding: the domain-anatomy reason (not wave-speed) [corrected from G4/G5]
- Correct reason to reduce σ_EE (if needed): NOT to match wave propagation velocity (H1
  shows spindle bursts don't propagate, so matching any velocity is irrelevant), but because
  our 1mm sheet should support 2-5 INDEPENDENT non-propagating 200-400µm spindle-burst
  domains. For independent domains, σ_EE must be short enough that one domain's E→E
  recurrence does NOT reach the neighboring domain. Rough estimate: domain diameter ~300µm,
  inter-domain gap ~200µm → σ_EE << 200µm = 0.20 sheet-units. Current σ_EE=0.08 (80µm)
  already satisfies this (3σ ≈ 240µm < 300µm domain). So σ_EE is NOT obviously wrong by
  this criterion — the failure to see local domains may be a DENSITY problem (only N=1200
  cells at SCALE=0.1 → sparse representation of domains) or HOMOGENEITY problem (all cells
  equally tuned, no seeds for local ignition). At SCALE=1.0 with N=12000, 300µm domain
  would contain ~π×0.3²×12000 ≈ 3393 cells — should see local domains IF dynamics allow.
  The SCALE=1.0 continuous-hum problem (non-zero median) prevents proper domain analysis.
- Confidence: reasoning A; anatomical anchor for σ_EE = still confidence F (no primary
  source found for lateral axon collateral spread in dw20 fetal cortex — open literature item).

## H3. Subplate: human fetal activity confirmed at GW20 + role in spindle-burst generation
- Moore et al. 2009 (PMC3564513): human fetal SP neurons (GW16–22) show spontaneous electrical
  activity in vitro, functional glutamate and GABA receptors present from ~GW20. Direct human
  anchor for dw20 window SP activity. Luhmann 2009 numbers: SP RMP ~ −55 mV, R_in ~ 1–1.2 GΩ.
  (Our SP_E: V_eff=−55.2 mV, R_in=0.709 GΩ — V_eff very close! R_in ~40% low but same order.)
- "Subplate Neurons Promote Spindle Bursts and Thalamocortical Patterning" (PMC3517992):
  CAUSAL in vivo evidence that SP neurons are needed for spindle burst organization (rodent).
  This upgrades the severity of "SP silent in our model" — it's not merely "amplifier might
  not be needed", it's "active SP contributes causally to producing the phenomenon we target."
- MODEL STATUS: SP silent (0 Hz) in our headline run. Gap junctions in BS/SP DAMP voltage
  fluctuations (noise_damp×0.37–0.46), not amplify. The "amplifier" role is not positively
  demonstrated. SP silence is less-catastrophic than reviewer implied (Luhmann: not a required
  pacemaker), but more concerning than I framed it (causal role in spindle bursts per PMC3517992).
  Honest call: SP silence is an OPEN PROBLEM, partially mitigated, not fully resolved. Defer
  to Phase 2 where SP apoptosis becomes the mechanism of interest.
- Confidence: H3 mechanism A; causal role A (rodent); human activity B (in vitro, GW16-22).

## H4. TH → cortical plate: "waiting period" 15–24 PCW, cortical invasion after 24 PCW
- Confirmed: human "waiting period" = 15–24 PCW (Corbett-Detig et al., PMC3041913). dw20 IS
  inside this window → TH axons waiting IN subplate, not yet in cortical plate.
- Kostović 2010 (Acta Paediatrica): 22-23 PCW accumulation → after 24 PCW cortical plate
  invasion → after 29 PCW intra-cortical elaboration.
- Allendoerfer & Shatz 1990 (Nature 347:179): SP neuron removal disrupts thalamocortical
  innervation — thalamic axons "accumulate and wait in subplate for several weeks."
- MODEL IMPLICATION: our S_TH_P1_E (direct TH→P1, bypassing SP) is ANACHRONISTIC at dw20 —
  biologically, TH should not have direct functional cortical-plate synapses yet. SP relay
  should be the primary TH→cortex route. In our model, SP is silent → TH→SP→P1 relay fails
  → TH→P1 direct provides the drive. This is a correct "emergency bypass" for the snapshot
  but is NOT biologically accurate for the developmental sequence. Flagged for Phase 2
  trajectory work, not urgent for dw20 single-episode science.
- QUANTITATIVE ANCHOR: TH axon invasion of CP begins at ~24 PCW → at dw20 (4 weeks before
  the invasion), TH→CP direct transmission should be weak/absent. If we want to model this
  correctly: gate TH→P1_E strength to near-zero at dw<24, rising sigmoidal to full by dw28.
- Confidence: A (two independent primary sources + a Nature landmark paper).

## H5. STDP at depol-GABA: neocortex anchor now available (Wang & Kriegstein)
- Wang & Kriegstein 2009 (PMC2684685): NKCC1 knockdown → impairs AMPAR synapse formation in
  neonatal NEOCORTEX; rescued by Mg-insensitive NMDA receptor → causal chain CONFIRMED in
  cortex: depol-GABA (via NKCC1) → NMDA Ca → synapse maturation/formation.
  This is the FIRST NEOCORTEX-SPECIFIC anchor for depol-GABA→NMDA→plasticity (Leinekugel
  was hippocampus only; now we have cortex). Transfer to our dw20 model: from B to A confidence.
- Leinekugel 1997 numbers (already in G2): Mg-block affinity 16→118 µM via -82→-59mV.
  Mechanism confirmed (hippocampus + neocortex together), but the SPECIFIC Mg-block-relief
  calculation was hippocampal resting potential (-82mV), which is far from our immature
  model cell's V_eff (-55mV). In our model, cells may be ALREADY largely de-blocked
  (shallow Mg-block) even before GABA activation — which means the facilitation effect is
  already built-in structurally, not requiring GABA to add further relief.
- ✗ STDP window width: still UNVERIFIED. "Wider in immature" is not sourced with a number
  (confirmed in H4-STDP of Part G). Keep τ_pre=τ_post=20ms. STDP review (PMC3059680)
  notes dw20-equivalent neocortex data is largely absent — keep as "established mechanism,
  unquantified window at this age."
- Confidence: mechanism A (cortex, Wang & Kriegstein); window width UNVERIFIED.

## H6. Phase 2: it's synaptogenesis (not pruning) + transient circuit refinement
- Synaptogenesis at dw20: "Synaptogenesis in layer I, first half of gestation" (Cereb Cortex
  1998, 8:245): EM confirms synaptogenesis in layer I by ~20 gw. Synaptophysin IHC atlas
  (Front Cell Neurosci 2023) shows progressive increase dw20→26→33→40.
  → dw20-40 IS synaptogenesis, NOT mass synaptic pruning. Phase 2 targeting local E→E
  elimination is BIOLOGICALLY WRONG for this window.
- What IS eliminated prentally: exuberant collaterals + transient long-distance projections
  (Innocenti & Price 2005, Nat Rev Neurosci, "Exuberance in the development of cortical
  networks"). The "refinement" is of TRANSIENT EXUBERANT projections (especially SP-centered),
  not bulk local E→E synapse elimination. SP neuron partial apoptosis IS part of this.
- SP apoptosis timeline (Kostović): NOT uniform at dw30-35. Region-specific:
    somatosensory cortex (≈ our P1): SP persists even at term (40 wk)!
    prefrontal/association (≈ our A2): fades over 6 postnatal months.
  → Single-wave apoptosis model for all SP is wrong for P1 specifically.
- Huttenlocher & Dabholkar 1997, Petanjek et al. 2011 (PNAS PMC3156171): synaptic pruning
  peaks POSTNATALLY, not prenatally — confirmed from primary sources.
- ★ CORRECT Phase 2 target: "Activity-dependent transient circuit refinement" —
    (a) gradual weakening of TH-waiting-period S_TH_P1_E as dw approaches 24 (gating)
    (b) SP apoptosis (partial, region-specific, starting dw30-35 for some regions)
    (c) Elimination of weak INTER-REGIONAL projections (the SP-relay ones, not local E→E)
    (d) Local E→E: STDP can model STRENGTHENING (synaptic potentiation within 200-400µm
        domains) — this is biologically consistent since local E→E IS growing (synaptogenesis)
        and preferential strengthening within co-active domains IS the right dw20-40 mechanism.
  NOT: bulk elimination of local E→E cortical synapses.
- Confidence: synaptogenesis A; pruning-timing A (Huttenlocher, Petanjek); exuberance A;
  SP regional timing A (Kostović); Phase 2 redesign = synthesis of multiple A sources.

================================================================================
# PART I — Third literature round (2026-06-22): σ_EE anatomical anchor + SP survival data
================================================================================
Closes the two open debts from Part H (H2 anatomical σ_EE anchor; H6 SP survival curve).
Sources are cross-species/cross-age proxies throughout — translation flagged per item.

## I1. σ_EE anatomical anchor found — CURRENT VALUE ALREADY CONSISTENT [upgrades B6/H2]
- Feldmeyer et al. (developing rat barrel cortex, L4): spiny stellate cell axonal field is
  "roughly cylindrical... cross-section ~250 µm diameter" (~125 µm radius).
- Converting to σ_EE: with connection probability ~exp(-d²/2σ²), characteristic radius ≈2-3σ.
  125 µm radius → σ ≈ 40-62 µm = 0.04-0.062 sheet-units. Our CURRENT σ_EE=0.08 (80 µm, 3σ=240µm
  diameter) is the SAME ORDER, slightly broader than this specific anchor but consistent.
  → NO PARAMETER CHANGE INDICATED. This is a genuine anatomical anchor, not a fit — current
  value happened to already be in the right range (set at confidence F/pragmatic originally;
  this is a coincidence worth flagging, not a retroactive justification — we did NOT choose
  σ_EE=0.08 because of this number, we are checking it AFTER the fact).
- CAVEAT (important, not to be glossed over): L4 spiny stellate is a SPECIFIC barrel-cortex
  cell type/layer, not necessarily representative of our model's generic "E population" per
  region (no explicit layers in our architecture). Species: rat (age within "developing",
  not dw20-human-specific). Proxy chain: barrel-cortex-L4 → generic immature cortical E-cell
  is an ASSUMPTION, not a direct match.
- SEPARATE FINDING — real cortical L2/3 horizontal connectivity is NOT single-Gaussian:
  clustered "patches" 300–500 µm diameter, with 800–1000 µm INTER-PATCH spacing (developing
  ferret/cat visual cortex, functional mapping). This reveals our single-Gaussian σ_EE kernel
  is a STRUCTURAL SIMPLIFICATION of real patchy/clustered horizontal connectivity — flagged as
  an architecture limitation for future work (patchy connectivity / structural plasticity that
  self-clusters), NOT actioned now (out of scope for current dw20 work).
- BONUS (Phase 3 anchor, not used yet): horizontal axon growth rate 29.9±1.7 µm/hour (range
  9.5–46.7), mean horizontal axon length in culture 1540±236 µm. Useful for grounding FUTURE
  activity-dependent long-range wiring growth rate (Phase 3), independent of any wave-speed
  number we derived ourselves (avoids circularity for that future task).
- Confidence: B6 σ_EE upgraded F→B (anatomical anchor found, but proxy cell-type/species);
  patchy-structure caveat A (real finding, unaddressed); growth-rate anchor A (for later use).

## I2. ★ SP "survival curve" — found, but mostly POSTNATAL, and the underlying premise
     ("subplate must mass-die") is CONTESTED depending on definition [critical caveat]
- Torres-Reveron & Friedlander 2007 (rat visual cortex, NeuN/MAP-2 counts): SP density
  P2→P30: 61% reduction, ending at ~29,000±1,800 neurons/mm³ (P30). White matter (WM)
  interstitial neurons: 85% reduction, ending ~500±60 neurons/mm³ (P30). Real curve with
  5 age points (P2,P9,P16,P23,P30) — a genuine survival curve, but POSTNATAL rat.
- eLife 2021 (mouse S1, Lpar1-EGFP-marked SPNs, P3-4 vs P5-6): EGFP+ SPN density drops
  significantly P3-4→P5-6, BUT caspase-3+ (apoptotic marker) cells are RARE (2/420 across
  5 animals) — i.e. the density drop is NOT clearly explained by caspase-3-mediated apoptosis
  alone; broader layer-specific TUNEL+ signal increases at P5-6 but isn't subplate-specific.
  METHODOLOGICAL FLAG: "density decline" ≠ "death" necessarily — could partly reflect marker
  downregulation or migration, not pure apoptosis. Do not equate density curve with death curve.
- Allendoerfer et al. 1990 (PNAS; ferret/cat, NGFR/p75 immunostaining): strong staining by E30
  (~3 weeks), declines ~E52, GONE by E60 — "at which time subplate neurons begin to die." This
  is a START-OF-DEATH-WINDOW marker, not a %-survival curve. Ferret/cat, not rodent or human;
  translatingtime to human GW not performed (flagged, open).
- ★★ CRITICAL CAVEAT — Valverde et al. 1995 (J Neurosci): DIRECTLY TESTED whether rodent
  subplate (defined as layer VIb) shows the expected mass death. RESULT: early-born VIb cohort
  count is STATISTICALLY UNCHANGED P1→P63 (ANOVA n.s.); VIb cell-death rate is NOT elevated
  vs other layers (except layer II and white matter, which DO show elevated death). This
  CONTRADICTS a simple "rodent subplate must mass-die" framing.
  → The survival-curve numbers above (Torres-Reveron 61% SP density drop) and this null result
  are NOT necessarily contradictory — they may be measuring DIFFERENT POPULATIONS under
  different operational definitions of "subplate" (transient SP-proper that disperses/migrates
  vs. layer VIb that persists vs. white-matter interstitial neurons that do show high death).
  The honest state: "does subplate die, persist as VIb, or migrate to WM interstitial pool?"
  is NOT a settled question across the field — it depends on which marker/definition is used.
- MODEL IMPLICATION (reprioritizes Phase 2, see I3): SP "apoptosis" should be implemented, if
  at all, as an EXPLICITLY FLAGGED HYPOTHESIS (one credible interpretation among several), not
  as settled fact. The quantitative timing (mostly POSTNATAL per Torres-Reveron/eLife 2021) ALSO
  means within our dw20-40 PRENATAL window, SP cell death may not even be the dominant SP-related
  transition — see I3.
- Confidence: density-decline numbers A (primary, but postnatal rodent); death-mechanism C
  (contested — Valverde 1995 directly disputes the "mass apoptosis" framing); translation to
  human dw20-40 timing F (no direct human data, postnatal-rodent timing doesn't map cleanly).

## I3. ★ Phase 2 re-prioritization: within dw20-40, the SP transition that matters most is
     FUNCTIONAL (waiting-period end, Kostović/Corbett-Detig/Allendoerfer 1990, ledger H4) —
     NOT cellular death (which per I2 is contested and mostly postnatal anyway)
- Given I2's finding that quantified SP density decline is largely POSTNATAL (rat P2-P30 ≈
  human term+ in translatingtime, very roughly) and that the underlying "mass death" premise
  is contested (Valverde 1995), SP apoptosis is DEMOTED from "Phase 2 primary mechanism" to
  "Phase 2 secondary/later mechanism, explicitly flagged as one hypothesis."
- PROMOTED to primary dw20-40 Phase 2 mechanism (already grounded, ledger H4, confidence A):
  the THALAMOCORTICAL WAITING-PERIOD TRANSITION — TH axons wait in SP (dw~15-24 PCW per
  Corbett-Detig), then invade cortical plate (dw 24-29+ per Kostović 2010), with NGFR/p75
  loss (Allendoerfer 1990, translated timing) marking a related structural transition window.
  This is a FUNCTIONAL/CONNECTIVITY gating transition (TH→SP relay weakens, TH→CP direct
  strengthens), independent of whether SP cells die, persist as VIb, or migrate — it does NOT
  require resolving the I2 controversy to implement.
- REVISED Phase 2 plan (supersedes the apoptosis-centric version from Part H):
    (a) PRIMARY: gate S_TH_P1_E (direct) to rise from near-zero at dw20 to full strength by
        dw~28-29, and correspondingly gate TH→SP→P1 relay strength to be the dominant pathway
        before dw24 — grounded in H4 (confidence A, two independent primary sources).
    (b) PRIMARY: local E→E STDP-driven potentiation within spindle-burst-like co-active
        domains (synaptogenesis-consistent, NOT pruning) — grounded in H6 (confidence A).
    (c) SECONDARY/HYPOTHESIS-FLAGGED: optional SP density decline as a slow process, with
        explicit uncertainty about mechanism (death vs. migration vs. marker change) and
        explicit uncertainty about prenatal vs. postnatal timing — implement only if/when
        directly useful, always labeled as contested.
    (d) Exuberant/transient long-distance projection elimination (Innocenti & Price 2005,
        ledger H6) remains grounded and relevant, independent of (c).
- Confidence: gating mechanism (a) A; STDP potentiation (b) A; SP density mechanism (c)
  explicitly C/contested; transient-projection elimination (d) A.

================================================================================
# PART G — Reviewer-blow verification round (2026-06-21): subplate, STDP, pruning, V_cond, spatial scale
================================================================================
Context: a "Reviewer 2" pass challenged three findings; user independently fetched primary
sources for the literature claims involved. Verifying BOTH the reviewer's claims AND my own
counter-claims (neither was checked before being asserted — protocol failure, logged).

## G1. Luhmann 2009 — subplate is an "AMPLIFIER", not a mandated pacemaker [VERIFIED, reframes #48]
- Exact source: Luhmann et al. 2009, "Subplate cells: amplifiers of neuronal activity in the
  developing cerebral cortex", Front Neuroanat 3:19. Core thesis: SP neurons CAN intrinsically
  discharge 10–20 Hz under cholinergic activation AND are connexin-36 gap-junction-coupled
  columnarly to developing cortical neurons → proposed role = LOCAL AMPLIFIER, grounded in
  PASSIVE properties (low resonant frequency + slow membrane τ ideal for summating repetitive
  SUBTHRESHOLD synaptic input) + gap-junction coupling — NOT "must spike as a pacemaker."
  A separate line (Lischalk/Moody) suggests pacemaker role but is NOT Luhmann's main thesis.
- Human anchor: Moore et al. 2009 — human fetal SP neurons at GW16–22 show similar electrophysiology
  to rodent → DIRECT human anchor for our dw20 window (not just rodent translatingtime proxy).
- MODEL IMPLICATION (reframe of #48's "BS_E/SP_E silent = not a bug"): the silence claim is LESS
  wrong than the reviewer implied — subthreshold amplifier function doesn't require overt spiking.
  BUT: our gap junctions measurably DAMPEN voltage fluctuations (noise_damp×0.37–0.46, finding #46
  section F), not amplify them — this is the OPPOSITE of "amplifier" in the voltage sense (though
  damping noise while passing coherent input IS a form of SNR amplification — untested in our model).
  HONEST STATUS: SP's "amplifier" role is NOT positively demonstrated in our model either — it's
  merely not strictly falsified by 0 Hz spiking. Open item, not resolved.
- Confidence: A (primary source, direct quote-level detail); human-anchor B (Moore 2009, properties
  not full circuit function).

## G2. STDP + depol-GABA — mechanism confirmed quantitatively, CORTEX TRANSFER UNCONFIRMED,
     WINDOW-WIDTH CLAIM UNCONFIRMED [important downgrade]
- Leinekugel et al. 1997 (Neuron) CONFIRMED with a real number: GABA-A activation drops the Mg2+
  block affinity from 16 µM to 118 µM, ENTIRELY via depolarization from −82 mV to −59 mV. System:
  rat CA3 HIPPOCAMPUS, P2–5. NOT cortex, NOT dw20 human — transfer to our model remains an
  ASSUMPTION, not a finding (downgrade from how the reviewer presented it as settled).
- ✗ NOT CONFIRMED: "STDP window is wider in immature synapses (tens of ms) vs adult (10–20 ms)" —
  general STDP literature does report "tens of ms" as TYPICAL even in MATURE synapses; a specific
  immature-vs-mature τ-window COMPARISON with a sourced number was NOT found. Mark [UNVERIFIED] —
  do NOT widen our STDP τ_pre/τ_post (currently 20 ms) based on this claim. Keep at validated default.
- INTERESTING CROSS-CHECK: our model's resting V_eff for immature E-pops (−54 to −58 mV, finding #46)
  is ALREADY about as depolarized as Leinekugel's "GABA-activated" endpoint (−59 mV) — i.e. our
  immature baseline already sits near where their acute hippocampal preparation needs active GABA
  input to reach. Possibly consistent (immature cortex is intrinsically more depolarized than acute
  hippocampal slice baseline) or possibly a sign our background depolarization is set high — NOT
  resolved either way; flagged for awareness, not actionable without more data.
- Confidence: mechanism A (primary, but wrong tissue/age); cortex transfer C (assumption);
  window-width claim REJECTED as unsourced — remove from any future Phase 2 STDP parameter changes.

## G3. Kostovic & Rakic 1990 — precise subplate timeline is REGION-SPECIFIC, not a single window
- Thalamic axons WAIT IN subplate GW17–26 ("waiting period") — dw20 (our model's window) is
  SQUARELY INSIDE this waiting period, BEFORE axons invade the cortical plate (GW26–34, when first
  thalamocortical synapses form). MODEL IMPLICATION: at dw20, functional TH→cortical-plate synaptic
  transmission (bypassing SP) is biologically premature — our S_TH_P1_E DIRECT projection may be
  anachronistic for dw20 (per this timeline, TH should communicate via SP relay only, weakly, during
  the waiting period). NOT yet acted on — flagged as an architecture question for the dw-trajectory,
  not an immediate dw20-snapshot fix (removing it now would need separate verification of whether P1
  bursting survives — F5-style test).
- SP dissolution is REGION-SPECIFIC, not "dw30–35 globally" (correcting both my and the reviewer's
  earlier vague framing): general dissociation ~GW33–34, BUT somatosensory cortex SP neurons PERSIST
  even at TERM (40 wk); prefrontal/association cortex SP gradually disappears over 6 POSTNATAL months.
  → our P1 ("primary somatosensory"-like) should NOT have SP apoptose by dw30–35 in any future Phase 2
  apoptosis mechanism — region-specific timing is required, with primary sensory regions persisting
  far longer than association cortex. A uniform global SP-apoptosis rule (as both I and the reviewer
  sketched) would be WRONG for P1 specifically.
- ✗ NOT FOUND: quantitative survival-fraction-vs-week curve (Allendoerfer & Shatz or equivalent) —
  remains an open, unverified item (Q3 from the literature-question list is only PARTIALLY closed).
- Confidence: A (primary source, precise GW windows); region-specificity A; survival curve OPEN.

## G4. V_COND=0.1 mm/ms — axonal-CV vs emergent-wave-speed are DIFFERENT quantities; emergent
     speed is ~40× too fast for the PV-absent dw20 regime [actionable, reframes #48 risk]
- CLARIFICATION (my own conflation, caught in verification): "V_COND" in gen_bundle models SINGLE-
  AXON conduction velocity (used as dist/V_COND per synapse delay) — ledger B7 already correctly
  scopes this as axonal CV (~0.1–0.3 m/s immature unmyelinated, TRAINING-sourced, confidence B), NOT
  network-level wave-propagation speed. The retinal-wave (451±91 µm/s = 0.451 mm/s, gap-junction,
  stage-1 ganglion cells) and PV-interneuron-gap-junction (59.1 mm/s, elevated K+) numbers the user
  found are EMERGENT MULTI-HOP network wave speeds — the correct comparison is against OUR MODEL'S
  OWN EMERGENT propagation speed (computed from delay+integration chained over hops), NOT directly
  against V_COND.
- COMPUTED: our model's emergent E→E propagation speed ≈ 1 mm / 55 ms ≈ 18.2 mm/s (from finding #46/
  this session: delay 2 ms + τm 13.5 ms per ~6.25 hops to cross a 1 mm sheet at σ_EE=0.08).
  vs retinal-wave 0.451 mm/s → our emergent speed is **40.3× faster**.
  vs PV-gap-junction wave 59.1 mm/s → our emergent speed is 0.31× (same order, plausible match).
- THE PROBLEM: PV interneurons are LARGELY ABSENT at dw20 in our OWN model (finding #32) and per
  literature (ledger D6, PV postnatal) — so the FAST PV-gap-junction mechanism (59.1 mm/s) is NOT
  biologically available yet. The SLOW retinal-wave-LIKE mechanism (chemical synapse + immature gap
  junction, no fast-spiking interneurons) is the appropriate comparison for a PV-poor network. Our
  emergent ~18.2 mm/s is therefore ~40× faster than the dw20-appropriate biological regime.
- ACTIONABLE HYPOTHESIS (testable, not yet run): reducing σ_EE (forces more, smaller hops to cross
  the sheet) should slow emergent propagation toward the ~0.45–1 mm/s retinal-wave-like range. At
  that speed, 1 mm sheet crossing ≈ 1–2 s — COMPARABLE to bouffée duration (1–3 s) rather than
  negligible relative to it (55 ms) — meaning genuine spatial/wave structure should become VISIBLE
  (testable via the per-neuron raster logger) rather than washed out by the bouffée's much longer
  duration. This is the corrected, properly-grounded version of the reviewer's σ_EE suggestion —
  not because "flat autocorrelation proves instant ignition" (disproven, see G5) but because the
  PV-absent dw20 regime should match a specific, slower literature-measured wave speed.
================================================================================
# PART G7 — TH→SP synapse strength + SP timing in spindle bursts (dw20 relay)
================================================================================
## G7.1. TH→SP synaptic strength: QUANTIFIED [Hanganu, Kilb & Luhmann 2002, J Neurosci]
- PRIMARY SOURCE: Hanganu IL, Kilb W, Luhmann HJ (2002). Functional synaptic projections
  onto subplate neurons in neonatal rat somatosensory cortex. J Neurosci 22(16):7165-76.
  Rat P0-P3, S1 slices, whole-cell recordings of SP neurons, electrical stimulation of
  thalamocortical afferents (TA stimulation).
- MEASURED (evoked monosynaptic TA-PSC at Vhold=-70mV, AMPA-component):
    Range: 12–79 pA; mean: 29.7 ± 2.7 pA (n=40 cells)
    Onset latency: 5.7 ± 0.3 ms (monosynaptic confirmed by TTX block, latency distribution)
    68% of SP neurons showed TA-PSC (59 cells total)
  → Converting PSC to conductance (g = PSC / ΔV, ΔV = E_AMPA - Vhold = 0-(-70) = 70 mV):
    g_min  = 12/70 = 0.171 nS
    g_mean = 29.7/70 = 0.424 nS  ← USE AS MODEL TH→SP MEDIAN WEIGHT (upgrades B6 confidence)
    g_max  = 79/70 = 1.129 nS
- R_in measured in same cells: ~1342 MΩ ≈ 1.342 GΩ
    Our model SP_E R_in = 1/(0.5+0.325+0.585) = 709 MΩ → model is 1.9× LESS sensitive.
    This means our SP needs ~1.9× more synaptic drive than biology to fire — a known limitation.
- EPSP per single synapse (undepressed, charge approx Q=g×ΔV×τ/C; ΔV=55.2mV, τ=3ms, C=23pF):
    bio g_mean:   0.424 × 55.2 × 3 / 23 = 3.05 mV → need ~1.7 coincident to reach 5.2mV threshold
    Our 0.28nS:   0.28  × 55.2 × 3 / 23 = 2.02 mV → need ~2.6 coincident (undepressed)
    With STP_MILD U=0.25 (our model): eff_g_first_spike = g_mean × 0.25 = 0.106 nS
    → EPSP first-spike = 0.76 mV → need 6.8 coincident inputs to fire SP
    BOTTLENECK: STP_MILD's U=0.25 reduces first-spike EPSP to ~25% of undepressed value.
    Without STP (pure conductance), relay IS feasible with just 2 inputs biologically.
- NMDA component at +10 mV: decay τ = 68.7 ± 7.7 ms (NR2B-dominated by ifenprodil result)
- MODEL ACTION: update TH→SP median weight 0.28 → 0.42 nS (bio-grounded, direct measurement)
- Confidence: A (primary source, direct measurement, age-matched P0-P3 rat=~GW20 equiv);
  R_in mismatch documented as model limitation (our SP R_in 1.9× too low → less excitable).

## G7.2. SP is ACTIVE during spindle bursts — leads cortical oscillation cycles by 13.8ms
- Yang et al. 2009 (J Neurosci 29:13496, already in H1/D2): during spindle bursts, MUA in
  subplate LEADS the negative peak of each oscillatory cycle by 13.8 ± 14.3 ms (n=281 cycles
  from 80 spindle bursts, 6 animals). This directly confirms SP is an ACTIVE PARTICIPANT in
  spindle burst generation, not a passive bystander — supports Luhmann "amplifier" framing.
- Tolner et al. 2012 (PMC3517992, already H3): SP removal strongly abolishes spindle bursts.
  Causal evidence that SP is needed.
- HONEST STATUS: direct "TH→SP delay in ms" (cross-correlation between TH burst timing and
  SP burst timing in simultaneous recording) not found in this literature search. Relay
  architecture supported INDIRECTLY (strong) but "TH fires → SP fires with X ms delay" is
  an open specific quantitative debt.
- Confidence: A (Yang 2009 primary, SP leads CP cycles); TH→SP delay number OPEN.

## G7.3. SYNTHESIS — current relay feasibility in model (honest)
- With bio weight (0.42 nS) + STP_MILD (U=0.25): need ~6.8 coincident TH spikes to fire SP
  (vs ~1.7 in biology without STP). Our 4 TH inputs per SP cell (p=0.006 density) can't
  provide 6.8 simultaneous → relay NOT feasible with current density and STP.
- To enable BIOLOGICALLY GROUNDED relay, need EITHER: remove/weaken STP on TH→SP (no source),
  OR increase TH→SP density 3-5× (no direct convergence measurement found, open debt).
- CURRENT MODEL BEHAVIOR: SP fires INTRINSICALLY at IBI_GAIN=4.0 (suprathreshold V_eff,
  regulated by STD-SOC + I_CAN + Ca-AHP, Luhmann intrinsic discharge ✓), NOT via TH→SP relay.
  SP firing is nonetheless biologically appropriate (Moore 2009: SP active at GW16-22).
  The "relay" function of SP is present architecturally but not the dominant IBI-pacing mechanism.
- ACTION: update TH→SP weight to 0.42 nS (grounded). Flag density as an unverified debt.
  Keep IBI_GAIN=4.0 as SP activation mechanism (intrinsic, not relay).

- Confidence: A (Hanganu 2002 weight measurement); density OPEN (no convergence data); 
  IBI_GAIN=4.0 confidence C (calibrated to match Moore 2009 "active" constraint).

- Confidence: V_COND itself B (unchanged); emergent-speed computation A (our own model, computed);
  retinal/PV-gap comparison numbers A (primary, user-fetched); the σ_EE fix is a HYPOTHESIS to test.

## G7b. TH→SP convergence: BOUNDED INFERENCE ~4-9 effective inputs per SP neuron
- PRIMARY SOURCE: Cerebral Cortex 2019 ("Mechanisms of spontaneous electrical activity in the
  developing mouse subplate zone"): WM-evoked compound EPSC on SP neurons (P1-P6 mouse):
  mean 106.8 ± 1.8 pA (1131 stimulations, 15 neurons); failure rate = 2.5% (very low).
- BOUNDED INFERENCE (NOT measured N): if single-synapse contribution ~12-30 pA (Hanganu 2002),
  then N_eff = 106.8 / (12-30) ≈ 3.6-8.9 → WORKING RANGE: ~4-9 effective thalamic inputs/SPN.
  This is a FUNCTIONAL CONVERGENCE ESTIMATE (inputs that contribute to observed compound EPSC),
  NOT the anatomical synapse count (which would require EM reconstruction of labeled TH axons
  onto an identified SP neuron — not found in accessible literature).
- CURRENT MODEL: ~4 TH inputs per SP cell (p=0.006×800/640 at SCALE=0.1) — at LOWER end of 4-9.
  Could justify increasing to 6-8, but density is a calibration choice; honest marking = C.
- Confidence: amplitude scale A (Hanganu 2002 primary); N_eff INFERRED (C, bounded by amplitudes);
  anatomical N = OPEN (not found as measured parameter in accessible sources).

## G8. SP region-specific background reduction: GROUNDED R_in FIX [Cerebral Cortex 2019]
- PRIMARY SOURCE: Cerebral Cortex 2019 (mouse P1-P6 SP zone, referenced above): SP spontaneous
  plateau depolarizations PERSIST under full AP-dependent synaptic blocker cocktail (TTX 1µM +
  DNQX + APV + bicuculline + strychnine) but are DRASTICALLY REDUCED by gap junction blockers
  (octanol; lanthanum). → SP baseline activity is PREDOMINANTLY gap-junction/hemichannel-driven,
  NOT AP-dependent synaptic. Therefore, high ge0+gi0 background conductance for SP is biologically
  UNJUSTIFIED — it artificially suppresses R_in while the real baseline comes from gap junctions,
  not from tonic synaptic release at AMPA/GABA-A receptors.
- R_IN TARGET (Hanganu 2002, Luhmann 2009): developing SPN in vitro R_in ≈ 1-1.342 GΩ.
  Our model SP_E: ge0+gi0 = 0.325+0.585 = 0.910 nS → R_in = 709 MΩ (1.9× too low).
- IMPLEMENTATION: SP_BG_SCALE = 0.2694 ≈ 0.27 (ADAM_SP_BG_SCALE=0.27) scales ge0, gi0, sig_e,
  sig_i for SP_E only. Not a free parameter — SET TO MATCH MEASURED R_in.
  After: ge0=0.088nS, gi0=0.158nS, g_total=0.745nS → R_in=1342 MΩ ✓ (exact Hanganu 2002 match).
  V_rest with SP_BG_SCALE=0.27: -63.9 mV (more hyperpolarized due to less ge0 excitatory drive).
- CONSEQUENCE FOR IBI_GAIN: with V_rest=-63.9 mV, IBI_GAIN threshold for V_eff>VT is 2.6 (vs 2.82
  before). IBI_GAIN=3.0 → V_eff=-47.8mV (2.2mV above VT) → intrinsic bursting.
  IBI_GAIN=3.0 is LOWER than previous 4.0 (more conservative), biologically better.
- ADDITIONAL BENEFIT: TH→SP EPSP per synapse improves: 0.42nS × 55.2 × 3/23 with R_in=1342MΩ
  gives EPSP upper bound (I_peak × R_in) ≈ 29.7pA × 1342MΩ = 40 mV — larger than before (R_in
  correction makes synaptic inputs more effective, consistent with Hanganu 2002 observed PSC scale).
- Confidence: gap-junction-driven baseline A (Cerebral Cortex 2019 primary); R_in target A
  (Hanganu 2002 + Luhmann 2009 primary); IBI_GAIN=3.0 CALIBRATED C (confidence C, Moore 2009
  constraint: SP active — satisfied).

## G5. Flat spatial autocorrelation does NOT prove instant ignition [self-correction, retracted claim]
- Reviewer claimed flat mean-rate spatial autocorrelation (#48) proves "instantaneous global
  ignition" (infinite propagation speed) and is an artifact requiring σ_EE reduction.
- VERIFIED FALSE AS STATED: computed emergent propagation time ≈55 ms to cross 1 mm sheet (G4) —
  finite, not instant. But 55 ms ≪ bouffée duration (1–12 s), so BOTH a 55 ms wave-sweep followed by
  region-wide co-activation, AND genuinely instant ignition, predict the SAME flat mean-rate
  autocorrelation when averaged over a multi-second bouffée. The mean-rate metric CANNOT distinguish
  these two scenarios — neither my original "expected global biology" claim NOR the reviewer's
  "proves instant ignition" claim is supported by the data we measured. Both were unverified leaps.
  → The σ_EE-reduction experiment is still worth running, but justified by G4 (PV-absent wave-speed
  mismatch), NOT by this disproven autocorrelation argument.

## G6. The "200–400 µm" domain size is conflated across THREE distinct phenomena — disambiguated
- Ledger D2 (PMC4806652, spindle bursts) is CORRECTLY the functional synchronization domain
  (200–400 µm diameter, ~5 bursts/min, 5–10 APs/neuron/burst) — this citation was already right.
- User found TWO additional, DISTINCT uses of similar numbers that must NOT be conflated with D2:
  (a) a SEPARATE paper where "200–400 µm" is the size of an EXPERIMENTER-IMPOSED optogenetic
      PV-stimulation zone (not a measured natural scale at all — irrelevant to grounding σ_EE);
  (b) calcium "neuronal domains" in neonatal rat cortex: 50–100 µm radius from initiating cells,
      recurring ~every 4 MINUTES — a much finer spatial scale AND a wildly different timescale
      (~240 s) than our IBI target (10–30 s). This is likely a DIFFERENT observable (fine-grained
      calcium microdomain signaling) than population-level tracé-discontinu/spindle-burst events,
      not a smaller version of the same phenomenon. Do not use to re-tune σ_EE for the IBI/bouffée
      mechanism — it answers a different question at a different timescale.
- ACTION: when citing "200–400 µm" going forward, cite D2/PMC4806652 specifically and note it is the
  SPINDLE-BURST functional domain, not blend with (a) or (b).
- Confidence: D2 A (unchanged, correctly sourced); (a) and (b) now logged to prevent future conflation.

================================================================================
# PART J — Phase 3: long-range wiring timeline + activity-dependent stabilization
================================================================================
Context: implementing activity-grown P2↔A2 long-range connectivity (lr_gate).
Current DEFAULT_LR_GATE onset_dw=34 is ~14 weeks too late. User-fetched literature
closes the timeline and the stabilization mechanism. All three sub-questions answered.

## J1. P2↔A2 structural onset: parietal terminations of indirect SLF appear ~dw20 [Horgos 2020]
- Horgos et al. 2020 (Front Neuroanatomy, white matter dissection of fetal human brains):
  SLF indirect component — including inferior parietal ↔ posterior STG branch (= P2↔A2) —
  has parietal terminations visible by DISSECTION at ~20 GW. Timeline:
    14 GW: peri-insular SLF portion identifiable.
    17 GW: frontal and temporal terminations begin to spread.
    20 GW: PARIETAL TERMINATIONS START TO APPEAR; SLF indirect widens.
    ≥23 GW: rapid addition of radiating fibers; tract becomes multilayered.
- Why DTI misses it: SLF not visible in fetal DTI/tractography at 19-20 GW (Huang et al.
  2006, NeuroImage) because fibers are thin/low-anisotropy — structural substrate exists
  BEFORE tractography can resolve it. DTI absence ≠ axon absence.
- MODEL IMPLICATION: lr_gate onset must start at dw20 (axons present), NOT dw34.
  Current DEFAULT_LR_GATE onset_dw=34 is WRONG by ~14 weeks — treating 14 weeks of real
  structural connectivity as zero is the same anachronism as TH→P1 waiting-period mistake.
- Confidence: A (primary human fetal dissection, Horgos 2020); DTI caveat A (Huang 2006).
  URL: https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2020.584266/pdf

## J2. Quantitative lr_gate growth rate: functional connectivity spurts from rs-fMRI [Hadders-Algra 2018]
- Hadders-Algra 2018 (review, fetal rs-fMRI summary citing Jakab 2014 and others):
    Temporal region connectivity SPURT: peak ~26 weeks PMA
    Parietal region connectivity SPURT: peak ~27-28 weeks PMA
  → lr_gate should reach ~50% around dw26-28 (parietal-temporal functional connectivity).
- Doria et al. 2010 (PNAS): RSNs recognizable from ~30 weeks PMA as fragments, becoming
  "facsimiles of adult networks" by term (~40 weeks).
- dHCP: primary RSNs (sensorimotor/auditory/visual) identifiable from ~26 weeks PMA.
- REVISED lr_gate sigmoid: onset_dw≈20 (structure appears), center_dw≈27 (functional spurt),
  full_dw≈38 (near-term RSN maturation). Steepness set to match 3% at dw20, 50% at dw27.
- Refinement/pruning for P2↔A2 specifically in dw20-40: NOT FOUND as quantitative curve.
  Available data describes GROWTH/invasion in this period, not elimination of exuberant
  branches. True refinement likely postnatal (consistent with H6/I3 synaptogenesis framework).
- Confidence: functional timeline A (Hadders-Algra 2018, cited primary fMRI data); pruning
  curve OPEN (confirmed as data absence, not a gap to fill — postnatal is the honest call).

## J3. LR stabilization mechanism: coactivation + NMDA coincidence, NOT millisecond STDP
- Wang et al. 2007 (J Neurosci, mouse S1 callosal projections):
    Kir2.1 suppression of pre-neuron excitability → abnormal targeting/arborization.
    TeNT-LC block of synaptic transmission → strong projection reduction (near-elimination).
  → LR refinement requires SPIKING + SYNAPTIC TRANSMISSION, not just structural presence.
  → Mechanism = activity + transmission dependent; STDP millisecond timing NOT measured.
- NMDA coincidence detection (NCBI Bookshelf "NMDA Receptors and Brain Development"):
  NMDAR as coincidence detector for synapse stabilization. Coactivation + NMDA-dependent
  retrograde signal stabilizes branches — operationally = HEBBIAN COACTIVATION not STDP.
- For human cortex dw20-40: no published STDP rule for LR projections. Honest status:
  VERIFIED: activity + synaptic transmission needed for LR refinement (callosal analog).
  NOT VERIFIED: exact STDP timing window for LR at this age. Mark as ASSUMPTION if used.
- MODEL IMPLEMENTATION: lr_gate grows faster when P2 and A2 are coactive (burst together).
  Proxy metric = cross-correlation of episode spike rates (fast to compute, no STDP kernel).
  Mechanism confidence: A (Wang 2007 callosal analog); NMDA coincidence A (review);
  coactivation proxy as STDP substitute: C (assumption flagged).

## J-SYNTHESIS: revised lr_gate for Phase 3 implementation
- Genetic baseline: sigmoid_dw(dw, center=27, steepness=0.5) → 3% at dw20, 50% at dw27,
  97% at dw34. This replaces the biologically-wrong onset_dw=34 full-gate-off.
- Activity-dependent boost: lr_gate += learning_rate × max(0, corr(P2_rate, A2_rate) - theta)
  where corr is measured per episode (Pearson over 50ms bins), theta = baseline correlation.
  Boost is ADDITIVE to genetic baseline, capped at 1.0.
  → When P2+A2 burst together (high corr) → lr_gate grows faster than genetic alone.
  → When P2+A2 burst independently (low corr) → only genetic baseline.
- Falsification: silence P2 afferents → P2/A2 decorrelate → lr_gate grows slower → at dw30
  P2↔A2 connection is weaker than intact. Measurable in the trajectory (not just snapshot).
- Confidence: genetic timeline A (J1/J2); activity-boost mechanism B (callosal analog + NMDA
  coincidence, but transferred from callosal to ipsilateral LR = assumption); boost magnitude
  (learning_rate, theta) CALIBRATION C — no direct human numbers available.
