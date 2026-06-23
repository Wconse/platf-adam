# ADAM → Custom CUDA/C++ Engine — Detailed Migration Plan

> **Audience:** the AI agent that will write all the code.
> **Source of truth for biology:** `Platform-Adam/` (Brian2 reference implementation) + its YAML configs.
> **Source of truth for the port:** this document.
> **Golden rule:** every milestone ends with a numeric comparison against a Brian2 reference. No milestone is "done" until its acceptance tests pass. Never advance with a failing or skipped test — fix or explicitly document the deviation.

---

## 0. Orientation: what we are porting and why

### 0.1 The system in one paragraph

ADAM simulates the developmental trajectory of a spiking neural network modelling 8 brain regions from 20 to ~144 developmental weeks (DW). The hot loop is a spiking simulation (AdEx excitatory neurons, LIF inhibitory neurons, AMPA/NMDA/GABA synapses, OU background noise, intrinsic homeostasis, STDP on P1/P2 E→E). The slow loop runs *between* episodes and updates developmental parameters (E_GABA reversal, Mg block, NMDA ratio, long-range gate, structural plasticity, synaptic scaling) via `DevStateVector`. Episodes are 30–120 s of physical time; between them the network is checkpointed to HDF5.

### 0.2 Why move off Brian2CUDA

The current pain (documented in `Platform-Adam/README.md`) is entirely a consequence of `cuda_standalone`:
- Topology is frozen at compile time → structural plasticity needs `SynapseBank`, `TopologyCache`, recompiles (30–120 s each).
- Checkpoint needs the brittle `run_args` string→object mapping.
- Each episode runs in an isolated `multiprocessing` process; state cannot live on the GPU across episodes.

A custom engine collapses all of this:
- Structural plasticity = mutate CSR arrays / weight masks in device memory. No recompile.
- Checkpoint = `cudaMemcpy` device→host→HDF5. No `run_args`.
- Episodes = one persistent process, state stays on GPU between episodes.

So the migration is justified by **architecture simplification**, not just speed.

### 0.3 What stays in Python (do NOT port to C++)

Keep these in Python; they are not hot and they encode biology that must stay readable:
- Config loading + unit parsing + `$add/$mul/$set` merge operators (`config_loader.py`, `unit_parser.py`).
- `DevStateVector`, `curves.py` (sigmoids), interventions/mechanisms, pharmacology, pathology.
- Metrics, spectral analysis, validation, comparator, report.
- Episode scheduling, HDF5 trajectory recording.

The C++/CUDA layer is **only the per-episode hot loop**: build device state from arrays handed in by Python → integrate N steps → return spikes/traces/state arrays. The cleanest boundary is: **Python computes all per-episode inputs as plain numpy arrays (weights, connectivity CSR, delays, per-step TimedArray samples, scalar params), writes them to a directory or shared buffer; the C++ engine reads them, runs, writes outputs back.** Later this can become a pybind11 in-process call, but file/array hand-off is the M0 contract and is good enough through M6.

### 0.4 Current state (already done — M0)

`cuda_adam/` already contains a verified **Milestone 0**:
- One region (P1), 80 AdEx E + 20 LIF I, **no synapses**.
- Kernels: `homeostasis_step → ou_noise_step → adex_dynamics_step → spike_detect_reset` (order matters, see §1.2).
- Pre-generated OU noise (`gen_noise.py`, seed 7) so Python-Euler and CUDA match bit-closely; Brian2 differs only by its internal RNG.
- Custom `.npy` reader/writer in `main.cu`, validation in `validation/compare_traces.py`.
- **Verified result:** Python-Euler (same noise) ≈ CUDA, E≈0.33 Hz, I≈8.8 Hz. Difference vs Brian2 is purely noise-driven.

This is the correct foundation. The plan below extends it.

### 0.5 Reference parameters & scale

- 8 regions: BS, TH, SP, P1, P2, A1, A2, M1. Full-scale neuron counts in `config/base/regions.yaml` total ~143k E + ~36k I. **Always develop at `SCALE=0.1`** (`simulation.yaml`) → ~14k E + 3.6k I, then validate scale-invariance at the end.
- `dt_phys = 0.1 ms`. Integrator: forward Euler (must match Brian2's `method='euler'`).
- Synapse counts: connection probabilities are tiny (e.g. P1 p_EE=0.0042) → at scale 0.1 each projection is hundreds–thousands of synapses. The full network is on the order of 10^5–10^6 synapses at scale 0.1. Plan data structures for up to ~10^7 synapses at full scale.

---

## 1. Numerical-equivalence contract (read before writing any kernel)

These rules make the port reproducible against Brian2. Every kernel must obey them.

### 1.1 Units convention (already in `units.cuh`)

All device values in "neuronal units": voltage mV, current pA, conductance nS, capacitance pF, time ms, rate Hz. Then `[pA]/[pF] = [mV/ms]` and `[nS]*[mV] = [pA]` hold with no scale factors. Keep this. Do **not** introduce SI conversions inside kernels.

### 1.2 Update order within one timestep

Brian2 with `method='euler'` computes every state derivative from the **state at the start of the step**, then applies all updates, then checks thresholds/resets, then propagates spikes through synapses (respecting delays). Our kernel decomposition must reproduce this *semantics*, not Brian2's internal scheduling. Fixed per-step order:

```
for each step:
  0. sample TimedArrays at t   (E_GABA[r], Mg, nmda_ratio, lr_gate, stress, afferent rates) — host or a tiny kernel
  1. homeostasis_step          (decay r_hat estimate, integrate I_homeo)   [reads old state]
  2. ou_noise_step             (update ge_bg, gi_bg via OU Euler-Maruyama)  [reads old state]
  3. deliver_spikes            (apply delayed spikes scheduled for THIS step to g_ampa/g_nmda/g_gaba) — see §3
  4. adex_dynamics_step        (update v, w; uses g_* incl. just-delivered)
  5. synapse_decay_step        (decay g_ampa, g_nmda, g_gaba)  — OR fold into step 4 RHS; see §1.3
  6. spike_detect_reset        (threshold, reset v/w, bump r_hat, set spike flags, enqueue spikes with delay)
  7. stdp updates              (on_pre/on_post weight changes for the spikes from step 6 / arrivals from step 3) — see §4
```

> **Important subtlety about conductance decay.** In Brian2 the equations integrate `dg_ampa/dt = -g_ampa/tau` *and* `on_pre` does `g_ampa_post += w` as a discrete event. Brian2 applies the synaptic jump at spike *arrival* (after delay), and the continuous decay is part of the neuron ODE solved each step. To match: treat `g_*` jumps as discrete additions in `deliver_spikes` (step 3), and decay `g_*` as part of the Euler step. The exact ordering of "add then decay" vs "decay then add" within a step produces a `O(dt/tau)` difference. **Match Brian2:** Brian2 evaluates the jump as an event at the clock step boundary, then the ODE update uses the post-jump value over the step. Implement as: deliver (add) at start of step → decay over the step. Validate the choice in M1 against the synapse reference; if rates drift >2%, flip the order and re-test.

### 1.3 float32 vs float64

- M0 used float32 for everything; rates matched. For longer runs and STDP, drift accumulates.
- **Decision:** keep `v`, `w`, `w_syn`, `I_homeo`, `r_hat` in **float64**; keep `g_ampa/g_nmda/g_gaba`, `ge_bg/gi_bg`, and noise in float32 (cheaper, decays fast, low sensitivity). This is the M1+ default. Make the precision a `typedef real_t` so it can be flipped globally.
- Document expected drift: float64 v drift vs Brian2 < 1e-6 mV/step is unattainable (different RNG) — we compare **statistics**, not traces.

### 1.4 Determinism policy

`atomicAdd` ordering on GPU is non-deterministic → bit-exact runs are impossible and **not required** (Platform README INVARIANT 6: reproducibility = KPI within tolerance, not bytes). Acceptance is always statistical. BUT: for debugging, provide a `--deterministic` flag that runs synapse delivery single-threaded-per-target (one thread per postsynaptic neuron accumulates its inputs) so results are run-to-run identical on a fixed GPU. Use this mode for all equivalence tests; use the fast atomic mode for production.

### 1.5 RNG

- Background OU noise and afferent Poisson draws need RNG. Use **cuRAND** with `curandStatePhilox4_32_10_t` (counter-based → reproducible given seed + step + neuron index, independent of thread scheduling). Seed scheme: `seed = global_seed (137/42/7) ^ hash(stream_id, neuron_id)`; advance by step via the Philox counter. This gives reproducible noise without the pre-generated-noise-file crutch used in M0.
- **Keep the pre-generated noise path too** (M0 style): it is the only way to get an exact Python-Euler vs CUDA match for the neuron-equivalence tests. So: M1 neuron tests use noise files; production uses cuRAND. Both code paths must exist.

### 1.6 The reference harness (build this FIRST, it pays for itself)

Generalize the existing `reference/` machinery into a reusable tool **before** M1:
- `reference/run_brian2_ref.py` should accept a config (region subset, scale, duration, seed, which features enabled) and dump: ICs, connectivity (CSR: `conn_pre, conn_post, conn_w, conn_delay, conn_type, conn_offset`), per-step noise (optional), per-step TimedArray samples, and outputs (`ref_spikes.npy`, `ref_rates.npy`, optional `ref_v.npy`, optional `ref_weights_final.npy`).
- `validation/compare_traces.py` should grow into `validation/validate.py` with named checks (rates, spike counts, weight distributions, PSD, KPI) and a JSON verdict per check. Each milestone adds checks; none are removed.

---

## 2. Data structures (define once, in a header, reuse everywhere)

Create `cuda/include/engine_types.cuh`:

### 2.1 Neuron population (Structure-of-Arrays)

```
struct NeuronPop {
  int   N;                 // neurons in this population
  bool  is_excitatory;     // E uses AdEx (w, exp term); I is LIF (no w, DeltaT=0)
  // state (device pointers)
  real_t *v, *w;           // w only allocated for E
  float  *g_ampa, *g_nmda, *g_gaba;
  float  *ge_bg, *gi_bg;
  real_t *r_hat, *I_homeo;
  float  *last_spike_t;
  int    *spike_flag;      // 0/1 this step
  int    *refrac_until;    // step index until which refractory (cleaner than float compare)
  // per-neuron shared scalars promoted to params struct (see 2.3)
};
```

One `NeuronPop` per (region, type). 8 regions × {E,I} = 16 populations. Keep them in arrays indexed by a global population id. Maintain a global neuron id space `[0, N_total)` with per-population offsets so spikes/monitors use absolute ids (mirrors Brian2 monitor indexing in the reference scripts, e.g. `spike_i + N_E`).

### 2.2 Synapse projection (CSR, source-indexed for spike delivery)

For each projection (e.g. `S_P1_EE_STDP`, `S_TH_P1_E`, `S_AUD_TH`) store **outgoing** CSR keyed by presynaptic neuron, because spike delivery iterates "for each spiking pre-neuron, touch its targets":

```
struct Projection {
  char name[32];
  int  src_pop, dst_pop;       // population ids
  int  syn_type;               // 0=AMPA(+NMDA split), 1=GABA
  bool plastic;                // STDP on?
  bool gated;                  // multiply by lr_gate(t)?
  int  n_syn;
  // CSR over source neurons (size src_N+1):
  int    *row_offset;          // device, len src_N+1
  int    *col_index;           // device, len n_syn  → postsynaptic local id
  real_t *w_syn;               // device, len n_syn  (nS)
  int    *delay_steps;         // device, len n_syn  (delay in integer timesteps)
  // STDP per-synapse traces (only if plastic):
  real_t *Apre, *Apost;        // device, len n_syn
  // for on_post we ALSO need incoming CSR (keyed by dst) — see §4.3
  int    *in_row_offset;       // len dst_N+1   (optional, plastic only)
  int    *in_syn_index;        // len n_syn      maps dst-CSR slot → synapse index
};
```

### 2.3 Parameter structs

Extend the existing `AdExParams` (it's already good) to carry the per-population constants. Add a separate `SynParams` (tau_ampa, tau_nmda, tau_gaba, reversals E_AMPA/E_NMDA/E_GABA-handled-via-TimedArray) and `StdpParams` (tau_pre, tau_post, A_plus, A_minus, eta, wmax). Pass by value to kernels (cheap, register-resident).

### 2.4 Delay ring buffer (the one genuinely hard piece — see §3)

```
struct SpikeDelayBuffer {
  int   max_delay_steps;       // = ceil(max_delay_ms / dt) + 1
  int   n_targets;             // dst_N for this projection's target pop
  // ring of per-(target, ring-slot) accumulated conductance, OR event lists.
  // Chosen design: per-projection conductance ring (see §3.2).
  float *ring;                 // len = max_delay_steps * dst_N  (or per syn-type)
  int    head;                 // current step modulo max_delay_steps
};
```

---

## 3. Spike delivery with heterogeneous delays (core of M1–M2)

This is the part Brian2 hides and the single biggest source of bugs. Read this whole section before coding.

### 3.1 The problem

Each synapse has its own delay (uniform random in [d_min, d_max], see `cpu_delays` in `synapse_factory.py`; local 1–5 ms, thalamic 2–6 ms, long-range 8–20 ms). A presynaptic spike at step `t` must increment the postsynaptic conductance at step `t + delay_steps`. Delays span 1–200 steps (20 ms / 0.1 ms). We need an efficient way to "schedule" future conductance jumps.

### 3.2 Chosen design: per-target conductance ring buffer (recommended)

Because synaptic input is *additive* and we only ever read "what arrives at this target this step", we don't need to remember individual events — we can pre-accumulate into a ring indexed by arrival time.

**On spike of pre-neuron i (in `enqueue` after spike detection):**
for each outgoing synapse `s` of `i`: compute `arrival = (head + delay_steps[s]) % max_delay_steps`; `atomicAdd(&ring[arrival * dst_N + col_index[s]], w_syn[s] * gate)`. (Split AMPA/NMDA: maintain two rings, or one ring of a small struct; simplest is two rings `ring_ampa`, `ring_nmda` for excitatory projections, one `ring_gaba` for inhibitory.)

**At start of each step (in `deliver`):**
the slot `ring[head * dst_N + j]` holds the total conductance arriving at target `j` now. Add it to `g_ampa[j]` (etc.), then zero that slot, then `head = (head + 1) % max_delay_steps`.

**Cost:** memory = `max_delay_steps * dst_N * sizeof(float) * (#syn_types)`. For A2 at scale 0.1 (dst ~3600 E) and max_delay 200 steps: 200*3600*4*2 ≈ 11 MB per projection. Acceptable. At full scale A2 is 36000 → 115 MB per projection — watch total budget; long-range projections are few so it's fine, but **assert** total ring memory at build and log it.

**Why this over event-list ring:** no per-step sorting, no dynamic allocation, coalesced writes mostly fail (scatter by `col_index`) but atomics on conductance are cheap and the read-back is a clean coalesced pass. STDP needs spike *times*, not just summed conductance — handle that separately (§4), the conductance ring is only for the neuron drive.

### 3.3 Alternative (only if ring memory blows up at full scale): per-pre event queues

Keep a fixed-capacity ring of *spike events* per delay slot: `events[slot]` = list of (pre or syn ids) that arrive at that slot. More memory-efficient when firing is sparse, but needs atomic counters and a compaction pass. **Do not build this in M1.** Only consider it in M7 (scale-up) if profiling shows the conductance ring is the bottleneck. Document the decision either way.

### 3.4 Validation of delays specifically

Delays are the highest-risk code. Dedicated micro-test in M1 (see §M1 tests T1.4): two neurons, one synapse, known delay → assert the post conductance jumps exactly `delay_steps` after the pre spike, to the step. Then a 10-synapse fan-out with distinct delays → assert each arrival lands on the right step. These are *exact* tests (deterministic mode), not statistical.

---

## 4. STDP (M4, but design the hooks in M1)

From `synapse_factory.make_stdp_syn` (P1/P2 E→E only):

```
model:  dApre/dt  = -Apre /tau_pre  (event-driven)
        dApost/dt = -Apost/tau_post (event-driven)
on_pre:  g_ampa += w*(1-nmda_ratio); g_nmda += w*nmda_ratio
         Apre += A_plus
         w = clip(w + eta*Apost, 0, wmax)
on_post: Apost += A_minus
         w = clip(w + eta*Apre, 0, wmax)
```

Key points for the port:
- **Event-driven traces:** Brian2 decays `Apre/Apost` only when touched, using elapsed time since last touch. To match exactly, store `last_update_step` per synapse and decay by `exp(-(now-last)/tau)` on access. (Simpler alternative: clock-driven decay every step — diverges slightly; the event-driven form is required for equivalence. Implement event-driven.)
- **on_pre fires at spike arrival (post-delay)**, on_post fires at the postsynaptic spike (no delay in this model — Brian2 default post pathway has zero delay unless set). Be careful: `on_pre` weight update uses `Apost` *at arrival time*; `on_post` uses `Apre` at post-spike time.
- **Race conditions:** a synapse's `w` can be touched by on_pre (pre arrival) and on_post (post spike) in the same step. In Brian2 these are ordered (pre pathway then post pathway by default scheduling). Reproduce: run the pre-pathway weight update kernel, then the post-pathway kernel. Within each, different synapses are independent (one synapse touched by at most one pre arrival and one post spike per step given refractory ≥ dt). Use atomics only if a target neuron's multiple incoming synapses are processed concurrently — but each synapse is a distinct memory slot, so **no atomics on `w` needed**; parallelize over synapses.
- **on_post needs incoming CSR** (find all synapses whose dst = the spiking post neuron). Build `in_row_offset/in_syn_index` (§2.2) at projection construction.

---

## 5. Afferents (M5): Poisson input populations

From `afferents.py`: AUD, VEST, TACT, GSW are `NeuronGroup`s with `threshold='rand()<rates*dt'`, rates driven by `TimedArray`s (audio/vest/tact/gsw + tact_pulses). They are pure spike sources (no dynamics). Port as: per-afferent-neuron, each step draw `u ~ U(0,1)` (cuRAND), spike if `u < rate(t,channel)*dt`. `rate(t,channel)` comes from per-step TimedArray samples Python hands in (a `[n_steps, n_channels]` array). Channel index = `i % n_channels` (matches `ta_audio(t, i%32)` etc.). Then deliver through their projections like any other (they have synapses S_AUD_TH etc.).

---

## 6. TimedArrays: how time-varying developmental params enter the engine

Brian2 evaluates `ta_egaba(t)`, `ta_mg(t)`, `ta_nmda_ratio(t)`, `ta_lr_gate(t)`, `ta_stress(t)`, and afferent rate TAs each step. In the port, **Python precomputes the sampled arrays for the whole episode** and hands them to the engine:
- Scalars-over-time (Mg, nmda_ratio, lr_gate, stress): `float[n_steps]` each. The engine indexes by step (with the TA's own dt → upsample to phys dt by `floor(t / ta_dt)`, matching Brian2 `TimedArray` semantics exactly — **piecewise constant, not interpolated**).
- Per-region E_GABA: `float[n_regions][n_steps_lfp]`. Same piecewise-constant indexing at `dt_lfp`.
- Afferent rates: `float[n_channels][n_steps_audio_or_mid]`.

**Critical:** Brian2 `TimedArray(values, dt=DT)` returns `values[int(t/DT)]` clamped to last. Reproduce that indexing exactly (`int(t/dt)`, clamp to `len-1`). Off-by-one here shifts E_GABA and silently changes dynamics. Unit-test the indexer against Brian2 (§M2 T2.1).

---

## 7. Milestones

Each milestone: **Goal → Deliverables → Algorithm notes → Reference to generate → Tests (with counts & tolerances) → Acceptance → Pitfalls.**

Tolerances follow Platform INVARIANT 6 / ITERATION acceptance: Level-1/2 KPI within ±5% (mean rates), distributions by quantile, never bit-exact across RNG. "Exact" tests (deterministic mode, same noise file) require machine-epsilon agreement and are noted as such.

---

### MILESTONE 0 — DONE (baseline, single region, no synapses)

Already complete and verified. **Action for the agent:** do not modify M0 behaviour. Add a regression test `tests/test_m0_regression.py` that re-runs the existing P1-no-synapse case and asserts E and I mean rates equal the recorded M0 numbers (E≈0.33, I≈8.8 Hz with seed-7 noise) within 0.05 Hz. This guards against breaking the neuron core while adding synapses. **(1 test.)**

---

### MILESTONE 1 — Synapses + heterogeneous delays, single region P1 (E↔I, no STDP, no NMDA split yet)

**Goal:** P1 region (80E+20I at test size, or scale-0.1 sizes) with the four intra-region projections wired (E→E AMPA, E→I AMPA, I→E GABA, I→I GABA), delays honoured, matching the Brian2 synapse reference (`run_brian2_syn_ref.py`, which already exists and loads `connectivity.npz`).

**Deliverables:**
- `cuda/cuda_src/kernels/synapse_kernel.cu` (already referenced in CMakeLists — create it).
- `engine_types.cuh` with `NeuronPop`, `Projection`, conductance ring (§2, §3).
- Spike enqueue + deliver kernels (§3.2).
- A `connectivity` loader in `main.cu` that reads the CSR arrays the reference produces (`conn_pre, conn_post, conn_w, conn_delay, conn_offset, conn_type` — these `.npy` already exist in `reference/`).
- For M1 keep AMPA without NMDA split (set nmda_ratio=0) to isolate delay/delivery correctness. NMDA split is M2.

**Algorithm notes:**
- Build outgoing CSR from `(conn_pre, conn_post, conn_w, conn_delay)` sorted by pre. The reference already provides `conn_offset` (CSR row offsets) — use it; assert `conn_offset[src_N] == n_syn`.
- `delay_steps = round(delay_ms / dt_phys)`. Assert all ≥1 (Brian2 min delay here is 1 ms = 10 steps; a 0-delay would need same-step handling — not present in this model, assert against it).
- Deliver before dynamics (§1.2 step 3 → 4). Decay g after (or fold into RHS). Pick one, validate, document.
- Use deterministic mode + the **same noise files** the reference consumes so this is an (almost) exact test, isolating synaptic code from RNG differences.

**Reference to generate:**
- Run `reference/run_brian2_syn_ref.py` → `syn_ref_spikes.npy`, `syn_ref_rates.npy`, `syn_ref_params.json` (already wired).
- Add: dump final `g_ampa/g_gaba` snapshots and, for the micro-tests, a hand-built 2-neuron / 10-synapse connectivity with known delays.

**Tests (8 total):**
- **T1.1 (exact, micro):** 2 neurons, 1 synapse, delay = 30 steps, force a pre spike at step 100 (inject via I_ext or a forced threshold). Assert post `g_ampa` jumps by exactly `w` at step 130, zero before. Tolerance: |Δ| < 1e-6.
- **T1.2 (exact, micro):** 1 pre → 10 post, delays = 10,20,…,100 steps, one pre spike. Assert each post receives its jump at the exact predicted step. Tolerance: 1e-6.
- **T1.3 (exact, micro):** fan-in: 5 pre → 1 post, same delay, simultaneous spikes. Assert post g = sum of 5 weights (tests atomic accumulation correctness). Tolerance: 1e-6.
- **T1.4 (exact, conductance decay):** single synapse, one spike; sample `g_ampa(t)` for 100 steps post-arrival; compare to analytic `w*exp(-(t-t_arr)/tau_ampa)` and to Brian2's trace. Tolerance vs Brian2: < 1e-4 nS (this validates the add-then-decay ordering choice).
- **T1.5 (statistical, full P1):** run 10 s, same noise file as reference, deterministic delivery. Compare mean E rate and mean I rate to `syn_ref_rates.npy`. Tolerance: ±5% or ±0.3 Hz (whichever larger), per Platform L1 KPI.
- **T1.6 (statistical, full P1):** per-neuron rate vector correlation vs reference > 0.9; mean |Δrate| < 1.0 Hz; max |Δrate| < 3 Hz.
- **T1.7 (spike count):** total E and I spike counts within ±5% of reference.
- **T1.8 (M0 regression):** re-run T from M0 — neuron core unchanged.

**Acceptance:** T1.1–T1.4 exact-pass; T1.5–T1.8 within tolerance. If T1.5 fails but micro-tests pass → it's the decay-ordering or the EI/IE balance; flip ordering (T1.4) and re-check.

**Pitfalls:**
- Local→local index spaces: I→E synapses use *local* post ids (reference subtracts N_E). Keep per-population local ids in `col_index`; the population offset is applied only for global monitors.
- Forgetting to zero the ring slot after delivery → phantom recurring input.
- `head` advance must happen exactly once per step, after delivery, before enqueue of this step's new spikes (new spikes are scheduled relative to the *current* head).

---

### MILESTONE 2 — NMDA split + Mg block + per-step TimedArrays (still P1)

**Goal:** add the AMPA/NMDA split on excitatory synapses, the voltage-dependent Mg block `B_nmda`, and the TimedArray machinery (E_GABA(t), Mg(t), nmda_ratio(t), stress(t)). Reproduce Brian2 P1 with NMDA enabled.

**Deliverables:**
- NMDA conductance state `g_nmda` (already in M0 struct), decay `tau_nmda` (100 ms).
- In `deliver`: excitatory arrival splits `g_ampa += w*(1-nmda_ratio(t))`, `g_nmda += w*nmda_ratio(t)` (nmda_ratio sampled at arrival step). Matches `make_syn`/`make_stdp_syn` on_pre.
- In `adex_dynamics_step` RHS: `I_syn = g_ampa*(E_AMPA - v) + g_nmda*B_nmda*(E_NMDA - v) + g_gaba*(E_GABA(t) - v)` with `B_nmda = 1/(1 + Mg(t)*exp(-0.062*v))` and `E_GABA(t)` from the per-region TimedArray. (Exactly `neuron_models.py` EQS_E / `run_brian2_syn_ref.py`.)
- TimedArray sampler (§6): piecewise-constant, `int(t/ta_dt)` clamped.

**Reference to generate:** extend `run_brian2_syn_ref.py` to use non-trivial (time-varying) E_GABA, Mg, nmda_ratio TimedArrays over a 30 s episode (not the constant single-value TAs it currently uses). Dump the sampled TA arrays so the engine consumes the identical samples.

**Tests (6 total):**
- **T2.1 (exact, TA indexer):** unit-test the sampler against Brian2 `TimedArray` for a ramp array at several dts and query times incl. boundaries (`t=0`, `t` just below/above a tick, `t > end`). Tolerance: exact equality of selected index.
- **T2.2 (exact, B_nmda):** evaluate `B_nmda(v, Mg)` on a grid v∈[-80,-20] mV, Mg∈{0.3,1.0} vs the Python formula. Tolerance: < 1e-6.
- **T2.3 (statistical, full P1, constant TAs):** reproduce M1 result with NMDA enabled at constant nmda_ratio (matches current `run_brian2_syn_ref.py` NMDA_RATIO=0.08). Mean rates within ±5%.
- **T2.4 (statistical, time-varying TAs):** 30 s episode with ramping E_GABA (depolarizing→hyperpolarizing). Mean rates within ±5%; assert qualitative E/I-ratio shift (GABA flips from excitatory to inhibitory) appears in both.
- **T2.5 (conservation/sanity):** with nmda_ratio=0 the engine must reproduce M1 bit-for-bit (deterministic mode) — guards the split code. Exact.
- **T2.6 (regression):** M1 + M0 tests still pass.

**Acceptance:** all exact tests pass; T2.3/T2.4 within ±5%.

**Pitfalls:** sampling `nmda_ratio` at *arrival* step (delivery) vs the neuron's RHS using `Mg(t)`/`E_GABA(t)` at the *current* step — they're different times; match Brian2 (delivery uses on_pre time = arrival; RHS uses current t). `exp(-0.062*v)` with v in mV (not volts) — the reference divides by mV; keep v in mV so the literal `0.062` matches.

---

### MILESTONE 3 — Multi-region wiring (all 8 regions, hierarchical + long-range + gate), no STDP yet

**Goal:** scale from 1 region to the full 8-region topology with inter-region projections (TH→P1, P1→P2, P2→A1, A1→A2, BS→M1, P2↔A2 long-range gated by lr_gate(t)), region-specific birth times and maturation gates. This is mostly bookkeeping on top of M1–M2.

**Deliverables:**
- Generalize the single-region loader to N populations + M projections, all driven from arrays Python emits (Python already builds this in `network_builder.py`; mirror its projection list exactly — see lines 202–336 there).
- Per-region `t_birth` and `T_mature` (A1/A2 differ) feeding the maturation gate `mat` already in the kernels.
- Long-range gate: `lr_gate(t)` multiplies the P2↔A2 weights at delivery (`base_expr = w_syn * lr_gate_ta(t)` in `make_syn`).
- Afferents still **off** in M3 (replace with their background drive only) to isolate inter-region wiring; afferents are M5.

**Reference to generate:** new `reference/run_brian2_multiregion_ref.py` — build the full network via the actual `network_builder.build_network` at SCALE=0.1, afferents muted, fixed seeds, 30 s. Dump per-region CSR, ICs, TA samples, and per-region spike counts/rates.

**Tests (7 total):**
- **T3.1 (build integrity):** assert engine's per-projection `n_syn`, `row_offset[-1]`, and weight sums equal the reference's per-projection values. Exact (same connectome seed → identical (i,j,w,delay)). This is a strong test: the connectome is generated by the *same* numpy RNG (`sample_pairs`, `lognormal_w`, `cpu_delays`) so it must match exactly when Python hands the arrays over. Tolerance: exact.
- **T3.2 (per-region rates):** all 16 populations' mean rates within ±5% (or ±0.3 Hz) of reference.
- **T3.3 (birth ordering):** assert no spikes in region r before its `t_birth` (e.g. A2 born at 37 DW → silent early). Exact boolean per region.
- **T3.4 (maturation gate):** sample `mat(t)` per region vs analytic; assert ramp from 0→1 over T_mature. Tolerance 1e-6.
- **T3.5 (long-range gate):** with lr_gate ramped 0→1, assert P2→A2 effective input scales accordingly (measure A2 input conductance with gate=0 vs gate=1). Statistical, monotone increasing.
- **T3.6 (inter-region propagation):** correlation between TH and P1 rate trajectories present (driven coupling), within qualitative agreement of reference.
- **T3.7 (regression):** M0–M2 pass.

**Acceptance:** T3.1, T3.3, T3.4 exact; T3.2 within ±5%; T3.5/T3.6 qualitative match.

**Pitfalls:** global vs local neuron ids across 16 populations; projection delay class (local/thal/long) selects the delay range — keep per-projection. Memory: sum the ring buffers across all projections and log it (§3.2). At SCALE=0.1 it's modest; assert < some budget (e.g. 2 GB) and fail loudly otherwise.

---

### MILESTONE 4 — STDP on P1/P2 E→E

**Goal:** event-driven STDP exactly as `make_stdp_syn`, on the two plastic projections (P1 E→E, P2 E→E). Weights evolve; final weight distribution matches reference quantiles.

**Deliverables:**
- Per-synapse `Apre, Apost, last_update_step` (event-driven decay, §4).
- Incoming CSR (`in_row_offset, in_syn_index`) for the post pathway.
- Pre-pathway kernel (on arrival): decay Apre/Apost to now, `g_*` jump (already in deliver), `Apre += A_plus`, `w = clip(w + eta*Apost, 0, wmax)`.
- Post-pathway kernel (on post spike): decay, `Apost += A_minus`, `w = clip(w + eta*Apre, 0, wmax)`.
- Ordering: deliver+pre-pathway, then dynamics, then spike detect, then post-pathway (matches Brian2 default pre-before-post scheduling).

**Reference to generate:** extend multiregion reference with STDP enabled, dump **final `w_syn` per plastic projection** (`ref_weights_final.npy`) and weight quantiles [10,25,50,75,90].

**Tests (7 total):**
- **T4.1 (exact, pair protocol):** classic STDP curve — one pre, one post, sweep Δt ∈ [-50,50] ms, single pairing, measure Δw. Compare to Brian2 same protocol. Tolerance: < 1e-4 nS per point. This is the definitive STDP-correctness test.
- **T4.2 (exact, trace decay):** after a spike, sample Apre over time vs `A_plus*exp(-Δt/tau_pre)`. Tolerance 1e-5.
- **T4.3 (no-op equivalence):** eta=0 → weights frozen → reproduces M3 exactly (deterministic). Exact.
- **T4.4 (statistical, distribution):** 30 s run; final weight quantiles [10,25,50,75,90] within ±10% of reference (Platform ITER-0 criterion is quantile match).
- **T4.5 (statistical, mean weight drift):** mean w trajectory direction matches reference (both potentiate or both depress); final mean within ±10%.
- **T4.6 (stability):** no NaN/Inf in weights; `dead_synapse_fraction` and `saturated_fraction` within ±5% of reference.
- **T4.7 (regression):** M0–M3 pass.

**Acceptance:** T4.1–T4.3 exact; T4.4–T4.6 within stated tolerances.

**Pitfalls:** event-driven decay needs `last_update_step` updated on *every* touch (both pathways). The sign convention: `A_minus` is negative (−1.05 in config), `eta>0`; depression comes from `w += eta*Apost` where Apost<0. Don't double-apply. Float64 for `w_syn` and traces (drift over long runs).

---

### MILESTONE 5 — Afferents (Poisson sources) + their projections

**Goal:** add AUD/VEST/TACT/GSW Poisson populations driven by per-step rate TimedArrays, wired through S_AUD_TH, S_AUD_P1, S_VEST_BS, S_TACT_BS, S_GSW_SP, S_GSW_P1. Reproduces the full sensory-driven network.

**Deliverables:**
- Poisson kernel: `spike if curand_uniform() < rate(t,channel)*dt` (§5). Channel = `i % n_ch`.
- Python emits the rate arrays (it already builds env via `base_env.generate_episode_environment`; reuse it to dump `audio/vest/tact/tact_pulses/gsw` sampled arrays).
- Afferent spikes feed the normal delivery path.

**Reference to generate:** full network reference (`run_brian2_multiregion_ref.py` with afferents on). Because Poisson draws differ between Brian2 RNG and cuRAND, afferent tests are **statistical only** (rate of the Poisson source, and downstream effect).

**Tests (5 total):**
- **T5.1 (statistical, source rate):** measured mean firing rate of each afferent population ≈ commanded `rate(t)` time-average, within ±5%. (Validates the `rand()<rate*dt` thinning.)
- **T5.2 (channel structure):** assert neurons in the same channel share the same rate signal (cross-correlation high within channel, lower across) — matches `i%n_ch` indexing.
- **T5.3 (downstream effect):** target regions (TH, BS, SP, P1) show increased drive with afferents on vs off; mean rates within ±10% of reference (looser — Poisson + RNG).
- **T5.4 (TimedArray 2D indexer):** unit-test `ta(t, channel)` 2D sampling vs Brian2 for audio/tact. Exact index selection.
- **T5.5 (regression):** M0–M4 pass.

**Acceptance:** T5.1, T5.2, T5.4 pass; T5.3 within ±10%.

**Pitfalls:** `rate*dt` must use dt in the same time unit as rate (Hz·ms needs dt in *seconds* → `rate_Hz * dt_ms * 1e-3`). Brian2's `rand()<rates*dt` uses dt in seconds implicitly. Get this unit right or afferents fire 1000× too much/little. Unit-test it (fold into T5.1).

---

### MILESTONE 6 — Episode boundary: checkpoint save/load + the Python↔C++ contract

**Goal:** make the engine resumable. Run an episode, dump full device state to HDF5 (same schema as `checkpoint.py`), reload into a fresh engine, continue, and show KPI continuity. This replaces Brian2's `run_args` machinery entirely.

**Deliverables:**
- `engine.save_state(path)`: `cudaMemcpy` all neuron SoA + all projection `w_syn,i,j,delay` (+ STDP traces) → host → HDF5 with the **exact group/dataset names** in `checkpoint.py` (`neurons/{r}_E/{var}`, `synapses/{name}/{w_syn,i,j,delay}`, `meta`, `dev_state`). This guarantees the existing Python analysis stack reads engine checkpoints unchanged.
- `engine.load_state(path)`: inverse.
- Define the **hand-off contract** explicitly in `cuda/IO_CONTRACT.md`: directory layout of input arrays (connectivity CSR, ICs, TA samples, scalar params JSON) and output arrays (spikes, rates, traces, final state). Python writes inputs, spawns/links engine, reads outputs.
- Decide carry-over policy per Platform "episode boundary" note: by default **do not** carry STDP `Apre/Apost` or in-flight delay-queue spikes across episodes (their timescales ≪ episode length). Provide a flag to carry them (drain the ring into the next episode) for strict-continuity mode. Implement the default (drop) first.

**Reference to generate:** a two-episode Brian2 run using the existing checkpoint round-trip (`episode_runner.py` already does save→load→run). Compare engine's episode-2 KPI to Brian2's episode-2 KPI.

**Tests (6 total):**
- **T6.1 (exact, round-trip):** save state → load into new engine → assert every device array equals the saved one (deterministic). Tolerance 0 (bytes) for state arrays since no RNG involved in copy.
- **T6.2 (exact, HDF5 schema):** open the engine's checkpoint with the Python `CheckpointManager`-reading code path; assert all expected groups/datasets/attrs present and shapes match. Boolean pass.
- **T6.3 (statistical, continuity):** episode-1 (30 s) → checkpoint → episode-2 (30 s). Compare episode-2 mean rates to a single continuous 60 s run (same noise). Within ±5% (the boundary drops short traces, so small diff expected). Matches Platform ITER-1 criterion (KPI within ±5% after load).
- **T6.4 (cross-engine):** load a checkpoint produced by the **Brian2** pipeline (`episode_runner`) into the CUDA engine; run; compare to Brian2 continuation. Within ±5%. (Proves drop-in compatibility.)
- **T6.5 (carry-over mode):** with STDP-trace carry-over enabled, continuity tightens (episode-2 vs continuous within ±3%). Confirms the flag works.
- **T6.6 (regression):** M0–M5 pass.

**Acceptance:** T6.1, T6.2 exact; T6.3, T6.4 within ±5%.

**Pitfalls:** HDF5 dataset dtypes (Brian2 stores SI? No — `checkpoint.py` stores raw `getattr(G,var)[:]` which are unitless arrays in Brian2's internal units — **check units**: Brian2 internal stores SI (volts, siemens, amps). The engine uses mV/nS/pA. **You must convert on save/load** (×1e-3 for mV→V etc.) so the schema is bit-compatible with the Brian2 checkpoints, OR define the engine checkpoint as "neuronal units" and add a converter for cross-engine tests T6.4. Pick: engine-native units in its own checkpoints + an explicit converter module for Brian2 interop. Document loudly.)

---

### MILESTONE 7 — Scale-up, performance, full trajectory

**Goal:** run the full schedule (e.g. `prenatal_postnatal.yaml`, 20→52 DW, dozens of episodes) at SCALE=0.1 and then push SCALE toward 1.0; profile and optimize; confirm scale-invariance of KPIs.

**Deliverables:**
- Profiling harness (CUDA events around each kernel; per-step and per-episode timing; memory high-water mark).
- Optimizations as needed: fuse decay into dynamics RHS; coalesce ring read-back; consider event-queue delays (§3.3) only if conductance ring memory is the wall at full scale; tune block sizes; use `__ldg`/restrict; pinned host buffers for checkpoint copies; overlap copy with compute via streams.
- Multi-GPU / multi-process screening (Platform "subprocess parallelism" note): one episode-config per process; keep it simple.

**Reference to generate:** Brian2 baseline trajectory at SCALE=0.1 over the full schedule (the existing `experiments/run_baseline.py`). Compare KPI trajectories.

**Tests (7 total):**
- **T7.1 (trajectory KPI):** for each checkpoint DW (28/32/36/40/44/52), engine vs Brian2 Level-1 KPIs (mean rates per region, burst_rate, EI_ratio) within ±5–10%.
- **T7.2 (developmental KPI, Level-4):** assert the qualitative milestones (no spikes before birth_dw; discontinuity high early then dropping; E_GABA flip visible) — same pass/warn/fail verdicts as Brian2.
- **T7.3 (scale invariance):** run SCALE=0.1 and SCALE=0.3; assert mean rates and KPI shapes stable within ±15% (per ITER-4 spirit). Documents that the engine isn't size-pathological.
- **T7.4 (stability, long run):** 20+ episodes; assert no NaN/Inf, rates stay within ±20% of r_target (homeostasis works), `dead_synapse_fraction` < 5%.
- **T7.5 (performance):** report speedup vs Brian2CUDA on the same case; assert engine is at least not slower (target: ≥3× on the spiking hot loop at scale 0.1; record the number).
- **T7.6 (memory):** assert peak device memory < GPU budget at the target scale; log ring-buffer share.
- **T7.7 (regression):** M0–M6 pass.

**Acceptance:** T7.1–T7.4 within tolerances; T7.5/T7.6 reported (perf is informational, correctness is gating).

**Pitfalls:** runaway excitation if homeostasis/scaling timing is off; ring memory at full scale (long-range projections to A2); float32 g-decay error accumulating over 10^6 steps (it shouldn't — g decays fast — but watch). Don't optimize before T7.1 passes; correctness first.

---

### MILESTONE 8 — Slow loop integration + structural plasticity (the payoff)

**Goal:** wire the Python slow loop (DevStateVector, interventions, synaptic scaling, structural plasticity) to the persistent engine, exploiting the no-recompile advantage. Demonstrate a DNT experiment (VPA) end-to-end.

**Deliverables:**
- Python orchestrator: per episode, compute DevState → TA samples + scalar params → call engine.run_episode → read state → apply synaptic scaling (multiplicative on weights) + structural pruning/growth (mutate CSR/weights in place on device via a small kernel or host-side rebuild of CSR) → next episode. No recompilation ever.
- Structural plasticity on device: pruning = set `w_syn=0` where `w < threshold` (trivial kernel); growth = activate reserve synapses or rebuild CSR host-side and re-upload (cheap now — no compile). Implement the rebuild path (simplest, matches `structural_plasticity.apply_pruning`).
- Intervention plumbing: `EI_Imbalance` → per-region `I_bias` delta (already a scalar param); `KCC2_Inhibition/GABA_Flip_Delay` → shifted E_GABA TA; `LR_Connectivity_Deficit` → lr_gate multiplier; all computed in Python, handed in as arrays/scalars.

**Reference to generate:** Brian2 VPA experiment (`experiments/run_dnt.py --compound valproate --dose 100`) trajectory.

**Tests (6 total):**
- **T8.1 (baseline parity):** full slow-loop baseline trajectory vs Brian2 baseline — same KPIs as T7.1, within ±5–10%.
- **T8.2 (synaptic scaling):** after scaling, weight multipliers match `apply_synaptic_scaling` formula exactly (deterministic, host-side). Exact.
- **T8.3 (pruning):** after pruning, the set of zeroed synapses matches `apply_pruning(threshold)` exactly. Exact.
- **T8.4 (intervention effect):** VPA-100 run shows delayed GABA flip (E_GABA at 32 DW shifted, per the experiment's `$add` deltas) — same direction & magnitude as Brian2. Within ±5% on the affected KPI.
- **T8.5 (no-recompile proof):** structural change mid-trajectory completes in < 1 s (vs Brian2's 30–120 s recompile) — record the timing as the headline win.
- **T8.6 (regression):** M0–M7 pass.

**Acceptance:** T8.1 within tolerance; T8.2, T8.3 exact; T8.4 qualitative+quantitative match; T8.5 timing recorded.

**Pitfalls:** keep all biology in Python — the engine only takes numbers. Don't let structural plasticity logic leak into C++. CSR rebuild must preserve STDP traces for surviving synapses (map old→new synapse index) or explicitly reset them (document which; resetting is acceptable per the trace-timescale argument).

---

## 8. Cross-cutting test infrastructure (build alongside M1, extend every milestone)

- `validation/validate.py`: registry of named checks; each returns `{name, value, tolerance, pass}`. CI-style summary + JSON. **Every milestone adds checks, removes none.** Final suite ≈ 50+ checks.
- `tests/` mirrors milestones: `test_m1_synapses.py`, … `test_m8_slowloop.py`. Each is runnable standalone and in a `run_all.sh`.
- **Reference cache:** generate Brian2 references once per config hash, cache under `reference/cache/<hash>/`. Regenerate only when the relevant YAML changes (hash the inputs).
- **Deterministic mode** is the default for equivalence tests; **cuRAND mode** for perf/production.
- **Tolerance table (single source):** Level-1 rates ±5% (or ±0.3 Hz); per-neuron |Δrate| mean < 1 Hz; weight quantiles ±10%; exact tests < 1e-4 (traces) / 1e-6 (algebraic) / 0 (copies); afferent/Poisson ±10%.
- Always run the **full regression** (all prior milestones) before declaring a milestone done — listed as the last test of each.

---

## 9. Total test count & timeline

| Milestone | Focus | Tests | Est. effort (1 dev) |
|-----------|-------|-------|---------------------|
| M0 | neuron core (done) + regression | 1 | done |
| M1 | synapses + delays (P1) | 8 | 1.5–2.5 wk |
| M2 | NMDA/Mg + TimedArrays | 6 | 1 wk |
| M3 | 8-region wiring | 7 | 1–1.5 wk |
| M4 | STDP | 7 | 1.5–2 wk |
| M5 | afferents | 5 | 0.5–1 wk |
| M6 | checkpoint + IO contract | 6 | 1–1.5 wk |
| M7 | scale-up + perf | 7 | 1.5–2 wk |
| M8 | slow loop + structural | 6 | 1.5–2 wk |
| **Total** | | **53** | **~2.5–3.5 months** |

Critical path / highest risk: **M1 delays (§3)** and **M4 STDP event-driven traces (§4)**. Budget extra time there; both have exact micro-tests that catch errors early — write those tests *first* (TDD) within each milestone.

---

## 10. Order of operations for the implementing agent (checklist)

1. Build the reference harness generalization (§1.6) and `validate.py` skeleton. Add M0 regression test (T from M0). Confirm green.
2. **M1:** write micro-tests T1.1–T1.4 FIRST (they need only a tiny hand-built connectome). Implement ring buffer + deliver/enqueue until they pass. Then T1.5–T1.8 against `run_brian2_syn_ref.py`.
3. **M2:** TA sampler + B_nmda unit tests first (T2.1, T2.2), then NMDA split, then statistical.
4. **M3:** connectome-equality test first (T3.1), then rates.
5. **M4:** STDP pair-protocol test first (T4.1), then distributions.
6. **M5:** Poisson rate test first (T5.1), then wiring.
7. **M6:** round-trip + schema tests first (T6.1, T6.2), then continuity + cross-engine.
8. **M7:** trajectory KPIs, then optimize (never optimize before T7.1 green).
9. **M8:** baseline parity, then structural exactness, then VPA demo.

At every step: if a statistical test fails but the exact micro-tests pass, the bug is in ordering/units/timescale, not in the algebra — check §1.2, §6 unit handling, and the decay-ordering choice (§1.2 note) before touching kernel math.

---

## 11. Non-goals / explicit scope guards

- Do **not** port config/curves/interventions/metrics to C++ — Python owns biology (§0.3).
- Do **not** chase bit-exact agreement with Brian2 — statistical equivalence only (§1.4, INVARIANT 6).
- Do **not** build the event-queue delay design (§3.3) unless M7 profiling demands it.
- Do **not** add T-channel/spindles/three-factor STDP — those are Platform ITERATION 7+, out of scope for the port (reach feature-parity with the *current* Brian2 engine first).
- Keep `--deterministic` and pre-generated-noise paths alive permanently — they are the test harness's backbone.
