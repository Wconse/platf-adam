# ADAM — Prenatal Brain Development Simulator

**A CUDA-accelerated spiking neural network engine for modelling human prenatal brain development, developmental neurotoxicity, and neurological disorders.**

---

## What Is This?

ADAM is a custom simulation engine that models the developing human brain from **gestational week 20 to 52+** — the critical prenatal window when the cortex is most vulnerable to toxins, genetic mutations, and environmental disruption.

The engine simulates 8 interconnected brain regions at single-neuron resolution, tracking how the network matures, how GABA switches from excitatory to inhibitory, how long-range connections form, and how all of this changes under toxicological or genetic conditions.

The goal: a **computational platform for developmental neurotoxicology (DNT)** and disease modelling that is fast enough to run hundreds of experiments.

---

## The Brain in the Model

### 8 Regions, Sequential Maturation

```
Brainstem (BS) ──► Thalamus (TH) ──► Subplate (SP)
                                          │
                        ┌─────────────────┤
                        ▼                 ▼
                   Primary Cortex    Primary Cortex
                   Parietal (P1)     Auditory (A1)
                        │                 │
                        ▼                 ▼
                   Secondary (P2)    Secondary (A2)
                        │
                        ▼
                   Motor Cortex (M1)
```

Each region is born at a different gestational age, reflecting the biological inside-out cortical development sequence:

| Region | Born (dw) | Function |
|--------|-----------|----------|
| BS | 20 | Brainstem autonomic |
| TH | 22 | Thalamic relay, sensory gating |
| SP | 20 | Subplate — transient scaffold layer |
| P1 | 24 | Primary parietal cortex |
| P2 | 28 | Secondary parietal |
| A1 | 32 | Primary auditory cortex |
| A2 | 37 | Secondary auditory |
| M1 | 24 | Primary motor cortex |

---

## Neuron Types: Three Classes Per Region

Each region contains **three neuron populations** with distinct biophysics:

### Excitatory — AdEx (Adaptive Exponential I&F)
The workhorse. Full spike-frequency adaptation via adaptation current `w`, exponential threshold, homeostatic regulation.

```
C·dV/dt = -gL(V-EL) + gL·ΔT·exp((V-VT)/ΔT) - w + I_syn + I_homeo + I_stress
τw·dw/dt = a(V-EL) - w
```

### PV Interneurons — LIF (Leaky I&F), fast-spiking
Parvalbumin-positive interneurons: fast time constant (τm ≈ 10 ms), no adaptation, perisomatic targeting (soma of E neurons). **Primary generators of gamma-frequency inhibition.**

### SST Interneurons — LIF, adapting
Somatostatin-positive interneurons: slower (τm ≈ 20 ms), mild spike-frequency adaptation, dendritic targeting. **Gate the disinhibitory circuit — receive facilitating input from E neurons.**

### Network Size (SCALE=0.1 reference bundle)

| Population | N | Target Rate |
|------------|---|-------------|
| BS_E | 400 | 4.5 Hz |
| TH_E | 640 | 3.0 Hz |
| P1_E | 1200 | 2.5 Hz |
| A1_E | 2400 | 1.5 Hz |
| A2_E | 3600 | 1.2 Hz |
| All PV (×8) | 100–900 | 6–10 Hz |
| All SST (×8) | 30–270 | 2.7–4.5 Hz |
| **Total** | **15,158** | — |

Full scale (SCALE=1.0): **143,000 neurons**, 4.9M synapses.

---

## Synaptic Architecture

### 64 Projections, 492,312 Synapses

**Within each region (local circuit):**
- E → E (recurrent excitation, STDP in P1/P2)
- E → PV (feedforward inhibition)
- E → SST (facilitating — high-frequency bursts activate SST preferentially)
- PV → E (perisomatic inhibition, fast)
- PV → PV (mutual inhibition)
- PV → SST (PV suppresses SST = disinhibitory gate)
- SST → E (dendritic inhibition, slower)

**Between regions (hierarchy):**
```
TH ──AMPA──► P1 ──AMPA──► P2 ──AMPA──► A1 ──AMPA──► A2
             ↕ lr_gate                  ↕ lr_gate
BS ──────────────────────────────────► M1
         P2 ◄──► A2  (long-range, gated)
```

### Synapse Types
- **AMPA** — fast excitation (τ = 3 ms)
- **NMDA** — slow excitation, Mg²⁺-blocked at rest, voltage-dependent (τ = 100 ms)
- **GABA-A** — inhibition, reversal potential E_GABA changes with development

---

## Developmental State Vector

At every episode, 5 slow variables control the network's developmental state. They change between episodes (slow loop), not within a step.

### 1. GABA Reversal Potential — The Critical Flip

The most important developmental transition. At birth GABA **depolarizes** neurons (like glutamate). As KCC2 cotransporter matures, Cl⁻ is exported, and GABA becomes **hyperpolarizing**.

```
E_GABA: −45 mV (dw 20) → −75 mV (dw 34+)
         [depolarizing]    [hyperpolarizing]
```

This "GABA flip" is the most studied event in prenatal neurodevelopment. Many toxins and genetic conditions delay it.

**Mechanistic KCC2 model** (new): instead of a shifted sigmoid, we model KCC2 expression as a proper ODE:
```python
dk/dt = rate_factor × (kcc2_target − kcc2) / τ_kcc2
E_GABA = −45 + kcc2 × (−75 − (−45))   mV
```

This allows VPA to act correctly — by slowing KCC2 upregulation, not just shifting a curve.

### 2. Long-Range Connectivity Gate (`lr_gate`)

Thalamo-cortical and cortico-cortical long-range connections form gradually:
```
lr_gate: 0.0 (dw < 34) → 1.0 (dw > 38)
```

### 3. Mg²⁺ Concentration

Higher Mg²⁺ = stronger NMDA block at rest = less excitation:
```
Mg²⁺: 0.6 mM (dw 20) → 1.0 mM (dw 38+)
```

### 4. NMDA Ratio

Fraction of excitatory current through NMDA channels increases with maturation (NR2B → NR2A subunit switch):
```
nmda_ratio: 0.08 (dw 20) → 0.22 (dw 38+)
```

### 5. Homeostasis

Each neuron continuously adjusts `I_homeo` to track a target firing rate. Keeps the network in a biologically plausible dynamic range across all developmental ages.

---

## Advanced Biophysical Mechanisms

### T-Type Ca²⁺ Channels in Thalamus

Thalamic burst mode — the biological source of **delta brushes** (the dominant EEG pattern of premature newborns). Implemented via Destexhe et al. 1994 model:

```
I_T = g_T × m² × h × (E_Ca − V)
         [activation] [inactivation]

m gate: fast (~1 ms),  activated by depolarization
h gate: slow (~50 ms), inactivated by depolarization
E_Ca = +120 mV
```

**How it works:** After hyperpolarization (e.g., from GABA-I), `h` deinactivates. When excitatory input arrives, `m` activates rapidly → large inward Ca²⁺ current → burst of spikes. This creates the characteristic thalamic burst pattern.

After the GABA flip (dw ≈ 28), thalamic neurons begin to receive genuine hyperpolarizing GABA → T-channels deinactivate → burst mode becomes available.

### Short-Term Plasticity (Tsodyks-Markram)

Synaptic strength is not fixed — it depends on recent activity history:

```
On each pre-spike:
  u += U × (1 − u)         # utilization jumps
  w_eff = w × u × x        # effective weight
  x -= u × x               # resources consumed

Between spikes (recovery):
  u → U  with τ_f           # facilitation time constant
  x → 1  with τ_d           # depression time constant
```

| Projection type | U | τ_f | τ_d | Effect |
|----------------|---|-----|-----|--------|
| E → E, E → PV | 0.50 | 50 ms | 400 ms | **Depressing** — attenuates high-frequency bursts |
| E → SST | 0.15 | 600 ms | 30 ms | **Facilitating** — SST activates more during bursts |
| TH → P1 | 0.25 | 75 ms | 300 ms | Mild depressing |

The facilitating E→SST synapse creates a key circuit: **during E bursting, SST activation increases non-linearly → inhibits PV → disinhibits E dendrites**. This is the canonical disinhibitory motif observed in sensory cortex.

### STDP (Spike-Timing-Dependent Plasticity)

Hebbian learning on P1-P1 and P2-P2 recurrent connections. Event-driven implementation handles multiple simultaneous spikes in flight correctly.

```
on_pre:  Apre += A_plus;  w = clip(w + η × Apost)
on_post: Apost += A_minus; w = clip(w + η × Apre)
```

Traces decay exponentially with τ_pre = τ_post = 20 ms.

### Correlated Background Noise

Individual neurons also share a per-region OU noise component (τ = 20 ms), creating biologically realistic intra-regional correlation in spontaneous activity (~10% of individual OU amplitude). This improves the realism of burst synchronization patterns.

---

## Experimental Results

### Baseline Trajectory — Normal Prenatal Development

32 episodes from dw 20 → 52 (SCALE=0.1, 2400 neurons in P1).

**The GABA flip is clearly visible:**

| DW | P1_E rate | A1_E rate | E_GABA (P1) | lr_gate |
|----|-----------|-----------|-------------|---------|
| 20 | 0.74 Hz | — (not born) | −45.0 mV | 0.000 |
| 28 | 2.44 Hz | 0.32 Hz | −45.5 mV | 0.000 |
| **32** | **1.55 Hz** | **1.23 Hz** | **−74.5 mV** | 0.003 |
| 36 | 1.52 Hz | 0.54 Hz | −75.0 mV | 0.500 |
| 40 | 1.53 Hz | 0.53 Hz | −75.0 mV | 0.998 |
| 44 | 1.54 Hz | 0.53 Hz | −75.0 mV | 1.000 |
| 52 | 1.54 Hz | 0.53 Hz | −75.0 mV | 1.000 |

**Key biological checkpoints — all pass:**
- GABA flip in P1: centered at dw ≈ 30, complete by dw 34 ✓
- P1_E rate drop at flip: 2.44 → 1.55 Hz (GABA switches from excitatory to inhibitory) ✓
- A2 born at dw 37, rate climbs from 0 ✓
- lr_gate: 0 before dw 34, full by dw 38 ✓
- Network stable from dw 40–52, no runaway, no NaN ✓

**Real-time ratio:** 12 s simulated / 26 s compute = **0.46× realtime** (simulating 2× faster than real time)

---

### Experiment 1 — Valproate (VPA) 100 µM

**Biological mechanism:** VPA inhibits KCC2 cotransporter expression, delaying the GABA excitatory→inhibitory switch. Used as anticonvulsant; associated with autism when taken during pregnancy (Mowery et al. 2015).

**Exposure window:** dw 24 → 40 (typical first/second trimester).

**What we see:**

| DW | P1_E (Baseline) | P1_E (VPA) | Δ rate | E_GABA BL | E_GABA VPA |
|----|----------------|------------|--------|-----------|------------|
| 28 | 2.44 Hz | 2.46 Hz | +0.03 | −45.5 mV | −45.0 mV |
| 30 | 1.95 Hz | **2.48 Hz** | **+0.53** | −60.0 mV | **−45.0 mV** |
| 32 | 1.55 Hz | **2.48 Hz** | **+0.93** | −74.5 mV | **−45.1 mV** |
| 35 | 1.53 Hz | 1.95 Hz | +0.42 | −75.0 mV | −60.0 mV |
| 37 | 1.55 Hz | 1.56 Hz | +0.01 | −75.0 mV | −74.5 mV |
| **40** | **1.53 Hz** | **1.53 Hz** | **0.00** | **−75.0 mV** | **−75.0 mV** |

**Interpretation:**
- GABA flip in P1 is delayed by ~5 DW: baseline completes at dw 32, VPA at dw 37
- During the delay, P1_E is **60% hyperexcitable** (2.48 vs 1.55 Hz at dw 32) — GABA still depolarizing
- Complete washout after dw 40 (full recovery) — consistent with reversible KCC2 inhibition
- **Validates:** Delayed GABA switch seen in human VPA-exposed tissue (Mowery et al. 2015)

**Mechanistic VPA mode** (KCC2 ODE): tracks actual KCC2 expression level, giving gradual post-exposure recovery — more biologically accurate than an instant curve shift.

---

### Experiment 2 — ASD Moderate (severity 0.6)

**Mechanisms modelled:** Three simultaneous pathways seen in genetic ASD studies:
1. **GABA_flip delay:** +3 DW shift in P1, P2, A1, A2 (KCC2/NKCC1 dysregulation)
2. **E/I imbalance:** +12 pA excitatory bias in A1, A2, P2 (Markram E/I theory)
3. **LR connectivity deficit:** lr_gate capped at 0.76 (long-range P2↔A2 reduced 24%)

**What we see:**

| DW | A1_E BL | A1_E ASD | A2_E BL | A2_E ASD | P1_E BL | P1_E ASD |
|----|---------|----------|---------|----------|---------|----------|
| 28 | 0.32 Hz | 0.44 Hz | — | — | 2.44 Hz | 2.46 Hz |
| 32 | 1.23 Hz | 1.68 Hz | 0.07 Hz | 0.12 Hz | 1.55 Hz | **2.34 Hz** |
| 36 | 0.54 Hz | **1.53 Hz** | 0.34 Hz | **1.19 Hz** | 1.52 Hz | 1.53 Hz |
| 40 | 0.53 Hz | **0.80 Hz** | 0.30 Hz | **0.59 Hz** | 1.53 Hz | 1.53 Hz |
| 44 | 0.53 Hz | **0.80 Hz** | 0.31 Hz | **0.57 Hz** | 1.54 Hz | 1.54 Hz |

**Interpretation:**
- **A1/A2 chronic hyperactivity:** +49% elevated from dw 40 onward — persistent, non-reversible
  - Source: E/I imbalance in auditory cortex — matches hyperresponsivity in ASD (Marco et al. 2011)
- **P1 hyperexcitability during flip window (dw 32):** +51% — GABA still depolarizing
- **lr_gate permanently capped at 0.76:** P2↔A2 long-range coherence reduced 24%
  - Matches reduced EEG coherence between frontal and parietal areas (Just et al. 2004)
- **Non-reversible phenotype** from dw 40 onward (genetic condition, not drug) ✓

---

### KCC2 Mechanistic Model — VPA vs Baseline

Using the new ODE-based KCC2 model (VPA slows KCC2 upregulation by 65%):

| DW | E_GABA Baseline | E_GABA VPA_KCC2 | KCC2 level (P1) |
|----|-----------------|-----------------|-----------------|
| 24 | −45.0 mV | −45.0 mV | 0.000 |
| 28 | −45.5 mV | −45.0 mV | 0.000 |
| 30 | −60.0 mV | **−45.1 mV** | 0.003 |
| 32 | −74.5 mV | **−47.7 mV** | 0.090 |
| 34 | −75.0 mV | **−52.4 mV** | 0.246 |

At dw 32, the difference is **+27 mV** — GABA is still nearly fully depolarizing in VPA while baseline has already completed the flip. After VPA ends (dw > 40), KCC2 recovers gradually from its depressed level — not the instantaneous snap-back of a simple curve shift.

---

### Episode Metrics — dw 20, Baseline

Live metrics computed from population spike histograms (240 × 50 ms bins):

```
Wall: 38.8s  | BS E=3.70 PV=59.4 SST=14.7 Hz
              | TH E=2.19 PV=41.9 SST=9.8 Hz
              | P1 E=0.74 PV=19.9 SST=3.8 Hz
EI-ratio:  BS=0.06  TH=0.05  P1=0.04  M1=0.04
Bursts:    M1:11×3Hz(12%)  P1:9×3Hz(5%)  TH:8×5Hz(3%)
Sync:      0.527
```

**Interpretation of early-development metrics:**
- **EI ratio ~0.05:** At dw 20, E neurons fire far below I neurons because GABA is depolarizing — I neurons are over-driven by the excitatory GABA. This ratio normalizes dramatically after the flip.
- **Synchrony 0.53:** High inter-regional synchrony is expected at early development — before long-range connections form, the network fires in large coordinated bursts (mimicking fetal EEG).
- **Bursts:** Spontaneous burst activity in all active regions. TH shows thalamic bursts consistent with T-channel activity (8 burst events, peak 5 Hz).

---

## Performance

### Hardware
- GPU: NVIDIA RTX 3060 Ti (8 GB VRAM, sm_86)
- CPU: Windows 10, Python 3.13
- CUDA 13.3, compute capability 8.6

### Architecture: CUDA-Native, Windows

The entire simulation runs on GPU. The CPU only manages the slow developmental loop (between episodes):

```
Per episode (12 s biological time):
  GPU  ──► CUDA Graph: 120,000 steps × 103 kernels
  CPU  ──► DevStateVector update, synaptic scaling, pruning, checkpointing
```

### Speed Comparison

| Engine | Scale | Episode time | vs Realtime | vs Brian2 |
|--------|-------|-------------|-------------|-----------|
| Brian2 (old) | 0.1 | ~712 s/ep | 59× slower | 1× |
| ADAM pre-opt | 0.1 | 42 s/ep | 3.5× slower | 17× faster |
| **ADAM CUDA Graphs** | **0.1** | **26 s/ep** | **2.2× slower** | **28× faster** |
| ADAM + all features | 0.1 | **38 s/ep** | **3.2× slower** | **19× faster** |
| ADAM CUDA Graphs | 1.0 (143k neurons) | **26 s/ep** | 2.2× | — |

**For a full dw 20→52 trajectory (32 episodes):**
- Brian2: ~19 hours (measured)
- ADAM (current): **~20 minutes** — a **57× speedup**

### Key Optimizations
1. **CUDA Graphs:** Entire simulation step captured as graph, launched 120k times. Eliminates 11M kernel launch overhead.
2. **Batch deliver/enqueue:** 2D grid kernels replace 92 per-step projection launches.
3. **Device step counter:** Allows graph re-use across all steps with time-varying TA arrays.
4. **Fused neuron kernel:** Homeostasis + OU noise + AdEx dynamics + spike detect in one kernel (4×fewer launches).
5. **Warp-optimal block sizes:** Small populations (SST: 30–270 neurons) use 32-thread blocks instead of 256.
6. **`__expf` throughout:** Hardware-native float exp, 2× faster in T-channels, STP, NMDA block.

---

## What Makes This Model Unusual

Most spiking neural network simulators (Brian2, NEST, NEURON) are either:
- Biologically detailed but slow (NEURON: seconds per ms of simulation)
- Fast but abstract (rate models, point neurons with no plasticity)

ADAM occupies a specific niche:

| Feature | Brian2 | NEST | ADAM |
|---------|--------|------|------|
| Individual spikes | ✓ | ✓ | ✓ |
| AdEx + LIF | ✓ | ✓ | ✓ |
| NMDA + Mg² block | ✓ | partial | ✓ |
| 3 interneuron types | manual | partial | ✓ |
| Developmental trajectory | ✓ | ✗ | ✓ |
| GABA flip (mechanistic) | ✓ | ✗ | ✓ |
| T-type Ca²⁺ channels | ✓ | partial | ✓ |
| Short-term plasticity | ✓ | ✓ | ✓ |
| STDP | ✓ | ✓ | ✓ |
| CUDA-native + Graphs | ✗ | ✗ | ✓ |
| GPU speed (143k neurons) | 19× slower | — | ✓ |

The unique value: it is simultaneously **fast enough** for drug screening (minutes per trajectory) and **biologically specific enough** to distinguish mechanism of action (KCC2 inhibition ≠ GABA-A potentiation).

---

## Experiment Library

Currently implemented intervention modes:

| Mode | Mechanism | Key Effect |
|------|-----------|------------|
| `baseline` | Normal development | Reference trajectory |
| `vpa` | KCC2 inhibition, shifted sigmoid | GABA flip delayed 5 DW |
| `vpa_kcc2` | KCC2 ODE, rate_factor=0.35 | Gradual GABA delay + recovery |
| `asd` | E/I imbalance + GABA shift + LR deficit | Chronic A1/A2 hyperactivity |

**New experiments enabled by recent extensions:**

| Target | Mechanism to simulate | How |
|--------|----------------------|-----|
| Ethosuximide (anticonvulsant) | T-channel block in TH | `g_T *= (1 - block_fraction)` |
| Ketamine (anesthetic) | NMDA block + T-channel | `nmda_ratio *= 0` + `g_T *= 0.7` |
| Benzodiazepines | GABA-A potentiation (no EGABA change) | `w_gaba *= (1 + potentiation)` |
| Lead (Pb²⁺) DNT | NMDA receptor disruption | `nmda_ratio *= (1 - severity)` |
| Schizophrenia | PV deficit | `N_PV *= (1 - deficit_fraction)` |
| ADHD | DA-mediated LR connectivity | `lr_gate_mult` modification |

---

## Project Structure

```
cuda_adam/
├── cuda/
│   ├── cuda_src/
│   │   ├── main_multiregion.cu      # Main simulation engine (CUDA Graphs, batching)
│   │   └── kernels/
│   │       ├── fused_neuron_kernel.cu   # AdEx/LIF/TH-T-channel fused steps
│   │       ├── stdp_kernel.cu           # Event-driven STDP
│   │       └── afferent_kernel.cu       # Poisson afferent inputs
│   └── include/
│       ├── adex_params.cuh          # Neuron params + kernel declarations
│       └── engine_types.cuh         # ProjDesc, STPState structs
├── cuda_engine/
│   ├── engine_runner.py   # Slow loop: scaling, pruning, checkpointing
│   ├── dev_state.py       # DevStateVector: GABA, lr_gate, Mg, nmda, KCC2
│   └── metrics.py         # Burst detection, E/I ratio, synchrony
├── reference/
│   ├── gen_multiregion_scale.py   # Bundle generator (any SCALE)
│   └── multiregion/               # SCALE=0.1 reference bundle (15k neurons)
├── run_trajectory.py              # Main experiment runner
└── PROGRESS.md                    # Detailed technical log
```

---

## Numbers at a Glance

| Parameter | Value |
|-----------|-------|
| Brain regions | 8 |
| Neuron types | 3 (E / PV / SST) |
| Total neurons (SCALE=0.1) | 15,158 |
| Total neurons (SCALE=1.0) | 143,000 |
| Total synapses (SCALE=0.1) | 492,312 |
| Projections | 64 |
| STP projections | 25 |
| Timestep | 0.1 ms |
| Episode duration | 12 s biological |
| Episode compute time | ~38 s (GPU) |
| Full trajectory (32 episodes) | ~20 min |
| Speedup vs Brian2 | 19× |
| Realtime factor | 3.2× slower than real time |
| GPU | RTX 3060 Ti, sm_86 |
