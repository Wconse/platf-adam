# CUDA Engine I/O Contract

> Version: 1.0 (M6+). This document defines the data hand-off between the Python
> orchestrator and the C++ CUDA engine. Both sides must honour this contract.

## Directory layout

```
<episode_bundle>/           ← Python writes, engine reads
  network_config.json       ← topology, pop params, proj list (static across episodes)
  aff_config.json           ← afferent params (optional)
  pop_{i}_v.npy             ← initial conditions per population
  pop_{i}_ge.npy, pop_{i}_gi.npy
  pop_{i}_w.npy             ← E only
  proj_{j}_row_off.npy      ← outgoing CSR per projection
  proj_{j}_col_idx.npy
  proj_{j}_w_syn.npy        ← weights (nS), mutable per episode (STDP)
  proj_{j}_delay_steps.npy  ← integer steps, fixed
  proj_{j}_in_row_off.npy   ← incoming CSR for STDP projs
  proj_{j}_in_syn_idx.npy
  aff_rate_{NAME}.npy       ← [T_steps x n_channels] float32, Hz
  aff_proj_{k}_*.npy        ← afferent outgoing CSR
  ta_egaba_{region}.npy     ← [T_steps] E_GABA per region (mV)
  ta_lr_gate.npy            ← [T_steps] lr_gate scalar
  ta_mg.npy                 ← [T_steps] Mg concentration (mM)
  ta_nmda_ratio.npy         ← [T_steps] NMDA split ratio

<output_dir>/               ← engine writes, Python reads
  cuda_rates.npy            ← [N_total] mean firing rate (Hz) per neuron
  cuda_spikes.npy           ← [2 x n_spikes] float32: [neuron_global_id, t_ms]
  pop{i}_{var}.npy          ← final state (with --save-state)
  stdp_w_proj{j}.npy        ← final STDP weights (with --save-state)
  stdp_Apre_proj{j}.npy     ← final Apre traces (with --save-state)
  stdp_Apost_proj{j}.npy    ← final Apost traces
```

## Units (ALL files in neuronal units)

| Quantity | Unit | Notes |
|----------|------|-------|
| Voltage v, E_GABA, E_AMPA, E_NMDA | mV | NOT volts |
| Conductance g_ampa/nmda/gaba, w_syn | nS | NOT siemens |
| Current I_homeo | pA | NOT amps |
| Rate r_hat, r_tgt | Hz | |
| Time delays | ms or integer steps | delay_steps = round(ms/DT) |
| Mg | mM | Mg block formula: 1/(1+Mg*exp(-0.062*v_mV)) |

**Brian2-interop:** Brian2 stores SI (V, S, A). Convert when reading Brian2 checkpoints:
```python
v_mV = np.array(G.v[:]) / 1e-3        # V → mV
g_nS = np.array(G.g_ampa[:]) / 1e-9   # S → nS
I_pA = np.array(G.I_homeo[:]) / 1e-12 # A → pA
```

## Episode boundary policy

- OU conductances (`ge_bg`, `gi_bg`): carried over (continuity)
- Synaptic conductances (`g_ampa/nmda/gaba`): **dropped** (reset to 0)
  Rationale: tau_ampa=3ms, tau_gaba=10ms — they decay to ~0 within 10× tau = 100ms,
  which is negligible vs episode length (30–120 s). Dropping simplifies the boundary.
- Delay ring buffers: **dropped** (pending spikes lost)
  Rationale: max delay = 20ms; ring state is ≤ 200 steps of context.
  Flag `--carry-ring` enables carrying (future).
- STDP traces (Apre/Apost): **dropped** by default, carried with `--carry-stdp`
  (timescale tau_pre/post = 20ms << episode, so dropping is correct default)
- Homeostasis (r_hat, I_homeo): carried over (slow timescale tau_homeo=1866ms)
- Weights w_syn: carried over (STDP updates accumulate across episodes)

## Engine invocation

```bash
./sim_multiregion <bundle_dir> <output_dir> [--save-state] [--carry-stdp]
```

## Output format

- `cuda_rates.npy`: float32, shape `(N_total,)`, mean rate in Hz over full episode.
  Index = global neuron id (pop.global_offset + local_id).
- `cuda_spikes.npy`: float32, shape `(2, n_spikes)`.
  Row 0 = global neuron ids (as float). Row 1 = spike times in ms.
  Empty episode: shape `(2, 0)` or `(2, 2)` with zeros.

## Performance targets (M7)

- Scale 0.1 (14300 neurons, 40 projs): < 2× real-time at DT=0.1ms (1s sim in < 2s wall)
- Scale 1.0: < 20× real-time (profiling target, not hard requirement for M7)
- Memory: total VRAM < 4 GB at SCALE=0.1 including ring buffers
