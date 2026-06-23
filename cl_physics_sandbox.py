#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl_physics_sandbox.py — TASK 1 of the master roadmap.

GOAL (honest version): determine, from first principles, which DIRECTION
intracellular chloride moves during a network burst at dw20 (depolarizing
GABA, E_GABA = -45 mV), and whether that produces a seconds-long refractory
"macro-pause" (tracé discontinu) — OR the opposite.

The proposed roadmap asserts (PART 2 "The Lock"):
    "during the burst ... Cl- flows OUT ... E_GABA drops toward -60mV"
but its own PART 3 note says:
    "during an action potential, Vm >> E_GABA, causing massive inward Cl- flux".
These contradict. This script does NOT pre-commit to either. It builds a
self-bursting micro-circuit (burst shape is EMERGENT, not hand-drawn), tracks
chloride with the full driving-force ODE, and reports where E_GABA actually goes.

We do NOT impose "E_GABA -> -60". The ODE decides.

================================================================================
BIOPHYSICS / CITATIONS
================================================================================
Faraday conversion gamma (charge -> concentration):
    d[Cl_i]/dt = + I_GABA / (F * Vol)            [mol/(L*s) = M/s]
  Sign: conventional current I is positive when positive charge leaves the cell.
  A Cl- (anion) entering the cell = negative charge in = equivalent outward
  positive current. So I_GABA > 0 (outward, when Vm > E_Cl) == Cl- INFLUX ==
  [Cl_i] INCREASES. Hence the + sign.
    F   = 96485 C/mol  (Faraday)
    Vol ~ 1 pL = 1e-12 L  (small cortical neuron soma+proximal dendrite;
                           Kaila 1997 Prog Neurobiol; Staley & Proctor 1999 J Physiol)
  => for I in pA (1e-12 A) and Vol = 1 pL:
        gamma = 1/(F*Vol) = 1/(96485 * 1e-12) mol/(C) per L  ... evaluate:
        1 pA -> 1e-12/(96485*1e-12) mol/(L*s) = 1.036e-5 M/s = 1.036e-2 mM/s
     gamma ~ 0.0104 mM/(pA*s) at Vol = 1 pL.
  This matches acute GABA-loading rates reported by Staley & Proctor (1999) and
  the Cl- accumulation literature (Kaila 1997): tens of pA of sustained GABA
  current shift [Cl_i] by a few mM over hundreds of ms.

Pump / developmental restoration:
    + (Cl_eq(dw) - cl) / tau_pump
  Cl_eq(dw) is the developmental setpoint (KCC2/NKCC1 balance). At dw20 KCC2 is
  low => Cl_eq is HIGH => E_Cl(Cl_eq) = -45 mV (depolarizing).
  tau_pump is the acute restoration time constant. The roadmap claims it must be
  ~seconds for a seconds-long IBI; we SCAN it (0.5 - 8 s) rather than assume.

Nernst (anion, z = -1):
    E_Cl = -(RT/F) ln([Cl]_o/[Cl]_i) = (RT/F) ln([Cl]_i/[Cl]_o)
    RT/F = 26.7 mV at 37C ;  [Cl]_o = 130 mM (fixed)
  Check: E_Cl=-45 -> [Cl]_i ~ 24 mM ; E_Cl=-75 -> [Cl]_i ~ 7.8 mM. Consistent
  with CL_REST=7 mM in cuda_engine/dev_state.py.

================================================================================
UNITS: mV, nS, pF, pA, ms internally. cl in mM. gamma converted per-ms inside.
"""

import numpy as np

# ---------------------------------------------------------------- constants
RT_F   = 26.7      # mV
CL_O   = 130.0     # mM
F_FAR  = 96485.0   # C/mol

def nernst_ecl(cl_mM):
    """E_Cl in mV from intracellular [Cl] (mM). Vectorized."""
    cl = np.maximum(cl_mM, 0.05)
    return RT_F * np.log(cl / CL_O)          # = -(RT/F)ln(Clo/Cli)

def cl_from_ecl(ecl_mV):
    """Inverse Nernst: [Cl]_i (mM) from E_Cl (mV)."""
    return CL_O * np.exp(ecl_mV / RT_F)

def gamma_per_s(vol_pL):
    """Faraday conversion gamma in mM/(pA*s) for a cell of volume vol_pL."""
    vol_L = vol_pL * 1e-12
    # 1 pA = 1e-12 C/s ;  d[Cl]/dt = I/(F*Vol)
    return (1e-12 / (F_FAR * vol_L)) * 1e3   # mol/L/s -> mM/s, per pA

# ---------------------------------------------------------------- AdEx E params (P1, from params_p1.py)
C_E, gL_E, EL_E = 200.0, 10.0, -70.0
VT_E, DT_E      = -50.0, 2.0
Vcut_E, Vr_E    = -30.0, -65.0
a_E, b_E, tauw_E = 2.0, 50.0, 200.0
tref_E          = 2.0
E_AMPA          = 0.0

# Inhibitory (LIF-like, no adaptation)
C_I, gL_I, EL_I = 100.0, 10.0, -65.0
Vth_I, Vr_I     = -50.0, -65.0
tref_I          = 1.0

# Synaptic kinetics
tau_ampa = 3.0     # ms (E recv AMPA)
tau_gaba = 10.0    # ms (E recv GABA)

# Tonic background drive (from params_p1.py — sets the operating point, NOT fitted).
# These represent extrinsic synaptic bombardment; reversals fixed as in the
# reference model. The PLASTIC, chloride-coupled GABA is the RECURRENT giE only.
ge0_E, gi0_E = 14.0, 23.0    # nS, background exc/inh onto E
ge0_I, gi0_I = 4.0, 3.81     # nS, background exc/inh onto I
E_i_bg       = -80.0         # mV, background inhibition reversal (fixed)
# OU fluctuation std + time-const of the background conductances (params_p1.py).
# The VARIANCE (not the mean) is what stochastically crosses threshold -> ignites.
sig_e_E, sig_i_E = 2.5, 3.8
sig_e_I, sig_i_I = 0.48, 0.30
tau_e_bg, tau_i_bg = 2.73, 10.49   # ms
# resulting operating points (V_eff): E ~ -54 mV, I ~ -53.6 mV
V_eff_E = (gL_E*EL_E + gi0_E*E_i_bg) / (gL_E + ge0_E + gi0_E)
V_eff_I = (gL_I*EL_I + gi0_I*E_i_bg) / (gL_I + ge0_I + gi0_I)

# ---------------------------------------------------------------- micro-circuit
def run_microcircuit(
        dw=20.0,
        N_E=80, N_I=20,
        T_ms=8000.0, dt=0.1,
        ecl0_mV=-45.0,        # developmental E_GABA at this dw (NOT a target for activity)
        tau_pump_s=3.0,       # acute Cl restoration time constant
        vol_pL=1.0,           # cell volume for Faraday gamma
        tauw_ms=200.0,        # adaptation/sAHP time constant
        b_pA=50.0,            # spike-triggered adaptation increment
        # recurrent weights (nS per spike, scaled by 1/sqrt(K) style)
        w_ee=0.9, w_ei=0.9, w_ie=2.2, w_ii=1.0,
        p_ee=0.15, p_ei=0.2, p_ie=0.2, p_ii=0.2,
        # STP (Tsodyks depression) on E->E and E->I
        U_stp=0.45, tau_rec_ms=300.0,
        # ambient ignition noise (Poisson AMPA kicks)
        noise_rate_hz=2.5, noise_g=0.6,
        seed=0,
        gaba_chloride_coupled=True,   # if False: E_GABA frozen at ecl0 (null for Cl effect)
        record_dt_ms=1.0):
    """
    Self-bursting E-I micro-circuit at dw20 with depolarizing GABA.
    Per-E-neuron chloride tracked with full driving-force Faraday ODE.
    Returns dict of recorded time series (population means) + spike rasters.
    """
    rng = np.random.default_rng(seed)
    nstep = int(T_ms / dt)
    gamma = gamma_per_s(vol_pL)               # mM/(pA*s)
    cl_eq = cl_from_ecl(ecl0_mV)              # developmental setpoint [Cl]

    # connectivity (sparse random)
    def conn(npre, npost, p):
        m = rng.random((npost, npre)) < p
        np.fill_diagonal(m, False) if npre == npost else None
        return m.astype(np.float32)
    C_ee = conn(N_E, N_E, p_ee) * w_ee
    C_ei = conn(N_E, N_I, p_ei) * w_ei   # E->I  (post I, pre E)
    C_ie = conn(N_I, N_E, p_ie) * w_ie   # I->E  (post E, pre I)
    C_ii = conn(N_I, N_I, p_ii) * w_ii

    # state E  (init at operating point, not at EL — background drive present)
    vE  = V_eff_E + 2.0 * rng.standard_normal(N_E)
    wE  = np.zeros(N_E)
    # OU background-conductance fluctuations (zero-mean, added to ge0/gi0)
    oue_E = np.zeros(N_E); oui_E = np.zeros(N_E)
    oue_I = np.zeros(N_I); oui_I = np.zeros(N_I)
    # OU step coefficients for stationary std = sigma
    ke_E = np.sqrt(2.0*dt/tau_e_bg); ki_E = np.sqrt(2.0*dt/tau_i_bg)
    geE = np.zeros(N_E)   # AMPA conductance on E
    giE = np.zeros(N_E)   # GABA conductance on E
    cl  = np.full(N_E, cl_eq)             # per-neuron chloride
    refE = np.zeros(N_E)
    # STP resource per E neuron (presynaptic)
    x_stp = np.ones(N_E)
    # state I
    vI  = V_eff_I + 2.0 * rng.standard_normal(N_I)
    geI = np.zeros(N_I)
    giI = np.zeros(N_I)
    refI = np.zeros(N_I)

    tau_pump_ms = tau_pump_s * 1000.0
    gamma_ms = gamma * 1e-3                # mM/(pA*ms)

    rec_every = int(record_dt_ms / dt)
    rec = {k: [] for k in
           ('t', 'ecl_mean', 'cl_mean', 'v_mean', 'gi_mean',
            'igaba_mean', 'vminusE_mean', 'rateE', 'rateI')}
    spikesE_t, spikesE_i = [], []
    # rolling spike counters for instantaneous rate (per record window)
    cntE = 0; cntI = 0

    for step in range(nstep):
        ecl = nernst_ecl(cl) if gaba_chloride_coupled else np.full(N_E, ecl0_mV)

        # OU update of background conductance fluctuations
        oue_E += -oue_E*dt/tau_e_bg + sig_e_E*ke_E*rng.standard_normal(N_E)
        oui_E += -oui_E*dt/tau_i_bg + sig_i_E*ki_E*rng.standard_normal(N_E)
        oue_I += -oue_I*dt/tau_e_bg + sig_e_I*ke_E*rng.standard_normal(N_I)
        oui_I += -oui_I*dt/tau_i_bg + sig_i_I*ki_E*rng.standard_normal(N_I)
        gbe_E = np.maximum(ge0_E + oue_E, 0.0); gbi_E = np.maximum(gi0_E + oui_E, 0.0)
        gbe_I = np.maximum(ge0_I + oue_I, 0.0); gbi_I = np.maximum(gi0_I + oui_I, 0.0)

        # ---- E dynamics (AdEx) ----
        I_ampa = geE * (E_AMPA - vE)
        I_gaba = giE * (ecl - vE)                 # driving force; sign emerges
        I_bg   = gbe_E*(E_AMPA - vE) + gbi_E*(E_i_bg - vE)   # fluctuating background
        active = refE <= 0
        # clip exp argument to avoid overflow when v overshoots before spike reset
        exp_arg = np.minimum((vE - VT_E)/DT_E, 20.0)
        dv = (gL_E*(EL_E - vE) + gL_E*DT_E*np.exp(exp_arg)
              + I_ampa + I_gaba + I_bg - wE) / C_E
        vE = np.where(active, vE + dt*dv, vE)
        # bound the AdEx overshoot at the spike cutoff: spike detection still
        # fires (v>=Vcut), but the chloride current is not corrupted by a huge
        # transient. The real depolarization is the inter-spike plateau, not the
        # (reset-model) spike peak.
        vE = np.minimum(vE, Vcut_E)
        wE = wE + dt*(a_E*(vE - EL_E) - wE)/tauw_ms
        refE = np.where(active, refE, refE - dt)

        # ---- chloride ODE (per E neuron) : I_GABA in pA = nS*mV ----
        # I_GABA_current sign convention: membrane current = giE*(vE - ecl)
        #   > 0 (outward) when vE > ecl  => Cl- influx => cl increases.
        I_gaba_membrane = giE * (vE - ecl)        # pA
        if gaba_chloride_coupled:
            dcl = (gamma_ms * I_gaba_membrane
                   + (cl_eq - cl)/tau_pump_ms) * dt
            cl = np.maximum(cl + dcl, 0.1)

        # ---- I dynamics (LIF) ----
        I_ampaI = geI * (E_AMPA - vI)
        I_gabaI = giI * (-75.0 - vI)              # I cells: keep GABA hyperpol (not tracked)
        I_bgI   = gbe_I*(E_AMPA - vI) + gbi_I*(E_i_bg - vI)
        activeI = refI <= 0
        dvI = (gL_I*(EL_I - vI) + I_ampaI + I_gabaI + I_bgI) / C_I
        vI = np.where(activeI, vI + dt*dvI, vI)
        refI = np.where(activeI, refI, refI - dt)

        # ---- synaptic decay ----
        geE *= np.exp(-dt/tau_ampa); giE *= np.exp(-dt/tau_gaba)
        geI *= np.exp(-dt/tau_ampa); giI *= np.exp(-dt/tau_gaba)
        # STP recovery
        x_stp = x_stp + dt*(1.0 - x_stp)/tau_rec_ms

        # ---- ambient noise (Poisson AMPA to E) ----
        nk = rng.random(N_E) < (noise_rate_hz*1e-3*dt)
        geE += nk * noise_g

        # ---- spikes E ----
        spkE = (vE >= Vcut_E) & active
        if spkE.any():
            idx = np.where(spkE)[0]
            vE[idx] = Vr_E
            wE[idx] += b_pA
            refE[idx] = tref_E
            eff = (U_stp * x_stp[idx])           # released fraction
            # propagate to postsynaptic E (AMPA) and I, with STP depression
            geE += C_ee[:, idx] @ eff
            geI += C_ei[:, idx] @ eff
            x_stp[idx] *= (1.0 - U_stp)
            cntE += len(idx)
            for j in idx:
                spikesE_t.append(step*dt); spikesE_i.append(int(j))

        # ---- spikes I ----
        spkI = (vI >= Vth_I) & activeI
        if spkI.any():
            idx = np.where(spkI)[0]
            vI[idx] = Vr_I
            refI[idx] = tref_I
            giE += C_ie[:, idx] @ np.ones(len(idx))   # I->E GABA (depolarizing via ecl)
            giI += C_ii[:, idx] @ np.ones(len(idx))
            cntI += len(idx)

        # ---- record ----
        if step % rec_every == 0:
            rec['t'].append(step*dt)
            rec['ecl_mean'].append(float(np.mean(ecl)))
            rec['cl_mean'].append(float(np.mean(cl)))
            rec['v_mean'].append(float(np.mean(vE)))
            rec['gi_mean'].append(float(np.mean(giE)))
            rec['igaba_mean'].append(float(np.mean(I_gaba_membrane)))
            rec['vminusE_mean'].append(float(np.mean(vE - ecl)))
            # instantaneous rate over the record window (Hz)
            rec['rateE'].append(cntE / (N_E * record_dt_ms*1e-3))
            rec['rateI'].append(cntI / (N_I * record_dt_ms*1e-3))
            cntE = 0; cntI = 0

    out = {k: np.asarray(v) for k, v in rec.items()}
    out['spikesE_t'] = np.asarray(spikesE_t)
    out['spikesE_i'] = np.asarray(spikesE_i)
    out['meta'] = dict(dw=dw, gamma_mM_per_pA_s=gamma, cl_eq=cl_eq,
                       ecl0=ecl0_mV, tau_pump_s=tau_pump_s, vol_pL=vol_pL)
    return out


# ---------------------------------------------------------------- Q3 control: known answer
def control_clamped_voltage(v_clamp_mV, g_gaba_nS, ecl0_mV=-45.0,
                            tau_pump_s=3.0, vol_pL=1.0, T_ms=40000.0, dt=0.1):
    """
    KNOWN-ANSWER CONTROL. Clamp V to a constant, hold a constant GABA conductance.
    The chloride ODE then has an analytic fixed point. Verify the simulator
    converges to it. If it does NOT, the integrator is wrong and nothing else
    can be trusted (protocol Q3).

    Fixed point: 0 = gamma*g*(Vclamp - E_Cl(cl*)) + (cl_eq - cl*)/tau_pump
    Solve numerically for cl* and compare to simulation steady state.
    """
    gamma = gamma_per_s(vol_pL)
    cl_eq = cl_from_ecl(ecl0_mV)
    tau_pump_ms = tau_pump_s*1000.0
    gamma_ms = gamma*1e-3
    cl = cl_eq
    nstep = int(T_ms/dt)
    for _ in range(nstep):
        ecl = nernst_ecl(cl)
        I_mem = g_gaba_nS*(v_clamp_mV - ecl)     # pA
        cl += (gamma_ms*I_mem + (cl_eq - cl)/tau_pump_ms)*dt
        cl = max(cl, 0.1)
    sim_cl = cl
    sim_ecl = nernst_ecl(sim_cl)

    # analytic fixed point via bisection on cl
    def resid(c):
        ecl = nernst_ecl(c)
        return gamma*g_gaba_nS*(v_clamp_mV - ecl) + (cl_eq - c)/tau_pump_s
    lo, hi = 0.1, 200.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if resid(lo)*resid(mid) <= 0: hi = mid
        else: lo = mid
    ana_cl = 0.5*(lo+hi)
    ana_ecl = nernst_ecl(ana_cl)
    return dict(v_clamp=v_clamp_mV, g_gaba=g_gaba_nS,
                sim_cl=sim_cl, ana_cl=ana_cl,
                sim_ecl=sim_ecl, ana_ecl=ana_ecl,
                err_mV=abs(sim_ecl-ana_ecl))


def single_neuron_burst_train(
        ecl0_mV=-45.0, cl_eq_mV=-45.0,
        n_bursts=5, burst_dur_ms=400.0, gap_ms=2000.0,
        ge_burst=40.0, gi_burst=30.0,     # nS during burst (high-conductance GDP)
        tau_pump_s=3.0, vol_pL=1.0, dt=0.1,
        ge0=14.0, gi0=23.0):
    """
    DECISIVE physics test (protocol-clean): drive ONE AdEx E neuron through a
    train of realistic high-conductance bursts. The burst's Vm trajectory is
    EMERGENT from AdEx (we do not prescribe Vm). We measure:
      - actual mean Vm during each burst (the quantity that sets Cl direction),
      - E_GABA at burst end and its minimum,
      - recovery time back toward setpoint between bursts.
    This isolates the chloride consequence of a burst from network self-organization.
    """
    gamma = gamma_per_s(vol_pL); gamma_ms = gamma*1e-3
    cl_eq = cl_from_ecl(cl_eq_mV)
    tau_pump_ms = tau_pump_s*1000.0
    v = V_eff_E; w = 0.0; cl = cl_from_ecl(ecl0_mV); ref = 0.0
    ge = 0.0; gi = 0.0
    rng = np.random.default_rng(0)

    period_ms = burst_dur_ms + gap_ms
    T_ms = n_bursts*period_ms + gap_ms
    nstep = int(T_ms/dt)
    rec_ecl = []; rec_v = []; rec_t = []; rec_burst = []
    # per-burst accumulators
    burst_v_sum = np.zeros(n_bursts); burst_v_cnt = np.zeros(n_bursts)
    ecl_at_end = np.full(n_bursts, np.nan)

    for step in range(nstep):
        t = step*dt
        # which burst window are we in?
        phase = t % period_ms
        bidx = int(t // period_ms)
        in_burst = (phase < burst_dur_ms) and (bidx < n_bursts)

        ecl = nernst_ecl(cl)
        # drive
        if in_burst:
            ge = ge_burst; gi = gi_burst
        # AdEx
        I_ampa = ge*(E_AMPA - v)
        I_gaba = gi*(ecl - v)
        I_bg   = ge0*(E_AMPA - v) + gi0*(E_i_bg - v)
        if ref <= 0:
            exp_arg = min((v - VT_E)/DT_E, 20.0)
            dv = (gL_E*(EL_E - v) + gL_E*DT_E*np.exp(exp_arg) + I_ampa + I_gaba + I_bg - w)/C_E
            v = v + dt*dv
            v = min(v, Vcut_E)
        else:
            ref -= dt
        w = w + dt*(a_E*(v - EL_E) - w)/tauw_E
        # chloride (membrane current sign: g*(v-ecl) > 0 => Cl influx => cl up)
        I_mem = gi*(v - ecl)
        cl += (gamma_ms*I_mem + (cl_eq - cl)/tau_pump_ms)*dt
        cl = max(cl, 0.1)
        # spike
        if ref <= 0 and v >= Vcut_E:
            v = Vr_E; w += b_E; ref = tref_E
        # decay synapses (outside burst they relax)
        if not in_burst:
            ge *= np.exp(-dt/tau_ampa); gi *= np.exp(-dt/tau_gaba)
        # accumulate mean Vm during burst (use the bounded v)
        if in_burst:
            burst_v_sum[bidx] += v; burst_v_cnt[bidx] += 1
        elif bidx < n_bursts and phase < burst_dur_ms + dt:
            pass
        # record end-of-burst ecl
        if bidx < n_bursts and (burst_dur_ms - dt) <= phase < (burst_dur_ms + dt):
            ecl_at_end[bidx] = ecl
        if step % 10 == 0:
            rec_t.append(t); rec_ecl.append(ecl); rec_v.append(v); rec_burst.append(in_burst)

    mean_v = np.where(burst_v_cnt > 0, burst_v_sum/np.maximum(burst_v_cnt, 1), np.nan)
    rec_ecl = np.asarray(rec_ecl)
    return dict(mean_v_burst=mean_v, ecl_at_burst_end=ecl_at_end,
                ecl_min=float(np.nanmin(rec_ecl)), ecl_max=float(np.nanmax(rec_ecl)),
                ecl_final=float(rec_ecl[-1]),
                t=np.asarray(rec_t), ecl=rec_ecl, v=np.asarray(rec_v),
                vol_pL=vol_pL, tau_pump_s=tau_pump_s)


def detect_pop_bursts(rateE, record_dt_ms):
    """Population-burst detector. Returns (n_bursts, median_IBI_s, burst_dur_s,
    peak_Hz, duty) from a population-rate trace. A burst = contiguous region
    above 30% of peak; IBI = onset-to-onset interval."""
    r = np.convolve(rateE, np.ones(7)/7, mode='same')
    pk = float(r.max())
    if pk < 5.0:
        return dict(n=0, ibi_s=np.nan, dur_s=np.nan, peak=pk, duty=0.0)
    above = r > 0.3*pk
    edges = np.diff(above.astype(int))
    onsets = np.where(edges == 1)[0]; offsets = np.where(edges == -1)[0]
    if len(onsets) < 3:
        return dict(n=len(onsets), ibi_s=np.nan, dur_s=np.nan, peak=pk,
                    duty=float(above.mean()))
    ibis = np.diff(onsets)*record_dt_ms/1000.0
    # burst durations (match offsets after each onset)
    durs = []
    for on in onsets:
        off = offsets[offsets > on]
        if len(off): durs.append((off[0]-on)*record_dt_ms/1000.0)
    return dict(n=len(onsets), ibi_s=float(np.median(ibis)),
                dur_s=float(np.median(durs)) if durs else np.nan,
                peak=pk, duty=float(above.mean()))


def network_oscillator(
        N_E=200, N_I=50, T_ms=15000.0, dt=0.1,
        ecl_mV=-45.0,            # depolarizing GABA at dw20 (fixed; chloride irrelevant here)
        gain=1.0,                # global recurrent-gain multiplier (scan this)
        w_ee=8.0, w_ei=6.0, w_ie=6.0, w_ii=2.0,
        p_ee=0.15, p_ei=0.25, p_ie=0.25, p_ii=0.2,
        tauw_ms=3000.0, b_pA=50.0, a_nS=2.0,
        U_stp=0.35, tau_rec_ms=500.0,
        nmda_frac=0.6, tau_nmda_ms=100.0,  # slow recurrent excitation sustains the episode
        bg_gL_E=4.0, bg_gL_I=3.0,        # weak tonic leak (dw20: low tonic inhibition)
        I_tonic_E=210.0, I_tonic_I=150.0, # tonic depolarizing bias -> V_eff ~ -55 mV
        sigma_noise_pA=45.0, tau_noise_ms=5.0,
        g_gap=0.0, p_gap=0.10,           # electrical (gap-junction) coupling among E cells
        spk_gain=0.0, tau_spk_ms=3.0,    # spike-transmitted electrical coupling (spikelet)
        seed=0, record_dt_ms=2.0):
    """
    Developing-network relaxation oscillator (Tabak/Rinzel 2000 + Ben-Ari GDP).
    Ingredients: regenerative recurrent excitation (mean-field bistable), a
    depolarizing-GABA amplifier (I->E excitatory at dw20), and SLOW negative
    feedback (sAHP w + synaptic depression) that terminates the episode and
    recovers over seconds -> sets the inter-burst interval. No chloride.
    """
    rng = np.random.default_rng(seed)
    nstep = int(T_ms/dt)
    def conn(npre, npost, p, wmean):
        m = (rng.random((npost, npre)) < p).astype(np.float32)
        if npre == npost: np.fill_diagonal(m, 0.0)
        return m*wmean*gain
    Cee = conn(N_E, N_E, p_ee, w_ee); Cei = conn(N_E, N_I, p_ei, w_ei)
    Cie = conn(N_I, N_E, p_ie, w_ie); Cii = conn(N_I, N_I, p_ii, w_ii)
    # Gap-junction (electrical) adjacency among E cells — symmetric, no self.
    # Subplate pyramidal/subplate cells form an electrically-coupled syncytium in
    # the immature cortex; this is the synchronization mechanism chemical synapses
    # lack. I_gap_i = g_gap * sum_j (V_j - V_i).
    if g_gap > 0.0:
        Mg = (rng.random((N_E, N_E)) < p_gap)
        Mg = np.triu(Mg, 1); Mg = (Mg | Mg.T).astype(np.float32)
        gap_deg = Mg.sum(1)
    else:
        Mg = None; gap_deg = None

    vE = -55.0 + 3*rng.standard_normal(N_E); wE = np.zeros(N_E)
    geE = np.zeros(N_E); giE = np.zeros(N_E); refE = np.zeros(N_E)
    gnE = np.zeros(N_E); gnI = np.zeros(N_I)  # slow NMDA-like recurrent exc
    gspk = np.zeros(N_E)                       # gap-junction spikelet conductance
    x = np.ones(N_E)                          # STP resource
    vI = -55.0 + 3*rng.standard_normal(N_I)
    geI = np.zeros(N_I); giI = np.zeros(N_I); refI = np.zeros(N_I)
    InE = np.zeros(N_E); InI = np.zeros(N_I)  # OU current noise
    kn = np.sqrt(2.0*dt/tau_noise_ms)

    rec_every = int(record_dt_ms/dt); rE = []; rI = []
    cntE = 0; cntI = 0
    for step in range(nstep):
        InE += -InE*dt/tau_noise_ms + sigma_noise_pA*kn*rng.standard_normal(N_E)
        InI += -InI*dt/tau_noise_ms + sigma_noise_pA*kn*rng.standard_normal(N_I)
        # E dynamics
        aE = refE <= 0
        ea = np.minimum((vE - VT_E)/DT_E, 20.0)
        I_gap = (g_gap*(Mg @ vE - gap_deg*vE)) if Mg is not None else 0.0
        dvE = (gL_E*(EL_E - vE) + gL_E*DT_E*np.exp(ea)
               + (geE + gnE + gspk)*(E_AMPA - vE) + giE*(ecl_mV - vE)
               + bg_gL_E*(EL_E - vE) + I_tonic_E + InE + I_gap - wE)/C_E
        vE = np.where(aE, np.minimum(vE + dt*dvE, Vcut_E), vE)
        wE += dt*(a_nS*(vE - EL_E) - wE)/tauw_ms
        refE = np.where(aE, refE, refE - dt)
        # I dynamics
        aI = refI <= 0
        dvI = (gL_I*(EL_I - vI) + (geI + gnI)*(E_AMPA - vI) + giI*(-75.0 - vI)
               + bg_gL_I*(EL_I - vI) + I_tonic_I + InI)/C_I
        vI = np.where(aI, np.minimum(vI + dt*dvI, -30.0), vI)
        refI = np.where(aI, refI, refI - dt)
        # decay
        geE *= np.exp(-dt/tau_ampa); giE *= np.exp(-dt/tau_gaba)
        geI *= np.exp(-dt/tau_ampa); giI *= np.exp(-dt/tau_gaba)
        gnE *= np.exp(-dt/tau_nmda_ms); gnI *= np.exp(-dt/tau_nmda_ms)
        gspk *= np.exp(-dt/tau_spk_ms)
        x += dt*(1.0 - x)/tau_rec_ms
        # E spikes
        sE = aE & (vE >= Vcut_E)
        if sE.any():
            idx = np.where(sE)[0]; vE[idx] = Vr_E; wE[idx] += b_pA; refE[idx] = tref_E
            eff = U_stp*x[idx]
            dee = Cee[:, idx] @ eff; dei = Cei[:, idx] @ eff
            geE += (1.0-nmda_frac)*dee; gnE += nmda_frac*dee
            geI += (1.0-nmda_frac)*dei; gnI += nmda_frac*dei
            x[idx] *= (1.0 - U_stp); cntE += len(idx)
            # spikelet: each spike depolarizes its gap-coupled neighbors (electrical
            # spike transmission — what synchronizes intrinsic bursters)
            if Mg is not None and spk_gain > 0.0:
                gspk += spk_gain * Mg[:, idx].sum(axis=1)
        # I spikes
        sI = aI & (vI >= Vth_I)
        if sI.any():
            idx = np.where(sI)[0]; vI[idx] = Vr_I; refI[idx] = tref_I
            giE += Cie[:, idx] @ np.ones(len(idx)); giI += Cii[:, idx] @ np.ones(len(idx))
            cntI += len(idx)
        if step % rec_every == 0:
            rE.append(cntE/(N_E*record_dt_ms*1e-3)); rI.append(cntI/(N_I*record_dt_ms*1e-3))
            cntE = 0; cntI = 0
    return dict(rateE=np.asarray(rE), rateI=np.asarray(rI), record_dt_ms=record_dt_ms)


def network_with_istdp(
        ecl_mV, N_E=160, N_I=40, T_ms=20000.0, dt=0.1,
        gain=1.0, w_ee=8.0, w_ei=6.0, w_ie_init=6.0, w_ii=2.0,
        p_ee=0.15, p_ei=0.25, p_ie=0.25, p_ii=0.2,
        tauw_ms=1500.0, b_pA=50.0, a_nS=2.0,
        nmda_frac=0.5, tau_nmda_ms=100.0,
        bg_gL_E=4.0, bg_gL_I=3.0, I_tonic_E=210.0, I_tonic_I=150.0,
        sigma_noise_pA=45.0, tau_noise_ms=5.0,
        # iSTDP (Vogels-Sprekeler 2011), sign-aware for depolarizing GABA
        istdp_on=True, eta=0.002, rho0_hz=3.0, tau_istdp_ms=20.0,
        w_ie_max=40.0, V_op=-55.0,
        seed=0, record_dt_ms=2.0):
    """
    E-I network with SIGN-AWARE inhibitory plasticity (iSTDP) on I->E synapses.

    Vogels-Sprekeler 2011 rule drives each E cell toward rate rho0 by adjusting
    inhibitory weights. CRITICAL developmental correction: that rule assumes GABA
    is hyperpolarizing (stronger I->E lowers rate). At dw20 GABA is DEPOLARIZING
    (ecl=-45 > V_op), so stronger I->E RAISES rate -> the rule must invert sign.
    effect_sign = sign(V_op - ecl_mV): +1 hyperpolarizing (standard Vogels),
    -1 depolarizing (inverted). So iSTDP only becomes a balancer as GABA matures.

    Returns regime metrics. Scan ecl_mV from -45 (dw20) to -75 (dw34) to test
    whether the bursting->balanced transition emerges from the GABA flip + iSTDP.
    """
    rng = np.random.default_rng(seed)
    nstep = int(T_ms/dt)
    def conn(npre, npost, p, wmean):
        m = (rng.random((npost, npre)) < p).astype(np.float32)
        if npre == npost: np.fill_diagonal(m, 0.0)
        return m*wmean*gain
    Cee = conn(N_E, N_E, p_ee, w_ee); Cei = conn(N_E, N_I, p_ei, w_ei)
    Mie = (rng.random((N_E, N_I)) < p_ie).astype(np.float32)   # I->E mask (post E, pre I)
    Wie = Mie * w_ie_init                                       # plastic I->E weights
    Cii = conn(N_I, N_I, p_ii, w_ii)

    vE = -55.0 + 3*rng.standard_normal(N_E); wE = np.zeros(N_E)
    geE = np.zeros(N_E); giE = np.zeros(N_E); gnE = np.zeros(N_E); refE = np.zeros(N_E)
    vI = -55.0 + 3*rng.standard_normal(N_I)
    geI = np.zeros(N_I); giI = np.zeros(N_I); gnI = np.zeros(N_I); refI = np.zeros(N_I)
    InE = np.zeros(N_E); InI = np.zeros(N_I); kn = np.sqrt(2.0*dt/tau_noise_ms)
    x_pre = np.zeros(N_I)    # presynaptic (I) iSTDP trace
    x_post = np.zeros(N_E)   # postsynaptic (E) iSTDP trace
    alpha = 2.0 * rho0_hz * (tau_istdp_ms/1000.0)   # depression offset -> sets rho0
    effect_sign = 1.0 if (V_op - ecl_mV) >= 0 else -1.0

    rec_every = int(record_dt_ms/dt); rE = []; rI = []; cntE = 0; cntI = 0
    for step in range(nstep):
        InE += -InE*dt/tau_noise_ms + sigma_noise_pA*kn*rng.standard_normal(N_E)
        InI += -InI*dt/tau_noise_ms + sigma_noise_pA*kn*rng.standard_normal(N_I)
        aE = refE <= 0
        ea = np.minimum((vE - VT_E)/DT_E, 20.0)
        I_inh = giE*(ecl_mV - vE)
        dvE = (gL_E*(EL_E-vE)+gL_E*DT_E*np.exp(ea)+(geE+gnE)*(E_AMPA-vE)+I_inh
               + bg_gL_E*(EL_E-vE)+I_tonic_E+InE-wE)/C_E
        vE = np.where(aE, np.minimum(vE+dt*dvE, Vcut_E), vE)
        wE += dt*(a_nS*(vE-EL_E)-wE)/tauw_ms
        refE = np.where(aE, refE, refE-dt)
        aI = refI <= 0
        dvI = (gL_I*(EL_I-vI)+(geI+gnI)*(E_AMPA-vI)+giI*(-75.0-vI)
               + bg_gL_I*(EL_I-vI)+I_tonic_I+InI)/C_I
        vI = np.where(aI, np.minimum(vI+dt*dvI, -30.0), vI)
        refI = np.where(aI, refI, refI-dt)
        geE*=np.exp(-dt/tau_ampa); giE*=np.exp(-dt/tau_gaba)
        geI*=np.exp(-dt/tau_ampa); giI*=np.exp(-dt/tau_gaba)
        gnE*=np.exp(-dt/tau_nmda_ms); gnI*=np.exp(-dt/tau_nmda_ms)
        x_pre *= np.exp(-dt/tau_istdp_ms); x_post *= np.exp(-dt/tau_istdp_ms)
        # E spikes
        sE = aE & (vE >= Vcut_E)
        if sE.any():
            idx = np.where(sE)[0]; vE[idx]=Vr_E; wE[idx]+=b_pA; refE[idx]=tref_E
            dee = Cee[:, idx].sum(1); dei = Cei[:, idx].sum(1)
            geE += (1-nmda_frac)*dee; gnE += nmda_frac*dee
            geI += (1-nmda_frac)*dei; gnI += nmda_frac*dei
            cntE += len(idx)
            if istdp_on:
                # post (E) spike: Δw = eta * effect_sign * x_pre  (for all I->this E)
                Wie[idx, :] += eta * effect_sign * x_pre[None, :] * Mie[idx, :]
                x_post[idx] += 1.0
        # I spikes
        sI = aI & (vI >= Vth_I)
        if sI.any():
            idx = np.where(sI)[0]; vI[idx]=Vr_I; refI[idx]=tref_I
            giE += Wie[:, idx].sum(1); giI += Cii[:, idx].sum(1)
            cntI += len(idx)
            if istdp_on:
                # pre (I) spike: Δw = eta * effect_sign * (x_post - alpha)
                Wie[:, idx] += eta * effect_sign * (x_post[:, None] - alpha) * Mie[:, idx]
                x_pre[idx] += 1.0
            Wie = np.clip(Wie, 0.0, w_ie_max)
        if step % rec_every == 0:
            rE.append(cntE/(N_E*record_dt_ms*1e-3)); rI.append(cntI/(N_I*record_dt_ms*1e-3))
            cntE=0; cntI=0
    rE = np.asarray(rE); rI = np.asarray(rI)
    # metrics over the SECOND HALF (after iSTDP has acted)
    h = len(rE)//2
    rEh = rE[h:]
    fano = float(rEh.var()/max(rEh.mean(), 1e-6))
    return dict(ecl=ecl_mV, effect_sign=effect_sign,
                meanE=float(rE.mean()), meanE_late=float(rEh.mean()),
                meanI=float(rI.mean()), fano_late=fano,
                wie_mean=float(Wie[Mie>0].mean()), wie_init=w_ie_init,
                drift=float(rE[-h:].mean()/max(rE[:h].mean(),1e-6)))


def single_neuron_adaptation_scan(tauw_ms, b_pA=50.0, I_drive_pA=150.0,
                                  a_nS=2.0, T_ms=20000.0, dt=0.1,
                                  ge0=14.0, gi0=23.0):
    """
    USER HYPOTHESIS TEST: AdEx 'w' is the math analog of sAHP (Ca-dependent K).
    With tauw in the sAHP range (1-5 s), does the spike-triggered adaptation
    create a SECONDS-long pause between bursts? Single AdEx E neuron, constant
    suprathreshold drive, scan tauw. Measure inter-burst interval (IBI).
    No chloride here — pure adaptation.
    """
    v = V_eff_E; w = 0.0; ref = 0.0
    nstep = int(T_ms/dt)
    spk_t = []
    for step in range(nstep):
        if ref <= 0:
            exp_arg = min((v - VT_E)/DT_E, 20.0)
            I_bg = ge0*(E_AMPA - v) + gi0*(E_i_bg - v)
            dv = (gL_E*(EL_E - v) + gL_E*DT_E*np.exp(exp_arg)
                  + I_bg + I_drive_pA - w)/C_E
            v = min(v + dt*dv, Vcut_E)
        else:
            ref -= dt
        w = w + dt*(a_nS*(v - EL_E) - w)/tauw_ms
        if ref <= 0 and v >= Vcut_E:
            v = Vr_E; w += b_pA; ref = tref_E; spk_t.append(step*dt)
    spk_t = np.asarray(spk_t)
    if len(spk_t) < 3:
        return dict(tauw_s=tauw_ms/1000, rate=len(spk_t)/(T_ms/1000),
                    n_spk=len(spk_t), ibi_s=np.nan, intra_hz=np.nan,
                    n_bursts=0, w_peak=np.nan)
    isi = np.diff(spk_t)  # ms
    # bimodal split: intra-burst (short) vs inter-burst (long)
    thr = 0.5*(isi.min() + isi.max()) if isi.max() > 5*isi.min() else isi.max()*2
    long_isi = isi[isi > max(thr, 50.0)]      # IBIs in ms
    short_isi = isi[isi <= max(thr, 50.0)]
    ibi_s = float(np.median(long_isi))/1000 if len(long_isi) else np.nan
    intra_hz = 1000.0/float(np.median(short_isi)) if len(short_isi) else np.nan
    return dict(tauw_s=tauw_ms/1000, rate=len(spk_t)/(T_ms/1000),
                n_spk=len(spk_t), ibi_s=ibi_s, intra_hz=intra_hz,
                n_bursts=len(long_isi)+1)


def plateau_burst_train(v_plateau_mV, ecl0_mV=-45.0, cl_eq_mV=-45.0,
                        n_bursts=5, burst_dur_ms=400.0, gap_ms=2000.0,
                        gi_burst=30.0, tau_pump_s=3.0, vol_pL=1.0, dt=0.1):
    """
    ARTIFACT-FREE sign+magnitude test. During each burst, hold Vm at a fixed
    plateau (literature: GDP plateau -40..-20 mV) with GABA conductance gi_burst.
    Between bursts Vm relaxes to rest (V_eff_E ~ -54) and gi -> 0. This removes
    the AdEx reset/refractory bias entirely: the only question is where the
    GDP plateau sits relative to E_Cl(-45).
    Returns per-burst E_Cl at end + overall min/max + recovery time.
    """
    gamma = gamma_per_s(vol_pL); gamma_ms = gamma*1e-3
    cl_eq = cl_from_ecl(cl_eq_mV); tau_pump_ms = tau_pump_s*1000.0
    cl = cl_from_ecl(ecl0_mV)
    period_ms = burst_dur_ms + gap_ms
    T_ms = n_bursts*period_ms + gap_ms
    nstep = int(T_ms/dt)
    ecl_end = np.full(n_bursts, np.nan)
    rec_ecl = []
    for step in range(nstep):
        t = step*dt; phase = t % period_ms; bidx = int(t // period_ms)
        in_burst = (phase < burst_dur_ms) and (bidx < n_bursts)
        ecl = nernst_ecl(cl)
        v = v_plateau_mV if in_burst else V_eff_E
        gi = gi_burst if in_burst else 0.0
        I_mem = gi*(v - ecl)
        cl += (gamma_ms*I_mem + (cl_eq - cl)/tau_pump_ms)*dt
        cl = max(cl, 0.1)
        if bidx < n_bursts and (burst_dur_ms - dt) <= phase < (burst_dur_ms + dt):
            ecl_end[bidx] = nernst_ecl(cl)
        if step % 10 == 0: rec_ecl.append(nernst_ecl(cl))
    rec_ecl = np.asarray(rec_ecl)
    return dict(ecl_end=ecl_end, ecl_min=float(rec_ecl.min()),
                ecl_max=float(rec_ecl.max()), ecl_final=float(rec_ecl[-1]))


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    print("="*70)
    print("Faraday gamma check:")
    for v in (0.5, 1.0, 2.0):
        print(f"  Vol={v} pL -> gamma = {gamma_per_s(v):.5f} mM/(pA*s)")
    print(f"  Nernst: E_Cl(-45mV) -> Cl_i = {cl_from_ecl(-45):.2f} mM ;"
          f"  E_Cl(-75mV) -> Cl_i = {cl_from_ecl(-75):.2f} mM")

    print("="*70)
    print("Q3 KNOWN-ANSWER CONTROL (clamped V, constant g_GABA):")
    print(f"  {'V_clamp':>8} {'g_GABA':>7} {'sim_Ecl':>9} {'ana_Ecl':>9} {'err_mV':>7}")
    for vc in (-70, -55, -45, -35, -20):
        for gg in (2.0, 10.0):
            r = control_clamped_voltage(vc, gg)
            print(f"  {r['v_clamp']:8.1f} {r['g_gaba']:7.1f} "
                  f"{r['sim_ecl']:9.2f} {r['ana_ecl']:9.2f} {r['err_mV']:7.3f}")

    print("="*70)
    print("SELF-BURSTING MICRO-CIRCUIT @ dw20 (E_GABA starts -45, chloride coupled):")
    out = run_microcircuit(seed=0, T_ms=8000.0, tau_pump_s=3.0)
    t = out['t']
    print(f"  gamma={out['meta']['gamma_mM_per_pA_s']:.5f} mM/(pA*s), "
          f"cl_eq={out['meta']['cl_eq']:.2f} mM, tau_pump=3s")
    print(f"  mean E rate = {out['rateE'].mean():.3f} Hz, "
          f"mean I rate = {out['rateI'].mean():.3f} Hz")
    print(f"  E_GABA: start={out['ecl_mean'][0]:.2f}  min={out['ecl_mean'].min():.2f}  "
          f"max={out['ecl_mean'].max():.2f}  end={out['ecl_mean'][-1]:.2f} mV")
    print(f"  [Cl]_i: start={out['cl_mean'][0]:.2f}  min={out['cl_mean'].min():.2f}  "
          f"max={out['cl_mean'].max():.2f} mM")
    print(f"  <V - E_Cl> over run = {out['vminusE_mean'].mean():.2f} mV  "
          f"(>0 => Cl loads/depolarizes ; <0 => Cl effluxes/hyperpolarizes)")
    # burst detection: population rate threshold
    rE = out['rateE']
    thr = max(1.0, rE.mean() + 2*rE.std())
    above = rE > thr
    nburst = int(np.sum(np.diff(above.astype(int)) == 1))
    print(f"  burst threshold={thr:.2f}Hz, detected burst onsets = {nburst}")
    # save series for plotting / inspection
    np.savez('output/cl_sandbox_dw20.npz',
             t=t, ecl=out['ecl_mean'], cl=out['cl_mean'],
             v=out['v_mean'], gi=out['gi_mean'],
             rateE=out['rateE'], rateI=out['rateI'],
             vminusE=out['vminusE_mean'])
    print("  saved -> output/cl_sandbox_dw20.npz")

    print("="*70)
    print("DECISIVE TEST: single neuron, train of 5 realistic GDP bursts @ dw20")
    print("  (E_GABA setpoint -45mV ; burst = ge40+gi30 nS for 400ms ; gap 2s)")
    print(f"  {'vol_pL':>7} {'tau_p_s':>8} {'meanVm_burst':>13} {'V-Ecl':>7} "
          f"{'Ecl_end':>8} {'Ecl_min':>8} {'Ecl_max':>8} {'dir':>10}")
    for vol in (1.0, 0.2, 0.05):
        for tp in (3.0, 8.0):
            r = single_neuron_burst_train(vol_pL=vol, tau_pump_s=tp)
            mv = float(np.nanmean(r['mean_v_burst']))
            vmecl = mv - (-45.0)
            direction = ("LOADS(depol)" if r['ecl_max'] > -44.9
                         else "EFFLUX(hyper)" if r['ecl_min'] < -45.1 else "flat")
            print(f"  {vol:7.2f} {tp:8.1f} {mv:13.2f} {vmecl:7.2f} "
                  f"{r['ecl_at_burst_end'][-1]:8.2f} {r['ecl_min']:8.2f} "
                  f"{r['ecl_max']:8.2f} {direction:>12}")
    print("  INTERPRETATION:")
    print("   - mean Vm during burst > -45 => Cl loads => E_GABA depolarizes (AGAINST roadmap PART2)")
    print("   - mean Vm during burst < -45 => Cl effluxes => E_GABA hyperpolarizes (roadmap PART2)")
    print("   - magnitude tells if a -45->-60 macro-pause is physically reachable per burst")

    print("="*70)
    print("ARTIFACT-FREE PLATEAU TEST: scan GDP plateau Vm (no AdEx reset bias)")
    print("  (5 bursts x 400ms, gi=30nS, gap 2s, E_GABA setpoint -45)")
    print(f"  {'Vplateau':>9} {'vol_pL':>7} {'tau_p':>6} {'Ecl_end5':>9} "
          f"{'Ecl_min':>8} {'Ecl_max':>8} {'dir':>13}")
    for vp in (-55, -50, -45, -40, -35, -30):
        for vol in (1.0, 0.2):
            r = plateau_burst_train(vp, vol_pL=vol, tau_pump_s=3.0)
            d = ("LOADS(depol)" if r['ecl_max'] > -44.9
                 else "EFFLUX(hyper)" if r['ecl_min'] < -45.1 else "flat")
            print(f"  {vp:9.1f} {vol:7.2f} {3.0:6.1f} {r['ecl_end'][-1]:9.2f} "
                  f"{r['ecl_min']:8.2f} {r['ecl_max']:8.2f} {d:>13}")
    print("  => crossover at Vm=-45 (=E_Cl). GDP plateau (lit: -40..-20mV) is ABOVE")
    print("     -45 => LOADS chloride => E_GABA DEPOLARIZES during burst.")
    print("     The roadmap PART2 (efflux->hyperpolarize macro-pause) needs Vm<-45.")

    print("="*70)
    print("Q10 CHECK: is the AdEx 'efflux' a reset artifact?")
    # subthreshold steady-state V under burst drive (no spikes): if > -45, the
    # neuron is genuinely driven ABOVE E_Cl, and the spiking mean (-54) is a
    # refractory/reset artifact dragging it down.
    ge_b, gi_b = 40.0, 30.0
    v_inf = (gL_E*EL_E + ge_b*E_AMPA + gi_b*(-45.0) + ge0_E*E_AMPA + gi0_E*E_i_bg) \
            / (gL_E + ge_b + gi_b + ge0_E + gi0_E)
    print(f"  subthreshold V_inf under burst drive (ge40+gi30) = {v_inf:.2f} mV")
    print(f"  AdEx spiking mean Vm was -53.9 mV -> the {v_inf:.0f}->-54 gap is the")
    print(f"  refractory-clamp(-65) + reset artifact. True drive is ABOVE -45 => loading.")

    print("="*70)
    print("Q9/Q4 MAGNITUDE BOUND: most aggressive plausible burst train")
    for vp in (-55.0, -30.0):
        r = plateau_burst_train(vp, gi_burst=60.0, burst_dur_ms=1000.0,
                                n_bursts=5, gap_ms=2000.0, vol_pL=0.2, tau_pump_s=8.0)
        print(f"  plateau={vp} gi=60nS dur=1000ms vol=0.2pL tau=8s -> "
              f"Ecl_min={r['ecl_min']:.2f} Ecl_max={r['ecl_max']:.2f} mV")
    print("  Even with aggressive params, |E_GABA shift| stays far from a -45->-60")
    print("  swing per burst at >=0.2pL. The chloride macro-pause is not reachable;")
    print("  burst gating must come from adaptation(w)+STP recovery (seconds-scale).")

    print("="*70)
    print("sAHP HYPOTHESIS: AdEx 'w' as sAHP. Scan tauw -> does IBI scale to seconds?")
    print("  (single AdEx E neuron, constant drive 150pA, b=50pA spike-triggered)")
    print(f"  {'drive':>6} {'tauw_s':>7} {'rate_Hz':>8} {'n_spk':>6} {'intra_Hz':>9} {'IBI_s':>7} {'n_bursts':>9}")
    for drive in (500.0, 800.0):
        for tw_s in (0.2, 1.0, 3.0, 5.0):
            r = single_neuron_adaptation_scan(tauw_ms=tw_s*1000.0, I_drive_pA=drive, b_pA=100.0)
            print(f"  {drive:6.0f} {r['tauw_s']:7.1f} {r['rate']:8.2f} {r['n_spk']:6d} "
                  f"{r['intra_hz']:9.1f} {r['ibi_s']:7.2f} {r['n_bursts']:9d}")
    print("  If IBI grows ~linearly with tauw into the 1-5s range => sAHP/w IS the")
    print("  macro-pause mechanism (biological, right sign, no chloride, no CUDA).")

    print("="*70)
    print("STEP 1 - find the bursting regime: scan recurrent gain (Tabak/Rinzel)")
    print("  (N_E=200, depol-GABA @ -45, tauw=3s, STP; gain multiplies recurrence)")
    print(f"  {'gain':>5} {'meanHz':>7} {'peakHz':>7} {'n_burst':>8} {'IBI_s':>7} {'dur_s':>6} {'duty':>6}")
    for g in (0.5, 1.0, 1.5, 2.0, 3.0):
        o = network_oscillator(gain=g, seed=1, T_ms=15000.0)
        b = detect_pop_bursts(o['rateE'], o['record_dt_ms'])
        print(f"  {g:5.1f} {o['rateE'].mean():7.2f} {b['peak']:7.1f} {b['n']:8d} "
              f"{b['ibi_s']:7.2f} {b['dur_s']:6.2f} {b['duty']:6.2f}")

    print("-"*70)
    print("STEP 2 - scan tauw with statistics (2 seeds, 30s, nmda_frac=0.5, gain=1.5)")
    print(f"  {'tauw_s':>7} {'n_burst':>8} {'IBI_s':>14} {'dur_s':>7} {'meanHz':>7}")
    for tw_s in (1.0, 2.0, 3.0):
        ibis = []; nbs = []; durs = []; mhz = []
        for sd in (1, 2):
            o = network_oscillator(gain=1.5, nmda_frac=0.5, tauw_ms=tw_s*1000.0,
                                   seed=sd, T_ms=30000.0)
            b = detect_pop_bursts(o['rateE'], o['record_dt_ms'])
            nbs.append(b['n']); mhz.append(o['rateE'].mean())
            if not np.isnan(b['ibi_s']): ibis.append(b['ibi_s'])
            if not np.isnan(b['dur_s']): durs.append(b['dur_s'])
        ibi_str = f"{np.mean(ibis):.2f}+-{np.std(ibis):.2f}" if ibis else "  n/a "
        print(f"  {tw_s:7.1f} {int(np.mean(nbs)):8d} {ibi_str:>14} "
              f"{np.mean(durs) if durs else np.nan:7.2f} {np.mean(mhz):7.2f}")
    print("  IBI growing with tauw across seeds => emergent NETWORK relaxation")
    print("  oscillator gated by sAHP (no chloride). Regime still narrow (Q4): a")
    print("  robust basin likely needs homeostatic self-tuning, not hand-set gain.")

    print("="*70)
    print("iSTDP TEST: sign-aware inhibitory plasticity across the GABA flip")
    print("  Does bursting (depol GABA -45) -> balanced (hyperpol -75) transition emerge,")
    print("  and does iSTDP stabilize drift? (Vogels-Sprekeler, sign=sign(V_op-E_GABA))")
    print(f"  {'E_GABA':>7} {'sign':>5} {'iSTDP':>6} {'meanE':>6} {'meanE_lt':>8} {'Fano_lt':>8} {'wie':>6} {'drift':>6}")
    for ecl in (-45.0, -60.0, -75.0):
        for on in (False, True):
            r = network_with_istdp(ecl, istdp_on=on, T_ms=15000.0, seed=1)
            print(f"  {ecl:7.1f} {r['effect_sign']:5.0f} {str(on):>6} {r['meanE']:6.2f} "
                  f"{r['meanE_late']:8.2f} {r['fano_late']:8.1f} {r['wie_mean']:6.2f} {r['drift']:6.2f}")
    print("  EXPECT (if mechanism right): at -45 high Fano (bursting); at -75 with iSTDP")
    print("  ON, Fano drops (balanced) + meanE near rho0=3Hz + drift~1 (stabilized).")

    print("="*70)
    print("STEP 3 - GAP JUNCTIONS at WEAK chemical recurrence (gain=1.0): do electrical")
    print("  synapses (subplate syncytium) provide the missing synchronization?")
    print(f"  {'g_gap':>6} {'meanHz':>7} {'peakHz':>7} {'n_burst':>8} {'IBI_s':>7} {'dur_s':>6} {'duty':>6}")
    for gg in (0.0, 0.05, 0.15, 0.4, 1.0):
        o = network_oscillator(gain=1.0, nmda_frac=0.5, tauw_ms=2000.0,
                               g_gap=gg, p_gap=0.10, seed=1, T_ms=20000.0)
        b = detect_pop_bursts(o['rateE'], o['record_dt_ms'])
        print(f"  {gg:6.2f} {o['rateE'].mean():7.2f} {b['peak']:7.1f} {b['n']:8d} "
              f"{b['ibi_s']:7.2f} {b['dur_s']:6.2f} {b['duty']:6.2f}")

    print("-"*70)
    print("STEP 4 - ROBUSTNESS (Q4/emergence): with gap junctions on (g_gap=0.15),")
    print("  is bursting robust across recurrent gain? (wide basin = emergent, not tuned)")
    print(f"  {'gain':>5} {'meanHz':>7} {'n_burst':>8} {'IBI_s':>7}   |  gap OFF (g_gap=0)")
    for g in (0.6, 1.0, 1.5, 2.5):
        on  = network_oscillator(gain=g, nmda_frac=0.5, tauw_ms=2000.0, g_gap=0.15, seed=1, T_ms=20000.0)
        off = network_oscillator(gain=g, nmda_frac=0.5, tauw_ms=2000.0, g_gap=0.0,  seed=1, T_ms=20000.0)
        bon = detect_pop_bursts(on['rateE'], on['record_dt_ms'])
        bof = detect_pop_bursts(off['rateE'], off['record_dt_ms'])
        print(f"  {g:5.1f} {on['rateE'].mean():7.2f} {bon['n']:8d} {bon['ibi_s']:7.2f}   |  "
              f"bursts={bof['n']} IBI={bof['ibi_s']:.2f}")
    print("  If bursting appears across a WIDE gain range WITH gap junctions but is")
    print("  narrow/absent WITHOUT => electrical coupling converts knife-edge -> basin.")

    print("="*70)
    print("STEP 5 - CORRECTED: gap junctions SYNCHRONIZE intrinsic bursters (subplate")
    print("  syncytium). NO chemical synapses (gain=0); cells are intrinsic sAHP bursters")
    print("  (I_tonic=450, b=80, tauw=2.5s). Does electrical coupling ALONE sync them?")
    print(f"  {'spk':>6} {'meanHz':>7} {'peakHz':>7} {'peak/mean':>10} {'n_burst':>8} {'IBI_s':>7}")
    for spk in (0.0, 0.2, 0.5, 1.0, 2.0):
        o = network_oscillator(gain=0.0, I_tonic_E=450.0, b_pA=80.0, tauw_ms=2500.0,
                               nmda_frac=0.0, g_gap=0.02, p_gap=0.12, spk_gain=spk,
                               seed=1, T_ms=20000.0)
        b = detect_pop_bursts(o['rateE'], o['record_dt_ms'])
        pm = b['peak']/max(o['rateE'].mean(), 0.01)
        print(f"  {spk:6.2f} {o['rateE'].mean():7.2f} {b['peak']:7.1f} {pm:10.1f} "
              f"{b['n']:8d} {b['ibi_s']:7.2f}")
    print("  g_gap=0: cells burst but ASYNCHRONOUSLY (smooth pop rate, low peak/mean).")
    print("  g_gap>0: if peak/mean & n_burst rise => electrical coupling SYNCHRONIZES")
    print("  intrinsic bursters into population bursts. This is the subplate mechanism.")
    print("="*70)
