"""
M1 micro-tests T1.1 – T1.4: exact synapse-delivery tests.

These tests run a tiny hand-built simulation in Python Euler (same algorithm as CUDA)
to verify ring-buffer delay delivery, fan-out, fan-in, and conductance decay.
All tests use exact deterministic logic (no RNG).

Usage:
    python3 tests/test_m1_microtests.py

Pass criteria (all EXACT):
  T1.1: 2 neurons, 1 synapse, delay=30 steps → post g_ampa jumps at step 130, zero before
  T1.2: 1 pre → 10 post, delays 10..100 → each jumps at exact step
  T1.3: 5 pre → 1 post, same delay, simultaneous spikes → g = sum of 5 weights
  T1.4: single synapse, 1 spike; g_ampa(t) vs analytic w*exp(-t/tau) and vs step-decay
"""
import numpy as np
import sys

MAX_DELAY = 256
DT = 0.1  # ms
TOL = 1e-5  # conductance tolerance (nS)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def run_ring_sim(
    n_pre, n_post,
    row_offset, col_index, w_syn, delay_steps,
    pre_spike_steps,   # dict: step → list of pre neuron ids that spike
    n_steps,
    tau_ampa
):
    """Minimal ring-buffer simulation. Returns g_ampa[step][j] trace."""
    ring = np.zeros((MAX_DELAY, n_post), dtype=np.float64)
    g_ampa = np.zeros(n_post, dtype=np.float64)
    dt_ta = DT / tau_ampa

    g_trace = np.zeros((n_steps, n_post), dtype=np.float64)

    head = 0
    for step in range(n_steps):
        # Deliver
        g_ampa += ring[head]
        ring[head] = 0.0

        # (no dynamics / spike detection in micro-tests — we inject spikes externally)

        # Conductance decay
        g_ampa *= (1.0 - dt_ta)

        # Record after decay (matches CUDA step order: deliver → dynamics → decay → detect → enqueue)
        # We actually want to record BEFORE enqueue so we can see the delivered conductance
        # For micro-tests we record right after delivery+decay, before enqueue
        # But to test "does g_ampa jump at arrival step", record AFTER deliver but we need
        # to check the value of g_ampa AFTER the ring read on the arrival step.
        # Order in CUDA: deliver → dynamics → decay → detect → enqueue → advance_head
        # So g_ampa after delivery+decay is what the neuron sees.
        g_trace[step] = g_ampa.copy()

        # Enqueue spikes
        if step in pre_spike_steps:
            for i in pre_spike_steps[step]:
                start = row_offset[i]
                end = row_offset[i+1]
                for s in range(start, end):
                    j = col_index[s]
                    d = delay_steps[s]
                    slot = (head + d) % MAX_DELAY
                    ring[slot, j] += w_syn[s]

        # Advance head
        head = (head + 1) % MAX_DELAY

    return g_trace


def test_T1_1():
    """T1.1: 2 neurons, 1 synapse, delay=30 steps. Pre spikes at step 100."""
    n_pre, n_post = 2, 2
    row_offset = np.array([0, 1, 1], dtype=np.int32)  # neuron 0 has 1 synapse, neuron 1 has 0
    col_index  = np.array([1], dtype=np.int32)
    w_syn      = np.array([0.5], dtype=np.float64)   # 0.5 nS
    delay_steps= np.array([30], dtype=np.int32)
    tau_ampa   = 3.0  # ms

    n_steps = 200
    spike_at = {100: [0]}   # pre neuron 0 spikes at step 100

    g = run_ring_sim(n_pre, n_post, row_offset, col_index, w_syn, delay_steps,
                     spike_at, n_steps, tau_ampa)

    # Arrival should be at step 100 + 30 = 130
    arrival = 130
    arrival_val = g[arrival, 1]
    # Before arrival: all zeros
    before = g[:arrival, 1]
    # After decay from arrival, the first sample after the jump is at step 130
    # g_ampa[130] = (0 + w) * (1 - dt/tau) after delivery+decay
    expected_first = w_syn[0] * (1.0 - DT/tau_ampa)

    ok = True
    if before.max() > TOL:
        print(f"  T1.1 FAIL: g_ampa non-zero before arrival: max={before.max():.2e}")
        ok = False
    if abs(arrival_val - expected_first) > TOL:
        print(f"  T1.1 FAIL: g_ampa at arrival={arrival_val:.6f}, expected={expected_first:.6f}")
        ok = False
    # Check that step 129 is truly zero
    if g[arrival-1, 1] > TOL:
        print(f"  T1.1 FAIL: g_ampa at step 129 = {g[arrival-1,1]:.2e} (should be 0)")
        ok = False

    label = PASS if ok else FAIL
    print(f"  [{label}] T1.1: 1-synapse delay=30, arrival at step 130")
    results.append(ok)


def test_T1_2():
    """T1.2: 1 pre → 10 post, delays = 10, 20, …, 100 steps. One pre spike at step 0."""
    n_pre, n_post = 1, 10
    delays = np.arange(10, 110, 10, dtype=np.int32)
    row_offset  = np.array([0, 10], dtype=np.int32)
    col_index   = np.arange(10, dtype=np.int32)
    w_syn       = np.ones(10, dtype=np.float64) * 0.3  # 0.3 nS each
    delay_steps = delays
    tau_ampa    = 3.0

    n_steps = 120
    spike_at = {0: [0]}

    g = run_ring_sim(n_pre, n_post, row_offset, col_index, w_syn, delay_steps,
                     spike_at, n_steps, tau_ampa)

    ok = True
    for k, d in enumerate(delays):
        arrival = int(d)
        # before arrival: zero
        if g[:arrival, k].max() > TOL:
            print(f"  T1.2 FAIL: post[{k}] (delay={d}) non-zero before arrival: {g[:arrival,k].max():.2e}")
            ok = False
        # at arrival: positive
        if g[arrival, k] < TOL:
            print(f"  T1.2 FAIL: post[{k}] (delay={d}) zero at arrival step {arrival}: {g[arrival,k]:.2e}")
            ok = False

    label = PASS if ok else FAIL
    print(f"  [{label}] T1.2: 1→10 fan-out, delays 10..100 steps, each arrives at exact step")
    results.append(ok)


def test_T1_3():
    """T1.3: 5 pre → 1 post, same delay (20 steps), simultaneous spikes. g = sum of 5 weights."""
    n_pre, n_post = 5, 1
    # Each pre neuron has 1 synapse targeting post 0
    row_offset = np.arange(6, dtype=np.int32)  # [0,1,2,3,4,5]
    col_index  = np.zeros(5, dtype=np.int32)
    w_syn      = np.array([0.10, 0.20, 0.30, 0.15, 0.25], dtype=np.float64)
    delay_steps= np.full(5, 20, dtype=np.int32)
    tau_ampa   = 3.0

    n_steps = 50
    # All 5 pre neurons spike at step 5
    spike_at = {5: list(range(5))}

    g = run_ring_sim(n_pre, n_post, row_offset, col_index, w_syn, delay_steps,
                     spike_at, n_steps, tau_ampa)

    arrival = 5 + 20  # = 25
    expected = w_syn.sum() * (1.0 - DT / tau_ampa)
    actual   = g[arrival, 0]

    ok = abs(actual - expected) < TOL
    if not ok:
        print(f"  T1.3 FAIL: g_ampa at arrival={actual:.6f}, expected={expected:.6f}")

    label = PASS if ok else FAIL
    print(f"  [{label}] T1.3: 5-fan-in, g={actual:.4f} nS, expected={expected:.4f} nS")
    results.append(ok)


def test_T1_4():
    """T1.4: Single synapse, one spike at step 0. Sample g_ampa for 150 steps.
       Compare to analytic w * prod_k (1 - dt/tau)^k (exact Euler decay).
    """
    n_pre, n_post = 1, 1
    row_offset  = np.array([0, 1], dtype=np.int32)
    col_index   = np.array([0], dtype=np.int32)
    delay_steps = np.array([10], dtype=np.int32)
    w0          = 0.4  # nS
    w_syn       = np.array([w0], dtype=np.float64)
    tau_ampa    = 3.0

    n_steps = 150
    spike_at = {0: [0]}

    g = run_ring_sim(n_pre, n_post, row_offset, col_index, w_syn, delay_steps,
                     spike_at, n_steps, tau_ampa)

    # Arrival at step 10.
    # After delivery at step 10: g_ampa = w0, then decay → g[10,0] = w0 * (1 - dt/tau)
    # At step 10+k: g[10+k, 0] = w0 * (1 - dt/tau)^(k+1)
    arrival = 10
    dt_ta = DT / tau_ampa

    ok = True
    max_err = 0.0
    for k in range(0, 100):
        step = arrival + k
        expected = w0 * (1.0 - dt_ta)**(k + 1)
        actual   = g[step, 0]
        err = abs(actual - expected)
        max_err = max(max_err, err)
        if err > TOL:
            print(f"  T1.4 FAIL at step {step} (k={k}): actual={actual:.6e}, expected={expected:.6e}, err={err:.2e}")
            ok = False
            break  # one failure is enough to report

    label = PASS if ok else FAIL
    print(f"  [{label}] T1.4: conductance decay matches analytic; max_err={max_err:.2e} nS (tol={TOL:.0e})")
    results.append(ok)


if __name__ == '__main__':
    print("=" * 60)
    print("M1 Micro-tests T1.1 – T1.4 (Python Euler, exact)")
    print("=" * 60)

    test_T1_1()
    test_T1_2()
    test_T1_3()
    test_T1_4()

    print()
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  ALL {n_total} TESTS PASSED")
        sys.exit(0)
    else:
        print(f"  {n_pass}/{n_total} PASSED, {n_total - n_pass} FAILED")
        sys.exit(1)
