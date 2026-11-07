import json
import time

import numpy as np

import screen as legacy
import torch
from screen import OUT, K, base, error, integrate, terminal


def aligned_action(x, limit, mode, terminal_fn=terminal):
    if mode == "policy":
        return base(x).clamp(-limit, limit)
    n = len(x)
    nc = 10
    z = x.double().repeat_interleave(nc, 0)
    pulse = torch.tensor(
        [0.0, -1.0, -0.5, 0.0, 0.5, 1.0, -1.0, 1.0, -1.0, 1.0],
        dtype=torch.float64,
    ).repeat(n) * limit
    duration = torch.tensor(
        [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.30, 0.30],
        dtype=torch.float64,
    ).repeat(n)
    index = torch.arange(nc).repeat(n)
    held = base(x).double().clamp(-limit, limit)
    held_repeated = held.repeat_interleave(nc)
    cost = torch.zeros(n * nc, dtype=torch.float64)
    grid = sorted(set(np.round(np.r_[np.arange(0.0, 1.20001, 0.02), 0.05, 0.15, 0.30], 10)))
    for t0, t1 in zip(grid[:-1], grid[1:]):
        u = base(z.float()).double().clamp(-limit, limit)
        if t0 < 0.05 - 1e-12:
            u = torch.where(index == 0, held_repeated, u)
        u = torch.where(t0 < duration - 1e-12, pulse, u)
        e = torch.cat((error(z), z[:, 2:]), -1)
        local = (e[:, :2].abs().amax(-1) < 0.3) & (e[:, 2:].abs().amax(-1) < 1.5)
        local_u = (-(e.float() @ K.T).squeeze(-1)).double().clamp(-limit, limit)
        u = torch.where(local, local_u, u)
        dt = t1 - t0
        z = integrate(z, u, dt / 2.0)
        z = integrate(z, u, dt / 2.0)
        cost += dt * ((1.0 - torch.cos(error(z))).sum(-1) + 0.01 * (z[:, 2:] ** 2).sum(-1))
    cost += terminal_fn(z, mode)
    best = cost.reshape(n, nc).argmin(-1)
    return torch.where(best == 0, held, pulse.reshape(n, nc)[torch.arange(n), best])


def simulate_aligned(xs, limit, mode, terminal_fn=terminal):
    x = torch.tensor(xs, dtype=torch.float64)
    n = len(x)
    arrival = np.full(n, 20.0)
    hold = np.zeros(n)
    uptime = np.zeros(n)
    effort = np.zeros(n)
    trace = []
    calls = []
    with torch.no_grad():
        for k in range(10000):
            if mode == "policy" or k % 25 == 0:
                tic = time.perf_counter()
                u = aligned_action(x.float(), limit, mode, terminal_fn).double()
                calls.append(time.perf_counter() - tic)
            e = torch.cat((error(x), x[:, 2:]), -1)
            local = (e[:, :2].abs().amax(-1) < 0.3) & (e[:, 2:].abs().amax(-1) < 1.5)
            local_u = (-(e.float() @ K.T).squeeze(-1)).double().clamp(-limit, limit)
            applied = torch.where(local, local_u, u)
            x = integrate(x, applied, 0.002)
            inside = ((error(x).abs().amax(-1) < 0.15) & (x[:, 2:].abs().amax(-1) < 0.6)).numpy()
            hold = np.where(inside, hold + 1, 0)
            arrival = np.where((hold >= 250) & (arrival == 20.0), (k + 1) * 0.002 - 0.5, arrival)
            uptime += inside
            effort += applied.numpy() ** 2 * 0.002
            if k % 10 == 0:
                trace.append(torch.cat((x, applied[:, None]), -1).numpy())
    return {
        "arrival": arrival,
        "success": arrival < 20.0,
        "uptime": uptime / 10000,
        "effort": effort,
        "trace": np.asarray(trace),
        "batch_call_seconds": np.asarray(calls),
    }


def no_energy_terminal(x, mode):
    return (1.0 - torch.cos(error(x))).sum(-1) + 0.1 * (x[..., 2:] ** 2).sum(-1)


if __name__ == "__main__":
    xs = np.load(OUT / "confirmation_states.npy")
    critic_scale = 0.1

    def tuned_critic_terminal(x, mode):
        return -critic_scale * legacy.p.value(x.float())

    cases = [
        ("energy", terminal),
        ("no_energy", no_energy_terminal),
        ("tuned_critic", tuned_critic_terminal),
    ]
    rows = []
    for name, terminal_fn in cases:
        start = time.perf_counter()
        result = simulate_aligned(xs, 0.5, name, terminal_fn)
        assert np.isfinite(result["trace"]).all()
        assert np.abs(result["trace"][..., -1]).max() <= 0.500001
        np.savez_compressed(OUT / f"aligned_confirm_{name}.npz", **result)
        row = {
            "mode": name,
            "n": len(xs),
            "success": int(result["success"].sum()),
            "arrival": float(result["arrival"].mean()),
            "effort": float(result["effort"].mean()),
            "batch_call_mean": float(result["batch_call_seconds"].mean()),
            "batch_call_p95": float(np.quantile(result["batch_call_seconds"], 0.95)),
            "wall_seconds": time.perf_counter() - start,
        }
        rows.append(row)
        print(row, flush=True)
        (OUT / "aligned_confirmation_results.json").write_text(json.dumps(rows, indent=2))
