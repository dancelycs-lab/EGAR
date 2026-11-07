import json
import time

import numpy as np

from aligned_rollout import AlignedRollout
import torch
from screen import BASE, OUT, error, simulate


class NoEnergyAligned(AlignedRollout):
    def terminal(self, x):
        return (1.0 - torch.cos(error(x))).sum(-1) + 0.1 * (x[:, 2:] ** 2).sum(-1)


if __name__ == "__main__":
    xs = np.load(BASE / "heldout_initial_states.npy")
    cases = []
    energy = AlignedRollout(len(xs))
    cases.append(("energy", energy))
    critic = AlignedRollout(len(xs))
    critic.objective = "critic"
    cases.append(("critic", critic))
    cases.append(("no_energy", NoEnergyAligned(len(xs))))
    rows = []
    for name, controller in cases:
        start = time.perf_counter()
        result = simulate(controller, xs)
        assert np.isfinite(result["trace"]).all()
        np.savez_compressed(OUT / f"aligned_core_{name}.npz", **result)
        row = {
            "mode": name,
            "n": len(xs),
            "success": int(result["success"].sum()),
            "arrival": float(result["arrival"].mean()),
            "effort": float(result["effort"].mean()),
            "batch_call_mean": float(result["batch_replan_seconds"].mean()),
            "batch_call_p95": float(np.quantile(result["batch_replan_seconds"], 0.95)),
            "wall_seconds": time.perf_counter() - start,
        }
        rows.append(row)
        print(row, flush=True)
        (OUT / "aligned_core_results.json").write_text(json.dumps(rows, indent=2))
