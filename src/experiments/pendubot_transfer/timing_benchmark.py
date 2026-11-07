import json
import time

import numpy as np

from aligned_confirm import aligned_action, terminal
from ilqr_mpc_baseline import create_controller
from screen import OUT
import torch


def summarize(samples):
    samples = np.asarray(samples)
    return {
        "mean_seconds": float(samples.mean()),
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.quantile(samples, 0.95)),
        "max_seconds": float(samples.max()),
    }


if __name__ == "__main__":
    state = np.load(OUT / "confirmation_states.npy")[0]
    tensor_state = torch.tensor(state[None, :], dtype=torch.float32)
    rollout_samples = []
    with torch.no_grad():
        for index in range(520):
            start = time.perf_counter()
            aligned_action(tensor_state, 0.5, "energy", terminal)
            elapsed = time.perf_counter() - start
            if index >= 20:
                rollout_samples.append(elapsed)

    ilqr = create_controller(1.0, 2.0, max_iter=10)
    controller_state = state.copy()
    controller_state[0] %= 2.0 * np.pi
    controller_state[1] = (controller_state[1] + np.pi) % (2.0 * np.pi) - np.pi
    ilqr_samples = []
    for index in range(520):
        start = time.perf_counter()
        ilqr.get_control_output(controller_state, index * 0.05)
        elapsed = time.perf_counter() - start
        if index >= 20:
            ilqr_samples.append(elapsed)

    result = {
        "state_index": 0,
        "warmup_calls": 20,
        "timed_calls": 500,
        "energy_rollout": summarize(rollout_samples),
        "strengthened_ilqr_mpc": summarize(ilqr_samples),
        "replan_period_seconds": 0.05,
        "scope": "Single CPU process, single fixed state, steady-call timing; not a hardware-independent complexity comparison.",
    }
    (OUT / "single_state_timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
