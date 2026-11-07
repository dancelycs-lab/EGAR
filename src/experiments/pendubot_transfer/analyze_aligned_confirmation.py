import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
modes = ["energy", "tuned_critic", "no_energy"]
data = {mode: np.load(ROOT / f"aligned_confirm_{mode}.npz") for mode in modes}
rng = np.random.default_rng(20261123)
bootstrap_index = rng.integers(0, 64, (20000, 64))
output = {}
for mode, result in data.items():
    trace = result["trace"]
    angle_error = (trace[..., :2] - [np.pi, 0.0] + np.pi) % (2 * np.pi) - np.pi
    inside = (np.abs(angle_error).max(-1) < 0.15) & (np.abs(trace[..., 2:4]).max(-1) < 0.6)
    difference = result["arrival"] - data["energy"]["arrival"]
    output[mode] = {
        "success": int(result["success"].sum()),
        "n": 64,
        "mean_capped_time": float(result["arrival"].mean()),
        "mean_effort": float(result["effort"].mean()),
        "last_2s_all_saved_samples_in_capture": int(inside[-100:].all(0).sum()),
        "paired_time_difference_from_energy": float(difference.mean()),
        "paired_bootstrap95": np.quantile(
            difference[bootstrap_index].mean(1), [0.025, 0.975]
        ).tolist(),
    }
result = {
    "methods": output,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20261123,
    "scope": "Paired descriptive intervals over the frozen 64 initial states; no multiplicity adjustment.",
}
(ROOT / "aligned_confirmation_analysis.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
