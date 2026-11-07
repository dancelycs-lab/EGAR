import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261122)
bootstrap_index = rng.integers(0, 64, (20000, 64))


def summarize(prefix, modes, reference):
    data = {mode: np.load(ROOT / f"{prefix}_{mode}.npz") for mode in modes}
    output = {}
    for mode, result in data.items():
        trace = result["trace"]
        angle_error = (trace[..., :2] - [np.pi, 0.0] + np.pi) % (2 * np.pi) - np.pi
        inside = (np.abs(angle_error).max(-1) < 0.15) & (np.abs(trace[..., 2:4]).max(-1) < 0.6)
        difference = result["arrival"] - data[reference]["arrival"]
        output[mode] = {
            "success": int(result["success"].sum()),
            "n": 64,
            "mean_capped_time": float(result["arrival"].mean()),
            "mean_effort": float(result["effort"].mean()),
            "last_2s_all_saved_samples_in_capture": int(inside[-100:].all(0).sum()),
            "paired_time_difference_from_reference": float(difference.mean()),
            "paired_bootstrap95": np.quantile(
                difference[bootstrap_index].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    return output


result = {
    "terminal_cost_comparison": summarize(
        "aligned_core", ["energy", "critic", "no_energy"], "energy"
    ),
    "gain_noise_comparison": summarize(
        "final_aligned_gain", ["nominal", "dynamics", "oracle"], "dynamics"
    ),
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20261122,
    "scope": "Paired descriptive intervals over the 64 frozen initial states; no multiplicity adjustment.",
}
(ROOT / "final_aligned_analysis.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
