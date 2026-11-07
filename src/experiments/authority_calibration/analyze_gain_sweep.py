import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261124)
bootstrap_index = rng.integers(0, 32, (20000, 32))
rows = []
for gain in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
    nominal = np.load(ROOT / f"aligned_sweep_g{gain}_nominal.npz")
    adaptive = np.load(ROOT / f"aligned_sweep_g{gain}_dynamics.npz")
    difference = nominal["arrival"] - adaptive["arrival"]
    rows.append(
        {
            "gain": gain,
            "nominal_success": int(nominal["success"].sum()),
            "adaptive_success": int(adaptive["success"].sum()),
            "nominal_capped_time": float(nominal["arrival"].mean()),
            "adaptive_capped_time": float(adaptive["arrival"].mean()),
            "adaptive_time_improvement": float(difference.mean()),
            "paired_bootstrap95": np.quantile(
                difference[bootstrap_index].mean(1), [0.025, 0.975]
            ).tolist(),
            "adaptive_gain_mae": float(
                np.mean(np.abs(adaptive["theta"][-1, :, 0] - gain))
            ),
        }
    )
result = {
    "rows": rows,
    "n_per_gain": 32,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20261124,
    "scope": "Sensitivity on a previously observed 32-state subset; paired descriptive intervals, not independent confirmation.",
}
(ROOT / "aligned_gain_sweep_analysis.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
