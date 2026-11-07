import json
import time

import numpy as np

from aligned_rollout import AlignedRollout
from screen import OUT, experiment


if __name__ == "__main__":
    xs = np.load(OUT / "gain_confirmation_states.npy")
    true_parameters = [5.0 / 6.0, 0.0, 0.0]
    rows = []
    for mode in ["nominal", "dynamics", "oracle"]:
        start = time.perf_counter()
        result = experiment(
            xs,
            mode,
            true_parameters,
            observer=True,
            controller_cls=AlignedRollout,
        )
        assert np.isfinite(result["trace"]).all()
        np.savez_compressed(OUT / f"final_aligned_gain_{mode}.npz", **result)
        row = {
            "mode": mode,
            "n": len(xs),
            "success": int(result["success"].sum()),
            "arrival": float(result["arrival"].mean()),
            "effort": float(result["effort"].mean()),
            "physical_parameter_mae": np.mean(
                np.abs(result["theta"][-1] - true_parameters) * [1.0, 0.03, 0.03], axis=0
            ).tolist(),
            "wall_seconds": time.perf_counter() - start,
        }
        rows.append(row)
        print(row, flush=True)
        (OUT / "final_aligned_gain_results.json").write_text(json.dumps(rows, indent=2))
