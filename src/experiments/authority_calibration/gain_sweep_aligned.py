import json
import time

import numpy as np

from aligned_rollout import AlignedRollout
from screen import OUT, experiment


if __name__ == "__main__":
    states = np.load(OUT / "gain_confirmation_states.npy")[:32]
    gains = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
    rows = []
    for gain in gains:
        for mode in ["nominal", "dynamics"]:
            start = time.perf_counter()
            result = experiment(
                states,
                mode,
                [gain, 0.0, 0.0],
                observer=True,
                controller_cls=AlignedRollout,
            )
            assert np.isfinite(result["trace"]).all()
            tag = f"aligned_sweep_g{gain}_{mode}"
            np.savez_compressed(OUT / f"{tag}.npz", **result)
            row = {
                "gain": gain,
                "mode": mode,
                "n": len(states),
                "success": int(result["success"].sum()),
                "arrival": float(result["arrival"].mean()),
                "effort": float(result["effort"].mean()),
                "gain_mae": float(np.mean(np.abs(result["theta"][-1, :, 0] - gain))),
                "wall_seconds": time.perf_counter() - start,
            }
            rows.append(row)
            print(row, flush=True)
            (OUT / "aligned_gain_sweep_results.json").write_text(json.dumps(rows, indent=2))
