from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def run(system, mode):
    sys.path.insert(0, str(EXPERIMENTS / ('authority_calibration' if system == 'acrobot' else 'pendubot_transfer')))
    if system == 'acrobot':
        from aligned_rollout import AlignedRollout
        import screen as core
        original = AlignedRollout(64)
        base = lambda x: original.tail_action(x.double(), original.theta)
        advance = lambda x, u, dt: core.integrate(x, u, original.theta.repeat_interleave(10, 0), dt)
        terminal = original.terminal
        states = np.load(core.BASE / 'heldout_initial_states.npy')
    else:
        import screen as core
        import aligned_confirm as aligned
        base = lambda x: core.base(x.float()).double().clamp(-.5, .5)
        advance = core.integrate
        terminal = lambda x: core.terminal(x, 'energy')
        states = np.load(core.OUT / 'confirmation_states.npy')
    torch = core.torch

    def action(x):
        n = len(x); nc = 10
        z = x.double().repeat_interleave(nc, 0)
        pulse = torch.tensor([0., -1, -.5, 0, .5, 1, -1, 1, -1, 1], dtype=torch.float64).repeat(n) * .5
        duration = torch.tensor([0, .05, .05, .05, .05, .05, .15, .15, .3, .3]).repeat(n)
        index = torch.arange(nc).repeat(n)
        held = base(x).double(); held_all = held.repeat_interleave(nc)
        cost = torch.zeros(n * nc, dtype=torch.float64)
        grid = sorted(set(np.round(np.r_[np.arange(0, 1.20001, .02), .05, .15, .3], 10)))
        for t, tn in zip(grid[:-1], grid[1:]):
            if mode == 'zero': u = torch.zeros_like(held_all)
            elif mode == 'hold': u = held_all.clone()
            else:
                u = (original.tail_action(z, original.theta.repeat_interleave(nc, 0))
                     if system == 'acrobot' else core.base(z.float()).double().clamp(-.5, .5))
            if t < .05 - 1e-9: u = torch.where(index == 0, held_all, u)
            u = torch.where(t < duration - 1e-9, pulse, u)
            e = torch.cat((core.error(z), z[:, 2:]), -1)
            local = (e[:, :2].abs().amax(-1) < .3) & (e[:, 2:].abs().amax(-1) < 1.5)
            local_u = (-(e.float() @ core.K.T).squeeze(-1)).double().clamp(-.5, .5)
            u = torch.where(local, local_u, u)
            dt = tn - t
            z = advance(z, u, dt/2); z = advance(z, u, dt/2)
            cost += dt * ((1-torch.cos(core.error(z))).sum(-1)+.01*(z[:, 2:]**2).sum(-1))
        cost += terminal(z)
        best = cost.reshape(n, nc).argmin(-1)
        result = torch.where(best == 0, held, pulse.reshape(n, nc)[torch.arange(n), best])
        return result.float() if system == 'acrobot' else result

    # Test unchanged first-action mapping against the original in policy mode.
    with torch.no_grad():
        if mode == 'policy':
            x = torch.tensor(states, dtype=torch.float32)
            reference = original.action(x) if system == 'acrobot' else aligned.aligned_action(x, .5, 'energy')
            np.testing.assert_array_equal(action(x).numpy(), reference.numpy())
    start = time.perf_counter()
    if system == 'acrobot':
        class Controller:
            def action(self, x): return action(x)
        result = core.simulate(Controller(), states)
    else:
        aligned.aligned_action = lambda x, limit, objective, terminal_fn=None: action(x)
        result = aligned.simulate_aligned(states, .5, 'energy')
    result.update(initial_states=states, time=.002+np.arange(1000)*.02)
    assert np.isfinite(result['trace']).all()
    assert abs(result['trace'][..., 4]).max() <= .500001
    np.savez_compressed(HERE / f'tail_{system}_{mode}.npz', **result)
    row = dict(system=system, continuation=mode, n=len(states), recovery=int(result['success'].sum()),
               time=float(result['arrival'].mean()), effort=float(result['effort'].mean()),
               seconds=time.perf_counter()-start)
    (HERE / f'tail_{system}_{mode}.json').write_text(json.dumps(row, indent=2))
    print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=['acrobot', 'pendubot'], required=True)
    parser.add_argument('--mode', choices=['zero', 'hold', 'policy'], required=True)
    args = parser.parse_args(); run(args.system, args.mode)
