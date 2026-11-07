from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
AUTH = HERE.parent / 'authority_calibration'
sys.path.insert(0, str(AUTH))
from aligned_rollout import AlignedRollout
from screen import torch, derivative, integrate, features, rhs, error, K


def run(mode, scenario, seed, backend=None, output=None):
    if backend is None:
        Controller = AlignedRollout
        advance, vectorfield, regressor, drift, local_gain = integrate, derivative, features, rhs, K
        states = np.load(AUTH / 'gain_confirmation_states.npy')
    else:
        Controller = backend.Controller
        advance, vectorfield, regressor, drift, local_gain = backend.integrate, backend.derivative, backend.features, backend.rhs, backend.K
        states = np.load(backend.STATE_PATH)
    destination = HERE if output is None else Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    n = len(states); dt = .002; change = 1. if scenario == 'step' else 0.
    x = torch.tensor(states, dtype=torch.float64); controller = Controller(n)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.tensor([.001, .001, .01, .01], dtype=torch.float64)
    y = x + torch.randn(x.shape, generator=gen, dtype=torch.float64)*noise
    xh = y.clone(); R = torch.diag(noise**2).repeat(n, 1, 1); cov = R.clone()
    Q = torch.diag(torch.tensor([1e-12, 1e-12, 1e-7, 1e-7], dtype=torch.float64))
    eye = torch.eye(4, dtype=torch.float64)
    G = torch.eye(3, dtype=torch.float64).repeat(n, 1, 1)*1e-4
    r = G @ torch.tensor([1., 0., 0.], dtype=torch.float64)
    v0 = y[:, 2:].clone(); phi = torch.zeros((n, 2, 3), dtype=torch.float64)
    f = torch.zeros((n, 2), dtype=torch.float64)
    arrival = np.full(n, 20.); post_arrival = np.full(n, 20.-change)
    hold = np.zeros(n, dtype=int); post_hold = hold.copy(); effort = np.zeros(n)
    iae = np.zeros(n); post_iae = np.zeros(n)
    trace = [np.c_[states, np.zeros(n)]]; estimates = [controller.theta.numpy().copy()]
    error_previous = abs(error(x)).sum(-1).numpy(); times = [0.]; true_gains = [1. if change else 5/6]
    arrived_before = None; start = time.perf_counter()
    with torch.no_grad():
        for k in range(10000):
            gain = (1. if k*dt < change else .8) if scenario == 'step' else 5/6
            true = torch.tensor([gain, 0., 0.], dtype=torch.float64).repeat(n, 1)
            if mode == 'oracle': controller.theta = true.clone()
            if k == round(change/dt): arrived_before = arrival < 20.
            if k % 25 == 0: u = controller.action(xh.float()).double().clamp(-.5, .5)
            e = torch.cat((error(xh), xh[:, 2:]), -1)
            local = (e[:, :2].abs().amax(-1) < .3) & (e[:, 2:].abs().amax(-1) < 1.5)
            act = torch.where(local, (-(e.float() @ local_gain.T).squeeze(-1)).double().clamp(-.5, .5), u)
            xn = advance(x, act, true, dt)
            yn = xn + torch.randn(x.shape, generator=gen, dtype=torch.float64)*noise
            phi += (regressor(y, act)+regressor(yn, act))*.001
            f += (drift(y, torch.zeros(n))[:, 2:]+drift(yn, torch.zeros(n))[:, 2:])*.001
            perturb = eye[None]*1e-5
            jac = ((vectorfield(xh[:, None]+perturb, act[:, None].expand(-1, 4), controller.theta[:, None])
                    -vectorfield(xh[:, None]-perturb, act[:, None].expand(-1, 4), controller.theta[:, None]))/2e-5).transpose(-1, -2)
            F = eye+dt*jac; cov = F@cov@F.transpose(-1, -2)+Q
            kg = torch.linalg.solve(cov+R, cov.transpose(-1, -2)).transpose(-1, -2)
            pred = advance(xh, act, controller.theta, dt)
            xh = pred+torch.einsum('nij,nj->ni', kg, yn-pred)
            J = eye-kg; cov = J@cov@J.transpose(-1, -2)+kg@R@kg.transpose(-1, -2)
            x = xn; y = yn
            if (k+1) % 100 == 0:
                if mode == 'dynamics':
                    target = y[:, 2:]-v0-f
                    G = .95*G+torch.einsum('nki,nkj->nij', phi, phi)
                    r = .95*r+torch.einsum('nki,nk->ni', phi, target)
                    controller.theta = torch.linalg.solve(G, r).clamp(torch.tensor([.4, 0, 0]), torch.tensor([1.6, 2., 2.]))
                v0 = y[:, 2:].clone(); phi.zero_(); f.zero_()
            ok = ((error(x).abs().amax(-1) < .15) & (x[:, 2:].abs().amax(-1) < .6)).numpy()
            hold = np.where(ok, hold+1, 0)
            arrival = np.where((hold >= 250) & (arrival == 20.), (k+1)*dt-.5, arrival)
            if k*dt >= change:
                post_hold = np.where(ok, post_hold+1, 0)
                post_arrival = np.where((post_hold >= 250) & (post_arrival == 20.-change),
                                        max(0., (k+1)*dt-.5-change), post_arrival)
            effort += act.numpy()**2*dt
            error_now = abs(error(x)).sum(-1).numpy()
            increment = .5*dt*(error_previous+error_now); iae += increment
            if k*dt >= change: post_iae += increment
            error_previous = error_now
            if (k+1) % 10 == 0:
                trace.append(torch.cat((x, act[:, None]), -1).numpy()); times.append((k+1)*dt)
                estimates.append(controller.theta.numpy().copy()); true_gains.append(gain)
    result = dict(initial_states=states, arrival=arrival, success=arrival < 20, post_arrival=post_arrival,
                  post_success=post_arrival < 20.-change, recovered_before_change=arrived_before,
                  effort=effort, angle_iae=iae, post_angle_iae=post_iae,
                  trace=np.asarray(trace), theta=np.asarray(estimates), time=np.asarray(times),
                  true_gain=np.asarray(true_gains), seed=seed, change_time=change)
    assert np.isfinite(result['trace']).all() and abs(result['trace'][..., 4]).max() <= .500001
    tag = f'{scenario}_{mode}_seed{seed}'
    np.savez_compressed(destination / f'{tag}.npz', **result)
    row = dict(scenario=scenario, mode=mode, seed=seed, n=n, recovery=int(result['success'].sum()),
               time=float(arrival.mean()), post_recovery=int(result['post_success'].sum()),
               post_time=float(post_arrival.mean()), angle_iae=float(iae.mean()),
               post_angle_iae=float(post_iae.mean()), effort=float(effort.mean()), seconds=time.perf_counter()-start)
    (destination / f'{tag}.json').write_text(json.dumps(row, indent=2)); print(row, flush=True)
    if backend is None and scenario == 'static' and seed == 51:
        with np.load(AUTH / f'final_aligned_gain_{mode}.npz') as legacy:
            np.testing.assert_array_equal(arrival, legacy['arrival'])
            np.testing.assert_allclose(effort, legacy['effort'], atol=1e-12, rtol=1e-12)
        print('PASS: seed-51 arrival and effort reproduce existing archive', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['nominal','dynamics','oracle'], required=True)
    ap.add_argument('--scenario', choices=['static','step'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args(); run(args.mode, args.scenario, args.seed)
