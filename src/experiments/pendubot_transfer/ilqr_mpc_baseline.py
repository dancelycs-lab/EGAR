from runtime import ASSETS, BUILD
import json
import sys
import time
from pathlib import Path


import numpy as np

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.model.model_parameters import model_parameters
from screen import K, OUT, error, integrate
import torch


MODEL_PARAMETERS = ASSETS / "model_parameters.yml"
INITIAL_TRAJECTORY = ASSETS / "ilqr_initial_trajectory.csv"


def create_controller(horizon, terminal_energy, max_iter=2):
    parameters = model_parameters(filepath=str(MODEL_PARAMETERS))
    parameters.set_motor_inertia(0.0)
    parameters.set_damping([0.0, 0.0])
    parameters.set_cfric([0.0, 0.0])
    parameters.set_torque_limit([0.5, 0.0])
    controller = ILQRMPCCPPController(model_pars=parameters)
    controller.set_goal([np.pi, 0.0, 0.0, 0.0])
    controller.set_parameters(
        N=round(horizon / 0.005),
        dt=0.005,
        max_iter=max_iter,
        regu_init=1.0,
        max_regu=10000.0,
        min_regu=0.0001,
        break_cost_redu=1e-6,
        integrator="runge_kutta",
        trajectory_stabilization=False,
        shifting=10,
    )
    controller.set_cost_parameters(
        sCu=[0.0001, 0.0001],
        sCp=[0.0, 0.0],
        sCv=[0.0, 0.0],
        sCen=0.0,
        fCp=[10.0, 10.0],
        fCv=[0.5, 0.5],
        fCen=terminal_energy,
    )
    controller.load_init_traj(str(INITIAL_TRAJECTORY), num_break=40, poly_degree=3)
    controller.init()
    return controller


def run_trial(initial_state, horizon, terminal_energy, max_iter=2):
    controller = create_controller(horizon, terminal_energy, max_iter=max_iter)
    x = torch.tensor(initial_state, dtype=torch.float64)
    arrival = 20.0
    hold = 0
    uptime = 0
    effort = 0.0
    calls = []
    trace = []
    u = torch.tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        for k in range(10000):
            if k % 25 == 0:
                tic = time.perf_counter()
                controller_state = x.numpy().copy()
                # Match the coordinate convention used by the public C++ cost
                # and its supplied trajectory without changing the plant state.
                controller_state[0] = controller_state[0] % (2.0 * np.pi)
                controller_state[1] = (controller_state[1] + np.pi) % (2.0 * np.pi) - np.pi
                raw = controller.get_control_output(controller_state, k * 0.002)
                calls.append(time.perf_counter() - tic)
                u = torch.tensor(float(raw[0]), dtype=torch.float64).clamp(-0.5, 0.5)
            e = torch.cat((error(x), x[2:]))
            local = (e[:2].abs().max() < 0.3) and (e[2:].abs().max() < 1.5)
            local_u = (-(e.float() @ K.T).squeeze()).double().clamp(-0.5, 0.5)
            applied = local_u if local else u
            x = integrate(x[None, :], applied[None], 0.002)[0]
            inside = bool((error(x).abs().max() < 0.15) and (x[2:].abs().max() < 0.6))
            hold = hold + 1 if inside else 0
            if hold >= 250 and arrival == 20.0:
                arrival = (k + 1) * 0.002 - 0.5
            uptime += int(inside)
            effort += float(applied**2) * 0.002
            if k % 10 == 0:
                trace.append(np.r_[x.numpy(), float(applied)])
    return {
        "arrival": arrival,
        "success": arrival < 20.0,
        "uptime": uptime / 10000,
        "effort": effort,
        "trace": np.asarray(trace),
        "call_mean": float(np.mean(calls)),
        "call_p95": float(np.quantile(calls, 0.95)),
        "call_max": float(np.max(calls)),
    }


def run_set(states, horizon, terminal_energy, tag, max_iter=2):
    trials = []
    start = time.perf_counter()
    for index, state in enumerate(states):
        trial = run_trial(state, horizon, terminal_energy, max_iter=max_iter)
        trials.append(trial)
        np.savez_compressed(OUT / f"{tag}_{index}.npz", **trial)
    row = {
        "tag": tag,
        "horizon": horizon,
        "terminal_energy": terminal_energy,
        "max_iter": max_iter,
        "n": len(states),
        "success": int(sum(trial["success"] for trial in trials)),
        "arrival": float(np.mean([trial["arrival"] for trial in trials])),
        "effort": float(np.mean([trial["effort"] for trial in trials])),
        "call_mean": float(np.mean([trial["call_mean"] for trial in trials])),
        "call_p95_mean": float(np.mean([trial["call_p95"] for trial in trials])),
        "call_max": float(np.max([trial["call_max"] for trial in trials])),
        "wall_seconds": time.perf_counter() - start,
    }
    return row


