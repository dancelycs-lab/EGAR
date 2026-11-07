from runtime import ASSETS, BUILD
from pathlib import Path
import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import Future
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'global_recovery'))
import acados_readiness
import double_pendulum.controller.acados_mpc.acados_mpc as vendor
import casadi as ca
from shared_model import Calibration, DT, K, rhs, step, noise_stream, validate, torch
from transfer_screen import Variant


def create(horizon, kind):
    name = f'p3_calibration_{kind}_H{horizon}_pid{os.getpid()}'.replace('.', 'p')
    pars = acados_readiness.model_parameters(filepath=str(ASSETS / "model_parameters.yml"))
    pars.set_torque_limit([0, .5]); pars.set_motor_inertia(0); pars.set_damping([0, 0]); pars.set_cfric([0, 0])
    c = vendor.AcadosMpc(model_pars=pars, generate_code_filename=name)
    c.set_start([0, 0, 0, 0]); c.set_goal([np.pi, 0, 0, 0])
    c.set_parameters(N_horizon=round(horizon/.025), prediction_horizon=horizon,
        Nlp_max_iter=40, max_solve_time=.01 if kind == 'rti' else .05,
        solver_type='SQP_RTI' if kind == 'rti' else 'SQP', wrap_angle=False,
        warm_start=kind!='warm', fallback_on_solver_fail=True, nonuniform_grid=False,
        cheating_on_inactive_joint=False, mpc_cycle_dt=.0025 if kind == 'rti' else .049,
        outer_cycle_dt=.002, qp_solver_tolerance=.001,
        qp_solver='PARTIAL_CONDENSING_HPIPM', hpipm_mode='ROBUST')
    c.set_velocity_constraints(v_max=30, v_final=30)
    c.set_cost_parameters(Q_mat=2*np.diag([100, 100, 10, 10]),
                         Qf_mat=2*np.diag([10000, 10000, 100, 100]), R_mat=2*np.diag([1e-6, 1e-6]))
    factory = vendor.AcadosOcpSolver
    def parameterized(ocp, **kw):
        model = ocp.model; theta = ca.SX.sym('actuation_parameters', 3)
        q2 = model.x[1]; v = model.x[2:]
        kap = .6255677234174437*.2*.25569305436052964; d = .05086984812807257
        a = .031887199591513114+d+.6255677234174437*.2**2+2*kap*ca.cos(q2)
        b = d+kap*ca.cos(q2); det = a*d-b*b
        visc = .03*ca.vertcat(theta[1]*v[0], theta[2]*v[1])
        accel_drag = ca.vertcat(d*visc[0]-b*visc[1], -b*visc[0]+a*visc[1])/det
        model.f_expl_expr = ca.substitute(model.f_expl_expr, model.u, theta[0]*model.u)-ca.vertcat(0, 0, accel_drag)
        model.f_impl_expr = model.xdot-model.f_expl_expr
        model.p = theta; ocp.parameter_values = np.array([1., 0., 0.])
        if kind == 'warm':
            # Exactly the EGAR state and physical-terminal objectives; negligible
            # input regularization (1e-6 u^2) keeps the public SQP well conditioned.
            q1=model.x[0]; q2=model.x[1]; v1=model.x[2]; v2=model.x[3]
            ga=(.5234602302310271*.2+.6255677234174437*.2)*9.81
            gb=.6255677234174437*.25569305436052964*9.81
            energy=.5*(a*v1*v1+2*b*v1*v2+d*v2*v2)-ga*ca.cos(q1)-gb*ca.cos(q1+q2)
            geometry=ca.vertcat(ca.sin(q1),ca.cos(q1),ca.sin(q2),ca.cos(q2),v1,v2)
            ocp.cost.cost_type='NONLINEAR_LS';ocp.cost.cost_type_e='NONLINEAR_LS'
            model.cost_y_expr=ca.vertcat(geometry,model.u)
            model.cost_y_expr_e=ca.vertcat(geometry,energy-ga-gb)
            ocp.cost.yref=np.array([0.,-1.,0.,1.,0.,0.,0.,0.])
            ocp.cost.yref_e=np.array([0.,-1.,0.,1.,0.,0.,0.])
            ocp.cost.W=np.diag([1.,1.,1.,1.,.02,.02,2e-6,2e-6])
            ocp.cost.W_e=np.diag([1.,1.,1.,1.,.2,.2,4.])
            for suffix in ['', '_e']:
                setattr(ocp.constraints,'idxbx'+suffix,np.array([],dtype=int))
                setattr(ocp.constraints,'lbx'+suffix,np.array([]))
                setattr(ocp.constraints,'ubx'+suffix,np.array([]))
            # Initial equality constraints are stored separately by acados.
            ocp.constraints.x0=c.x0
        fn = ca.Function('physical_check', [model.x, model.u, model.p], [model.f_expl_expr])
        rng = np.random.default_rng(7342); diffs = []
        for _ in range(50):
            x = rng.uniform([-6,-6,-12,-12], [6,6,12,12]); u = rng.uniform(-.5,.5)
            p = rng.uniform([.4,0,0], [1.6,2,2])
            diffs.append(float(abs(np.array(fn(x,[0,u],p)).ravel()-rhs(x,u,p)).max()))
        assert max(diffs) < 1e-9
        c.parameterized_dynamics_error = max(diffs)
        return factory(ocp, **kw)
    vendor.AcadosOcpSolver = parameterized
    work = BUILD/name; work.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd(); os.chdir(work)
    try:
        with (HERE/f'{name}_build.log').open('w') as log, contextlib.redirect_stdout(log): c.init()
    finally:
        vendor.AcadosOcpSolver = factory; os.chdir(old_cwd)
    if kind == 'rti':
        exc = c.async_mpc_future.exception()
        if isinstance(exc, TypeError):
            done = Future(); done.set_result(0); c.async_mpc_future = done
        elif exc is not None: raise exc
    if kind=='warm':
        c.seed_policy=Variant(local=True)
        c.v_max=float('inf')
    return c


def policy_warm_start(c, x, theta):
    z=x.copy();z[:2]=(z[:2]+2*np.pi)%(4*np.pi)-2*np.pi
    for j in range(c.N_horizon):
        c.ocp_solver.set(j,'x',z)
        with torch.no_grad():
            u=float(c.seed_policy.action(torch.tensor(z[None],dtype=torch.float32)).clamp(-.5,.5)[0])
        c.ocp_solver.set(j,'u',np.array([0.,u]))
        for _ in range(2):z=step(z,u,theta,c.prediction_horizon/c.N_horizon/2)
    c.ocp_solver.set(c.N_horizon,'x',z)


def set_parameters(c, theta, kind):
    if kind == 'rti': c.async_mpc_future.result()
    for j in range(c.N_horizon+1): c.ocp_solver.set(j, 'p', theta)
    if kind == 'rti':
        c.ocp_solver.options_set('rti_phase', 1); c.ocp_solver.solve()
        done = Future(); done.set_result(0); c.async_mpc_future = done


def simulate(c, state, noise, adaptive, kind):
    x = state.copy(); cal = Calibration(x+noise[0], adaptive)
    arrival = 20.; hold = 0; effort = 0.; iae = 0.; calls = []; statuses = {}
    trace = [np.r_[x, 0.]]; estimates = [cal.theta.copy()]
    previous_error = abs((x[:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi).sum()
    for k in range(10000):
        if k % 100 == 0: set_parameters(c, cal.theta, kind)
        if kind == 'rti' or k % 25 == 0:
            tic = time.perf_counter()
            if kind=='warm':policy_warm_start(c,cal.xh,cal.theta)
            output = c.get_control_output(cal.xh, k*DT)
            calls.append(time.perf_counter()-tic)
            assert abs(output[0]) < 1e-8
            u = float(output[1])
            if kind == 'rti': c.async_mpc_future.result()
            status = int(c.ocp_solver.get_status()); statuses[status] = statuses.get(status,0)+1
        applied = u
        if kind != 'rti':
            e = np.r_[(cal.xh[:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi, cal.xh[2:]]
            if abs(e[:2]).max() < .3 and abs(e[2:]).max() < 1.5:
                applied = float(np.clip(-(e.astype(np.float32)@K.numpy().T).squeeze(), -.5, .5))
        assert np.isfinite(applied) and abs(applied) <= .500001
        xn = step(x, applied, np.array([5/6, 0., 0.]))
        cal.update(applied, xn+noise[k+1], k); x = xn
        eq = (x[:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi
        inside = abs(eq).max() < .15 and abs(x[2:]).max() < .6
        hold = hold+1 if inside else 0
        if hold >= 250 and arrival == 20.: arrival = (k+1)*DT-.5
        effort += applied**2*DT; current_error = abs(eq).sum()
        iae += .5*DT*(previous_error+current_error); previous_error = current_error
        if (k+1) % 10 == 0:
            trace.append(np.r_[x, applied]); estimates.append(cal.theta.copy())
    return dict(initial_states=state, arrival=arrival, success=arrival<20, effort=effort,
                angle_iae=iae, trace=np.asarray(trace), theta=np.asarray(estimates),
                time=np.arange(1001)*.02, call_seconds=np.asarray(calls)), statuses


def run(args):
    if args.split == 'development':
        states = np.load(HERE.parent/'global_recovery/exploratory_initial_states.npy')[:8]
        seed = 7401
    else:
        states = np.load(HERE.parent/'authority_calibration/gain_confirmation_states.npy'); seed = 51
    noise = noise_stream(seed, len(states))
    c = create(args.horizon, args.kind)
    xx = [c.ocp_solver.get(i,'x').copy() for i in range(c.N_horizon+1)]
    uu = [c.ocp_solver.get(i,'u').copy() for i in range(c.N_horizon)]
    mode = 'adaptive' if args.adaptive else 'fixed'
    prefix = f'nmpc_{args.kind}_{args.split}_H{args.horizon}_{mode}'
    for i in range(args.start, min(args.stop, len(states))):
        path = HERE/f'{prefix}_{i:02d}.npz'
        if path.exists(): continue
        if args.kind == 'rti': c.async_mpc_future.result()
        c.ocp_solver.reset()
        for j, value in enumerate(xx): c.ocp_solver.set(j, 'x', value)
        for j, value in enumerate(uu): c.ocp_solver.set(j, 'u', value)
        set_parameters(c, np.array([1.,0.,0.]), args.kind)
        c.last_mpc_run_t = -float('inf'); c.last_good_solution_time = -float('inf')
        c.last_u[:] = 0; c.mpc_u[:] = 0
        tic = time.perf_counter()
        result, status = simulate(c, states[i], noise[:,i], args.adaptive, args.kind)
        assert np.isfinite(result['trace']).all()
        np.savez_compressed(path, **result)
        row = dict(index=i, mode=mode, kind=args.kind, split=args.split, horizon=args.horizon,
                   success=int(result['success']), arrival=result['arrival'], effort=result['effort'],
                   angle_iae=result['angle_iae'], status_counts=status,
                   parameterized_dynamics_max_abs=c.parameterized_dynamics_error,
                   call_mean=float(result['call_seconds'].mean()), seconds=time.perf_counter()-tic)
        path.with_suffix('.json').write_text(json.dumps(row, indent=2)); print(row, flush=True)
    c.spinner.shutdown(wait=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--kind', choices=['rti','sqp','warm'], default='rti')
    ap.add_argument('--horizon', type=float, default=1.2)
    ap.add_argument('--split', choices=['development','confirmation'], default='confirmation')
    ap.add_argument('--adaptive', action='store_true'); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, default=64); ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    if args.validate:
        record = validate(); (HERE/'shared_calibration_validation.json').write_text(json.dumps(record, indent=2)); print(record)
    else: run(args)
