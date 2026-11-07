from runtime import ASSETS, BUILD
import contextlib
import os
from pathlib import Path
import sys
import json
import time

HERE=Path(__file__).resolve().parent
PEND=HERE.parent/'pendubot_transfer'
sys.path.insert(0,str(HERE.parent/'global_recovery'))
from acados_readiness import np, model_parameters, AcadosMpc
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


def params():
    p=model_parameters(filepath=str(ASSETS / "model_parameters.yml"))
    p.set_torque_limit([.5,0]);p.set_motor_inertia(0);p.set_damping([0,0]);p.set_cfric([0,0]);return p


def create(horizon):
    tag=f'pendubot_L0p5_H{horizon}'.replace('.','p')
    c=AcadosMpc(model_pars=params(),generate_code_filename=tag)
    c.set_start([0,0,0,0]);c.set_goal([np.pi,0,0,0])
    c.set_parameters(N_horizon=round(horizon/.025),prediction_horizon=horizon,Nlp_max_iter=40,
       max_solve_time=.01,solver_type='SQP_RTI',wrap_angle=False,warm_start=True,
       fallback_on_solver_fail=True,nonuniform_grid=False,cheating_on_inactive_joint=False,
       mpc_cycle_dt=.0025,outer_cycle_dt=.002,qp_solver_tolerance=.001,
       qp_solver='PARTIAL_CONDENSING_HPIPM',hpipm_mode='ROBUST')
    c.set_velocity_constraints(v_max=30,v_final=30)
    c.set_cost_parameters(Q_mat=2*np.diag([100,100,10,10]),Qf_mat=2*np.diag([10000,10000,100,100]),R_mat=2*np.diag([1e-6,1e-6]))
    work=BUILD/tag;work.mkdir(parents=True,exist_ok=True);os.chdir(work)
    with open(HERE/f'{tag}_initialization.log','w') as log,contextlib.redirect_stdout(log):c.init()
    # Upstream initializes submit(self.ocp_solver.solve()), eagerly completing
    # preparation and then submitting an integer. Its own get_control_output
    # ignores this already-done failed future. Preserve the completed solver
    # preparation while replacing only that bookkeeping future.
    from concurrent.futures import Future
    initial_exception=c.async_mpc_future.exception()
    if isinstance(initial_exception,TypeError):
        done=Future();done.set_result(0);c.async_mpc_future=done
    elif initial_exception is not None:
        raise initial_exception
    return c


def rhs(x,u):
    q1,q2,v1,v2=x
    b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257
    a=.031887199591513114+d+.6255677234174437*.2**2+2*b*np.cos(q2);off=d+b*np.cos(q2)
    ga=(.5234602302310271*.2+.6255677234174437*.2)*9.81;gb=.6255677234174437*.25569305436052964*9.81
    f0=-ga*np.sin(q1)-gb*np.sin(q1+q2)+b*np.sin(q2)*(2*v1*v2+v2*v2)+u[0]
    f1=-gb*np.sin(q1+q2)-b*np.sin(q2)*v1*v1
    det=a*d-off*off
    return np.array([v1,v2,(d*f0-off*f1)/det,(-off*f0+a*f1)/det])


def run(states,horizon,tag):
    c=create(horizon);c.async_mpc_future.result()
    xx=[c.ocp_solver.get(i,'x').copy() for i in range(c.N_horizon+1)]
    uu=[c.ocp_solver.get(i,'u').copy() for i in range(c.N_horizon)]
    rows=[];traces=[]
    for index,x0 in enumerate(states):
        c.async_mpc_future.result();c.ocp_solver.reset()
        for i,v in enumerate(xx):c.ocp_solver.set(i,'x',v)
        for i,v in enumerate(uu):c.ocp_solver.set(i,'u',v)
        c.last_mpc_run_t=-float('inf');c.last_good_solution_time=-float('inf');c.last_u[:]=0;c.mpc_u[:]=0
        x=x0.copy();hold=0;arrival=20.;effort=0.;tr=[];statuses={};start=time.perf_counter()
        for k in range(10000):
            u=c.get_control_output(x,k*.002)
            assert abs(u[1])<1e-8 and max(abs(u))<=.5000001
            a=rhs(x,u);b=rhs(x+.001*a,u);d=rhs(x+.001*b,u);e=rhs(x+.002*d,u)
            x+=.002*(a+2*b+2*d+e)/6
            error=(x[:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi
            inside=max(abs(error))<.15 and max(abs(x[2:]))<.6
            hold=hold+1 if inside else 0
            if hold>=250 and arrival==20.:arrival=(k+1)*.002-.5
            effort+=float(u@u)*.002
            status=int(c.ocp_solver.get_status());statuses[status]=statuses.get(status,0)+1
            if k%10==0:tr.append(np.r_[x,u[0]])
        row=dict(index=index,arrival=arrival,success=arrival<20,effort=effort,status_counts=statuses,wall_seconds=time.perf_counter()-start)
        rows.append(row);traces.append(tr)
        (HERE/f'{tag}_progress.json').write_text(json.dumps(rows,indent=2))
        print(tag,row,flush=True)
    c.spinner.shutdown(wait=True)
    out={key:np.array([r[key] for r in rows]) for key in ['arrival','success','effort']}
    out.update(trace=np.array(traces).transpose(1,0,2),initial_states=states,time=.002+np.arange(1000)*.02)
    np.savez_compressed(HERE/f'{tag}.npz',**out)
    return dict(horizon=horizon,n=len(states),success=int(out['success'].sum()),time=float(out['arrival'].mean()),effort=float(out['effort'].mean()))


