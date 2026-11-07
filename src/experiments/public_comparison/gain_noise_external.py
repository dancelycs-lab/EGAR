from pathlib import Path
import sys
import json
import contextlib
import time
from concurrent.futures import Future

HERE = Path(__file__).resolve().parent
AUTH = HERE.parent/'authority_calibration'
sys.path.insert(0, str(AUTH))
from screen import torch, np, derivative, integrate, K, p

DT = .002
R = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
Q = np.diag([1e-12, 1e-12, 1e-7, 1e-7])
I4 = np.eye(4)


def rhs(x, u, gamma=1.):
    q1, q2, v1, v2 = np.moveaxis(x, -1, 0)
    kappa = .6255677234174437*.2*.25569305436052964
    d = .05086984812807257
    a = .031887199591513114+d+.6255677234174437*.2**2+2*kappa*np.cos(q2)
    b = d+kappa*np.cos(q2)
    ga = (.5234602302310271*.2+.6255677234174437*.2)*9.81
    gb = .6255677234174437*.25569305436052964*9.81
    f0 = -ga*np.sin(q1)-gb*np.sin(q1+q2)+kappa*np.sin(q2)*(2*v1*v2+v2*v2)
    f1 = -gb*np.sin(q1+q2)-kappa*np.sin(q2)*v1*v1+gamma*u
    det = a*d-b*b
    return np.stack([v1, v2, (d*f0-b*f1)/det, (-b*f0+a*f1)/det], -1)


def step(x, u, gamma=1.):
    a = rhs(x, u, gamma); b = rhs(x+DT*a/2, u, gamma)
    c = rhs(x+DT*b/2, u, gamma); d = rhs(x+DT*c, u, gamma)
    return x+DT*(a+2*b+2*c+d)/6


def observe(xh, cov, u, yn):
    perturb = I4*1e-5
    jac = np.swapaxes((rhs(xh[..., None, :]+perturb, u[..., None])
                      -rhs(xh[..., None, :]-perturb, u[..., None]))/(2e-5), -1, -2)
    F = I4+DT*jac
    cov = F@cov@np.swapaxes(F, -1, -2)+Q
    gain = np.swapaxes(np.linalg.solve(cov+R, np.swapaxes(cov, -1, -2)), -1, -2)
    pred = step(xh, u)
    nxt = pred+np.einsum('...ij,...j->...i', gain, yn-pred)
    J = I4-gain
    cov = J@cov@np.swapaxes(J, -1, -2)+gain@R@np.swapaxes(gain, -1, -2)
    return nxt, cov


def validate():
    rng = np.random.default_rng(7241)
    x = rng.uniform([-6,-6,-12,-12], [6,6,12,12], (128,4))
    u = rng.uniform(-.5,.5,128)
    errors = []
    for gamma in (1., 5/6):
        tx = torch.tensor(x); tu = torch.tensor(u)
        theta = torch.tensor([gamma,0.,0.],dtype=torch.float64).repeat(len(x),1)
        errors.append(float(np.max(abs(rhs(x,u,gamma)-derivative(tx,tu,theta).numpy()))))
        errors.append(float(np.max(abs(step(x,u,gamma)-integrate(tx,tu,theta,DT).numpy()))))
    # The existing estimator uses the same finite-difference nominal Jacobian.
    tx=torch.tensor(x);tu=torch.tensor(u);theta=torch.tensor([1.,0.,0.]).double().repeat(len(x),1)
    yn=x+rng.normal(size=x.shape)*[.001,.001,.01,.01]
    ti=torch.eye(4,dtype=torch.float64);tr=torch.tensor(R);tq=torch.tensor(Q)
    tcov=tr.repeat(len(x),1,1);delta=ti[None]*1e-5
    jac=((derivative(tx[:,None,:]+delta,tu[:,None],theta[:,None,:])
         -derivative(tx[:,None,:]-delta,tu[:,None],theta[:,None,:]))/2e-5).transpose(-1,-2)
    F=ti+DT*jac;tcov=F@tcov@F.transpose(-1,-2)+tq
    gain=torch.linalg.solve(tcov+tr,tcov.transpose(-1,-2)).transpose(-1,-2)
    pred=integrate(tx,tu,theta,DT)
    ref=pred+torch.einsum('nij,nj->ni',gain,torch.tensor(yn)-pred)
    J=ti-gain;pc=J@tcov@J.transpose(-1,-2)+gain@tr@gain.transpose(-1,-2)
    got,gcov=observe(x,np.broadcast_to(R,(len(x),4,4)).copy(),u,yn)
    from transfer_screen import Variant
    with torch.no_grad():
        policy_ref=Variant(local=True).action(torch.tensor(x,dtype=torch.float32)).clamp(-.5,.5).numpy()
    record=dict(dynamics_max_abs=max(errors),observer_state_max_abs=float(np.max(abs(got-ref.numpy()))),
                observer_covariance_max_abs=float(np.max(abs(gcov-pc.numpy()))),
                frozen_policy_command_max_abs=float(np.max(abs(policy_action(x,0)-policy_ref))))
    assert max(record.values())<1e-8,record
    (HERE/'gain_noise_source_validation.json').write_text(json.dumps(record,indent=2))


def noise_stream():
    gen=torch.Generator().manual_seed(51)
    sigma=torch.tensor([.001,.001,.01,.01],dtype=torch.float64)
    return np.stack([(torch.randn((64,4),generator=gen,dtype=torch.float64)*sigma).numpy()
                     for _ in range(10001)])


def simulate(states, noises, action):
    n=len(states);x=states.copy();xh=x+noises[0];cov=np.broadcast_to(R,(n,4,4)).copy()
    arrival=np.full(n,20.);hold=np.zeros(n,dtype=int);effort=np.zeros(n)
    trace=[np.c_[x,np.zeros(n)]]
    for k in range(10000):
        u=np.asarray(action(xh,k),dtype=float).reshape(n)
        assert np.isfinite(u).all() and np.max(abs(u))<=.500001
        xn=step(x,u,5/6);yn=xn+noises[k+1]
        xh,cov=observe(xh,cov,u,yn);x=xn
        eq=(x[:,:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi
        ok=(abs(eq).max(-1)<.15)&(abs(x[:,2:]).max(-1)<.6)
        hold=np.where(ok,hold+1,0)
        arrival=np.where((hold>=250)&(arrival==20.),(k+1)*DT-.5,arrival)
        effort+=u*u*DT
        if (k+1)%10==0:trace.append(np.c_[x,u])
    return dict(initial_states=states,arrival=arrival,success=arrival<20,effort=effort,
                trace=np.asarray(trace),time=np.arange(1001)*.02)


def policy_action(xh,k):
    x=torch.tensor(xh,dtype=torch.float32)
    with torch.no_grad():
        u=p.action(x)
        eq=(x[:,:2]-torch.tensor([np.pi,0])+np.pi)%(2*np.pi)-np.pi
        e=torch.cat([eq,x[:,2:]],-1)
        loc=(eq.abs().amax(-1)<.3)&(x[:,2:].abs().amax(-1)<1.5)
        u=torch.where(loc,-(e@K.T).squeeze(-1),u).clamp(-.5,.5)
    return u.numpy()


def run():
    validate();states=np.load(AUTH/'gain_confirmation_states.npy');noise=noise_stream()
    policy_file=HERE/'gain_noise_policy.npz'
    if not policy_file.exists():
        start=time.perf_counter();result=simulate(states,noise,policy_action)
        np.savez_compressed(policy_file,**result)
        print('policy',int(result['success'].sum()),float(result['arrival'].mean()),
              'seconds',time.perf_counter()-start,flush=True)
    from acados_readiness import create
    with open(HERE/'gain_noise_acados_initialization.log','w') as log,contextlib.redirect_stdout(log):
        controller=create(.5,1.2)
    exc=controller.async_mpc_future.exception()
    if isinstance(exc,TypeError):
        done=Future();done.set_result(0);controller.async_mpc_future=done
    elif exc is not None:raise exc
    xx=[controller.ocp_solver.get(i,'x').copy() for i in range(controller.N_horizon+1)]
    uu=[controller.ocp_solver.get(i,'u').copy() for i in range(controller.N_horizon)]
    rows=[];results=[]
    for i in range(64):
        path=HERE/f'gain_noise_acados_episode_{i:02d}.npz'
        if path.exists():
            with np.load(path) as z:result={k:z[k] for k in z.files}
            row=json.loads(path.with_suffix('.json').read_text())
        else:
            controller.async_mpc_future.result();controller.ocp_solver.reset()
            for j,v in enumerate(xx):controller.ocp_solver.set(j,'x',v)
            for j,v in enumerate(uu):controller.ocp_solver.set(j,'u',v)
            controller.last_mpc_run_t=-float('inf');controller.last_good_solution_time=-float('inf')
            controller.last_u[:]=0;controller.mpc_u[:]=0;statuses={}
            def action(xh,k):
                u=controller.get_control_output(xh[0],k*DT)
                assert abs(u[0])<1e-8
                status=int(controller.ocp_solver.get_status());statuses[status]=statuses.get(status,0)+1
                return np.array([u[1]])
            start=time.perf_counter();result=simulate(states[i:i+1],noise[:,i:i+1],action)
            row=dict(index=i,success=int(result['success'][0]),arrival=float(result['arrival'][0]),
                     effort=float(result['effort'][0]),seconds=time.perf_counter()-start,status_counts=statuses)
            np.savez_compressed(path,**result);path.with_suffix('.json').write_text(json.dumps(row,indent=2))
        results.append(result);rows.append(row)
        (HERE/'gain_noise_acados_progress.json').write_text(json.dumps(rows,indent=2))
        print('acados',row,flush=True)
    controller.spinner.shutdown(wait=True)
    out={k:np.concatenate([r[k] for r in results],axis=0) for k in ['initial_states','arrival','success','effort']}
    out.update(trace=np.concatenate([r['trace'] for r in results],axis=1),time=results[0]['time'])
    np.testing.assert_array_equal(out['initial_states'],states)
    np.savez_compressed(HERE/'gain_noise_acados.npz',**out)
    print('COMPLETE',int(out['success'].sum()),float(out['arrival'].mean()),flush=True)


if __name__=='__main__':run()
