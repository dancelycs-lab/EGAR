from pathlib import Path
import importlib.util
import argparse
import json
import sys
import numpy as np
import gain_trials

HERE = Path(__file__).resolve().parent
PEND = HERE.parent / 'pendubot_transfer'
spec = importlib.util.spec_from_file_location('pendubot_backend', PEND/'screen.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
torch, K, rhs, error = core.torch, core.K, core.rhs, core.error
STATE_PATH = PEND/'confirmation_states.npy'
OUT = HERE/'pendubot_step'

def features(x, u):
    q = x[..., 1]; k = .6255677234174437*.2*.25569305436052964
    d = .05086984812807257
    a = .031887199591513114+d+.6255677234174437*.2**2+2*k*torch.cos(q)
    b = d+k*torch.cos(q); det = a*d-b*b
    return torch.stack((torch.stack((torch.full_like(b,d), -b), -1)*u[..., None]/det[..., None],
        torch.stack((-d/det, b/det), -1)*x[..., 2, None]*.03,
        torch.stack((b/det, -a/det), -1)*x[..., 3, None]*.03), -1)

def derivative(x, u, theta):
    f = rhs(x, torch.zeros_like(u))
    f[..., 2:] += torch.einsum('...ij,...j->...i', features(x, u), theta)
    return f

def integrate(x, u, theta, dt):
    a=derivative(x,u,theta); b=derivative(x+dt*a/2,u,theta)
    c=derivative(x+dt*b/2,u,theta); d=derivative(x+dt*c,u,theta)
    return x+dt*(a+2*b+2*c+d)/6

class Controller:
    def __init__(self,n):
        self.theta=torch.tensor([1.,0.,0.],dtype=torch.float64).repeat(n,1)
    def action(self,x):
        n=len(x); nc=10; z=x.double().repeat_interleave(nc,0)
        pars=self.theta.repeat_interleave(nc,0)
        pulse=torch.tensor([0.,-1,-.5,0,.5,1,-1,1,-1,1],dtype=torch.float64).repeat(n)*.5
        duration=torch.tensor([0.,.05,.05,.05,.05,.05,.15,.15,.3,.3],dtype=torch.float64).repeat(n)
        index=torch.arange(nc).repeat(n); held=core.base(x.float()).double().clamp(-.5,.5)
        value=torch.zeros(n*nc,dtype=torch.float64)
        grid=sorted(set(np.round(np.r_[np.arange(0.,1.20001,.02),.05,.15,.3],10)))
        for t,tn in zip(grid[:-1],grid[1:]):
            u=core.base(z.float()).double().clamp(-.5,.5)
            if t<.05-1e-12: u=torch.where(index==0,held.repeat_interleave(nc),u)
            u=torch.where(t<duration-1e-12,pulse,u)
            e=torch.cat((error(z),z[:,2:]),-1)
            local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
            u=torch.where(local,(-(e.float()@K.T).squeeze(-1)).double().clamp(-.5,.5),u)
            dt=tn-t; z=integrate(z,u,pars,dt/2); z=integrate(z,u,pars,dt/2)
            value+=dt*((1-torch.cos(error(z))).sum(-1)+.01*(z[:,2:]**2).sum(-1))
        value+=core.terminal(z,'energy'); best=value.reshape(n,nc).argmin(-1)
        return torch.where(best==0,held,pulse.reshape(n,nc)[torch.arange(n),best])

def validate():
    # Import the original nominal rollout with its original Pendubot dependencies.
    previous=sys.modules['screen']; sys.modules['screen']=core
    try:
        spec=importlib.util.spec_from_file_location('original_pendubot_aligned',PEND/'aligned_confirm.py')
        ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    finally: sys.modules['screen']=previous
    rng=np.random.default_rng(731); x=torch.tensor(rng.uniform([-6,-6,-12,-12],[6,6,12,12],(100,4)))
    u=torch.tensor(rng.uniform(-.5,.5,100)); theta=torch.tensor([1.,0.,0.]).double().repeat(100,1)
    with torch.no_grad():
        err=float(abs(derivative(x,u,theta)-core.derivative(x,u)).max())
        xs=torch.tensor(np.load(STATE_PATH),dtype=torch.float32)
        actionerr=float(abs(Controller(64).action(xs)-ref.aligned_action(xs,.5,'energy')).max())
    assert err<1e-12 and actionerr<1e-12
    record=dict(nominal_derivative_max_abs=err,nominal_first_action_max_abs=actionerr,
                states=str(STATE_PATH),seeds=[51,52,53,54,55],change_time=1.,gain_before=1.,gain_after=.8)
    OUT.mkdir(exist_ok=True); (OUT/'validation.json').write_text(json.dumps(record,indent=2))
    print(record,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['nominal','dynamics','oracle'],default='dynamics')
    ap.add_argument('--seed',type=int,default=51); ap.add_argument('--validate',action='store_true')
    args=ap.parse_args()
    if args.validate: validate()
    else: gain_trials.run(args.mode,'step',args.seed,backend=sys.modules[__name__],output=OUT)
