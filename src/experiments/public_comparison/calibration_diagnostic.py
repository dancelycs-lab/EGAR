from pathlib import Path
import sys
import json

HERE=Path(__file__).resolve().parent
A=HERE.parent/'authority_calibration'
sys.path.insert(0,str(A))
from aligned_rollout import AlignedRollout,torch,np,integrate,error,K


def costs(x,theta):
    n=len(x);nc=10;c=AlignedRollout(n);c.theta=theta
    z=x.double().repeat_interleave(nc,0);pars=theta.repeat_interleave(nc,0)
    pulse=torch.tensor([0.,-1,-.5,0,.5,1,-1,1,-1,1],dtype=torch.float64).repeat(n)*.5
    dur=torch.tensor([0,.05,.05,.05,.05,.05,.15,.15,.3,.3]).repeat(n)
    held=c.tail_action(x.double(),theta);uhold=held.repeat_interleave(nc)
    idx=torch.arange(nc).repeat(n);value=torch.zeros(n*nc,dtype=torch.float64)
    grid=sorted(set(np.round(np.r_[np.arange(0,1.20001,.02),.05,.15,.3],10)))
    with torch.no_grad():
        for t,tn in zip(grid[:-1],grid[1:]):
            u=c.tail_action(z,pars)
            if t<.05-1e-9:u=torch.where(idx==0,uhold,u)
            u=torch.where(t<dur-1e-9,pulse,u)
            e=torch.cat((error(z),z[:,2:]),-1)
            loc=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
            localu=(-(e.float()@K.T).squeeze(-1)).double().clamp(-.5,.5)
            u=torch.where(loc,localu,u);dt=tn-t
            for _ in range(2):z=integrate(z,u,pars,dt/2)
            value+=dt*((1-torch.cos(error(z))).sum(-1)+.01*(z[:,2:]**2).sum(-1))
        value+=c.terminal(z)
    return value.reshape(n,nc)


if __name__=='__main__':
    d=np.load(A/'final_aligned_gain_dynamics.npz');ix=np.arange(0,201,10)
    x=torch.tensor(d['trace'][ix,:,:4].reshape(-1,4),dtype=torch.float32)
    adapted=torch.tensor(d['theta'][ix].reshape(-1,3),dtype=torch.float64)
    nominal=torch.tensor([1.,0,0],dtype=torch.float64).repeat(len(x),1)
    truth=torch.tensor(d['true_theta'],dtype=torch.float64).repeat(len(x),1)
    reference=costs(x,truth).numpy();nom=costs(x,nominal).numpy();adp=costs(x,adapted).numpy()
    assert np.isfinite(reference).all() and np.isfinite(nom).all() and np.isfinite(adp).all()
    # Equivalence to deployed selection on 12 fixed snapshots.
    with torch.no_grad():
        test=AlignedRollout(12);test.theta=adapted[:12]
        actual=test.action(x[:12]).numpy()
        chosen=np.argmin(adp[:12],1);pulse=np.array([0.,-1,-.5,0,.5,1,-1,1,-1,1])*.5
        expected=np.where(chosen==0,test.tail_action(x[:12].double(),adapted[:12]).numpy(),pulse[chosen])
        assert np.max(abs(actual-expected))<1e-6
    reference=reference.reshape(21,64,10);nom=nom.reshape(21,64,10);adp=adp.reshape(21,64,10)
    np.savez_compressed(HERE/'calibration_diagnostic.npz',time=.002+.02*ix,reference_cost=reference,
                        nominal_cost=nom,adaptive_cost=adp,source_indices=ix)
    out={}
    for name,c in [('nominal',nom),('adaptive',adp)]:
        selected=np.argmin(c,axis=-1)
        regret=np.take_along_axis(reference,selected[...,None],axis=-1)[...,0]-reference.min(-1)
        mae=abs(c-reference).mean(-1)
        out[name]=dict(mean_candidate_cost_error=mae.mean(1).tolist(),mean_selection_excess=regret.mean(1).tolist(),
                       minimizer_agreement=(selected==reference.argmin(-1)).mean(1).tolist(),
                       pooled_mean_selection_excess=float(regret.mean()),pooled_mean_candidate_cost_error=float(mae.mean()))
    (HERE/'calibration_diagnostic_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
