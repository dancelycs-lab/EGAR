from pathlib import Path
import sys
import json
import itertools
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'global_recovery'
sys.path.insert(0, str(BASE))
from transfer_screen import torch, np, K, p, error, step
from pfl_screen import PFL


def mechanics(x):
    q1,q2,v1,v2=x.unbind(-1)
    b=.6255677234174437*.2*.25569305436052964
    d=.05086984812807257
    a=.031887199591513114+d+.6255677234174437*.2**2+2*b*torch.cos(q2)
    off=d+b*torch.cos(q2)
    ep=(.5234602302310271*.2+.6255677234174437*.2)*9.81
    ee=.6255677234174437*.25569305436052964*9.81
    g0=-ep*torch.sin(q1)-ee*torch.sin(q1+q2)
    g1=-ee*torch.sin(q1+q2)
    h0=-b*torch.sin(q2)*(2*v1*v2+v2*v2)
    h1=b*torch.sin(q2)*v1*v1
    energy=.5*(a*v1*v1+2*off*v1*v2+d*v2*v2)-ep*torch.cos(q1)-ee*torch.cos(q1+q2)
    return a,off,d,a*d-off*off,h0,h1,g0,g1,energy-ep-ee


class Xin:
    def __init__(self, gains):
        self.kp,self.kd,self.kv=torch.as_tensor(gains,dtype=torch.float64).unbind(-1)
    def action(self,x):
        a,b,d,det,h0,h1,g0,g1,de=mechanics(x)
        q2=(x[...,1]+np.pi)%(2*np.pi)-np.pi
        return -((self.kv*x[...,3]+self.kp*q2)*det+self.kd*(b*(h0-g0)-a*(h1-g1)))/(self.kd*a+de*det)


def simulate(controller, states, save_trace=True):
    x=torch.tensor(states,dtype=torch.float64)
    n=len(x); arrival=np.full(n,20.); hold=np.zeros(n,dtype=int); effort=np.zeros(n)
    trace=[np.c_[x.numpy(),np.zeros(n)]] if save_trace else []
    with torch.no_grad():
        for k in range(10000):
            u=controller.action(x).double().clamp(-.5,.5)
            e=torch.cat((error(x),x[:,2:]),-1)
            local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
            ul=(-(e.float()@K.T).squeeze(-1)).double().clamp(-.5,.5)
            u=torch.where(local,ul,u)
            assert torch.isfinite(u).all(), 'Nonfinite public-controller command'
            x=step(x,u)
            inside=((error(x).abs().amax(-1)<.15)&(x[:,2:].abs().amax(-1)<.6)).numpy()
            hold=np.where(inside,hold+1,0)
            arrival=np.where((hold>=250)&(arrival==20.),(k+1)*.002-.5,arrival)
            effort+=u.numpy()**2*.002
            if save_trace and (k+1)%10==0: trace.append(np.c_[x.numpy(),u.numpy()])
    result=dict(initial_states=states,arrival=arrival,success=arrival<20,effort=effort)
    if save_trace:result.update(trace=np.array(trace),time=np.arange(1001)*.02)
    return result


def summary(r):
    return dict(n=len(r['arrival']),success=int(r['success'].sum()),time=float(r['arrival'].mean()),effort=float(r['effort'].mean()))




