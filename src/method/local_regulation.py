"""Deterministic illustrative local trajectories; proof is the interval certificate."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
d=np.load(HERE.parent/'experiments/global_recovery/lqr_matrices.npz')
P=d['P']; K=d['K'].ravel().astype(float)
cert=json.loads((HERE/'interval_sampled_certificate.json').read_text())
assert cert['certificate']
r=cert['P_norm_radius']; rng=np.random.default_rng(99318)
z=rng.normal(size=(64,4)); z/=np.linalg.norm(z,axis=1)[:,None]
e0=np.linalg.solve(np.linalg.cholesky(P).T,(r*z).T).T
gains=np.array([.8,1.,1.25]); results=[];all_states=[]

def rhs(x,u,gamma):
    q1,q2,v1,v2=x.T
    b=.6255677234174437*.2*.25569305436052964; d=.05086984812807257
    a=.031887199591513114+d+.6255677234174437*.2**2+2*b*np.cos(q2)
    off=d+b*np.cos(q2);det=a*d-off**2
    ga=(.5234602302310271*.2+.6255677234174437*.2)*9.81
    gb=.6255677234174437*.25569305436052964*9.81
    f0=-ga*np.sin(q1)-gb*np.sin(q1+q2)+b*np.sin(q2)*(2*v1*v2+v2*v2)
    f1=-gb*np.sin(q1+q2)-b*np.sin(q2)*v1*v1+gamma*u
    return np.column_stack((v1,v2,(d*f0-off*f1)/det,(-off*f0+a*f1)/det))

for gamma in gains:
    goal=np.array([np.pi,0,0,0]);x=e0+goal;states=[x.copy()];values=[np.ones(64)]
    for _ in range(1000):
        u=-(x-goal)@K
        assert abs(u).max()<.5
        # Two substeps inside each exact command hold; command remains fixed.
        for _ in range(2):
            h=.001;a=rhs(x,u,gamma);b=rhs(x+h*a/2,u,gamma);c=rhs(x+h*b/2,u,gamma);dd=rhs(x+h*c,u,gamma)
            x+=h*(a+2*b+2*c+dd)/6
        e=x-goal;values.append(np.einsum('ni,ij,nj->n',e,P,e)/r**2);states.append(x.copy())
    results.append(values);all_states.append(states)
results=np.array(results)
assert np.all(results <= (1-.0035/2)**np.arange(1001)[None,:,None]+1e-8)
np.savez_compressed(HERE/'local_regulation.npz',time=np.arange(1001)*.002,
                   gains=gains,initial_error=e0,states=all_states,normalized_V=results)
print('Local illustration: 3 gains x 64 directions; analytical envelope respected.')
