import sys, json, time
from pathlib import Path

import numpy as np
import torch
from policy import Policy
torch.set_num_threads(1)
ROOT=Path(__file__).parent

def rhs(x,u,scale=1.):
 q1,q2,v1,v2=x.unbind(-1)
 a=.031887199591513114+.05086984812807257+.6255677234174437*.2**2
 b=.6255677234174437*.2*.25569305436052964
 d=.05086984812807257
 m00=a+2*b*torch.cos(q2);m01=d+b*torch.cos(q2)
 g0=-(.5234602302310271*.2+.6255677234174437*.2)*9.81*torch.sin(q1)-.6255677234174437*.25569305436052964*9.81*torch.sin(q1+q2)
 g1=-.6255677234174437*.25569305436052964*9.81*torch.sin(q1+q2)
 f0=g0+b*torch.sin(q2)*(2*v1*v2+v2*v2)
 f1=g1-b*torch.sin(q2)*v1*v1+u/scale
 det=m00*d-m01*m01
 return torch.stack((v1,v2,(d*f0-m01*f1)/det,(m00*f1-m01*f0)/det),-1)
def step(x,u,dt=.002,scale=1.):
 a=rhs(x,u,scale);b=rhs(x+dt*a/2,u,scale);c=rhs(x+dt*b/2,u,scale);d=rhs(x+dt*c,u,scale)
 return x+dt*(a+2*b+2*c+d)/6

def error(x):
 q=(x[...,:2]-torch.tensor([np.pi,0.],dtype=x.dtype)+np.pi)%(2*np.pi)-np.pi
 return q

def run(policy,initial,limit=6.,mode='clip',velocity_scale=1.,duration=12.,dt=.002,mass_scale=1.):
 x=torch.tensor(initial,dtype=torch.float64);n=len(x);hold=torch.zeros(n,dtype=torch.int64);arrival=torch.full((n,),duration,dtype=torch.float64);up=torch.zeros(n);eff=torch.zeros(n);maxv=torch.zeros(n);history=[]
 t0=time.perf_counter()
 with torch.no_grad():
  for k in range(round(duration/dt)):
   z=x.float().clone();z[:,2:]*=velocity_scale
   u=policy.action(z).double()
   if mode=='scale':u=u*limit/6
   u=u.clamp(-limit,limit)
   x=step(x,u,dt,mass_scale)
   ok=(error(x).abs().amax(-1)<.15)&(x[:,2:].abs().amax(-1)<.6)
   hold=torch.where(ok,hold+1,0);arrival=torch.where((hold>=round(.5/dt))&(arrival==duration),(k+1)*dt-.5,arrival)
   up+=ok;eff+=(u*u*dt).float();maxv=torch.maximum(maxv,x[:,2:].abs().amax(-1).float())
   if k%10==0:history.append(torch.cat((x,u[:,None]),-1).numpy())
 return {'arrival':arrival.numpy(),'uptime':up.numpy()/(duration/dt),'effort':eff.numpy(),'maxv':maxv.numpy(),'success':(arrival<duration).numpy(),'final_error':error(x).numpy(),'trace':np.asarray(history),'seconds':time.perf_counter()-t0}


