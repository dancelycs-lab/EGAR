from batch_screen import *
from scipy.linalg import solve_continuous_are
p=Policy();goal=torch.tensor([np.pi,0,0,0],dtype=torch.float64)
A=torch.autograd.functional.jacobian(lambda x:rhs(x,torch.tensor(0.)),goal).numpy();B=torch.autograd.functional.jacobian(lambda u:rhs(goal,u),torch.tensor(0.,dtype=torch.float64)).numpy()[:,None]
P=solve_continuous_are(A,B,np.diag([10,10,1,1]),np.array([[.1]]));K=torch.tensor(np.linalg.solve(np.array([[.1]]),B.T@P),dtype=torch.float32)
class Variant:
 def __init__(self,gain=1.,vel=1.,local=False,mirror=False,anchor=False):self.gain=gain;self.vel=vel;self.local=local;self.mirror=mirror;self.anchor=anchor
 def action(self,x):
  z=x.clone();z[:,2:]*=self.vel
  u=p.action(z)
  if self.mirror:u=(u-p.action(-z))*.5
  if self.anchor:
   e=error(z);w=torch.exp(-torch.sum(e**2,dim=-1)/.25)
   u=u-.29236192*w
  u=u*self.gain
  if self.local:
   e=torch.cat((error(x),x[:,2:]),-1);local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
   ul=-(e@K.T).squeeze(-1);u=torch.where(local,ul,u)
  return u
