from screen import *
class AlignedRollout(CalibratedRollout):
 def action(self,x):
  n=len(x);nc=10;z=x.double().repeat_interleave(nc,0);pars=self.theta.repeat_interleave(nc,0)
  pulse=torch.tensor([0.,-1,-.5,0,.5,1,-1,1,-1,1],dtype=torch.float64).repeat(n)*.5;dur=torch.tensor([0,.05,.05,.05,.05,.05,.15,.15,.3,.3]).repeat(n)
  held=self.tail_action(x.double(),self.theta);uhold=held.repeat_interleave(nc);idx=torch.arange(nc).repeat(n);cost=torch.zeros(n*nc,dtype=torch.float64)
  grid=sorted(set(np.round(np.r_[np.arange(0,1.20001,.02),.05,.15,.3],10)))
  for t,tn in zip(grid[:-1],grid[1:]):
   u=self.tail_action(z,pars)
   if t<.05-1e-9:u=torch.where(idx==0,uhold,u)
   u=torch.where(t<dur-1e-9,pulse,u)
   e=torch.cat((error(z),z[:,2:]),-1);loc=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
   # Same priority as execution: the local stabilizer overrides a prefix.
   localu=(-(e.float()@K.T).squeeze(-1)).double().clamp(-.5,.5);u=torch.where(loc,localu,u)
   dt=tn-t
   for _ in range(2):z=integrate(z,u,pars,dt/2)
   cost+=dt*((1-torch.cos(error(z))).sum(-1)+.01*(z[:,2:]**2).sum(-1))
  cost+=self.terminal(z);best=cost.reshape(n,nc).argmin(-1)
  return torch.where(best==0,held,pulse.reshape(n,nc)[torch.arange(n),best]).float()
