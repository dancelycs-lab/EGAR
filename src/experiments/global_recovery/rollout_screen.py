from transfer_screen import *
class Rollout:
 def __init__(self,limit=.5,horizon=1.2,objective='physical'):
  self.limit=limit;self.horizon=horizon;self.objective=objective;self.base=Variant(local=True);self.calls=0
 def terminal(self,x):
  e=error(x);v=x[:,2:]
  pos=(1-torch.cos(e)).sum(-1)
  if self.objective=='critic':return -p.value(x.float())
  if self.objective=='physical':return 10*pos+.2*(v*v).sum(-1)
  b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257
  m00=.031887199591513114+d+.6255677234174437*.2**2+2*b*torch.cos(x[:,1]);m01=d+b*torch.cos(x[:,1])
  ep=(.5234602302310271*.2+.6255677234174437*.2)*9.81;ee=.6255677234174437*.25569305436052964*9.81
  E=.5*(m00*v[:,0]**2+2*m01*v[:,0]*v[:,1]+d*v[:,1]**2)-ep*torch.cos(x[:,0])-ee*torch.cos(x[:,0]+x[:,1])
  return 2*(E-ep-ee)**2+pos+.1*(v*v).sum(-1)

 def action(self,x):
  # Nine constant input pulses, including nominal policy, followed by the same policy.
  n=len(x);nc=10;z=x.double().repeat_interleave(nc,dim=0)
  pulses=torch.tensor([0.,-1,-.5,0,.5,1,-1,1,-1,1],dtype=torch.float64).repeat(n)*self.limit
  lengths=torch.tensor([0,.05,.05,.05,.05,.05,.15,.15,.3,.3]).repeat(n)
  idx=torch.arange(nc).repeat(n);baseu=self.base.action(x).double().clamp(-self.limit,self.limit);cost=torch.zeros(n*nc,dtype=torch.float64)
  dt=.02
  for j in range(round(self.horizon/dt)):
   u=self.base.action(z.float()).double().clamp(-self.limit,self.limit)
   u=torch.where(j*dt<lengths,pulses,u)
   z=step(z,u,dt/2,scale=getattr(self,'model_scale',1.));z=step(z,u,dt/2,scale=getattr(self,'model_scale',1.))
   e=error(z);cost+=dt*((1-torch.cos(e)).sum(-1)+.01*(z[:,2:]**2).sum(-1))
  cost+=self.terminal(z)
  best=cost.reshape(n,nc).argmin(-1)
  return torch.where(best==0,baseu,pulses.reshape(n,nc)[torch.arange(n),best]).float()


def simulate(c,xs,limit=.5,duration=20.,replan=.05,mass_scale=1.):
 x=torch.tensor(xs,dtype=torch.float64);n=len(x);arr=np.full(n,duration);hold=np.zeros(n);up=np.zeros(n);eff=np.zeros(n);trace=[];times=[]
 with torch.no_grad():
  for k in range(round(duration/.002)):
   if k%round(replan/.002)==0:
    t=time.perf_counter();u=c.action(x.float()).double().clamp(-limit,limit);times.append(time.perf_counter()-t)
   # Common high-rate local stabilizer applied to every rollout variant.
   e=torch.cat((error(x),x[:,2:]),-1);local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
   ul=-(e.float()@K.T).squeeze(-1).double().clamp(-limit,limit);applied=torch.where(local,ul,u)
   x=step(x,applied,scale=mass_scale)
   ok=((error(x).abs().amax(-1)<.15)&(x[:,2:].abs().amax(-1)<.6)).numpy();hold=np.where(ok,hold+1,0);arr=np.where((hold>=250)&(arr==duration),(k+1)*.002-.5,arr);up+=ok;eff+=applied.numpy()**2*.002
   if k%10==0:trace.append(torch.cat((x,applied[:,None]),-1).numpy())
 return {'arrival':arr,'success':arr<duration,'uptime':up/(duration/.002),'effort':eff,'trace':np.asarray(trace),'batch_replan_seconds':np.asarray(times)}
