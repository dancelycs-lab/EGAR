import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'global_recovery';sys.path.insert(0,str(BASE))
from work_balance_screen import energy
from rollout_screen import *
OUT=Path(__file__).resolve().parent

def mass_entries(x):
 c=torch.cos(x[...,1]);b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257
 a=.031887199591513114+d+.6255677234174437*.2**2+2*b*c;off=d+b*c;det=a*d-off*off
 return a,off,d,det

def features(x,u):
 a,b,d,det=mass_entries(x);v=x[...,2:];z=torch.zeros_like(u)
 # Columns multiply normalized parameters [gamma, d1/.03, d2/.03].
 B=torch.stack((-b/det,a/det),-1)*u[...,None]
 D1=torch.stack((-d/det,b/det),-1)*v[...,0,None]*.03
 D2=torch.stack((b/det,-a/det),-1)*v[...,1,None]*.03
 return torch.stack((B,D1,D2),-1)
def derivative(x,u,theta):
 f=rhs(x,torch.zeros_like(u));f[...,2:]+=torch.einsum('...ij,...j->...i',features(x,u),theta)
 return f

def integrate(x,u,theta,dt):
 a=derivative(x,u,theta);b=derivative(x+dt*a/2,u,theta);c=derivative(x+dt*b/2,u,theta);d=derivative(x+dt*c,u,theta)
 return x+dt*(a+2*b+2*c+d)/6

class CalibratedRollout(Rollout):
 def __init__(self,n,transport='none'):super().__init__(.5,1.2,'energy');self.theta=torch.tensor([1.,0.,0.],dtype=torch.float64).repeat(n,1);self.transport=transport;self.long_prefix=False
 def tail_action(self,x,pars):
  u=self.base.action(x.float()).double()
  if self.transport!='none':
   comp=.03*pars[...,2]*x[...,3]
   if self.transport=='metric':
    a,b,d,det=mass_entries(x);comp-=b/a*.03*pars[...,1]*x[...,2]
   u=(u+comp)/pars[...,0]
  return u.clamp(-.5,.5)

 def action(self,x):
  n=len(x);amps=[0.,-1,-.5,0,.5,1,-1,1,-1,1];dur=[0,.05,.05,.05,.05,.05,.15,.15,.3,.3]
  if self.long_prefix:amps+=[-1,1,-1,1,-1,1];dur+=[.6,.6,.9,.9,1.2,1.2]
  nc=len(amps);z=x.double().repeat_interleave(nc,0);pars=self.theta.repeat_interleave(nc,0)
  pulse=torch.tensor(amps,dtype=torch.float64).repeat(n)*.5
  lengths=torch.tensor(dur).repeat(n)
  bu=self.tail_action(x.double(),self.theta);cost=torch.zeros(n*nc,dtype=torch.float64)
  for j in range(60):
   u=self.tail_action(z,pars);u=torch.where(j*.02<lengths,pulse,u)
   z=integrate(z,u,pars,.01);z=integrate(z,u,pars,.01)
   cost+=.02*((1-torch.cos(error(z))).sum(-1)+.01*(z[:,2:]**2).sum(-1))
  cost+=self.terminal(z);best=cost.reshape(n,nc).argmin(-1)
  return torch.where(best==0,bu,pulse.reshape(n,nc)[torch.arange(n),best]).float()


def experiment(xs,mode,truepars,sigma=(.001,.01),seed=51,observer=False,transport='none',long_prefix=False,controller_cls=CalibratedRollout):
 n=len(xs);x=torch.tensor(xs,dtype=torch.float64);c=controller_cls(n,transport);c.long_prefix=long_prefix;true=torch.tensor(truepars,dtype=torch.float64).repeat(n,1)
 if mode=='oracle':c.theta=true.clone()
 gen=torch.Generator().manual_seed(seed);noise=torch.tensor([sigma[0],sigma[0],sigma[1],sigma[1]],dtype=torch.float64)
 y=x+torch.randn(x.shape,generator=gen,dtype=torch.float64)*noise
 xh=y.clone();cov=torch.diag(noise**2).repeat(n,1,1);Robs=torch.diag(noise**2).repeat(n,1,1);Qobs=torch.diag(torch.tensor([1e-12,1e-12,1e-7,1e-7],dtype=torch.float64));I4=torch.eye(4,dtype=torch.float64)
 G=torch.eye(3,dtype=torch.float64).repeat(n,1,1)*1e-4;r=G@torch.tensor([1.,0.,0.],dtype=torch.float64);E0=energy(y);v0=y[:,2:].clone();work=torch.zeros((n,3),dtype=torch.float64);Iphi=torch.zeros((n,2,3),dtype=torch.float64);If=torch.zeros((n,2),dtype=torch.float64)
 arr=np.full(n,20.);hold=np.zeros(n);up=np.zeros(n);eff=np.zeros(n);tr=[];est=[];cond=[]
 with torch.no_grad():
  for k in range(10000):
   yc=xh if observer else y
   if k%25==0:u=c.action(yc.float()).double().clamp(-.5,.5)
   e=torch.cat((error(yc),yc[:,2:]),-1);local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
   act=torch.where(local,(-(e.float()@K.T).squeeze(-1)).double().clamp(-.5,.5),u)
   xn=integrate(x,act,true,.002);yn=xn+torch.randn(x.shape,generator=gen,dtype=torch.float64)*noise
   work[:,0]+=act*(y[:,3]+yn[:,3])*.001
   work[:,1:]-=.03*((y[:,2:]**2+yn[:,2:]**2)*.5-sigma[1]**2)*.002
   Iphi+=(features(y,act)+features(yn,act))*.001
   If+=(rhs(y,torch.zeros(n))[:,2:]+rhs(yn,torch.zeros(n))[:,2:])*.001
   if observer:
    delta=I4[None,:,:]*1e-5;xp=xh[:,None,:]+delta;xm=xh[:,None,:]-delta
    jac=((derivative(xp,act[:,None].expand(-1,4),c.theta[:,None,:])-derivative(xm,act[:,None].expand(-1,4),c.theta[:,None,:]))/(2e-5)).transpose(-1,-2)
    F=I4+.002*jac;cov=F@cov@F.transpose(-1,-2)+Qobs
    gain=torch.linalg.solve(cov+Robs,cov.transpose(-1,-2)).transpose(-1,-2)
    pred=integrate(xh,act,c.theta,.002);xh=pred+torch.einsum('nij,nj->ni',gain,yn-pred)
    J=I4-gain;cov=J@cov@J.transpose(-1,-2)+gain@Robs@gain.transpose(-1,-2)
   x=xn;y=yn
   if (k+1)%100==0:
    if mode in ['work','scalar','dynamics']:
     if mode=='dynamics':F=Iphi;target=(y[:,2:]-v0-If)
     else:
      F=work[:,None,:].clone();target=(energy(y)-E0)[:,None]
      if mode=='scalar':F[:,:,1:]=0
     G=.95*G+torch.einsum('nki,nkj->nij',F,F);r=.95*r+torch.einsum('nki,nk->ni',F,target)
     c.theta=torch.linalg.solve(G,r).clamp(torch.tensor([.4,0,0]),torch.tensor([1.6,2.,2.]))
    E0=energy(y);v0=y[:,2:].clone();work.zero_();Iphi.zero_();If.zero_()
   ok=((error(x).abs().amax(-1)<.15)&(x[:,2:].abs().amax(-1)<.6)).numpy();hold=np.where(ok,hold+1,0);arr=np.where((hold>=250)&(arr==20),(k+1)*.002-.5,arr);up+=ok;eff+=act.numpy()**2*.002
   if k%10==0:tr.append(torch.cat((x,act[:,None]),-1).numpy());est.append(c.theta.numpy().copy())
 return {'arrival':arr,'success':arr<20,'uptime':up/10000,'effort':eff,'trace':np.asarray(tr),'theta':np.asarray(est),'true_theta':truepars}

