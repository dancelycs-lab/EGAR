import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'global_recovery';sys.path.insert(0,str(BASE))
from batch_screen import *
from scipy.linalg import solve_continuous_are
OUT=Path(__file__).resolve().parent
p=Policy('pendubot')
def derivative(x,u):
 c=torch.cos(x[...,1]);b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257;a=.031887199591513114+d+.6255677234174437*.2**2+2*b*c;off=d+b*c;det=a*d-off*off
 f=rhs(x,torch.zeros_like(u));f[...,2]+=d/det*u;f[...,3]-=off/det*u;return f

def integrate(x,u,dt):
 a=derivative(x,u);b=derivative(x+dt*a/2,u);c=derivative(x+dt*b/2,u);d=derivative(x+dt*c,u);return x+dt*(a+2*b+2*c+d)/6

g=torch.tensor([np.pi,0.,0.,0.],dtype=torch.float64);A=torch.autograd.functional.jacobian(lambda x:derivative(x,torch.tensor(0.)),g).numpy();B=torch.autograd.functional.jacobian(lambda u:derivative(g,u),torch.tensor(0.,dtype=torch.float64)).numpy()[:,None];P=solve_continuous_are(A,B,np.diag([10,10,1,1]),np.array([[.1]]));K=torch.tensor(10*B.T@P,dtype=torch.float32)
def base(x):
 e=torch.cat((error(x),x[...,2:]),-1);local=(e[...,:2].abs().amax(-1)<.3)&(e[...,2:].abs().amax(-1)<1.5)
 return torch.where(local,-(e@K.T).squeeze(-1),p.action(x))
def terminal(x,mode):
 if mode=='critic':return -p.value(x.float())
 if mode=='physical':return 10*(1-torch.cos(error(x))).sum(-1)+.2*(x[...,2:]**2).sum(-1)
 q1,q2,v1,v2=x.unbind(-1);b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257;m00=.031887199591513114+d+.6255677234174437*.2**2+2*b*torch.cos(q2);m01=d+b*torch.cos(q2);ep=(.5234602302310271*.2+.6255677234174437*.2)*9.81;ee=.6255677234174437*.25569305436052964*9.81
 E=.5*(m00*v1*v1+2*m01*v1*v2+d*v2*v2)-ep*torch.cos(q1)-ee*torch.cos(q1+q2)
 return 2*(E-ep-ee)**2+(1-torch.cos(error(x))).sum(-1)+.1*(x[...,2:]**2).sum(-1)
