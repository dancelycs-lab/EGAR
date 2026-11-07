from transfer_screen import *
class PFL:
 def __init__(self,kp,kd,ke):self.kp=kp;self.kd=kd;self.ke=ke
 def action(self,x):
  q1,q2,v1,v2=x.unbind(-1);b=.6255677234174437*.2*.25569305436052964;d=.05086984812807257
  m00=.031887199591513114+d+.6255677234174437*.2**2+2*b*torch.cos(q2);m01=d+b*torch.cos(q2)
  g0=-(.5234602302310271*.2+.6255677234174437*.2)*9.81*torch.sin(q1)-.6255677234174437*.25569305436052964*9.81*torch.sin(q1+q2)
  g1=-.6255677234174437*.25569305436052964*9.81*torch.sin(q1+q2)
  h0=-b*torch.sin(q2)*(2*v1*v2+v2*v2);h1=b*torch.sin(q2)*v1*v1
  ep=(.5234602302310271*.2+.6255677234174437*.2)*9.81;ee=.6255677234174437*.25569305436052964*9.81
  energy=.5*(m00*v1*v1+2*m01*v1*v2+d*v2*v2)-ep*torch.cos(q1)-ee*torch.cos(q1+q2)
  w=-self.kp*((q2+np.pi)%(2*np.pi)-np.pi)-self.kd*v2+self.ke*v1*(energy-ep-ee)
  u=(d-m01*m01/m00)*w+h1-m01/m00*h0+m01/m00*g0-g1
  e=torch.cat((error(x),x[:,2:]),-1);local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
  return torch.where(local,-(e@K.T).squeeze(-1),u)
