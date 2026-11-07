from runtime import ASSETS, BUILD
from rollout_screen import *
import mujoco
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
m1=.5234602302310271;m2=.6255677234174437;r1=.2;r2=.25569305436052964;ic1=.031887199591513114-m1*r1*r1;ic2=.05086984812807257-m2*r2*r2
xml=f'''<mujoco model="acrobot"><option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/><worldbody><body name="upper"><joint name="q1" type="hinge" axis="0 1 0"/><inertial pos="0 0 -{r1}" mass="{m1}" diaginertia="{ic1} {ic1} .001"/><geom type="capsule" fromto="0 0 0 0 0 -.2" size=".015" contype="0" conaffinity="0"/><body name="lower" pos="0 0 -.2"><joint name="q2" type="hinge" axis="0 1 0"/><inertial pos="0 0 -{r2}" mass="{m2}" diaginertia="{ic2} {ic2} .001"/><geom type="capsule" fromto="0 0 0 0 0 -.3" size=".015" contype="0" conaffinity="0"/></body></body></worldbody><actuator><motor joint="q2" ctrllimited="true" ctrlrange="-.5 .5"/></actuator></mujoco>'''
(ROOT/'acrobot.xml').write_text(xml);model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
pars=model_parameters(filepath=str(ASSETS / "model_parameters.yml"));pars.set_torque_limit([0,6]);pars.set_motor_inertia(0);pars.set_damping([0,0]);pars.set_cfric([0,0]);plant=SymbolicDoublePendulum(model_pars=pars)
rng=np.random.default_rng(71);dm=[];df=[]
for _ in range(100):
 x=rng.uniform([-6,-6,-10,-10],[6,6,10,10]);u=rng.uniform(-.5,.5);data.qpos[:]=x[:2];data.qvel[:]=x[2:];data.ctrl[:]=u;mujoco.mj_forward(model,data);M=np.empty((2,2));mujoco.mj_fullM(model,data,M);dm.append(np.max(abs(M-plant.mass_matrix(x))));df.append(np.max(abs(data.qacc-plant.rhs(0,x,[0,u])[2:])))
report={'random_states':100,'mass_max_abs':max(dm),'acceleration_max_abs':max(df),'mujoco_version':mujoco.__version__};print(report,flush=True);(ROOT/'mujoco_validation.json').write_text(json.dumps(report,indent=2))
xs=np.load(ROOT/'heldout_initial_states.npy')[:8];c=Rollout(.5,1.2,'energy');rows=[]
with torch.no_grad():
 for index,x0 in enumerate(xs):
  mujoco.mj_resetData(model,data);data.qpos[:]=x0[:2];data.qvel[:]=x0[2:];hold=0;arrival=20.;up=0;eff=0;tr=[]
  for k in range(10000):
   x=torch.tensor(np.r_[data.qpos,data.qvel][None],dtype=torch.float32)
   if k%25==0:u=float(c.action(x)[0])
   e=torch.cat((error(x),x[:,2:]),-1);local=(e[:,:2].abs().amax(-1)<.3)&(e[:,2:].abs().amax(-1)<1.5)
   act=float((-(e@K.T).squeeze(-1)).clamp(-.5,.5)[0]) if local[0] else u
   data.ctrl[:]=act;mujoco.mj_step(model,data);er=(data.qpos-[np.pi,0]+np.pi)%(2*np.pi)-np.pi;ok=max(abs(er))<.15 and max(abs(data.qvel))<.6;hold=hold+1 if ok else 0
   if hold>=250 and arrival==20:arrival=(k+1)*.002-.5
   up+=ok;eff+=act*act*.002
   if k%10==0:tr.append(np.r_[data.qpos,data.qvel,act])
  np.savez(ROOT/f'mujoco_energy_{index}.npz',trace=np.asarray(tr));row={'index':index,'arrival':arrival,'success':arrival<20,'uptime':up/10000,'effort':eff};rows.append(row);print(row,flush=True);(ROOT/'mujoco_results.json').write_text(json.dumps(rows,indent=2))
