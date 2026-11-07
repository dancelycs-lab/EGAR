from runtime import ASSETS, BUILD
import contextlib,time
from acados_readiness import *
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
import casadi as ca
OUT=ROOT
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--split',default='development',choices=['development','heldout']);ap.add_argument('--horizon',type=float);args=ap.parse_args()
xs=np.load(OUT/('heldout_initial_states.npy' if args.split=='heldout' else 'exploratory_initial_states.npy'));xs=xs if args.split=='heldout' else xs[:8];rows=[]
configs=[(.5,args.horizon)] if args.horizon else [(6.,.5),(.5,.5),(.5,1.2)]
prefix='acados' if not args.horizon else f'acados_{args.split}_H{args.horizon}'
pars=model_parameters(filepath=str(ASSETS / "model_parameters.yml"));pars.set_torque_limit([0,6]);pars.set_motor_inertia(0);pars.set_damping([0,0]);pars.set_cfric([0,0]);plant=SymbolicDoublePendulum(model_pars=pars)
def advance(x,u,dt=.002):
 a=plant.rhs(0,x,u);b=plant.rhs(0,x+dt*a/2,u);d=plant.rhs(0,x+dt*b/2,u);e=plant.rhs(0,x+dt*d,u);return x+dt*(a+2*b+2*d+e)/6
for limit,horizon in configs:
 with open(OUT/f'acados_init_{limit}_{horizon}.log','w') as log,contextlib.redirect_stdout(log):c=create(limit,horizon)
 model=c.pendulum_model.acados_model();f=ca.Function('f',[model.x,model.u],[model.f_expl_expr]);rng=np.random.default_rng(91);diff=[]
 for _ in range(100):
  x=rng.uniform([-6,-6,-10,-10],[6,6,10,10]);u=[0,rng.uniform(-limit,limit)];diff.append(np.max(abs(np.array(f(x,u)).ravel()-plant.rhs(0,x,u))))
 print('dynamics',limit,horizon,max(diff),flush=True)
 # Save the original post-initialization solution and restore it for each trial.
 initx=[c.ocp_solver.get(i,'x').copy() for i in range(c.N_horizon+1)];initu=[c.ocp_solver.get(i,'u').copy() for i in range(c.N_horizon)]
 for index,x0 in enumerate(xs):
  if not c.async_mpc_future.done():c.async_mpc_future.result()
  c.ocp_solver.reset()
  for i,z in enumerate(initx):c.ocp_solver.set(i,'x',z)
  for i,z in enumerate(initu):c.ocp_solver.set(i,'u',z)
  c.last_mpc_run_t=-float('inf');c.last_good_solution_time=-float('inf');c.last_u[:]=0;c.mpc_u[:]=0
  x=x0.copy();hold=0;arrival=20.;up=0;eff=0;tr=[];statuses={};times=[]
  for k in range(10000):
   t=time.perf_counter();u=c.get_control_output(x,k*.002);times.append(time.perf_counter()-t)
   x=advance(x,u);e=(x[:2]-[np.pi,0]+np.pi)%(2*np.pi)-np.pi;ok=max(abs(e))<.15 and max(abs(x[2:]))<.6;hold=hold+1 if ok else 0
   if hold>=250 and arrival==20.:arrival=(k+1)*.002-.5
   up+=ok;eff+=float(u@u)*.002;status=int(c.ocp_solver.get_status());statuses[status]=statuses.get(status,0)+1
   if k%10==0:tr.append(np.r_[x,u])
  tag=f'{prefix}_L{limit}_H{horizon}_{index}';np.savez(OUT/(tag+'.npz'),trace=np.asarray(tr),times=times)
  row={'name':tag,'arrival':arrival,'success':arrival<20,'uptime':up/10000,'effort':eff,'status_counts':statuses,'dynamics_max_abs':max(diff),'call_mean':float(np.mean(times))};rows.append(row);print(row,flush=True);(OUT/(prefix+'_results.json')).write_text(json.dumps(rows,indent=2))
 c.spinner.shutdown(wait=True)
