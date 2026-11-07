from pathlib import Path
import argparse
import json
import time
import numpy as np
import calibrated_nmpc as baseline

HERE=Path(__file__).resolve().parent
OUT=HERE/'solver_quality'
CONFIGS={'original':(40,1e-3),'budget':(160,1e-3),'tight':(160,1e-6),'extended':(640,1e-6)}

def create():
    original=baseline.vendor.AcadosOcpSolver
    def factory(ocp,**kw):
        ocp.solver_options.nlp_solver_max_iter=640
        ocp.solver_options.timeout_max_time=10.
        return original(ocp,**kw)
    baseline.vendor.AcadosOcpSolver=factory
    try: c=baseline.create(1.2,'warm')
    finally: baseline.vendor.AcadosOcpSolver=original
    return c

def configure(c,key):
    iterations,tolerance=CONFIGS[key]
    c.ocp_solver.options_set('max_iter',iterations)
    for r in ['stat','eq','ineq','comp']:c.ocp_solver.options_set('qp_tol_'+r,tolerance)

def reset(c):
    c.ocp_solver.reset(); c.last_mpc_run_t=-float('inf'); c.last_good_solution_time=-float('inf')
    c.last_u[:]=0.; c.mpc_u[:]=0.

def snapshot(c):
    rows=[]
    for index in range(8):
        d=np.load(HERE/f'nmpc_warm_development_H1.2_adaptive_{index:02d}.npz')
        for j in [0,25,50,100,200,400,600,800]:
            x=d['trace'][j,:4];theta=d['theta'][j]
            for key in CONFIGS:
                reset(c);configure(c,key);baseline.set_parameters(c,theta,'warm')
                baseline.policy_warm_start(c,x,theta)
                tic=time.perf_counter();u=c.get_control_output(x,0.)
                row=dict(index=index,time=float(d['time'][j]),configuration=key,
                    residuals=c.ocp_solver.get_residuals(recompute=True).tolist(),
                    iterations=int(c.ocp_solver.get_stats('sqp_iter')),
                    cost=float(c.ocp_solver.get_cost()),status=int(c.ocp_solver.get_status()),
                    command=float(u[1]),seconds=time.perf_counter()-tic)
                assert np.isfinite(row['residuals']).all() and np.isfinite(row['cost'])
                rows.append(row)
        print('snapshot state',index,'complete',flush=True)
    (OUT/'snapshot_results.json').write_text(json.dumps(rows,indent=2))

def closed_loop(c,start,stop,split='development'):
    states=(np.load(HERE.parent/'global_recovery/exploratory_initial_states.npy')[:8]
            if split=='development' else np.load(HERE.parent/'authority_calibration/gain_confirmation_states.npy'))
    noise=baseline.noise_stream(7401 if split=='development' else 51,len(states));configure(c,'tight')
    for i in range(start,min(stop,len(states))):
        path=OUT/f'tight_{split}_{i:02d}.npz'
        if path.exists():continue
        reset(c);baseline.set_parameters(c,np.array([1.,0.,0.]),'warm')
        residuals=[];iterations=[];output=c.get_control_output
        def tracked(x,t):
            u=output(x,t)
            residuals.append(c.ocp_solver.get_residuals(recompute=True).copy())
            iterations.append(c.ocp_solver.get_stats('sqp_iter'))
            return u
        c.get_control_output=tracked
        tic=time.perf_counter()
        result,status=baseline.simulate(c,states[i],noise[:,i],True,'warm')
        c.get_control_output=output
        result.update(solver_residuals=np.array(residuals),solver_iterations=np.array(iterations))
        np.savez_compressed(path,**result)
        row=dict(index=i,split=split,recovery=int(result['success']),time=result['arrival'],
                 angle_iae=result['angle_iae'],effort=result['effort'],status_counts=status,
                 residual_median=np.median(residuals,axis=0).tolist(),seconds=time.perf_counter()-tic)
        path.with_suffix('.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['snapshot','closed'],required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=8)
    ap.add_argument('--split',choices=['development','confirmation'],default='development')
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.split=='development':
        (OUT/'protocol.json').write_text(json.dumps(dict(configurations=CONFIGS,
            states='Original eight development states; no confirmation state used for tuning',
            snapshot_times=[0,.5,1,2,4,8,12,16],initialization='Identical frozen-policy initialization',
            timeout_seconds=10.,closed_loop_configuration='tight',noise_seed=7401),indent=2))
    c=create()
    try:
        if args.part=='snapshot':snapshot(c)
        else:closed_loop(c,args.start,args.stop,args.split)
    finally:c.spinner.shutdown(wait=True)
