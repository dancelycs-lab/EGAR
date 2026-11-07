#!/usr/bin/env python3
"""Run the paper experiments from a self-contained source and input package."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE=Path(__file__).resolve().parent
STAGES=['nominal','gain','continuation','multiseed','public','shared','solver','certificate','supplementary','figures']
INPUTS={
 'acrobot_nominal.npy':'global_recovery/heldout_initial_states.npy',
 'acrobot_development.npy':'global_recovery/exploratory_initial_states.npy',
 'acrobot_noise.npy':'authority_calibration/gain_confirmation_states.npy',
 'acrobot_drag.npy':'authority_calibration/initial_states.npy',
 'pendubot_nominal.npy':'pendubot_transfer/confirmation_states.npy',
}

def prepare(output):
    work=output/'work'
    shutil.copytree(PACKAGE/'src',work,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name,target in INPUTS.items():
        shutil.copy2(PACKAGE/'assets'/name,work/'experiments'/target)
    (output/'figures').mkdir(parents=True,exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    (output/'completed').mkdir(exist_ok=True)
    env=os.environ.copy()
    external=PACKAGE/'external'
    pend=Path(env.get('DOUBLE_PENDULUM_DIR',external/'double_pendulum')).resolve()
    acados=Path(env.get('ACADOS_SOURCE_DIR',external/'acados')).resolve()
    paths=[str(work),str(pend/'src/python'),str(pend/'src/cpp/python'),str(acados/'interfaces/acados_template')]
    if env.get('PYTHONPATH'):paths.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(paths),EGAR_PACKAGE_ROOT=str(PACKAGE),EGAR_RESULTS=str(output),
               ACADOS_SOURCE_DIR=str(acados),MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    libraries=[str(acados/'lib'),str(pend/'src/cpp/controllers/ilqr/obj')]
    if env.get('LD_LIBRARY_PATH'):libraries.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH']=os.pathsep.join(libraries)
    return work,env

def jobs(stage):
    e='experiments/'
    if stage=='smoke':yield 'smoke','smoke.py',[]
    if stage=='nominal':
        for folder,name in [('authority_calibration','aligned_core_confirmation'),('pendubot_transfer','aligned_confirm'),
                            ('global_recovery','nominal_policy'),('pendubot_transfer','nominal_policy'),
                            ('public_comparison','classical')]:yield folder+'_'+name,e+folder+'/'+name+'.py',[]
    if stage=='gain':
        for folder,name in [('authority_calibration','final_aligned_confirmation'),('authority_calibration','gain_sweep_aligned'),
                            ('public_comparison','calibration_diagnostic')]:yield name,e+folder+'/'+name+'.py',[]
    if stage=='continuation':
        for system in ['acrobot','pendubot']:
            for mode in ['policy','zero','hold']:
                yield f'tail_{system}_{mode}',e+'strengthening/continuation.py',['--system',system,'--mode',mode]
    if stage=='multiseed':
        for scenario in ['static','step']:
            for seed in range(51,56):
                for mode in ['nominal','dynamics','oracle']:
                    yield f'acrobot_{scenario}_{seed}_{mode}',e+'strengthening/gain_trials.py',['--scenario',scenario,'--seed',str(seed),'--mode',mode]
        for seed in range(51,56):
            for mode in ['nominal','dynamics','oracle']:
                yield f'pendubot_step_{seed}_{mode}',e+'strengthening/pendubot_gain.py',['--seed',str(seed),'--mode',mode]
    if stage=='public':
        yield 'acrobot_nmpc',e+'global_recovery/acados_screen.py',['--split','heldout','--horizon','1.2']
        yield 'pendubot_nmpc',e+'public_comparison/pendubot_nmpc.py',[]
        yield 'ilqr',e+'pendubot_transfer/ilqr_comparison.py',[]
        yield 'gain_noise_external',e+'public_comparison/gain_noise_external.py',[]
    if stage=='shared':
        for kind in ['rti','warm']:
            for adaptive in [False,True]:
                args=['--kind',kind,'--horizon','1.2']+(['--adaptive'] if adaptive else [])
                yield f'nmpc_{kind}_{adaptive}',e+'strengthening/calibrated_nmpc.py',args
        yield 'nmpc_development',e+'strengthening/calibrated_nmpc.py',['--kind','warm','--split','development','--horizon','1.2','--adaptive','--stop','8']
    if stage=='solver':
        yield 'solver_snapshot',e+'strengthening/solver_quality.py',['--part','snapshot']
        for split,count in [('development',8),('confirmation',64)]:
            yield 'solver_'+split,e+'strengthening/solver_quality.py',['--part','closed','--split',split,'--stop',str(count)]
    if stage=='certificate':
        yield 'lqr_design',e+'global_recovery/lqr_design.py',[]
        yield 'interval_certificate','method/interval_sampled_certificate.py',[]
        yield 'local_regulation','method/local_regulation.py',[]
    if stage=='supplementary':
        yield 'timing',e+'pendubot_transfer/timing_benchmark.py',[]
        yield 'mujoco',e+'global_recovery/mujoco_check.py',[]
        yield 'dissipation',e+'authority_calibration/dissipation.py',[]
    if stage=='figures':
        for folder,name in [('authority_calibration','analyze_final_aligned'),('authority_calibration','analyze_gain_sweep'),
                            ('pendubot_transfer','analyze_aligned_confirmation')]:yield name,e+folder+'/'+name+'.py',[]
        for name,args in [('plot_results',[]),('compare_public',[]),('analyze_revision',['--part','all']),
                          ('analyze_strengthening',['--part','all']),('draw_simulation_systems',[]),('export_tables',[])]:
            yield name,'analysis/'+name+'.py',args

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['all','smoke']+STAGES)
    parser.add_argument('--output',type=Path,default=PACKAGE/'results',help='Output directory (default: results).')
    parser.add_argument('--only',help='Run only a job whose name contains this text.')
    parser.add_argument('--list',action='store_true',help='List jobs without importing numerical libraries.')
    args=parser.parse_args()
    todo=[job for stage in (STAGES if args.stage=='all' else [args.stage]) for job in jobs(stage)]
    if args.only:todo=[j for j in todo if args.only in j[0]]
    if not todo:parser.error('No matching jobs.')
    if args.list:
        for name,path,opts in todo:print(name,':',path,' '.join(opts))
        return
    output=args.output.expanduser().resolve();work,env=prepare(output)
    for name,path,opts in todo:
        marker=output/'completed'/name
        if marker.exists():print('Completed:',name,flush=True);continue
        logpath=output/'logs'/(name+'.log')
        print('Running:',name,'| log:',logpath,flush=True)
        with logpath.open('w') as log:
            # Each experiment has its own module namespace and working directory.
            jobenv=env.copy()
            if name=='mujoco':jobenv['PYTHONPATH']=str(work/'experiments/authority_calibration')+os.pathsep+env['PYTHONPATH']
            result=subprocess.run([sys.executable,str(work/path),*opts],cwd=(work/path).parent,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(logpath.read_text()[-6000:],file=sys.stderr)
            raise SystemExit(f'{name} failed; see {logpath}. Earlier jobs are retained.')
        marker.touch();print('Finished:',name,flush=True)
    print('Results:',output)

if __name__=='__main__':main()
