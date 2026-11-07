"""Export numerical results as CSV tables; values are computed from rerun data."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from plot_results import ROOT,ACRO,PEND
from compare_public import display_name

DEST=ROOT.parent/'tables';DEST.mkdir(exist_ok=True)
EX=ROOT/'experiments';PUBLIC=EX/'public_comparison';EXT=EX/'strengthening'

def read(path):return json.loads(path.read_text())
def flatten(record,prefix=''):
    out={}
    for key,value in record.items():
        name=prefix+key
        if isinstance(value,dict):out.update(flatten(value,name+'_'))
        elif isinstance(value,list):out[name]=json.dumps(value)
        else:out[name]=value
    return out

def save(name,rows):
    rows=[flatten(row) for row in rows]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

public=read(PUBLIC/'comparison_analysis.json')
save('table3_public_controllers',[dict(system=system,controller=display_name(name),**values)
    for system,methods in public.items() for name,values in methods.items() if isinstance(values,dict)])
rows=[]
for system,folder,prefix,modes in [('Acrobot',ACRO,'aligned_core',['energy','critic','no_energy']),
                                 ('Pendubot',PEND,'aligned_confirm',['energy','tuned_critic','no_energy'])]:
    for mode in modes:
        with np.load(folder/f'{prefix}_{mode}.npz') as d:
            rows.append(dict(system=system,terminal=mode,recovery=int(d['success'].sum()),
                n=len(d['success']),time=float(d['arrival'].mean()),effort=float(d['effort'].mean())))
save('table4_terminal_cost',rows)
save('table5_policy_continuation',[dict(system=system,continuation=mode,**values)
    for system,methods in read(EXT/'continuation_summary.json').items() for mode,values in methods.items()])
save('table6_gain_loss_noise',[dict(controller=display_name(name),**values)
    for name,values in read(PUBLIC/'gain_noise_comparison.json')['methods'].items()])
nmpc=read(EXT/'nmpc_summary.json')
solver=read(EXT/'solver_quality/summary.json')
rows=[dict(controller=name,**values) for name,values in nmpc.items()]
# The accuracy summary includes both snapshot and closed-loop results.
for key,value in solver.items():
    if key=='confirmation' and isinstance(value,dict):rows.append(dict(controller='high_accuracy_'+key,**value))
save('table7_shared_identification',rows)
save('table8_solver_accuracy',[dict(configuration=name,**value) for name,value in solver.items()
    if isinstance(value,dict) and ('converged' in value or 'median_stationarity' in value)])
rows=[]
for scenario,methods in read(EXT/'stochastic_summary.json').items():
    for mode in ['dynamics','nominal','oracle']:
        rows.append(dict(system='Acrobot',scenario=scenario,model=mode,**methods[mode]))
for mode,values in read(EXT/'pendubot_step/summary.json').items():
    if mode in ['dynamics','nominal','oracle']:rows.append(dict(system='Pendubot',scenario='step',model=mode,**values))
save('table9_repeated_noise',rows)
# Full paired intervals and numerical details accompany the compact CSV tables.
for name,path in [('public',PUBLIC/'comparison_analysis.json'),('gain_noise',PUBLIC/'gain_noise_comparison.json'),
                  ('terminal_acrobot',ACRO/'final_aligned_analysis.json'),('terminal_pendubot',PEND/'aligned_confirmation_analysis.json'),
                  ('shared',EXT/'nmpc_summary.json'),('solver',EXT/'solver_quality/summary.json'),
                  ('noise',EXT/'stochastic_summary.json'),('pendubot_step',EXT/'pendubot_step/summary.json'),
                  ('gain_sweep',ACRO/'aligned_gain_sweep_analysis.json'),('ranking',EXT/'ranking_margin_summary.json')]:
    shutil.copy2(path,DEST/(name+'.json'))
print('Tables written to',DEST)
