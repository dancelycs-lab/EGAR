"""Evidence-bound summaries and vector figures for the three added experiments."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_results import configure, load_npz, wrap_error, OUT, ROOT
from compare_public import extra_outcomes, descriptive

DATA = ROOT/'experiments/strengthening'
MODES = ['dynamics', 'nominal', 'oracle']
LABELS = {'dynamics':'Integral calibration', 'nominal':'Fixed nominal', 'oracle':'True parameters'}
STYLE = {'dynamics':('#0072B2','-'), 'nominal':('#882255','-.'), 'oracle':('#009E73','--')}


def tail_summary():
    ix = np.random.default_rng(64291).integers(0,64,(20000,64))
    out = {}; fig, axes = plt.subplots(1,2,figsize=(6.8,2.65),sharey=True)
    for ax, system in zip(axes,['acrobot','pendubot']):
        out[system] = {}; base = load_npz(DATA/f'tail_{system}_policy.npz')
        legacy = load_npz(ROOT/'experiments'/('authority_calibration/aligned_core_energy.npz'
                         if system=='acrobot' else 'pendubot_transfer/aligned_confirm_energy.npz'))
        for key in ['arrival','effort','trace']:np.testing.assert_array_equal(base[key],legacy[key])
        for mode, name, color, ls in [('policy','Policy tail','#0072B2','-'),
                                      ('zero','Zero tail','#D55E00','--'),
                                      ('hold','Held tail','#777777',':')]:
            data = load_npz(DATA/f'tail_{system}_{mode}.npz'); extra = extra_outcomes(data)
            np.testing.assert_array_equal(data['initial_states'],base['initial_states'])
            arrivals = np.sort(data['arrival'][data['success']]); n = len(arrivals)
            ax.step(np.r_[0,arrivals,20],np.r_[0,np.arange(1,n+1)/64,n/64],where='post',
                    color=color,ls=ls,label=f'{name} ({n}/64)')
            out[system][mode] = dict(recovery=n,retained=int(extra['retained'].sum()),
                                    time=descriptive(data['arrival'],ix),angle_iae=descriptive(extra['angle_iae'],ix),
                                    effort=descriptive(data['effort'],ix),
                                    paired_time_vs_policy=descriptive(data['arrival']-base['arrival'],ix))
        ax.set(xlim=(0,20),ylim=(-.025,1.06),xlabel='Dwell-qualified entry time (s)',
               title='(a) Acrobot' if system=='acrobot' else '(b) Pendubot')
        ax.grid(alpha=.2)
        ax.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.35 if system=='acrobot' else .65),
                  fontsize=7.3,handlelength=1.7,labelspacing=.35)
    axes[0].set_ylabel('Fraction of 64 initial states')
    fig.tight_layout(w_pad=.8);fig.savefig(OUT/'continuation_ablation.pdf');plt.close(fig)
    (DATA/'continuation_summary.json').write_text(json.dumps(out,indent=2));return out


def nmpc_summary():
    horizon = 1.2
    ix = np.random.default_rng(64292).integers(0,64,(20000,64));out={}
    reference = load_npz(DATA/'static_dynamics_seed51.npz')
    all_data={}
    for kind, h, mode in [('rti',1.2,'fixed'),('rti',1.2,'adaptive'),('warm',horizon,'fixed'),('warm',horizon,'adaptive')]:
        prefix=f'nmpc_{kind}_confirmation_H{h}_{mode}'
        rows=[load_npz(DATA/f'{prefix}_{i:02d}.npz') for i in range(64)]
        records=[json.loads((DATA/f'{prefix}_{i:02d}.json').read_text()) for i in range(64)]
        data={k:np.stack([r[k] for r in rows]) for k in ['initial_states','arrival','success','effort','angle_iae']}
        data.update(trace=np.stack([r['trace'] for r in rows],1),theta=np.stack([r['theta'] for r in rows],1),time=rows[0]['time'])
        np.testing.assert_array_equal(data['initial_states'],reference['initial_states'])
        extra=extra_outcomes(data);statuses={}
        for record in records:
            for k,v in record['status_counts'].items():statuses[k]=statuses.get(k,0)+v
        out[kind+'_'+mode]=dict(recovery=int(data['success'].sum()),retained=int(extra['retained'].sum()),
            time=descriptive(data['arrival'],ix),angle_iae=descriptive(data['angle_iae'],ix),
            effort=descriptive(data['effort'],ix),status_counts=statuses,
            nonzero_status_fraction=1-statuses.get('0',0)/sum(statuses.values()),
            parameter_gain_mae=float(abs(data['theta'][-1,:,0]-5/6).mean()),
            paired_time_vs_egar=descriptive(data['arrival']-reference['arrival'],ix),
            paired_angle_vs_egar=descriptive(data['angle_iae']-reference['angle_iae'],ix),
            mean_call_ms=float(np.mean([r['call_mean'] for r in records])*1000))
        np.savez_compressed(DATA/f'{prefix}.npz',**data)
        all_data[kind+'_'+mode]=data
    shared_calibration_figure(reference,all_data)
    for kind in ['rti','warm']:
        out[kind+'_adaptive']['paired_time_vs_fixed']=descriptive(
            all_data[kind+'_fixed']['arrival']-all_data[kind+'_adaptive']['arrival'],ix)
    out['egar']=dict(recovery=int(reference['success'].sum()),retained=int(extra_outcomes(reference)['retained'].sum()),
        time=descriptive(reference['arrival'],ix),angle_iae=descriptive(reference['angle_iae'],ix),
        effort=descriptive(reference['effort'],ix))
    (DATA/'nmpc_summary.json').write_text(json.dumps(out,indent=2));return out


def shared_calibration_figure(egar,methods):
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.9))
    cases=[('EGAR',egar,'#0072B2','-'),
           ('RTI, calibrated',methods['rti_adaptive'],'#D55E00','--'),
           ('Energy NMPC, calibrated',methods['warm_adaptive'],'#009E73','-.'),
           ('Energy NMPC, fixed',methods['warm_fixed'],'#777777',':')]
    tight_path=DATA/'solver_quality/tight_confirmation.npz'
    if tight_path.exists():
        tight=load_npz(tight_path)
        assert tight['arrival'].shape==(64,)
        np.testing.assert_array_equal(tight['initial_states'],egar['initial_states'])
        cases.insert(3,('Energy NMPC, high accuracy',tight,'#882255',(0,(5,2,1,2))))
    for label,data,color,ls in cases:
        arrivals=np.sort(data['arrival'][data['success']]);n=len(arrivals)
        short={'EGAR':'EGAR','RTI, calibrated':'Calibrated RTI',
               'Energy NMPC, calibrated':'Calibrated SQP','Energy NMPC, fixed':'Fixed SQP',
               'Energy NMPC, high accuracy':'High-accuracy SQP'}[label]
        axes[0].step(np.r_[0,arrivals,20],np.r_[0,np.arange(1,n+1)/64,n/64],where='post',
                     color=color,ls=ls,label=f'{short} ({n}/64)')
        t=data['time'];integrand=abs(wrap_error(data['trace'][...,:2])).sum(-1)
        cum=np.vstack([np.zeros(64),np.cumsum(.5*np.diff(t)[:,None]*(integrand[1:]+integrand[:-1]),axis=0)])
        axes[1].plot(t,cum.mean(1),color=color,ls=ls,label=label)
    axes[0].set(xlabel='Dwell-qualified entry time (s)',ylabel='Fraction of 64 initial states',
                title='(a) Recovery with shared calibration',ylim=(-.025,1.06))
    axes[1].set(xlabel='Time (s)',ylabel='Mean cumulative angle error (rad s)',
                title='(b) Complete-response error',ylim=(0,None))
    for ax in axes:ax.set_xlim(0,20);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=7,loc='center right',bbox_to_anchor=(1,.70),handlelength=1.5)
    axes[1].legend(frameon=False,fontsize=6.8,loc='upper left',handlelength=1.5)
    fig.tight_layout();fig.savefig(OUT/'shared_calibration_comparison.pdf');plt.close(fig)


def clustered(values, ix):
    # values shape=(five paired noise realizations, 64 independent initial states)
    assert values.shape == (5,64)
    by_state=values.mean(0)
    return dict(mean=float(values.mean()),seed_means=values.mean(1).tolist(),
                bootstrap95=np.quantile(by_state[ix].mean(1),[.025,.975]).tolist())


def stochastic_summary():
    ix=np.random.default_rng(64293).integers(0,64,(20000,64));out={};episodes=[];loaded={}
    for scenario in ['static','step']:
        out[scenario]={};loaded[scenario]={}
        for mode in MODES:
            runs=[load_npz(DATA/f'{scenario}_{mode}_seed{seed}.npz') for seed in range(51,56)]
            loaded[scenario][mode]=runs
            for r in runs:np.testing.assert_array_equal(r['initial_states'],runs[0]['initial_states'])
            outcomes={key:np.stack([r[key] for r in runs]) for key in
                      ['arrival','effort','angle_iae','post_arrival','post_angle_iae','post_success']}
            retained=[]
            for seed,r in zip(range(51,56),runs):
                assert np.isfinite(r['trace']).all() and abs(r['trace'][...,4]).max()<=.500001
                inside=(abs(wrap_error(r['trace'][...,:2])).max(-1)<.15)&(abs(r['trace'][...,2:4]).max(-1)<.6)
                after=r['time'][:,None]>=float(r['change_time'])+r['post_arrival'][None]+.5-1e-10
                keep=r['post_success']&after.any(0)&~((~inside)&after).any(0);retained.append(keep)
                for i in range(64):
                    episodes.append(dict(scenario=scenario,mode=mode,seed=seed,index=i,
                        post_success=int(r['post_success'][i]),post_retained=int(keep[i]),
                        time=float(r['arrival'][i]),post_time=float(r['post_arrival'][i]),
                        angle_iae=float(r['angle_iae'][i]),post_angle_iae=float(r['post_angle_iae'][i]),
                        effort=float(r['effort'][i])))
            out[scenario][mode]={k:clustered(v,ix) for k,v in outcomes.items() if k!='post_success'}
            out[scenario][mode].update(episodes=320,initial_states=64,noise_seeds=5,
                recovery=int(sum(r['success'].sum() for r in runs)),
                post_recovery=int(outcomes['post_success'].sum()),
                recovery_by_seed=[int(r['post_success'].sum()) for r in runs],
                post_retained=int(np.asarray(retained).sum()),
                recovered_before_change=int(sum(r['recovered_before_change'].sum() for r in runs)))
        base=np.stack([r['post_arrival'] for r in loaded[scenario]['dynamics']])
        for mode in ['nominal','oracle']:
            out[scenario][mode]['paired_post_time_vs_calibrated']=clustered(
                np.stack([r['post_arrival'] for r in loaded[scenario][mode]])-base,ix)
    with (DATA/'stochastic_episode_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)
    (DATA/'stochastic_summary.json').write_text(json.dumps(out,indent=2))
    step_figure(loaded['step'])
    return out


def step_figure(runs):
    adaptive=runs['dynamics'];flat=np.concatenate([r['post_arrival'] for r in adaptive])
    selected=int(np.argmin(abs(flat-np.median(flat))));seed_index,state_index=divmod(selected,64)
    record=dict(seed=51+seed_index,state_index=state_index,rule='closest to pooled median calibrated post-change capped time, first index for ties')
    (DATA/'step_representative.json').write_text(json.dumps(record,indent=2))
    fig,axes=plt.subplots(3,2,figsize=(6.8,6.2))
    for mode in MODES:
        color,ls=STYLE[mode];data=runs[mode][seed_index];t=data['time'];state=data['trace'][:,state_index,:4]
        for row,(ax,val) in enumerate(zip(axes[:,0],[abs(wrap_error(state[:,:2])).max(-1),data['trace'][:,state_index,4],data['theta'][:,state_index,0]])):
            ax.plot(t,val,color=color,ls=ls,label=LABELS[mode],
                    lw=(1.15 if mode=='dynamics' else .7) if row==1 else 1.35,
                    zorder=4 if mode=='dynamics' else 2);ax.set_xlim(0,20)
        post=np.concatenate([r['post_arrival'][r['post_success']] for r in runs[mode]])
        short={'dynamics':'Calibrated','nominal':'Fixed','oracle':'Known gain'}[mode]
        n=len(post);axes[0,1].step(np.r_[0,np.sort(post),19],np.r_[0,np.arange(1,n+1)/320,n/320],
                                where='post',color=color,ls=ls,label=f'{short} ({n}/320)')
        values=[];gains=[]
        for r in runs[mode]:
            integrand=abs(wrap_error(r['trace'][...,:2])).sum(-1)
            cumulative=np.vstack([np.zeros(64),np.cumsum(.5*np.diff(t)[:,None]*(integrand[1:]+integrand[:-1]),axis=0)])
            cumulative=cumulative-cumulative[50]
            values.append(cumulative);gains.append(r['theta'][...,0])
        # Population trajectories across the five fixed noise streams; no iid time-sample inference.
        value=np.stack(values).mean(0);mask=t>=1
        axes[1,1].plot(t[mask]-1,value[mask].mean(1),color=color,ls=ls,label=LABELS[mode])
        gain_values=np.concatenate(gains,axis=1)
        axes[2,1].plot(t,gain_values.mean(1),color=color,ls=ls,label=LABELS[mode])
        if mode=='dynamics':
            lo,hi=np.quantile(gain_values,[.1,.9],axis=1)
            axes[2,1].fill_between(t,lo,hi,color=color,alpha=.14,lw=0)
    axes[0,0].set(title=f'(a) State response: seed {51+seed_index}, state {state_index}',ylabel=r'$\|e_q\|_\infty$ (rad)')
    axes[1,0].set(title='(c) Commanded torque',ylabel=r'$u$ (N m)',ylim=(-.56,.56))
    axes[2,0].set(title='(e) Individual gain estimate',ylabel=r'$\hat\gamma$',xlabel='Time (s)',xlim=(0,5))
    axes[0,1].set(title='(b) Post-change recovery',ylabel='Fraction of 320 episodes',xlabel='Time after gain loss (s)',xlim=(0,19),ylim=(-.025,1.06))
    axes[1,1].set(title='(d) Post-change angular deviation',ylabel='Mean angle integral (rad s)',xlabel='Time after gain loss (s)',xlim=(0,19),ylim=(0,None))
    axes[2,1].set(title='(f) Population gain estimate',ylabel=r'$\hat\gamma$',xlabel='Time (s)',xlim=(0,5))
    for ax in axes[:,0]:ax.axvline(1,color='.4',lw=.8,ls=':')
    axes[2,1].axvline(1,color='.4',lw=.8,ls=':')
    axes[0,0].axhline(.15,color='.4',lw=.7,ls=':')
    axes[1,0].axhline(.5,color='.4',lw=.7,ls=':');axes[1,0].axhline(-.5,color='.4',lw=.7,ls=':')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[1,1].legend(loc='upper left',frameon=False,fontsize=7)
    axes[0,1].legend(loc='center right',bbox_to_anchor=(1,.57),frameon=False,fontsize=6.8,handlelength=1.5)
    axes[2,0].legend(loc='upper right',frameon=False,fontsize=7)
    fig.tight_layout(h_pad=.6,w_pad=1);fig.savefig(OUT/'abrupt_gain_recovery.pdf');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['tail','nmpc','stochastic','all'],default='all')
    args=ap.parse_args();configure()
    for name,fn in [('tail',tail_summary),('nmpc',nmpc_summary),('stochastic',stochastic_summary)]:
        if args.part in [name,'all']:print(name,json.dumps(fn(),indent=2))
