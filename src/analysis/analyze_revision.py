"""Ranking-margin, cross-actuation and SQP-accuracy evidence for the revision."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_results import configure, load_npz, wrap_error, OUT, ROOT
from analyze_strengthening import clustered, MODES, LABELS, STYLE
from compare_public import descriptive, extra_outcomes

DATA=ROOT/'experiments/strengthening'

def margins():
    d=load_npz(ROOT/'experiments/public_comparison/calibration_diagnostic.npz')
    reference=d['reference_cost'];out={};fig,axes=plt.subplots(2,2,figsize=(6.8,5.0))
    for key,label,color,ls in [('nominal_cost','Fixed nominal','#555555','--'),
                               ('adaptive_cost','Integral calibration','#0072B2','-')]:
        cost=d[key];pick=cost.argmin(-1)
        discrepancy=abs(cost-reference)
        selected_cost=np.take_along_axis(cost,pick[...,None],-1)
        selected_error=np.take_along_axis(discrepancy,pick[...,None],-1)
        separation=cost-selected_cost-discrepancy-selected_error
        np.put_along_axis(separation,pick[...,None],np.inf,-1)
        margin=separation.min(-1);resolved=margin>1e-12
        excess=np.take_along_axis(reference,pick[...,None],-1)[...,0]-reference.min(-1)
        agreement=excess<=1e-10
        assert np.all(agreement[resolved])
        assert np.all(excess<=2*discrepancy.max(-1)+1e-10)
        arrays=[discrepancy.mean(-1),excess,agreement,resolved]
        for panel,(ax,a) in enumerate(zip(axes.flat,arrays)):
            if panel<2:
                ix=np.random.default_rng(21471).integers(0,64,(5000,64))
                lo,hi=np.quantile(a[:,ix].mean(-1),[.025,.975],axis=1)
                ax.fill_between(d['time'],lo,hi,color=color,alpha=.14,lw=0)
            ax.plot(d['time'],a.mean(1),color=color,ls=ls,label=label)
        out[key]=dict(n_decisions=int(pick.size),resolved_count=int(resolved.sum()),
            resolved_fraction=float(resolved.mean()),reference_optimal_count=int(agreement.sum()),
            reference_optimal_fraction=float(agreement.mean()),positive_margin_all_reference_optimal=True,
            mean_discrepancy=float(discrepancy.mean()),mean_excess=float(excess.mean()),
            resolved_fraction_by_time=resolved.mean(1).tolist(),agreement_by_time=agreement.mean(1).tolist())
    labels=['Mean candidate-cost discrepancy','Mean excess reference cost',
            'Reference-optimal selection fraction','Positive-margin fraction']
    titles=['(a) Prediction accuracy','(b) Selection accuracy','(c) Reference agreement','(d) Resolved ranking margins']
    for ax,label,title in zip(axes.flat,labels,titles):
        ax.set(xlabel='Replay snapshot time (s)',ylabel=label,title=title,xlim=(0,4),ylim=(0,None))
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,loc='upper right',fontsize=7.3)
    axes[0,1].legend(frameon=False,loc='upper right',fontsize=7.3)
    axes[1,0].set_ylim(0,1.06);axes[1,1].set_ylim(0,1.06)
    axes[1,0].legend(frameon=False,loc='lower right',fontsize=7.3)
    axes[1,1].legend(frameon=False,loc='upper right',fontsize=7.3)
    fig.tight_layout();fig.savefig(OUT/'calibration_evaluation.pdf');plt.close(fig)
    (DATA/'ranking_margin_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

def pendubot():
    folder=DATA/'pendubot_step';all_data={};out={}
    ix=np.random.default_rng(73211).integers(0,64,(20000,64))
    fig,axes=plt.subplots(2,2,figsize=(6.8,4.8));times=None
    for mode in MODES:
        rows=[load_npz(folder/f'step_{mode}_seed{s}.npz') for s in range(51,56)]
        assert len(rows)==5
        for r in rows:
            assert r['trace'].shape==(1001,64,5)
            assert np.isfinite(r['trace']).all() and abs(r['trace'][...,4]).max()<=.500001
            np.testing.assert_array_equal(r['initial_states'],rows[0]['initial_states'])
        all_data[mode]=rows
        arrival=np.stack([r['post_arrival'] for r in rows]);success=np.stack([r['post_success'] for r in rows])
        retained=[]
        for r in rows:
            inside=(abs(wrap_error(r['trace'][...,:2])).max(-1)<.15)&(abs(r['trace'][...,2:4]).max(-1)<.6)
            after=r['time'][:,None]>=1+r['post_arrival'][None,:]+.5-1e-10
            retained.append(r['post_success']&~((~inside)&after).any(0))
        out[mode]=dict(recovery=int(success.sum()),retained=int(np.sum(retained)),
            per_seed=success.sum(1).tolist(),time=clustered(arrival,ix),
            angle_iae=clustered(np.stack([r['post_angle_iae'] for r in rows]),ix),
            effort=clustered(np.stack([r['effort'] for r in rows]),ix),
            recovered_before_change=sum(int(r['recovered_before_change'].sum()) for r in rows))
        col,ls=STYLE[mode];label=LABELS[mode];times=rows[0]['time']
        a=np.sort(arrival[success]);n=len(a)
        short={'dynamics':'Calibrated','nominal':'Fixed','oracle':'Known gain'}[mode]
        axes[0,0].step(np.r_[0,a,19],np.r_[0,np.arange(1,n+1)/320,n/320],where='post',
                      color=col,ls=ls,label=f'{short} ({n}/320)')
        state=np.stack([r['trace'] for r in rows],axis=1).reshape(1001,320,5)
        y=abs(wrap_error(state[50:,:,:2])).sum(-1);t=times[50:]-1
        cum=np.vstack([np.zeros(320),np.cumsum(.5*np.diff(t)[:,None]*(y[1:]+y[:-1]),axis=0)])
        axes[0,1].plot(t,cum.mean(1),color=col,ls=ls,label=label)
        gain=np.stack([r['theta'][:,:,0] for r in rows],axis=1).reshape(1001,320)
        axes[1,0].plot(times,gain.mean(1),color=col,ls=ls,label=label)
        if mode=='dynamics':
            lo,hi=np.quantile(gain,[.1,.9],axis=1);axes[1,0].fill_between(times,lo,hi,color=col,alpha=.15,lw=0)
    nominal=np.stack([r['post_arrival'] for r in all_data['nominal']])
    adapted=np.stack([r['post_arrival'] for r in all_data['dynamics']])
    out['paired_fixed_minus_calibrated']=clustered(nominal-adapted,ix)
    # Representative chosen by proximity to pooled median, with deterministic first tie.
    idx=int(np.argmin(abs(adapted-np.median(adapted))))
    seed,index=np.unravel_index(idx,adapted.shape)
    out['representative']=dict(seed=51+int(seed),index=int(index),rule='closest to pooled median calibrated post-change time')
    for mode in MODES:
        r=all_data[mode][seed];col,ls=STYLE[mode]
        axes[1,1].plot(times,abs(wrap_error(r['trace'][:,index,:2])).max(-1),color=col,ls=ls,label=LABELS[mode])
    axes[0,0].set(xlabel='Time after gain loss (s)',ylabel='Fraction of 320 episodes',title='(a) Pendubot post-change recovery',xlim=(0,19),ylim=(-.02,1.05))
    axes[0,1].set(xlabel='Time after gain loss (s)',ylabel='Mean angle integral (rad s)',title='(b) Accumulated angular deviation',xlim=(0,19),ylim=(0,None))
    axes[1,0].set(xlabel='Time (s)',ylabel=r'Mean estimated gain $\hat\gamma$',title='(c) Gain adaptation',xlim=(0,5),ylim=(.76,1.04))
    axes[1,1].set(xlabel='Time (s)',ylabel=r'$\|e_q\|_\infty$ (rad)',title=f'(d) State response: seed {51+seed}, state {index}',xlim=(0,20))
    axes[1,1].axhline(.15,color='.4',ls=':',lw=.7)
    for ax in axes[1]:ax.axvline(1,color='.4',ls=':',lw=.7)
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,loc='upper left',fontsize=7)
    axes[0,1].legend(frameon=False,loc='upper left',fontsize=7)
    axes[1,0].legend(frameon=False,loc='center right',bbox_to_anchor=(1,.63),fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'pendubot_gain_step.pdf');plt.close(fig)
    (folder/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

def solver():
    folder=DATA/'solver_quality';records=json.loads((folder/'snapshot_results.json').read_text());out={}
    for key in ['original','budget','tight','extended']:
        rows=[r for r in records if r['configuration']==key];assert len(rows)==64
        out[key]=dict(n=64,converged=sum(r['status']==0 for r in rows),
            median_residuals=np.median([r['residuals'] for r in rows],axis=0).tolist(),
            mean_cost=float(np.mean([r['cost'] for r in rows])),
            mean_iterations=float(np.mean([r['iterations'] for r in rows])))
    dev=[load_npz(folder/f'tight_development_{i:02d}.npz') for i in range(8)]
    out['development']=dict(recovery=sum(int(r['success']) for r in dev),mean_time=float(np.mean([r['arrival'] for r in dev])))
    if all((folder/f'tight_confirmation_{i:02d}.npz').exists() for i in range(64)):
        rows=[load_npz(folder/f'tight_confirmation_{i:02d}.npz') for i in range(64)]
        data={k:np.stack([r[k] for r in rows]) for k in ['initial_states','arrival','success','effort','angle_iae']}
        data.update(trace=np.stack([r['trace'] for r in rows],1),theta=np.stack([r['theta'] for r in rows],1),time=rows[0]['time'])
        reference=load_npz(DATA/'static_dynamics_seed51.npz')
        np.testing.assert_array_equal(data['initial_states'],reference['initial_states'])
        ix=np.random.default_rng(64292).integers(0,64,(20000,64));status={}
        for i in range(64):
            meta=json.loads((folder/f'tight_confirmation_{i:02d}.json').read_text())
            for k,v in meta['status_counts'].items():status[k]=status.get(k,0)+v
        assert sum(status.values())==25600
        out['confirmation']=dict(recovery=int(data['success'].sum()),retained=int(extra_outcomes(data)['retained'].sum()),
            time=descriptive(data['arrival'],ix),angle_iae=descriptive(data['angle_iae'],ix),effort=descriptive(data['effort'],ix),
            status_counts=status,converged_fraction=status.get('0',0)/25600,
            paired_time_vs_egar=descriptive(data['arrival']-reference['arrival'],ix),
            median_residuals=np.median(np.concatenate([r['solver_residuals'] for r in rows]),axis=0).tolist())
        np.savez_compressed(folder/'tight_confirmation.npz',**data)
    (folder/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['margin','pendubot','solver','all'],default='all')
    args=ap.parse_args();configure()
    if args.part in ['margin','all']:margins()
    if args.part in ['pendubot','all']:pendubot()
    if args.part in ['solver','all']:solver()
