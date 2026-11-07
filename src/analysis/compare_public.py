"""Recompute public comparisons from recorded trajectories; never synthesize data."""
import json
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from plot_results import (HERE, ROOT, ACRO, PEND, GLOBAL, OUT, configure,
                          load_npz, wrap_error, mechanical_energy)

PUBLIC = ROOT / 'experiments/public_comparison'
STYLE = {'EGAR': ('#0072B2', '-'), 'acados NMPC': ('#D55E00', '--'),
         'Frozen policy + LQR': ('#555555', ':'),
         'Xin–Kaneda + LQR': ('#009E73', '-.'),
         'Spong-type PFL + LQR': ('#882255', (0, (5, 2, 1, 2))),
         'iLQR-MPC': ('#882255', '-.')}


def legacy(path):
    d = load_npz(path)
    d['time'] = .002 + np.arange(d['trace'].shape[0])*.02
    return d


def collect():
    ac = {'EGAR': legacy(ACRO/'aligned_core_energy.npz'),
          'Frozen policy + LQR': legacy(GLOBAL/'heldout_L0.5_actor_lqr.npz'),
          'Xin–Kaneda + LQR': load_npz(PUBLIC/'acrobot_xin.npz'),
          'Spong-type PFL + LQR': load_npz(PUBLIC/'acrobot_pfl.npz')}
    rows = json.loads((GLOBAL/'acados_heldout_H1.2_results.json').read_text())
    assert len(rows) == 64
    traces = [load_npz(GLOBAL/(r['name']+'.npz'))['trace'][:, [0,1,2,3,5]] for r in rows]
    ac['acados NMPC'] = {k: np.array([r[k] for r in rows]) for k in ['arrival','success','effort']}
    ac['acados NMPC'].update(trace=np.stack(traces,1), time=.002+np.arange(1000)*.02)
    pe = {'EGAR': legacy(PEND/'aligned_confirm_energy.npz'),
          'Frozen policy + LQR': legacy(PEND/'confirm_policy.npz')}
    rows = [load_npz(PEND/f'ilqr_wrapped_strengthened_confirmation_{i}.npz') for i in range(64)]
    pe['iLQR-MPC'] = {k: np.array([r[k] for r in rows]) for k in ['arrival','success','effort']}
    pe['iLQR-MPC'].update(trace=np.stack([r['trace'] for r in rows],1),time=.002+np.arange(1000)*.02)
    pe['acados NMPC'] = load_npz(PUBLIC/'pendubot_acados_confirmation.npz')
    for methods, path in [(ac, GLOBAL/'heldout_initial_states.npy'),
                          (pe, PEND/'confirmation_states.npy')]:
        initial = np.load(path)
        assert initial.shape == (64, 4) and np.isfinite(initial).all()
        for data in methods.values():
            if 'initial_states' in data:
                np.testing.assert_allclose(data['initial_states'], initial, rtol=0, atol=1e-12)
            data['initial_states'] = initial
    return {'Acrobot': ac, 'Pendubot': pe}


def extra_outcomes(data):
    """Descriptive quadrature and retention on actual archived state samples.

    The endpoint extension is used only for quadrature. It is never used as
    evidence of a simulated state, threshold satisfaction, or sustained capture.
    """
    t = np.asarray(data['time'], dtype=float)
    state = np.asarray(data['trace'][..., :4], dtype=float)
    assert state.shape == (len(t), 64, 4) and np.isfinite(state).all()
    assert np.all(np.diff(t) > 0) and t[0] >= 0 and t[-1] <= 20 + 1e-10
    assert np.max(np.diff(t)) <= .02000001
    angle = np.abs(wrap_error(state[..., :2]))
    integrand = angle.sum(-1)
    tq, yq = t.copy(), integrand.copy()
    if t[0] > 1e-12:
        tq = np.r_[0., tq]
        yq = np.vstack([np.abs(wrap_error(data['initial_states'][:, :2])).sum(-1), yq])
    else:
        np.testing.assert_allclose(state[0], data['initial_states'], rtol=0, atol=1e-10)
    if t[-1] < 20 - 1e-12:
        tq = np.r_[tq, 20.]
        yq = np.vstack([yq, yq[-1]])
    assert np.isclose(tq[0], 0) and np.isclose(tq[-1], 20)
    angle_iae = np.trapezoid(yq, tq, axis=0)
    inside = (angle.max(-1) < .15) & (np.abs(state[..., 2:4]).max(-1) < .6)
    success = np.asarray(data['success'], dtype=bool)
    arrival = np.asarray(data['arrival'], dtype=float)
    np.testing.assert_array_equal(success, arrival < 20)
    after_dwell = t[:, None] >= arrival[None, :] + .5 - 1e-10
    retained = success & after_dwell.any(0) & ~((~inside) & after_dwell).any(0)
    first_exit = np.full(64, np.nan)
    for i in np.flatnonzero(success & ~retained):
        ix = np.flatnonzero((~inside[:, i]) & after_dwell[:, i])
        if len(ix):
            first_exit[i] = t[ix[0]]
    assert np.all(retained <= success) and np.isfinite(angle_iae).all()
    assert np.all((angle_iae >= 0) & (angle_iae <= 40*np.pi + 1e-9))
    return dict(angle_iae=angle_iae, retained=retained, first_exit=first_exit,
                first_sample=float(t[0]), last_sample=float(t[-1]),
                endpoint_extension=float(max(0., 20-t[-1])),
                endpoint_error_bound=float(2*np.pi*max(0., 20-t[-1])))


def descriptive(values, bootstrap_indices):
    values = np.asarray(values, dtype=float)
    assert values.shape == (64,) and np.isfinite(values).all()
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                median=float(np.median(values)), minimum=float(values.min()),
                maximum=float(values.max()),
                bootstrap95=np.quantile(values[bootstrap_indices].mean(1), [.025, .975]).tolist())


def summarize(systems):
    rng = np.random.default_rng(74620)
    ix = rng.integers(0,64,(20000,64)); out = {}; episodes = []
    for system, methods in systems.items():
        base=methods['EGAR']; out[system]={}
        extra = {name: extra_outcomes(data) for name, data in methods.items()}
        for name,d in methods.items():
            assert d['arrival'].shape == (64,) and np.isfinite(d['trace']).all()
            assert np.max(abs(d['trace'][...,4])) <= .500001
            delta=d['arrival']-base['arrival']
            out[system][name] = dict(n=64, success=int(d['success'].sum()),
                capped_time=float(d['arrival'].mean()), effort=float(d['effort'].mean()),
                baseline_minus_egar_time=float(delta.mean()),
                paired_bootstrap95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),
                wins_egar=int((delta>1e-9).sum()), wins_baseline=int((delta<-1e-9).sum()))
            ex = extra[name]
            angle_delta = ex['angle_iae'] - extra['EGAR']['angle_iae']
            out[system][name].update(
                retained=int(ex['retained'].sum()),
                lost_after_dwell=int((np.asarray(d['success'], dtype=bool) & ~ex['retained']).sum()),
                angle_iae=descriptive(ex['angle_iae'], ix),
                time_summary=descriptive(d['arrival'], ix),
                effort_summary=descriptive(d['effort'], ix),
                baseline_minus_egar_angle_iae=descriptive(angle_delta, ix),
                first_saved_time=ex['first_sample'], last_saved_time=ex['last_sample'],
                quadrature_endpoint_extension=ex['endpoint_extension'],
                quadrature_endpoint_error_bound=ex['endpoint_error_bound'])
            for i in range(64):
                episodes.append(dict(system=system, controller=name, initial_state_index=i,
                    recovery=int(d['success'][i]), retained=int(ex['retained'][i]),
                    capped_time_s=float(d['arrival'][i]), effort_N2m2s=float(d['effort'][i]),
                    angle_iae_rad_s=float(ex['angle_iae'][i]),
                    first_saved_exit_after_dwell_s=(float(ex['first_exit'][i])
                        if np.isfinite(ex['first_exit'][i]) else '')))
        med=np.median(base['arrival']);index=int(np.argmin(abs(base['arrival']-med)))
        out[system]['representative_index']=index
    (PUBLIC/'comparison_analysis.json').write_text(json.dumps(out,indent=2))
    with (PUBLIC/'comparison_episode_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader(); writer.writerows(episodes)
    print(json.dumps(out,indent=2))


def display_name(name):
    return {'Frozen policy + LQR': 'Pretrained policy + LQR'}.get(name, name)


def recovery(systems):
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.0),sharey=True)
    for ax,(system,methods) in zip(axes,systems.items()):
        for name in ['EGAR','acados NMPC','iLQR-MPC','Frozen policy + LQR','Xin–Kaneda + LQR','Spong-type PFL + LQR']:
            if name not in methods: continue
            d=methods[name]; ts=np.sort(d['arrival'][d['success']])
            x=np.r_[0.,ts,20.]; y=np.r_[0.,np.arange(1,len(ts)+1)/64,len(ts)/64]
            color,style=STYLE[name]
            ax.step(x,y,where='post',color=color,linestyle=style,
                    label=f'{display_name(name)} ({len(ts)}/64)')
            if name in ['EGAR','acados NMPC']:
                ax.annotate(f'{len(ts)}/64',xy=(20,len(ts)/64),xytext=(-4,5),
                            textcoords='offset points',ha='right',color=color,
                            fontsize=8.5,fontweight='bold' if name=='EGAR' else 'normal')
        ax.set(xlim=(0,20),ylim=(-.025,1.12),xlabel='Dwell-qualified entry time (s)',title=system)
        ax.set_xticks([0,5,10,15,20])
        ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=7.7,loc='center right',
                                     bbox_to_anchor=(1,.43 if system=='Pendubot' else .35))
    axes[0].set_ylabel('Fraction of 64 initial states')
    fig.tight_layout();fig.savefig(OUT/'public_recovery.pdf');plt.close(fig)


def cumulative_angle(data):
    """Same full-episode quadrature as Table 3, retaining every initial state."""
    t=np.asarray(data['time']); y=np.abs(wrap_error(data['trace'][...,:2])).sum(-1)
    if t[0]>1e-12:
        t=np.r_[0.,t]
        y=np.vstack([np.abs(wrap_error(data['initial_states'][:,:2])).sum(-1),y])
    if t[-1]<20-1e-12:
        t=np.r_[t,20.];y=np.vstack([y,y[-1]])
    result=np.vstack([np.zeros((1,64)),np.cumsum((y[1:]+y[:-1])*np.diff(t)[:,None]/2,axis=0)])
    np.testing.assert_allclose(result[-1],extra_outcomes(data)['angle_iae'],rtol=1e-12,atol=1e-10)
    return t,result


def bootstrap_weights(seed=20261107, repetitions=5000):
    ix=np.random.default_rng(seed).integers(0,64,(repetitions,64))
    return np.stack([np.bincount(row,minlength=64) for row in ix])/64


def mean_band(values, weights):
    draws=values@weights.T
    low,high=np.quantile(draws,[.025,.975],axis=1)
    return values.mean(1),low,high


def population_error(systems):
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.2),sharey=True)
    weights=bootstrap_weights();saved={};largest=0;handles={}
    for ax,(system,methods) in zip(axes,systems.items()):
        for name in ['EGAR','acados NMPC','Frozen policy + LQR','Xin–Kaneda + LQR','Spong-type PFL + LQR','iLQR-MPC']:
            if name not in methods:continue
            t,values=cumulative_angle(methods[name]);mean,low,high=mean_band(values,weights)
            color,style=STYLE[name]
            ax.fill_between(t,low,high,color=color,alpha=.10,lw=0)
            line,=ax.plot(t,mean,color=color,linestyle=style,label=display_name(name))
            handles[name]=line;largest=max(largest,float(high.max()))
            key=system+'__'+name
            saved[key+'__time']=t;saved[key+'__mean']=mean;saved[key+'__lower']=low;saved[key+'__upper']=high
        ax.set(xlim=(0,20),xlabel='Time (s)',title=system)
        ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.22)
        ax.legend(loc='upper left',frameon=False,fontsize=8)
    axes[0].set_ylim(0,largest*1.06)
    axes[0].set_ylabel(r'Mean cumulative $J_q(t)$ (rad s)')
    fig.tight_layout();fig.savefig(OUT/'public_cumulative_error.pdf');plt.close(fig)
    np.savez_compressed(PUBLIC/'population_error_curves.npz',**saved)


def responses(systems):
    fig,axs=plt.subplots(5,2,figsize=(6.8,7.0),sharex=True)
    estar=mechanical_energy(np.array([np.pi,0,0,0]))
    for col,(system,methods) in enumerate(systems.items()):
        index=int(np.argmin(abs(methods['EGAR']['arrival']-np.median(methods['EGAR']['arrival']))))
        chosen=['EGAR', 'acados NMPC', 'Frozen policy + LQR']
        for name in chosen:
            d=methods[name]; tr=d['trace'][:,index]; color,style=STYLE[name]
            vals=[np.abs(wrap_error(tr[:,:2])).max(1),np.abs(tr[:,2:4]).max(1),
                  mechanical_energy(tr[:,:4])-estar,tr[:,4]]
            for row,v in enumerate(vals): axs[row,col].plot(d['time'],v,color=color,linestyle=style,label=display_name(name))
            tq,cum=cumulative_angle(d)
            axs[4,col].plot(tq,cum[:,index],color=color,linestyle=style,label=display_name(name))
            if d['success'][index]:
                for row in [0,1,4]:axs[row,col].axvline(d['arrival'][index],color=color,ls=(0,(2,3)),lw=.8,alpha=.65)
        for row,bound in enumerate([.15,.6,0,.5]):
            axs[row,col].axhline(bound,color='.3',ls=(0,(2,3)),lw=.6)
            axs[row,col].grid(alpha=.2)
            axs[row,col].set_xlim(0,20)
            axs[row,col].set_xticks([0,5,10,15,20])
        axs[3,col].axhline(-.5,color='.3',ls=(0,(2,3)),lw=.6)
        times=[]
        for name,short in [('EGAR','EGAR'),('acados NMPC','acados'),('Frozen policy + LQR','Policy')]:
            d=methods[name]
            times.append(f'{short}: '+(f"{d['arrival'][index]:.2f} s" if d['success'][index] else 'not recovered'))
        if all(not methods[name]['success'][index] for name in chosen[1:]):
            time_note=f"EGAR: {methods['EGAR']['arrival'][index]:.2f} s; baselines unrecovered"
        else:
            time_note='; '.join(times)
        axs[0,col].set_title(f'{system}: initial state {index}\n'+time_note,fontsize=8.5)
        axs[4,col].set(xlabel='Time (s)',xlim=(0,20));axs[4,col].grid(alpha=.2)
        axs[4,col].set_xticks([0,5,10,15,20]);axs[4,col].set_ylim(bottom=0)
        axs[4,col].legend(loc='upper left',frameon=False,fontsize=7)
        if system=='Acrobot':
            for row,limits in [(1,(-1,35)),(2,(-5,5))]:
                inset=axs[row,col].inset_axes([.08,.42,.40,.53])
                for name in chosen:
                    data=methods[name];tr=data['trace'][:,index]
                    y=(np.abs(tr[:,2:4]).max(1) if row==1
                       else mechanical_energy(tr[:,:4])-estar)
                    color,style=STYLE[name]
                    inset.plot(data['time'],y,color=color,ls=style,lw=.8)
                inset.set(xlim=(0,5),ylim=limits,xticks=[0,2.5,5])
                inset.tick_params(labelsize=5.5,pad=1,length=2)
                inset.grid(alpha=.15)
                inset.text(.96,.9,'0--5 s',transform=inset.transAxes,ha='right',fontsize=6,
                           bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3))
                inset.axhline(0,color='.4',ls=':',lw=.5)
    for ax,label in zip(axs[:,0],[r'$\|e_q\|_\infty$ (rad)',r'$\|v\|_\infty$ (rad/s)',r'$E-E_\star$ (J)',r'$u$ (N m)',r'$J_q(t)$ (rad s)']):ax.set_ylabel(label)
    fig.tight_layout(h_pad=.35,w_pad=1);fig.savefig(OUT/'public_responses.pdf');plt.close(fig)


def noisy_response():
    methods={name:legacy(ACRO/f'final_aligned_gain_{key}.npz') for name,key in
             [('Integral calibration','dynamics'),('Fixed nominal','nominal'),('True parameters','oracle')]}
    d=methods['Integral calibration'];index=int(np.argmin(abs(d['arrival']-np.median(d['arrival']))))
    fig,grid=plt.subplots(2,2,figsize=(6.8,4.4))
    axes=grid.ravel()
    for (name,d),(color,style) in zip(methods.items(),[('#0072B2','-'),('#882255','-.'),('#009E73','--')]):
        tr=d['trace'][:,index];t=d['time']
        axes[0].plot(t,np.abs(wrap_error(tr[:,:2])).max(1),c=color,ls=style,label=display_name(name))
        axes[1].plot(t,np.abs(tr[:,2:4]).max(1),c=color,ls=style)
        axes[2].plot(t,d['theta'][:,index,0],c=color,ls=style)
    theta=methods['Integral calibration']['theta'][...,0]
    t=methods['Integral calibration']['time']
    axes[3].fill_between(t,np.quantile(theta,.1,axis=1),np.quantile(theta,.9,axis=1),color='#0072B2',alpha=.16,lw=0,label='10--90% range')
    axes[3].plot(t,theta.mean(1),color='#0072B2',label='Mean estimate')
    axes[3].axhline(1,color='#882255',ls='-.',lw=.9)
    axes[3].axhline(5/6,color='#009E73',ls='--',lw=.9)
    axes[3].legend(frameon=False,fontsize=7.5,loc='upper right')
    for ax,bound in zip(axes[:2],[.15,.6]):ax.axhline(bound,c='.3',ls=':',lw=.7)
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,20)
    for ax,label in zip(axes,[r'$\|e_q\|_\infty$ (rad)',r'$\|v\|_\infty$ (rad/s)',r'$\hat\gamma$ (selected state)',r'$\hat\gamma$ (64 states)']):ax.set_ylabel(label)
    for ax,title in zip(axes,['(a) Angular regulation','(b) Velocity regulation','(c) Individual calibration','(d) Population calibration']):ax.set_title(title,fontsize=8.5)
    hh,ll=axes[0].get_legend_handles_labels()
    axes[2].legend(hh,ll,frameon=False,loc='upper right',fontsize=8,bbox_to_anchor=(1,.9))
    axes[2].set_xlabel('Time (s)');axes[3].set_xlabel('Time (s)')
    for ax in axes:ax.set_xticks([0,5,10,15,20])
    for ax in axes[2:]:ax.set_xlim(0,4);ax.set_xticks([0,1,2,3,4])
    fig.tight_layout(h_pad=.25);fig.savefig(OUT/'noisy_response.pdf');plt.close(fig)


def calibration_evaluation():
    d=load_npz(PUBLIC/'calibration_diagnostic.npz');ref=d['reference_cost']
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.7))
    ix=np.random.default_rng(21471).integers(0,64,(5000,64))
    for key,label,color,style in [('nominal_cost','Fixed nominal','#555555','--'),('adaptive_cost','Integral calibration','#0072B2','-')]:
        c=d[key];pick=np.argmin(c,axis=-1)
        metrics=[abs(c-ref).mean(-1),np.take_along_axis(ref,pick[...,None],axis=-1)[...,0]-ref.min(-1)]
        for ax,metric in zip(axes,metrics):
            mean=metric.mean(1);boots=metric[:,ix].mean(-1);lo,hi=np.quantile(boots,[.025,.975],axis=1)
            ax.fill_between(d['time'],lo,hi,color=color,alpha=.14,lw=0)
            ax.plot(d['time'],mean,color=color,ls=style,label=label)
    for ax in axes:ax.set_xlim(0,4);ax.set_xlabel('Replay snapshot time (s)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean candidate-cost discrepancy')
    axes[1].set_ylabel('Mean excess reference cost')
    axes[0].set_title('(a) Prediction accuracy');axes[1].set_title('(b) Candidate selection')
    for ax in axes:
        ax.set_ylim(0,ax.get_ylim()[1]*1.20)
        ax.legend(frameon=False,loc='upper right',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'calibration_evaluation.pdf');plt.close(fig)


def gain_noise_comparison():
    """Complete external confirmation; never plot a partially completed set."""
    initial=np.load(ACRO/'gain_confirmation_states.npy')
    methods={
        'EGAR':legacy(ACRO/'final_aligned_gain_dynamics.npz'),
        'acados NMPC':load_npz(PUBLIC/'gain_noise_acados.npz'),
        'Frozen policy + LQR':load_npz(PUBLIC/'gain_noise_policy.npz'),
        'EGAR, fixed predictor':legacy(ACRO/'final_aligned_gain_nominal.npz'),
        'EGAR, true parameters':legacy(ACRO/'final_aligned_gain_oracle.npz')}
    ix=np.random.default_rng(74620).integers(0,64,(20000,64))
    extras={};summary={};episodes=[]
    for name,data in methods.items():
        assert data['arrival'].shape==(64,)
        if 'initial_states' in data:np.testing.assert_array_equal(data['initial_states'],initial)
        data['initial_states']=initial
        assert np.isfinite(data['trace']).all() and abs(data['trace'][...,4]).max()<=.500001
        ex=extra_outcomes(data);extras[name]=ex
        summary[name]=dict(n=64,recovery=int(data['success'].sum()),retained=int(ex['retained'].sum()),
            time=descriptive(data['arrival'],ix),effort=descriptive(data['effort'],ix),
            angle_iae=descriptive(ex['angle_iae'],ix),
            paired_time_vs_egar=descriptive(data['arrival']-methods['EGAR']['arrival'],ix),
            paired_angle_vs_egar=descriptive(ex['angle_iae']-extras['EGAR']['angle_iae'],ix))
        for i in range(64):
            episodes.append(dict(controller=name,initial_state_index=i,recovery=int(data['success'][i]),
                retained=int(ex['retained'][i]),capped_time_s=float(data['arrival'][i]),
                effort_N2m2s=float(data['effort'][i]),angle_iae_rad_s=float(ex['angle_iae'][i])))
    progress=json.loads((PUBLIC/'gain_noise_acados_progress.json').read_text())
    assert len(progress)==64 and [r['index'] for r in progress]==list(range(64))
    statuses={}
    for row in progress:
        for key,val in row['status_counts'].items():statuses[key]=statuses.get(key,0)+val
    assert sum(statuses.values())==640000
    record=dict(methods=summary,acados_status_counts=statuses,n=64,seed=51,gain=5/6)
    (PUBLIC/'gain_noise_comparison.json').write_text(json.dumps(record,indent=2))
    with (PUBLIC/'gain_noise_episode_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.25))
    weights=bootstrap_weights();handles=[];labels=[];saved={}
    names=['EGAR','acados NMPC','Frozen policy + LQR','EGAR, fixed predictor']
    for name in names:
        data=methods[name]
        color,style=STYLE[name] if name in STYLE else ('#882255','-.')
        arrivals=np.sort(data['arrival'][data['success']]);n=len(arrivals)
        line,=axes[0].step(np.r_[0.,arrivals,20.],np.r_[0.,np.arange(1,n+1)/64,n/64],
                          where='post',color=color,linestyle=style)
        short_name={'Frozen policy + LQR':'Pretrained + LQR','EGAR, fixed predictor':'EGAR, fixed'}.get(name,name)
        labels.append(f'{short_name} ({n}/64)');handles.append(line)
        t,values=cumulative_angle(data);mean,low,high=mean_band(values,weights)
        axes[1].fill_between(t,low,high,color=color,alpha=.12,lw=0)
        axes[1].plot(t,mean,color=color,linestyle=style)
        saved[name+'__time']=t;saved[name+'__mean']=mean;saved[name+'__lower']=low;saved[name+'__upper']=high
    axes[0].set(xlabel='Dwell-qualified entry time (s)',ylabel='Fraction of 64 initial states',
                title='(a) Recovery under gain loss and noise',ylim=(-.025,1.08))
    axes[1].set(xlabel='Time (s)',ylabel=r'Mean cumulative $J_q(t)$ (rad s)',
                title='(b) Accumulated angular deviation',ylim=(0,None))
    for ax in axes:ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.22)
    axes[0].legend(handles,labels,frameon=False,fontsize=8,loc='center right',bbox_to_anchor=(1,.64),
                   handlelength=1.5,handletextpad=.5)
    axes[1].legend(handles,[display_name(name) for name in names],frameon=False,fontsize=8,loc='upper left')
    fig.tight_layout();fig.savefig(OUT/'gain_noise_public.pdf');plt.close(fig)
    np.savez_compressed(PUBLIC/'gain_noise_comparison_curves.npz',**saved)
    print('Noisy external comparison:',json.dumps(record,indent=2))


if __name__=='__main__':
    systems=collect();summarize(systems)
    if '--metrics-only' not in sys.argv:
        configure();recovery(systems);responses(systems);population_error(systems);noisy_response();calibration_evaluation()
        if (PUBLIC/'gain_noise_acados.npz').exists():gain_noise_comparison()
