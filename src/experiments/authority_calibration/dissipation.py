"""Reported development-set dissipative operating-limit experiment."""
from screen import OUT,experiment,np
if __name__ == '__main__':
    states=np.load(OUT/'initial_states.npy')
    for mode in ['dynamics','oracle']:
        result=experiment(states,mode,[.833333333333,1/3,1/3],observer=True)
        np.savez_compressed(OUT/f'dissipation_{mode}.npz',**result)
        print(mode,int(result['success'].sum()),flush=True)
    result=experiment(states,'oracle',[.833333333333,1/3,1/3],sigma=(0.,0.),observer=False)
    np.savez_compressed(OUT/'dissipation_exact_state.npz',**result)
    print('exact-state oracle',int(result['success'].sum()),flush=True)
