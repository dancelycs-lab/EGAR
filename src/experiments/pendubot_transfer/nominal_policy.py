"""Clipped pretrained Pendubot policy with the common high-rate local LQR."""
from aligned_confirm import simulate_aligned,OUT,np
if __name__ == '__main__':
    states=np.load(OUT/'confirmation_states.npy')
    result=simulate_aligned(states,.5,'policy')
    np.savez_compressed(OUT/'confirm_policy.npz',**result)
    print('Pendubot policy:',int(result['success'].sum()),float(result['arrival'].mean()),flush=True)
