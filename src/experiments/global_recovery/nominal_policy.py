"""Clipped pretrained Acrobot policy with the common high-rate local LQR."""
from transfer_screen import *
if __name__ == '__main__':
    states=np.load(ROOT/'heldout_initial_states.npy')
    result=run(Variant(local=True),states,.5,duration=20.)
    np.savez_compressed(ROOT/'heldout_L0.5_actor_lqr.npz',**result)
    print('Acrobot policy:',int(result['success'].sum()),float(result['arrival'].mean()),flush=True)
