"""Selected public iLQR-MPC configuration (1 s, ten iterations)."""
from ilqr_mpc_baseline import OUT,run_set,np
if __name__ == '__main__':
    print(run_set(np.load(OUT/'confirmation_states.npy'),1.,2.,'ilqr_wrapped_strengthened_confirmation',max_iter=10),flush=True)
