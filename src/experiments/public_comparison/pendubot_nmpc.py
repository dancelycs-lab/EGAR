"""Selected public Pendubot NMPC configuration (2.4 s, 96 nodes)."""
from acados_pendubot import *
if __name__ == '__main__':
    result=run(np.load(PEND/'confirmation_states.npy'),2.4,'pendubot_acados_confirmation')
    print(result,flush=True)
