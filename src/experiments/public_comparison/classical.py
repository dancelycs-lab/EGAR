"""Classical baselines at the manuscript's development-selected gains."""
from complete_baselines import *
if __name__ == '__main__':
    states=np.load(BASE/'heldout_initial_states.npy')
    class PublicPFL:
        def action(self,x):return PFL(3.,.5,1.).action(x.float())
    # Xin constructor order is (kp, kd, kv).
    for name,controller in [('xin',Xin([3.,.1,1.])),('pfl',PublicPFL())]:
        result=simulate(controller,states)
        np.savez_compressed(HERE/f'acrobot_{name}.npz',**result)
        print(name,summary(result),flush=True)
