"""Generate the local controller matrices used by the interval certificate."""
from transfer_screen import A,B,P,K,ROOT,np
np.savez(ROOT/'lqr_matrices.npz',A=A,B=B,P=P,K=K.numpy())
