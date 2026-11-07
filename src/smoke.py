"""Small end-to-end run without the external NMPC/iLQR toolchains."""
from pathlib import Path
import sys,subprocess,os
import numpy as np
from runtime import RESULTS
HERE=Path(__file__).resolve().parent
# Separate processes avoid the two systems' intentionally different screen modules.
for system in ['acrobot','pendubot']:
    code='''
import sys,numpy as np,torch
from pathlib import Path
from runtime import ASSETS,RESULTS
system=sys.argv[1];root=Path(sys.argv[2])
sys.path.insert(0,str(root/'experiments'/('authority_calibration' if system=='acrobot' else 'pendubot_transfer')))
if system=='acrobot':
 from aligned_rollout import AlignedRollout
 from screen import simulate
 states=np.load(ASSETS/'acrobot_nominal.npy')[:2]
 result=simulate(AlignedRollout(2),states,duration=.1)
else:
 from aligned_confirm import aligned_action
 from screen import integrate,base
 states=np.load(ASSETS/'pendubot_nominal.npy')[:2]
 x=torch.tensor(states,dtype=torch.float64)
 with torch.no_grad():
  for k in range(50):
   if k%25==0:u=aligned_action(x.float(),.5,'energy').double()
   x=integrate(x,u,.002)
 result={'trace':x.numpy(),'command':u.numpy()}
assert np.isfinite(result['trace']).all()
print(system,'short simulation completed')
'''
    subprocess.run([sys.executable,'-c',code,system,str(HERE)],check=True,env=os.environ)
print('Both systems: model loading, candidate selection and plant integration passed.')
