"""Inference-only AR-EAPO actor and critic; no training framework is required."""
import numpy as np
import torch
from torch import nn
from runtime import ASSETS

class Policy:
    def __init__(self, system='acrobot'):
        with np.load(ASSETS / f'{system}_policy.npz') as data:
            self.mean = torch.tensor(data['mean'], dtype=torch.float32)
            self.std = torch.tensor(data['std'], dtype=torch.float32)
            self.clip = float(data['clip'])
            self.actor_hidden = nn.Sequential(nn.Linear(4,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
            self.critic_hidden = nn.Sequential(nn.Linear(4,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU())
            self.actor_output = nn.Linear(256,1)
            self.critic_output = nn.Linear(512,1)
            for name in ['actor_hidden','critic_hidden','actor_output','critic_output']:
                module = getattr(self,name)
                module.load_state_dict({k:torch.from_numpy(data[name+'.'+k].copy()) for k in module.state_dict()})
                module.eval()
                module.requires_grad_(False)

    def obs(self,x):
        z=torch.cat(((x[...,:2]%(2*np.pi)-np.pi)/np.pi,torch.clamp(x[...,2:],-30,30)/30),-1)
        return torch.clamp((z-self.mean)/self.std,-self.clip,self.clip)

    def action(self,x):
        return 6*torch.tanh(self.actor_output(self.actor_hidden(self.obs(x)))).squeeze(-1)

    def value(self,x):
        return self.critic_output(self.critic_hidden(self.obs(x))).squeeze(-1)
