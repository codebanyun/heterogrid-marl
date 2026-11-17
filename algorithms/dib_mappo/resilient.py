"""DIB Resilient-v3: local geometric actions and outcome-aware PPO sampling.

No preferred movement, enemy state input, hand-coded tactic or new action is
introduced. Geometry is a deterministic featurization of legal local records;
all action preferences are learned. Capability masking is training-only and
never truncates the environment, critic returns, or the episode budget.
"""
from types import MethodType

import torch
from torch import nn

from algorithms.dib_mappo import tactical


def actor_support(obs,filled):
    alive=filled.expand(*filled.shape[:-1],obs.shape[-2])*(obs[...,3]>0)
    armed=(obs[...,3]>0)&(obs[...,9]>0)
    viable=armed.any(-1,keepdim=True)
    return alive*viable,alive


def episode_weights(mask):
    """Equal total policy weight per nonempty episode, not per surviving tick."""
    return mask/mask.sum((1,2),keepdim=True).clamp_min(1.)


def geometric_features(agent,obs):
    shape=obs.shape[:-1];start=agent.entity_layout.target_start
    move=agent.candidate_displacements
    size=agent.coordinate_scale
    allies=obs[...,12:start].reshape(*shape,agent.n_agents-1,8)
    targets=obs[...,start:start+11*agent.n_agents].reshape(*shape,agent.n_agents,11)
    def summarize(tokens,enemy):
        rel=tokens[...,2:5] if enemy else tokens[...,1:4]
        hp=tokens[...,5:6] if enemy else tokens[...,4:5]
        role=tokens[...,6:9] if enemy else tokens[...,5:8]
        present=(tokens[...,0]>0)&(hp.squeeze(-1)>0)
        relative=rel*size
        before=relative.norm(dim=-1,keepdim=True)
        after=relative.unsqueeze(-3)-move.view(*([1]*len(shape)),7,1,3)
        distance=after.norm(dim=-1,keepdim=True)
        repeated=lambda x:x.unsqueeze(-3).expand(*shape,7,*x.shape[-2:])
        token=torch.cat((after/10.,repeated(before)/10.,distance/10.,
            distance-repeated(before),repeated(hp),repeated(role)),-1)
        weights=present.to(obs.dtype)/(1.+before.squeeze(-1))
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        mean=(token*weights.unsqueeze(-2).unsqueeze(-1)).sum(-2)
        nearest=distance.squeeze(-1).masked_fill(~present.unsqueeze(-2),1e6).min(-1)[0]/10.
        any_present=present.any(-1)
        nearest=torch.where(any_present.unsqueeze(-1),nearest,torch.zeros_like(nearest))
        return torch.cat((mean,nearest.unsqueeze(-1),
            any_present.to(obs.dtype).unsqueeze(-1).unsqueeze(-1).expand(*shape,7,1)),-1)
    future=obs[...,:3].unsqueeze(-2)+move/size
    return torch.cat((future,move.expand(*shape,7,3),
        torch.eye(7,device=obs.device,dtype=obs.dtype).expand(*shape,7,7),
        summarize(allies,False),summarize(targets,True)),-1)


def action_logits(self,hidden,obs):
    geometry=geometric_features(self,obs)
    repeated=lambda x:x.unsqueeze(-2).expand(*x.shape[:-1],7,x.shape[-1])
    movement=self.geometric_move_head(torch.cat((repeated(hidden),repeated(obs[...,:12]),geometry),-1))
    movement=(movement*obs[...,4:7].unsqueeze(-2)).sum(-1)
    # Existing entity-shared attack scorer; no forced firing or target priority.
    attacks=tactical.action_logits(self,hidden,obs)[...,7:]
    return torch.cat((movement,attacks),-1)


def install_agent(agent):
    # Installed after the intact Tactical-v2 agent, before learner/old-MAC.
    h=agent.hidden_dim
    agent.geometric_move_head=nn.Sequential(nn.Linear(h+12+37,h),nn.Tanh(),nn.Linear(h,3,bias=False))
    nn.init.orthogonal_(agent.geometric_move_head[-1].weight,gain=.01)
    agent.register_buffer('candidate_displacements',torch.tensor(
        [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=torch.float32))
    agent.register_buffer('coordinate_scale',torch.tensor(agent.args.resilient_coordinate_scale,dtype=torch.float32))
    agent.action_logits=MethodType(action_logits,agent)
