"""HeteroGrid-specific DIB variant; not an equivalent upstream optimization.

Only ``robust_comm_v1`` installs these methods. Supervised entity validity is
training-only. Execution scores hypotheses against *its own* observed fields;
it never reads another agent's observation, simulator state or target tracks.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class EntityLayout:
    """Compact observation: self[alive,xyz,hp], allies[seen,xyz,hp],
    enemy records[present,visible,xyz,hp,age,reported]. Fixed-role one-hots,
    public time, static capabilities and unused message vectors are not targets.
    Input observations and action masks are unchanged.
    """
    def __init__(self, n_agents, obs_dim):
        self.n = n_agents
        self.raw = obs_dim
        self.width = 5 + 5 * (n_agents - 1) + 8 * n_agents
        self.target_start = 12 + 8 * (n_agents - 1)
        if obs_dim != self.target_start + 11 * n_agents + 10 * n_agents:
            raise ValueError("robust_comm_v1 requires balanced HeteroGrid message_dim=8 layout")

    def pack(self, obs):
        own = torch.cat(((obs[..., 3:4] > 0).to(obs.dtype), obs[..., :4]), -1)
        allies = obs[..., 12:self.target_start].reshape(*obs.shape[:-1], self.n - 1, 8)[..., :5]
        targets = obs[..., self.target_start:self.target_start + 11*self.n]
        targets = targets.reshape(*obs.shape[:-1], self.n, 11)
        targets = torch.cat((targets[..., :6], targets[..., 9:11]), -1)
        return torch.cat((own, allies.flatten(-2), targets.flatten(-2)), -1)

    def fields(self, value):
        if value.shape[-1] % self.width:
            raise ValueError("Invalid structured target width")
        slots = value.shape[-1] // self.width
        x = value.reshape(*value.shape[:-1], slots, self.width)
        own = x[..., :5]
        allies = x[..., 5:5 + 5*(self.n-1)].reshape(*x.shape[:-1], self.n-1, 5)
        targets = x[..., 5 + 5*(self.n-1):].reshape(*x.shape[:-1], self.n, 8)
        return own, allies, targets

    @staticmethod
    def mean(value, weight, dimensions):
        return (value * weight).sum(dimensions) / weight.expand_as(value).sum(dimensions).clamp_min(1.)

    def transform(self, raw):
        own, allies, targets = self.fields(raw)
        own = own.sigmoid()
        allies = torch.cat((allies[..., :1].sigmoid(), allies[..., 1:4].tanh(), allies[..., 4:].sigmoid()), -1)
        targets = torch.cat((targets[..., :2].sigmoid(), targets[..., 2:5].tanh(), targets[..., 5:].sigmoid()), -1)
        return torch.cat((own, allies.flatten(-2), targets.flatten(-2)), -1).flatten(-2)

    def error(self, raw, target, logstd=None, filtering=False):
        """Per-sample, per-hypothesis grouped proper labels/conditional values.

        Negative presence labels remain supervised. Only undefined attributes
        are masked. A real coordinate zero is a valid regression target.
        ``filtering`` excludes self position (largely determined by own action).
        Positive classification labels get fixed weight 3; no test-set fitting.
        """
        raw, target = torch.broadcast_tensors(raw, target)
        po, pa, pe = self.fields(raw)
        yo, ya, ye = self.fields(target)
        vo, va, ve = self.fields(self.transform(raw))
        if logstd is not None:
            so, sa, se = self.fields(torch.broadcast_tensors(logstd, raw)[0].clamp(-2., 1.))
        else:
            so = sa = se = None
        alive = yo[..., :1]
        ally = alive.unsqueeze(-2) * ya[..., :1]
        record = alive.unsqueeze(-2) * ye[..., :1]
        def binary(p, y):
            return F.binary_cross_entropy_with_logits(p, y, reduction='none') * (1. + 2.*y)
        def regression(p, y, std, scale):
            e = F.smooth_l1_loss((p-y)/scale, torch.zeros_like(p), reduction='none')
            if std is not None:
                # Relative Gaussian NLL: shifted by a constant so its value is
                # nonnegative; no negative easy-field NLL dominates loss caps.
                e = .5*e*torch.exp(-2.*std) + std + 2.
            return e
        own_start = 4 if filtering else 1
        own_scale = raw.new_tensor([.2,.2,.2,.25])[own_start-1:]
        groups = [
            (.5, binary(po[..., :1], yo[..., :1]), torch.ones_like(alive), (-1,-2)),
            (.5, regression(vo[..., own_start:],yo[..., own_start:],so[..., own_start:] if so is not None else None,own_scale),alive,(-1,-2)),
            (1.,binary(pa[..., :1],ya[..., :1]),alive.unsqueeze(-2),(-1,-2,-3)),
            (1.,regression(va[..., 1:],ya[..., 1:],sa[..., 1:] if sa is not None else None,raw.new_tensor([.2,.2,.2,.25])),ally,(-1,-2,-3)),
            (2.,binary(pe[..., :1],ye[..., :1]),alive.unsqueeze(-2),(-1,-2,-3)),
            (.5,binary(torch.cat((pe[..., 1:2],pe[..., 7:8]),-1),torch.cat((ye[..., 1:2],ye[..., 7:8]),-1)),record,(-1,-2,-3)),
            (2.,regression(ve[..., 2:7],ye[..., 2:7],se[..., 2:7] if se is not None else None,raw.new_tensor([.2,.2,.2,.25,.5])),record,(-1,-2,-3))]
        numerator, denominator = 0., 0.
        for weight, value, valid, axes in groups:
            support = valid.expand_as(value).sum(axes) > 0
            numerator = numerator + weight*self.mean(value,valid,axes)
            denominator = denominator + weight*support.to(raw.dtype)
        return numerator / denominator.clamp_min(1.)

    def valid_features(self, target):
        own, allies, targets = self.fields(target)
        alive = own[..., :1]
        mo = torch.cat((torch.ones_like(alive),alive.expand_as(own[...,1:])), -1)
        ma = torch.cat((alive.unsqueeze(-2).expand_as(allies[...,:1]),
                        (alive.unsqueeze(-2)*allies[...,:1]).expand_as(allies[...,1:])), -1)
        present = alive.unsqueeze(-2)*targets[...,:1]
        me = torch.cat((alive.unsqueeze(-2).expand_as(targets[...,:1]),present.expand_as(targets[...,1:])), -1)
        return torch.cat((mo,ma.flatten(-2),me.flatten(-2)),-1).flatten(-2)

    def binary_features(self, reference):
        one = reference.new_zeros(self.width)
        one[0] = 1
        for j in range(self.n-1):one[5+5*j] = 1
        for j in range(self.n):
            start = 5+5*(self.n-1)+8*j
            one[start] = one[start+1] = one[start+7] = 1
        return one.repeat(reference.shape[-1]//self.width)


def probe_entropy(agent, mu, logstd):
    """Compact-target uncertainty, not entropy over unused raw message fields."""
    probabilities = agent.entity_layout.transform(mu)
    mask = agent.entity_layout.valid_features(probabilities).detach()
    continuous = .5*(math.log(2.*math.pi*math.e)+2.*logstd.clamp(-2.,1.))
    p = probabilities.clamp(1e-6,1-1e-6)
    discrete = -p*p.log()-(1-p)*(1-p).log()
    binary = agent.entity_layout.binary_features(mu).bool()
    entropy = torch.where(binary,discrete,continuous)
    return (entropy*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1.)


def local_normalized_error(agent, prediction, target, raw_obs):
    """AHS/teacher local dynamics retain upstream normalized output units,
    but ignore static/message fields and undefined entity attributes."""
    weights = torch.zeros_like(raw_obs)
    weights[..., :4] = 1
    start = agent.entity_layout.target_start
    allies = raw_obs[...,12:start].reshape(*raw_obs.shape[:-1],agent.n_agents-1,8)
    wa = weights[...,12:start].reshape_as(allies)
    wa[...,:1] = 1
    wa[...,1:5] = allies[...,:1]
    targets = raw_obs[...,start:start+11*agent.n_agents].reshape(*raw_obs.shape[:-1],agent.n_agents,11)
    wt = weights[...,start:start+11*agent.n_agents].reshape_as(targets)
    wt[...,:1] = 2
    wt[...,1:6] = 2*targets[...,:1]
    wt[...,9:11] = 2*targets[...,:1]
    e = F.smooth_l1_loss(prediction,target.detach(),reduction='none')
    return (e*weights).sum(-1)/weights.sum(-1).clamp_min(1.)


def ahs_forward(self, history, mask, obs, action, prev_obs, has_prev):
    from algorithms.dib_mappo.auxiliary import history_context
    return history_context(self,history,mask,obs,action,prev_obs,has_prev,
                           self.query_proj(history[:,-1]),self.key_proj(history),self.value_proj(history))


def other_targets(self, obs):
    compact = self.entity_layout.pack(obs)
    return torch.stack([torch.cat([compact[:,:,j] for j in range(self.n_agents) if j!=i],-1)
                        for i in range(self.n_agents)],2)


def student_forward(self, context):
    mu,std = self.student(context).chunk(2,-1)
    return self._finite(mu,clip=self.latent_clip),std.clamp(-2.,1.)


def predict_next(self, obs, action, z):
    leading = z.shape[:-1]
    value = self.forward_dynamics_model(torch.cat((obs.reshape(-1,self.obs_dim),
                        action.reshape(-1,self.n_actions),z.reshape(-1,self.z_dim)),-1))
    return self._finite(value).reshape(*leading,self.entity_layout.width)


def completed_filter(self, pi, z, prev_obs, obs, action, has_prev):
    if not self.use_online_filter:
        return pi,torch.zeros_like(pi)
    prediction = self.predict_next_obs(prev_obs.unsqueeze(1).expand(-1,self.n_hypotheses,-1),
                    action.unsqueeze(1).expand(-1,self.n_hypotheses,-1),z)
    target = self.entity_layout.pack(obs).unsqueeze(1)
    error = self.entity_layout.error(prediction,target,filtering=True)
    posterior = torch.softmax(torch.log(pi.clamp_min(1e-8))-error/self.tau_d,-1)
    return torch.where(has_prev>0,posterior,pi),error


def compose_logits(agent, base_logits, residual, base_hidden, context):
    # The trust head learns only from task PPO, not privileged labels. No
    # entropy-based forced confidence: ambiguous beliefs may still be useful.
    trust = torch.sigmoid(agent.belief_trust(torch.cat((base_hidden.detach(),context.detach()),-1)))
    bound = agent.args.dib_residual_logit_bound
    bounded = bound*torch.tanh(residual/bound)
    logits = base_logits + agent.residual_actor_scale*agent.residual_actor_gate*trust.unsqueeze(-1)*bounded
    return agent._finite(logits,clip=agent.args.dib_logit_clip), bounded, trust


def forward(self, obs, action, hidden):
    output, state = self._robust_original_forward(obs,action,hidden)
    aux = output['aux']
    logits,residual,trust = compose_logits(self,aux['base_logits'],aux['residual_logits'],
                                            state['base_policy_h'],aux['c'])
    output['logits_k'] = aux['logits_k'] = logits
    aux['residual_logits'],aux['belief_trust'] = residual,trust
    return output,state


def ahs_losses(self, aux, y, valid):
    probe = self.entity_layout.error(aux['probe_mu'],y,aux['probe_logstd'])
    probe = self._masked_mean(probe,valid)
    transition = local_normalized_error(self,aux['transition_pred'],aux['transition_target'],aux['_local_obs'])
    transition = self._masked_mean(transition,valid*aux['transition_has_prev'].squeeze(-1))
    length = self._masked_mean(aux['soft_mask'].mean(-1),valid)
    sharp = self._masked_mean((aux['soft_mask']*(1-aux['soft_mask'])).mean(-1),valid)
    total = probe+transition+self.args.dib_ahs_len_coef*length+self.args.dib_ahs_sharpness_coef*sharp
    return {'ahs_loss':total,'ahs_probe_loss':probe,'ahs_transition_loss':transition,
            'ahs_length_loss':length,'ahs_sharpness_loss':sharp}


def teacher_losses(self, aux, obs, next_obs, actions, prev_actions, y, valid):
    from algorithms.dib_mappo.auxiliary import teacher_centre
    b,t,n,_ = aux['x'].shape
    stats = self._finite(self.teacher_head(teacher_centre(self,aux['x'])))
    mu,std = stats.chunk(2,-1)
    mu = self._finite(mu,clip=self.latent_clip).reshape(b,n,t,self.z_dim).transpose(1,2)
    std = std.clamp(-2.,1.).reshape(b,n,t,self.z_dim).transpose(1,2)
    decoded = self.belief_decoder(mu)
    recon = self._masked_mean(self.entity_layout.error(decoded,y),valid)
    prior = self._masked_mean(self._normal_kl(mu,std,torch.zeros_like(mu),torch.zeros_like(std)).mean(-1),valid)
    teacher_values = self.entity_layout.transform(decoded).detach()
    student_values = self.entity_layout.transform(aux['y_hat_s'])
    support = self.entity_layout.valid_features(y)
    consistency = ((student_values-teacher_values).square()*support).sum(-1)/support.sum(-1).clamp_min(1.)
    consistency = self._masked_mean(consistency,valid)
    distill = self._masked_mean(self._normal_kl(mu.detach(),std.detach(),aux['student_mu'],
                                             aux['student_logstd'].clamp(-2.,1.)).mean(-1),valid)
    normalized = self.obs_norm(obs)
    pf = self._finite(self.teacher_forward_dynamics(torch.cat((mu,normalized,actions),-1)))
    pb = self._finite(self.teacher_backward_dynamics(torch.cat((mu,normalized,prev_actions),-1)))
    previous = torch.cat((obs[:,:1],obs[:,:-1]),1)
    pmask = torch.ones_like(valid);pmask[:,0]=0
    dynamics = self._masked_mean(local_normalized_error(self,pf,self.obs_norm(next_obs),next_obs),valid)
    dynamics = dynamics+self._masked_mean(local_normalized_error(self,pb,self.obs_norm(previous),previous),valid*pmask)
    teacher = recon+self.args.dib_teacher_dyn_coef*dynamics+self.args.dib_teacher_prior_beta*prior
    distillation = self.args.dib_distill_kl_coef*distill+self.args.dib_distill_output_coef*consistency
    return {'teacher_loss':teacher,'teacher_recon_loss':recon,'teacher_dyn_loss':dynamics,
            'teacher_prior_kl_loss':prior,'distill_loss':distillation,'distill_kl_loss':distill,
            'distill_output_consistency_loss':consistency}


def mhf_losses(self, aux, obs, next_obs, actions, y, valid):
    prior = aux['pi'].clamp_min(1e-8)
    weights = aux['weights'].clamp_min(1e-8)
    error = aux.get('_structured_hyp_error')
    if error is None:error = self.entity_layout.error(aux['y_hat_k'],y.unsqueeze(-2))
    mix = self._masked_mean(-torch.logsumexp(prior.log()-error/self.args.dib_target_sigma,-1),valid)
    features = self.entity_layout.transform(aux['y_hat_k'])
    mask = self.entity_layout.valid_features(y).unsqueeze(-2)
    features = features*mask.sqrt()/mask.sum(-1,keepdim=True).clamp_min(1.).sqrt()
    if self.n_hypotheses > 1:
        from algorithms.dib_mappo.auxiliary import SelfDistances
        distances = SelfDistances.apply(features.reshape(-1,self.n_hypotheses,self.y_dim))
        eye = torch.eye(self.n_hypotheses,device=obs.device,dtype=torch.bool).unsqueeze(0)
        separation = F.relu(self.args.dib_sep_margin-distances).square().masked_fill(eye,0.)
        separation = separation.sum((1,2))/(self.n_hypotheses*(self.n_hypotheses-1))
        separation = self._masked_mean(separation.reshape_as(valid),valid)
    else:
        separation = features.sum()*0.
    prediction = self.predict_next_obs(obs.unsqueeze(-2).expand(-1,-1,-1,self.n_hypotheses,-1),
                        actions.unsqueeze(-2).expand(-1,-1,-1,self.n_hypotheses,-1),aux['z'])
    next_error = self.entity_layout.error(prediction,self.entity_layout.pack(next_obs).unsqueeze(-2),filtering=True)
    fwd = self._masked_mean(-torch.logsumexp(weights.log()-next_error/self.args.dib_obs_sigma,-1),valid)
    hypothesis = mix+self.args.dib_lambda_sep*separation
    entropy = -(weights*weights.log()).sum(-1)
    trust = aux.get('belief_trust',torch.zeros_like(valid).unsqueeze(-1)).squeeze(-1)
    result = {'hyp_loss':hypothesis,'hyp_mixture_loss':mix,'hyp_separation_loss':separation,
              'mhf_loss':hypothesis+fwd,'fwd_loss':fwd,'mhf_forward_loss':fwd,
              'mhf_entropy_loss':-self._masked_mean(entropy,valid),
              'mhf_posterior_entropy':self._masked_mean(entropy,valid),
              'mhf_weight_update_l1':self._masked_mean((weights-prior).abs().sum(-1),valid),
              'belief_trust_mean':self._masked_mean(trust,valid),
              'ahs_rho_mean':self._masked_mean(aux['rho'].squeeze(-1),valid),
              'structured_other_alive_fraction':self._masked_mean(y.reshape(*y.shape[:-1],self.n_agents-1,-1)[...,0].mean(-1),valid)}
    if 'action_prob_k' in aux:
        pk = aux['action_prob_k']
        diversity = .5*(pk-pk[...,:1,:]).abs().sum(-1).mean(-1)
        result['hyp_policy_tv'] = self._masked_mean(diversity,valid)
    return result


def auxiliary_losses(self, aux, obs, actions, mask):
    aux['_local_obs'] = obs[:,:aux['x'].shape[1]].float()
    return self._robust_original_auxiliary(aux,obs,actions,mask)


def clip_actor_gradients(learner):
    """Float64 norm prevents FP32 sum-of-squares overflow. Separate groups
    prevent a large auxiliary gradient shrinking the isolated base policy."""
    groups = {'base':[], 'belief':[]}
    for name,p in learner.mac.agent.named_parameters():
        if p.grad is None:continue
        if not bool(torch.isfinite(p.grad).all()):
            raise FloatingPointError('Non-finite robust DIB gradient: '+name)
        base=name.startswith(('base_policy_','base_actor_head'))
        if getattr(learner.args,'dib_tactical',False):
            base=base or name.startswith(('entity_','ally_encoder','enemy_encoder','attack_head','geometric_move_head'))
        groups['base' if base else 'belief'].append(p)
    norms = {}
    for name,parameters in groups.items():
        norm = torch.stack([p.grad.norm(p=2,dtype=torch.float64) for p in parameters]).norm() if parameters else torch.tensor(0.,dtype=torch.float64)
        coefficient = min(1.,learner.args.grad_norm_clip/(float(norm)+1e-12))
        for p in parameters:p.grad.mul_(coefficient)
        norms[name] = norm
        learner.logger.log_stat(name+'_actor_grad_norm',float(norm),learner.mac._learning_clock)
    return torch.stack(list(norms.values())).norm()


def install(agent):
    """Install before learner construction so optimizer/old-MAC include heads."""
    agent.entity_layout = EntityLayout(agent.n_agents,agent.obs_dim)
    agent.y_dim = (agent.n_agents-1)*agent.entity_layout.width
    if agent.n_agents < 2:raise ValueError('robust_comm_v1 requires at least two agents')
    agent.probe_head = nn.Linear(agent.ahs_hidden_dim,2*agent.y_dim)
    agent.belief_decoder[-1] = nn.Linear(agent.hidden_dim,agent.y_dim)
    agent.forward_dynamics_model[-1] = nn.Linear(agent.hidden_dim,agent.entity_layout.width)
    agent.belief_trust = nn.Linear(agent.hidden_dim+agent.ahs_hidden_dim,1)
    nn.init.zeros_(agent.belief_trust.weight);nn.init.constant_(agent.belief_trust.bias,-2.)
    agent._robust_original_forward = agent.forward
    agent._robust_original_auxiliary = agent.compute_auxiliary_losses
    for name,function in {'forward':forward,'_ahs_forward':ahs_forward,'_build_other_agent_targets':other_targets,
          'predict_next_obs':predict_next,'_completed_transition_filter':completed_filter,'_student_forward':student_forward,
          '_compute_ahs_losses':ahs_losses,'_compute_teacher_losses':teacher_losses,
          '_compute_mhf_losses':mhf_losses,'compute_auxiliary_losses':auxiliary_losses}.items():
        setattr(agent,name,MethodType(function,agent))
