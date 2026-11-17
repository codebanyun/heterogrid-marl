"""DIB-Tactical-v2: causal entity actor, streaming AHS, trajectory BBD, anchored MHF.

This is a method variant, not an upstream-equivalent optimization. Privileged
observations/actions and future context appear only in auxiliary supervision.
No expert policy, hidden target access, new action or reward shaping is used.
"""
import copy
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.dib_mappo import robust

PRESETS = {'policy_first': .03, 'balanced': .10, 'belief_first': .30}


def encode_local(self, obs, action, ids):
    start = self.entity_layout.target_start
    own = torch.cat((obs[..., :12], action, ids), -1)
    query = self.entity_query(own)
    allies = obs[..., 12:start].reshape(*obs.shape[:-1], self.n_agents-1, 8)
    enemies = obs[..., start:start+11*self.n_agents].reshape(*obs.shape[:-1], self.n_agents, 11)
    def pool(tokens, encoder):
        features = encoder(tokens)
        score = (features*query.unsqueeze(-2)).sum(-1)/math.sqrt(features.shape[-1])
        visible = tokens[..., 0] > 0
        weights = torch.softmax(score.masked_fill(~visible, -1e9), -1)*visible
        weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return (features*weights.unsqueeze(-1)).sum(-2)
    return torch.tanh(self.entity_fusion(torch.cat((own,pool(allies,self.ally_encoder),
                                                    pool(enemies,self.enemy_encoder)),-1)))


def action_logits(self, hidden, obs):
    start = self.entity_layout.target_start
    targets = obs[..., start:start+11*self.n_agents].reshape(*obs.shape[:-1], self.n_agents, 11)
    features = self.enemy_encoder(targets)
    linear=self.attack_head[0];h=self.hidden_dim
    # Same concatenated linear layer, projecting shared context only once.
    context=F.linear(hidden,linear.weight[:,:h],linear.bias)+F.linear(obs[...,:12],linear.weight[:,h+32:])
    target=F.linear(features,linear.weight[:,h:h+32])
    attacks=self.attack_head[-1](torch.tanh(context.unsqueeze(-2)+target)).squeeze(-1)
    return torch.cat((self.base_actor_head(hidden),attacks),-1)


def streaming_context(self, history, history_mask, obs, action, prev_obs, has_prev, query=None, keys=None, values=None):
    # History contains causal GRU states, not raw tokens re-encoded L times.
    minimum = history_mask[:, -self.l_min:].unsqueeze(-1)
    pooled = (history[:, -self.l_min:]*minimum).sum(1)/minimum.sum(1).clamp_min(1.)
    mu, std = self.probe_head(pooled).chunk(2,-1)
    mu = self._finite(mu);std=std.clamp(-2.,1.)
    entropy = robust.probe_entropy(self,mu,std)
    predicted = self._finite(self.transition_predictor(torch.cat((self.obs_norm(prev_obs),action),-1)))
    target = self.obs_norm(obs)
    error = robust.local_normalized_error(self,predicted,target,obs).unsqueeze(-1)*has_prev
    ambiguity = self.ambiguity_norm(torch.cat((entropy.detach(),error.detach()),-1))
    rho = self.l_min+(self.l_max-self.l_min)*torch.sigmoid(self.boundary_mlp(torch.cat((history[:,-1],ambiguity),-1)))
    distances = self.history_distances.view(1,self.l_max)
    soft = torch.sigmoid((rho-distances-.5)/self.kappa)*history_mask
    query = self.query_proj(history[:,-1]) if query is None else query
    keys = self.key_proj(history) if keys is None else keys
    values = self.value_proj(history) if values is None else values
    score = (query.unsqueeze(1)*keys).sum(-1)/math.sqrt(self.ahs_hidden_dim)
    score = score+self.relative_bias[self.history_distances.long()].view(1,-1)+soft.clamp_min(1e-6).log()
    weights = torch.softmax(score,-1)
    context = self._finite((weights.unsqueeze(-1)*values).sum(1))
    return context,{'probe_mu':mu,'probe_logstd':std,'transition_pred':predicted,
                    'transition_target':target,'rho':rho,'soft_mask':soft}


def hypotheses(self, context, mu, std, decode=True):
    belief = torch.cat((context,mu,2.*std),-1).unsqueeze(1).expand(-1,self.n_hypotheses,-1)
    query = self.hypothesis_queries.unsqueeze(0).expand(mu.shape[0],-1,-1)
    trunk = self.hypothesis_trunk(torch.cat((belief,query),-1))
    delta = .25*torch.tanh(self.hypothesis_delta(trunk))
    offsets = self.hypothesis_anchors.unsqueeze(0)+delta
    z = self._finite(mu.unsqueeze(1)+std.exp().unsqueeze(1)*offsets,clip=self.latent_clip)
    prior = torch.softmax(self.hypothesis_gate(trunk).squeeze(-1),-1)
    if decode:return z,prior,self._finite(self.belief_decoder(z))
    return z,prior


def student(self, context):
    mu,std=self.student(context).chunk(2,-1)
    return self._finite(mu,clip=self.latent_clip),std.clamp(math.log(.25),1.)


def forward(self, obs, last_action, hidden):
    b,n,_=obs.shape;flat=b*n
    o=self._finite(obs.reshape(flat,self.obs_dim));a=last_action.reshape(flat,self.n_actions)
    ids=torch.eye(n,device=obs.device,dtype=obs.dtype).unsqueeze(0).expand(b,-1,-1).reshape(flat,n)
    bh=self.base_policy_gru(self.encode_local(o,a,ids),hidden['base_policy_h'].reshape(flat,self.hidden_dim))
    bh=self._finite(bh)
    x=self._finite(self.obs_action_encoder(torch.cat((o,a,ids),-1)))
    h=self._finite(self.policy_gru(x,hidden['policy_h'].reshape(flat,self.hidden_dim)))
    history=torch.cat((hidden['x_history'].reshape(flat,self.l_max,self.embed_dim)[:,1:],h.unsqueeze(1)),1)
    hm=torch.cat((hidden['history_mask'].reshape(flat,self.l_max)[:,1:],o.new_ones(flat,1)),1)
    prev=hidden['prev_obs'].reshape(flat,self.obs_dim);has=hidden['has_prev'].reshape(flat,1)
    c,ahs=self.streaming_context(history,hm,o,a,prev,has)
    mu,std=self._student_forward(c);z,prior=self._mhf_forward(c,mu,std,decode=False)
    weights,error=self._completed_transition_filter(prior,z,prev,o,a,has)
    base=self.action_logits(bh,o).unsqueeze(1)
    residual=self.actor_head(torch.cat((h.unsqueeze(1).expand(-1,self.n_hypotheses,-1),
                      c.unsqueeze(1).expand(-1,self.n_hypotheses,-1),z),-1))
    logits,residual,trust=robust.compose_logits(self,base,residual,bh,c)
    def shape(v):return v.reshape(b,n,*v.shape[1:])
    aux={'x':shape(x),'c':shape(c),'probe_mu':shape(ahs['probe_mu']),
         'probe_logstd':shape(ahs['probe_logstd']),'transition_pred':shape(ahs['transition_pred']),
         'transition_target':shape(ahs['transition_target']),'transition_has_prev':shape(has),
         'rho':shape(ahs['rho']),'soft_mask':shape(ahs['soft_mask']),
         'student_mu':shape(mu),'student_logstd':shape(std),'z':shape(z),'pi':shape(prior),
         'weights':shape(weights),
         'filter_error':shape(error),'base_logits':shape(base),'residual_logits':shape(residual),
         'logits_k':shape(logits),'belief_trust':shape(trust)}
    state={'base_policy_h':shape(bh),'policy_h':shape(h),'x_history':shape(history),'history_mask':shape(hm),
           'filter_weights':shape(weights),'prev_obs':obs.clone(),'has_prev':obs.new_ones(b,n,1)}
    return {'logits_k':aux['logits_k'],'weights':aux['weights'],'aux':aux},state


def trajectory_teacher(agent,tokens,valid):
    b,t,n,e=tokens.shape
    sequence=tokens.transpose(1,2).reshape(b*n,t,e)
    lengths=(valid.sum(-1)>0).sum(-1).clamp_min(1).long().repeat_interleave(n).cpu()
    packed=nn.utils.rnn.pack_padded_sequence(sequence,lengths,batch_first=True,enforce_sorted=False)
    output,_=agent.teacher_gru(packed)
    output,_=nn.utils.rnn.pad_packed_sequence(output,batch_first=True,total_length=t)
    return output.reshape(b,n,t,-1).transpose(1,2)


def teacher_losses(self,aux,obs,next_obs,actions,prev_actions,y,valid):
    if '_teacher_mu' in aux:
        mu,std=aux['_teacher_mu'],aux['_teacher_std']
    else:
        stats=self.teacher_head(trajectory_teacher(self,aux['x'],valid))
        mu,std=stats.chunk(2,-1);mu=self._finite(mu,clip=self.latent_clip);std=std.clamp(math.log(.25),1.)
    decoded=self.belief_decoder(mu)
    recon=self._masked_mean(self.entity_layout.error(decoded,y),valid)
    prior=self._masked_mean(self._normal_kl(mu,std,torch.zeros_like(mu),torch.zeros_like(std)).mean(-1),valid)
    support=self.entity_layout.valid_features(y)
    delta=self.entity_layout.transform(aux['y_hat_s'])-self.entity_layout.transform(decoded).detach()
    consistency=self._masked_mean((delta.square()*support).sum(-1)/support.sum(-1).clamp_min(1.),valid)
    kl=self._masked_mean(self._normal_kl(mu.detach(),std.detach(),aux['student_mu'],aux['student_logstd']).mean(-1),valid)
    # Task-relevant intent supervision: other agents' actual current actions.
    if '_intent_labels' in aux:
        labels,others=aux['_intent_labels'],aux['_intent_valid']
    else:
        labels=torch.stack([torch.stack([actions[:,:,j] for j in range(self.n_agents) if j!=i],2)
                            for i in range(self.n_agents)],2).argmax(-1)
        alive=obs[...,3]>0
        others=torch.stack([torch.stack([alive[:,:,j] for j in range(self.n_agents) if j!=i],2)
                            for i in range(self.n_agents)],2).float()
    def intent(z):
        logits=self.intent_decoder(z).reshape(*z.shape[:-1],self.n_agents-1,self.n_actions)
        loss=F.cross_entropy(logits.reshape(-1,self.n_actions),labels.reshape(-1),reduction='none').reshape_as(others)
        loss=(loss*others).sum(-1)/others.sum(-1).clamp_min(1.)
        return self._masked_mean(loss,valid)
    teacher_intent=intent(mu);student_intent=intent(aux['student_mu'])
    teacher=recon+.001*prior+.25*teacher_intent
    distill=kl+.5*consistency+.25*student_intent
    zero=mu.sum()*0.
    return {'teacher_loss':teacher,'teacher_recon_loss':recon,'teacher_dyn_loss':zero,
            'teacher_prior_kl_loss':prior,'distill_loss':distill,'distill_kl_loss':kl,
            'distill_output_consistency_loss':consistency,'teacher_intent_loss':teacher_intent,
            'student_intent_loss':student_intent}


def mhf_losses(self,aux,obs,next_obs,actions,y,valid):
    aux['_structured_hyp_error']=self.entity_layout.error(aux['y_hat_k'],y.unsqueeze(-2))
    result=robust.mhf_losses(self,aux,obs,next_obs,actions,y,valid)
    error=aux['_structured_hyp_error']
    winner=error.detach().argmin(-1)
    best=self._masked_mean(error.min(-1)[0],valid)
    gate=self._masked_mean(-aux['pi'].clamp_min(1e-8).log().gather(-1,winner.unsqueeze(-1)).squeeze(-1),valid)
    mean_prior=(aux['pi']*valid.unsqueeze(-1)).sum((0,1,2))/valid.sum().clamp_min(1.)
    balance=(mean_prior*(mean_prior.clamp_min(1e-8)*self.n_hypotheses).log()).sum()
    original=result['hyp_loss']
    result['hyp_loss']=best+.05*gate+.10*original+.01*balance
    result['mhf_loss']=result['hyp_loss']+result['fwd_loss']
    result['mhf_best_error']=best;result['mhf_gate_assignment_loss']=gate
    result['mhf_batch_balance']=balance
    result['hyp_latent_spread']=self._masked_mean(aux['z'].std(-2,unbiased=False).mean(-1),valid)
    return result


def auxiliary_losses(self,aux,obs,actions,mask):
    """Decode/score only active, filled actor samples. Teacher still sees the
    entire real sequence, so this removes zero-weight computation, not context."""
    t=mask.shape[1];current=obs[:,:t].float();following=obs[:,1:t+1].float()
    select=mask>0
    def take(value):return value[select].unsqueeze(0).unsqueeze(0)
    y=take(self._build_other_agent_targets(current))
    small={k:take(v) for k,v in aux.items() if torch.is_tensor(v)}
    valid=mask[select].reshape(1,1,-1)
    raw=self.teacher_head(trajectory_teacher(self,aux['x'],mask))
    mu,std=raw.chunk(2,-1)
    small['_teacher_mu']=take(self._finite(mu,clip=self.latent_clip))
    small['_teacher_std']=take(std.clamp(math.log(.25),1.))
    labels=torch.stack([torch.stack([actions[:,:t,j] for j in range(self.n_agents) if j!=i],2)
                        for i in range(self.n_agents)],2).argmax(-1)
    alive=current[...,3]>0
    others=torch.stack([torch.stack([alive[:,:,j] for j in range(self.n_agents) if j!=i],2)
                        for i in range(self.n_agents)],2).float()
    small['_intent_labels']=take(labels);small['_intent_valid']=take(others)
    small['_local_obs']=take(current)
    small['y_hat_s']=self.belief_decoder(small['student_mu'])
    small['y_hat_k']=self.belief_decoder(small['z'])
    local=take(current);next_local=take(following);chosen=take(actions[:,:t])
    previous=torch.cat((torch.zeros_like(actions[:,:1]),actions[:,:t-1]),1)
    ahs=self._compute_ahs_losses(small,y,valid)
    teacher=self._compute_teacher_losses(small,local,next_local,chosen,take(previous),y,valid)
    mhf=self._compute_mhf_losses(small,local,next_local,chosen,y,valid)
    return {**ahs,**teacher,**mhf,'bbd_core_loss':teacher['teacher_loss']+teacher['distill_loss']}


class TeamValue(nn.Module):
    """One common-reward team value, expanded only for API compatibility."""
    def __init__(self,args):
        super().__init__();self.n=args.n_agents
        self.net=nn.Sequential(nn.Linear(args.state_shape+args.n_agents*args.n_actions,args.hidden_dim),
                               nn.Tanh(),nn.Linear(args.hidden_dim,args.hidden_dim),nn.Tanh(),nn.Linear(args.hidden_dim,1))
    def forward(self,batch):
        action=batch['actions_onehot'];previous=torch.cat((torch.zeros_like(action[:,:1]),action[:,:-1]),1)
        value=self.net(torch.cat((batch['state'],previous.flatten(-2)),-1))
        return value.unsqueeze(2).expand(-1,-1,self.n,-1)


def gae(reward,values,terminated,mask,gamma=.99,lam=.95):
    running=torch.zeros_like(values[:,0]);advantages=torch.zeros_like(values[:,:-1])
    for t in range(reward.shape[1]-1,-1,-1):
        alive=1-terminated[:,t]
        delta=reward[:,t]+gamma*values[:,t+1]*alive-values[:,t]
        running=(delta+gamma*lam*alive*running)*mask[:,t]
        advantages[:,t]=running
    return advantages,advantages+values[:,:-1]


def train(self,batch,clock,episodes):
    mask=batch['filled'][:,:-1].float();terminal=batch['terminated'][:,:-1].float()
    mask[:,1:]*=1-terminal[:,:-1]
    team_mask=mask.expand(-1,-1,self.n_agents)
    actor_mask=team_mask*(batch['obs'][:,:-1,:,3]>0)
    original_alive=actor_mask
    if getattr(self.args,'dib_resilient',False):
        from algorithms.dib_mappo.resilient import actor_support,episode_weights
        actor_mask,original_alive=actor_support(batch['obs'][:,:-1],mask)
        policy_weight=episode_weights(actor_mask)
    else:policy_weight=actor_mask
    if actor_mask.sum()==0:return
    gate=self.mac.set_residual_actor_gate_for_t(clock);self.old_mac.set_residual_actor_gate(gate)
    def replay(mac,auxiliary=False):
        mac.init_hidden(batch.batch_size)
        out=[mac.forward(batch,t,return_aux=auxiliary) for t in range(batch.max_seq_length-1)]
        return (torch.stack([x[0] for x in out],1),[x[1] for x in out]) if auxiliary else torch.stack(out,1)
    actions=batch['actions'][:,:-1]
    with torch.no_grad():
        old=replay(self.old_mac)
        old_log=old.gather(-1,actions).squeeze(-1).clamp_min(1e-10).log()
        advantage,returns=gae(batch['reward'][:,:-1].expand_as(team_mask),self.critic(batch).squeeze(-1),
                               terminal,mask,self.args.gamma,self.args.gae_lambda)
        if getattr(self.args,'dib_resilient',False):
            centre=(advantage*policy_weight).sum()/policy_weight.sum()
            variance=((advantage-centre).square()*policy_weight).sum()/policy_weight.sum()
            advantages=(advantage-centre)/variance.sqrt().clamp_min(1e-6)
        else:
            valid=advantage[actor_mask>0]
            advantages=(advantage-valid.mean())/valid.std(unbiased=False).clamp_min(1e-6)
        old_values=self.critic(batch)[:,:-1,:,0].detach()
    logs={};aux_losses={};epochs=0
    for epoch in range(self.args.epochs):
        pi,aux=replay(self.mac,True)
        log=pi.gather(-1,actions).squeeze(-1).clamp_min(1e-10).log()
        difference=(log-old_log).clamp(-20,20);ratios=difference.exp()
        mean=lambda x:(x*policy_weight).sum()/policy_weight.sum()
        kl=mean((ratios-1)-difference)
        if epoch>0 and self.args.ppo_target_kl>0 and float(kl)>self.args.ppo_target_kl:break
        values=self.critic(batch)[:,:-1,:,0]
        clipped=old_values+(values-old_values).clamp(-.2,.2)
        error=torch.maximum((values-returns).square(),(clipped-returns).square())
        value_loss=.5*(error*team_mask).sum()/team_mask.sum()
        self.critic_optimiser.zero_grad();value_loss.backward()
        cn=torch.nn.utils.clip_grad_norm_(self.critic_params,self.args.grad_norm_clip,error_if_nonfinite=True)
        self.critic_optimiser.step()
        entropy=-(pi*pi.clamp_min(1e-10).log()).sum(-1)
        objective=torch.minimum(ratios*advantages,ratios.clamp(1-self.args.eps_clip,1+self.args.eps_clip)*advantages)
        pg=-mean(objective+self._entropy_coef(clock)*entropy)
        belief=pg.new_tensor(0.)
        if self._get_stage(clock)!='warmup':
            aux_losses=self.mac.compute_auxiliary_losses(aux,batch,actor_mask)
            raw=.2*aux_losses['ahs_loss']+.2*aux_losses['bbd_core_loss']+.2*aux_losses['hyp_loss']+.1*aux_losses['fwd_loss']
            belief=self.args.tactical_aux_weight*self._belief_scale(clock)*raw
        self.agent_optimiser.zero_grad();(pg+belief).backward()
        norm=robust.clip_actor_gradients(self);self.agent_optimiser.step();epochs+=1
        logs={'pg_loss':float(pg),'belief_loss':float(belief),'critic_loss':float(value_loss),
              'critic_grad_norm':float(cn),'agent_grad_norm':float(norm),'ppo_approx_kl':float(kl),
              'ppo_ratio_mean':float(mean(ratios)),'ppo_clip_fraction':float(mean(((ratios-1).abs()>.2).float())),
              'entropy':float(mean(entropy)),'ppo_epochs':epochs,'agent_update_skipped':0.,
              'active_actor_fraction':float(actor_mask.sum()/team_mask.sum()),
              'dead_actor_samples':float(team_mask.sum()-original_alive.sum()),
              'dib_actor_residual_gate':gate,'belief_loss_scale':self._belief_scale(clock)}
        if getattr(self.args,'dib_resilient',False):
            logs.update(incapable_actor_samples=float(original_alive.sum()-actor_mask.sum()),
                viable_actor_fraction=float(actor_mask.sum()/original_alive.sum().clamp_min(1.)),
                episode_policy_weight_min=float(policy_weight.sum((1,2)).min()),
                episode_policy_weight_max=float(policy_weight.sum((1,2)).max()))
    self.old_mac.load_state(self.mac)
    self.target_critic.load_state_dict(self.critic.state_dict());self.critic_training_steps+=1
    for key,value in {**logs,**{k:float(v.detach()) for k,v in aux_losses.items()}}.items():
        self.logger.log_stat(key,value,clock)


def install_agent(agent):
    robust.install(agent)
    n,h=agent.n_agents,agent.hidden_dim
    if agent.embed_dim!=h or agent.ahs_hidden_dim!=h:raise ValueError('Streaming state widths must agree')
    agent.entity_query=nn.Linear(12+agent.n_actions+n,32)
    agent.ally_encoder=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
    agent.enemy_encoder=nn.Sequential(nn.Linear(11,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
    agent.entity_fusion=nn.Linear(12+agent.n_actions+n+64,h)
    agent.base_actor_head=nn.Linear(h,7)
    agent.attack_head=nn.Sequential(nn.Linear(h+32+12,h),nn.Tanh(),nn.Linear(h,1))
    # An initial preference, not a forced action: PPO can learn to override it.
    nn.init.constant_(agent.attack_head[-1].bias,1.)
    anchors=torch.empty(agent.z_dim,agent.n_hypotheses);nn.init.orthogonal_(anchors)
    agent.hypothesis_anchors=nn.Parameter(anchors.t()*(math.sqrt(agent.z_dim)*.5))
    nn.init.orthogonal_(agent.hypothesis_queries)
    agent.intent_decoder=nn.Sequential(nn.Linear(agent.z_dim,h),nn.Tanh(),nn.Linear(h,(n-1)*agent.n_actions))
    # Replaced by streaming causal policy-GRU states; no unused window GRU.
    del agent.ahs_gru
    for name,fn in {'encode_local':encode_local,'action_logits':action_logits,'streaming_context':streaming_context,
                    'forward':forward,'_mhf_forward':hypotheses,'_student_forward':student,
                    '_compute_teacher_losses':teacher_losses,'_compute_mhf_losses':mhf_losses,
                    'compute_auxiliary_losses':auxiliary_losses}.items():
        setattr(agent,name,MethodType(fn,agent))


def install_learner(learner):
    learner.critic=TeamValue(learner.args);learner.target_critic=copy.deepcopy(learner.critic)
    learner.critic_params=list(learner.critic.parameters())
    learner.critic_optimiser=torch.optim.Adam(learner.critic_params,lr=learner.args.lr)
    learner.target_critic.requires_grad_(False)
    learner.train=MethodType(train,learner)
