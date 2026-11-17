"""Native MA2E modules with compact history replay and local-only execution."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import yaml

from algorithms.common.actor_critic.backend import MetricLogger, external_module
from algorithms.common.evaluation import observe
from heterogrid.core import REWARD_COMPONENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORRECTIONS = {
    "own_history_slot": "Scatter, execution mask, overwrite and gather all use the actual agent slot.",
    "causal_actions": "At t=0 past actions are zero, including replay training; no current/future action input.",
    "padding": "MAE training uses filled transition rows, includes terminal actions, excludes bootstrap states and padding.",
    "mae_action_offset": "Reconstruction and execution action windows both end at a_(t-1).",
    "frozen_mae_mode": "Frozen MAEs stay in eval mode during Q-learning; clear reconstruction gradients before Adam.",
    "storage": "Store raw transitions once; reconstruct the native five-step history on demand.",
    "templates": "Broadcast zero templates instead of fixed 2000/3000-row allocations; identical model layers.",
    "activation_storage": "Exact recomputation checkpoints per recurrent timestep reduce full-episode gradient memory.",
    "mae_cache": "Cache frozen decoder latents in float32 per replay episode; clear after every MAE update and overwrite.",
}


def resolve_root(root):
    root = Path(root).expanduser().resolve()
    if (root / "src").is_dir():
        root = root / "src"
    for name in ("learners/nq_learner.py", "modules/agents/n_rnn_agent.py",
                 "components/masked_transformer.py", "config/algs/qmix.yaml"):
        if not (root / name).is_file():
            raise ValueError("Incomplete MA2E source: " + name)
    return root


def settings(root, device="cpu"):
    result = yaml.safe_load((root / "config/default.yaml").read_text())
    result.update(yaml.safe_load((root / "config/algs/qmix.yaml").read_text()))
    result.update(use_MT_mode=True, device=device, use_cuda=device == "cuda", env="heterogrid",
                  use_wandb=False, use_tensorboard=False, buffer_cpu_only=True,
                  learner_log_interval=1, test_noise=0.0)
    return result


def history(batch, args, time=None, previous_actions=False):
    """Reconstruct native [batch,time,agent,history,feature] without future data."""
    obs, actions = batch["obs"], batch["actions"]
    b, length, n, _ = obs.shape
    times = torch.arange(length, device=obs.device) if time is None else torch.tensor([time], device=obs.device)
    positions = times[:, None] + torch.arange(1 - args.MT_traj_length, 1, device=obs.device)
    valid = positions >= 0
    selected = positions.clamp(min=0, max=length - 1)
    observations = obs[:, selected].permute(0, 1, 3, 2, 4) * valid[None, :, None, :, None]
    previous = positions - 1
    last = actions[:, previous.clamp(min=0, max=length - 1), :, 0]
    last = torch.nn.functional.one_hot(last.long(), args.n_actions).to(obs.dtype)
    last = last.permute(0, 1, 3, 2, 4) * (previous >= 0)[None, :, None, :, None]
    ids = torch.eye(n, device=obs.device).view(1, 1, n, 1, n).expand(b, len(times), n, args.MT_traj_length, n)
    features = torch.cat((observations, last, ids), dim=-1)
    action_positions = positions - int(previous_actions)
    action_values = actions[:, action_positions.clamp(min=0, max=length - 1)].permute(0, 1, 3, 2, 4)
    action_values = action_values * (action_positions >= 0)[None, :, None, :, None]
    return (features[:, 0], action_values[:, 0]) if time is not None else (features, action_values)


class HistoryBatch:
    def __init__(self, batch, args, mae_training=False):
        self.batch, self.args, self.mae_training = batch, args, mae_training
        self._histories = None

    def __getattr__(self, name):
        return getattr(self.batch, name)

    def __getitem__(self, key):
        if isinstance(key, str) and key in ("obs_all", "action_all"):
            if self._histories is None:
                self._histories = history(self.batch, self.args, previous_actions=True)
            return self._histories[0 if key == "obs_all" else 1]
        if isinstance(key, str) and key == "terminated" and self.mae_training:
            valid = self.batch["filled"].clone()
            valid[:, 1:] *= 1 - self.batch["terminated"][:, :-1]
            return 1 - valid
        return self.batch[key]


def build(root, parameters, env_info):
    sys.path.insert(0, str(root))
    # The native learner only needs matrix-game diagnostics, not StarCraft's registry.
    if "envs" not in sys.modules:
        package = types.ModuleType("envs")
        package.__path__ = [str(root / "envs")]
        sys.modules["envs"] = package
    episode = importlib.import_module("components.episode_buffer")
    onehot = importlib.import_module("components.transforms").OneHot
    transformer = importlib.import_module("components.masked_transformer")
    agents = importlib.import_module("modules.agents")
    original_agent = agents.REGISTRY["n_rnn"]
    basic = external_module(root, "controllers/basic_controller.py").BasicMAC
    selector = importlib.import_module("components.action_selectors").REGISTRY
    learner_class = external_module(root, "learners/nq_learner.py").NQLearner
    args = SimpleNamespace(**{**parameters, **env_info})
    args.input_shape = args.obs_shape + args.n_actions + args.n_agents
    if not args.obs_last_action or not args.obs_agent_id:
        raise ValueError("MA2E history requires the native previous-action and agent-ID inputs")

    class CompactTransformer(transformer.Base_Transformer):
        def __init__(self, obs_dim, action_dim, config, device, positional_type="agent"):
            torch.nn.Module.__init__(self)
            self.n_embd, self.n_heads = config.MT_n_embd, config.MT_n_heads
            self.n_encoder, self.n_decoder = config.MT_n_enc_layer, config.MT_n_dec_layer
            self.traj_length, self.n_agent = config.MT_traj_length, config.n_agents
            self.max_len, self.device = self.traj_length * 2, device
            self.obs_dim, self.action_dim = obs_dim, action_dim
            self.transformer_model = torch.nn.Transformer(d_model=self.n_embd, nhead=self.n_heads,
                num_encoder_layers=self.n_encoder, num_decoder_layers=self.n_decoder, batch_first=True).to(device)
            self.state_embed = torch.nn.Linear(obs_dim, self.n_embd).to(device)
            self.action_embed = torch.nn.Linear(action_dim, self.n_embd).to(device)
            self.decoder_state_embed = torch.nn.Linear(self.n_embd, obs_dim).to(device)
            self.decoder_action_embed = torch.nn.Linear(self.n_embd, action_dim).to(device)
            self.positional_type = positional_type
            self.positional_encoding = transformer.Positional_Encoding(self.max_len * self.n_agent, self.n_embd, device)
            self.add_new_position()
            count = self.n_agent * self.traj_length
            def template(*shape):
                return torch.zeros((1,) + shape, device=device).expand((100000,) + shape)
            self.input_obs_base = template(count, obs_dim)
            self.input_action_base = template(count, action_dim)
            self.embeded_input_1_base = template(2 * count, self.n_embd)
            self.masked_data_base = template(2 * count, self.n_embd)
            self.mask_base = template(2 * count)
            self.execution_masking = torch.zeros(self.n_agent, 2 * count, device=device)
            for index in range(self.n_agent):
                self.execution_masking[index, index * 2::self.n_agent * 2] = 100
                self.execution_masking[index, index * 2 + 1::self.n_agent * 2] = 100

        def decode(self, tgt, memory):
            out = self.transformer_model.decoder(tgt, memory)
            self.last_latents = out[:, 0::2][:, -self.n_agent:].detach()
            return self.decoder_state_embed(out[:, 0::2]), self.decoder_action_embed(out[:, 1::2])

    class CompactAgent(original_agent):
        def __init__(self, input_shape, config):
            torch.nn.Module.__init__(self)
            self.args, self.input_shape, self.mt_pretraining = config, input_shape, False
            self.fc0 = torch.nn.Linear(input_shape, config.rnn_hidden_dim // 2)
            self.fc1 = torch.nn.Linear(input_shape, config.rnn_hidden_dim // 2)
            self.rnn = torch.nn.GRUCell(config.rnn_hidden_dim, config.rnn_hidden_dim)
            self.fc2 = torch.nn.Linear(config.rnn_hidden_dim, config.n_actions)
            self.mae = CompactTransformer(input_shape, 1, config, config.device, config.positional_encoding_target)
            self.mae.requires_grad_(False)
            attention = importlib.import_module("modules.layer.self_without_atten").Self_Without_Attention
            self.self_obs_attention = attention(input_shape, 1, input_shape)
            self.self_action_attention = attention(1, 1, 1)
            shape = (1, config.n_agents * config.MT_traj_length)
            self.input_mae_obs_base = torch.zeros(*shape, input_shape, device=config.device).expand(100000, -1, -1)
            self.input_mae_action_base = torch.zeros(*shape, 1, device=config.device).expand(100000, -1, -1)
            if config.use_layer_norm:
                self.layer_norm = torch.nn.LayerNorm(config.rnn_hidden_dim)
            if config.use_orthogonal:
                init = importlib.import_module("utils.th_utils").orthogonal_init_
                init(self.fc1)
                init(self.fc2, gain=config.gain)

        def train(self, mode=True):
            super().train(mode)
            self.mae.eval()
            return self

        def generate_mae_input(self, all_info):
            b, n, length, width = all_info[0].shape
            count = b * n
            row = torch.arange(count, device=self.args.device)[:, None]
            slots = torch.arange(length, device=self.args.device)[None] * n + (row % n)
            observations = torch.zeros(count, n * length, width, device=self.args.device)
            actions = torch.zeros(count, n * length, 1, device=self.args.device)
            observations[row, slots] = all_info[0].reshape(count, length, width)
            actions[row, slots] = all_info[1].reshape(count, length, 1) / self.args.n_actions
            return observations, actions

        def forward(self, inputs, all_info, hidden_state, latent=None):
            b, n, width = inputs.shape
            if latent is None:
                observations, actions = self.generate_mae_input(all_info)
                reconstructed, _ = self.mae(1, observations, actions, train=False)
                start = (self.args.MT_traj_length - 1) * n
                inferred = reconstructed[:, start:start + n].clone()
            else:
                inferred = self.mae.decoder_state_embed(latent.reshape(b * n, n, -1))
            row = torch.arange(b * n, device=inputs.device)
            slot = row % n
            inferred[row, slot] = inputs.reshape(-1, width)
            _, memory = self.self_obs_attention(inferred)
            own_memory = memory[row, slot]
            x = torch.cat((torch.relu(self.fc0(own_memory)), torch.relu(self.fc1(inputs.reshape(-1, width)))), dim=-1)
            hidden = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
            values = self.fc2(self.layer_norm(hidden) if self.args.use_layer_norm else hidden)
            return values.view(b, n, -1), hidden.view(b, n, -1)

    class LocalHistoryMAC(basic):
        def __init__(self, scheme, groups, config):
            self.n_agents, self.args = config.n_agents, config
            self.agent = CompactAgent(self._get_input_shape(scheme), config)
            self.agent_output_type = config.agent_output_type
            self.action_selector = selector[config.action_selector](config)
            self.hidden_states = None

        def forward(self, batch, t, test_mode=False):
            if test_mode:
                self.agent.eval()
            inputs = self._build_inputs(batch, t)
            latent = getattr(batch, "mae_latents", None)
            if latent is not None:
                latent = latent[:, t]
                if torch.is_grad_enabled() and self.agent.training:
                    def cached_forward(inputs, latent, hidden):
                        return self.agent(inputs, None, hidden, latent)
                    values, self.hidden_states = activation_checkpoint(cached_forward, inputs, latent,
                        self.hidden_states, use_reentrant=False)
                else:
                    values, self.hidden_states = self.agent(inputs, None, self.hidden_states, latent)
                return values
            histories = history(batch, self.args, t, previous_actions=True)
            if torch.is_grad_enabled() and self.agent.training:
                def forward(inputs, observations, actions, hidden):
                    return self.agent(inputs, (observations, actions), hidden)
                values, self.hidden_states = activation_checkpoint(forward, inputs, *histories,
                    self.hidden_states, use_reentrant=False)
            else:
                values, self.hidden_states = self.agent(inputs, histories, self.hidden_states)
            return values

    scheme = {"state": {"vshape": args.state_shape}, "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    groups, preprocess = {"agents": args.n_agents}, {"actions": ("actions_onehot", [onehot(out_dim=args.n_actions)])}
    class CompactReplay(episode.ReplayBuffer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.latent_cache = {}

        def insert_episode_batch(self, batch):
            for i in range(batch.batch_size):
                self.latent_cache.pop((self.buffer_index + i) % self.buffer_size, None)
            super().insert_episode_batch(batch)

    buffer = CompactReplay(scheme, groups, args.buffer_size, args.episode_limit + 1,
                                  preprocess=preprocess, device="cpu")
    mac = LocalHistoryMAC(buffer.scheme, groups, args)
    logger = MetricLogger()
    learner = learner_class(mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    learner.target_mac.agent.requires_grad_(False)
    learner.target_mixer.requires_grad_(False)
    return SimpleNamespace(args=args, mac=mac, learner=learner, logger=logger, buffer=buffer,
        scheme=scheme, groups=groups, preprocess=preprocess, EpisodeBatch=episode.EpisodeBatch)


def new_batch(backend, count, length):
    return backend.EpisodeBatch(backend.scheme, backend.groups, count, length,
        preprocess=backend.preprocess, device=backend.args.device)


def update_batch(batch, values, **kwargs):
    batch.update({key: torch.as_tensor(np.asarray(value), dtype=batch[key].dtype, device=batch.device)
                  for key, value in values.items()}, **kwargs)


@torch.no_grad()
def collect(backend, envs, steps, rng):
    batch = new_batch(backend, len(envs), envs[0].episode_limit + 1)
    batch["avail_actions"][..., 0] = 1
    seeds = [int(rng.integers(0, 2 ** 31 - 1)) for _ in envs]
    for env, seed in zip(envs, seeds):
        env.reset(seed=seed)
    def transition(indices, time):
        values = [observe(envs[i]) for i in indices]
        update_batch(batch, {"obs": [v[0] for v in values], "state": [v[1][0] for v in values],
                            "avail_actions": [v[2] for v in values]}, bs=indices, ts=time)
    transition(list(range(len(envs))), 0)
    backend.mac.agent.eval()
    backend.mac.init_hidden(len(envs))
    done, totals = [False] * len(envs), [0.] * len(envs)
    components = [dict.fromkeys(REWARD_COMPONENTS, 0.) for _ in envs]
    rows, collected = [], 0
    for time in range(envs[0].episode_limit):
        active = [i for i, value in enumerate(done) if not value]
        actions = backend.mac.select_actions(batch, time, steps, bs=active, test_mode=False).cpu().numpy()
        available = batch["avail_actions"][active, time].cpu().numpy()
        if not np.take_along_axis(available, actions[..., None], axis=-1).all():
            raise RuntimeError("MA2E selected an illegal action")
        update_batch(batch, {"actions": actions[..., None]}, bs=active, ts=time, mark_filled=False)
        rewards, terminal = [], []
        for local, index in enumerate(active):
            reward, done[index], info = envs[index].step(actions[local].tolist())
            if any(name.startswith("red_") for name in info["invalid_actions"]):
                raise RuntimeError("MA2E executed an illegal action")
            collected += 1
            totals[index] += reward
            for name in REWARD_COMPONENTS:
                components[index][name] += info["reward_components"][name]
            rewards.append([reward])
            terminal.append([done[index]])
            if done[index]:
                row = dict(seed=seeds[index], length=time + 1, **{"return": totals[index]}, winner=info["winner"],
                    reason=info["termination_reason"], invalid_red_actions=0, reward_components=components[index], **info["episode_metrics"])
                if "team_episode_metrics" in info:
                    row["team_episode_metrics"] = info["team_episode_metrics"]
                rows.append(row)
        update_batch(batch, {"reward": rewards, "terminated": terminal}, bs=active, ts=time, mark_filled=False)
        transition(active, time + 1)
        if all(done):
            break
    if not all(done):
        raise RuntimeError("MA2E collection exceeded the episode limit")
    return batch, collected, rows


@torch.no_grad()
def prepare_latents(backend, batch, ids):
    args, agent, cache = backend.args, backend.mac.agent, backend.buffer.latent_cache
    agent.mae.eval()
    latent = torch.zeros(batch.batch_size, batch.max_seq_length, args.n_agents, args.n_agents,
                         args.MT_n_embd, device=args.device)
    for index, key in enumerate(ids):
        length = int(batch["filled"][index].sum().item())
        if key not in cache:
            episode = batch[index:index + 1, :length]
            observations, actions = history(episode, args, previous_actions=True)
            values = []
            for start in range(0, length, 32):
                stop = min(length, start + 32)
                info = (observations[:, start:stop].reshape(-1, args.n_agents, args.MT_traj_length, args.input_shape),
                        actions[:, start:stop].reshape(-1, args.n_agents, args.MT_traj_length, 1))
                own_obs, own_actions = agent.generate_mae_input(info)
                agent.mae(1, own_obs, own_actions, train=False)
                values.append(agent.mae.last_latents.reshape(stop - start, args.n_agents,
                    args.n_agents, args.MT_n_embd).cpu().clone())
            cache[key] = torch.cat(values)
        value = cache[key]
        latent[index, :len(value)] = value.to(args.device)
    batch.mae_latents = latent


def sample(backend, count, mae=False):
    buffer = backend.buffer
    assert buffer.can_sample(count)
    ids = np.arange(count) if buffer.episodes_in_buffer == count else np.random.choice(buffer.episodes_in_buffer, count, replace=False)
    result = buffer[ids]
    result = result[:, :int(result.max_t_filled().item())]
    result.to(backend.args.device)
    if not mae:
        prepare_latents(backend, result, ids)
    return result


def train_mae(backend, batch, steps, episodes, repeats):
    result = backend.learner.MT_train(HistoryBatch(batch, backend.args, mae_training=True), steps, episodes,
        n_repeat=repeats, write_log=False)
    if not np.isfinite(result):
        raise FloatingPointError("Non-finite MAE loss")
    for parameter in backend.mac.agent.mae.parameters():
        parameter.grad = None
    backend.buffer.latent_cache.clear()
    return result


class EvaluationPolicy:
    def __init__(self, backend):
        self.backend, self.actor = backend, backend.mac.agent
        self.batch, self.time = None, 0

    @torch.no_grad()
    def act(self, obs, hidden, masks, available, deterministic=False):
        b = self.backend
        if not np.asarray(hidden).any() or self.batch is None:
            self.batch, self.time = new_batch(b, 1, b.args.episode_limit + 1), 0
        update_batch(self.batch, {"obs": np.asarray(obs)[None], "avail_actions": np.asarray(available)[None]}, ts=self.time)
        b.mac.hidden_states = torch.as_tensor(hidden, device=b.args.device).reshape(1, b.args.n_agents, -1)
        values = b.mac.forward(self.batch, self.time, test_mode=True)
        # Sampled evaluation uses the trained epsilon floor; greedy uses epsilon zero.
        actions = b.mac.action_selector.select_action(values, self.batch["avail_actions"][:, self.time],
            b.args.epsilon_anneal_time, test_mode=deterministic)
        update_batch(self.batch, {"actions": actions.cpu().numpy()[..., None]}, ts=self.time, mark_filled=False)
        self.time += 1
        return actions.reshape(-1, 1), b.mac.hidden_states.reshape(b.args.n_agents, 1, -1)


def modules(backend):
    return {"actor": backend.mac.agent, "target_actor": backend.learner.target_mac.agent,
            "mixer": backend.learner.mixer, "target_mixer": backend.learner.target_mixer}


def snapshot(backend):
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules(backend).items()}


def checkpoint(path, backend, config, steps, counters):
    states = {name: module.state_dict() for name, module in modules(backend).items()}
    # MAE targets are exact copies; shared checkpoint storage is lossless, not pruning.
    for source, target in (("actor", "target_actor"), ("mixer", "target_mixer")):
        for key, value in states[target].items():
            if torch.equal(value, states[source][key]):
                states[target][key] = states[source][key]
    value = {"algorithm": "ma2e", "environment_steps": steps, "env_config": config.to_dict(),
        "algorithm_args": vars(backend.args), "modules": states,
        "optimizer": backend.learner.optimiser.state_dict(), "counters": counters,
        "last_target_update_episode": backend.learner.last_target_update_episode, "corrections": CORRECTIONS}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def restore(path, backend):
    value = torch.load(path, map_location=backend.args.device)
    for name, module in modules(backend).items():
        module.load_state_dict(value["modules"][name])
    backend.learner.optimiser.load_state_dict(value["optimizer"])
    backend.buffer.latent_cache.clear()
    backend.learner.last_target_update_episode = value["last_target_update_episode"]
    return value


def network_configuration(backend):
    a = backend.args
    return {"algorithm": "QMIX + MA2E", "actor_uses_global_state": False,
        "actor_uses_other_agents_observations": False, "mae_training_uses_joint_histories": True,
        "input_shape": a.input_shape, "rnn_hidden_dim": a.rnn_hidden_dim, "history_length": a.MT_traj_length,
        "mae_embedding": a.MT_n_embd, "mae_encoder_layers": a.MT_n_enc_layer, "mae_decoder_layers": a.MT_n_dec_layer,
        "corrections": CORRECTIONS, "replay_storage": "native raw EpisodeBatch, histories reconstructed exactly",
        "modules": {name: {"architecture": str(module), "parameters": sum(p.numel() for p in module.parameters())}
                    for name, module in modules(backend).items()}}
