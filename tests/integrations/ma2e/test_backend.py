"""Optional native MA2E contracts, isolated from other PyMARL module registries."""
import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MA2E_ROOT = Path(os.environ.get("MA2E_ROOT", "/root/research/MA2E")).expanduser()
SOURCE_ROOT = MA2E_ROOT / "src" if (MA2E_ROOT / "src").is_dir() else MA2E_ROOT
NATIVE_AVAILABLE = (importlib.util.find_spec("torch") is not None
                    and (SOURCE_ROOT / "learners/nq_learner.py").is_file())


SETUP = """
import copy
import importlib
import random
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from algorithms.common.density_study import conditions
from algorithms.ma2e import backend as api
from heterogrid import EnvConfig, SMACAdapter

torch.set_num_threads(1)
torch.manual_seed(42)
random.seed(42)
root = api.resolve_root(MA2E_ROOT)
parameters = api.settings(root)
parameters.update(MT_n_embd=8, MT_n_enc_layer=1, MT_n_dec_layer=1,
                  MT_traj_length=2, batch_size_run=2, batch_size=2, buffer_size=4)
base = EnvConfig.load(PROJECT_ROOT / 'configs/scenarios/6v6_rule_distance.json')


def build(condition='D1_3v3'):
    values = conditions(base)[condition].to_dict()
    values['max_steps'] = 4
    config = EnvConfig.from_dict(values)
    env = SMACAdapter(config=config)
    try:
        model = api.build(root, parameters, env.get_env_info())
    finally:
        env.close()
    assert model.args.rnn_hidden_dim == 128
    return model, config


b, config = build()


def episode(model=b):
    batch = api.new_batch(model, 2, 5)
    a = model.args
    obs = torch.arange(2 * 5 * a.n_agents * a.obs_shape, dtype=torch.float32)
    obs = (obs.reshape(2, 5, a.n_agents, a.obs_shape) % 97) / 97
    state = torch.linspace(-0.1, 0.1, 2 * 5 * a.state_shape).reshape(2, 5, a.state_shape)
    actions = torch.arange(2 * 5 * a.n_agents).reshape(2, 5, a.n_agents, 1)
    actions = actions % (a.n_actions - 1) + 1
    api.update_batch(batch, {'obs': obs, 'state': state, 'actions': actions,
                            'avail_actions': torch.ones(2, 5, a.n_agents, a.n_actions)},
                     mark_filled=False)
    for i, length in enumerate((2, 4)):
        batch['reward'][i, :length, 0] = 0.25
        batch['terminated'][i, length - 1, 0] = 1
        # Collection marks bootstrap observations filled, but they are not transitions.
        batch['filled'][i, :length + 1, 0] = 1
    return batch


def evaluate(batch, time, model=b):
    model.mac.agent.eval()
    hidden = torch.linspace(-0.1, 0.1, batch.batch_size * model.args.n_agents * 128)
    model.mac.hidden_states = hidden.reshape(batch.batch_size, model.args.n_agents, 128).clone()
    with torch.no_grad():
        values = model.mac.forward(batch, time, test_mode=True)
    return values.clone(), model.mac.hidden_states.clone()


def equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
"""


@unittest.skipUnless(NATIVE_AVAILABLE, "Optional native MA2E tests require torch and MA2E_ROOT")
class NativeMA2EBackendTests(unittest.TestCase):
    def run_native(self, source):
        # Do not import native `components`, `modules`, `utils` or `envs` in pytest's process.
        script = (f"MA2E_ROOT = {str(MA2E_ROOT)!r}\n"
                  f"PROJECT_ROOT = __import__('pathlib').Path({str(PROJECT_ROOT)!r})\n"
                  + textwrap.dedent(SETUP) + "\n" + textwrap.dedent(source))
        result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def test_compact_history_matches_causal_raw_replay_windows(self):
        self.run_native("""
            batch = episode()
            a = b.args
            assert 'obs_all' not in batch.scheme and 'action_all' not in batch.scheme
            assert 'obs_all' not in b.buffer.scheme and 'action_all' not in b.buffer.scheme
            assert b.buffer.buffer_size == 4 and b.buffer.max_seq_length == 5
            expected = torch.zeros(2, 5, a.n_agents, 2, a.input_shape)
            actions = torch.zeros(2, 5, a.n_agents, 2, 1, dtype=torch.long)
            for time in range(5):
                for offset in range(2):
                    past = time - 1 + offset
                    expected[:, time, :, offset, -a.n_agents:] = torch.eye(a.n_agents)
                    if past >= 0:
                        expected[:, time, :, offset, :a.obs_shape] = batch['obs'][:, past]
                    if past > 0:
                        previous = batch['actions'][:, past - 1, :, 0]
                        expected[:, time, :, offset, a.obs_shape:a.obs_shape + a.n_actions] = (
                            torch.nn.functional.one_hot(previous, a.n_actions).float())
                        actions[:, time, :, offset] = batch['actions'][:, past - 1]
            features, previous = api.history(batch, a, previous_actions=True)
            equal(features, expected)
            equal(previous, actions)
            wrapped = api.HistoryBatch(batch, a)
            equal(wrapped['obs_all'], expected)
            equal(wrapped['action_all'], actions)
            for time in range(5):
                local, action = api.history(batch, a, time, previous_actions=True)
                equal(local, expected[:, time])
                equal(action, actions[:, time])
            assert not previous[:, 0].any()
            assert not features[:, 0, :, :, a.obs_shape:a.obs_shape + a.n_actions].any()
            b.buffer.insert_episode_batch(batch)
            replay = api.sample(b, 2)
            assert replay.batch_size == 2 and replay.max_seq_length == 5
        """)

    def test_actor_t0_and_current_or_future_actions_are_independent(self):
        self.run_native("""
            batch = episode()
            for time in range(5):
                baseline = evaluate(batch, time)
                altered = copy.deepcopy(batch)
                replacement = (altered['actions'][:, time:] + 1) % b.args.n_actions
                api.update_batch(altered, {'actions': replacement}, ts=slice(time, 5),
                                 mark_filled=False)
                if time < 4:
                    altered['obs'][:, time + 1:] += 100
                for before, after in zip(api.history(batch, b.args, time, previous_actions=True),
                                         api.history(altered, b.args, time, previous_actions=True)):
                    equal(after, before)
                equal(b.mac._build_inputs(altered, time), b.mac._build_inputs(batch, time))
                changed = evaluate(altered, time)
                equal(changed[0], baseline[0])
                equal(changed[1], baseline[1])
        """)

    def test_filled_mae_rows_include_terminal_but_not_bootstrap_or_padding(self):
        self.run_native("""
            batch = episode()
            original_terminated = batch['terminated'].clone()
            original_filled = batch['filled'].clone()
            wrapped = api.HistoryBatch(batch, b.args, mae_training=True)
            equal(wrapped['terminated'].squeeze(-1), torch.tensor([
                [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=wrapped['terminated'].dtype))
            equal(api.HistoryBatch(batch, b.args)['terminated'], original_terminated)
            expected_obs, expected_actions = api.history(batch, b.args, previous_actions=True)
            seen = []
            def record(module, inputs):
                assert module.training
                seen.append((inputs[1].detach().clone(), inputs[2].detach().clone()))
            handle = b.mac.agent.mae.register_forward_pre_hook(record)
            try:
                loss = api.train_mae(b, batch, 8, 2, repeats=1)
            finally:
                handle.remove()
            assert np.isfinite(loss)
            assert len(seen) == 2 * b.args.n_agents
            for i, length in enumerate((2, 4)):
                for observations, actions in seen[i * b.args.n_agents:(i + 1) * b.args.n_agents]:
                    equal(observations, expected_obs[i, :length])
                    equal(actions, expected_actions[i, :length].float() / b.args.n_actions)
            equal(batch['terminated'], original_terminated)
            equal(batch['filled'], original_filled)
        """)

    def test_local_q_is_invariant_to_other_agents_observations(self):
        self.run_native("""
            batch = episode()
            baseline, baseline_hidden = evaluate(batch, 2)
            for agent in range(b.args.n_agents):
                altered = copy.deepcopy(batch)
                others = [index for index in range(b.args.n_agents) if index != agent]
                altered['obs'][:, :, others] += 50
                altered['state'].add_(100)
                changed, changed_hidden = evaluate(altered, 2)
                equal(changed[:, agent], baseline[:, agent])
                equal(changed_hidden[:, agent], baseline_hidden[:, agent])
            architecture = api.network_configuration(b)
            assert not architecture['actor_uses_global_state']
            assert not architecture['actor_uses_other_agents_observations']
        """)

    def test_each_agents_q_uses_its_own_older_history(self):
        self.run_native("""
            batch = episode()
            baseline, _ = evaluate(batch, 2)
            for agent in range(b.args.n_agents):
                altered = copy.deepcopy(batch)
                altered['obs'][:, 1, agent] += 5
                equal(b.mac._build_inputs(altered, 2), b.mac._build_inputs(batch, 2))
                changed, _ = evaluate(altered, 2)
                delta = (changed[:, agent] - baseline[:, agent]).abs().max().item()
                assert delta > 1e-7, (agent, 'older own history has no Q influence', delta)
                others = [index for index in range(b.args.n_agents) if index != agent]
                equal(changed[:, others], baseline[:, others])
        """)

    def test_twelve_agents_use_actual_scatter_mask_overwrite_and_gather_slots(self):
        self.run_native("""
            model, _ = build('D3_12v12')
            a = model.args
            assert a.n_agents == 12
            batch = episode(model)
            histories = api.history(batch, a, 2, previous_actions=True)
            scattered_obs, scattered_actions = model.mac.agent.generate_mae_input(histories)
            for episode_index in range(2):
                for agent in range(12):
                    row = episode_index * 12 + agent
                    expected_obs = torch.zeros(24, a.input_shape)
                    expected_actions = torch.zeros(24, 1)
                    expected_obs[agent::12] = histories[0][episode_index, agent]
                    expected_actions[agent::12] = histories[1][episode_index, agent].float() / a.n_actions
                    equal(scattered_obs[row], expected_obs)
                    equal(scattered_actions[row], expected_actions)
                    expected_mask = torch.zeros(48)
                    expected_mask[2 * agent::24] = 100
                    expected_mask[2 * agent + 1::24] = 100
                    equal(model.mac.agent.mae.execution_masking[agent], expected_mask)
            tokens = torch.arange(24 * 48 * 8, dtype=torch.float32).reshape(24, 48, 8) + 1
            masked = model.mac.agent.mae.masking(1, False, tokens, 0, 0)
            for row in range(24):
                expected = torch.zeros_like(tokens[row])
                agent = row % 12
                expected[2 * agent::24] = tokens[row, 2 * agent::24]
                expected[2 * agent + 1::24] = tokens[row, 2 * agent + 1::24]
                equal(masked[row], expected)

            # Observe real transformer output before and after the actor's slot overwrite.
            inferred = []
            attention_inputs = []
            memories = []
            fc0_inputs = []
            handles = [
                model.mac.agent.mae.register_forward_hook(
                    lambda module, inputs, output: inferred.append(output[0].detach().clone())),
                model.mac.agent.self_obs_attention.register_forward_pre_hook(
                    lambda module, inputs: attention_inputs.append(inputs[0].detach().clone())),
                model.mac.agent.self_obs_attention.register_forward_hook(
                    lambda module, inputs, output: memories.append(output[1].detach().clone())),
                model.mac.agent.fc0.register_forward_pre_hook(
                    lambda module, inputs: fc0_inputs.append(inputs[0].detach().clone())),
            ]
            try:
                values, hidden = evaluate(batch, 2, model)
            finally:
                for handle in handles:
                    handle.remove()
            inputs = model.mac._build_inputs(batch, 2).reshape(24, -1)
            expected = inferred[0][:, 12:24].clone()
            for row in range(24):
                expected[row, row % 12] = inputs[row]
            equal(attention_inputs[0], expected)
            equal(fc0_inputs[0], torch.stack([memories[0][row, row % 12] for row in range(24)]))
            assert values.shape == (2, 12, a.n_actions) and hidden.shape == (2, 12, 128)
            assert torch.isfinite(values).all()
        """)

    def test_native_parameter_shapes_are_preserved_except_scratch_storage(self):
        self.run_native("""
            native = importlib.import_module('modules.agents.n_rnn_agent')
            zeros = torch.zeros
            def compact_scratch(*args, **kwargs):
                shape = args[0] if args and isinstance(args[0], (tuple, list)) else args
                if len(shape) > 1 and shape[0] in (2000, 3000):
                    return zeros((1,) + tuple(shape[1:]), **kwargs).expand(tuple(shape))
                return zeros(*args, **kwargs)
            # Only native scratch tensors are shrunk; every learned layer is constructed natively.
            with mock.patch.object(torch, 'zeros', side_effect=compact_scratch):
                reference = native.NRNNAgent(b.args.input_shape, b.args)
            shapes = lambda module: {name: tuple(value.shape) for name, value in module.named_parameters()}
            assert shapes(b.mac.agent) == shapes(reference)
            assert {name: tuple(value.shape) for name, value in b.mac.agent.state_dict().items()} == {
                name: tuple(value.shape) for name, value in reference.state_dict().items()}
            assert isinstance(b.mac.agent, native.NRNNAgent)
            assert isinstance(b.mac.agent.mae, importlib.import_module('components.masked_transformer').Base_Transformer)
            assert b.learner.__class__.__name__ == 'NQLearner'
            assert b.learner.mixer.__class__.__module__ == 'modules.mixers.nmix'
            for module, names in (
                (b.mac.agent, ('input_mae_obs_base', 'input_mae_action_base')),
                (b.mac.agent.mae, ('input_obs_base', 'input_action_base', 'embeded_input_1_base',
                                   'masked_data_base', 'mask_base')),
            ):
                for name in names:
                    value = getattr(module, name)
                    assert value.stride(0) == 0, name
                    assert not value.requires_grad, name
            defaults = api.settings(root)
            assert (defaults['MT_n_embd'], defaults['MT_n_enc_layer'], defaults['MT_n_dec_layer'],
                    defaults['MT_traj_length'], defaults['rnn_hidden_dim']) == (24, 3, 2, 5, 128)
        """)

    def test_rl_keeps_maes_in_eval_and_targets_frozen_without_gradients(self):
        self.run_native("""
            batch = episode()
            before = api.snapshot(b)
            modes = {'live': [], 'target': []}
            handles = []
            for name, actor in (('live', b.mac.agent), ('target', b.learner.target_mac.agent)):
                actor.train()
                assert actor.training and not actor.mae.training
                handles.append(actor.mae.register_forward_pre_hook(
                    lambda module, inputs, name=name: modes[name].append(module.training)))
            try:
                b.learner.train(batch, 8, 2)
            finally:
                for handle in handles:
                    handle.remove()
            # Activation checkpointing may recompute live forwards during backward.
            assert len(modes['live']) >= 5 and not any(modes['live']), modes
            assert modes['target'] == [False] * 5, modes
            for module in (b.mac.agent.mae, b.learner.target_mac.agent, b.learner.target_mixer):
                assert all(not parameter.requires_grad for parameter in module.parameters())
                assert all(parameter.grad is None for parameter in module.parameters())
            after = api.snapshot(b)
            for name in ('target_actor', 'target_mixer'):
                for key, value in before[name].items():
                    equal(after[name][key], value)
            for key, value in before['actor'].items():
                if key.startswith('mae.'):
                    equal(after['actor'][key], value)
            assert any(not torch.equal(value, after['actor'][key])
                       for key, value in before['actor'].items() if not key.startswith('mae.'))
            assert any(not torch.equal(value, after['mixer'][key]) for key, value in before['mixer'].items())
            assert np.isfinite(b.logger.values['loss_td'])
        """)

    def test_train_mae_uses_sgd_updates_and_refreezes_live_and_target_maes(self):
        self.run_native("""
            before = api.snapshot(b)
            with mock.patch.object(torch.optim, 'SGD', wraps=torch.optim.SGD) as sgd:
                loss = api.train_mae(b, episode(), 8, 2, repeats=1)
            sgd.assert_called_once()
            assert np.isfinite(loss)
            after = api.snapshot(b)
            assert any(not torch.equal(value, after['actor'][key])
                       for key, value in before['actor'].items() if key.startswith('mae.'))
            for key, value in after['actor'].items():
                if key.startswith('mae.'):
                    equal(after['target_actor'][key], value)
                else:
                    equal(value, before['actor'][key])
            for name in ('mixer', 'target_mixer'):
                for key, value in after[name].items():
                    equal(value, before[name][key])
            for actor in (b.mac.agent, b.learner.target_mac.agent):
                actor.train()
                assert not actor.mae.training
                assert all(not parameter.requires_grad for parameter in actor.mae.parameters())
            assert all(parameter.grad is None for parameter in b.learner.target_mac.agent.parameters())
        """)

    def test_cached_live_and_target_q_and_hidden_match_direct_without_transformer_calls(self):
        self.run_native("""
            batch = episode()
            batch['filled'].fill_(1)
            batch['terminated'].zero_()
            batch['terminated'][:, 3, 0] = 1
            b.buffer.insert_episode_batch(batch)
            controllers = {'live': b.mac, 'target': b.learner.target_mac}
            calls = {(name, layer): 0 for name in controllers for layer in ('encoder', 'decoder')}
            def record(name, layer):
                def hook(module, inputs):
                    calls[name, layer] += 1
                return hook
            handles = []
            for name, mac in controllers.items():
                mac.agent.eval()
                transformer = mac.agent.mae.transformer_model
                for layer in ('encoder', 'decoder'):
                    handles.append(getattr(transformer, layer).register_forward_pre_hook(record(name, layer)))
            try:
                cached = api.sample(b, 2)
                assert calls['live', 'encoder'] > 0 and calls['live', 'decoder'] > 0
                assert calls['target', 'encoder'] == calls['target', 'decoder'] == 0
                assert cached.mae_latents.dtype == torch.float32 and not cached.mae_latents.requires_grad
                assert set(b.buffer.latent_cache) == {0, 1}
                for value in b.buffer.latent_cache.values():
                    assert value.dtype == torch.float32 and not value.requires_grad
                    assert value.shape == (5, b.args.n_agents, b.args.n_agents, 8)
                warmed_calls = dict(calls)
                again = api.sample(b, 2)
                assert calls == warmed_calls
                equal(again.mae_latents, cached.mae_latents)
                direct = api.sample(b, 2, mae=True)
                assert not hasattr(direct, 'mae_latents') and calls == warmed_calls
                for key in batch.scheme:
                    equal(direct[key], cached[key])
                hidden = torch.linspace(-0.1, 0.1, 2 * b.args.n_agents * 128).reshape(2, b.args.n_agents, 128)
                expected = {}
                with torch.no_grad():
                    for name, mac in controllers.items():
                        mac.hidden_states = hidden.clone()
                        expected[name] = []
                        for time in range(4):
                            values = mac.forward(direct, time, test_mode=True)
                            expected[name].append((values.clone(), mac.hidden_states.clone()))
                    direct_calls = dict(calls)
                    for name in controllers:
                        for layer in ('encoder', 'decoder'):
                            assert direct_calls[name, layer] - warmed_calls[name, layer] == 4
                    for repeat in range(2):
                        for name, mac in controllers.items():
                            mac.hidden_states = hidden.clone()
                            for time in range(4):
                                values = mac.forward(cached, time, test_mode=True)
                                q, state = expected[name][time]
                                # Different transformer batch sizes can change float32 rounding.
                                torch.testing.assert_close(values, q, rtol=1e-5, atol=1e-7)
                                torch.testing.assert_close(mac.hidden_states, state, rtol=1e-5, atol=1e-7)
                assert calls == direct_calls, (calls, direct_calls)
                b.learner.train(cached, 8, 2)
                assert calls == direct_calls, 'cached Q-learning must not execute either transformer'
            finally:
                for handle in handles:
                    handle.remove()
        """)

    def test_train_mae_invalidates_replay_latents_and_clears_gradients_before_adam(self):
        self.run_native("""
            b.buffer.insert_episode_batch(episode())
            calls = []
            handle = b.mac.agent.mae.transformer_model.encoder.register_forward_pre_hook(
                lambda module, inputs: calls.append(module.training))
            try:
                api.sample(b, 2)
                old_cache = {key: value.clone() for key, value in b.buffer.latent_cache.items()}
                assert set(old_cache) == {0, 1}
                warmed_calls = len(calls)
                pretraining = api.sample(b, 2, mae=True)
                assert not hasattr(pretraining, 'mae_latents') and len(calls) == warmed_calls
                assert set(b.buffer.latent_cache) == set(old_cache)
                before = {key: value.clone() for key, value in b.mac.agent.mae.state_dict().items()}
                loss = api.train_mae(b, pretraining, 8, 2, repeats=1)
                assert np.isfinite(loss)
                assert any(not torch.equal(value, b.mac.agent.mae.state_dict()[key])
                           for key, value in before.items())
                assert not b.buffer.latent_cache, 'MAE updates must invalidate every replay latent'
                assert any(calls[warmed_calls:]), 'pretraining must execute the transformer in train mode'
                for actor in (b.mac.agent, b.learner.target_mac.agent):
                    actor.train()
                    assert not actor.mae.training
                    assert all(not parameter.requires_grad for parameter in actor.mae.parameters())
                    assert all(parameter.grad is None for parameter in actor.mae.parameters())
                trained_calls = len(calls)
                cached = api.sample(b, 2)
                assert len(calls) > trained_calls, 'sampling after MAE training must recompute latents'
                assert set(b.buffer.latent_cache) == set(old_cache)
                assert any(not torch.equal(value, old_cache[key])
                           for key, value in b.buffer.latent_cache.items())
                refreshed_calls = len(calls)
                repeated = api.sample(b, 2)
                equal(repeated.mae_latents, cached.mae_latents)
                assert len(calls) == refreshed_calls
                frozen = {name: {key: value.clone() for key, value in actor.mae.state_dict().items()}
                          for name, actor in (('live', b.mac.agent), ('target', b.learner.target_mac.agent))}
                b.learner.train(cached, 8, 2)
                assert len(calls) == refreshed_calls
                for name, actor in (('live', b.mac.agent), ('target', b.learner.target_mac.agent)):
                    for key, value in actor.mae.state_dict().items():
                        equal(value, frozen[name][key])
                    assert all(parameter.grad is None for parameter in actor.mae.parameters())
            finally:
                handle.remove()
        """)

    def test_checkpoint_roundtrip_restores_modules_counters_and_predictions(self):
        self.run_native("""
            batch = episode()
            api.train_mae(b, batch, 8, 2, repeats=1)
            b.learner.train(batch, 8, 2)
            b.learner.last_target_update_episode = 17
            counters = {'episodes': 2, 'mae_updates': 1}
            before = api.snapshot(b)
            predictions = evaluate(batch, 2)
            optimizer_before = copy.deepcopy(b.learner.optimiser.state_dict())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                api.checkpoint(path, b, config, 8, counters)
                assert path.is_file() and not path.with_suffix('.pt.tmp').exists()
                with torch.no_grad():
                    for module in api.modules(b).values():
                        for parameter in module.parameters():
                            parameter.add_(1)
                b.learner.last_target_update_episode = -1
                saved = api.restore(path, b)
                fresh, _ = build()
                assert not fresh.learner.optimiser.state_dict()['state']
                api.restore(path, fresh)
            assert saved['algorithm'] == 'ma2e'
            assert saved['environment_steps'] == 8 and saved['counters'] == counters
            assert saved['env_config'] == config.to_dict()
            assert saved['algorithm_args']['rnn_hidden_dim'] == 128
            assert saved['optimizer']['state']
            assert b.learner.last_target_update_episode == 17
            for name, values in api.snapshot(b).items():
                for key, value in values.items():
                    equal(value, before[name][key])
            restored = evaluate(batch, 2)
            equal(restored[0], predictions[0])
            equal(restored[1], predictions[1])
            for name, values in api.snapshot(fresh).items():
                for key, value in values.items():
                    equal(value, before[name][key])
            actual_optimizer = fresh.learner.optimiser.state_dict()
            assert actual_optimizer['param_groups'] == optimizer_before['param_groups']
            assert actual_optimizer['state'].keys() == optimizer_before['state'].keys(), (
                'restore must reload the saved optimizer into a fresh backend')
            for key, expected_state in optimizer_before['state'].items():
                for name, expected_value in expected_state.items():
                    actual = actual_optimizer['state'][key][name]
                    if torch.is_tensor(expected_value):
                        equal(actual, expected_value)
                    else:
                        assert actual == expected_value
        """)


if __name__ == "__main__":
    unittest.main()
