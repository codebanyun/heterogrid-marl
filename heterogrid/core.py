"""Algorithm-independent, simultaneous grid combat and legal message transport."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import EnvConfig, ROLES, RoleStats, UnitSpec

# Shared action numbering, regardless of role. Dead units only allow WAIT.
WAIT = 0
ATTACK_OFFSET = 7
MOVES = np.asarray([(0, 0, 0), (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int32)
SELF_FEATURES = 12
ALLY_FEATURES = 8
TARGET_FEATURES = 11
REWARD_COMPONENTS = ("damage", "kill", "terminal", "time", "discovery", "sharing")


@dataclass
class Unit:
    name: str
    team: str
    spec: UnitSpec
    stats: RoleStats
    position: np.ndarray
    hp: float

    @property
    def alive(self):
        return self.hp > 0


@dataclass
class Track:
    position: np.ndarray
    hp: float
    role: str
    seen_at: int
    reported: bool = False


class GridCombat:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.names = {t: [f"{t}_{i}" for i in range(len(getattr(config, t)))] for t in ("red", "blue")}
        self.all_names = self.names["red"] + self.names["blue"]
        self.rng = np.random.default_rng()
        self.team_communication = {t: config.communication_for(t) for t in self.names}
        self._independent_communication = config.red_communication is not None or config.blue_communication is not None
        self._communication_rng = {t: np.random.default_rng() for t in self.names}
        self.units: dict[str, Unit] = {}
        self.step_count = 0
        self.finished = False
        self.winner: str | None = None
        self.reason: str | None = None
        self.last_attacks: list[dict] = []
        self.last_deliveries: list[dict] = []
        self.episode_returns = {"red": 0.0, "blue": 0.0}
        self.tracks: dict[str, dict[str, Track]] = {}
        self.inboxes: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
        self.online: dict[str, bool] = {}
        self.last_metrics: dict[str, Any] = {}
        self.episode_metrics: dict[str, dict] = {}

    @staticmethod
    def enemy_team(team: str):
        return "blue" if team == "red" else "red"

    def reset(self, seed=None, positions=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self._independent_communication:
                streams = np.random.SeedSequence(seed).spawn(2)
                self._communication_rng = {t: np.random.default_rng(s) for t, s in zip(self.names, streams)}
        self.units = {}
        positions = {} if positions is None else positions
        if set(positions) - set(self.all_names):
            raise ValueError("positions contains an unknown unit ID")
        size = np.asarray(self.config.map_size, dtype=np.int32)
        flank = max(1, int(size[0]) // 4)
        for team in ("red", "blue"):
            for i, spec in enumerate(getattr(self.config, team)):
                name = self.names[team][i]
                position = self.rng.integers(0, size, dtype=np.int32)
                position[0] = self.rng.integers(0, flank) if team == "red" else self.rng.integers(size[0] - flank, size[0])
                if name in positions:
                    raw = np.asarray(positions[name])
                    if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                        raise ValueError(f"Invalid integer grid position for {name}")
                    if np.any(raw < 0) or np.any(raw >= size):
                        raise ValueError(f"Position for {name} is outside map")
                    position = raw.astype(np.int32)
                stats = self.config.role_stats[spec.role]
                self.units[name] = Unit(name, team, spec, stats, position, stats.hp)
        self.step_count = 0
        self.finished = False
        self.winner = self.reason = None
        self.last_attacks = []
        self.last_deliveries = []
        self.episode_returns = {"red": 0.0, "blue": 0.0}
        self.last_metrics = {}
        self.episode_metrics = {t: {"first_contact_step": None, "discovered_targets": 0,
                                  "useful_reports": 0, "damage": 0.0, "kills": 0,
                                  "information_reward": 0.0} for t in self.names}
        self._discovered = {t: set() for t in self.names}
        self._rewarded_reports = {t: set() for t in self.names}
        self._pending_information = {t: {"discovery": 0, "sharing": 0} for t in self.names}
        self.tracks = {name: {} for name in self.all_names}
        self.inboxes = {name: {} for name in self.all_names}
        self._sample_network()
        self._refresh_perception(reward_discoveries=False)
        if self.config.target_sharing == "auto":
            for team in self.names:
                self.share_visible_targets(team, reward_reports=False)

    def _sample_network(self):
        # A random node outage applies to both send and receive, for one decision tick.
        if self._independent_communication:
            self.online = {}
            for team, names in self.names.items():
                config = self.team_communication[team]
                # Draw for ALL slots each tick, including dead/offline slots. Neither
                # the other team's configuration nor early deaths shift this stream.
                draws = self._communication_rng[team].random(len(names))
                for name, draw in zip(names, draws):
                    self.online[name] = (self.units[name].alive and config.mode != "none" and
                                         draw >= config.dropout_probability)
            return
        # No overrides: preserve historical RNG consumption and trajectories exactly.
        probability = self.config.dropout_probability if self.config.communication == "random" else 0.0
        self.online = {n: self.units[n].alive and self.config.communication != "none" and self.rng.random() >= probability
                       for n in self.all_names}

    def can_communicate(self, sender: str, receiver: str):
        a, b = self.units[sender], self.units[receiver]
        if sender == receiver or a.team != b.team or not self.online[sender] or not self.online[receiver]:
            return False
        config = self.team_communication[a.team]
        return config.mode != "none" and (config.mode != "distance" or
                np.linalg.norm(a.position - b.position) <= config.radius)

    def comm_mask(self, team: str):
        names = self.names[team]
        return np.asarray([[self.can_communicate(a, b) for b in names] for a in names], dtype=np.int8)

    def visible(self, observer: str, target: str):
        a, b = self.units[observer], self.units[target]
        return a.alive and b.alive and np.linalg.norm(a.position - b.position) <= a.stats.vision

    def _refresh_perception(self, reward_discoveries=True):
        for name, unit in self.units.items():
            self.tracks[name] = {n: track for n, track in self.tracks[name].items()
                                 if self.step_count - track.seen_at <= self.config.target_ttl}
            self.inboxes[name] = {n: item for n, item in self.inboxes[name].items()
                                  if self.step_count - item[1] <= self.config.target_ttl}
            if not unit.alive:
                self.tracks[name].clear()
                self.inboxes[name].clear()
                continue
            for target_name in self.names[self.enemy_team(unit.team)]:
                target = self.units[target_name]
                # A visible corpse is locally observable; a remote death never clears a track.
                if np.linalg.norm(unit.position - target.position) <= unit.stats.vision:
                    self.tracks[name][target_name] = Track(target.position.copy(), target.hp,
                                                         target.spec.role, self.step_count)
                    if target.alive and target_name not in self._discovered[unit.team]:
                        self._discovered[unit.team].add(target_name)
                        metrics = self.episode_metrics[unit.team]
                        metrics["discovered_targets"] += 1
                        if metrics["first_contact_step"] is None:
                            metrics["first_contact_step"] = self.step_count
                        if reward_discoveries:
                            self._pending_information[unit.team]["discovery"] += 1

    def exchange_messages(self, messages: dict[str, dict], allowed_senders: set[str] | None = None,
                          *, reward_reports=True):
        """Deliver one-hop messages at the current tick, without advancing physics.

        Payload: {sender: {content: [-1..1]*message_dim, targets: [enemy IDs],
                           recipients: [ally IDs]}}. Omitted recipients means broadcast.
        Target reports are built ONLY from the sender's current direct perception.
        Validation is atomic; invalid messages never partially update a receiver.
        """
        if self.finished:
            raise RuntimeError("Episode has ended; reset before exchanging messages")
        planned = []
        for sender, payload in messages.items():
            if sender not in self.units or (allowed_senders is not None and sender not in allowed_senders):
                raise ValueError(f"Invalid sender {sender!r}")
            if not isinstance(payload, dict) or set(payload) - {"content", "targets", "recipients"}:
                raise ValueError("Messages must contain only content, targets and recipients")
            unit = self.units[sender]
            if not unit.alive:
                raise ValueError("Dead units cannot send messages")
            content = payload.get("content")
            if content is not None:
                content = np.asarray(content, dtype=np.float32)
                if content.shape != (self.config.message_dim,) or not np.all(np.isfinite(content)) or np.any(np.abs(content) > 1):
                    raise ValueError("content must be a finite vector in [-1,1] with message_dim elements")
            targets = list(payload.get("targets", []))
            reports = {}
            for target_name in targets:
                if target_name not in self.names[self.enemy_team(unit.team)] or not self.visible(sender, target_name):
                    raise ValueError(f"{sender} cannot report unseen target {target_name!r}")
                target = self.units[target_name]
                reports[target_name] = Track(target.position.copy(), target.hp, target.spec.role, self.step_count, True)
            recipients = list(payload.get("recipients", [n for n in self.names[unit.team] if n != sender]))
            if any(n not in self.names[unit.team] or n == sender for n in recipients):
                raise ValueError("Recipients must be other units on the sender's team")
            planned.append((sender, content, reports, set(recipients)))
        for sender, content, reports, recipients in planned:
            team = self.units[sender].team
            for receiver in sorted(recipients):
                if not self.can_communicate(sender, receiver):
                    continue
                if content is not None:
                    self.inboxes[receiver][sender] = (content.copy(), self.step_count)
                for target_name, report in reports.items():
                    current = self.tracks[receiver].get(target_name)
                    if current is None or current.seen_at < report.seen_at:
                        ally = self.units[receiver]
                        actionable_before = (current is not None and current.hp > 0 and
                                             np.linalg.norm(ally.position - current.position) <= ally.stats.attack_range)
                        actionable_after = (ally.stats.damage > 0 and report.hp > 0 and
                                            np.linalg.norm(ally.position - report.position) <= ally.stats.attack_range)
                        pair = (receiver, target_name)
                        # Credit an actual new firing opportunity, once per receiver/target per episode.
                        if actionable_after and not actionable_before and pair not in self._rewarded_reports[team]:
                            self._rewarded_reports[team].add(pair)
                            self.episode_metrics[team]["useful_reports"] += 1
                            if reward_reports:
                                self._pending_information[team]["sharing"] += 1
                        self.tracks[receiver][target_name] = Track(report.position.copy(), report.hp,
                                                                 report.role, report.seen_at, True)
                if content is not None or reports:
                    self.last_deliveries.append({"sender": sender, "receiver": receiver,
                                                 "targets": list(reports), "has_content": content is not None,
                                                 "step": self.step_count})

    def share_visible_targets(self, team: str, *, reward_reports=True):
        messages = {}
        for name in self.names[team]:
            if self.units[name].alive:
                targets = [t for t in self.names[self.enemy_team(team)] if self.visible(name, t)]
                if targets:
                    messages[name] = {"targets": targets}
        self.exchange_messages(messages, reward_reports=reward_reports)

    def n_actions(self, team: str):
        return ATTACK_OFFSET + len(self.names[self.enemy_team(team)])

    def action_mask(self, name: str):
        unit = self.units[name]
        mask = np.zeros(self.n_actions(unit.team), dtype=np.int8)
        mask[WAIT] = 1
        if not unit.alive:
            return mask
        size = np.asarray(self.config.map_size)
        for a in range(1, ATTACK_OFFSET):
            new = unit.position + MOVES[a]
            mask[a] = int(np.all(new >= 0) and np.all(new < size))
        if unit.stats.damage > 0:
            for j, target_name in enumerate(self.names[self.enemy_team(unit.team)]):
                track = self.tracks[name].get(target_name)
                # Use only cached local information, never the enemy's hidden live state.
                if track is not None and track.hp > 0 and np.linalg.norm(unit.position - track.position) <= unit.stats.attack_range:
                    mask[ATTACK_OFFSET + j] = 1
        return mask

    def obs_size(self, team: str):
        return SELF_FEATURES + ALLY_FEATURES * (len(self.names[team]) - 1) + TARGET_FEATURES * len(self.names[self.enemy_team(team)])

    def observation(self, name: str):
        unit = self.units[name]
        size = np.maximum(np.asarray(self.config.map_size) - 1, 1)
        diagonal = max(float(np.linalg.norm(size)), 1.0)
        max_damage = max(1.0, *(s.damage for s in self.config.role_stats.values()))
        role = np.eye(3, dtype=np.float32)[ROLES.index(unit.spec.role)]
        self_vec = [*(unit.position / size), unit.hp / unit.stats.hp, *role,
                    min(unit.stats.vision / diagonal, 1), min(unit.stats.attack_range / diagonal, 1),
                    unit.stats.damage / max_damage, self.step_count / self.config.max_steps, float(self.online[name])]
        values = list(self_vec)
        for ally_name in self.names[unit.team]:
            if ally_name == name:
                continue
            ally = self.units[ally_name]
            if self.visible(name, ally_name):
                values.extend([1, *((ally.position - unit.position) / size), ally.hp / ally.stats.hp,
                               *np.eye(3)[ROLES.index(ally.spec.role)]])
            else:
                values.extend([0] * ALLY_FEATURES)
        for target_name in self.names[self.enemy_team(unit.team)]:
            track = self.tracks[name].get(target_name)
            if track is None:
                values.extend([0] * TARGET_FEATURES)
            else:
                values.extend([1, float(self.visible(name, target_name)), *((track.position - unit.position) / size),
                               track.hp / self.config.role_stats[track.role].hp,
                               *np.eye(3)[ROLES.index(track.role)],
                               (self.step_count - track.seen_at) / self.config.target_ttl, float(track.reported)])
        messages = np.zeros((len(self.names[unit.team]), self.config.message_dim), dtype=np.float32)
        msg_mask = np.zeros(len(self.names[unit.team]), dtype=np.int8)
        ages = np.zeros(len(self.names[unit.team]), dtype=np.float32)
        for j, sender in enumerate(self.names[unit.team]):
            if sender in self.inboxes[name]:
                content, tick = self.inboxes[name][sender]
                messages[j] = content
                msg_mask[j] = 1
                ages[j] = (self.step_count - tick) / self.config.target_ttl
        vector = np.asarray(values, dtype=np.float32) if unit.alive else np.zeros(self.obs_size(unit.team), dtype=np.float32)
        return {"observation": vector, "action_mask": self.action_mask(name),
                "messages": messages, "message_mask": msg_mask, "message_age": ages}

    def state(self):
        values = []
        size = np.maximum(np.asarray(self.config.map_size) - 1, 1)
        for name in self.all_names:
            unit = self.units[name]
            values.extend([float(unit.team == "red"), float(unit.team == "blue"),
                           *(unit.position / size), unit.hp / unit.stats.hp, float(unit.alive),
                           *np.eye(3)[ROLES.index(unit.spec.role)], float(self.online[name])])
        values.append(self.step_count / self.config.max_steps)
        return np.asarray(values, dtype=np.float32)

    def step(self, actions: dict[str, int]):
        if self.finished:
            raise RuntimeError("Episode has ended; reset before stepping")
        alive = [n for n in self.all_names if self.units[n].alive]
        if set(actions) != set(alive):
            raise ValueError("Core step requires exactly one action per living unit")
        clean = {}
        invalid = []
        for name, action in actions.items():
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions(self.units[name].team):
                raise ValueError(f"Action out of space for {name}: {action!r}")
            if not self.action_mask(name)[action]:
                invalid.append(name)
                action = WAIT
            clean[name] = int(action)
        damage = {name: 0.0 for name in self.all_names}
        self.last_attacks = []
        for name in alive:
            action = clean[name]
            if action < ATTACK_OFFSET:
                continue
            unit = self.units[name]
            target_name = self.names[self.enemy_team(unit.team)][action - ATTACK_OFFSET]
            target = self.units[target_name]
            aim = self.tracks[name][target_name].position
            hit = target.alive and np.array_equal(aim, target.position)
            if hit:
                damage[target_name] += unit.stats.damage
            self.last_attacks.append({"attacker": name, "target": target_name, "origin": unit.position.tolist(),
                                      "aim": aim.tolist(), "hit": bool(hit)})
        # Fire from the shared start-of-tick snapshot. All damage is applied together.
        actual_damage = {"red": 0.0, "blue": 0.0}
        kills = {"red": 0, "blue": 0}
        died = []
        for name in alive:
            unit = self.units[name]
            effective = min(unit.hp, damage[name])
            scoring_team = self.enemy_team(unit.team)
            actual_damage[scoring_team] += effective
            unit.hp = max(0.0, unit.hp - effective)
            if not unit.alive:
                died.append(name)
                kills[scoring_team] += 1
        # Only survivors move; stacking on a cell is deliberately permitted.
        for name in alive:
            action = clean[name]
            if self.units[name].alive and 0 < action < ATTACK_OFFSET:
                self.units[name].position += MOVES[action]
        self.step_count += 1
        counts = {t: sum(self.units[n].alive for n in self.names[t]) for t in ("red", "blue")}
        self.finished = not counts["red"] or not counts["blue"] or self.step_count >= self.config.max_steps
        if self.finished:
            if not counts["red"] and not counts["blue"]:
                self.reason = "mutual_elimination"
            elif not counts["blue"]:
                self.winner, self.reason = "red", "elimination"
            elif not counts["red"]:
                self.winner, self.reason = "blue", "elimination"
            else:
                self.reason = "timeout"
        self._sample_network()
        self.last_deliveries = []
        self._refresh_perception()
        if self.config.target_sharing == "auto" and not self.finished:
            for team in self.names:
                self.share_visible_targets(team)
        components = {t: dict.fromkeys(REWARD_COMPONENTS, 0.0) for t in self.names}
        events = {t: dict(self._pending_information[t]) for t in self.names}
        for team in self.names:
            metrics = self.episode_metrics[team]
            metrics["damage"] += actual_damage[team]
            metrics["kills"] += kills[team]
            components[team]["time"] = -self.config.time_penalty
            if self.config.reward_mode == "dense":
                components[team]["damage"] = self.config.damage_reward * actual_damage[team]
                components[team]["kill"] = self.config.kill_reward * kills[team]
                remaining = max(0.0, self.config.information_reward_cap - metrics["information_reward"])
                for kind, weight in (("discovery", self.config.discovery_reward), ("sharing", self.config.sharing_reward)):
                    bonus = min(remaining, events[team][kind] * weight)
                    components[team][kind] = bonus
                    remaining -= bonus
                    metrics["information_reward"] += bonus
            self._pending_information[team] = {"discovery": 0, "sharing": 0}
        if self.winner is not None:
            components[self.winner]["terminal"] = self.config.win_reward
            components[self.enemy_team(self.winner)]["terminal"] = self.config.loss_reward
        elif self.reason == "timeout" and self.config.timeout_as_loss:
            components["red"]["terminal"] = self.config.loss_reward
        rewards = {t: sum(components[t].values()) for t in self.names}
        for team in rewards:
            self.episode_returns[team] += rewards[team]
        self.last_metrics = {"damage": actual_damage, "kills": kills, "died": died, "invalid_actions": invalid,
                             "team_rewards": rewards, "reward_components": components,
                             "information_events": events, "winner": self.winner, "reason": self.reason}
        return rewards, self.last_metrics

    def snapshot(self):
        """Privileged data for rendering/replays; never feed this into a deployed actor."""
        return {"schema_version": 1, "step": self.step_count, "map_size": list(self.config.map_size),
                "max_steps": self.config.max_steps, "finished": self.finished,
                "winner": self.winner, "reason": self.reason, "returns": dict(self.episode_returns),
                "reward_components": self.last_metrics.get("reward_components", {}),
                "episode_metrics": {t: dict(m) for t, m in self.episode_metrics.items()},
                "communication": self.config.communication,
                "independent_communication": self._independent_communication,
                "team_communication": {t: asdict(c) for t, c in self.team_communication.items()},
                "units": [{"id": n, "team": u.team, "role": u.spec.role, "platform": u.spec.platform,
                           "position": u.position.tolist(), "hp": u.hp, "max_hp": u.stats.hp, "alive": u.alive,
                           "vision": u.stats.vision, "attack_range": u.stats.attack_range,
                           "damage": u.stats.damage, "online": self.online[n]} for n, u in self.units.items()],
                "links": [[a, b] for t in ("red", "blue") for i, a in enumerate(self.names[t])
                          for b in self.names[t][i + 1:] if self.can_communicate(a, b)],
                "attacks": list(self.last_attacks), "deliveries": list(self.last_deliveries)}
