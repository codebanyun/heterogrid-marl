"""Opt-in privileged statistics: never used for observations, RNG or rewards."""
from collections import defaultdict

import numpy as np


class CombatDiagnostics:
    def reset(self, core):
        self.sums = {team: defaultdict(float) for team in core.names}
        self.counts = {team: defaultdict(int) for team in core.names}
        self.initial = {}
        self.shots = {team: dict(attack_attempts=0, invalid_attack_attempts=0,
                                executed_attacks=0, hit_attacks=0, missed_attacks=0,
                                raw_hit_damage=0.0, effective_damage=0.0, overkill_damage=0.0)
                      for team in core.names}
        self.samples = 0
        self.record_state(core)

    def record_state(self, core):
        for team, names in core.names.items():
            alive = [name for name in names if core.units[name].alive]
            enemies = [name for name in core.names[core.enemy_team(team)] if core.units[name].alive]
            scouts = [name for name in alive if core.units[name].spec.role == "scout"]
            strikes = [name for name in alive if core.units[name].spec.role == "strike"]
            degree = sum(core.can_communicate(a, b) for a in alive for b in alive) / len(alive) if alive else 0.0
            values = {"comm_degree": degree,
                      "comm_degree_fraction": degree / (len(alive) - 1) if len(alive) > 1 else None,
                      "alive_fraction": len(alive) / len(names),
                      "scout_coverage_fraction": sum(any(core.visible(s, e) for s in scouts) for e in enemies) / len(enemies) if enemies else None,
                      "strike_scout_link_fraction": sum(any(core.can_communicate(s, a) for s in scouts) for a in strikes) / len(strikes) if strikes else None}
            if self.samples == 0:
                distances = [min(float(np.linalg.norm(core.units[a].position - core.units[b].position))
                                 for b in enemies) for a in alive]
                self.initial[team] = {"initial_agents": len(names),
                    "initial_team_density": len(names) / float(np.prod(core.config.map_size, dtype=np.float64)),
                    "initial_mean_nearest_enemy_distance": sum(distances) / len(distances),
                    **{"initial_" + key: value for key, value in values.items()}}
            for key, value in values.items():
                if value is not None:
                    self.sums[team][key] += value
                    self.counts[team][key] += 1
        self.samples += 1

    def record_attacks(self, core, requested, invalid):
        # Called before simultaneous damage is applied; target HP is start-of-tick HP.
        invalid = set(invalid)
        for name, action in requested.items():
            if action >= 7:
                team = core.units[name].team
                self.shots[team]["attack_attempts"] += 1
                self.shots[team]["invalid_attack_attempts"] += int(name in invalid)
        hit_damage = defaultdict(float)
        for attack in core.last_attacks:
            unit = core.units[attack["attacker"]]
            stats = self.shots[unit.team]
            stats["executed_attacks"] += 1
            stats["hit_attacks"] += int(attack["hit"])
            stats["missed_attacks"] += int(not attack["hit"])
            if attack["hit"]:
                hit_damage[attack["target"]] += unit.stats.damage
        for target, damage in hit_damage.items():
            stats = self.shots[core.enemy_team(core.units[target].team)]
            effective = min(core.units[target].hp, damage)
            stats["raw_hit_damage"] += damage
            stats["effective_damage"] += effective
            stats["overkill_damage"] += damage - effective

    def metrics(self, core, team):
        stats = dict(self.shots[team])
        def ratio(value, denominator):
            return value / denominator if denominator else None
        names = core.names[team]
        combat_agents = sum(core.units[name].stats.damage > 0 for name in names)
        return {"schema_version": 1, "state_samples": self.samples, **self.initial[team],
                **{"mean_" + key: ratio(value, self.counts[team][key])
                   for key, value in self.sums[team].items()},
                # Include undefined metrics too (e.g. no strike nodes at all).
                **{"mean_" + key: None for key in ("comm_degree_fraction", "scout_coverage_fraction", "strike_scout_link_fraction")
                   if key not in self.sums[team]},
                "final_alive_fraction": sum(core.units[n].alive for n in names) / len(names),
                "final_alive_by_role": {role: sum(core.units[n].alive and core.units[n].spec.role == role for n in names)
                                        for role in ("scout", "strike", "hybrid")},
                "useful_reports_per_initial_agent": core.episode_metrics[team]["useful_reports"] / len(names),
                "useful_reports_per_initial_combat_agent": ratio(core.episode_metrics[team]["useful_reports"], combat_agents),
                **stats,
                "ineffective_attack_attempts": stats["invalid_attack_attempts"] + stats["missed_attacks"],
                "invalid_attack_rate": ratio(stats["invalid_attack_attempts"], stats["attack_attempts"]),
                "attack_hit_rate": ratio(stats["hit_attacks"], stats["executed_attacks"]),
                "overkill_fraction": ratio(stats["overkill_damage"], stats["raw_hit_damage"])}
