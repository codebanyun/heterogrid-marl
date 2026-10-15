"""Episode-weighted diagnostic means with explicit non-null observation counts."""


def summarize_team_metrics(rows):
    records = [row["team_episode_metrics"] for row in rows if "team_episode_metrics" in row]
    if not records:
        return {}

    def summarize(values):
        values = [value for value in values if value is not None]
        if values and isinstance(values[0], dict):
            means, counts = {}, {}
            for key in sorted({key for value in values for key in value}):
                if key == "schema_version":
                    continue
                means[key], counts[key] = summarize([value.get(key) for value in values])
            return means, counts
        return (sum(values) / len(values) if values else None), len(values)

    means, counts = summarize(records)
    return {"mean_team_episode_metrics": means, "team_episode_metrics_observed_episodes": counts}
