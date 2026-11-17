"""Independent, allocation-ordered environment seeds for replicated studies."""


class EpisodeSeedSequence:
    def __init__(self, training_seed, episode_index=0):
        if training_seed < 0 or episode_index < 0:
            raise ValueError("Training seed and episode index must be nonnegative")
        self.base = 1_000_000 + training_seed * 100_000_000
        self.episode_index = episode_index

    def next_seeds(self, count):
        if count < 1:
            raise ValueError("Episode count must be positive")
        start = self.episode_index
        self.episode_index += count
        return list(range(self.base + start, self.base + self.episode_index))


def episode_seeds(generator, count):
    if isinstance(generator, EpisodeSeedSequence):
        return generator.next_seeds(count)
    return [int(generator.integers(0, 2 ** 31 - 1)) for _ in range(count)]
