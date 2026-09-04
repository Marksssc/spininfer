from dataclasses import dataclass
from stats.moments import Moments

@dataclass
class McmcEstimator:
    n_samples: int
    iterations: int = 1000
    seed: int = 0
    def estimate(self, model, h, J) -> Moments:
        samples = model.simulate(h, J, self.n_samples, self.iterations, seed=self.seed)
        return model.compute_moments(samples)