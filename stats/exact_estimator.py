from dataclasses import dataclass
from stats.moments import Moments

@dataclass
class ExactEstimator:
    def estimate(self, model, h, J) -> Moments:
        mean_s, mean_ss = model.exact_statistics(h, J)
        return Moments(mean_s=mean_s, mean_ss=mean_ss)