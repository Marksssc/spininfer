from dataclasses import dataclass
from objectives.gradient import Gradient

@dataclass
class MomentMatchingObjective:
    estimator: object

    def compute_gradient(self, model, h, J, dataset) -> Gradient:
        empirical = dataset.moments
        model_moments = self.estimator.estimate(model, h, J)
        return Gradient(
            grad_h=empirical.mean_s - model_moments.mean_s,
            grad_J=empirical.mean_ss - model_moments.mean_ss,
        )