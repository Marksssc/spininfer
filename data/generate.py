from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class SyntheticDataset:
    h: Any
    J: Any
    samples: Any
    seed: int

def generate_data(model, n_samples, iterations=1000, seed=0, h=None, J=None,
                   loc_h=0.0, scale_h=0.2, loc_J=0.0, scale_J=0.3) -> SyntheticDataset:
    if h is None or J is None:
        h, J = model.random_params(loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J, seed=seed)
    samples = model.simulate(h, J, n_samples, iterations, seed=seed)
    return SyntheticDataset(h=h, J=J, samples=samples, seed=seed)