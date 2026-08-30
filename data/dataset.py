from dataclasses import dataclass, field
import numpy as np
from stats.moments import Moments

@dataclass
class Dataset:
    samples: np.ndarray 
    model: object
    moments: Moments = field(init=False)

    def __post_init__(self):
        n_sites = self.samples.shape[1]
        if n_sites != self.model.n_sites:
            raise ValueError(f"data has {n_sites} sites, model expects {self.model.n_sites}")
        self.moments = self.model.compute_moments(self.samples)   # computed ONCE, cached