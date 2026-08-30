from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Moments:
    mean_s: np.ndarray
    mean_ss: np.ndarray