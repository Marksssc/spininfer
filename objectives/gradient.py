from dataclasses import dataclass
import numpy as np

@dataclass
class Gradient:
    grad_h: np.ndarray
    grad_J: np.ndarray