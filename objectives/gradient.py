from dataclasses import dataclass
from typing import Any

@dataclass
class Gradient:
    grad_h: Any
    grad_J: Any