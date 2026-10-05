from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class Moments:
    """First and second moments (mean_s, mean_ss) of a spin/label configuration."""

    mean_s: Any
    mean_ss: Any