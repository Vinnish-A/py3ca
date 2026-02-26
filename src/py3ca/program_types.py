from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Program:
    program_id: str
    sample_id: str
    study_id: str
    k: int
    component: int
    genes: List[str]
    weights: np.ndarray
    gene_index: Dict[str, int]

    @property
    def gene_set(self) -> set:
        return set(self.genes)