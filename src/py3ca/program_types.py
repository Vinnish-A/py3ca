from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

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
    gene_set: FrozenSet[str] = field(init=False)

    def __post_init__(self) -> None:
        self.gene_set = frozenset(self.genes)
