from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ProcessorConfig:
    steps: List[Tuple[int, int]]
