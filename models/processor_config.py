from typing import List, Tuple


class ProcessorConfig:
    steps: List[Tuple[int, int]]

    def __init__(self, steps: List[Tuple[int, int]]):
        self.steps = steps
