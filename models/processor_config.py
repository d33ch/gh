from typing import List, Tuple


class ProcessorConfig:
    steps: List[Tuple[int, int]]

    def __init__(self, steps: List[Tuple[int, int]]):
        self.steps = steps

    def get(self, time: int) -> Tuple[int, int]:
        minStep = self.steps[0]
        for step in self.steps[1:]:
            if time <= step[0] and minStep[0] > step[0]:
                minStep = step
        return minStep
