from abc import ABC, abstractmethod
from typing import Tuple


class BaseDataUtils(ABC):
    @abstractmethod
    def as_str(self, value):
        pass

    @abstractmethod
    def get_min_positive(a: float, b: float):
        pass

    @abstractmethod
    def slice(l, n, attr):
        pass

    @abstractmethod
    def pull_ladder(ladder, size) -> Tuple[float, float]:
        pass
