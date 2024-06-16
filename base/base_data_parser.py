from abc import ABC, abstractmethod


class BaseDataParser(ABC):
    @abstractmethod
    def parse(self, file) -> list:
        pass
