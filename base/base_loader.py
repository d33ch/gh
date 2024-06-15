from abc import ABC, abstractmethod
from bz2 import BZ2File


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> BZ2File:
        pass
