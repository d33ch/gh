from abc import ABC, abstractmethod
from betfair_data import bflw


class BaseMarketFilter(ABC):
    @abstractmethod
    def filter(self, market_book: bflw.MarketBook) -> bool:
        pass
