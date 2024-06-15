from abc import ABC, abstractmethod
from betfair_data import MarketBook

class BaseMarketFilter(ABC):
    @abstractmethod
    def filter(self, market_book: MarketBook) -> bool:
        pass
