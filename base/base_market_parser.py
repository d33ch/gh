from abc import ABC, abstractmethod
from betfair_data import bflw

from dtos.market import Market
from dtos.market_state import MarketState


class BaseMarketParser(ABC):
    @abstractmethod
    def parse_market_info(self, market_book: bflw.MarketBook) -> Market:
        pass

    @abstractmethod
    def parse_market_state(self, market_book: bflw.MarketBook) -> MarketState:
        pass
