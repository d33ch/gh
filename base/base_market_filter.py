from abc import ABC, abstractmethod
import datetime
from typing import List, Tuple
from betfair_data import bflw


class BaseMarketFilter(ABC):
    @abstractmethod
    def filter_market(self, market_book: bflw.MarketBook) -> bool:
        pass

    @abstractmethod
    def filter_time(self, market_time: datetime, publish_time: datetime, last_consumed_time: datetime, steps: List[Tuple[int, int]]) -> bool:
        pass
