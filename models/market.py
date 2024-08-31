from typing import List

from models.market_state import MarketState


class Market:
    def __init__(self, name, symbol, market_states: List[MarketState]):
        self.name = name
        self.symbol = symbol
        self.market_states = market_states
