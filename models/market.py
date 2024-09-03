import datetime
from typing import List

from models.market_state import MarketState
from models.runner import Runner


class Market:
    betting_type: str
    country_code: str
    event_name: str
    event_type_id: str
    event_id: str
    market_id: str
    name: str
    venue: str
    runners: List[Runner]
    market_states: List[MarketState]
    market_base_rate: float
    market_time: datetime
    market_type: str
    open_date: datetime
    race_type: str
    regulators: str
    settled_time: datetime
    suspend_time: datetime
    timezone: str

    def __init__(self):
        self.runners = []
        self.market_states = []

    def to_dict(self):
        return {
            **self.__dict__,
            "runners": [r.__dict__ for r in self.runners],
            "market_states": [market_state.to_dict() for market_state in self.market_states],
        }
