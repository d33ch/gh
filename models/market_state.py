from datetime import datetime
from typing import List

from models.runner_state import RunnerState


class MarketState:
    market_id: str
    bsp_reconciled: bool
    inplay: bool
    number_of_active_runners: int
    number_of_runners: int
    number_of_winners: int
    publish_time_epoch: int
    publish_time: datetime
    status: str
    runners_states: List[RunnerState]
    total_available: float
    total_matched: float

    def __init__(self):
        self.runners_states = []

    def to_dict(self):
        return {**self.__dict__, "runners_states": [rs.to_dict() for rs in self.runners_states]}
