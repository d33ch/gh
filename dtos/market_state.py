from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from dtos.runner_state import RunnerState


@dataclass
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
    total_available: float
    total_matched: float
    runners_states: List[RunnerState] = field(default_factory=list)

    def to_dict(self):
        return {**self.__dict__, "runners_states": [rs.to_dict() for rs in self.runners_states]}
