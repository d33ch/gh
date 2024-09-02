from datetime import datetime
from typing import List

from models.runner_state import RunnerState


class MarketState:
    market_id: str
    bsp_reconciled: bool
    complete: bool
    inplay: bool
    last_match_time: datetime
    number_of_active_runners: int
    number_of_runners: int
    number_of_winners: int
    runners_state: str
    publish_time_epoch: int
    publish_time: datetime
    status: str
    runners_states: List[RunnerState] = []
    total_available: float
    total_matched: float
