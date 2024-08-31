from datetime import datetime
from typing import List

from models.runner_state import RunnerState


class MarketState:

    def __init__(
        self,
        market_id: str,
        runners_count: int,
        runners_state: str,
        state_time: datetime,
        runners: List[RunnerState],
    ):
        self.market_id = market_id
        self.runners_count = runners_count
        self.runners_state = runners_state
        self.state_time = state_time
