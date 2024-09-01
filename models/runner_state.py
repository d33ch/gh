import datetime
from models.price_ladder import PriceLadder


class RunnerState:
    runner_id: str
    adjustment_factor: float
    bet_ladder: PriceLadder
    handicap: float
    last_price_traded: float
    lay_ladder: PriceLadder
    removal_date: datetime
    runner_id: int
    sp_actual: float
    sp_near: float
    sp_far: float
    status: str
    traded_volume: float
    total_matched: float
    wap: float
