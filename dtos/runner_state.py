import datetime
from typing import List
from dtos.ladder_position import LadderPosition


class RunnerState:
    runner_id: str
    name: str
    adjustment_factor: float
    back_ladder: List[LadderPosition]
    handicap: float
    last_price_traded: float
    lay_ladder: List[LadderPosition]
    removal_date: datetime
    runner_id: int
    sp_actual: float
    sp_near: float
    sp_far: float
    status: str
    traded_volume: float
    total_matched: float
    wap: float

    def to_dict(self):
        return {
            **self.__dict__,
            "back_ladder": [p.__dict__ for p in self.back_ladder],
            "lay_ladder": [p.__dict__ for p in self.lay_ladder],
        }
