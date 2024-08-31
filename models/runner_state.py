from models.price_ladder import PriceLadder


class RunnerState:
    def __init__(
        self,
        runner_id: str,
        traded_volume: float,
        wap: float,
        ltp: float,
        status: str,
        bet_ladder: PriceLadder,
        lay_ladder: PriceLadder,
        sp_near: float,
        sp_far: float,
    ):
        self.runner_id = runner_id
        self.traded_volume = traded_volume
        self.wap = wap
        self.ltp = ltp
        self.status = status
        self.bet_ladder = bet_ladder
        self.lay_ladder = lay_ladder
        self.sp_near = sp_near
        self.sp_far = sp_far
