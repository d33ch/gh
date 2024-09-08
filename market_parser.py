import bz2
from betfair_data import bflw, PriceSize
from typing import List
from base.base_market_parser import BaseMarketParser
from dtos.ladder_position import LadderPosition
from dtos.market import Market
from dtos.market_state import MarketState
from dtos.runner import Runner
from dtos.runner_state import RunnerState


class MarketParser(BaseMarketParser):
    def __init__(self):
        super().__init__()

    def parse_market_info(self, market_book: bflw.MarketBook) -> Market:
        return Market(
            market_book.market_definition.betting_type,
            market_book.market_definition.country_code,
            market_book.market_definition.event_name,
            market_book.market_definition.event_id,
            market_book.market_definition.event_type_id,
            market_book.market_id,
            market_book.market_definition.name,
            market_book.market_definition.venue,
            market_book.market_definition.market_base_rate,
            market_book.market_definition.market_time,
            market_book.market_definition.market_type,
            market_book.market_definition.open_date,
            market_book.market_definition.race_type,
            market_book.market_definition.regulators,
            market_book.market_definition.settled_time,
            market_book.market_definition.suspend_time,
            market_book.market_definition.timezone,
            self.parse_runners(market_book.market_definition.runners),
        )

    def parse_market_state(self, market_book: bflw.MarketBook) -> MarketState:
        return MarketState(
            market_book.market_id,
            market_book.bsp_reconciled,
            market_book.inplay,
            market_book.number_of_active_runners,
            market_book.number_of_runners,
            market_book.number_of_winners,
            market_book.publish_time_epoch,
            market_book.publish_time,
            market_book.status,
            market_book.total_available,
            market_book.total_matched,
            self.parse_runners_states(market_book.runners),
        )

    def parse_ladder(self, ladder: List[PriceSize], depth=5) -> List[LadderPosition]:
        return [LadderPosition(position.price, position.size) for position in ladder[:depth]]

    def parse_runners(self, market_runners: List[bflw.MarketDefinitionRunner]) -> List[Runner]:
        return [
            Runner(market_runner.selection_id, market_runner.name, market_runner.handicap, market_runner.removal_date, market_runner.sort_priority)
            for market_runner in market_runners
        ]

    def parse_runners_states(self, market_runners: List[bflw.RunnerBook]) -> List[RunnerState]:
        runners_states: List[RunnerState] = []
        for runner in market_runners:
            try:
                traded_volume = sum([traded.size for traded in runner.ex.traded_volume])
                runner_state = RunnerState(
                    runner.adjustment_factor,
                    self.parse_ladder(runner.ex.available_to_back),
                    runner.handicap,
                    runner.last_price_traded,
                    self.parse_ladder(runner.ex.available_to_lay),
                    runner.removal_date,
                    runner.selection_id,
                    runner.sp.actual_sp,
                    runner.sp.near_price,
                    runner.sp.far_price,
                    runner.status,
                    traded_volume,
                    runner.total_matched,
                )
                runners_states.append(runner_state)
            except Exception as e:
                print(f"error {e}")
        return runners_states
