import bz2
from betfair_data import bflw, PriceSize
from typing import List
from base.base_market_parser import BaseMarketParser
from models.ladder_position import LadderPosition
from models.market import Market
from models.market_state import MarketState
from models.runner import Runner
from models.runner_state import RunnerState


class MarketParser(BaseMarketParser):
    def __init__(self):
        super().__init__()

    def parse_market_info(self, market_book: bflw.MarketBook) -> Market:
        market = Market()
        market.betting_type = market_book.market_definition.betting_type
        market.country_code = market_book.market_definition.country_code
        market.event_name = market_book.market_definition.event_name
        market.event_id = market_book.market_definition.event_id
        market.event_type_id = market_book.market_definition.event_type_id
        market.market_id = market_book.market_id
        market.name = market_book.market_definition.name
        market.venue = market_book.market_definition.venue
        market.market_base_rate = market_book.market_definition.market_base_rate
        market.market_time = market_book.market_definition.market_time
        market.market_type = market_book.market_definition.market_type
        market.open_date = market_book.market_definition.open_date
        market.race_type = market_book.market_definition.race_type
        market.regulators = market_book.market_definition.regulators
        market.runners = self.parse_runners(market_book.market_definition.runners)
        market.settled_time = market_book.market_definition.settled_time
        market.suspend_time = market_book.market_definition.suspend_time
        market.timezone = market_book.market_definition.timezone

        return market

    def parse_market_state(self, market_book: bflw.MarketBook) -> MarketState:
        market_state = MarketState()
        market_state.market_id = market_book.market_id
        market_state.bsp_reconciled = market_book.bsp_reconciled
        market_state.inplay = market_book.inplay
        market_state.number_of_active_runners = market_book.number_of_active_runners
        market_state.number_of_runners = market_book.number_of_runners
        market_state.number_of_winners = market_book.number_of_active_runners
        market_state.publish_time_epoch = market_book.publish_time_epoch
        market_state.publish_time = market_book.publish_time
        market_state.runners_states = self.parse_runners_states(market_book.runners)
        market_state.status = market_book.status
        market_state.total_available = market_book.total_available
        market_state.total_matched = market_book.total_matched

        return market_state

    def parse_ladder(self, ladder: List[PriceSize], depth=5) -> List[LadderPosition]:
        return [LadderPosition(position.price, position.size) for position in ladder[:depth]]

    def parse_runners(self, market_runners: List[bflw.MarketDefinitionRunner]) -> List[Runner]:
        return [Runner(market_runner.selection_id, market_runner.name, market_runner.handicap, market_runner.removal_date) for market_runner in market_runners]

    def parse_runners_states(self, market_runners: List[bflw.RunnerBook]) -> List[RunnerState]:
        runners_states: List[RunnerState] = []
        for runner in market_runners:
            try:
                traded_volume = sum([traded.size for traded in runner.ex.traded_volume])
                wap = sum(position.size * position.price for position in runner.ex.traded_volume) / traded_volume if traded_volume > 0 else 0
                wap = round(wap, 2)
                runner_state = RunnerState()
                runner_state.adjustment_factor = runner.adjustment_factor
                runner_state.runner_id = runner.selection_id
                runner_state.traded_volume = traded_volume
                runner_state.back_ladder = self.parse_ladder(runner.ex.available_to_back)
                runner_state.lay_ladder = self.parse_ladder(runner.ex.available_to_lay)
                runner_state.last_price_traded = runner.last_price_traded
                runner_state.sp_actual = runner.sp.actual_sp
                runner_state.sp_near = runner.sp.near_price
                runner_state.sp_far = runner.sp.far_price
                runner_state.wap = wap
                runner_state.status = runner.status
                runners_states.append(runner_state)
            except Exception as e:
                print(f"error {e}")
        return runners_states
