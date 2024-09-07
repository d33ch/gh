import datetime
from typing import List, Tuple
from betfair_data import bflw
from base.base_market_filter import BaseMarketFilter
from dtos.market_info import MarketInfo
from dtos.processor_config import ProcessorConfig


def get_step(steps: List[Tuple[int, int]], time: int) -> Tuple[int, int]:
    minStep = steps[0]
    for step in steps[1:]:
        if time <= step[0] and minStep[0] > step[0]:
            minStep = step
    return minStep


class MarketFilter(BaseMarketFilter):
    # TODO: to pass filter config
    def filter_market(self, market_book: bflw.MarketBook) -> bool:
        definition = market_book.market_definition
        date = f"{definition.market_time.year}-{definition.market_time.month}-{definition.market_time.day}"
        return (
            definition != None
            and definition.country_code == "AU"
            and definition.market_type == "WIN"
            and date == "2016-9-21"
            # and (race_type := MarketInfo(definition.name).race_type) != "trot"
            # and race_type != "pace"
        )

    def filter_time(self, market_time: datetime, publish_time: datetime, last_consumed_time: datetime, steps: List[Tuple[int, int]]) -> bool:
        seconds_to_start = (market_time - publish_time).total_seconds()
        step = get_step(steps, seconds_to_start)
        if seconds_to_start > step[0]:
            return False
        delta_time_seconds = (publish_time - last_consumed_time).total_seconds()
        if delta_time_seconds < step[1]:
            return False
        return True
