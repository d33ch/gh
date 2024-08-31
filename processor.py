import csv
from typing import List
from base.base_data_parser import BaseDataParser
from base.base_loader import BaseLoader
from base.base_market_filter import BaseMarketFilter
from injector import inject
import betfairlightweight
from betfair_data import bflw, PriceSize
from betfairlightweight import StreamListener
from unittest.mock import patch
from models.ladder_position import LadderPosition
from models.price_ladder import PriceLadder
from models.processor_config import ProcessorConfig
from models.runner_state import RunnerState


class Processor:
    filter: BaseMarketFilter
    loader: BaseLoader
    parser: BaseDataParser
    trading = betfairlightweight.APIClient(
        username="username", password="password", app_key="app_key"
    )
    listener = StreamListener(max_latency=None)

    @inject
    def __init__(
        self, loader: BaseLoader, parser: BaseDataParser, filter: BaseMarketFilter
    ):
        self.filter = filter
        self.loader = loader
        self.parser = parser

    def fill_ladder(ladder: List[PriceSize], depth=5) -> PriceLadder:
        if len(ladder) == 0:
            raise Exception("More than one market book in a single market")
        return PriceLadder(
            [LadderPosition(position.price, position.size)] for position in ladder
        )

    def fill_runners(self, market_book: bflw.MarketBook) -> List[RunnerState]:
        runners = List[RunnerState]
        for runner in market_book.runners:
            try:
                traded_volume = sum([traded.size for traded in runner.ex.traded_volume])
                wap = (
                    sum(
                        [position.size * position.price]
                        for position in runner.ex.traded_volume
                    )
                    / traded_volume
                    if traded_volume > 0
                    else 0
                )
                back_ladder = self.fill_ladder(runner.ex.available_to_back)
                lay_ladder = self.fill_ladder(runner.ex.available_to_lay)
                status = runner.status
                # adjustment_factor = runner.adjustment_factor
                spn = runner.sp.near_price
                spf = runner.sp.far_price
                runner = RunnerState(
                    id=runner.selection_id,
                    traded_volume=traded_volume,
                    back_ladder=back_ladder,
                    lay_ladder=lay_ladder,
                    ltp=runner.last_price_traded,
                    sp_near=spn,
                    sp_far=spf,
                    wap=wap,
                    status=runner.status,
                )
                runners.append(runner)
            except:
                print(runner)
        return runners

    def process(self, file, config: ProcessorConfig):
        file_paths = self.loader.load(file)
        with open("./out.txt", "w") as output:
            for file_path in file_paths:
                stream = self.trading.streaming.create_historical_generator_stream(
                    file_path=file_path,
                    listener=self.listener,
                )
                with patch("builtins.open", lambda f, _: f):
                    generator = stream.get_generator()
                    last_consumed_time = None
                    market_id = None
                    for market_books in generator():
                        if self.filter.filter(market_books[0]) == False:
                            break
                        if len(market_books) > 1:
                            raise Exception(
                                "More than one market book in a single market"
                            )
                        market_book = market_books[0]
                        if market_book.market_id != market_id:
                            market_id = market_book.market_id
                            last_consumed_time = market_book.publish_time

                        seconds_to_start = (
                            market_book.market_definition.market_time
                            - market_book.publish_time
                        ).total_seconds()
                        wait = config.get(seconds_to_start)
                        if seconds_to_start > wait[0]:
                            continue
                        delta_time_seconds = (
                            market_book.publish_time - last_consumed_time
                        ).total_seconds()
                        if delta_time_seconds < wait[1]:
                            continue
                        last_consumed_time = market_book.publish_time

                        output.write(
                            f"{market_book.market_definition.name}-{market_book.market_definition.venue} e: {market_book.market_definition.event_id} m: {market_book.market_id} {last_consumed_time}\n"
                        )
