from base.base_data_parser import BaseDataParser
from base.base_loader import BaseLoader
from base.base_market_filter import BaseMarketFilter
from injector import inject
import betfairlightweight
from betfairlightweight import StreamListener
from unittest.mock import patch


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

    def process(self, file):
        file_paths = self.loader.load(file)
        for file_path in file_paths:
            stream = self.trading.streaming.create_historical_generator_stream(
                file_path=file_path,
                listener=self.listener,
            )
            with patch("builtins.open", lambda f, _: f):
                generator = stream.get_generator()
                for market_books in generator():
                    if len(market_books) > 1:
                        raise Exception("More than one market book in a single market")
                    market_book = market_books[0]

                    if self.filter.filter(market_book):
                        print(market_book.market_definition.venue)
                        print(market_book.market_definition.event_id)

                    # if(market_book.bsp_reconciled):
                    # sp = market_book.runners[0].sp
                    # lpt = market_book.runners[0].last_price_traded
                    # print(market_book.json())
                    # break
                    # print(
                    # f"{market_book.market_id} - {market_book.status} - {market_book.total_matched} - {market_book.bet_delay}"
                    # )
                    # print(f"{sp.near_price}-{sp.far_price}-{sp.actual_sp}-{lpt}")
                    # print(dir(market_books[0]))
                    # for runner in market_books[0].runners:
                    # if len(runner.ex.available_to_back) > 0:
                    # to_back = runner.ex.available_to_back
                    # print(
                    # f"{to_back[0].size} {to_back[0].price} {runner.selection_id} {market_books[0].total_matched}"
                    # )
                    # break
                    # self.parser.parse(d)
