from typing import List
from repository_factory import RepositoryFactory
from base.base_market_parser import BaseMarketParser
from base.base_loader import BaseLoader
from base.base_market_filter import BaseMarketFilter
from injector import inject
import betfairlightweight
from betfairlightweight import StreamListener
from unittest.mock import patch
from models.market import Market
from models.processor_config import ProcessorConfig
from repository import Repository


class Processor:
    filter: BaseMarketFilter
    loader: BaseLoader
    parser: BaseMarketParser
    market_repository: Repository
    trading = betfairlightweight.APIClient(username="username", password="password", app_key="app_key")
    listener = StreamListener(max_latency=None)

    @inject
    def __init__(self, loader: BaseLoader, parser: BaseMarketParser, filter: BaseMarketFilter, repository_factory: RepositoryFactory):
        self.filter = filter
        self.loader = loader
        self.parser = parser
        self.market_repository = repository_factory.create_repository("market")

    def process(self, file, config: ProcessorConfig):
        file_paths = self.loader.load(file)
        markets: List[Market] = []
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
                    if len(market_books) > 1:
                        raise Exception("More than one market book in a single market")

                    if not self.filter.filter_market(market_books[0]):
                        break

                    market_book = market_books[0]
                    if market_book.market_id != market_id:
                        market_id = market_book.market_id
                        last_consumed_time = market_book.publish_time
                        market = self.parser.parse_market_info(market_book)
                        # result = self.market_repository.add(market)
                        markets.append(market)

                    if not self.filter.filter_time(market_book.market_definition.market_time, market_book.publish_time, last_consumed_time, config.steps):
                        continue

                    market_state = self.parser.parse_market_state(market_book)
                    market.market_states.append(market_state)
                    last_consumed_time = market_book.publish_time
                print(len(markets))
