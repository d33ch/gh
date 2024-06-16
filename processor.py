from base.base_data_parser import BaseDataParser
from base.base_loader import BaseLoader
from injector import inject
import betfairlightweight
from betfairlightweight import StreamListener
from unittest.mock import patch


class Processor():
    loader: BaseLoader
    parser: BaseDataParser
    trading = betfairlightweight.APIClient(
        username="username",
        password="password",
        app_key="app_key"
    )
    listener = StreamListener(max_latency=None)

    @inject
    def __init__(self, loader: BaseLoader, parser: BaseDataParser):
        self.loader = loader
        self.parser = parser

    def process(self, file):
        for data in self.loader.load(file):
            # with data as d:
            # print(d.read())
            stream = self.trading.streaming.create_historical_generator_stream(
                file_path=data,
                listener=self.listener,
            )

            with patch("builtins.open", lambda f, _: f):
                gen = stream.get_generator()

                marketID = None
                tradeVols = None
                time = None

                for market_books in gen():
                    print(market_books[0].market_id)
                    # self.parser.parse(d)
                    break

        # for d in data[:2]:
        #     )
