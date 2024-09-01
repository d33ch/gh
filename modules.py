from injector import Module, provider

from base.base_market_parser import BaseMarketParser

# from base_data_utils import BaseDataUtils
from base.base_loader import BaseLoader

from base.base_market_filter import BaseMarketFilter

# from data_utils import DataUtils
from market_filter import MarketFilter
from processor import Processor
from tar_loader import TarLoader
from market_parser import MarketParser


class ProcessorModule(Module):
    def configure(self, binder):
        binder.bind(BaseLoader, to=TarLoader)
        binder.bind(BaseMarketParser, to=MarketParser)
        binder.bind(BaseMarketFilter, to=MarketFilter)
        # binder.bind(BaseDataUtils, to=DataUtils)
        binder.bind(Processor)
