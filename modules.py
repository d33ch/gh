from injector import Module, provider

from base.base_data_parser import BaseDataParser

# from base_data_utils import BaseDataUtils
from base.base_loader import BaseLoader

from base.base_market_filter import BaseMarketFilter

# from data_utils import DataUtils
from market_filter import MarketFilter
from processor import Processor
from tar_loader import TarLoader
from data_parser import DataParser


class ProcessorModule(Module):
    def configure(self, binder):
        binder.bind(BaseLoader, to=TarLoader)
        binder.bind(BaseDataParser, to=DataParser)
        binder.bind(BaseMarketFilter, to=MarketFilter)
        # binder.bind(BaseDataUtils, to=DataUtils)
        binder.bind(Processor)
