
import bz2
from base.base_data_parser import BaseDataParser


class DataParser(BaseDataParser):
    def __init__(self):
        super().__init__()

    def parse(self, file) -> list:
        markets = []
        with bz2.open(file, 'rt') as f:
            for line in f:
                print(line)
                market_data = line.strip().split(',')
                # market = Market(market_data[0], market_data[1])
                # markets.append(market)
        return markets