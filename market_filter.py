from betfair_data import bflw
from base.base_market_filter import BaseMarketFilter
from models.market_info import MarketInfo


class MarketFilter(BaseMarketFilter):
    def filter(self, market_book: bflw.MarketBook) -> bool:
        definition = market_book.market_definition
        return (
            definition != None
            and definition.country_code == "AU"
            and definition.market_type == "WIN"
            and (race_type := MarketInfo(definition.name).race_type) != "trot"
            and race_type != "pace"
        )
