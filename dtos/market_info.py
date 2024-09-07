class MarketInfo:
    def __init__(self, market_name: str):
        parts = market_name.split(' ')

        self.market_name = market_name
        self.race_number = parts[0]
        self.race_length = parts[1].split('m')[0]
        self.race_type = parts[2].lower()