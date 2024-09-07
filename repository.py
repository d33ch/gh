from bson import ObjectId
from dtos.market import Market
from dtos.market_state import MarketState
from dtos.runner import Runner


class Repository:
    def __init__(self, collection):
        self.collection = collection

    def add(self, data) -> str:
        data_dict = data.to_dict()
        result = self.collection.insert_one(data_dict)
        return str(result.inserted_id)
