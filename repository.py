from models.market import Market


class Repository:
    def __init__(self, collection):
        self.collection = collection

    def add(self, data) -> str:
        result = self.collection.insert_one(data)
        return str(result.inserted_id)
