from pymongo import MongoClient

from repository import Repository

client = MongoClient("mongodb://localhost:27017/")
db = client["gh"]


class RepositoryFactory:
    @staticmethod
    def create_repository(collection_name: str) -> Repository:
        return Repository(db[collection_name])
