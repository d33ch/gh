from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pymongo import MongoClient
from utils import Utils


class DatabaseHandler:
    def __init__(self, uri, db_name):
        self.db = MongoClient(uri)[db_name]

    def get_runs_from_collection(self, collection_name, state, year, limit=None):
        collection = self.db[collection_name]
        cursor = collection.find({})
        if limit:
            cursor = cursor.limit(limit)
        races = []
        for doc in cursor:
            races.append(doc)
        return races

    def load(self, states, years, limit_per_collection=None):
        all_races = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_params = {
                executor.submit(
                    self.get_runs_from_collection, f"history_{state.lower()}_{year}", state, year, limit_per_collection
                ): (
                    state,
                    year,
                )
                for state in states
                for year in years
            }
            for future in as_completed(future_to_params):
                state, year = future_to_params[future]
                try:
                    races = future.result()
                    all_races.extend(races)
                    print(f"Retrieved {len(races)} races from {state} {year}")
                except Exception as exc:
                    print(f"{state} {year} generated an exception: {exc}")
        return all_races
