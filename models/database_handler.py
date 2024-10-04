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
            race_data = {}
            race_data["state"] = state
            race_data["year"] = year
            race_data.update({k: v for k, v in doc.items() if k not in ["runs", "form", "splitTimes"]})

            runs = doc.get("runs", [])
            form = doc.get("form", [])
            split_times = doc.get("splitTimes", [])

            form_map = {f.get("runId"): f for f in form}
            split_times_map = {st.get("runId"): st for st in split_times}

            # Process runs
            for run in runs:
                combined_data = race_data.copy()
                combined_data.update({f"run_{k}": v for k, v in run.items()})
                run_id = run.get("runId")

                if run_id in form_map:
                    excluded_keys = ["formSummary", "formStatistics", "form"]
                    form_data = form_map[run_id]
                    combined_data["run_weightInKg"] = (
                        next(
                            (form_item["weightInKg"] for form_item in form_data["form"] if form_item["weightInKg"] is not None),
                            combined_data["run_weightInKg"],
                        )
                        if combined_data["run_weightInKg"] is None
                        else combined_data["run_weightInKg"]
                    )
                    combined_data.update({f"form_{k}": v for k, v in form_data.items() if k not in excluded_keys})
                    combined_data.update({f"form_{k}": v for k, v in form_data.get("formSummary").items()})
                    combined_data.update(Utils.flatten_dict(form_data.get("formStatistics"), "form_formStatistics"))

                if run_id in split_times_map:
                    split_times_data = split_times_map[run_id]
                    combined_data.update({f"split_{k}": v for k, v in split_times_data.items()})

                races.append(combined_data)

        return races

    def load(self, states, years, limit_per_collection=None):
        all_races = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_params = {
                executor.submit(self.get_runs_from_collection, f"history_{state.lower()}_{year}", state, year, limit_per_collection): (
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

        return pd.DataFrame(all_races)
