import time
from pymongo import MongoClient
from datetime import timedelta
import json
from difflib import SequenceMatcher

YEAR = "2017"
STATES = ["NSW", "VIC", "QLD", "NZ", "NT", "SA", "TAS", "WA"]
client = MongoClient("mongodb://localhost:27017/")
db = client["gh"]

market_collection = db["markets_2017"]

with open("track_mapping.json", "r") as f:
    track_mapping = json.load(f)


def similarity_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_historic_track_name(betfair_track_name):
    track_info = track_mapping.get(betfair_track_name)
    if track_info == None:
        return []
    if isinstance(track_info, list):
        return track_info
    else:
        return [track_info]


def get_start_time(start_time):
    start_time_str = start_time.isoformat()[:10]
    min_start_time = start_time - timedelta(days=1)
    max_start_time = start_time + timedelta(days=1)
    min_start_time_str = min_start_time.isoformat()[:10]
    max_start_time_str = max_start_time.isoformat()[:10]

    return [min_start_time_str, start_time_str, max_start_time_str]


def normalize_market_runner(runner):
    return runner.split(". ", 1)[1].replace(" ", "").replace(".", "").upper()


def normalize_historic_runner(runner):
    return runner.replace("'", "").replace(" ", "").replace(".", "").upper()


def map_runner(betfair_runner, historic_runners):
    runner_found = next(
        (runner for runner in historic_runners if normalize_historic_runner(runner["dogName"]) == betfair_runner), None
    )
    return map_runner_using_sequencer(betfair_runner, historic_runners) if not runner_found else (1, runner_found)


def map_runner_using_sequencer(betfair_runner, historic_runners):
    best_ratio = 0
    best_match = None
    for runner in historic_runners:
        normalized_historic_runner = normalize_historic_runner(runner["dogName"])
        ratio = similarity_ratio(betfair_runner, normalized_historic_runner)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = runner
    return (best_ratio, best_match)


def find_matching_historic(market_doc):
    betfair_runners = [(normalize_market_runner(runner["name"]), runner["id"]) for runner in market_doc["runners"]]
    betfair_track_name = market_doc["venue"]
    betfair_race_times = get_start_time(market_doc["suspend_time"])
    historic_track_infos = get_historic_track_name(betfair_track_name)

    for historic_track_info in historic_track_infos:
        historic_track_name = historic_track_info.get("trackName")
        historic_track_state = historic_track_info.get("owningAuthorityCode")
        historic_collection = db[f"history_{historic_track_state.lower()}_{YEAR}"]

        for race_time in betfair_race_times:
            historic_matches = historic_collection.find(
                {"raceStart": {"$regex": f"^{race_time}"}, "runs.track": historic_track_name}
            )

            for historic_match in historic_matches:
                # unmatched_runners = []
                runners = []
                if f"R{historic_match['raceNumber']} " in market_doc["name"]:
                    for betfair_runner in betfair_runners:
                        runner_found = map_runner(betfair_runner[0], historic_match["runs"])

                        if runner_found[0] > 0.83:
                            runners.append(
                                {
                                    "betfair_id": betfair_runner[1],
                                    "historic_id": runner_found[1]["runId"],
                                    "dog_id": runner_found[1]["dogId"],
                                    "ratio": runner_found[0],
                                    "betfair_name": betfair_runner,
                                    "historic_name": normalize_historic_runner(runner_found[1]["dogName"]),
                                }
                            )
                        else:
                            if runner_found[0] > 0.8:
                                print(f"{runner_found[0]}-{betfair_runner}-{runner_found[1]['dogName']}")

                    if len(betfair_runners) == len(runners):
                        return (historic_match, runners)

    return (None, None)


def map_datasets():
    mapped_count = 0
    unmapped_records = []
    mapped_records = []

    start_time = time.time()

    for market_doc in market_collection.find():
        if market_doc["venue"] is None:
            continue

        (matching_historic, runners) = find_matching_historic(market_doc)

        if matching_historic:
            mapped_count += 1

            mapped_records.append(
                {
                    "history_id": matching_historic["_id"],
                    "owningAuthorityCode": matching_historic["owningAuthorityCode"],
                    "market_id": market_doc["_id"],
                    "name": market_doc["name"],
                    "runners": runners,
                }
            )
        else:
            record = {
                "id": market_doc["_id"],
                "venue": market_doc["venue"],
                "name": market_doc["name"],
                "time": market_doc["suspend_time"],
                "runners": [run["name"] for run in market_doc["runners"]],
            }
            unmapped_records.append(record)
            print(f"no match {market_doc['_id']}")

    print(f"Mapped {mapped_count} records")
    print(f"Failed to map {len(unmapped_records)} records")

    with open("unmapped_records.json", "w") as f:
        json.dump(unmapped_records, f, default=str, indent=2)

    with open("mapped_records.json", "w") as f:
        json.dump(mapped_records, f, default=str, indent=2)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    map_datasets()
