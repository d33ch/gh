import time
from datetime import datetime, timedelta
import json
import os
import sys

import requests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from apis.topaz_client import TopazClient
from repository_factory import RepositoryFactory


def get_races_for_day(day, code, log):
    retry_count = 10
    sleep_time = 30
    while retry_count >= 0:
        try:
            races_str = client.get_races(day, day, code)
            if races_str == None:
                print(f"No races found for {day}-{code}", file=log)
            return races_str
        except requests.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retry_count -= 1
                if retry_count >= 0:
                    print(f"Rate limit. Retrying in {sleep_time * 5} seconds.", file=log)
                    time.sleep(sleep_time * 5)
                else:
                    print(f"Retry count limit reached for {day}-{code}", file=log)
            else:
                print(f"Error fetching races for {day} {code}: {http_err.response.status_code}", file=log)
                retry_count -= 5
                if retry_count >= 0:
                    print(f"Retrying in {sleep_time / 6} seconds.", file=log)
                    time.sleep(sleep_time / 6)
                else:
                    print(f"Retry count limit reached for {day}-{code}", file=log)
        except Exception as ex:
            print(ex)
            retry_count -= 3
            if retry_count >= 0:
                print(f"Retrying in {sleep_time} seconds.", file=log)
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for {day}-{code}", file=log)
    return None


def get_race_form(race_id, log):
    sleep_time = 30
    retry_count = 10
    print(f"Downloading runners form for race with id: {race_id}", file=log)
    while retry_count >= 0:
        try:
            form_str = client.get_race_runs_form(race_id)
            if form_str == None:
                print(f"No race found with id: {race_id}", file=log)
                break
            return json.loads(form_str)
        except requests.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retry_count -= 1
                if retry_count >= 0:
                    print(f"Rate limit. Retrying in {sleep_time * 5} seconds.", file=log)
                    time.sleep(sleep_time * 5)
                else:
                    print(f"Retry count limit reached for race with id: {race_id}", file=log)
            else:
                print(f"Error fetching race results: {http_err.response.status_code}", file=log)
                retry_count -= 5
                if retry_count >= 0:
                    print(f"Retrying in {sleep_time / 6} seconds.", file=log)
                    time.sleep(sleep_time / 6)
                else:
                    print(f"Retry count limit reached for race with id: {race_id}", file=log)
        except Exception as ex:
            print(ex, file=log)
            retry_count -= 3
            if retry_count >= 0:
                print(f"Retrying in {sleep_time} seconds.", file=log)
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for race with id: {race_id}", file=log)
    return None


def get_race_result(race_id, log):
    sleep_time = 30
    retry_count = 10
    print(f"Downloading result for race with id: {race_id}", file=log)
    while retry_count >= 0:
        try:
            result_str = client.get_race_result(race_id)
            if result_str == None:
                break
            return json.loads(result_str)
        except requests.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retry_count -= 1
                if retry_count >= 0:
                    print(f"Rate limit. Retrying in {sleep_time * 5} seconds.", file=log)
                    time.sleep(sleep_time * 5)
                else:
                    print(f"Retry count limit reached for race with id {race_id}", file=log)
            else:
                print(f"Error fetching race results: {http_err.response.status_code}", file=log)
                retry_count -= 5
                if retry_count >= 0:
                    print(f"Retrying in {sleep_time / 6} seconds.", file=log)
                    time.sleep(sleep_time / 6)
                else:
                    print(f"Retry count limit reached for race with id {race_id}", file=log)
        except Exception as ex:
            print(ex, file=log)
            retry_count -= 3
            if retry_count >= 0:
                print(f"Retrying in {sleep_time} seconds.", file=log)
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for race with id: {race_id}", file=log)
    return None


def download_historic_data(start_date, end_date, code):
    with open(output_log, "a") as log:
        print(f"Download data from {start_date} to {end_date}", file=log)
        current_date = start_date
        while current_date <= end_date:
            day = current_date.strftime("%Y-%m-%d")
            current_date += timedelta(days=1)
            print(f"Downloading for {day} {code}", file=log)
            races_str = get_races_for_day(day, code, log)
            if races_str == None:
                continue
            race_ids = [race["raceId"] for race in json.loads(races_str)]
            for race_id in race_ids:
                result_json = get_race_result(race_id, log)
                if result_json == None:
                    continue
                result_json["form"] = get_race_form(race_id, log)
                history_repository.insert_from_json(result_json)


factory = RepositoryFactory()
client = TopazClient("")

codes = ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA", "NZ"]
years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
if len(sys.argv) == 3:
    if sys.argv[1] not in codes:
        raise ValueError(f"{sys.argv[1]} is not a valid code. Use {codes}")
    if sys.argv[2] not in years:
        raise ValueError(f"{sys.argv[2]} is not a valid year. Use {years}")
else:
    raise ValueError(f"Please provide authority code and year. Available codes are {codes}. Available years are {years}")

code = sys.argv[1]
year = sys.argv[2]
history_repository = factory.create_repository(f"history_{code.lower()}_{year}")
start_date = datetime.strptime(f"01-01-{year}", "%d-%m-%Y").date()
end_date = datetime.strptime(f"31-12-{year}", "%d-%m-%Y").date()
output_log = f"outputs/output_{code.lower()}_{year}.log"

print(f"Download historic data for {code} {year}")
download_historic_data(start_date, end_date, code)
