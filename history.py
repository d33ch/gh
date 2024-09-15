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


def get_races_for_day(day, code):
    retry_count = 10
    sleep_time = 30
    try:
        while retry_count > 0:
            races_str = client.get_races(day, day, code)
            if races_str == None:
                print(f"No races found for {day}-{code}")
            return races_str
    except requests.HTTPError as http_err:
        if http_err.response.status_code == 429:
            retry_count -= 1
            if retry_count > 0:
                print(f"Rate limit. Retrying in {sleep_time * 5} seconds.")
                time.sleep(sleep_time * 5)
            else:
                print(f"Retry count limit reached for {day}-{code}")
        else:
            print(f"Error fetching races for {code}: {http_err.response.status_code}")
            retry_count -= 1
            if retry_count > 0:
                print(f"Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for {day}-{code}")
    return None


def get_race_form(race_id):
    sleep_time = 30
    retry_count = 10
    print(f"Downloading runners form for race with id: {race_id}")
    try:
        while retry_count > 0:
            form_str = client.get_race_runs_form(race_id)
            if form_str == None:
                print(f"No race found with id: {race_id}")
                break
            return json.loads(form_str)
    except requests.HTTPError as http_err:
        if http_err.response.status_code == 429:
            retry_count -= 1
            if retry_count > 0:
                print(f"Rate limit. Retrying in {sleep_time * 5} seconds.")
                time.sleep(sleep_time * 5)
            else:
                print(f"Retry count limit reached for race with id: {race_id}")
        else:
            print(f"Error fetching race results: {http_err.response.status_code}")
            retry_count -= 1
            if retry_count > 0:
                print(f"Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for race with id: {race_id}")
    return None


def get_race_result(race_id):
    sleep_time = 30
    retry_count = 10
    print(f"Downloading result for race with id: {race_id}")
    try:
        while retry_count > 0:
            result_str = client.get_race_result(race_id)
            if result_str == None:
                retry_count -= 1
                continue
            return json.loads(result_str)
    except requests.HTTPError as http_err:
        if http_err.response.status_code == 429:
            retry_count -= 1
            if retry_count > 0:
                print(f"Rate limit. Retrying in {sleep_time * 5} seconds.")
                time.sleep(sleep_time * 5)
            else:
                print(f"Retry count limit reached for race with id {race_id}")
        else:
            print(f"Error fetching race results: {http_err.response.status_code}")
            retry_count -= 1
            if retry_count > 0:
                print(f"Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                print(f"Retry count limit reached for race with id {race_id}")
    return None


def download_historic_data(start_date, end_date):
    print(f"Download data from {start_date} to {end_date}")
    codes = ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA", "NZ"]
    current_date = start_date
    while current_date <= end_date:
        day = current_date.strftime("%Y-%m-%d")
        for code in codes[:-7]:
            print(f"Downloading for {day} {code}")
            races_str = get_races_for_day(day, code)
            if races_str == None:
                continue
            race_ids = [race["raceId"] for race in json.loads(races_str)]
            for race_id in race_ids:
                result_json = get_race_result(race_id)
                result_json["form"] = get_race_form(race_id)
                history_repository.insert_from_json(result_json)
        current_date += timedelta(days=1)


factory = RepositoryFactory()
history_repository = factory.create_repository("history")
client = TopazClient("")
start_date = datetime.strptime("01-10-2016", "%d-%m-%Y").date()
end_date = datetime.strptime("31-12-2020", "%d-%m-%Y").date()

download_historic_data(start_date, end_date)
