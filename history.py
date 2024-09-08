import calendar
from datetime import date, timedelta
import json
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from apis.topaz_client import TopazClient
from repository_factory import RepositoryFactory


def loop_through_calendar(year, month):
    cal = calendar.monthcalendar(year, month)
    first_day = date(year, month, 1)
    for week in cal:
        for day in week:
            if day != 0:
                current_date = first_day + timedelta(days=day - 1)
                current_date = current_date.strftime("%Y-%m-%d")
                download_races_for_day(current_date)


def download_races_for_day(day: str):
    for code in codes:
        print(f"Downloading for {day} {code}")
        races_str = client.races(day, day, code)
        if races_str == None:
            continue
        races = json.loads(races_str)
        print(f"Found {len(races)}")
        for race in races:
            try:
                raceId = race["raceId"]
                result_str = client.race_result(raceId)
                result_json = json.loads(result_str)
                history_repository.add_json(result_json)
            except Exception as ex:
                print(f"ex: {raceId} - {ex}")


codes = ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA", "NZ"]
factory = RepositoryFactory()
history_repository = factory.create_repository("history")
client = TopazClient("")

from repository_factory import RepositoryFactory

loop_through_calendar(2016, 9)
