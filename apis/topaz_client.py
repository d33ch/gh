from topaz import TopazAPI
import pandas as pd


class TopazClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.topaz_api = TopazAPI(api_key)
        self.csv_extension = "csv"
        self.txt_extension = "txt"
        self.json_extension = "json"

    def meetings(self, from_date, to_date, code):
        filename = f"get_meetings_{from_date}_{to_date}_vic.{self.csv_extension}"
        meetings = self.topaz_api.get_meetings(from_date, to_date, code)
        meetings.to_csv(filename, index=False)

    def meeting_races(self, meeting_id):
        filename = f"get_meeting_races_{meeting_id}.{self.json_extension}"
        races = self.topaz_api.get_meeting_races(meeting_id)
        self.list_to_txt(filename, races)

    def meeting_results(self, meeting_id):
        filename = f"get_meeting_results_{meeting_id}.{self.json_extension}"
        results = self.topaz_api.get_meeting_results(meeting_id)
        self.dict_to_txt(filename, results)

    def meeting(self, meeting_id):
        filename = f"get_meeting_{meeting_id}"
        meeting = self.topaz_api.get_meeting_details(meeting_id, "full")
        json = meeting.to_json(orient="records")
        self.to_json_file(filename, json)

    def meeting_form(self, meeting_id):
        filename = f"get_meeting_form_{meeting_id}.{self.json_extension}"
        form = self.topaz_api.get_meeting_form(meeting_id)
        self.dict_to_txt(filename, form)

    def meeting_field(self, meeting_id):
        filename = f"get_meeting_field_{meeting_id}.{self.json_extension}"
        field = self.topaz_api.get_meeting_field(meeting_id)
        self.list_to_txt(filename, field)

    def races(self, from_date, to_date, code):
        filename = f"get_races_{from_date}_{to_date}_vic.{self.csv_extension}"

        races = self.topaz_api.get_races(from_date=from_date, to_date=to_date, owning_authority_code=code)
        races = races.drop(self.column_races_to_drop, axis=1)
        races.to_csv(filename, index=False)

    def race_field(self, race_id):
        filename = f"get_race_field_{race_id}.{self.json_extension}"
        field = self.topaz_api.get_race_field(race_id)
        self.dict_to_txt(filename, field)

    def race_runs(self, race_id):
        filename = f"get_race_runs_{race_id}.{self.csv_extension}"
        runs = self.topaz_api.get_race_runs(race_id)
        runs.to_csv(filename, index=False)

    def race_runs_form(self, race_id):
        filename = f"get_race_runs_form_{race_id}.{self.csv_extension}"
        form = self.topaz_api.get_race_runs_form(race_id)
        self.list_to_txt(filename, form)

    def race_result(self, race_id):
        filename = f"get_race_result_{race_id}.{self.json_extension}"
        result = self.topaz_api.get_race_result(race_id)
        self.dict_to_txt(filename, result)

    def to_json_file(self, filename: str, data: str):
        with open(f"{filename}.{self.json_extension}", "w") as file:
            file.write(data)

    def dict_to_txt(self, filename, data):
        with open(filename, "w") as file:
            file.write("{")
            for key, value in data.items():
                file.write(f'"{key}": "{value}"\n')
                file.write(",")
            file.write("}")

    def list_to_txt(self, filename, data):
        with open(filename, "w") as file:
            file.write("[")
            for item in data:
                file.write(f"{item}\n")
                file.write(",")
            file.write("]")

    def get_track_codes(self):
        filename = f"get_track_codes.{self.json_extension}"
        track_codes = self.topaz_api.get_track_codes()
        track_codes_json = track_codes.to_json(orient="records")
        self.to_json_file(filename, track_codes_json)
