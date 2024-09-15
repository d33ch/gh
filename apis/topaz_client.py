import json
from topaz import TopazAPI


class TopazClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.topaz_api = TopazAPI(api_key)

    def get_races(self, from_date, to_date, code):
        try:
            races = self.topaz_api.get_races(from_date=from_date, to_date=to_date, owning_authority_code=code)
            return races.to_json(orient="records")
        except Exception as e:
            if e.response.status_code == 404:
                print(f"Error 404 not found: {from_date} {code}")
                return None
            else:
                print(f"Error: {from_date} {code}")

    def get_race_runs(self, race_id):
        runs = self.topaz_api.get_race_runs(race_id)
        return runs.to_json(orient="records")

    def get_race_runs_form(self, race_id):
        form = self.topaz_api.get_race_runs_form(race_id)
        return json.dumps(form)

    def get_race_result(self, race_id):
        result = self.topaz_api.get_race_result(race_id)
        return json.dumps(result)

    def get_track_codes(self):
        track_codes = self.topaz_api.get_track_codes()
        return track_codes.to_json(orient="records")
