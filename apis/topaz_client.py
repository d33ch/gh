import json
from topaz import TopazAPI


class TopazClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.topaz_api = TopazAPI(api_key)

    def races(self, from_date, to_date, code):
        races = self.topaz_api.get_races(from_date=from_date, to_date=to_date, owning_authority_code=code)
        return races.to_json(orient="records")

    def race_runs(self, race_id):
        runs = self.topaz_api.get_race_runs(race_id)
        return runs.to_json(orient="records")

    def race_runs_form(self, race_id):
        form = self.topaz_api.get_race_runs_form(race_id)
        return json.dumps(form)

    def race_result(self, race_id):
        result = self.topaz_api.get_race_result(race_id)
        return json.dumps(result)

    def get_track_codes(self):
        track_codes = self.topaz_api.get_track_codes()
        return track_codes.to_json(orient="records")
