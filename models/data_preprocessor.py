import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from fastai.tabular.core import add_datepart
from utils import Utils


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def flatten_race_info(self, race_doc):
        race_info = {
            k: v
            for k, v in race_doc.items()
            if k
            not in [
                "exoticBetType",
                "positionInRunning",
                "resultsSummary",
                "tabTipRunners",
                "splitTimes",
                "dividends",
                "runs",
                "form",
            ]
        }
        race_info["_id"] = str(race_info["_id"])
        return race_info

    def process_runs(self, runs, race_id):
        processed_runs = []
        for run in runs:
            run_info = run.copy()
            run_info["raceId"] = race_id
            processed_runs.append(run_info)
        return processed_runs

    def process_form(self, form, race_id):
        processed_form = []
        for dog_form in form:
            # dog_id = dog_form["dogId"]
            for historical_run in dog_form["form"]:
                historical_run["currentRaceId"] = race_id
                # historical_run["dogId"] = dog_id
                processed_form.append(historical_run)
        return processed_form

    def process_race_document(self, race):
        race_info = self.flatten_race_info(race)
        race_id = race_info["raceId"]
        race_df = pd.DataFrame([race_info])
        runs_df = pd.DataFrame(self.process_runs(race["runs"], race_id))
        form_df = pd.DataFrame(self.process_form(race["form"], race_id))
        return {"race_df": race_df, "runs_df": runs_df, "form_df": form_df}

    def transform(self, races):
        races_df = []
        for race in races:
            races_df.append(self.process_race_document(race))
        return races_df

        X = pd.DataFrame(races)

        # df["run_pir"] = df["run_pir"].apply(Utils.split_and_encode_pir)
        # df["run_pir"] = df["run_pir"].apply(lambda x: Utils.encode_missing_value(x, -1))

        X["run_last5"] = X["run_last5"].apply(Utils.split_and_encode_last_results)
        for i in range(5):
            X[f"run_last_race_{i+1}"] = X["run_last5"].apply(lambda x: x[i] if i < len(x) else -1)

        X["run_careerPrizeMoney"] = X["run_careerPrizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))
        X["run_rating"] = X["run_rating"].apply(lambda x: Utils.encode_missing_value(x, -1))
        X["run_incomingGrade"] = X["run_incomingGrade"].apply(Utils.encode_grade_feature)
        X["run_boxNumber"] = X["run_boxNumber"].apply(lambda x: Utils.encode_missing_value(x, -1))

        X[["raceStartHour", "raceStartMinute"]] = X["startTime"].apply(lambda x: pd.Series(Utils.time_to_components(x)))
        X["run_weightInKg"] = X["run_weightInKg"].apply(lambda x: Utils.encode_missing_value(x, -1))

        X["run_trainerPostCode"] = X["run_trainerPostCode"].apply(lambda x: Utils.encode_missing_value(x, -1))

        # df["run_startingPrice"] = df["run_startingPrice"].apply(lambda x: Utils.encode_missing_value(x, 0))
        # df["run_place"] = df["run_place"].apply(lambda x: Utils.encode_missing_value(x, 0))
        # df["run_prizeMoney"] = df["run_prizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))
        # df["run_resultTime"] = df["run_resultTime"].apply(lambda x: Utils.encode_missing_value(x, -1))
        # df["run_resultMargin"] = df["run_resultMargin"].apply(lambda x: Utils.encode_missing_value(x, -1))

        X["form_bestFinishTrackAndDistance"] = X["form_bestFinishTrackAndDistance"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_averageSpeed"] = X["form_averageSpeed"].apply(lambda x: Utils.encode_missing_value(x, -1))
        X["form_bestFinishTrackDistance"] = X["form_bestFinishTrackDistance"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_bestFirstSplitTrackDistance"] = X["form_bestFirstSplitTrackDistance"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_avgFirstSplitTrackDistance"] = X["form_avgFirstSplitTrackDistance"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_bestFirstSplitTrackDistanceBox"] = X["form_bestFirstSplitTrackDistanceBox"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_careerPrizeMoney"] = X["form_careerPrizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))

        X = X.drop("startTime", axis=1)

        add_datepart(X, "raceStart", drop=True)
        # add_datepart(df, "run_dateWhelped", drop=True)
        X["run_age"] = Utils.calculate_dog_age(X["raceStart", X["run_dateWhelped"]])

        categorical_columns = [
            "raceTypeCode",
            "owningAuthorityCode",
            "run_trackCode",
            "run_gradedTo",
            "run_colourCode",
            "run_sex",
            "run_ownerState",
            "run_trainerName",
            "run_trainerState",
            "run_trainerSuburb",
            # "run_pir",
        ]

        for col in categorical_columns:
            X[col] = self.le.fit_transform(X[col])

        # df["run_scratched"] = df["run_scratched"].astype(int)
        # df["run_isLateScratching"] = df["run_isLateScratching"].astype(int)
        # df["run_resultMarginLengths"] = df["run_resultMarginLengths"].apply(Utils.extract_numeric)

        # df.loc[df["run_place"] == 1, "run_resultMargin"] = 0
        # df.loc[df["run_place"] == 1, "run_resultMarginLengths"] = 0

        X["run_isWinner"] = (X["run_place"] == 1).astype(int)

        return X
