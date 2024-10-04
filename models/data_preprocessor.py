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

    def transform(self, X):
        df = X.copy()

        df["run_last5"] = df["run_last5"].apply(Utils.split_and_encode_last_results)
        df["run_pir"] = df["run_pir"].apply(Utils.split_and_encode_pir)

        for i in range(5):
            df[f"run_last_race_{i+1}"] = df["run_last5"].apply(lambda x: x[i] if i < len(x) else -1)

        df["run_incomingGrade"] = df["run_incomingGrade"].apply(Utils.encode_grade_feature)
        df["run_outgoingGrade"] = df["run_outgoingGrade"].apply(Utils.encode_grade_feature)
        df["run_boxNumber"] = df["run_boxNumber"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["run_startingPrice"] = df["run_startingPrice"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df["run_place"] = df["run_place"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df["run_prizeMoney"] = df["run_prizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df["run_resultTime"] = df["run_resultTime"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["run_resultMargin"] = df["run_resultMargin"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["run_boxNumber"] = df["run_boxNumber"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["run_careerPrizeMoney"] = df["run_careerPrizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df["run_rating"] = df["run_rating"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df["run_weightInKg"] = df["run_weightInKg"].apply(lambda x: Utils.encode_missing_value(x, 0))
        df[["raceStartHour", "raceStartMinute"]] = df["startTime"].apply(lambda x: pd.Series(Utils.time_to_components(x)))
        df["run_trainerPostCode"] = df["run_trainerPostCode"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["run_pir"] = df["run_pir"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_bestFinishTrackAndDistance"] = df["form_bestFinishTrackAndDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_averageSpeed"] = df["form_averageSpeed"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_bestFinishTrackDistance"] = df["form_bestFinishTrackDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_bestFirstSplitTrackDistance"] = df["form_bestFirstSplitTrackDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_avgFirstSplitTrackDistance"] = df["form_avgFirstSplitTrackDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        df["form_bestFirstSplitTrackDistanceBox"] = df["form_bestFirstSplitTrackDistanceBox"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        df["form_careerPrizeMoney"] = df["form_careerPrizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))

        df = df.drop("startTime", axis=1)

        add_datepart(df, "raceStart", drop=True)
        add_datepart(df, "run_dateWhelped", drop=True)

        # Encode categorical variables
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
            "run_pir",
        ]

        for col in categorical_columns:
            df[col] = self.le.fit_transform(df[col])

        df["run_scratched"] = df["run_scratched"].astype(int)
        df["run_isLateScratching"] = df["run_isLateScratching"].astype(int)
        df["run_resultMarginLengths"] = df["run_resultMarginLengths"].apply(Utils.extract_numeric)

        df.loc[df["run_place"] == 1, "run_resultMargin"] = 0
        df.loc[df["run_place"] == 1, "run_resultMarginLengths"] = 0

        df["run_isWinner"] = (df["run_place"] == 1).astype(int)

        return df
