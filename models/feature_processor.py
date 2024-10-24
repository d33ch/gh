from datetime import datetime
import math
from numbers import Number
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from fastai.tabular.core import add_datepart
from features import ALL_FEATURES
from enum import Enum
from scipy import stats


class Direction(Enum):
    EARLY = "early"
    LATE = "late"


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ALL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, races):
        results = []
        for race_data in races:
            race_df = race_data["race_df"]
            runs_df = race_data["runs_df"]
            form_df = race_data["form_df"]
            stats_df = race_data["stats_df"]
            track = runs_df["trackCode"].iloc[0]
            track_distance = runs_df["distance"].iloc[0]
            box_numbers_df = runs_df[["dogId", "boxNumber"]]
            whelped_dates_df = runs_df[["dogId", "dateWhelped"]]
            race_date = pd.to_datetime(race_df.loc[0, "raceStart"])
            auth_code = race_df.loc[0, "owningAuthorityCode"]
            dogs = runs_df["dogId"].values.tolist()
            dog_features = self.create_features_from_form(form_df, box_numbers_df, dogs, track, track_distance, race_date)
            run_features = self.create_features_from_runs(runs_df, whelped_dates_df, race_date)
            result_df = pd.merge(run_features, dog_features, on="dogId", how="left")
            result_df = result_df.merge(stats_df, on="dogId", how="left")
            result_df["auth_code"] = auth_code
            result_df[["raceStartHour", "raceStartMinute"]] = race_df["startTime"].apply(lambda x: pd.Series(self.time_to_components(x))).loc[0]
            result_df = result_df.reset_index(drop=True)
            results.append(result_df)
        X = pd.DataFrame([row for df in results for _, row in df.iterrows()])
        X = add_datepart(X, "meetingDate", drop=True)
        X = X.reset_index(drop=True)
        return X

    def create_features_from_form(self, form_df, box_numbers_df, dogs, track, distance, race_date):
        form_df["pir_with_place"] = form_df["pir"].astype(str).replace("0", "") + form_df["place"].astype("Int64").astype(str).replace("nan", "")
        form_df["meetingDate"] = pd.to_datetime(form_df["meetingDate"])
        form_df["place"] = form_df["place"].apply(self.encode_race_result)
        form_df_sorted = form_df.sort_values(["dogId", "meetingDate"], ascending=[True, False])
        x = form_df_sorted.groupby("dogId")
        pir_features = self.calculate_pir(x["pir_with_place"])
        race_result_features = self.race_result_features(x)
        track_dist_features = self.track_distance_features(x, track, distance)
        box_features = self.box_features(x, box_numbers_df)
        dog_features = self.dog_features(x, race_date)
        features = pd.merge(race_result_features, track_dist_features, on="dogId", how="left")
        features = pd.merge(features, box_features, on="dogId", how="left")
        features = pd.merge(features, dog_features, on="dogId", how="left")
        features = pd.merge(features, pir_features, on="dogId", how="left")
        features = features.reset_index(drop=True)
        present_dogs = features["dogId"].unique()
        missing_dogs = set(dogs) - set(present_dogs)
        missing_features = pd.DataFrame({"dogId": list(missing_dogs), **{col: -1 for col in features.columns if col != "dogId"}})
        final_features = pd.concat([features, missing_features], ignore_index=True)
        final_features = final_features.reset_index(drop=True)
        return final_features

    def create_features_from_runs(self, runs_df, whelped_dates_df, race_date):
        age_features = whelped_dates_df.apply(lambda x: self.calculate_dog_age(x, race_date), axis=1)
        runs_df.loc[runs_df["unplacedCode"].notna(), "place"] = runs_df["unplacedCode"].apply(self.encode_race_result)
        runs_df = runs_df.merge(age_features, on="dogId", how="left")
        return runs_df[
            [
                "dogId",
                "colourCode",
                "trackCode",
                "distance",
                "raceId",
                "meetingDate",
                "raceTypeCode",
                "runId",
                "weightInKg",
                "careerPrizeMoney",
                "incomingGrade",
                "rating",
                "raceNumber",
                "boxNumber",
                "rugNumber",
                "startPrice",
                "place",
                "unplacedCode",
                "prizeMoney",
                "resultTime",
                "sex",
                "ownerId",
                "trainerId",
                "damId",
                "sireId",
            ]
        ]

    def calculate_track_distance_features(self, x, track_code, distance):
        tr_dist_df = x[(x["trackCode"] == track_code) & (x["distance"] == distance) & (~np.isnan(x["resultTime"]))]
        distance_df_filtered = x[(x["distance"] == distance) & (~np.isnan(x["resultTime"]))]
        track_distance_num = len(tr_dist_df)
        distance_num_races = len(distance_df_filtered)
        return pd.Series(
            {
                "track_distance_num_races_last_5": track_distance_num,
                "track_distance_wins_last_5": (tr_dist_df["place"] == 1).sum(),
                "track_distance_win_rate_last_5": round((tr_dist_df["place"] == 1).mean() if track_distance_num > 0 else 0, 2),
                "track_distance_top_3_finishes_last_5": (tr_dist_df["place"] <= 3).sum(),
                "track_distance_top_3_rate_last_5": round((tr_dist_df["place"] <= 3).mean() if track_distance_num > 0 else 0, 2),
                "track_distance_avg_finish_last_5": (round(tr_dist_df["place"].mean(), 2) if track_distance_num > 0 else -1),
                "track_distance_median_finish_last_5": (round(tr_dist_df["place"].median(), 2) if track_distance_num > 0 else -1),
                "track_distance_avg_time_last_5": (round(tr_dist_df["resultTime"].mean(), 2) if track_distance_num > 0 else -1),
                "track_distance_best_time_last_5": (tr_dist_df["resultTime"].min() if track_distance_num > 0 else -1),
                "track_distance_worst_time_last_5": (tr_dist_df["resultTime"].max() if track_distance_num > 0 else -1),
                "distance_num_races_last_5": distance_num_races,
                "distance_wins_last_5": (distance_df_filtered["place"] == 1).sum(),
                "distance_win_rate_last_5": round((distance_df_filtered["place"] == 1).mean() if distance_num_races > 0 else 0, 2),
                "distance_top_3_finishes_last_5": (distance_df_filtered["place"] <= 3).sum(),
                "distance_top_3_rate_last_5": round((distance_df_filtered["place"] <= 3).mean() if distance_num_races > 0 else 0, 2),
                "distance_avg_finish_last_5": (round(distance_df_filtered["place"].mean(), 2) if distance_num_races > 0 else -1),
                "distance_avg_time_last_5": (round(distance_df_filtered["resultTime"].mean(), 2) if distance_num_races > 0 else -1),
                "distance_best_time_last_5": (distance_df_filtered["resultTime"].min() if distance_num_races > 0 else -1),
                "distance_worst_time_last_5": (distance_df_filtered["resultTime"].max() if distance_num_races > 0 else -1),
            }
        )

    def track_distance_features(self, x, track_code, distance):
        track_features = x.apply(lambda x: self.calculate_track_distance_features(x, track_code, distance), include_groups=False)
        return track_features

    def box_features(self, x, box_numbers_df):
        feat = x.apply(
            lambda x: self.calculate_box_features(x, box_numbers_df.loc[box_numbers_df["dogId"] == x.name, "boxNumber"].values[0]),
            include_groups=False,
        )
        return feat

    def dog_features(self, x, race_date):
        feat = pd.DataFrame()
        weight_features = x.apply(self.calculate_dog_features, include_groups=False)
        rating_features = x.apply(self.calculate_rating_features, include_groups=False)
        feat = pd.merge(weight_features, rating_features, on="dogId", how="left")
        feat["days_since_last_race"] = ((race_date - x["meetingDate"].first()).dt.total_seconds() / 86400).fillna(0).astype(int)
        return feat

    def race_result_features(self, x):
        feat = pd.DataFrame()
        feat["performance_trend"] = x["place"].apply(self.calculate_performance_trend)
        race_result_features = x.apply(self.calculate_race_result_features)
        margin_features = x.apply(self.calculate_margin_features, include_groups=False)
        feat = pd.merge(feat, race_result_features, on="dogId", how="left")
        feat = pd.merge(feat, margin_features, on="dogId", how="left")
        # feat = feat.reset_index(drop=True)
        return feat

    def calculate_race_result_features(self, x):
        x_last_5 = x[(~np.isnan(x["place"])) & (~np.isnan(x["resultTime"]))]
        x_last_3 = x.head(3)
        x_last_3 = x_last_3[(~np.isnan(x_last_3["place"])) & (~np.isnan(x_last_3["resultTime"]))]
        x_last_5_num = len(x_last_5)
        x_last_3_num = len(x_last_3)
        return pd.Series(
            {
                "num_races_last_5": x_last_5_num,
                "avg_finish_last_5": round(x_last_5["place"].mean(), 2) if x_last_5_num > 0 else 0,
                "median_finish_last_5": round(x_last_5["place"].median(), 2) if x_last_5_num > 0 else 0,
                "best_finish_last_5": x_last_5["place"].min() if x_last_5_num > 0 else 0,
                "worst_finish_last_5": x_last_5["place"].max() if x_last_5_num > 0 else 0,
                "var_finish_last_5": round(x_last_5["place"].var(), 2) if x_last_5_num > 1 else 0,
                "avg_speed_last_5": round((x_last_5["distance"] / x_last_5["resultTime"]).mean() * 3.6, 2) if x_last_5_num > 0 else 0,
                "max_speed_last_5": round((x_last_5["distance"] / x_last_5["resultTime"]).max() * 3.6, 2) if x_last_5_num > 0 else 0,
                "min_speed_last_5": round((x_last_5["distance"] / x_last_5["resultTime"]).min() * 3.6, 2) if x_last_5_num > 0 else 0,
                "win_rate_last_5": round((x_last_5["place"] == 1).mean(), 2) if x_last_5_num > 0 else 0,
                "top_3_rate_last_5": round((x_last_5["place"] <= 3).mean(), 2) if x_last_5_num > 0 else 0,
                "avg_time_per_100m_last_5": round(((x_last_5["resultTime"] / x_last_5["distance"]) * 100).mean(), 2) if x_last_5_num > 0 else 0,
                "min_time_per_100m_last_5": round(((x_last_5["resultTime"] / x_last_5["distance"]) * 100).min(), 2) if x_last_5_num > 0 else 0,
                "max_time_per_100m_last_5": round(((x_last_5["resultTime"] / x_last_5["distance"]) * 100).max(), 2) if x_last_5_num > 0 else 0,
                "num_races_last_3": x_last_3_num,
                "avg_finish_last_3": round(x_last_3["place"].mean(), 2) if x_last_3_num > 0 else 0,
                "median_finish_last_3": round(x_last_3["place"].median(), 2) if x_last_3_num > 0 else 0,
                "best_finish_last_3": round(x_last_3["place"].min(), 2) if x_last_3_num > 0 else 0,
                "worst_finish_last_3": round(x_last_3["place"].max(), 2) if x_last_3_num > 0 else 0,
                "var_finish_last_3": round(x_last_3["place"].var(), 2) if x_last_3_num > 1 else 0,
                "avg_speed_last_3": round((x_last_3["distance"] / x_last_3["resultTime"]).mean() * 3.6, 2) if x_last_3_num > 0 else 0,
                "max_speed_last_3": round((x_last_3["distance"] / x_last_3["resultTime"]).max() * 3.6, 2) if x_last_3_num > 0 else 0,
                "min_speed_last_3": round((x_last_3["distance"] / x_last_3["resultTime"]).min() * 3.6, 2) if x_last_3_num > 0 else 0,
                "win_rate_last_3": round((x_last_3["place"] == 1).mean(), 2) if x_last_3_num > 0 else 0,
                "top_3_rate_last_3": round((x_last_3["place"] <= 3).mean(), 2) if x_last_3_num > 0 else 0,
                "avg_time_per_100m_last_3": round(((x_last_3["resultTime"] / x_last_3["distance"]) * 100).mean(), 2) if x_last_3_num > 0 else 0,
                "min_time_per_100m_last_3": round(((x_last_3["resultTime"] / x_last_3["distance"]) * 100).min(), 2) if x_last_3_num > 0 else 0,
                "max_time_per_100m_last_3": round(((x_last_3["resultTime"] / x_last_3["distance"]) * 100).max(), 2) if x_last_3_num > 0 else 0,
            }
        )

    def calculate_dog_features(self, x):
        weight = x["weightInKg"].dropna()
        weight_num = len(weight)
        return pd.Series(
            {
                "avg_weight_last_5": round(weight.mean(), 2) if weight_num > 0 else 0,
                "max_weight_last_5": round(weight.max(), 2) if weight_num > 0 else 0,
                "min_weight_last_5": round(weight.min(), 2) if weight_num > 0 else 0,
                "var_weight_last_5": round(weight.var(), 2) if weight_num > 1 else 0,
            }
        )

    def calculate_box_features(self, x, box_number):
        box_races = x[x["boxNumber"] == box_number]
        box_num_races = len(box_races)
        return pd.Series(
            {
                "box_num_races_last_5": box_num_races,
                "box_wins_last_5": (box_races["place"] == 1).sum() if box_num_races > 0 else -1,
                "box_top_3_finishes_last_5": (box_races["place"] <= 3).sum() if box_num_races > 0 else -1,
                "box_win_rate_last_5": (round((box_races["place"] == 1).sum() / box_num_races, 2) if box_num_races > 0 else -1),
                "box_top_3_finishes_rate_last_5": (round((box_races["place"] <= 3).sum() / box_num_races, 2) if box_num_races > 0 else -1),
                "box_avg_finish_last_5": round(box_races["place"].mean(), 2) if box_num_races > 0 else -1,
            }
        )

    def calculate_rating_features(self, x):
        positive_rating = x[x["rating"] > 0]["rating"]
        return pd.Series(
            {
                "avg_rating_last_5": round(positive_rating.mean(), 2) if not positive_rating.empty else 0,
                "min_rating_last_5": round(positive_rating.min(), 2) if not positive_rating.empty else 0,
                "max_rating_last_5": round(positive_rating.max(), 2) if not positive_rating.empty else 0,
                "var_rating_last_5": round(positive_rating.var(), 2) if len(positive_rating) > 1 else 0,
            }
        )

    def calculate_dog_age(self, x, race_date):
        return pd.Series(
            {
                "dogId": x["dogId"],
                "dog_age_days": int((race_date - pd.to_datetime(x["dateWhelped"])).total_seconds() / 86400) if x["dateWhelped"] != None else -1,
            }
        )

    def calculate_margin_features(self, x):
        winning_races = x[(x["place"] == 1) & (~np.isnan(x["resultMargin"]))]
        losing_races = x[(x["place"] > 1) & (~np.isnan(x["resultMargin"]))]
        win_races_num = len(winning_races)
        losing_races_num = len(losing_races)
        total_races_num = win_races_num + losing_races_num
        margin_treshold = 0.2
        lossMargin = losing_races["resultMargin"]
        winMargin = winning_races["resultMargin"]
        return pd.Series(
            {
                "loss_margin_avg_last_5": round(lossMargin.mean(), 2) if losing_races_num > 0 else -1,
                "loss_margin_max_last_5": round(lossMargin.max(), 2) if losing_races_num > 0 else -1,
                "loss_margin_min_last_5": round(lossMargin.min(), 2) if losing_races_num > 0 else -1,
                "loss_margin_consistency_last_5": round(
                    (1 / (1 + lossMargin.std()) if losing_races_num > 1 and not np.isinf(lossMargin.std()) else (1 if losing_races_num == 1 else -1)),
                    2,
                ),
                "close_loss_rate_last_5": round(
                    (len(losing_races[lossMargin < margin_treshold]) / total_races_num if total_races_num > 0 else -1),
                    2,
                ),
                "close_loss_rate_loss_last_5": round(
                    (len(losing_races[lossMargin < margin_treshold]) / losing_races_num if losing_races_num > 0 else -1),
                    2,
                ),
                "win_margin_avg_last_5": round(winMargin.mean(), 2) if win_races_num > 0 else -1,
                "win_margin_max_last_5": round(winMargin.max(), 2) if win_races_num > 0 else -1,
                "win_margin_min_last_5": round(winMargin.min(), 2) if win_races_num > 0 else -1,
                "win_margin_consistency_last_5": round(
                    (1 / (1 + winMargin.std()) if win_races_num > 1 and not np.isinf(winMargin.std()) else (1 if win_races_num == 1 else -1)),
                    2,
                ),
                "close_win_rate_last_5": round(
                    (len(winning_races[winMargin < margin_treshold]) / total_races_num if total_races_num > 0 else -1),
                    2,
                ),
                "close_win_rate_wins_last_5": round(
                    (len(winning_races[winMargin < margin_treshold]) / win_races_num if win_races_num > 0 else -1),
                    2,
                ),
            }
        )

    def calculate_pir(self, pir):
        pir_features = pir.apply(self.calculate_pir_features)
        pir_features = pir_features.unstack()
        pir_features = pir_features.reset_index()
        pir_features = pir_features.rename(columns={"pir_dogId": "dogId"})
        return pir_features

    def calculate_pir_features(self, pir_series):
        pir_lists = pir_series.apply(self.split_and_encode_pir)
        pir_list_flat = np.array([item for sublist in pir_lists for item in sublist])
        pir_num = len(pir_list_flat)
        return pd.Series(
            {
                "pir_avg": round(np.mean(pir_list_flat), 2) if pir_num > 0 else 0,
                "pir_volatility": round(np.std(pir_list_flat), 2) if pir_num > 0 else 0,
                "pir_starting_avg": round(np.mean([pir[-1] for pir in pir_lists]), 2) if pir_num > 0 else 0,
                "pir_improvement_avg": round(np.mean([pir[-1] - pir[0] for pir in pir_lists]), 2) if pir_num > 0 else 0,
                "pir_best": np.min(pir_list_flat) if pir_num > 0 else 0,
                "pir_worst": np.max(pir_list_flat) if pir_num > 0 else 0,
                "pir_range": (np.min(pir_list_flat) - np.max(pir_list_flat)) if pir_num > 0 else 0,
                "pir_median": np.median(pir_list_flat) if pir_num > 0 else 0,
                "pir_early_avg": round(np.mean([self.half_array_mean(pir, Direction.EARLY) for pir in pir_lists]), 1) if pir_num > 0 else 0,
                "pir_late_avg": round(np.mean([self.half_array_mean(pir, Direction.LATE) for pir in pir_lists]), 1) if pir_num > 0 else 0,
                "pir_consistency": round(1 - (max(pir_list_flat) - min(pir_list_flat)) / pir_num, 2) if pir_num > 0 else 0,
                "pir_top3_ratio": round(sum(1 for pir in pir_list_flat if pir <= 3) / pir_num, 2) if pir_num > 0 else 0,
                "pir_win_ratio": round(sum(1 for pir in pir_list_flat if pir == 1) / pir_num, 2) if pir_num > 0 else 0,
                "pir_trend": round(0 if pir_num == 1 else stats.linregress(np.arange(pir_num), pir_list_flat).slope, 2) if pir_num > 0 else 0,
                "pir_trend_per_race_avg": (
                    round(np.mean([0 if len(set(pir)) == 1 else stats.linregress(np.arange(len(pir)), pir).slope for pir in pir_lists]), 2)
                    if pir_num > 0
                    else 0
                ),
            }
        )

    def calculate_performance_trend(self, places):
        places = places.tolist()
        if len(places) < 1:
            return 0
        max_place = max(places)
        points = [max_place - p + 1 for p in places]
        weights = list(range(len(points), 0, -1))
        weighted_sum = sum(p * w for p, w in zip(points, weights))
        total_weight = sum(weights)
        weighted_average = weighted_sum / total_weight
        overall_average = sum(points) / len(points)
        diff = weighted_average - overall_average
        return round(diff, 2) if not np.isnan(diff) else 0

    def half_array_mean(self, arr, direction: Direction) -> float:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if len(arr) == 0:
            return np.nan
        elif len(arr) == 1:
            return float(arr[0])

        midpoint = len(arr) // 2

        if direction == Direction.EARLY:
            return float(np.mean(arr[max(1, midpoint) :]))
        elif direction == Direction.LATE:
            return float(np.mean(arr[:midpoint]))
        else:
            raise ValueError("direction must be either Direction.EARLY or Direction.LATE")

    def distance_category(self, distance):
        if distance < 400:
            return "sprint"
        elif distance < 600:
            return "middle"
        else:
            return "staying"

    def split_and_encode_pir(self, x):
        if pd.isna(x) or x == "":
            return [0]
        return [self.encode_race_result(i) for i in list(x) if i][::-1]

    def encode_race_result(self, result):
        if isinstance(result, Number) and not math.isnan(result):
            return result
        if isinstance(result, str) and result.isdigit():
            return int(result)
        return 10

    def time_to_components(self, time_str):
        try:
            time_obj = datetime.strptime(time_str, "%I:%M%p")
            hour = time_obj.hour
            minute = time_obj.minute
            return hour, minute
        except ValueError as e:
            print(f"Error parsing time string '{time_str}': {e}")
            return None, None
