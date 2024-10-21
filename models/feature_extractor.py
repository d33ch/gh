import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from features import ALL_FEATURES
from enum import Enum
from scipy import stats


class Direction(Enum):
    EARLY = "early"
    LATE = "late"


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ALL_FEATURES

    def fit(self, X, y=None):
        return self

    def safe_concat(dfs):
        dfs = [df.dropna(axis=1, how="all") for df in dfs if not df.empty]
        return pd.concat(dfs, ignore_index=True)

    def transform(self, races):
        final_df = pd.DataFrame()

        for race_data in races:
            race_df = race_data["race_df"]
            runs_df = race_data["runs_df"]
            form_df = race_data["form_df"]
            track_code = runs_df["trackCode"][0]
            track_distance = runs_df["distance"][0]
            box_numbers_df = runs_df[["dogId", "boxNumber"]]
            whelped_dates_df = runs_df[["dogId", "dateWhelped"]]
            race_date = pd.to_datetime(race_df.loc[0, "raceStart"])
            dog_features = self.calculate_dog_performance_features(
                form_df, box_numbers_df, whelped_dates_df, track_code, track_distance, race_date
            )
            runs_features = self.calculate_race_features(runs_df)
            result_df = pd.merge(runs_features, dog_features, on="dogId", how="left")
            final_df = pd.concat([final_df, result_df], ignore_index=True)
        return final_df

    def calculate_race_features(self, race_df):
        return race_df[
            [
                "dogId",
                "trackCode",
                "distance",
                "raceId",
                "meetingDate",
                "raceTypeCode",
                "runId",
                "weightInKg",
                "incomingGrade",
                "rating",
                "raceNumber",
                "boxNumber",
                "rugNumber",
                "startPrice",
                "place",
                "scratched",
                "prizeMoney",
                "resultTime",
                "resultMargin",
                "sex",
                "ownerId",
                "trainerId",
                "damId",
                "sireId",
            ]
        ]

    def track_distance_features(self, x, track_code, distance):
        track_features = {}
        track_distance_df_filtered = x[(x["trackCode"] == track_code) & (x["distance"] == distance)]
        distance_df_filtered = x[x["distance"] == distance]
        track_distance_num_races = len(track_distance_df_filtered)
        distance_num_races = len(distance_df_filtered)
        track_features = {
            "track_distance_num_races_last_5": track_distance_num_races,
            "track_distance_wins_last_5": (track_distance_df_filtered["place"] == 1).sum(),
            "track_distance_win_rate_last_5": round(
                (track_distance_df_filtered["place"] == 1).mean() if track_distance_num_races > 0 else 0, 2
            ),
            "track_distance_top_3_finishes_last_5": (track_distance_df_filtered["place"] <= 3).sum(),
            "track_distance_top_3_rate_last_5": round(
                (track_distance_df_filtered["place"] <= 3).mean() if track_distance_num_races > 0 else 0, 2
            ),
            "track_distance_avg_finish_last_5": (
                round(track_distance_df_filtered["place"].mean(), 2) if track_distance_num_races > 0 else -1
            ),
            "track_distance_median_finish_last_5": (
                round(track_distance_df_filtered["place"].median(), 2) if track_distance_num_races > 0 else -1
            ),
            "track_distance_avg_time_last_5": (
                round(track_distance_df_filtered["resultTime"].mean(), 2)
                if track_distance_num_races > 0
                else float("inf")
            ),
            "track_distance_best_time_last_5": (
                track_distance_df_filtered["resultTime"].min() if track_distance_num_races > 0 else float("inf")
            ),
            "distance_num_races_last_5": distance_num_races,
            "distance_wins_last_5": (distance_df_filtered["place"] == 1).sum(),
            "distance_win_rate_last_5": round(
                (distance_df_filtered["place"] == 1).mean() if distance_num_races > 0 else 0, 2
            ),
            "distance_top_3_finishes_last_5": (distance_df_filtered["place"] <= 3).sum(),
            "distance_top_3_rate_last_5": round(
                (distance_df_filtered["place"] <= 3).mean() if distance_num_races > 0 else 0, 2
            ),
            "distance_avg_finish_last_5": (
                round(distance_df_filtered["place"].mean(), 2) if distance_num_races > 0 else -1
            ),
            "distance_avg_time_last_5": (
                round(distance_df_filtered["resultTime"].mean(), 2) if distance_num_races > 0 else float("inf")
            ),
            "distance_best_time_last_5": (
                distance_df_filtered["resultTime"].min() if distance_num_races > 0 else float("inf")
            ),
        }
        return pd.Series(track_features)

    def calculate_track_distance_features(self, form_df_group, track_code, distance):
        track_features = form_df_group.apply(
            lambda x: self.track_distance_features(x, track_code, distance), include_groups=False
        )
        track_features = track_features.reset_index()
        track_features = track_features.rename(columns={"track_dogId": "dogId"})
        return track_features

    def box_features(self, x, box_number):
        box_races = x[x["boxNumber"] == box_number]
        box_num_races = len(box_races)
        box_features = {
            "box_num_races_last_5": box_num_races,
            "box_wins_last_5": (box_races["place"] == 1).sum() if box_num_races > 0 else -1,
            "box_top_3_finishes_last_5": (box_races["place"] <= 3).sum() if box_num_races > 0 else -1,
            "box_win_rate_last_5": (
                round((box_races["place"] == 1).sum() / box_num_races, 2) if box_num_races > 0 else -1
            ),
            "box_top_3_finishes_rate_last_5": (
                round((box_races["place"] <= 3).sum() / box_num_races, 2) if box_num_races > 0 else -1
            ),
            "box_avg_finish_last_5": round(box_races["place"].mean(), 2) if box_num_races > 0 else -1,
        }
        return pd.Series(box_features)

    def calculate_box_features(self, form_df_group, box_numbers_df):
        features = form_df_group.apply(
            lambda x: self.box_features(
                x, box_numbers_df.loc[box_numbers_df["dogId"] == x.name, "boxNumber"].values[0]
            ),
            include_groups=False,
        )
        features = features.reset_index()
        # features = features.rename(columns={"box_dogId": "dogId"})
        return features

    def calculate_rating_features(self, x):
        positive_rating = x[x["rating"] > 0]["rating"]
        rating_features = {
            "avg_rating_last_5": round(positive_rating.mean(), 2) if not positive_rating.empty else 0,
            "min_rating_last_5": round(positive_rating.min(), 2) if not positive_rating.empty else 0,
            "max_rating_last_5": round(positive_rating.max(), 2) if not positive_rating.empty else 0,
            "var_rating_last_5": round(positive_rating.var(), 2) if not positive_rating.empty else 0,
        }
        return pd.Series(rating_features)

    def calculate_dog_performance_features(self, form_df, box_numbers_df, whelped_dates_df, track, distance, race_date):
        form_df["pir_with_place"] = form_df["pir"].fillna("") + form_df["place"].astype(str).fillna("")
        form_df["meetingDate"] = pd.to_datetime(form_df["meetingDate"])
        form_df_sorted = form_df.sort_values(["dogId", "meetingDate"], ascending=[True, False])
        form_df_sorted["speed"] = form_df_sorted["distance"] / form_df_sorted["resultTime"]
        form_df_sorted["time_per_100m"] = (form_df_sorted["resultTime"] / form_df_sorted["distance"]) * 100
        form_df_sorted["distance_category"] = form_df_sorted["distance"].apply(self.distance_category)
        form_df_group = form_df_sorted.groupby("dogId")
        result_features = self.calculate_result_features(form_df_group)
        track_features = self.calculate_track_distance_features(form_df_group, track, distance)
        box_features = self.calculate_box_features(form_df_group, box_numbers_df)
        dog_features = self.calculate_dog_features(form_df_group, whelped_dates_df, race_date)
        features = {
            "avg_speed_last_5": lambda x: round(x.mean(), 2),
            "max_speed_last_5": lambda x: x.max(),
            "min_speed_last_5": lambda x: x.min(),
            "avg_time_per_100m_last_5": lambda x: x.head(5).mean(),
            "min_time_per_100m_last_5": lambda x: x.head(5).min(),
            "max_time_per_100m_last_5": lambda x: x.head(5).max(),
            "avg_odds_last_5": lambda x: x.head(5).mean(),
            "avg_prize_last_5": lambda x: x.head(5).mean(),
            "total_prize_last_5": lambda x: x.head(5).sum(),
        }
        features = pd.merge(result_features, track_features, on="dogId", how="left")
        features = pd.merge(features, box_features, on="dogId", how="left")
        features = pd.merge(features, dog_features, on="dogId", how="left")
        return features

    def calculate_dog_features(self, form_df_group, whelped_dates_df, race_date):
        features = pd.DataFrame()
        features["avg_weight_last_5"] = round(form_df_group["weightInKg"].mean(), 2)
        features["max_weight_last_5"] = round(form_df_group["weightInKg"].max(), 2)
        features["min_weight_last_5"] = round(form_df_group["weightInKg"].min(), 2)
        features["var_weight_last_5"] = round(form_df_group["weightInKg"].var(), 2)
        features["days_since_last_race"] = (
            ((race_date - form_df_group["meetingDate"].first()).dt.total_seconds() / 86400).fillna(0).astype(int)
        )
        age_features = whelped_dates_df.apply(lambda x: self.calculate_dog_age(x, race_date), axis=1)
        rating_features = form_df_group.apply(self.calculate_rating_features, include_groups=False)
        features = pd.merge(features, rating_features, on="dogId", how="left")
        features = pd.merge(features, age_features, on="dogId", how="left")
        return features

    def calculate_result_features(self, group):
        df = pd.DataFrame()
        df["avg_finish_last_5"] = round(group["place"].mean(), 2)
        df["median_finish_last_5"] = round(group["place"].median(), 2)
        df["best_finish_last_5"] = group["place"].min()
        df["worst_finish_last_5"] = group["place"].max()
        df["var_finish_last_5"] = group["place"].apply(lambda x: round(x.var(), 2) if x.count() >= 2 else 0)
        df["avg_finish_last_3"] = group["place"].apply(lambda x: round(x.head(3).mean(), 2))
        df["median_finish_last_3"] = group["place"].apply(lambda x: round(x.head(3).median(), 2))
        df["best_finish_last_3"] = group["place"].apply(lambda x: round(x.head(3).min(), 2))
        df["worst_finish_last_3"] = group["place"].apply(lambda x: round(x.head(3).max(), 2))
        df["var_finish_last_3"] = group["place"].apply(
            lambda x: round(x.head(3).var(), 2) if x.head(3).count() >= 2 else 0
        )
        df["avg_speed_last_5"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).mean() * 3.6, 2), include_groups=False
        )
        df["max_speed_last_5"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).max() * 3.6, 2), include_groups=False
        )
        df["min_speed_last_5"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).min() * 3.6, 2), include_groups=False
        )
        df["avg_speed_last_3"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).head(3).mean() * 3.6, 2), include_groups=False
        )
        df["max_speed_last_3"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).head(3).max() * 3.6, 2), include_groups=False
        )
        df["min_speed_last_3"] = group.apply(
            lambda x: round((x["distance"] / x["resultTime"]).head(3).min() * 3.6, 2), include_groups=False
        )
        df["win_rate_last_5"] = group["place"].agg(lambda x: round((x == 1).mean(), 2))
        df["win_rate_last_3"] = group["place"].agg(lambda x: round((x == 1).head(3).mean(), 2))
        df["top_3_rate_last_5"] = group["place"].agg(lambda x: round((x <= 3).mean(), 2))
        df["top_3_rate_last_3"] = group["place"].agg(lambda x: round((x <= 3).head(3).mean(), 2))
        df["performance_trend"] = group["place"].apply(self.calculate_performance_trend)
        margin_features = group.apply(self.calculate_margin_features, include_groups=False)
        df = pd.merge(df, margin_features, on="dogId", how="left")
        return df

    def calculate_dog_age(self, x, race_date):
        return pd.Series(
            {
                "dogId": x["dogId"],
                "dog_age_days": int((race_date - pd.to_datetime(x["dateWhelped"])).total_seconds() / 86400),
            }
        )

    def calculate_margin_features(self, group):
        features = {}
        winning_races = group[group["place"] == 1]
        losing_races = group[group["place"] > 1]
        winning_races_num = len(winning_races)
        losing_races_num = len(losing_races)
        total_races_num = winning_races_num + losing_races_num
        margin_treshold = 0.2
        features["loss_margin_avg_last_5"] = (
            round(losing_races["resultMargin"].mean(), 2) if losing_races_num > 0 else -1
        )
        features["loss_margin_max_last_5"] = (
            round(losing_races["resultMargin"].max(), 2) if losing_races_num > 0 else -1
        )
        features["loss_margin_min_last_5"] = (
            round(losing_races["resultMargin"].min(), 2) if losing_races_num > 0 else -1
        )
        features["loss_margin_consistency_last_5"] = round(
            (
                1 / (1 + losing_races["resultMargin"].std())
                if losing_races_num > 1 and not np.isinf(losing_races["resultMargin"].std())
                else (1 if losing_races_num == 1 else -1)
            ),
            2,
        )
        features["close_loss_rate_last_5"] = round(
            (
                len(losing_races[losing_races["resultMargin"] < margin_treshold]) / total_races_num
                if total_races_num > 0
                else 0
            ),
            2,
        )
        features["close_loss_rate_loss_last_5"] = round(
            (
                len(losing_races[losing_races["resultMargin"] < margin_treshold]) / losing_races_num
                if losing_races_num > 0
                else 0
            ),
            2,
        )
        features["win_margin_avg_last_5"] = (
            round(winning_races["resultMargin"].mean(), 2) if winning_races_num > 0 else -1
        )
        features["win_margin_max_last_5"] = (
            round(winning_races["resultMargin"].max(), 2) if winning_races_num > 0 else -1
        )
        features["win_margin_min_last_5"] = (
            round(winning_races["resultMargin"].min(), 2) if winning_races_num > 0 else -1
        )
        features["win_margin_consistency_last_5"] = round(
            (
                1 / (1 + winning_races["resultMargin"].std())
                if winning_races_num > 1 and not np.isinf(winning_races["resultMargin"].std())
                else (1 if winning_races_num == 1 else -1)
            ),
            2,
        )
        features["close_win_rate_last_5"] = round(
            (
                len(winning_races[winning_races["resultMargin"] < margin_treshold]) / total_races_num
                if total_races_num > 0
                else 0
            ),
            2,
        )
        features["close_win_rate_wins_last_5"] = round(
            (
                len(winning_races[winning_races["resultMargin"] < margin_treshold]) / winning_races_num
                if winning_races_num > 0
                else 0
            ),
            2,
        )
        return pd.Series(features)

    def calculate_pir(self, pir):
        pir_features = pir.apply(self.calculate_pir_features)
        pir_features = pir_features.unstack()
        pir_features = pir_features.reset_index()
        pir_features = pir_features.rename(columns={"pir_dogId": "dogId"})
        return pir_features

    def calculate_pir_features(self, pir_series):
        pir_lists = pir_series.head(5).apply(self.split_and_encode_pir)
        pir_list_flat = np.array([item for sublist in pir_lists for item in sublist])

        if len(pir_list_flat) == 0:
            return pd.Series(
                {
                    "pir_avg": 0,
                    "pir_volatility": 0,
                    "pir_starting_avg": 0,
                    "pir_improvement_avg": 0,
                    "pir_best": 0,
                    "pir_worst": 0,
                    "pir_range": 0,
                    "pir_median": 0,
                    "pir_early_avg": 0,
                    "pir_late_avg": 0,
                    "pir_consistency": 0,
                    "pir_top3_ratio": 0,
                    "pir_trend": 0,
                    "pir_trend_per_race_avg": 0,
                }
            )

        pir_avg = round(np.mean(pir_list_flat), 2)
        pir_volatility = round(np.std(pir_list_flat), 2)
        pir_starting_avg = round(np.mean([pir[-1] for pir in pir_lists]), 2)
        pir_improvement_avg = round(np.mean([pir[-1] - pir[0] for pir in pir_lists]), 2)
        pir_best = np.min(pir_list_flat)
        pir_worst = np.max(pir_list_flat)
        pir_range = pir_worst - pir_best
        pir_consistency = round(1 - (max(pir_list_flat) - min(pir_list_flat)) / len(pir_list_flat), 2)
        pir_median = np.median(pir_list_flat)
        pir_early_avg = round(np.mean([self.half_array_mean(pir, Direction.EARLY) for pir in pir_lists]), 1)
        pir_late_avg = round(np.mean([self.half_array_mean(pir, Direction.LATE) for pir in pir_lists]), 1)
        pir_top3_ratio = round(sum(1 for pir in pir_list_flat if pir <= 3) / len(pir_list_flat), 2)
        pir_trend = round(stats.linregress(np.arange(len(pir_list_flat)), pir_list_flat).slope, 2)
        pir_trend_per_race_avg = round(
            np.mean(
                [0 if len(set(pir)) == 1 else stats.linregress(np.arange(len(pir)), pir).slope for pir in pir_lists]
            ),
            2,
        )
        return pd.Series(
            {
                "pir_avg": pir_avg,
                "pir_volatility": pir_volatility,
                "pir_starting_avg": pir_starting_avg,
                "pir_improvement_avg": pir_improvement_avg,
                "pir_best": pir_best,
                "pir_worst": pir_worst,
                "pir_range": pir_range,
                "pir_median": pir_median,
                "pir_early_avg": pir_early_avg,
                "pir_late_avg": pir_late_avg,
                "pir_consistency": pir_consistency,
                "pir_top3_ratio": pir_top3_ratio,
                "pir_trend": pir_trend,
                "pir_trend_per_race_avg": pir_trend_per_race_avg,
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

    def get_feature_names(self):
        return self.features

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
        if result.isdigit():
            return int(result)
        return 9
