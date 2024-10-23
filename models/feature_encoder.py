from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils import Utils


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_columns = ["colourCode", "trackCode", "raceTypeCode", "incomingGrade", "sex"]
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(X[col].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        X = self.drop_columns(X)
        X = self.drop_rows(X)
        X = self.encode_categorical(X)
        X = self.fill_empty_values(X)
        for column in X.columns:
            null_count = X[column].isnull().sum()
            if null_count > 0:
                print(f"{column}: {null_count} null values ({(null_count/len(X)*100):.2f}%)")
        return X

    def drop_rows(self, X):
        return X.dropna(subset=["place", "weightInKg"], how="all")

    def drop_columns(self, X):
        return X.drop("unplacedCode", axis=1)

    def fill_empty_values(self, X):
        X["prizeMoney"] = X["prizeMoney"].apply(lambda x: Utils.encode_missing_value(x, 0))
        X["rating"] = X["rating"].apply(lambda x: Utils.encode_missing_value(x, 0))
        X["form_summary_bestFinishTrackDistance"] = X["form_summary_bestFinishTrackDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        X["form_summary_bestFirstSplitTrackDistance"] = X["form_summary_bestFirstSplitTrackDistance"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X["form_summary_avgFirstSplitTrackDistance"] = X["form_summary_avgFirstSplitTrackDistance"].apply(lambda x: Utils.encode_missing_value(x, -1))
        X["form_summary_bestFirstSplitTrackDistanceBox"] = X["form_summary_bestFirstSplitTrackDistanceBox"].apply(
            lambda x: Utils.encode_missing_value(x, -1)
        )
        X.loc[X["avg_weight_last_5"].notna(), "weightInKg"] = X["avg_weight_last_5"]
        return X.reset_index(drop=True)

    def encode_categorical(self, X):
        for col in self.categorical_columns:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X
