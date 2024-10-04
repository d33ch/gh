from sklearn.base import BaseEstimator, TransformerMixin
from features import ALL_FEATURES


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ALL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features
