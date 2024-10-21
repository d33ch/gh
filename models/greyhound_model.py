from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from data_preprocessor import DataPreprocessor
from feature_extractor import FeatureExtractor
from model_evaluator import ModelEvaluator


class GreyhoundModel:
    def __init__(self):
        self.preprocessor_pipeline = Pipeline(
            [
                ("preprocessor", DataPreprocessor()),
                ("feature_extractor", FeatureExtractor()),
            ]
        )
        self.model_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        self.evaluator = ModelEvaluator()

    def preprocess(self, races):
        return self.preprocessor_pipeline.fit_transform(races)

    def fit(self, X, y):
        self.model_pipeline.fit(X, y)

    def predict(self, X):
        return self.model_pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return self.evaluator.evaluate(y, y_pred)

    def get_feature_importance(self):
        feature_importance = self.model_pipeline.named_steps["regressor"].feature_importances_
        feature_names = self.preprocessor_pipeline.named_steps["feature_extractor"].get_feature_names()
        return dict(zip(feature_names, feature_importance))
