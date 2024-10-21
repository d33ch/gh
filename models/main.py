from sklearn.model_selection import train_test_split
from database_handler import DatabaseHandler
from greyhound_model import GreyhoundModel

CONFIG = {
    "states": ["NSW", "VIC", "QLD", "NZ", "NT", "SA", "TAS", "WA"],
    "years": [2016],
    "mongodb_uri": "mongodb://localhost:27017/",
    "db_name": "gh",
}


def main():
    handler = DatabaseHandler(CONFIG["mongodb_uri"], CONFIG["db_name"])
    races = handler.load(CONFIG["states"], CONFIG["years"], 10)

    model = GreyhoundModel()
    X = model.preprocess(races)

    X.to_csv("X.csv", index=False)
    #    y = X["run_isWinner"]
    #    X = X.drop("run_isWinner", axis=1)
    #
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    #    model.fit(X_train, y_train)
    #
    #    evaluation_results = model.evaluate(X_test, y_test)
    #    print(f"Model evaluation results: {evaluation_results}")
    #
    #    feature_importance = model.get_feature_importance()
    #    print("Feature Importance:")
    #    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    #        print(f"{feature}: {importance}")


if __name__ == "__main__":
    main()
