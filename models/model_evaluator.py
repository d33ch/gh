from sklearn.metrics import mean_squared_error, r2_score


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "R2": r2}
