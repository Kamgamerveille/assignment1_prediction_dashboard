from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent

        models_dir = project_root / "models"

        self.preprocessor = joblib.load(
            models_dir / "preprocessor.pkl"
        )

        self.model = joblib.load(
            models_dir / "best_random_forest.pkl"
        )

    def predict(self, input_data: pd.DataFrame):
        processed_data = self.preprocessor.transform(
            input_data
        )

        prediction = self.model.predict(
            processed_data
        )

        probabilities = self.model.predict_proba(
            processed_data
        )

        return prediction, probabilities