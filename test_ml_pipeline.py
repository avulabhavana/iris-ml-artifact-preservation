
import json
import os
import unittest

import joblib
import pandas as pd


class TestIrisMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("iris.csv"))

    def test_dataset_not_empty(self):
        df = pd.read_csv("iris.csv")
        self.assertGreater(len(df), 0)

    def test_required_columns_exist(self):
        df = pd.read_csv("iris.csv")

        required_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species"
        ]

        for column in required_columns:
            self.assertIn(column, df.columns)

    def test_model_exists(self):
        self.assertTrue(os.path.exists("iris_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_prediction(self):
        model = joblib.load("iris_model.pkl")

        sample = pd.DataFrame([{
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }])

        prediction = model.predict(sample)
        self.assertEqual(len(prediction), 1)
        self.assertIn(
            prediction[0],
            ["setosa", "versicolor", "virginica"]
        )


if __name__ == "__main__":
    unittest.main()
