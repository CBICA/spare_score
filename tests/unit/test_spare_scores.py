import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("../../spare_scores")
from util import load_df, load_model
from mlp_torch import MLPDataset

from spare_scores import spare_test, spare_train

class CheckMLPDataset(unittest.TestCase):
    def test_len(self):
        # test case 1: testing length 
        self.X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.Y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.Dataset = MLPDataset(self.X, self.Y)
        self.assertTrue(len(self.Dataset) == 8)
    
    def test_idx(self):
        # test case 2: testing getter 
        self.X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.Y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.Dataset = MLPDataset(self.X, self.Y)
        self.assertTrue(self.Dataset[0] == (1, 1))
        self.assertTrue(self.Dataset[len(self.Dataset) - 1] == (8, 8))

class CheckSpareScores(unittest.TestCase):

    def test_spare_test_SVM(self):
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")

        # Test case 1: Test with df
        result = spare_test(self.df_fixture, self.model_fixture)
        status_code, status, result = (
            result["status_code"],
            result["status"],
            result["data"],
        )
        self.assertTrue(status == "OK")
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(result.shape[0] == self.df_fixture.shape[0])
        self.assertTrue("SPARE_score" in result.columns)  # Column name

        # Test case 2: Test with csv file:
        filepath = (
            Path(__file__).resolve().parent.parent / "fixtures" / "sample_data.csv"
        )
        filepath = str(filepath)
        result = spare_test(filepath, self.model_fixture)
        status, result = result["status"], result["data"]
        self.assertTrue(status == "OK")
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(result.shape[0] == self.df_fixture.shape[0])
        self.assertTrue("SPARE_score" in result.columns)  # Column name

        # Test case 3: Column required by the model is missing
        self.df_fixture.drop(columns="ROI1", inplace=True)
        result = spare_test(self.df_fixture, self.model_fixture)
        # {'status' : "Not all predictors exist in the input dataframe: ['ROI1']",
        #  'data'   : ['ROI1']}
        _, status, result = (
            result["status_code"],
            result["status"],
            result["data"],
        )
        self.assertTrue(
            status == "Not all predictors exist in the input dataframe: ['ROI1']"
        )
        self.assertTrue(result == ["ROI1"])

    def test_spare_train_MLP(self): 
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")
        # Test case 1: Testing spare_train with MLP model
        result = spare_train(
            self.df_fixture,
            "Age",
            model_type="MLP",
            data_vars=[
                "ROI1",
                "ROI2",
                "ROI3",
                "ROI4",
                "ROI5",
                "ROI6",
                "ROI7",
                "ROI8",
                "ROI9",
                "ROI10",
            ],
        )
        status, result_data = result["status"], result["data"]
        metadata = result_data[1]
        self.assertTrue(status == "OK")
        self.assertTrue(metadata["mdl_type"] == "MLP")
        self.assertTrue(metadata["kernel"] == "linear")
        self.assertTrue(
            set(metadata["predictors"]) == set(self.model_fixture[1]["predictors"])
        )
        self.assertTrue(metadata["to_predict"] == self.model_fixture[1]["to_predict"])

    def test_spare_train_MLPTorch(self):
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")
        # Test case 1: testing training an MLPTorch model
        result = spare_train(
            self.df_fixture,
            "Age",
            model_type="MLPTorch",
            data_vars=[
                "ROI1",
                "ROI2",
                "ROI3",
                "ROI4",
                "ROI5",
                "ROI6",
                "ROI7",
                "ROI8",
                "ROI9",
                "ROI10",
            ],
        )

        status, result_data = result["status"], result["data"]

        metadata = result_data[1]
        self.assertTrue(status == "OK")
        self.assertTrue(metadata["mdl_type"] == "MLPTorch")
        self.assertTrue(metadata["kernel"] == "linear")
        self.assertTrue(
            set(metadata["predictors"]) == set(self.model_fixture[1]["predictors"])
        )
        self.assertTrue(metadata["to_predict"] == self.model_fixture[1]["to_predict"])

    def test_spare_train_SVM(self):
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")

        # Test case 1: Test with df
        result = spare_train(
            self.df_fixture,
            "Age",
            data_vars=[
                "ROI1",
                "ROI2",
                "ROI3",
                "ROI4",
                "ROI5",
                "ROI6",
                "ROI7",
                "ROI8",
                "ROI9",
                "ROI10",
            ],
        )

        status, result_data = result["status"], result["data"]

        metadata = result_data[1]
        self.assertTrue(status == "OK")
        self.assertTrue(metadata["mdl_type"] == self.model_fixture[1]["mdl_type"])
        self.assertTrue(metadata["kernel"] == self.model_fixture[1]["kernel"])
        self.assertTrue(
            set(metadata["predictors"]) == set(self.model_fixture[1]["predictors"])
        )
        self.assertTrue(metadata["to_predict"] == self.model_fixture[1]["to_predict"])
        self.assertTrue(
            metadata["categorical_var_map"]
            == self.model_fixture[1]["categorical_var_map"]
        )
