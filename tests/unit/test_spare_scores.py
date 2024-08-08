from pathlib import Path
import numpy as np
import pandas as pd
import os
from spare_scores.data_prep import check_test
from spare_scores.util import load_df, load_model
from spare_scores.mlp_torch import MLPDataset
from spare_scores.spare import spare_test, spare_train

def test_len():
    # test case 1: testing length
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Dataset = MLPDataset(X, Y)
    assert(len(Dataset) == 8)

def test_idx():
    # test case 2: testing getter
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Dataset = MLPDataset(X, Y)
    assert(Dataset[0] == (1, 1))
    assert(Dataset[len(Dataset) - 1] == (8, 8))

def test_spare_test_SVM():
    df_fixture = load_df("../fixtures/sample_data.csv")
    model_fixture = load_model("../fixtures/sample_model.pkl.gz")

    # Test case 1: Test with df
    result = spare_test(df_fixture, model_fixture)
    status_code, status, result = (
        result["status_code"],
        result["status"],
        result["data"],
    )
    assert(status == "OK")
    assert(isinstance(result, pd.DataFrame))
    assert(result.shape[0] == df_fixture.shape[0])
    assert("SPARE_score" in result.columns)  # Column name

    # Test case 2: Test with csv file:
    filepath = (
        Path(__file__).resolve().parent.parent / "fixtures" / "sample_data.csv"
    )
    filepath = str(filepath)
    result = spare_test(filepath, model_fixture)
    status, result = result["status"], result["data"]
    assert(status == "OK")
    assert(isinstance(result, pd.DataFrame))
    assert(result.shape[0] == df_fixture.shape[0])
    assert("SPARE_score" in result.columns)  # Column name

    # Test case 3: Column required by the model is missing
    df_fixture.drop(columns="ROI1", inplace=True)
    result = spare_test(df_fixture, model_fixture)
    # {'status' : "Not all predictors exist in the input dataframe: ['ROI1']",
    #  'data'   : ['ROI1']}
    _, status, result = (
        result["status_code"],
        result["status"],
        result["data"],
    )
    assert(
        status == "Not all predictors exist in the input dataframe: ['ROI1']"
    )
    assert(result == ["ROI1"])

def test_spare_train_MLP():
    df_fixture = load_df("../fixtures/sample_data.csv")
    model_fixture = load_model("../fixtures/sample_model.pkl.gz")
    # Test case 1: Testing spare_train with MLP model
    result = spare_train(
        df_fixture,
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
    assert(status == "OK")
    assert(metadata["mdl_type"] == "MLP")
    assert(metadata["kernel"] == "linear")
    assert(
        set(metadata["predictors"]) == set(model_fixture[1]["predictors"])
    )
    assert(metadata["to_predict"] == model_fixture[1]["to_predict"])

    # test case 2: testing MLP regression model
    result = spare_train(
        df_fixture,
        "ROI1",
        model_type="MLP",
        data_vars = [
            "ROI2",
            "ROI3",
            "ROI4",
            "ROI5",
            "ROI6",
            "ROI7",
            "ROI8",
            "ROI9",
            "ROI10"
        ]
    )
    status, result_data = result["status"], result["data"]
    metadata = result_data[1]
    assert(status == "OK")
    assert(metadata["mdl_type"] == "MLP")
    assert(metadata["kernel"] == "linear")
    # assert(metadata["to_predict"] == "to_predict")

def test_spare_train_MLPTorch():
    df_fixture = load_df("../fixtures/sample_data.csv")
    model_fixture = load_model("../fixtures/sample_model.pkl.gz")
    # Test case 1: testing training an MLPTorch model
    result = spare_train(
        df_fixture,
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
    assert(status == "OK")
    assert(metadata["mdl_type"] == "MLPTorch")
    assert(metadata["kernel"] == "linear")
    assert(
        set(metadata["predictors"]) == set(model_fixture[1]["predictors"])
    )
    assert(metadata["to_predict"] == model_fixture[1]["to_predict"])

    # test case 2: testing MLPTorch regression model
    result = spare_train(
        df_fixture,
        "ROI1",
        model_type="MLPTorch",
        data_vars = [
            "ROI2",
            "ROI3",
            "ROI4",
            "ROI5",
            "ROI6",
            "ROI7",
            "ROI8",
            "ROI9",
            "ROI10",
        ]
    )
    status, result_data = result["status"], result["data"]
    metadata = result_data[1]
    assert(status == "OK")
    assert(metadata["mdl_type"] == "MLPTorch")
    assert(metadata["kernel"] == "linear")
    # assert(metadata["to_predict"] == "to_predict")

def test_spare_train_SVM():
    df_fixture = load_df("../fixtures/sample_data.csv")
    model_fixture = load_model("../fixtures/sample_model.pkl.gz")

    # Test case 1: Test with df
    result = spare_train(
        df_fixture,
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
    assert(status == "OK")
    assert(metadata["mdl_type"] == model_fixture[1]["mdl_type"])
    assert(metadata["kernel"] == model_fixture[1]["kernel"])
    assert(
        set(metadata["predictors"]) == set(model_fixture[1]["predictors"])
    )
    assert(metadata["to_predict"] == model_fixture[1]["to_predict"])
    assert(
        metadata["categorical_var_map"]
        == model_fixture[1]["categorical_var_map"]
    )

    # test case 2: testing SVM regression model
    result = spare_train(
        df_fixture,
        "ROI1",
        data_vars = [
            "ROI2",
            "ROI3",
            "ROI4",
            "ROI5",
            "ROI6",
            "ROI7",
            "ROI8",
            "ROI9",
            "ROI10"
        ]
    )
    status, result_data = result["status"], result["data"]
    metadata = result_data[1]
    assert(status == "OK")
    assert(metadata["mdl_type"] == "SVM")
    assert(metadata["kernel"] == "linear")
    # assert(metadata["to_predict"] == "to_predict")

def test_spare_train_SVM_None():
    df_fixture = load_df("../fixtures/sample_data.csv")
    # Test case 1: Training with no data vars
    result = spare_train(
        df_fixture,
        "Age"
    )
    assert(result is not None)


def test_spare_train_SVM2():
    df_fixture = load_df("../fixtures/sample_data.csv")
    # Test case 1: Test overwrites
    result = spare_train(
        df_fixture,
        "Age",
        output="test_util.py"
    )
    assert(result["status_code"] == 2)

    # Test case 2: Train with non existing output file
    result = spare_train(
        df_fixture,
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
        output="results"
    )
    assert(os.path.isfile("results.pkl.gz") == True)
    os.remove("results.pkl.gz")

def test_spare_train_non_existing_model():
    df_fixture = load_df("../fixtures/sample_data.csv")
    # Test case 1: training with non existing model type
    result = spare_train(
        df_fixture,
        "Age",
        model_type="CNN",
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
    assert(result["status_code"] == 2)

def test_spare_test_exceptions():
    df_fixture = load_df("../fixtures/sample_data.csv")
    model_fixture = load_model("../fixtures/sample_model.pkl.gz")

    # Test case 1: Test with existing output path
    if(not os.path.isfile("output.csv")):
        f = open("output.csv", "x")
    result = spare_test(df_fixture, model_fixture, output="output")
    assert(result["status_code"] == 0)
    os.remove("output.csv")

    # Test case 2: Test with predictors not existing in the original dataframe
    data = {
        "Var1": [x for x in range(100)],
        "Var2": [x for x in range(100)],
        "label": [x**2 for x in range(100)]
    }
    df_fixture = pd.DataFrame(data=data)
    meta_data = {
        "predictors": "Not_existing"
    }
    err, cols_not_found = check_test(df_fixture, meta_data)
    assert(len(err) != 0)
    assert(cols_not_found is not None)


def test_spare_train_regression_error():
    df_fixture = load_df("../fixtures/sample_data.csv")
    # Test case 1: testing with non-integer like as predictor
    result = spare_train(
        df_fixture,
        "ScanID",
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
        ]
    )

    assert(result["status_code"] == 2)
    assert(result["status"] == "Dataset check failed before training was initiated.")

    # Test case 2: testing with a too-small dataset
    data = {
        "Var1": [1,2,3,4,5],
        "Var2": [2,4,6,8,10],
        "label": [1.5,2.4,3.2,4.5,5.5]
    }
    df_fixture = pd.DataFrame(data=data)
    result = spare_train(
        df_fixture,
        "label",
        data_vars=[
            "Var1",
            "Var2"
        ]
    )

    assert(result["status_code"] == 2)
    assert(result["status"] == "Dataset check failed before training was initiated.")

    # Test case 3: testing with a label that has to variance
    data = {
        "Var1": [1,2,3,4,5],
        "Var2": [2,4,6,8,10],
        "label": [1,1,1,1,1]
    }
    df_fixture = pd.DataFrame(data=data)
    result = spare_train(
        df_fixture,
        "label",
        data_vars=[
            "Var1",
            "Var2"
        ]
    )
    assert(result["status_code"] == 2)
    assert(result["status"] == "Dataset check failed before training was initiated.")

    # Test case 4: testing with a dataset that may be too small
    data = {
        "Var1": [x for x in range(80)],
        "Var2": [x for x in range(80)],
        "Var3": [x for x in range(80)],
        "label": [x*2 for x in range(80)]
    }

    df_fixture = pd.DataFrame(data=data)
    result = spare_train(
        df_fixture,
        "label",
        data_vars=[
            "Var1",
            "Var2"
        ]
    )

    assert(result is not None)
