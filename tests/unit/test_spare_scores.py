import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spare_scores.spare_scores import (_expspace, _load_df, add_file_extension,
                                       spare_test, spare_train)


def test_spare_test(df_fixture, model_fixture):

    # Test case 1: No arguments given:
    with pytest.raises(TypeError):
        spare_test()

    # Test case 2: Test with df
    result = spare_test(df_fixture, model_fixture)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == df_fixture.shape[0]
    assert 'SPARE_scores' in result.columns  # Column name

    # Test case 3: Test with csv file:
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
    filepath = str(filepath)
    result = spare_test(filepath, model_fixture)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == df_fixture.shape[0]
    assert 'SPARE_scores' in result.columns  # Column name

    # Test case 4: Column required by the model is missing
    df_fixture.drop(columns='ROI1', inplace=True)
    with pytest.raises(KeyError):
        spare_test(df_fixture, model_fixture)

def test_spare_train(df_fixture, model_fixture):

    # Test case 1: No arguments given:
    with pytest.raises(TypeError):
        spare_train()

    # Test case 2: Test with df
    result = spare_train(df_fixture, 
                         'Age',
                         data_vars = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 
                                      'ROI6', 'ROI7', 'ROI8', 'ROI9', 'ROI10'],
                          )
    assert result[1]['mdl_type'] == model_fixture[1]['mdl_type']
    assert result[1]['kernel'] == model_fixture[1]['kernel']
    assert result[1]['predictors'] == model_fixture[1]['predictors']
    assert result[1]['to_predict'] == model_fixture[1]['to_predict']
    assert result[1]['categorical_var_map'] == model_fixture[1]['categorical_var_map']


def test_expspace():
    # Test case 1: span = [0, 2]
    span = [0, 2]
    expected_result = np.array([1., 2.71828183, 7.3890561])
    assert np.allclose(_expspace(span), expected_result)

    # Test case 2: span = [1, 5]
    span = [1, 5]
    expected_result = np.array([ 2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591])
    assert np.allclose(_expspace(span), expected_result)

    # Test case 3: span = [-2, 1]
    span = [-2, 1]
    expected_result = np.array([0.13533528, 0.36787944, 1., 2.71828183])
    assert np.allclose(_expspace(span), expected_result)


def test_load_df():
    # Test case 1: Input is a string (CSV file path)
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
    filepath = str(filepath)
    expected_df = pd.read_csv(filepath, low_memory=False)
    assert  _load_df(filepath).equals(expected_df)

    # Test case 2: Input is already a DataFrame
    input_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    expected_df = input_df.copy()
    assert  _load_df(input_df).equals(expected_df)

    # Test case 3: Empty DataFrame
    input_df = pd.DataFrame()
    expected_df = input_df.copy()
    assert  _load_df(input_df).equals(expected_df)

    # Test case 4: Large DataFrame
    input_df = pd.DataFrame({"A": range(100000), "B": range(100000)})
    expected_df = input_df.copy()
    assert  _load_df(input_df).equals(expected_df)

def test_add_file_extension():
    # Test case 1: File extension already present
    filename = "myfile.txt"
    extension = ".txt"
    assert add_file_extension(filename, extension) == "myfile.txt"

    # Test case 2: File extension not present
    filename = "myfile"
    extension = ".txt"
    assert add_file_extension(filename, extension) == "myfile.txt"

    # Test case 3: Different extension
    filename = "document"
    extension = ".docx"
    assert add_file_extension(filename, extension) == "document.docx"

    # Test case 4: Empty filename
    filename = ""
    extension = ".txt"
    assert add_file_extension(filename, extension) == ".txt"

    # Test case 5: Empty extension
    filename = "myfile"
    extension = ""
    assert add_file_extension(filename, extension) == "myfile"

    # Test case 6: Multiple extension dots in filename
    filename = "file.tar.gz"
    extension = ".gz"
    assert add_file_extension(filename, extension) == "file.tar.gz"