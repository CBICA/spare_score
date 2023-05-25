import re
from pathlib import Path

import pandas as pd
import pytest

from spare_scores.data_prep import (age_sex_match, check_test, check_train,
                                    load_model, logging_basic_config,
                                    smart_unique)


def test_load_model(model_fixture):

    # Test case 1: No arguments given:
    no_args = "load_model() missing 1 required positional " + \
                 "argument: 'mdl_path'"
    with pytest.raises(TypeError, match=re.escape(no_args)):
        load_model()

    # Test case 2: Load a model
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_model.pkl.gz'
    filepath = str(filepath)
    result = load_model(filepath)
    assert result[1]['mdl_type'] == model_fixture[1]['mdl_type']
    assert result[1]['kernel'] == model_fixture[1]['kernel']
    assert result[1]['predictors'] == model_fixture[1]['predictors']
    assert result[1]['to_predict'] == model_fixture[1]['to_predict']
    assert result[1]['categorical_var_map'] == model_fixture[1]['categorical_var_map']


def test_check_train(df_fixture):
    # Test case 1: Valid input dataframe and predictors
    predictors = ['ROI1', 'ROI2', 'ROI3']
    to_predict = 'Sex'
    pos_group = 'M'
    filtered_df, filtered_predictors, mdl_type = check_train(df_fixture, 
                                                             predictors, 
                                                             to_predict, 
                                                             pos_group)
    assert filtered_df.equals(df_fixture)  # Check if filtered dataframe is the same as the input dataframe
    assert filtered_predictors == predictors  # Check if filtered predictors are the same as the input predictors
    assert mdl_type == 'SVM Classification'  # Check if the SPARE model type is correct

    # Test case 2: Missing required columns
    df_missing_columns = pd.DataFrame({'ID': [1, 2, 3],
                                       'Var1': [1, 2, 3],
                                       'Var2': [4, 5, 6]})
    predictors = ['Var1', 'Var2']
    to_predict = 'ToPredict'
    pos_group = '1'
    res = check_train(df_missing_columns, predictors, to_predict, pos_group)
    assert res == 'Please check required columns: ID, Age, Sex.'

    # Test case 3: Predictor not in input dataframe
    df = pd.DataFrame({'ID': [1, 2, 3],
                       'Age': [30, 40, 50],
                       'Sex': ['M', 'F', 'M'],
                       'Var1': [1, 2, 3]})
    predictors = ['Var1', 'Var2']  # Var2 is not in the input dataframe
    to_predict = 'ToPredict'
    pos_group = '1'
    res = check_train(df, predictors, to_predict, pos_group)
    assert res == 'Not all predictors exist in the input dataframe.'


def test_check_test():
    # Test case 1: Valid input dataframe and meta_data
    df = pd.DataFrame({'ID': [1, 2, 3],
                       'Age': [30, 40, 50],
                       'Sex': ['M', 'F', 'M'],
                       'Var1': [1, 2, 3],
                       'Var2': [4, 5, 6]})
    meta_data = {'predictors': ['Var1', 'Var2'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}

    res = check_test(df, meta_data)
    assert res is None  # Check if filtered dataframe is the same as the input dataframe

    # Test case 2: Missing predictors in the input dataframe
    df_missing_predictors = pd.DataFrame({'ID': [1, 2, 3],
                                          'Age': [30, 40, 50],
                                          'Sex': ['M', 'F', 'M'],
                                          'Var1': [1, 2, 3]})
    meta_data = {'predictors': ['Var1', 'Var2', 'Var3'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}
    res = check_test(df_missing_predictors, meta_data)
    assert res == "Not all predictors exist in the input dataframe: ['Var2', 'Var3']"

    # Test case 3: Passing check.
    df_age_outside_range = pd.DataFrame({'ID': [1, 2, 3],
                                         'Age': [20, 45, 55],
                                         'Sex': ['M', 'F', 'M'],
                                         'Var1': [1, 2, 3],
                                         'Var2': [4, 5, 6]})
    meta_data = {'predictors': ['Var1', 'Var2'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}
    res = check_test(df_age_outside_range, meta_data)
    assert res == None


def test_smart_unique():
    pass

def test_age_sex_match():
    pass

def test_logging_basic_config():
    pass
