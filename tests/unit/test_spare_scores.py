from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from spare_scores.spare_scores import spare_test, spare_train
from spare_scores.util import load_df, load_model

class CheckSpareScores(unittest.TestCase):

    def test_spare_test(self):
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")

        # Test case 1: Test with df
        result = spare_test(self.df_fixture, self.model_fixture)
        status_code, status, result = result['status_code'], result['status'], result['data']
        self.assertTrue(status == 'OK')
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(result.shape[0] == self.df_fixture.shape[0])
        self.assertTrue('SPARE_score' in result.columns)  # Column name

        # Test case 2: Test with csv file:
        filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
        filepath = str(filepath)
        result = spare_test(filepath, self.model_fixture)
        status, result = result['status'], result['data']
        self.assertTrue(status == 'OK')
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(result.shape[0] == self.df_fixture.shape[0])
        self.assertTrue('SPARE_score' in result.columns) # Column name

        # Test case 3: Column required by the model is missing
        self.df_fixture.drop(columns='ROI1', inplace=True)
        result = spare_test(self.df_fixture, self.model_fixture)
        # {'status' : "Not all predictors exist in the input dataframe: ['ROI1']", 
        #  'data'   : ['ROI1']}
        status_code, status, result = result['status_code'], result['status'], result['data']
        self.assertTrue(status == 'Not all predictors exist in the input dataframe: [\'ROI1\']')
        self.assertTrue(result == ['ROI1'])


    def test_spare_train(self):
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        self.model_fixture = load_model("../fixtures/sample_model.pkl.gz")

        # Test case 1: Test with df
        result = spare_train(self.df_fixture, 
                             'Age',
                             data_vars = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 
                                          'ROI6', 'ROI7', 'ROI8', 'ROI9', 'ROI10'],
                              )

        status, result = result['status'], result['data']

        metadata = result[1] # For some reason, this is None
        self.assertTrue(status == 'OK')
        self.assertTrue(metadata['mdl_type'] == self.model_fixture[1]['mdl_type'])
        self.assertTrue(metadata['kernel'] == self.model_fixture[1]['kernel'])
        self.assertTrue(set(metadata['predictors']) == set(self.model_fixture[1]['predictors']))
        self.assertTrue(metadata['to_predict'] == self.model_fixture[1]['to_predict'])
        self.assertTrue(metadata['categorical_var_map'] == self.model_fixture[1]['categorical_var_map'])



