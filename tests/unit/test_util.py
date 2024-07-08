import re
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import pytest

from spare_scores.util import (add_file_extension, check_file_exists, expspace,
                               is_unique_identifier, load_df, load_examples,
                               load_model, save_file)

class CheckSpareScoresUtil(unittest.TestCase):
    def test_load_model(self):
        self.model_fixture = "../fixture/sample_model.pkl.gz"
        # Test case 1: No arguments given:
        no_args = "load_model() missing 1 required positional " + \
                     "argument: 'mdl_path'"
        with pytest.raises(TypeError, match=re.escape(no_args)):
            load_model()

        # Test case 2: Load a model
        filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_model.pkl.gz'
        filepath = str(filepath)
        result = load_model(filepath)
        self.assertTrue(result[1]['mdl_type'] == self.model_fixture[1]['mdl_type'])
        self.assertTrue(result[1]['kernel'] == self.model_fixture[1]['kernel'])
        self.assertTrue(result[1]['predictors'] == self.model_fixture[1]['predictors'])
        self.assertTrue(result[1]['to_predict'] == self.model_fixture[1]['to_predict'])
        self.assertTrue(result[1]['categorical_var_map'] == self.model_fixture[1]['categorical_var_map'])

    def test_expspace(self):
        # Test case 1: span = [0, 2]
        span = [0, 2]
        expected_result = np.array([1., 2.71828183, 7.3890561])
        self.assertTrue(np.allclose(expspace(span), expected_result))

        # Test case 2: span = [1, 5]
        span = [1, 5]
        expected_result = np.array([ 2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591])
        self.assertTrue(np.allclose(expspace(span), expected_result))

        # Test case 3: span = [-2, 1]
        span = [-2, 1]
        expected_result = np.array([0.13533528, 0.36787944, 1., 2.71828183])
        self.assertTrue(np.allclose(expspace(span), expected_result))

    def test_check_file_exists(self):
        pass

    def test_save_file(self):
        pass

    def test_is_unique_identifier(self):
        pass

    def test_load_model(self):
        pass

    def test_load_examples(self):
        pass

    def test_load_df(self):
        # Test case 1: Input is a string (CSV file path)
        filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
        filepath = str(filepath)
        expected_df = pd.read_csv(filepath, low_memory=False)
        self.assertTrue(load_df(filepath).equals(expected_df))

        # Test case 2: Input is already a DataFrame
        input_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

        # Test case 3: Empty DataFrame
        input_df = pd.DataFrame()
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

        # Test case 4: Large DataFrame
        input_df = pd.DataFrame({"A": range(100000), "B": range(100000)})
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

    def test_add_file_extension(self):
        # Test case 1: File extension already present
        filename = "myfile.txt"
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == "myfile.txt")

        # Test case 2: File extension not present
        filename = "myfile"
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == "myfile.txt")

        # Test case 3: Different extension
        filename = "document"
        extension = ".docx"
        self.assertTrue(add_file_extension(filename, extension) == "document.docx")

        # Test case 4: Empty filename
        filename = ""
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == ".txt")

        # Test case 5: Empty extension
        filename = "myfile"
        extension = ""
        self.assertTrue(add_file_extension(filename, extension) == "myfile")

        # Test case 6: Multiple extension dots in filename
        filename = "file.tar.gz"
        extension = ".gz"
        self.assertTrue(add_file_extension(filename, extension) == "file.tar.gz")