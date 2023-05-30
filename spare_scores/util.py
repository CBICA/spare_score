import gzip
import logging
import os
import pickle
from typing import Union

import numpy as np
import pandas as pd


def expspace(span: list):
    return np.exp(np.linspace(span[0], 
                              span[1], 
                              num=int(span[1])-int(span[0])+1))

def load_df(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
    return pd.read_csv(df, low_memory=False) if isinstance(df, str)\
                                             else df.copy()

def add_file_extension(filename, extension):
    if not filename.endswith(extension):
        filename += extension
    return filename

def check_file_exists(filename, logger):
    # Make sure that no overwrites happen:
    if filename is None or filename == '':
        return False
    if os.path.exists(filename):
        print("The output filename " + filename + ", corresponds to an "
              + "existing file, interrupting execution to avoid overwrite.")
        logger.info("The output filename " + filename + ", corresponds to an "
              + "existing file, interrupting execution to avoid overwrite.")
        return True
    return False

def save_file(result, output, action, logger):
    # Add the correct extension:
    if action == 'train':
        output = add_file_extension(output, '.pkl.gz')
    if action == 'test':
        output = add_file_extension(output, '.csv')

    # Make directory doesn't exist:
    if not os.path.exists(output):
        dirname, fname = os.path.split(output)
        try:
            os.mkdir(dirname)
            logger.info("Created directory {dirname}")
        except FileExistsError:
            logger.info("Directory of file already exists.")
        except FileNotFoundError:
            logger.info("Directory couldn't be created")

    # Create the file:
    if action == 'train':
        with gzip.open(output, 'wb') as f:
            pickle.dump(result, f)
            logger.info(f'Model {fname} saved to {dirname}/{fname}')
    
    if action == 'test':
        try:
            result.to_csv(output)
        except Exception as e:
            logger.info(e)
        logger.info(f'Spare scores {fname} saved to {dirname}/{fname}')
    
    return

def is_unique_identifier(df, column_names):
    # Check the number of unique combinations
    unique_combinations = df[column_names].drop_duplicates()
    num_unique_combinations = len(unique_combinations)

    # Check the total number of rows
    num_rows = df.shape[0]

    # Return True if the number of unique combinations is equal to the total number of rows
    return num_unique_combinations == num_rows