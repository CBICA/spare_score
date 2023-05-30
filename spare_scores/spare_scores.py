import gzip
import logging
import pickle
from typing import Tuple, Union

import numpy as np
import pandas as pd

from spare_scores.data_prep import *
from spare_scores.svm import run_SVM
from spare_scores.util import *
from spare_scores.classes import SpareModel, MetaData


def spare_train(
        df: Union[pd.DataFrame, str], 
        to_predict: str,
        pos_group: str = '',
        key_var: str = '',
        data_vars: list = [],
        ignore_vars: list = [],
        kernel: str = 'linear',
        output: str = '',
        verbose: int = 1,
        logs: str = '') -> Tuple[dict, dict]:
    """
    Trains a SPARE model, either classification or regression

    Args:
        df:         either a pandas dataframe or a path to a saved csv 
                    containing training data.
        to_predict: variable to predict. Binary for classification and 
                    continuous for regression. Must be one of the columnes in 
                    df.
        pos_group:  group to assign a positive SPARE score (only for 
                    classification).
        key_var:    The key variable to be used for training. If not 
                    given, the first column of the dataset is considered the 
                    primary key of the dataset.
        data_vars:  a list of predictors for the training. All must be present 
                    in columns of df. If empty list, then 
        ignore_vars:The list of predictors to be ignored for training. Can be 
                    a listkey_var, or empty. 
        kernel:     'linear' or 'rbf' (only linear is supported currently in 
                    regression).
        output:     path to save the trained model. '.pkl.gz' file extension 
                    optional. If None is given, no model will be saved.
        verbose:    Verbosity. Int, higher is more verbose. [0,1,2]
        logs:       Where to save log file. If not given, logs will only be 
                    printed out.

    Returns:
        a tuple of two dictionaries: first to contain SPARE model coefficients,
                                     and second to contain model information.
    """
    logger = logging_basic_config(verbose=verbose, filename=logs)
    
    # Make sure that no overwrites happen:
    if check_file_exists(output, logger):
        return

    # Load the data
    df = load_df(df)

    # Assume key_variable (if not given)
    if key_var == '' or key_var is None:
        key_var = df.columns[0]
        if not is_unique_identifier(df, key_var):
            logging.info("Assumed primary key is not capable of uniquely " 
                        + "identifying each row of the dataset. Assumed pkey: "
                        + key_var)

    # Assume predictors (if not given)
    if data_vars == [] or data_vars is None:
        # Predictors = all_vars - key_var - ignore_vars - to_predict
        if ignore_vars == [] or ignore_vars is None:
            data_vars = list( set(list(df))\
                              - set([key_var])\
                              - set([to_predict]))
        else:
            data_vars = list( set(list(df))\
                              - set([key_var])\
                              - set(ignore_vars)\
                              - set([to_predict]))
    predictors = data_vars

    # Check if it contains any errors.
    try:
        df, predictors, mdl_type = check_train(df, 
                                            predictors, 
                                            to_predict,
                                            key_var, 
                                            pos_group, 
                                            verbose=verbose)
    except Exception as e:
        logger.error("Dataset check failed before training was initiated.")
        print(e)
        return
    
    meta_data = MetaData(mdl_type, kernel, predictors, to_predict, key_var)
    meta_data.key_var = key_var

    # Convert categorical variables
    cat_vars = [var 
                for var in df[predictors].columns 
                if df[var].dtype.name == 'O']
    meta_data.categorical_var_map = {var: None for var in cat_vars}
    for var in cat_vars:
        if len(df[var].unique()) <= 2:
            meta_data.categorical_var_map[var] = {df[var].unique()[0]:  1, 
                                                  df[var].unique()[-1]: 2}
            df[var] = df[var].map(meta_data.categorical_var_map[var])
        
        elif len(df[var].unique()) > 2:
            raise ValueError('Categorical variables with more than 2 '
                             + 'categories are currently not supported.')
        
    # Prepare parameters
    if mdl_type == 'SVM Classification':
        meta_data.pos_group = pos_group
        to_predict_input = [to_predict,
                            [a 
                             for a in df[to_predict].unique() 
                             if a != pos_group] 
                            + [pos_group]]
        param_grid = {
                      'linear':{'C':    expspace([-9, 5])},
                      'rbf':{
                             'C':       expspace([-9, 5]), 
                             'gamma':   expspace([-5, 5])
                            }
                     }[kernel]
    
    elif mdl_type == 'SVM Regression':
        to_predict_input = to_predict
        param_grid = {'C':          expspace([-5, 5]), 
                      'epsilon':    expspace([-5, 5])}

    # Train model
    n_repeats = {'SVM Classification': 5, 'SVM Regression': 1}
    if len(df.index) > 1000:
        logger.info('Due to large dataset, first performing parameter tuning '
                     + 'with 500 randomly sampled data points.')
        
        sampled_df = df.sample(n=500, random_state=2023).reset_index(drop=True)
        _ , _ , _ , params, _ = run_SVM(sampled_df, 
                                        predictors, 
                                        to_predict_input, 
                                        param_grid=param_grid, 
                                        key_var=key_var,
                                        kernel=kernel, 
                                        n_repeats=1, 
                                        verbose=0)
        param_grid = {par: expspace([
                                       np.min(params[f'{par}_optimal']), 
                                       np.max(params[f'{par}_optimal'])
                                    ]) 
                           for par in param_grid}
    
    logger.info(f'Training {mdl_type} model...')
    try:
        df['predicted']\
        , mdl\
        , meta_data.stats\
        , meta_data.params\
        , meta_data.cv_folds = run_SVM(df, 
                                    predictors, 
                                    to_predict_input, 
                                    param_grid=param_grid, 
                                    key_var=key_var,
                                    kernel=kernel, 
                                    n_repeats=n_repeats[mdl_type], 
                                    verbose=verbose)
    except Exception as e:
        logger.info('\033[91m' + '\033[1m' 
                    + '\n\n\nspare_train(): run_SVM() failed.'
                    + '\033[0m')
        print(e)
        print("Please consider ignoring (-iv/--ignore_vars) any variables "
              + "that might not be needed for the training of the model, as "
              + "they could be causing problems.\n\n\n")
        return 
    
    meta_data.cv_results = df[list(dict.fromkeys([key_var, 
                                                  to_predict, 
                                                  'predicted']))]
    
    result = mdl, vars(meta_data)

    # Save model
    if output != '' and output is not None:
        save_file(result, output, 'train', logger)
        
    return result

def spare_test(df: Union[pd.DataFrame, str],
               mdl_path: Union[str, Tuple[dict, dict]],
               key_var: str = '',
               output: str = '',
               verbose: int = 1,
               logs: str = '') -> pd.DataFrame:
    """
    Applies a trained SPARE model on a test dataset

    Args:
        df:         either a pandas dataframe or a path to a saved csv 
                    containing the test sample.
        mdl_path:   either a path to a saved SPARE model ('.pkl.gz' file 
                    extension expected) or a tuple of SPARE model and 
                    meta_data.
        key_var:    The of key variable to be used for training. If not 
                    given, and the saved model does not contain it,the first 
                    column of the dataset is considered the primary key of the 
                    dataset.
        output:     path to save the calculated scores. '.csv' file extension 
                    optional. If None is given, no data will be saved.
        verbose:    Verbosity. Int, higher is more verbose. [0,1,2]
        logs:       Where to save log file. If not given, logs will only be 
                    printed out.

    Returns:
        a pandas dataframe containing predicted SPARE scores.
    """
    logger = logging_basic_config(verbose=verbose, filename=logs)

    # Make sure that no overwrites happen:
    if check_file_exists(output, logger):
        return

    df = load_df(df)

    # Load & check for errors / compatibility the trained SPARE model
    mdl, meta_data = load_model(mdl_path) if isinstance(mdl_path, str) \
                                          else mdl_path
    check_test(df, meta_data)

    # Assume key_variable (if not given)
    if key_var == '' or key_var is None:
        key_var = df.columns[0]
        if not is_unique_identifier(df, key_var):
            logging.info("Assumed primary key(s) are not capable of uniquely " 
                        + "identifying each row of the dataset. Assumed pkeys: "
                        + key_var)

    # Convert categorical variables
    for var, map_dict in meta_data.get('categorical_var_map',{}).items():
        if not isinstance(map_dict, dict):
            continue
        if df[var].isin(map_dict.keys()).any():
            df[var] = df[var].map(map_dict)
        else:
            expected_var = list(map_dict.keys())
            logger.error(f'Column "{var}" expected {expected_var}, but '
                          + f'received {list(df[var].unique())}')

    # Output model description
    n = len(meta_data['cv_results'].index)
    if 'Age' in meta_data['cv_results'].keys():
        a1 = int(np.floor(np.min((meta_data['cv_results']['Age']))))
        a2 = int(np.ceil(np.max((meta_data['cv_results']['Age']))))
    else:
        a1 = None
        a2 = None
    stats_metric = list(meta_data['stats'].keys())[0]
    stats = '{:.3f}'.format(np.mean(meta_data['stats'][stats_metric]))
    logger.info(f'Model Info: training N = {n} / ages = {a1} - {a2} / '
                 + f'expected {stats_metric} = {stats}')
    
    # Calculate SPARE scores
    n_ensemble = len(mdl['scaler'])
    ss = np.zeros([len(df.index), n_ensemble])
    for i in range(n_ensemble):
        X = mdl['scaler'][i].transform(df[meta_data['predictors']])
        if meta_data['mdl_type'] == 'SVM Regression':
            ss[:, i] = mdl['mdl'][i].predict(X)
            ss[:, i] = (ss[:, i] - mdl['bias_correct']['int'][i]) \
                       / mdl['bias_correct']['slope'][i]
        else:
            ss[:, i] = mdl['mdl'][i].decision_function(X)
        
        if key_var in df.columns:
            index_to_nan = df[key_var].isin(meta_data['cv_results'][key_var]\
                                   .drop(meta_data['cv_folds'][i]))
            ss[index_to_nan, i] = np.nan
    ss_mean = np.nanmean(ss, axis=1)
    ss_mean[np.all(np.isnan(ss),axis=1)] = np.nan
    
    d = {}
    d['SPARE_scores'] = ss_mean
    # Unique primary key:
    out_df = pd.DataFrame(data=d, index=df[key_var])
    
    if output != '' and output is not None:
        save_file(out_df, output, 'test', logger)
    return out_df
