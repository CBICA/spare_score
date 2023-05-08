import gzip
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Union
from dataclasses import dataclass
from spare_scores.svm import run_SVM
from spare_scores.data_prep import *


@dataclass
class MetaData:
  """Stores training information on its paired SPARE model
  """
  mdl_type: str
  kernel: str
  predictors: list
  to_predict: str

def spare_train(df: Union[pd.DataFrame, str], 
                predictors: list,
                to_predict: str,
                pos_group: str = '',
                kernel: str = 'linear',
                verbose: int = 1,
                save_path: str = None) -> Tuple[dict, dict]:
  """Trains a SPARE model, either classification or regression

  Args:
    df: either a pandas dataframe or a path to a saved csv containing training data.
    predictors: a list of predictors for the training. All must be present in columns of df.
    to_predict: variable to predict. Binary for classification and continuous for regression.
      Must be one of the columnes in df.
    pos_group: group to assign a positive SPARE score (only for classification).
    kernel: 'linear' or 'rbf' (only linear is supported currently in regression).
    save_path: path to save the trained model. '.pkl.gz' file extension expected.
      If None is given, no model will be saved.

  Returns:
    a tuple of two dictionaries: first to contain SPARE model coefficients, and
      second to contain model information
  """
  logging_basic_config(verbose)
  df = _load_df(df)
  df, predictors, mdl_type = check_train(df, predictors, to_predict, pos_group, verbose=verbose)
  meta_data = MetaData(mdl_type, kernel, predictors, to_predict)

  # Convert categorical variables
  var_categorical = [var for var in df[predictors].columns if df[var].dtype == 'O']
  meta_data.categorical_var_map = {var: None for var in var_categorical}
  for var in var_categorical:
    if len(df[var].unique()) == 2:
      meta_data.categorical_var_map[var] = {df[var].unique()[0]: 1, df[var].unique()[1]: 2}
      df[var] = df[var].map(meta_data.categorical_var_map[var])
    elif len(df[var].unique()) > 2:
      raise ValueError('Categorical variables with more than 2 categories are currently not supported.')
    
  # Prepare parameters
  if mdl_type == 'SVM Classification':
    meta_data.pos_group = pos_group
    to_predict_input = [to_predict, [a for a in df[to_predict].unique() if a != pos_group] + [pos_group]]
    param_grid = {'linear':{'C': _expspace([-9, 5])},
                  'rbf':{'C': _expspace([-9, 5]), 'gamma': _expspace([-5, 5])}}[kernel]
  elif mdl_type == 'SVM Regression':
    to_predict_input = to_predict
    param_grid = {'C': _expspace([-5, 5]), 'epsilon': _expspace([-5, 5])}

  # Train model
  n_repeats = {'SVM Classification': 5, 'SVM Regression': 1}
  if len(df.index) > 1000:
    logging.info('Due to large dataset, first performing parameter tuning with 500 randomly sampled data points.')
    _, _, _, params, _ = run_SVM(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors,
              to_predict_input, param_grid=param_grid, kernel=kernel, n_repeats=1, verbose=0)
    param_grid = {par: _expspace([np.min(params[f'{par}_optimal']), np.max(params[f'{par}_optimal'])]) for par in param_grid}
  logging.info(f'Training {mdl_type} model...')
  df['predicted'], mdl, meta_data.stats, meta_data.params, meta_data.cv_folds = run_SVM(df, predictors,
              to_predict_input, param_grid=param_grid, kernel=kernel, n_repeats=n_repeats[mdl_type], verbose=verbose)
  meta_data.cv_results = df[list(dict.fromkeys(['ID', 'Age', 'Sex', to_predict, 'predicted']))]
  
  # Save model
  if save_path is not None:
    save_path = save_path.rstrip('.pkl.gz') + '.pkl.gz'
    with gzip.open(save_path, 'wb') as f:
      pickle.dump((mdl, vars(meta_data)), f)
      logging.info(f'Model saved to {save_path}')

  return mdl, vars(meta_data)

def spare_test(df: Union[pd.DataFrame, str],
               mdl_path: Union[str, Tuple[dict, dict]],
               verbose: int = 1) -> pd.DataFrame:
  """Applies a trained SPARE model on a test dataset

  Args:
    df: either a pandas dataframe or a path to a saved csv containing the test sample.
    mdl_path: either a path to a saved SPARE model ('.pkl.gz' file extension expected) or
      a tuple of SPARE model and meta_data.

  Returns:
    a pandas dataframe containing predicted SPARE scores.
  """
  logging_basic_config(verbose)
  df = _load_df(df)

  # Load trained SPARE model
  mdl, meta_data = load_model(mdl_path) if isinstance(mdl_path, str) else mdl_path
  check_test(df, meta_data)

  # Convert categorical variables
  for var, map_dict in meta_data.get('categorical_var_map',{}).items():
    if not isinstance(map_dict, dict):
      continue
    if df[var].isin(map_dict.keys()).any():
      df[var] = df[var].map(map_dict)
    else:
      expected_var = list(map_dict.keys())
      return logging.error(f'Column "{var}" expected {expected_var}, but received {list(df[var].unique())}')

  # Output model description
  n = len(meta_data['cv_results'].index)
  a1 = int(np.floor(np.min((meta_data['cv_results']['Age']))))
  a2 = int(np.ceil(np.max((meta_data['cv_results']['Age']))))
  stats_metric = list(meta_data['stats'].keys())[0]
  stats = '{:.3f}'.format(np.mean(meta_data['stats'][stats_metric]))
  logging.info(f'Model Info: training N = {n} / ages = {a1} - {a2} / expected {stats_metric} = {stats}')
  
  # Calculate SPARE scores
  n_ensemble = len(mdl['scaler'])
  ss = np.zeros([len(df.index), n_ensemble])
  for i in range(n_ensemble):
    X = mdl['scaler'][i].transform(df[meta_data['predictors']])
    if meta_data['kernel'] == 'linear':
      ss[:, i] = np.sum(X * mdl['mdl'][i].coef_, axis=1) + mdl['mdl'][i].intercept_
    else:
      ss[:, i] = mdl['mdl'][i].decision_function(X)
    if meta_data['mdl_type'] == 'SVM Regression':
      ss[:, i] = (ss[:, i] - mdl['bias_correct']['int'][i]) / mdl['bias_correct']['slope'][i]
    if 'ID' in df.columns:
      ss[df['ID'].isin(meta_data['cv_results']['ID'].drop(meta_data['cv_folds'][i])), i] = np.nan
  ss_mean = np.nanmean(ss, axis=1)
  ss_mean[np.all(np.isnan(ss),axis=1)] = np.nan

  return pd.DataFrame(data={'SPARE_scores': ss_mean})

def _expspace(span: list):
  return np.exp(np.linspace(span[0], span[1], num=int(span[1])-int(span[0])+1))

def _load_df(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
  return pd.read_csv(df, low_memory=False) if isinstance(df, str) else df.copy()