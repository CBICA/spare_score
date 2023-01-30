import gzip
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Union
from dataclasses import dataclass
from spare_scores.svm import run_SVC, run_SVR
from spare_scores.data_prep import col_names, check_train, check_test

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class MetaData:
  """Stores training information on its paired SPARE model
  """
  spare_type: str
  kernel: str
  predictors: list
  to_predict: str

def spare_train(df: Union[pd.DataFrame, str], 
                predictors: list,
                to_predict: str,
                pos_group: str = '',
                kernel: str = 'linear',
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

  df = _load_df(df)
  col_id, col_age, col_sex = col_names(df)
  df, predictors, spare_type = check_train(df, predictors, to_predict, pos_group)
  if spare_type == 'classification':
    groups_to_classify = [a for a in df[to_predict].unique() if a != pos_group] + [pos_group]

  # Initiate SPARE model
  meta_data = MetaData(spare_type, kernel, predictors, to_predict)
  meta_data.n = len(df.index)
  meta_data.age_range = [np.min(df[col_age]), np.max(df[col_age])]

  # Convert categorical variables
  var_categorical = df[predictors].dtypes == np.object
  var_categorical = var_categorical[var_categorical].index
  meta_data.categorical_var_map = dict(zip(var_categorical, [None]*len(var_categorical)))
  for var in var_categorical:
    if len(df[var].unique()) == 2:
      meta_data.categorical_var_map[var] = {df[var].unique()[0]: 1, df[var].unique()[1]: 2}
      df[var] = df[var].map(meta_data.categorical_var_map[var])
    elif len(df[var].unique()) > 2:
      return logging.error('Categorical variables with more than 2 categories are currently not supported.')

  # SPARE classification
  if spare_type == 'classification':
    if kernel == 'linear':
      meta_data.pos_group = pos_group
      param_grid = {'C': _expspace([-9, 5])}
    elif kernel == 'rbf':
      param_grid = {'C': _expspace([-9, 5]), 'gamma': _expspace([-5, 5])}
    if len(df.index) > 1000:
      _, _, _, params = run_SVC(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors,
                to_predict, groups_to_classify, param_grid=param_grid, kernel=kernel, n_repeats=1, verbose=0)
      for par in param_grid.keys():
        param_grid[par] = _expspace([np.min(params[f'{par}_optimal']), np.max(params[f'{par}_optimal'])])
    df['predicted'], mdl, meta_data.auc, meta_data.params = run_SVC(
                df, predictors, to_predict, groups_to_classify, param_grid=param_grid, kernel=kernel)

  # SPARE regression
  elif spare_type == 'regression':
    param_grid = {'C': _expspace([-5, 5]), 'epsilon': _expspace([-5, 5])}
    if len(df.index) > 1000:
      _, _, _, params = run_SVR(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors,
                to_predict, param_grid=param_grid, n_repeats=1, verbose=0)
      for par in param_grid.keys():
        param_grid[par] = _expspace([np.min(params[f'{par}_optimal']), np.max(params[f'{par}_optimal'])])
    df['predicted'], mdl, meta_data.mae, meta_data.params = run_SVR(
                  df, predictors, to_predict, param_grid=param_grid)
  meta_data.cv_results = df[list(dict.fromkeys([col_id, col_age, col_sex, to_predict, 'predicted']))]

  # Save model
  if save_path is not None:
    if ~save_path.endswith('.pkl.gz'):
      save_path = save_path + '.pkl.gz'
    with gzip.open(save_path, 'wb') as f:
      pickle.dump((mdl, vars(meta_data)), f)
      logging.info(f'Model saved to {save_path}')

  return mdl, vars(meta_data)

def load_model(mdl_path: str) -> Tuple[dict, dict]:
  with gzip.open(mdl_path, 'rb') as f:
    return pickle.load(f)

def spare_test(df: Union[pd.DataFrame, str],
               mdl_path: Union[str, Tuple[dict, dict]]) -> pd.DataFrame:

  """Applies a trained SPARE model on a test dataset

  Args:
    df: either a pandas dataframe or a path to a saved csv containing the test sample.
    mdl_path: either a path to a saved SPARE model ('.pkl.gz' file extension expected) or
      a tuple of SPARE model and meta_data.

  Returns:
    a pandas dataframe containing predicted SPARE scores.
  """

  df = _load_df(df)

  # Load trained SPARE model
  if isinstance(mdl_path, str):
    mdl, meta_data = load_model(mdl_path)
  else:
    mdl, meta_data = mdl_path
    
  check_test(df, meta_data)

  # Convert categorical variables
  for var in meta_data.get('categorical_var_map',{}).keys():
    if not isinstance(meta_data['categorical_var_map'][var], dict):
      continue
    if np.all(df[var].isin(meta_data['categorical_var_map'][var].keys())):
      df[var] = df[var].map(meta_data['categorical_var_map'][var])
    else:
      expected_var = list(meta_data['categorical_var_map'][var].keys())
      return logging.error(f'Column "{var}" expected {expected_var}, but received {list(df[var].unique())}')

  # Output model description
  print('Model Info: training N =', meta_data['n'], end=' / ')
  print('ages =', int(meta_data['age_range'][0]), '-', int(meta_data['age_range'][1]), end=' / ')
  if meta_data['spare_type'] == 'classification':
    print('expected AUC =', np.round(np.mean(meta_data['auc']), 3))
  elif meta_data['spare_type'] == 'regression':
    print('expected MAE =', np.round(np.mean(meta_data['mae']), 3))

  # Calculate SPARE scores
  n_ensemble = len(mdl['scaler'])
  ss, ss_mean = np.zeros([len(df.index), n_ensemble]), np.zeros([len(df.index), ])
  for i in range(n_ensemble):
    X = mdl['scaler'][i].transform(df[meta_data['predictors']])
    if meta_data['kernel'] == 'linear':
      ss[:, i] = np.sum(X * mdl['mdl'][i].coef_, axis=1) + mdl['mdl'][i].intercept_
    else:
      ss[:, i] = mdl['mdl'][i].decision_function(X)
    if meta_data['spare_type'] == 'regression':
      ss[:, i] = (ss[:, i] - mdl['bias_correct']['int'][i]) / mdl['bias_correct']['slope'][i]
    ss[df[col_names(df,['ID'])].isin(
      meta_data['cv_results'][col_names(meta_data['cv_results'],['ID'])][mdl['cv_folds'][i][0]]), i] = np.nan
  ss_mean = np.nanmean(ss, axis=1)
  ss_mean[np.all(np.isnan(ss),axis=1)] = np.nan

  return pd.DataFrame(data={'SPARE_scores': ss_mean})

def _expspace(span: list):
  return np.exp(np.linspace(span[0], span[1], num=int(span[1])-int(span[0])+1))

def _load_df(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
  if isinstance(df, str):
    return pd.read_csv(df, low_memory=False)
  else:
    return df.copy()