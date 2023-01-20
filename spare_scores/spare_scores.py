import gzip
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from spare_scores.svm import run_SVC, run_SVR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class MetaData:
  spare_type: str
  kernel: str
  n: int
  age_range: list
  to_predict: str
  predictors: list
  
def spare_train(df: pd.DataFrame,
                predictors: list,
                to_predict: str,
                pos_group: str = '',
                kernel: str = 'linear',
                save_path: str = None,
                mdl_name: str = '') -> Tuple[dict, dict]:

  def _expspace(span: list):
    return np.exp(np.linspace(span[0], span[1], num=int(span[1])-int(span[0])+1))

  ################ FILTERS ################
  if not set(predictors).issubset(df.columns):
    return logging.error('Not all predictors exist in the input dataframe.')

  # Determine SPARE type
  if len(df[to_predict].unique()) == 2:
    if pos_group == '':
      return logging.error('"pos_group" not provided (group to assign a positive score).')
    elif pos_group not in df[to_predict].unique():
      return logging.error('"pos_group" does not match one of the two groups in variable to predict.')
    if np.min(df[to_predict].value_counts()) < 10:
      return logging.error('At least one of the groups to classify is too small (n<10).')
    elif np.min(df[to_predict].value_counts()) < 100:
      logging.warn('At least one of the groups to classify may be too small (n<100).')
    if np.sum((df['PTID']+df[to_predict]).duplicated()) > 0:
      logging.warn('Training dataset has duplicate participants.')
    spare_type = 'classification'
    groups_to_classify = [a for a in df[to_predict].unique() if a != pos_group] + [pos_group]
  elif len(df[to_predict].unique()) > 2:
    if df[to_predict].dtype not in ['int64', 'float64']:
      return logging.error('Variable to predict must be either binary or numeric.')
    if len(df.index) < 10:
      return logging.error('Sample size is too small (n<10).')
    elif len(df.index) < 100:
      logging.warn('Sample size may be too small (n<100).')
    if np.sum(df['PTID'].duplicated()) > 0:
      logging.warn('Training dataset has duplicate participants.')
    if pos_group != '':
      logging.info('SPARE regression model does not need a "pos_group". This will be ignored.')
    spare_type = 'regression'
  else:
    return logging.error('Variable to predict has no variance.')

  if to_predict in predictors:
    logging.info('Variable to predict is in the predictor set. This will be removed from the set.')
    predictors.remove(to_predict)
  if np.sum(np.sum(pd.isna(df[predictors]))) > 0:
    logging.info('Some participants have invalid predictor variables (such as n/a). They will be excluded from the training.')
    df = df.loc[np.sum(pd.isna(df[predictors]), axis=1) == 0].reset_index(drop=True)
  #########################################

  # Initiate SPARE model
  meta_data = MetaData(spare_type, kernel, len(df.index), [np.min(df['Age']), np.max(df['Age'])], to_predict, predictors)

  # Convert categorical variables
  var_categorical = df[predictors].dtypes == np.object
  var_categorical = var_categorical[var_categorical].index
  meta_data.categorical_var_map = dict(zip(var_categorical, [None]*len(var_categorical)))
  for var in var_categorical:
    if len(df[var].unique()) == 2:
      meta_data.categorical_var_map[var] = {df[var].unique()[0]: 1, df[var].unique()[1]: 2}
      df[var] = df[var].map(meta_data.categorical_var_map[var])

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
  meta_data.cv_results = df[list(dict.fromkeys(['PTID', 'Age', 'Sex', to_predict, 'predicted']))]

  # Save model
  if save_path is not None:
    if mdl_name == '':
      to_predict_ = to_predict.replace('.', '_')
      mdl_name = f'SPARE_{spare_type}_{to_predict_}'
    with gzip.open(f'{save_path}/mdl_{mdl_name}.pkl.gz', 'wb') as f:
      pickle.dump((mdl, vars(meta_data)), f)
      logging.info(f'Model saved to {save_path}/mdl_{mdl_name}.pkl.gz')

  return mdl, vars(meta_data)

def load_model(mdl_path: str):
  with gzip.open(mdl_path, 'rb') as f:
    return pickle.load(f)

def spare_test(df: pd.DataFrame,
               mdl_path: str) -> pd.DataFrame:

  # Load trained SPARE model
  mdl, meta_data = load_model(mdl_path)
  df = df.copy()

  ################ FILTERS ################
  if not set(meta_data['predictors']).issubset(df.columns):
    cols_not_found = sorted(set(meta_data['predictors']) - set(df.columns))
    assert len([a for a in cols_not_found if '_' not in a]) == 0, f'Not all predictors exist in the input dataframe: {cols_not_found}'
    try:
      roi_name = [a for a in meta_data['predictors'] if '_' in a]
      for roi_alter in [[int(a.split('_')[-1]) for a in roi_name],
                        [a.split('_')[-1] for a in roi_name],
                        ['R'+a.split('_')[-1] for a in roi_name]]:  
        if set(roi_alter).issubset(df.columns):
          df = df.rename(columns=dict(zip(roi_alter, roi_name)))
          logging.info(f'ROI names changed to match the model (e.g. {roi_alter[0]} to {roi_name[0]}).')
          continue
    except Exception:
      return logging.error(f'Not all predictors exist in the input dataframe: {cols_not_found}')
    cols_not_found = sorted(set(meta_data['predictors']) - set(df.columns))
    assert len(cols_not_found) == 0, f'Not all predictors exist in the input dataframe: {cols_not_found}'
  if (np.min(df['Age']) < meta_data['age_range'][0]) or (np.max(df['Age']) > meta_data['age_range'][1]):
    logging.warn('Some participants fall outside the age range of the SPARE model.')

  if np.sum(np.sum(pd.isna(df[meta_data['predictors']]))) > 0:
    logging.warn('Some participants have invalid predictor variables.')

  if np.any(df['PTID'].isin(meta_data['cv_results']['PTID'])):
    logging.info('Some participants seem to have been in the model training.')
  #########################################

  # Convert categorical variables
  for var in meta_data.get('categorical_var_map',{}).keys():
    if not isinstance(meta_data['categorical_var_map'][var], dict):
      continue
    if np.all(df[var].isin(meta_data['categorical_var_map'][var].keys())):
      df[var] = df[var].map(meta_data['categorical_var_map'][var])
    else:
      expected_var = list(meta_data['categorical_var_map'][var].keys())
      return logging.error(f'Column "{var}" contains value(s) other than expected: {expected_var}')

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
    ss[df['PTID'].isin(meta_data['cv_results']['PTID'][mdl['cv_folds'][i][0]]), i] = np.nan
  index_nan = np.all(np.isnan(ss),axis=1)
  ss_mean[~index_nan] = np.nanmean(ss[~index_nan,:], axis=1)

  return pd.DataFrame(data={'SPARE_scores': ss_mean})