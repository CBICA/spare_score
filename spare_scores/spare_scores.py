import os
import gzip
import pickle
import logging
import numpy as np
import pandas as pd
from .svm import run_SVC, run_SVR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_model(mdl_path: str):
  with gzip.open(mdl_path, 'rb') as f:
    return pickle.load(f)

def spare_train(df,
                predictors: list,
                to_predict: str,
                pos_group: str = '',
                kernel: str = 'linear',
                save_mdl: bool = True,
                out_path: str = './Mdl',
                mdl_name: str = ''):

  def _expspace(span: list):
    return np.exp(np.linspace(span[0], span[1], num=span[1]-span[0]+1))

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
      logging.warn('At least one of the groups to classify may be too small to build a robust SPARE classification model (n<100).')
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
      logging.warn('Sample size may be too small to build a robust SPARE regression model (n<100).')
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
    logging.info('Some participants have invalid predictor variables (such as n/a). They will be excluded from the training set.')
    df = df.loc[np.sum(pd.isna(df[predictors]), axis=1) == 0].reset_index(drop=True)
  #########################################

  # Initiate SPARE model
  metaData = {'spare_type': spare_type,
              'n': len(df.index),
              'age_range': np.floor([np.min(df['Age']), np.max(df['Age'])]),
              'to_predict': to_predict,
              'predictors': predictors}

  mdl = {'mdl': None,
         'scaler': None,
         'kernel': kernel}

  # Convert categorical variables
  var_categorical = df[predictors].dtypes == np.object
  var_categorical = var_categorical[var_categorical].index
  metaData['categorical_var_map'] = dict(zip(var_categorical, [None]*len(var_categorical)))
  for var in var_categorical:
    if len(df[var].unique()) == 2:
      metaData['categorical_var_map'][var] = {df[var].unique()[0]: 1, df[var].unique()[1]: 2}
      df[var] = df[var].map(metaData['categorical_var_map'][var])

  # SPARE classification
  if spare_type == 'classification':
    if kernel == 'linear':
      metaData['C_search'], metaData['pos_group'] = [-9, 5], pos_group
      if len(df.index) > 1000:
        _, _, _, _, C = run_SVC(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors, to_predict, groups_to_classify,
                                param_grid={'C': _expspace(metaData['C_search'])}, kernel=kernel, n_repeats=1, verbose=0)
        metaData['C_search'] = [int(np.min(C)), int(np.max(C))]
      df['predicted'], mdl['mdl'], mdl['scaler'], metaData['auc'], metaData['C_collect'] = run_SVC(
                  df, predictors, to_predict, groups_to_classify, param_grid={'C': _expspace(metaData['C_search'])}, kernel=kernel)
    elif kernel == 'rbf':
      metaData['C_search'], metaData['g_search'] = [-9, 5], [-5, 5]
      if len(df.index) > 1000:
        _, _, _, _, (C, g) = run_SVC(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors, to_predict, groups_to_classify,
                                     param_grid={'C': _expspace(metaData['C_search']), 'gamma': _expspace(metaData['g_search'])}, kernel=kernel, n_repeats=1, verbose=0)
        metaData['C_search'], metaData['g_search'] = [int(np.min(C)), int(np.max(C))], [int(np.min(g)), int(np.max(g))]
      df['predicted'], mdl['mdl'], mdl['scaler'], metaData['auc'], (metaData['C_collect'], metaData['g_collect']) = run_SVC(
                  df, predictors, to_predict, groups_to_classify, param_grid={'C': _expspace(metaData['C_search']), 'gamma': _expspace(metaData['g_search'])}, kernel=kernel)
  # SPARE regression
  elif spare_type == 'regression':
    metaData['C_search'], metaData['e_search'] = [-5, 5], [-5, 5]
    if len(df.index) > 1000:
      _, _, _, _, _, (C, e) = run_SVR(df.sample(n=500, random_state=2022).reset_index(drop=True), predictors, to_predict,
                                      param_grid={'C': _expspace(metaData['C_search']), 'epsilon': _expspace(metaData['e_search'])}, verbose=0)
      metaData['C_search'], metaData['e_search'] = [int(np.min(C)), int(np.max(C))], [int(np.min(e)), int(np.max(e))]
    df['predicted'], mdl['mdl'], mdl['scaler'], mdl['bias_correct'], metaData['mae'], (metaData['C_collect'], metaData['e_collect']) = run_SVR(
                  df, predictors, to_predict, param_grid={'C': _expspace(metaData['C_search']), 'epsilon': _expspace(metaData['e_search'])})

  metaData['cv_results'] = df[list(dict.fromkeys(['PTID', 'Age', 'Sex', to_predict, 'predicted']))]

  # Save model
  if save_mdl:
    if mdl_name == '':
      to_predict_ = to_predict.replace('.', '_')
      mdl_name = f'SPARE_{spare_type}_{to_predict_}'
    with gzip.open(f'{out_path}/mdl_{mdl_name}.pkl.gz', 'wb') as f:
      pickle.dump((mdl, metaData), f)

  return mdl, metaData


def spare_test(df,
               mdl_path: str,
               save_csv: bool = False,
               out_path: str = './Out'):

  # Load trained SPARE model
  mdl, metaData = load_model(mdl_path)

  ################ FILTERS ################
  if not set(metaData['predictors']).issubset(df.columns):
    print(set(metaData['predictors']) - set(df.columns))
    return logging.error('Not all predictors exist in the input dataframe.')

  if (np.min(df['Age']) < metaData['age_range'][0]) or (np.max(df['Age']) > metaData['age_range'][1]):
    logging.warn('Some participants fall outside of the age range of the SPARE model.')

  if np.sum(np.sum(pd.isna(df[metaData['predictors']]))) > 0:
    logging.warn('Some participants have invalid predictor variables.')
  #########################################

  # Output model description
  print('Trained on', metaData['n'], 'individuals ', end='/ ')
  print('Ages from', int(metaData['age_range'][0]), 'and', int(metaData['age_range'][1]), end=' / ')
  if metaData['spare_type'] == 'classification':
    print('Expected AUC =', np.round(np.mean(metaData['auc']), 3))
  elif metaData['spare_type'] == 'regression':
    print('Expected MAE =', np.round(np.mean(metaData['mae']), 3))

  # Convert categorical variables
  df = df.copy()
  if 'categorical_var_map' in metaData.keys():
    for var in metaData['categorical_var_map'].keys():
      if isinstance(metaData['categorical_var_map'][var], dict):
        df[var] = df[var].map(metaData['categorical_var_map'][var])

  # Calculate SPARE scores
  n_ensemble = len(mdl['scaler'])
  ss = np.zeros([len(df.index), n_ensemble])
  for i in range(n_ensemble):
    X = mdl['scaler'][i].transform(df[metaData['predictors']])
    if mdl['kernel'] == 'linear':
      ss[:, i] = np.sum(X * mdl['mdl'][i].coef_, axis=1) + mdl['mdl'][i].intercept_
    else:
      ss[:, i] = mdl['mdl'][i].decision_function(X)
    if metaData['spare_type'] == 'regression':
      ss[:, i] = (ss[:, i] - mdl['bias_correct']['int'][i]) / mdl['bias_correct']['slope'][i]

  df_results = pd.DataFrame(data={'SPARE_scores': np.mean(ss, axis=1)})

  # Save results csv
  if save_csv:
    mdl_name = mdl_path.split('/')[-1].split('.')[0]
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    df_results.to_csv(f'{out_path}/SPAREs_from_{mdl_name}.csv')

  return df_results
