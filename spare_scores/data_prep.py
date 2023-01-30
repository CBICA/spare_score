import random
import logging
import numpy as np
import pandas as pd

from scipy import stats
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
def col_names(df: pd.DataFrame,
              cols: list=['ID','Age','Sex']) -> tuple:
  """Matches required column names with common name variants. 

  Args:
    df: pandas dataframe.
    cols: columns to search for name variants.

  Returns:
    a string or a tuple with matched variants.
  """
  col_name_variants = {'ID':['ID','id','PTID','participant_id'],
                       'Age':['Age','age','AGE'],
                       'Sex':['Sex','sex','SEX']}
  col_name_variants = {a: col_name_variants[a] for a in cols}
  for k in col_name_variants.keys():
    if k not in cols:
      continue
    for i in col_name_variants[k]:
      if i in df.columns:
        col_name_variants[k] = i
        break
  col_not_found = [a for a in col_name_variants.keys() if type(col_name_variants[a])==list]
  if len(col_not_found) > 0:
    return logging.error(f'Required columns not found: {col_not_found}')
  if len(cols) == 1:
    return col_name_variants[cols[0]]
  else: 
    return tuple(col_name_variants.values())

def check_train(df: pd.DataFrame, 
                predictors: list,
                to_predict: str,
                pos_group: str = '') -> Tuple[pd.DataFrame, list, str]:
  """Checks training dataframe for errors.

  Args:
    df: a pandas dataframe containing training data.
    predictors: a list of predictors for SPARE model training.
    to_predict: variable to predict.
    pos_group: group to assign a positive SPARE score (only for classification).

  Returns:
    a tuple containing 1) filtered dataframe, 2) filtered predictors, 3) SPARE model type.
  """
  if not set(predictors).issubset(df.columns):
    return logging.error('Not all predictors exist in the input dataframe.')
  if to_predict not in df.columns:
    return logging.error('Variable to predict is not in the input dataframe.')
  if to_predict in predictors:
    logging.info('Variable to predict is in the predictor set. This will be removed from the set.')
    predictors.remove(to_predict)
  if np.sum(np.sum(pd.isna(df[predictors]))) > 0:
    logging.info('Some participants have invalid predictor variables (i.e. n/a). They will be excluded.')
    df = df.loc[np.sum(pd.isna(df[predictors]), axis=1) == 0].reset_index(drop=True)

  col_id = col_names(df,['ID'])
  if len(df[to_predict].unique()) == 2:
    if pos_group == '':
      return logging.error('"pos_group" not provided (group to assign a positive score).')
    elif pos_group not in df[to_predict].unique():
      return logging.error('"pos_group" is not one of the two groups in the variable to predict.')
    if np.min(df[to_predict].value_counts()) < 10:
      return logging.error('At least one of the groups to classify is too small (n<10).')
    elif np.min(df[to_predict].value_counts()) < 100:
      logging.warn('At least one of the groups to classify may be too small (n<100).')
    if np.sum((df[col_id]+df[to_predict]).duplicated()) > 0:
      logging.warn('Training dataset has duplicate participants.')
    spare_type = 'classification'

  elif len(df[to_predict].unique()) > 2:
    if df[to_predict].dtype not in ['int64', 'float64']:
      return logging.error('Variable to predict must be either binary or numeric.')
    if len(df.index) < 10:
      return logging.error('Sample size is too small (n<10).')
    elif len(df.index) < 100:
      logging.warn('Sample size may be too small (n<100).')
    if np.sum(df[col_id].duplicated()) > 0:
      logging.warn('Training dataset has duplicate participants.')
    if pos_group != '':
      logging.info('SPARE regression does not need a "pos_group". This will be ignored.')
    spare_type = 'regression'
  else:
    return logging.error('Variable to predict has no variance.')
    
  logging.info(f'Dataframe checked for SPARE {spare_type} training.')
  return df, predictors, spare_type

def check_test(df: pd.DataFrame, 
               meta_data: dict):
  """Checks testing dataframe for errors.

  Args:
    df: a pandas dataframe containing testing data.
    meta_data: a dictionary containing training information on its paired SPARE model.
  """
  col_id, col_age = col_names(df,['ID','Age'])
  if not set(meta_data['predictors']).issubset(df.columns):
    cols_not_found = sorted(set(meta_data['predictors']) - set(df.columns))
    return logging.error(f'Not all predictors exist in the input dataframe: {cols_not_found}')
    
  if (np.min(df[col_age]) < meta_data['age_range'][0]) or (
           np.max(df[col_age]) > meta_data['age_range'][1]):
    logging.warn('Some participants fall outside the age range of the SPARE model.')

  if np.sum(np.sum(pd.isna(df[meta_data['predictors']]))) > 0:
    logging.warn('Some participants have invalid predictor variables.')

  if np.any(df[col_id].isin(meta_data['cv_results'][col_id])):
    logging.info('Some participants seem to have been in the model training.')

def smart_unique(df1: pd.DataFrame,
                 df2: pd.DataFrame=None,
                 to_predict: str=None,
                 verbose: int=1) -> Union[pd.DataFrame, tuple]:
  """Select unique data points in a way that optimizes SPARE training.
  For SPARE regression, preserve data points with extreme values.
  For SPARE classification, preserve data points that help age match.

  Args:
    df1: a pandas dataframe.
    df2: a pandas dataframe (optional) if df1 and df2 are two groups to classify.
    to_predict: variable to predict. Binary for classification and continuous for regression.
      Must be one of the columnes in df. Ignored if df2 is given.

  Returns:
    a trimmed pandas dataframe or a tuple of two dataframes with only one time point per ID.
  """
  assert (isinstance(df2, pd.DataFrame) or (df2 is None)), (
    'Either provide a 2nd pandas dataframe for the 2nd argument or specify it with "to_predict"')
  if verbose == 0:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s', force=True)
  col_id = col_names(df1, cols=['ID'])
  if df2 is None:
    if to_predict is None:
      return logging.error('Either provide a second dataframe or provide a column "to_predict"')
    elif len(df1[to_predict].unique()) > 2:
      if ~np.any(df1[col_id].duplicated()):
        logging.info('No duplicated IDs.')
      else:
        logging.info('Select unique time points for SPARE regression training.')
        df1[f'{to_predict}_from_mean'] = np.abs(df1[to_predict] - np.mean(df1[to_predict]))
        df1 = df1[df1.groupby(col_id)[f'{to_predict}_from_mean'].transform(
                          max) == df1[f'{to_predict}_from_mean']].reset_index(drop=True)
        df1 = df1.drop(columns=f'{to_predict}_from_mean')
        df1 = df1[~df1[col_id].duplicated()].reset_index(drop=True)
      return df1
    elif len(df1[to_predict].unique()) < 2:
      return logging.error('Variable to predict has no variance.')
    if ~np.any((df1[col_id].astype(str) + df1[to_predict].astype(str)).duplicated()):
      logging.info('No duplicated IDs in either group.')
      return df1
    grps = list(df1[to_predict].unique())
    df1, df2 = df1[df1[to_predict] == grps[0]], df1[df1[to_predict] == grps[1]]
    no_df2 = True
  else:
    if to_predict is not None:
      logging.info('"to_predict" will be ignored.')
    if (~np.any(df1[col_id].duplicated())) and (~np.any(df2[col_id].duplicated())):
      logging.info('No duplicated IDs in either group.')
      return (df1, df2)
    no_df2 = False

  logging.info('Select unique time points for SPARE classification training.')
  col_age = col_names(df1, cols=['Age'])
  swap = False
  if stats.ttest_ind(df1[col_age], df2[col_age]).pvalue < 0.05:
    if np.mean(df1[col_age]) < np.mean(df2[col_age]):
        df1, df2, swap = df2.copy(), df1.copy(), True
    df2 = df2.loc[df2[col_age] >= np.min(df1[col_age])].reset_index(drop=True)
    df1 = df1[df1.groupby(col_id)[col_age].transform(min) == df1[col_age]].reset_index(drop=True)
    df2 = df2[df2.groupby(col_id)[col_age].transform(max) == df2[col_age]].reset_index(drop=True)
  else:
    logging.info('Age difference not significant between two groups.')
  df1 = df1[~df1[col_id].duplicated()].reset_index(drop=True)
  df2 = df2[~df2[col_id].duplicated()].reset_index(drop=True)
  if swap:
    df1, df2 = df2.copy(), df1.copy()
  if no_df2:
    return pd.concat([df1, df2], ignore_index=True)
  else:
    return (df1, df2)

def age_sex_match(df1: pd.DataFrame,
                  df2: pd.DataFrame = None,
                  to_match: str = None,
                  p_threshold: float = 0.15,
                  verbose: int = 1,
                  age_out_percentage: float = 20) -> pd.DataFrame:
  """Match two groups for age and sex.

  Args:
    df1: a pandas dataframe.
    df2: a pandas dataframe (optional) if df1 and df2 are two groups to classify.
    to_match: a binary variable of two groups. Must be one of the columns in df.
      Ignored if df2 is given.
    p_threshold: minimum p-value for matching.
    verbose: whether to output messages.
    age_out_percentage: percentage of the larger group to randomly select a participant to
      take out from during the age matching. For example, if age_out_percentage = 20 and the
      larger group is significantly older, then exclude one random participant from the fifth
      quintile based on age.

  Returns:
    a trimmed pandas dataframe or a tuple of two dataframes with age/sex matched groups.
  """
  assert (isinstance(df2, pd.DataFrame) or (df2 is None)), (
    'Either provide a 2nd pandas dataframe for the 2nd argument or specify it with "to_match"')
  if verbose == 0:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s', force=True)
  if df2 is None:
    if to_match is None:
      return logging.error('Either provide a second dataframe or provide a column "to_match"')
    if len(df1[to_match].unique()) != 2:
      return logging.error('Variable to match must be binary')
    grps = list(df1[to_match].unique())
    df1, df2 = df1[df1[to_match] == grps[0]], df1[df1[to_match] == grps[1]]
    no_df2 = True
  else:
    if to_match is not None:
      logging.info('"to_match" will be ignored.')
    no_df2 = False

  if (age_out_percentage <= 0) or (age_out_percentage >= 100):
    return logging.error('Age-out-percentage must be between 0 and 100')
 
  swap = 1
  random.seed(2022)
  n_orig = len(df1.index) + len(df2.index)
  col_age, col_sex = col_names(df1, ['Age', 'Sex'])
  s1, s2 = df1[col_sex].unique()

  p_age = stats.ttest_ind(df1[col_age], df2[col_age]).pvalue
  p_sex = stats.chi2_contingency([np.array(df1[col_sex].value_counts()), np.array(df2[col_sex].value_counts())])[1]
  if verbose > 1:
    print(f' Orig.: P_age: {np.round(p_age,2)}/ P_sex {np.round(p_sex,2)}')

  p_age_all, p_sex_all = np.array(p_age), np.array(p_sex)
  while np.min([p_age, p_sex]) < p_threshold:
    if len(df2.index) > len(df1.index):
      df1, df2 = df2.copy(), df1.copy()
      swap *= -1
    if p_age < p_threshold:
      if np.mean(df1[col_age]) < np.mean(df2[col_age]):
        i_age = df1[col_age] < np.percentile(df1[col_age], age_out_percentage)
      else:
        i_age = df1[col_age] > np.percentile(df1[col_age], 100-age_out_percentage)
    else:
      i_age = df1[col_age] >= 0
    if p_sex < p_threshold:
      if np.sum(df1[col_sex] == s1)/np.sum(df1[col_sex] == s2) > np.sum(df2[col_sex] == s1)/np.sum(df2[col_sex] == s2):
        i_sex = df1[col_sex] == s1
      else:
        i_sex = df1[col_sex] == s2
    else:
      i_sex = df1[col_sex].isin([s1, s2])

    try:
      df1 = df1.drop(random.sample(list(df1[i_age & i_sex].index), 1)).reset_index(drop=True)
    except:
      if np.min([len(df1.index), len(df2.index)]) > 10:
        print('Try increasing "age_out_percentage" parameter')
      return logging.error('Matching failed...')
    p_age = stats.ttest_ind(df1[col_age], df2[col_age]).pvalue
    p_sex = stats.chi2_contingency([np.array(df1[col_sex].value_counts()), np.array(df2[col_sex].value_counts())])[1]
    p_age_all = np.append(p_age_all, p_age)
    p_sex_all = np.append(p_sex_all, p_sex)
  if swap == -1:
    df1, df2 = df2.copy(), df1.copy()

  if verbose > 1:
    n_dropped = n_orig - len(df1.index) - len(df2.index)
    print(f' {n_dropped} participants excluded')
    print(f' Final: P_age: {np.round(p_age,2)}/ P_sex {np.round(p_sex,2)}')
  logging.info('Age/Sex matched!')
  if no_df2:
    return pd.concat([df1, df2], ignore_index=True)
  else:
    return (df1, df2)