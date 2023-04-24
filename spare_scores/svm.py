import logging
import numpy as np
from sklearn import metrics
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from spare_scores.data_prep import logging_basic_config

def train_initialize(df, k, n_repeats, param_grid):
  id_unique = df['ID'].unique()
  folds = list(RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=2022).split(id_unique))
  if len(id_unique) < len(df):
    folds = [[np.array(df.index[df['ID'].isin(id_unique[a])]) for a in folds[b]] for b in range(len(folds))]
  scaler = [StandardScaler()] * len(folds)
  params = param_grid.copy()
  params.update({f'{par}_optimal': np.zeros(len(folds)) for par in param_grid.keys()})
  stat, y_hat = np.zeros(len(folds)), np.zeros(len(df))
  return folds, scaler, params, stat, y_hat

def prepare_sample(df, fold, predictors, to_predict, scaler, classify=None):
  X_train, X_test = scaler.fit_transform(df.loc[fold[0], predictors]), scaler.transform(df.loc[fold[1], predictors])
  y_train, y_test = df.loc[fold[0], to_predict], df.loc[fold[1], to_predict]
  if classify is not None:
    y_train, y_test = y_train.map(dict(zip(classify, [-1, 1]))), y_test.map(dict(zip(classify, [-1, 1])))
  return X_train, X_test, y_train, y_test

@ignore_warnings(category=ConvergenceWarning)
def run_SVC(df, predictors, to_predict, param_grid, kernel='linear', k=5, n_repeats=5, verbose=1):

  logging_basic_config(verbose, content_only=True)
  folds, scaler, params, auc, y_hat = train_initialize(df, k, n_repeats, param_grid)
  if kernel == 'linear':
    mdl = [LinearSVC(max_iter=100000)] * len(folds)
  else:
    mdl = [SVC(max_iter=100000, kernel=kernel)] * len(folds)

  for i, fold in enumerate(folds):
    if i % n_repeats == 0: 
      logging.info(f'  FOLD {int(i/n_repeats+1)}...')

    X_train, X_test, y_train, y_test = prepare_sample(df, fold, predictors, to_predict[0], scaler[i], classify=to_predict[1])
    gs = GridSearchCV(mdl[i], param_grid, scoring='roc_auc', cv=k, return_train_score=True, verbose=0)
    gs.fit(X_train, y_train)
    mdl[i] = gs.best_estimator_
    fpr, tpr, _ = metrics.roc_curve(y_test, mdl[i].decision_function(X_test), pos_label=1)
    auc[i] = metrics.auc(fpr, tpr)
    for par in param_grid.keys():
      params[f'{par}_optimal'][i] = np.round(np.log(gs.best_params_[par]), 0)

    mdl[i].fit(X_train, y_train)
    y_hat[fold[1]] = mdl[i].decision_function(X_test)
    
  logging.info(f'>> AUC = {np.mean(auc):.3f} \u00B1 {np.std(auc):.3f}')
  
  return y_hat, {'mdl':mdl, 'scaler':scaler}, {'AUC':auc}, params, [a[1] for a in folds]

@ignore_warnings(category=ConvergenceWarning)
def run_SVR(df, predictors, to_predict, param_grid, kernel='linear', k=5, n_repeats=1, verbose=1):
  
  logging_basic_config(verbose, content_only=True)
  folds, scaler, params, mae, y_hat = train_initialize(df, k, n_repeats, param_grid)
  mdl = [LinearSVR(max_iter=100000)] * len(folds)
  bias_correct = {'slope':np.zeros((len(folds),)), 'int':np.zeros((len(folds),))}

  for i, fold in enumerate(folds):
    logging.info(f'  FOLD {int(i+1)}')
    
    X_train, X_test, y_train, y_test = prepare_sample(df, fold, predictors, to_predict, scaler[i])
    gs = GridSearchCV(mdl[i], param_grid, scoring='neg_mean_absolute_error', cv=k, return_train_score=True, verbose=0)
    gs.fit(X_train, y_train)
    mdl[i] = gs.best_estimator_
    mae[i] = -gs.best_score_
    for par in param_grid.keys():
      params[f'{par}_optimal'][i] = np.round(np.log(gs.best_params_[par]), 0)
    
    mdl[i].fit(X_train, y_train)
    y_hat[fold[1]] = mdl[i].predict(X_test)
    
    # Linear bias correction
    bias_correct['slope'][i], bias_correct['int'][i] = np.polyfit(y_test, y_hat[fold[1]], 1)
    if bias_correct['slope'][i] != 0:
      y_hat[fold[1]] = (y_hat[fold[1]] - bias_correct['int'][i]) / bias_correct['slope'][i]

  logging.info(f'>> MAE = {np.mean(mae):.3f} \u00B1 {np.std(mae):.3f}')

  return y_hat, {'mdl':mdl, 'scaler':scaler, 'bias_correct':bias_correct}, {'MAE':mae}, params, [a[1] for a in folds]