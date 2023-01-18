import numpy as np

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def train_initialize(df, k, n_repeats, param_grid):
  folds = [(train, test) for (train, test) in RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=2022).split(df['PTID'])]
  scaler = [StandardScaler()] * len(folds)
  params = param_grid.copy()
  for par in param_grid.keys():
    params[f'{par}_optimal'] = np.zeros((len(folds),))
  stat, y_hat = np.zeros((len(folds),)), np.zeros((len(df.index),))
  return folds, scaler, params, stat, y_hat

def prepare_sample(df, fold, predictors, to_predict, scaler, classify=None):
  X_train = df.loc[fold[0], predictors]
  X_test = df.loc[fold[1], predictors]
  scaler.fit(X_train)
  X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
  y_train, y_test = df.loc[fold[0], to_predict], df.loc[fold[1], to_predict]
  if classify is not None:
    y_train = y_train.map({classify[0]: -1, classify[1]: 1})
    y_test = y_test.map({classify[0]: -1, classify[1]: 1})
  return X_train, X_test, y_train, y_test

@ignore_warnings(category=ConvergenceWarning)
def run_SVC(df, predictors, to_predict, classify, param_grid, kernel='linear', k=5, n_repeats=5, verbose=1):

  folds, scaler, params, auc, y_hat = train_initialize(df, k, n_repeats, param_grid)
  mdl = [SVC(max_iter=100000, kernel=kernel)] * len(folds)

  for i, fold in enumerate(folds):
    if (i % n_repeats == 0) & (verbose==1):
      print(f'  FOLD {int(i/n_repeats+1)}...')

    X_train, X_test, y_train, y_test = prepare_sample(df, fold, predictors, to_predict, scaler[i], classify=classify)
    gs = GridSearchCV(mdl[i], param_grid, scoring='roc_auc', cv=k, return_train_score=True, verbose=0)
    gs.fit(X_train, y_train)
    mdl[i] = gs.best_estimator_
    fpr, tpr, _ = metrics.roc_curve(y_test, mdl[i].decision_function(X_test), pos_label=1)
    auc[i] = metrics.auc(fpr, tpr)
    for par in param_grid.keys():
      params[f'{par}_optimal'][i] = np.round(np.log(gs.best_params_[par]), 0)

    mdl[i].fit(X_train, y_train)
    y_hat[fold[1]] = mdl[i].decision_function(X_test)
    
  if verbose==1:
    print('>> AUC =', np.round(np.mean(auc), 3), '+/-', np.round(np.std(auc), 3))
  
  return y_hat, {'mdl':mdl, 'scaler':scaler, 'cv_folds':folds}, auc, params

@ignore_warnings(category=ConvergenceWarning)
def run_SVR(df, predictors, to_predict, param_grid, k=5, n_repeats=1, verbose=1):
  
  folds, scaler, params, mae, y_hat = train_initialize(df, k, n_repeats, param_grid)
  mdl = [LinearSVR(max_iter=100000)] * len(folds)
  bias_correct = {'slope':np.zeros((len(folds),)), 'int':np.zeros((len(folds),))}

  for i, fold in enumerate(folds):
    if verbose==1:
      print(f'  FOLD {int(i+1)}')
    
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

  if verbose==1:
    print('>> MAE =', np.round(np.mean(mae), 3), '+/-', np.round(np.std(mae), 3))

  return y_hat, {'mdl':mdl, 'scaler':scaler, 'bias_correct':bias_correct, 'cv_folds':folds}, mae, params
