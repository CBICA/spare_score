import numpy as np

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def prepare_sample(df_train, fold, predictors, scaler):
  X_train = df_train.loc[fold[0], predictors]
  X_test = df_train.loc[fold[1], predictors]
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  return X_train, X_test

@ignore_warnings(category=ConvergenceWarning)
def run_SVC(df, predictors, to_predict, classify, param_grid, kernel='linear', k=5, n_repeats=5, verbose=1):

  folds = [(train, test) for (train, test) in RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=2022).split(df['PTID'])]
  if kernel=='linear':
    mdl = [LinearSVC(max_iter=100000)] * len(folds)
  elif kernel=='rbf':
    mdl = [SVC(max_iter=100000)] * len(folds)
    g_collect = np.zeros((len(folds),))
  scaler = [StandardScaler()] * len(folds)

  auc, C_collect, y_hat = np.zeros((len(folds),)), np.zeros((len(folds),)), np.zeros((len(df.index),))
  for i, fold in enumerate(folds):
    if (i % n_repeats == 0) & (verbose==1):
      print(f'  FOLD {int(i/n_repeats+1)}...')

    X_train, X_test = prepare_sample(df, fold, predictors, scaler[i])
    y_train = df.loc[fold[0], to_predict].map({classify[0]: -1, classify[1]: 1})
    y_test = df.loc[fold[1], to_predict].map({classify[0]: -1, classify[1]: 1})

    gs = GridSearchCV(mdl[i], param_grid, scoring='roc_auc', cv=k, return_train_score=True, verbose=0)
    gs.fit(X_train, y_train)

    mdl[i] = gs.best_estimator_
    fpr, tpr, _ = metrics.roc_curve(y_test, mdl[i].decision_function(X_test), pos_label=1)
    auc[i] = metrics.auc(fpr, tpr)

    bp = gs.best_params_
    C_collect[i] = np.round(np.log(bp['C']), 0)
    if kernel=='rbf':
      g_collect[i] = np.round(np.log(bp['gamma']), 0)

    mdl[i].fit(X_train, y_train)
    y_hat[fold[1]] = mdl[i].decision_function(X_test)
    
  if verbose==1:
    print('>> AUC =', np.round(np.mean(auc), 3), '+/-', np.round(np.std(auc), 3))
  
  if kernel=='linear':
    return y_hat, mdl, scaler, auc, C_collect
  elif kernel=='rbf':
    return y_hat, mdl, scaler, auc, (C_collect, g_collect)

@ignore_warnings(category=ConvergenceWarning)
def run_SVR(df, predictors, to_predict, param_grid, score='neg_mean_absolute_error', k=5, verbose=1):
  
  folds = [(train, test) for (train, test) in KFold(n_splits=k, shuffle=True, random_state=2022).split(df['PTID'])]
  mdl = [LinearSVR(max_iter=100000)] * len(folds)
  scaler = [StandardScaler()] * len(folds)

  mae, C_collect, e_collect = np.zeros((len(folds),)), np.zeros((len(folds),)), np.zeros((len(folds),))
  bias_correct = {'slope':np.zeros((len(folds),)),
                  'int':np.zeros((len(folds),))}
  y_hat = np.zeros((len(df.index),))
  for i, fold in enumerate(folds):
    if verbose==1:
      print(f'  FOLD {int(i+1)}')
    gs = GridSearchCV(mdl[i], param_grid, scoring=score, cv=5, return_train_score=True, verbose=0)
    X_train, X_test = prepare_sample(df, fold, predictors, scaler[i])
    y_train, y_test = df.loc[fold[0], to_predict], df.loc[fold[1], to_predict]
    gs.fit(X_train, y_train)

    mdl[i] = gs.best_estimator_

    bp = gs.best_params_
    C_collect[i] = np.round(np.log(bp['C']), 0)
    e_collect[i] = np.round(np.log(bp['epsilon']), 0)
    mae[i] = -gs.best_score_

    mdl[i].fit(X_train, y_train)
    y_hat[fold[1]] = mdl[i].predict(X_test)
    
    # Linear bias correction
    bias_correct['slope'][i], bias_correct['int'][i] = np.polyfit(y_test, y_hat[fold[1]], 1)
    if bias_correct['slope'][i] != 0:
      y_hat[fold[1]] = (y_hat[fold[1]] - bias_correct['int'][i]) / bias_correct['slope'][i]
  if verbose==1:
    print('>> MAE =', np.round(np.mean(mae), 3), '+/-', np.round(np.std(mae), 3))

  return y_hat, mdl, scaler, bias_correct, mae, (C_collect, e_collect)
