import logging
import numpy as np
from sklearn import metrics
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from spare_scores.data_prep import logging_basic_config

class SVM_Model:
  def __init__(self, df, predictors, to_predict, param_grid, kernel, k, n_repeats):
    self.df = df
    self.predictors = predictors
    self.param_grid = param_grid
    self.kernel = kernel
    self.k = k
    self.n_repeats = n_repeats
    self.train_initialize(to_predict)

  def train_initialize(self, to_predict):
    id_unique = self.df['ID'].unique()
    self.folds = list(RepeatedKFold(n_splits=self.k, n_repeats=self.n_repeats, random_state=2022).split(id_unique))
    if len(id_unique) < len(self.df):
      self.folds = [[np.array(self.df.index[self.df['ID'].isin(id_unique[a])]) for a in self.folds[b]] for b in range(len(self.folds))]
    self.scaler = [StandardScaler()] * len(self.folds)
    self.params = self.param_grid.copy()
    self.params.update({f'{par}_optimal': np.zeros(len(self.folds)) for par in self.param_grid.keys()})
    self.y_hat = np.zeros(len(self.df))
    if isinstance(to_predict, list):
      self.type, self.scoring, metrics = 'SVC', 'roc_auc', ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
      self.to_predict, self.classify = to_predict[0], to_predict[1]
      self.mdl = ([LinearSVC(max_iter=100000)] if self.kernel == 'linear' else [SVC(max_iter=100000, kernel=self.kernel)]) * len(self.folds)
    else:
      self.type, self.scoring, metrics = 'SVR', 'neg_mean_absolute_error', ['MAE', 'RMSE', 'R2']
      self.to_predict, self.classify = to_predict, None
      self.mdl = [LinearSVR(max_iter=100000)] * len(self.folds)
      self.bias_correct = {'slope':np.zeros((len(self.folds),)), 'int':np.zeros((len(self.folds),))}
    self.stats = {metric: [] for metric in metrics}
    logging.info(f'Training a SPARE model ({self.type}) with {len(self.df.index)} participants')

  def run_CV(self):
    for i, fold in enumerate(self.folds):
      if i % self.n_repeats == 0:
        logging.info(f'  FOLD {int(i/self.n_repeats+1)}...')
      X_train, X_test, y_train, y_test = self.prepare_sample(fold, self.scaler[i], classify=self.classify)
      self.mdl[i] = self.param_search(self.mdl[i], X_train, y_train, scoring=self.scoring)
      for par in self.param_grid.keys():
        self.params[f'{par}_optimal'][i] = np.round(np.log(self.mdl[i].best_params_[par]), 0)
      if self.type == 'SVC':
        self.y_hat[fold[1]] = self.mdl[i].decision_function(X_test)
      if self.type == 'SVR':
        self.y_hat[fold[1]] = self.mdl[i].predict(X_test)
        self.bias_correct['slope'][i], self.bias_correct['int'][i] = self.correct_reg_bias(fold, y_test)
      self.get_stats(y_test, self.y_hat[fold[1]])
    self.output_stats()
    self.mdl = {'mdl':self.mdl, 'scaler':self.scaler}
    if self.type == 'SVR':
      self.mdl['bias_correct'] = self.bias_correct

  def prepare_sample(self, fold, scaler, classify=None):
    X_train, X_test = scaler.fit_transform(self.df.loc[fold[0], self.predictors]), scaler.transform(self.df.loc[fold[1], self.predictors])
    y_train, y_test = self.df.loc[fold[0], self.to_predict], self.df.loc[fold[1], self.to_predict]
    if classify is not None:
      y_train, y_test = y_train.map(dict(zip(classify, [-1, 1]))), y_test.map(dict(zip(classify, [-1, 1])))
    return X_train, X_test, y_train, y_test

  def param_search(self, mdl_i, X_train, y_train, scoring):
    gs = GridSearchCV(mdl_i, self.param_grid, scoring=scoring, cv=self.k, return_train_score=True, verbose=0)
    gs.fit(X_train, y_train)
    gs.best_estimator_.fit(X_train, y_train)
    return gs

  def get_stats(self, y_test, y_score):
    if len(y_test.unique()) == 2:
      fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
      self.stats['AUC'].append(metrics.auc(fpr, tpr))
      tn, fp, fn, tp = metrics.confusion_matrix(y_test, (y_score >= thresholds[np.argmax(tpr - fpr)])*2-1).ravel()
      self.stats['Accuracy'].append((tp + tn) / (tp + tn + fp + fn))
      self.stats['Sensitivity'].append(tp/(tp+fp))
      self.stats['Specificity'].append(tn/(tn+fn))
      precision, recall = tp / (tp + fp), tp / (tp + fn)
      self.stats['Precision'].append(precision)
      self.stats['Recall'].append(recall)
      self.stats['F1'].append(2 * precision * recall / (precision + recall))
    else:
      self.stats['MAE'].append(metrics.mean_absolute_error(y_test, y_score))
      self.stats['RMSE'].append(metrics.mean_squared_error(y_test, y_score, squared=False))
      self.stats['R2'].append(metrics.r2_score(y_test, y_score))
    logging.debug('   > ' + ' / '.join([f'{key}={value[-1]:#.4f}' for key, value in self.stats.items()]))

  def correct_reg_bias(self, fold, y_test):
    slope, interc = np.polyfit(y_test, self.y_hat[fold[1]], 1)
    if slope != 0:
      self.y_hat[fold[1]] = (self.y_hat[fold[1]] - interc) / slope
    return slope, interc

  def output_stats(self):
    [logging.info(f'>> {key} = {np.mean(value):#.4f} \u00B1 {np.std(value):#.4f}') for key, value in self.stats.items()]

@ignore_warnings(category=ConvergenceWarning)
def run_SVM(df, predictors, to_predict, param_grid, kernel='linear', k=5, n_repeats=1, verbose=1):

  logging_basic_config(verbose, content_only=True)
  SVM_mdl = SVM_Model(df, predictors, to_predict, param_grid, kernel, k, n_repeats)
  SVM_mdl.run_CV()  
  return SVM_mdl.y_hat, SVM_mdl.mdl, SVM_mdl.stats, SVM_mdl.params, [a[1] for a in SVM_mdl.folds]
  