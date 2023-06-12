import logging

import numpy as np
import torch

from spare_scores.data_prep import logging_basic_config


class MLP_pytorch:
    """
    A class for managing MLP models using pytorch.

    Static attributes:
        predictors (list): List of predictors used for modeling.
        to_predict (str): Target variable for modeling.
        key_var (str): Key variable for modeling.

    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.

    Methods:
        train_model(df, **kwargs):
            Trains the model using the provided dataframe.
        
        apply_model(df):
            Applies the trained model on the provided dataframe and returns
            the predictions.
        
        set_parameters(**parameters):
            Updates the model's parameters with the provided values. This also
            changes the model's attributes, while retaining the original ones.
    """
    def __init__(self, predictors, to_predict, key_var, verbose=1,**kwargs):
        logger = logging_basic_config(verbose, content_only=True)

        self.predictors = predictors
        self.to_predict = to_predict
        self.key_var = key_var
        self.verbose = verbose

        valid_parameters = ['k', 
                            'n_repeats', 
                            'task', 
                            'param_grid']

        for parameter in kwargs.keys():
            if parameter not in valid_parameters:
                print("Parameter '" + parameter + "' is not accepted for "
                        +"MLP_pytorch. Ignoring...")
                continue
            
            if parameter == 'task':
                if kwargs[parameter] not in ['Classification', 'Regression']:
                    logger.error("Only 'Classification' and 'Regression' "
                                    + "tasks are supported.")
                    raise ValueError("Only 'Classification' and 'Regression' "
                                    + "tasks are supported.")
                else:
                    self.task = kwargs[parameter]
                continue

            self.__dict__.update({parameter: kwargs[parameter]})

        # Set default values for the parameters if not provided
        if 'task' not in kwargs.keys():
            self.task = 'Regression'
            
    def set_parameters(self, **parameters):
        self.__dict__.update(parameters)

    def fit(self, df, verbose):
        logger = logging_basic_config(self.verbose, content_only=True)

        df = df.astype('float64')

        X = df[self.predictors]
        y = df[self.to_predict]

        if self.task == 'Classification':
            metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
            y = y.astype('int64')
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self.class_to_idx_ = {cls: i for i, cls in enumerate(self.classes_)}
            y = np.array([self.class_to_idx_[i] for i in y])

            self.n_features_in_ = X.shape[1]
            self.n_outputs_ = self.n_classes_

            self.model = torch.nn.Sequential(
                                    torch.nn.Linear(self.n_features_in_, 100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100, 100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100, self.n_outputs_),
                                    torch.nn.Softmax(dim=1))
            
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()

            for epoch in range(100):
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    print('Epoch: ', epoch, 'Loss: ', loss.item())

        elif self.task == 'Regression':
            metrics = ['MAE', 'RMSE', 'R2']
            self.n_features_in_ = X.shape[1]
            self.n_outputs_ = 1

            self.model = torch.nn.Sequential(
                                    torch.nn.Linear(self.n_features_in_, 100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100, 100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100, self.n_outputs_))
            
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()

            for epoch in range(100):
                self.optimizer.zero_grad()
                # convert X to tensor:
                X = torch.from_numpy(X.values).float()
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                if epoch % 10 == 0:
                    print('Epoch: ', epoch, 'Loss: ', loss.item())

        else:
            logger.error("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
            raise ValueError("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
        
        self.stats = {metric: [] for metric in metrics}

        return self
    
    def predict(self, df):
        logger = logging_basic_config(self.verbose, content_only=True)

        df = df.astype('float64')

        X = df[self.predictors]

        if self.task == 'Classification':
            self.model.eval()
            y_pred = self.model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = np.array([self.classes_[i] for i in y_pred])
        
        elif self.task == 'Regression':
            self.model.eval()
            y_pred = self.model(X)
            y_pred = y_pred.detach().numpy()
        
        else:
            logger.error("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
            raise ValueError("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
        
        return y_pred
    
    def get_stats(self, y_test, y_score):
        logger = logging_basic_config(self.verbose, content_only=True)

        if self.task == 'Classification':
            from sklearn.metrics import (accuracy_score, f1_score,
                                         precision_score, recall_score,
                                         roc_auc_score)

            self.stats = {'Accuracy': accuracy_score(y_test, y_score),
                    'Precision': precision_score(y_test, y_score, average='macro'),
                    'Recall': recall_score(y_test, y_score, average='macro'),
                    'F1': f1_score(y_test, y_score, average='macro'),
                    'AUC': roc_auc_score(y_test, y_score, average='macro')}
        
        elif self.task == 'Regression':
            from sklearn.metrics import (mean_absolute_error,
                                         mean_squared_error, r2_score)

            self.stats = {'R2': r2_score(y_test, y_score),
                    'MSE': mean_squared_error(y_test, y_score),
                    'MAE': mean_absolute_error(y_test, y_score)}
        
        else:
            logger.error("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
            raise ValueError("Only 'Classification' and 'Regression' tasks are "
                            + "supported.")
        
    def output_stats(self):
        [logging.info(f'>> {key} = {np.mean(value):#.4f} \u00B1 {np.std(value):#.4f}') for key, value in self.stats.items()]