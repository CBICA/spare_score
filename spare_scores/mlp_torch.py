import logging
import time

import numpy as np
from spare_scores.data_prep import logging_basic_config
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, mean_squared_error, roc_auc_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, num_features = 147, hidden_size = 256, classification = True, dropout = 0.2, use_bn = False):
        super(SimpleMLP, self).__init__()

        self.num_features   = num_features
        self.hidden_size    = hidden_size 
        self.dropout        = dropout
        self.classification = classification
        self.use_bn         = use_bn

        self.linear1 = nn.Linear(self.num_features, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,  self.hidden_size//2)
        self.norm = nn.InstanceNorm1d(self.hidden_size //2 , eps=1e-15) 
        self.linear3 = nn.Linear(self.hidden_size//2 , 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## first layer
        x = self.linear1(x)
        x = self.dropout(self.relu(x))

        ## second layer
        x = self.linear2(x)
        if self.use_bn:
            x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        ## thrid layer
        x = self.linear3(x)

        if self.classification:
            x = self.sigmoid(x)
        else:
            x = self.relu(x)

        return x.squeeze()

class MLPTorchModel:
    """
    A class for managing MLP models.

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

        valid_parameters = ['task']

        for parameter in kwargs.keys():
            if parameter not in valid_parameters:
                print("Parameter '" + parameter + "' is not accepted for "
                        +"MLPModel. Ignoring...")
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

        if device != 'cuda':
            print('You are not using the GPU! Check your device')

        ################################## MODEL SETTING ##################################################
        self.classification = True if self.task == 'Classification' else False
        self.TARGET = self.to_predict
        self.mdl = SimpleMLP(num_features = len(predictors), classification= self.classification).to(device)
        self.batch_size = 128
        self.num_epochs = 100
        self.loss_fn = nn.BCELoss() if self.classification else nn.L1Loss()
        self.optimizer = optim.Adam(self.mdl.parameters(), lr = 3e-4)
        self.evaluation_metric = 'Accuracy' if self.task == 'Classification' else 'MAE' 
        self.scaler = None
        self.stats  = None
        self.param  = None
        ################################## MODEL SETTING ##################################################

    
    def plot_loss(self, train_loss_list, val_loss_list, eval_metric_list, eval_metric_name):
        '''
        plot the training loss vs validation loss (figure 1)
        plot the validation data accuracy (figure 2)

        input:
            1. train_loss_list: list of training loss
            2. val_loss_list: list of validation loss
            3. val_acc_list: list of accuracy loss

        output:
            Two Plots for loss and accuracy
        '''
        plt.subplot(1,2,1)
        plt.plot(train_loss_list, label = 'Train Loss')
        plt.plot(val_loss_list, label = 'Val Loss')
        plt.legend()
        plt.xlabel('Num of Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss vs Val Loss ')
        plt.subplot(1,2,2)
        plt.plot(eval_metric_list)
        plt.xlabel('Num of Epoch')
        plt.ylabel(eval_metric_name)
        plt.title('Evaluation Metric: {}'.format(eval_metric_name))
        plt.savefig('./loss_curve.png', dpi=300, format='png')
        plt.show()

    def get_all_stats(self, y_hat, y, classification = True):
        """
        Input: 
            y:     ground truth y (1: AD, 0: CN) -> numpy 
            y_hat: predicted y -> numpy, notice y_hat is predicted value [0.2, 0.8, 0.1 ...]

        Output:
            A dictionary contains: Acc, F1, Sensitivity, Specificity, Balanced Acc, Precision, Recall
        """
        y = np.array(y)
        y_hat = np.array(y_hat)
        if classification: 
            auc = roc_auc_score(y, y_hat)

            y_hat = np.where(y_hat >= 0.5, 1 , 0)
            
            tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

            acc = (tp + tn) / (fp + fn + tp + tn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            balanced_acc = (sensitivity + specificity) / 2
            precision   = tp / (tp + fp)
            recall      = tp / (tp + fn)
            F1          = 2 * (precision * recall ) / (precision + recall)

            dict = {}
            dict['Accuracy']          = acc
            dict['AUC']               = auc
            dict['Sensitivity']       = sensitivity
            dict['Specificity']       = specificity
            dict['Balanced Accuarcy'] = balanced_acc
            dict['Precision']         = precision
            dict['Recall']            = recall
            dict['F1']                = F1
  
        else:
            dict = {}
            mae  = mean_absolute_error(y, y_hat)
            mrse = mean_squared_error(y, y_hat, squared=False)
            r2   = r2_score(y, y_hat)
            dict['MAE']  = mae
            dict['RMSE'] = mrse
            dict['R2']   = r2

        return dict 
    


    def train(self, 
              model, 
              train_dl, 
              val_dl, 
              num_epochs,
              loss_fn,
              optimizer,
              evaluation_metric = 'MAE',
              detail_flag = False):

        model.train()
        best_eval_metric = -np.inf if evaluation_metric == 'Accuracy' else np.inf 
        classification = True if evaluation_metric == 'Accuracy' else False
        val_interval  = 1

        best_model_state_dic = None

        train_loss_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(num_epochs):

            if detail_flag:
                print('epoch: ', epoch + 1)

            total_train_loss = 0
            step = 0

            for _, (x,y) in tqdm(enumerate(train_dl), total = len(train_dl), leave = False):
                step += 1
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                optimizer.zero_grad()

                loss = loss_fn(output, y)

                total_train_loss += loss.item()

                loss.backward()

                optimizer.step()

            total_train_loss = total_train_loss / step 
            train_loss_list.append(total_train_loss)

            if detail_flag:
                print('Train loss: {}'.format(total_train_loss))


            val_step = 0
            val_total_acc = 0
            val_total_loss = 0

            if (epoch + 1) % val_interval == 0:
                with torch.no_grad():
                    for _, (x, y) in tqdm(enumerate(val_dl), total = len(val_dl), leave = False):
                        val_step += 1
                        x = x.to(device)
                        y = y.to(device)
                        output = model(x.float())

                        loss = loss_fn(output, y)
                        val_total_loss += loss.item()
                        acc = self.get_all_stats(output.cpu().data.numpy(), y.cpu().data.numpy() , classification= classification)[evaluation_metric]
                        val_total_acc += acc

                    val_total_loss = val_total_loss / val_step
                    val_total_acc  = val_total_acc / val_step 

                    val_acc_list.append(val_total_acc)
                    val_loss_list.append(val_total_loss)
                    
                    if detail_flag:
                        print('val metric: {}'.format(val_total_acc))
                        print('val loss: {}'.format(val_total_loss))

                if val_total_acc >= best_eval_metric and classification:
                    best_model_state_dic = model.state_dict()

                elif val_total_acc <= best_eval_metric and not classification:
                    best_model_state_dic = model.state_dict()


        return train_loss_list, val_loss_list, val_acc_list, best_model_state_dic
    
   
    def set_parameters(self, **parameters):
        if 'linear1.weight' in parameters.keys():
            self.param = parameters
        else:
            self.__dict__.update(parameters)
        
    @ignore_warnings(category= (ConvergenceWarning,UserWarning))
    def fit(self, df, verbose=1, **kwargs):
        logger = logging_basic_config(verbose, content_only=True)
        
        
        # Time the training:
        start_time = time.time()

        logger.info(f'Training the MLP model...')
        
        ############################################ start training model here ####################################
        X = df[self.predictors]
        y = df[self.to_predict].tolist()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reset_index(drop = True)
        X_val = X_val.reset_index(drop = True)

        self.scaler = StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val  = self.scaler.transform(X_val)

        train_ds = MLPDataset(X_train, y_train)
        val_ds   = MLPDataset(X_val, y_val)

        train_dl = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)
        val_dl   = DataLoader(val_ds, batch_size = self.batch_size, shuffle = True)

        train_loss_list, val_loss_list, val_acc_list, best_model_state_dic = self.train(self.mdl, train_dl, val_dl, self.num_epochs, self.loss_fn, self.optimizer, self.evaluation_metric, detail_flag = False)
        self.plot_loss(train_loss_list, val_loss_list, val_acc_list, eval_metric_name = self.evaluation_metric)

        self.mdl.load_state_dict(best_model_state_dic)
        self.mdl.eval()

        X_total = self.scaler.transform( np.array(X, dtype = np.float32) )
        X_total = torch.tensor(X_total).to(device)
        
        self.y_pred = self.mdl(X_total).cpu().data.numpy()
        self.stats = self.get_all_stats(self.y_pred, y, classification = self.classification)

        ########################################################################################################### 

        training_time = time.time() - start_time
        self.stats['training_time'] = round(training_time, 4)


        result = {'predicted':self.y_pred, 
                  'model':self.mdl, 
                  'stats':self.stats, 
                  'best_params':best_model_state_dic,
                  'CV_folds': None,
                  'scaler': self.scaler}
    
        if self.task == 'Regression':
            print('>>MAE = ', self.stats['MAE'])
            print('>>RMSE = ', self.stats['RMSE'])
            print('>>R2 = ', self.stats['R2'])

        else:
            print('>>AUC = ', self.stats['AUC'])
            print('>>Accuracy = ', self.stats['Accuracy'])
            print('>>Sensityvity = ', self.stats['Sensitivity'])
            print('>>Specificity = ', self.stats['Specificity'])
            print('>>Precision = ', self.stats['Precision'])
            print('>>Recall = ', self.stats['Recall'])
            print('>>F1 = ', self.stats['F1'])

        return result 
    
    def predict(self, df):
        
        X = df[self.predictors]
        X = self.scaler.transform(np.array(X, dtype = np.float32))
        X = torch.tensor(X).to(device)

        self.mdl.load_state_dict(self.param)

        self.mdl.eval()
        y_pred = self.mdl(X).cpu().data.numpy()

        return y_pred if self.task == 'Regression' else np.where(y_pred >= 0.5, 1 , 0)

    def output_stats(self):
        [logging.info(f'>> {key} = {np.mean(value):#.4f} \u00B1 {np.std(value):#.4f}') for key, value in self.stats.items()]
