from dataclasses import dataclass
from spare_scores.svm import SVM_Model, run_SVM

class SpareModel:
    """
    A class for managing different spare models.

    Static attributes:
        predictors (list): List of predictors used for modeling.
        target (str): Target variable for modeling.
        model_type (str): Type of model to be used.
        model: The initialized model object.
        parameters (dict): Additional parameters for the model.

    Additionally, the class can be initialized with any number of keyword 
    arguments. These will be added as attributes to the class.
        
    Methods:
        initialize_model(**kwargs):
            Initializes the model based on the specified model type. This can 
            be either 'SVM' or 'MLP'.

        train_model(df, **kwargs):
            Trains the model using the provided dataframe.

        apply_model(df):
            Applies the trained model on the provided dataframe and returns 
            the predictions.

        set_parameters(**parameters):
            Updates the model's parameters with the provided values. This also 
            changes the model's attributes, while retaining the original ones.
    """
    def __init__(self, predictors, target, model_type, parameters={}) -> None:
        self.predictors = predictors
        self.target = target
        self.model_type = model_type
        self.model = None
        self.parameters = parameters

    def initialize_model(self, **kwargs):
        if self.model_type == 'SVM':
            self.model = SVM_Model(**kwargs)
        else:
            raise NotImplementedError("Only SVM is supported currently.")

    def train_model(self, df, **kwargs):
        if self.model is None:
            raise ValueError("Model is not initialized.")
      
        self.model.fit(self.df[self.predictors], self.df[self.target])

    def apply_model(self, df):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        
        return self.model.predict(df[self.predictors])
    
    def set_parameters(self, **parameters):
        self.parameters = parameters
        self.__dict__.update(parameters)



@dataclass
class MetaData:
    """Stores training information on its paired SPARE model"""
    mdl_type: str
    kernel: str
    predictors: list
    to_predict: str
    key_var: str
