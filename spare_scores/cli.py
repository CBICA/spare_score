from .spare_scores import load_model, spare_train, spare_test
import argparse
import pandas as pd
from typing import Tuple, Union

def main():
    prog="spare_scores"
    description = "SPARE model training & scores calculation"
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)

    # Action argument
    help = "The action to be performed, either 'train' or 'test'"
    parser.add_argument("-a", 
                        "--action", 
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)
    
    # Data argument
    help = "The dataset to be used for training / testing. Can be either a "\
            + "string filepath of a .csv file (str) or a pandas dataframe "\
            + "(pd.DataFrame)"
    parser.add_argument("-d", 
                        "--data", 
                        "--dataset",
                        "--data_file",
                        type=Union[str, pd.DataFrame],
                        help=help, 
                        default=None, 
                        required=True)
    
    # Model argument
    help = "The model to be used (only) for testing. Can be either a "\
            + "string filepath of a .pkl.gz file or a tuple (dict, dict)."
    parser.add_argument("-m", 
                        "--mdl", 
                        "--model",
                        "--model_file",
                        type=Union[str, Tuple],
                        help=help, 
                        default=None, 
                        required=True)
    
    # Predictors argument
    help = "The list of predictors to be used for training. Example: "\
            + "--predictors predictorA predictorB predictorC"
    parser.add_argument("-p", 
                        "--predictors",
                        type=str, 
                        nargs='+', 
                        default=None, 
                        required=True)
    
    # To_predict argument
    help = "The characteristic to be predicted in the course of the training."
    parser.add_argument("-t", 
                        "--to_predict", 
                        "--to_be_predicted",
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)

    # Pos_group argument
    help = "Group to assign a positive SPARE score (only for classification)"
    parser.add_argument("-pg", 
                        "--pos_group", 
                        "--positive_group",
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)
    
    # Verbosity argument
    help = "Verbosity"
    parser.add_argument("-v", 
                        "--verbose", 
                        "--verbosity",
                        type=int, 
                        help=help, 
                        default=1, 
                        required=False)
    
    # Save_path argument
    help = "Path to save the trained model. '.pkl.gz' file extension "\
            + "expected. If None is given, no model will be saved."
    parser.add_argument("-s", 
                        "--save_path", 
                        type=str, 
                        help=help, 
                        default='', 
                        required=False)
    
    print(parser)
    return