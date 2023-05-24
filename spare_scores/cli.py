import argparse
import pandas as pd
import pkg_resources  # part of setuptools

from spare_scores.spare_scores import spare_train, spare_test

VERSION = pkg_resources.require("spare_scores")[0].version

def main():

    prog="spare_scores"
    description = "SPARE model training & scores calculation"
    usage = """
    spare_scores  v{VERSION}.
    SPARE model training & scores calculation
    required arguments:
        [ACTION]        The action to be performed, either 'train' or 'test'
        [-a, --action]


        [DATA]          The dataset to be used for training / testing. Can be 
        [-d, --data,    a filepath string of a .csv file, or a string filepath  
        --dataset,      of a pandas df. 
        --data_file]    
                        
    optional arguments:
        [MODEL]         The model to be used (only) for testing. Can be a 
        [-m, --mdl,     filepath string of a .pkl.gz file. Required for testing
        --model,        
        --model_file]

        [PREDICTORS]    The list of predictors to be used for training. List.
        [-p,            Example: --predictors predictorA predictorB predictorC
        --predictors]   Required for training.

        [TO_PREDICT]    The characteristic to be predicted in the course of the
        [-t,            training. String. Required for training.
        --to_predict]

        [POS_GROUP]     Group to assign a positive SPARE score (only for 
        -pg,            classification). String. Required for training.
        --pos_group]

        [KERNEL]        The kernel for the training. 'linear' or 'rbf' (only 
        -k,             linear is supported currently in regression).
        --kernel]

        [VERBOSE]       Verbosity. Int, higher is more verbose. [0,1,2]     
        [-v, 
        --verbose, 
        --verbosity]

        [SAVE_PATH]     Path to save the trained model. '.pkl.gz' file 
        [-s,            extension optional. If None is given, no model will be 
        --save_path]    saved.
        
        [HELP]          Show this help message and exit.
        -h, --help
    """.format(VERSION=VERSION)

    
    parser = argparse.ArgumentParser(prog=prog,
                                     usage=usage,
                                     description=description,
                                     add_help=False)

    # Action argument
    help = "The action to be performed, either 'train' or 'test'"
    parser.add_argument("-a", 
                        "--action", 
                        type=str,
                        help=help, 
                        choices=['train', 'test'],
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
                        type=str,
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
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)
    
    # Predictors argument
    help = "The list of predictors to be used for training. Example: "\
            + "--predictors predictorA predictorB predictorC"
    parser.add_argument("-p", 
                        "--predictors",
                        type=str, 
                        nargs='+', 
                        default=None, 
                        required=False)
    
    # To_predict argument
    help = "The characteristic to be predicted in the course of the training."
    parser.add_argument("-t", 
                        "--to_predict",
                        type=str, 
                        help=help, 
                        default=None, 
                        required=False)

    # Pos_group argument
    help = "Group to assign a positive SPARE score (only for classification)"
    parser.add_argument("-pg", 
                        "--pos_group",
                        type=str, 
                        help=help, 
                        default=None, 
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
    
    # Kernel argument
    help = "The kernel for the training. 'linear' or 'rbf' (only linear is "\
            + "supported currently in regression)."
    parser.add_argument("-k", 
                        "--kernel", 
                        type=str, 
                        choices=['linear', 'rbf'],
                        help=help, 
                        default='linear',
                        required=False)
        
    # Verbosity argument
    help = "Verbosity"
    parser.add_argument("-v", 
                        "--verbose", 
                        "--verbosity",
                        type=int, 
                        help=help, 
                        default=1, 
                        required=False)
        
    # Help
    parser.add_argument('-h', 
                        '--help',
                        action='store_true')
    
    arguments = parser.parse_args()
    if arguments.help:
        print(usage)
        return
    
    if arguments.action == 'train':
        if arguments.predictors is None or arguments.to_predict is None:
            print(usage)
            print("The following arguments are required: -p/--predictors, "
                  +"-t/--to_predict")
            return
        
        spare_train(arguments.data, 
                    arguments.predictors, 
                    arguments.to_predict, 
                    arguments.pos_group, 
                    arguments.kernel, 
                    arguments.verbose, 
                    arguments.save_path)
        return
        

    if arguments.action == 'test':
        if arguments.mdl is None:
            print(usage)
            print("The following arguments are required: -m/--mdl/--model/"
                  +"--model_file")
            return
        
        spare_test(arguments.data,
                   arguments.mdl,
                   arguments.verbose)
        return

    return