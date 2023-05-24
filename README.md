For detailed documentation, please see here: **[spare_scores](https://cbica.github.io/spare_score/)**

# Compute SPARE Scores for Your Case
"SPARE" is short for "Spatial Pattern of Abnormalities for Recognition of ..." If you have brain images of a case population, such as the Alzheimer's disease (AD), the SPARE model will try to find characteristic brain patterns of AD with respect to a control population, such as cognitively normal. This would be an example of a classification-based SPARE model (currently powered by support vector machine or SVM). This model (that we named SPARE-AD) then computes SPARE-AD scores on an individual-basis that indicates how much the individual carries the learned brain patterns of AD.

Alternatively, you may want to find the spatial pattern related to brain aging (BA). In this case, you would provide sample images and indicate that chronological age is what you expect the model to learn patterns for. This would be an example of a regression-based SPARE model (also powered by SVM). This model (that we named SPARE-BA) then computes SPARE-BA scores on an individual-basis that predicts your brain age.
<br /><br />

## Installation

### Conda environment using pip
```
    conda create -n spare python=3.8
    conda activate spare
    conda install pip
    pip install spare_scores
```

### Python3 virtual environment using pip
```
    python3 -m venv env spare
    source spare/bin/activate
    pip install spare_scores
```

### Conda environment from Github repository
```
    git clone https://github.com/CBICA/spare_score.git
    cd spare_score
    pip install .
```


## Usage
```
spare_scores  v0.1.17.
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
    [-s,            extension expected. If None is given, no model will be 
    --save_path]    saved.
    
    [HELP]          Show this help message and exit.
    -h, --help
```

## Examples
<p>Example of training a model (given the example data):</p>

```
spare_score --action train \
            --data_file spare_scores/data/example_data.csv \
            --predictors H_MUSE_Volume_11 H_MUSE_Volume_23 H_MUSE_Volume_30 \
            --to_predict Age \
            --kernel linear \
            --verbose 2 \
            --save_path my_model.pkl.gz
```

<p>Example of testing (applying) a model (given the example data):</p>

```
spare_score -a test \
            -d spare_scores/data/example_data.csv  \
            --model saved_model_gai_240523.pkl.gz \
            -v 0
```

## References
- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

## Disclaimer
- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- By using spare_scores, the user agrees to the following license: https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

## Contact
For more information and support, please post on the [Discussions](https://github.com/CBICA/spare_score/discussionss) section or contact <a href="mailto:software@cbica.upenn.edu">CBICA Software</a>.
