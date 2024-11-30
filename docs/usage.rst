#####
Usage
#####

In order to use ``spare_scores``, you have to use our CLI. The CLI can perform training and testing. For training we use 2 models, a SVM and
a MLP. To perform **training**, you can just do: ::

    $ spare_score --action train \
        --input spare_scores/data/example_data.csv \
        --predictors H_MUSE_Volume_11 H_MUSE_Volume_23 H_MUSE_Volume_30 \
        --ignore_vars Sex \
        --to_predict Age \
        --kernel linear \
        --verbose 2 \
        --output my_model.pkl.gz

- With the ``--action`` parameter, you specify if you want to perform training or testing.
- With the ``--input`` parameter, you specify the directory of the input data(has to be .csv).
- The ``--predictors`` parameter is a list that represents the columns that will be used by the models for training.
- With the ``--ignore_vars`` parameter, you specify(if needed) any columns than you want the models to ignore.
- ``--to_predict`` represents the target column.
- ``--kernel`` represents the kernel of regression/classification. Currently only ``linear`` is supported as an option for regression.
- ``--verbose`` you can pass a value != 0 to enable verbosity on training/testing.
- With the ``--output`` parameter, you specify the directory of the output. This is where the trained model will be saved.

To perform **testing**, you can just do: ::

    $ spare_score -a test \
        -i spare_scores/data/example_data.csv  \
        --model my_model.pkl.gz \
        -o test_spare_data.csv \
        -v 0 \
        --logs test_logs.txt

The only new parameter here is ``--logs`` that represents the filename of the logger.
