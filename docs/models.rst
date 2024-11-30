#########
ML Models
#########

This document explains the inner workings of our 2 ML models.


MLP models
__________

We use a total of 2 MLP models. One is implemented with sklearn and the other
with torch.

****************
MLP with Sklearn
****************

The MLP model implemented with sklearn can be found at `mlp.py <../../../spare_scores/mlp.py>`_.
The MLP model class takes a total of 4 arguments, plus, extra parameters if needed.

.. code-block:: python

    def __init__(
        self,
        predictors: list,
        to_predict: str,
        key_var: str,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None

The usage of each parameter is explained at `usage.rst <../../usage.rst>`_. Valid extra
pamaters are "k", "n_repeats", "task" and "param_grid". We perform thorough checking to
always make sure that the parameters are valid and to ensure valid training and testing.

.. note::

    If you notice any bugs with training and testing, please report it with an issue.

If the extra parameters are not passed, we use our default ones.
For optimization, we use sklearn's ``GridSearchCV``. As we explained, you can specify the param grid,
otherwise we use our default. As metrics, we use ``AUC``, ``Accuracy``, ``Sensitivity``, ``Specificity``,
``Precision``, ``Recall`` and ``F1`` for classification and ``MAE``, ``RMSE`` and ``R2`` for regression. Also note that by default, the model performs regression,
so always **make sure that you specified the task if you want to perform classification**.
You can always get all the stats with the ``get_stats`` function and print them with ``output_stats``.

**************
MLP with torch
**************

The MLP model implemented with torch can be found at `mlp_torch.py <../../../spare_scores/mlp_torch.py>`_.
The torch MLP model class takes a total of 4 arguments, plus, extra parameters if needed.

.. code-block:: python

    def __init__(
        self,
        predictors: list,
        to_predict: str,
        key_var: str,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None

As you can see, it's the same parameters as the sklearn implementation. Valid extra parameters
are "task", "bs"(batch size) and "num_epochs". We use the same metrics as the sklearn model, but now,
we perform optimization with optuna instead of GridSearchCV. Each fit creates an optuna study that tries
to maximize the values if the task is classification, otherwise, trying to minimize the error if the task is
regression.
You can always get all the stats with the ``get_stats`` function and print them with ``output_stats``.


Note that the MLP implementation exists at a different class in the same file with name ``SimpleMLP``.
Also, we implemented a class to manage the data we pass to our MLP model. This class also exists in the
same file with name ``MLPDataset``.

SVM model
_________

The SVM model can be found at `svm.py <../../../spare_scores/svm.py>`_.
The SVL model class takes a total of 4 arguments, plus, extra parameters if needed.

.. code-block:: python

    def __init__(
        self,
        predictors: list,
        to_predict: str,
        key_var: str,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None

As you can see again, it's the same parameters as the other models. Valid extra parameters are
"kernel", "k", "n_repeats", "task" and "param_grid". If the task is classification and the kernel is
linear, we use ``LinearSVC``, if the kernel is not linear(i.e. rbf) we use ``SVC``. If the task is regression we use
``LinearSVR`` with ``squared epsilon insensitive`` as a loss function. As metrics, we use ``AUC``, ``Accuracy``,
``Sensitivity``, ``Specificity``, ``Precision``, ``Recall`` and ``F1`` for classification and ``MAE``, ``RMSE`` and ``R2``
for regression.
You can always get all the stats with the ``get_stats`` function and print them with ``output_stats``.
