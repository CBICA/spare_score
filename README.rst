spare scores
============

.. image:: https://codecov.io/gh/CBICA/spare_score/graph/badge.svg?token=7yk7pkydHE
    :target: https://codecov.io/gh/CBICA/spare_score
    :alt: Codecov

.. image:: https://github.com/CBICA/spare_score/actions/workflows/macos-tests-3.12.yml/badge.svg
    :alt: macos tests

.. image:: https://github.com/CBICA/spare_score/actions/workflows/ubuntu-tests-3.12.yml/badge.svg
    :alt: ubuntu tests

.. image:: https://img.shields.io/pypi/v/spare_scores
    :target: https://pypi.org/project/spare_scores/
    :alt: PyPI Stable


Overview
--------

"SPARE" is short for "Spatial Pattern of Abnormalities for Recognition of ..." If you have brain images of a case population, such as the Alzheimer's disease (AD), the SPARE model will try to find characteristic brain patterns of AD with respect to a control population, such as cognitively normal. This would be an example of a classification-based SPARE model (currently powered by support vector machine or SVM). This model (that we named SPARE-AD) then computes SPARE-AD scores on an individual-basis that indicates how much the individual carries the learned brain patterns of AD.

Alternatively, you may want to find the spatial pattern related to brain aging (BA). In this case, you would provide sample images and indicate that chronological age is what you expect the model to learn patterns for. This would be an example of a regression-based SPARE model (also powered by SVM). This model (that we named SPARE-BA) then computes SPARE-BA scores on an individual-basis that predicts your brain age.
\
\
\
For detailed documentation, please see here: **[spare_scores](https://cbica.github.io/spare_score/)**

Installation
____________

You can install the spare_score package for python 3.8 up to python 3.12
Please open an issue if you find any bugs for the newer versions of spare_score

*********
Using pip
*********

You can install our latest stable PyPI wheel: ::

  $ pip install spare_scores

**************************
Manually build spare_score
**************************

You can install spare_scores from source: ::

    # for python 3.12
    $ git clone https://github.com/CBICA/spare_score.git
      cd spare_score
      python -m pip install .

    # for python 3.8 and similar
    # python setup.py bdist_wheel
      cd dist && pip install <wheel file>


Usage
_____

Example of training a model (given the example data): ::

  $ spare_score --action train \
      --input spare_scores/data/example_data.csv \
      --predictors H_MUSE_Volume_11 H_MUSE_Volume_23 H_MUSE_Volume_30 \
      --ignore_vars Sex \
      --to_predict Age \
      --kernel linear \
      --verbose 2 \
      --output my_model.pkl.gz

Example of testing (applying) a model (given the example data): ::

  $ spare_score -a test \
      -i spare_scores/data/example_data.csv  \
      --model my_model.pkl.gz \
      -o test_spare_data.csv \
      -v 0 \
      --logs test_logs.txt

.. note::

  You can always see all of the CLI documentation with ``spare_score -h``

References
__________

- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

Disclaimer
__________

- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- By using spare_scores, the user agrees to the following license: [CBICA Software License](https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html)

Contact
_______

For more information and support, please post on the [Discussions](https://github.com/CBICA/spare_score/discussions) section or contact [CBICA Software](mailto:software@cbica.upenn.edu)
