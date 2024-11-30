############
Installation
############

You can install the ``spare_score`` package with Python 3.8 up to Python 3.12. Please open an issue if you find any bugs for the
newer versions of spare_score.

****************
Install with pip
****************

To install ``spare_score`` with pip, just do: ::

    $ pip install spare_scores

We always have our latest stable version on PyPI, so we highly suggest you to install it this way, as this package is under heavy development and
building from source can lead to crashes and bugs.

*******************
Manual installation
*******************

You can manually build the package from source by running: ::

    $ git clone https://github.com/CBICA/spare_score

    # for python 3.12
    $cd spare_score && python3 -m pip install -e .

    # for python 3.8
    $ python3 setup.py bdist_wheel
      cd dist && pip install <wheel file>

Currently, these are the only ways to install our package. We will add more ways soon.

.. note::
    We **do not** recommend installing the package directly from source as the repository above is under heavy development and can cause
    crashes and bugs.
