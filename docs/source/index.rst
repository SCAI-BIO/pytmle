.. PyTMLE documentation master file, created by
   sphinx-quickstart on Wed Apr 30 10:53:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTMLE's documentation!
==================================

PyTMLE is a flexible Python implementation of the Targeted Maximum Likelihood Estimation (TMLE) framework for survival and competing risks outcomes.
It is designed to be easy to use with default models for initial estimates of nuisance functions which are applied in a super learner framework.
However, it also allows for custom models to be used for the initial estimates or even passing initial estimates directly to the second TMLE stage.

.. code-block:: bash
   :caption: Installation from PyPI using pip

   pip install pytmle

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`



.. toctree::
   :maxdepth: 2
   :caption: API reference

   ./apiref.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorial notebooks

   notebooks/01_tutorial_basic.ipynb

   notebooks/02_tutorial_custom_model.ipynb

   notebooks/03_tutorial_skip_first_stage.ipynb
