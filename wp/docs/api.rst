.. _api:

API Reference
=========================================


Active Learning
----------------

.. autoclass:: acl.ActiveLearning
    :members:



Acquisition Function
---------------------

.. autoclass:: active_learning.AcquisitionFunction
   :members:





Models
===========================

Bayesian Model
----------------

.. autoclass:: bayesian.BayesModel
   :members:


MC Dropout
-----------------

.. autoclass:: bayesian.McDropout
   :members:


Moment Propagation
---------------------

.. autoclass:: bayesian.MomentPropagation
   :members:



Data loader
==========================

Loading utilities to ease the loading/pre-processing of datasets for benchmarking
of active leanring loops and models.

BenchmarkData
---------------
Load and pre-process a given dataset.

.. autoclass:: data.BenchmarkData
   :members:


DataSetType
--------------

An Enum representing different datasets.

.. autoclass:: data.DataSetType
   :members:



Supplementary
==========================

Metrics
----------------

.. autoclass:: active_learning.Metrics
   :members:


DataPools
----------------

.. autoclass:: active_learning.DataPool
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: active_learning.LabeledPool
   :members:


Config
----------------

.. autoclass:: active_learning.Config
   :members:


.. autoclass:: active_learning.TrainConfig
    :members:


Checkpoint
----------------------

.. autoclass:: bayesian.Checkpoint
   :members:



