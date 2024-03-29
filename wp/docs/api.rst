.. _api:

API Reference
=========================================


Active Learning
----------------

.. autoclass:: acl.ActiveLearning
    :members:


Active Learning Loop
----------------------
.. autoclass:: active_learning.ActiveLearningLoop
   :members:


Experiment Suit
----------------------
.. autoclass:: active_learning.ExperimentSuit
   :members:


Acquisition Function
---------------------

.. autoclass:: active_learning.AcquisitionFunction
   :members:





Models
===========================

Model Wrapper
----------------

.. autoclass:: wrapper.Model
   :members:


MC Dropout
-----------------

.. autoclass:: wrapper.McDropout
   :members:


Moment Propagation
---------------------

.. autoclass:: wrapper.MomentPropagation
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


DataPool
--------------------------------

.. autoclass:: active_learning.Pool
   :members:
   :undoc-members:
   :show-inheritance:



Config
----------------

.. autoclass:: active_learning.Config
   :members:


.. autoclass:: active_learning.TrainConfig
    :members:


Checkpoint
----------------------

.. autoclass:: wrapper.Checkpoint
   :members:



