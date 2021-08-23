.. _quickstart:

Quickstart
=========================================

.. toctree::
   :maxdepth: 2

   
Generic active learning loops
------------------------------

How to create an :ref:`link to a different section<Active Learning Loop>`


.. code-block:: python

   from active_learning import AcquistionFunction, Dataset, ActiveLearningLoop
   from wrapper import McDropout
   import tensorflow as tf
   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Softmax

   # Create and wrapp the model
   base_model = Sequential([
         Conv2D(128, 4,activation="relu", input_shape=(28, 28, 1)),
         MaxPool2D(),
         Dropout(.2),
         Conv2D(64, 3, activation="relu"),
         MaxPool2D(),
         Dropout(.2),
         Flatten(),
         Dense(512, activation="relu"),
         Dense(256, activation="relu"),
         Dense(128, activation="relu"),
         Dense(5, activation="softmax")
      ])
   mc_model = McDropout(base_model)

   # Create a pool for unlabeled and labeled data
   inputs, targets = load_mnist_data()
   pooled_dataset = Dataset(inputs, targets, init_size=1)

   # Create an iterable loop object
   query_fn = "max_entropy"
   al_iterations = ActiveLearningIterator(mc_model, pooled_dataset, query_fn)

   for meta in al_iterations:
      # meta returns information of i-th iteration
      # Accumulated information can now be used to be written out or else
      pass