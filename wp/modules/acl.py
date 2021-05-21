from IPython.display import clear_output

import tensorflow as tf
import os, time, datetime, importlib, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import bayesian.utils as butils

import active_learning
importlib.reload(active_learning)
from active_learning import AcquisitionFunction, Checkpoint, Config, DataPool, LabeledPool, UnlabeledPool
importlib.reload(active_learning)

# importlib.reload(active_learning)
# import torch
# import torch.nn as nn



class ActiveLearning:
    """
        Perform active learning for given model and data. 

        Args:
            model (tf.Layer): A neural network model written in tensorflow
            data (numpy.ndarray): The data to train the model on
            labels (numpy.ndarray): The labels of given data
            train_config (Config): The training configuration to be used
            acq_name (str): The acquisition functions to be used
            pseudo (bool): Use given labels as "user input" to evaluate active learning
    """

    def __init__(self, model, data, labels=None, config=None, train_config=None, acq_config=None, acq_name=None, pseudo=True):
        self.model = model
        self.checkpoint = Checkpoint()
        self.checkpoint.new(model) # Create initial checkpoint []

        # Perform pseudo active learning? (Meaning: use known labels and omit user input)
        self.pseudo = pseudo
        self.data = data
        self.labels = labels

        # Active learning specifics
        self.acquisition = AcquisitionFunction(acq_name)
        self.labeled_pool = LabeledPool(data, targets=labels)
        self.unlabeled_pool = UnlabeledPool(data, init_indices=self.labeled_pool.get_indices())
        self.train_config = train_config

        print(len(self.labeled_pool))
        print(len(self.unlabeled_pool))

        # Pool of labeled data
        self.history = []


    def start(self, limit=None, step_size=100):
        """
            Start the active learning loop.

            Parameters:
                limit (int): Limit the number of active learning loop iterations
        """

        if limit is None:
            limit = len(self.unlabeled_pool)

        # TODO: Initiale model selection, => random n-element, seed setzten
        # Bei wenigen gezogenen Daten: Klassenverteilung sollte gleich balanciert sein

        pg_bar = tqdm(range(0, len(self.unlabeled_pool), step_size), leave=True)
        # for i in tqdm(range(0, len(self.unlabeled_pool), step_size)):
        for i in pg_bar:

            # Is there any data left to label?
            if self.unlabeled_pool.is_empty():
                print("Is empty")
                print(len(self.unlabeled_pool))
                break
            
            # Train model on labeled data
            start = time.time()
            train_history = self.__train_model()
            end = time.time()
            train_time = end-start

            # Selected datapoints and label
            start = time.time()
            indices, _predictions = self.acquisition(self.model, self.unlabeled_pool, num=step_size)
            labels = self.__query(indices)
            self.unlabeled_pool.update(indices)

            end = time.time()
            acq_time = end - start

            # Update labeled pool
            start = time.time()
            self.labeled_pool[indices] = labels
            end = time.time()
            update_time = end - start

            # Debug

            # pg_bar.set_description("Training time: {}//Acquisition: {}//Update: {} // Labeled: {}".format(train_time, acq_time, update_time, len(self.labeled_pool)))
            self.__new_history_checkpoint(
                iteration=i,
                train_time=train_time,
                query_time=acq_time,
                training=(None if train_history is None else train_history.history)
            )

        print("Done: ")
        print(len(self.labeled_pool))
        print(len(self.unlabeled_pool))

        return self.history




    # -------------
    # Useable hooks
    # --------------------


    def pre_training_hook(self, model, **kwargs):
        pass


    def post_training_hook(self, model, **kwargs):
        pass

    
    def pre_query(self, model, **kwargs):
        pass


    def post_query(self, model, **kwargs):
        pass


    ## ----------------
    ## Private methods
    ## --------------------------

    def __new_history_checkpoint(self, **kwargs):
        self.history.append(kwargs)


    def __clean_history(self):
        self.history = []


    def __train_model(self):
        """
            Pre-train the network on already labeled data.
        """

        # Labeled data pool is empty, training not possible 
        if len(self.labeled_pool) == 0:
            print("Empty pool")
            return

        # # Skip model evaluation debugging purposes    
        else:
            return


        # Reset model weights
        self.checkpoint.load(self.model)

        # Compile model
        config = self.train_config
        optimizer = config["optimizer"]
        loss = config["loss"]
        metrics = config["metrics"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Fit model
        batch_size = config["batch_size"]
        epochs = config["epochs"]

        inputs, targets = self.labeled_pool[:]
        return self.model.fit(x=inputs, y=targets, batch_size=batch_size, epochs=epochs, verbose=0)


    def __query(self, indices):
        """
            Query for the next datapoints to be labeled.

            Returns:
                np.ndarray: The next datapoints to be labeled.
        """
        labels = None
        if not self.pseudo:
            # Let user label data
            labels = np.zeros(len(indices))
            for idx in indices:
                data_point = self.unlabeled_pool[idx]
                label = input("What's the label?")
                labels[idx] = label
                clear_output()

        else:
            # Auto label using known labels
            labels = self.labels[indices]

        return labels