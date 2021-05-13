from IPython.display import clear_output

import os, time, datetime, importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import bayesian.utils as butils

import active_learning
importlib.reload(active_learning)
from active_learning import AcquisitionFunction, Checkpoint, Config, DataPool, LabeledPool


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

    def __init__(self, model, data, labels=None, train_config=None, acq_name=None, pseudo=True):
        self.model = model
        self.checkpoint = Checkpoint()
        self.checkpoint.new(model) # Create initial checkpoint []

        # Perform pseudo active learning? (Meaning: use known labels and omit user input)
        self.pseudo = pseudo
        self.data = data
        self.labels = labels

        # Active learning specifics
        self.acquisition = AcquisitionFunction(acq_name)
        self.unlabeled_pool = DataPool(data)
        self.labeled_pool = LabeledPool(data)
        self.train_config = train_config
            
        # self.acq_config = acq_config
        # if acq_config is None:
        #     self.acq_config = Config(
        #         function="random"
        #     )


        # Pool of labeled data
        self.history = []


    def start(self, limit=None):
        """
            Start the active learning loop.

            Parameters:
                limit (int): Limit the number of active learning loop iterations
        """

        for i in tqdm(range(limit)):
            self.__pre_train()

            idx = self.acquisition(self.model, self.unlabeled_pool)
            data_point = self.unlabeled_pool[idx]
            # Display data
            label = input("What's the label?")
            clear_output()

            # Update pools of labeled/unlabeled data
            self.unlabeled_pool = np.delete(self.unlabeled_pool, idx, axis=0)
            self.inputs.append(data_point)
            self.targets.append(label)

    

    def __pre_train(self):
        """
            Pre-train the network on already labeled data.
        """

        # Labeled data pool is empty, training not possible 
        if len(self.inputs) == 0:
            return

        # Reset model weights
        self.checkpoint.load(self.model)

        # Compile model
        config = self.train_config
        optimizer = config["optimizer"]
        loss = config["optimizer"]
        metrics = config["metrics"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Fit model
        batch_size = config["batch_size"]
        epochs = config["epochs"]

        # inputs = self.labeled_pool.get_inputs()
        # targets = self.labeled_pool.get_labels()
        inputs, targets = self.labeled_pool[:]
        self.model.fit(x=self.inputs, y=self.targets, batch_size=batch_size, epochs=epochs)



    def __next_dp(self):
        """
            Query for the next datapoints to be labeled.

            Returns:
                np.ndarray: The next datapoints to be labeled.
        """
        pass
