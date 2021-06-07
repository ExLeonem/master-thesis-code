from IPython.display import clear_output
import os, time, datetime, importlib, time, gc, sys, logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
import bayesian.utils as butils

import active_learning
importlib.reload(active_learning)

from active_learning import AcquisitionFunction, Config, DataPool, LabeledPool, UnlabeledPool, Metrics
importlib.reload(active_learning)



class ActiveLearning:
    """
        Perform active learning for given model and data. 

        Args:
            model (tf.Layer): A neural network model written in tensorflow
            data (numpy.ndarray): The data to train the model on
            labels (numpy.ndarray): The labels of given data
            train_config (Config): The training configuration to be used
            acq_name (str|list): The acquisition functions to be used
            pseudo (bool): Use given labels as "user input" to evaluate active learning
    """

    def __init__(self, model, data, labels=None, config=None, train_config=None, acq_config=None, acq_name=None, pseudo=True, debug=True, **kwargs):

        self.setup_logger(debug)

        # Model itself
        self.model = model
        model.new_checkpoint()

        # self.checkpoint = Checkpoint()
        # self.checkpoint.new(model) # Create initial checkpoint []

        # Perform pseudo active learning? (Meaning: use known labels and omit user input)
        self.pseudo = pseudo

        # Active learning specifics
        self.inputs, self.targets = self.__train_test_val_split(data, labels)
        train_inputs = self.inputs["train"]
        train_targets = self.targets["train"]
        self.acquisition = AcquisitionFunction(acq_name, batch_size=700)
        self.labeled_pool = LabeledPool(train_inputs, targets=train_targets)
        self.unlabeled_pool = UnlabeledPool(train_inputs, init_indices=self.labeled_pool.get_indices())
        self.train_config = train_config

        # Pool of labeled data
        self.history = []
        # self.metrics = Metrics()

        # Callbacks
        self.pre_train_model_transform = dict.get(kwargs, "pre_train_model_transform")
        self.post_train_model_transform = dict.get(kwargs, "post_train_model_transform")


    def setup_logger(self, propagate):
        """
            Setup a logger for the active learning loop
        """

        logger = logging.Logger("ActiveLearning")
        log_level = logging.DEBUG if propagate else logging.CRITICAL

        logger.handler = logging.StreamHandler(sys.stdout)
        logger.handler.setLevel(log_level)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logger.handler.setFormatter(formatter)
        logger.addHandler(logger.handler)

        fh = logging.FileHandler("./logs/acl.log")
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        self.logger = logger


    def __train_test_val_split(self, data, labels):
        """
            Split the the into different sets.
        """
        x_train, x_test, y_train, y_test = train_test_split(data, labels)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test)

        inputs = {
            "train": x_train,
            "test": x_test,
            "valid": x_val
        }

        targets = {
            "train":y_train,
            "test": y_test,
            "valid":y_val
        }

        return inputs, targets


    def start(self, limit=None, step_size=100):
        """
            Start the active learning loop.

            Parameters:
                limit (int): Limit the number of active learning loop iterations
                step_size (int): 

            Returns:
                (list(dict))
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
            gc.collect()

            # TODO: Generalize. Only works for tensorflow
            # pg_bar.set_description("Training time: {}//Acquisition: {}//Update: {} // Labeled: {}".format(train_time, acq_time, update_time, len(self.labeled_pool)))
            train_metrics = ({} if train_history is None else train_history.history)
            self.logger.info(train_metrics)
            self.__new_history_checkpoint(
                iteration=i,
                train_time=train_time,
                query_time=acq_time,
                **train_metrics
            )

        return self.history


    # -------------
    # Active learning loop hooks
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
        
        self.logger.info("Start: Model fit")
        # Labeled data pool is empty, training not possible 
        if len(self.labeled_pool) == 0:
            return

        # # Skip model evaluation debugging purposes    
        # else:
        #     return

        # Reset model weights
        self.model.load_checkpoint()
        self.model.compile()

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

        history =  self.model.fit(x=inputs, y=targets, batch_size=batch_size, epochs=epochs, verbose=0)

        self.logger.info("End: Model fit")
        return history


    def __query(self, indices):
        """
            Query for the next datapoints to be labeled.

            Parameters:
                indices (numpy.ndarray): The indices for which to query a label.

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
            train_labels = self.targets["train"]
            labels = train_labels[indices]

        return labels