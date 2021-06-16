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

    def __init__(self, model, data, labels=None, config=None, train_config=None, eval_config=None, acq_config=None, acq_name=None, pseudo=True, debug=True, **kwargs):

        self.debug = debug
        self.setup_logger(debug)

        # Model itself
        self.model = model
        # if model.empty_weights():
        #     model.save_weights()
        
          # Compile model
        config = train_config
        optimizer = config["optimizer"]
        loss = config["loss"]
        metrics = config["metrics"]

        self.logger.info("Optimizer: {}".format(optimizer))
        self.logger.info("Loss: {}".format(loss))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # if model.empty_checkpoint():
        #     self.logger.info("Create new model checkpoint")
        #     model.new_checkpoint()

        # Perform pseudo active learning? (Meaning: use known labels and omit user input)
        self.pseudo = pseudo

        # Train/Test data
        self.inputs, self.targets = self.__train_test_val_split(data, labels)
        train_inputs = self.inputs["train"]
        train_targets = self.targets["train"]

        # Active learning specifics
        self.acquisition = AcquisitionFunction(acq_name, batch_size=700)
        self.labeled_pool = LabeledPool(train_inputs, targets=train_targets, pseudo=pseudo)
        self.unlabeled_pool = UnlabeledPool(train_inputs, init_indices=self.labeled_pool.get_indices())
        self.__init_pool_of_indices()

        # Configurations
        self.train_config = train_config
        self.eval_config = train_config if eval_config is None else eval_config
        self.acq_config = acq_config

        # Active learning history
        self.history = []

        # Callbacks
        self.pre_train_model_transform = dict.get(kwargs, "pre_train_model_transform")
        self.post_train_model_transform = dict.get(kwargs, "post_train_model_transform")


    def setup_logger(self, propagate):
        """
            Setup a logger for the active learning loop

            Parameters:
                propagate (bool): activate logging output in console?
        """

        logger = logging.Logger("ActiveLearning")
        log_level = logging.DEBUG if propagate else logging.CRITICAL

        logger.handler = logging.StreamHandler(sys.stdout)
        logger.handler.setLevel(log_level)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logger.handler.setFormatter(formatter)
        logger.addHandler(logger.handler)

        dir_name = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(dir_name, "logs", "acl.log")

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        self.logger = logger


    def __train_test_val_split(self, data, labels):
        """
            Split the the into different sets.
        """
        x_train, x_test, y_train, y_test = train_test_split(data, labels)

        inputs = {
            "train": x_train,
            "eval": x_test,
        }

        targets = {
            "train":y_train,
            "eval": y_test,
        }

        return inputs, targets


    def start(self, limit_iter=None, step_size=100):
        """
            Start the active learning loop.

            Parameters:
                limit (int): Limit the number of active learning loop iterations
                step_size (int): 

            Returns:
                (list(dict))
        """

        self.logger.info("Start: active learning loop")

        # TODO: Initiale model selection, => random n-element, seed setzten
        # Bei wenigen gezogenen Daten: Klassenverteilung sollte gleich balanciert sein

        current_iteration = 0
        pg_bar = tqdm(range(0, len(self.unlabeled_pool), step_size), leave=True)
        # for i in tqdm(range(0, len(self.unlabeled_pool), step_size)):
        for i in pg_bar:

            # Is there any data left to label?
            if self.unlabeled_pool.is_empty():
                break
            
            # Pre training evaluation
            pre_train_eval = self.__eval_model()
            self.logger.info("-------------")
            self.logger.info("Pre-fit eval: {}".format(pre_train_eval))

            # Train model on labeled data
            start = time.time()
            train_history = self.__train_model()
            end = time.time()
            train_time = end-start
            self.logger.info("Fitted model")

            # Selected datapoints and label
            start = time.time()
            indices, _predictions = self.acquisition(self.model, self.unlabeled_pool, num=step_size)
            labels = self.__query(indices)
            end = time.time()
            acq_time = end - start

            # Update unlabeled pool
            self.unlabeled_pool.update(indices)
            self.logger.info("Unlabeled pool size: {}".format(len(self.unlabeled_pool)))

            # Update labeled pool
            start = time.time()
            labeled_indices = self.unlabeled_pool.get_labeled_indices()
            self.labeled_pool[labeled_indices] = labels
            end = time.time()
            update_time = end - start
            self.logger.info("Labeled pool size {}".format(len(self.labeled_pool)))

            # Evaluate the model on test data
            eval_result = self.__eval_model()
            self.logger.info("Eval: {}".format(eval_result))

            # TODO: Generalize. Only works for tensorflow
            # pg_bar.set_description("Training time: {}//Acquisition: {}//Update: {} // Labeled: {}".format(train_time, acq_time, update_time, len(self.labeled_pool)))
            train_metrics = ({} if train_history is None else train_history.history)
            eval_metrics = ({} if eval_result is None else self.model.map_eval_values(eval_result))
            self.logger.info("Train: {}".format(train_metrics))
            self.__new_history_checkpoint(
                iteration=i,
                train_time=train_time,
                query_time=acq_time,
                **eval_metrics
            )

            # Iteration limit reached?
            current_iteration += 1
            if not (limit_iter is None) and current_iteration > limit_iter-1:
                self.logger.info("Reached iteration limit")
                break


        # self.model.clear_checkpoints()
        self.logger.info("finish active learning loop")
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
        # Labeled data pool is empty, training not possible 
        if len(self.labeled_pool) == 0:
            return

        # # Skip model evaluation debugging purposes    
        # else:
        #     return

        # Reset model weights
        # self.model.load_weights()

        # # Compile model
        # config = self.train_config
        # optimizer = config["optimizer"]
        # loss = config["loss"]
        # metrics = config["metrics"]

        # self.logger.info("Optimizer: {}".format(optimizer))
        # self.logger.info("Loss: {}".format(loss))

        # self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        config = self.train_config

        # Fit model
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        self.logger.info("Epochs: {}".format(epochs))
        self.logger.info("Batch-Size: {}".format(batch_size))
        inputs, targets = self.labeled_pool[:]

        self.logger.info("Use {}-inputs for training".format(len(inputs)))

        history = self.model.fit(inputs, targets, batch_size=batch_size, epochs=epochs, verbose=0)

        return history


    def __eval_model(self):
        """
            Evaluate the model 
        """

        inputs = self.inputs["eval"]
        targets = self.targets["eval"]

        config = self.eval_config
        batch_size = config["batch_size"]

        return self.model.evaluate(inputs, targets, batch_size=batch_size)


    def __query(self, indices):
        """
            Ask the oracle to label datapoints represented by given indices.
            Select targets automatically if in pseudo mode.

            Parameters:
                indices (numpy.ndarray): The indices for which to query a label.upda

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


    
    def __init_pool_of_indices(self, num_init_targets=10, seed=None):
        """
            Initialize the pool with randomly selected values.

            TODO:
                - Make it work for labels with additional dimensions (e.g. bag-of-words, one-hot vectors)
        """

        self.unlabeled_pool.update(indices)
        self.labeled_pool[indices] = labels

        # Use initial target values?
        if num_init_targets <= 0:
            return

        # Reproducability?
        if not (seed is None):
            np.random.seed(seed)

        # Select 'num_init_targets' per unique label 
        unique_targets = np.unique(self.targets["train"])
        for idx in range(len(unique_targets)):

            # Select indices of labels for unique label[idx]
            with_unique_value = self.targets["train"] == unique_targets[idx]
            indices_of_label = np.argwhere(with_unique_value)

            # Set randomly selected labels
            selected_indices = np.random.choice(indices_of_label.flatten(), num_init_targets, replace=True)

            # WILL NOT WORK FOR LABELS with more than 1 dimension
            self.unlabeled_pool.update(selected_indices)
            new_labels = np.full(len(selected_indices), unique_targets[idx])
            self.labeled_pool[selected_indices] = new_labels

            # self.labeled_indices[selected_indices] = 1
            # self.labels[selected_indices] = unique_targets[idx]