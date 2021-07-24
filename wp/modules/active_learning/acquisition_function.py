import os, sys, importlib
import time
import logging
import numpy as np
from enum import Enum

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import bayesian
importlib.reload(bayesian)
import bayesian.utils as butils

# from utils import setup_logger


class AcquisitionFunction:
    """
        Query a model for next datapoints that should be labeled.
        Already implemented bayesian models implement the following acquisition
        functions:

        - Max Entropy (\"max_entropy\")
        - Bald (\"bald\")
        - Variation Ratios (\"max_var_ratio\")
        - Mean standard deviation (\"mean_std\")
        - Randomized selection (\"random\")

        Parameters:
            fn_name (str): The name of the acquisition function to use.
            batch_size (int): The number of batches to split the data into for processing. (default=None)
            verbose (bool): Apply debugging? (default=False)

        Attributes:
            name (str): The name of the acquisition function to apply.
            fn (function): The acquisition function to execute.
            batch_size (int): to configure the processing in batches (default=None)
    """

    def __init__(self, fn_name, batch_size=None, verbose=False):
        self.setup_logger(verbose)

        self.name = fn_name
        self.fn = None
        self.batch_size = batch_size


    def setup_logger(self, propagate):
        """
            Setup a logger for the active learning loop
        """

        logger = logging.Logger("AcqLogger")
        log_level = logging.DEBUG if propagate else logging.CRITICAL

        logger.handler = logging.StreamHandler(sys.stdout)
        logger.handler.setLevel(log_level)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logger.handler.setFormatter(formatter)
        logger.addHandler(logger.handler)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(current_dir, "..", "logs", "acf.log")
        
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        self.logger = logger


    def __call__(self, model, pool, step_size=20, **kwargs):
        """
            
            Parameter:
                model (BayesModel): The model to use for the computation of acquistion functions.
                pool (Pool): The pool of unlabeled data.
                step_size (int): Number of datapoints to collect for next active learning iteration.

            Returns:
                (numpy.ndarray) Indices
        """

        # Set initial acquistion function
        if self.fn is None:
            self.fn = self._set_fn(model)

        data, indices = pool.get_unlabeled_data()
        # data = pool.get_data()
        # indices = pool.get_indices()

        # Select values randomly? 
        # No need for batch processing
        if self.name == "random":
            self.logger.info("Random function")
            return self.fn(indices, data, step_size=step_size, **kwargs)            

        # Iterate throug batches of data
        results = None
        num_datapoints = len(data)
        self.logger.info("Use {} unlabeled datapoints".format(num_datapoints))
        start = 0
        # end = self.batch_size if num_datapoints > self.batch_size else num_datapoints
        end = 0
        
        self.logger.info("Kwargs: {}".format(kwargs))

        # ---------
        # Alternative
        if self.batch_size is None:
            self.batch_size = len(data)

        num_batches = num_datapoints/self.batch_size
        batches = np.array_split(data, num_batches, axis=0)
        results = []
        for batch in batches:

            sub_result = self.fn(batch, **kwargs)
            self.logger.info("Result shape: {}".format(sub_result.shape))
            results.append(sub_result)

        stacked = np.hstack(results)
        self.logger.info("Stacked shaped: {}".format(stacked.shape))
        num_of_elements_to_select = self._adapt_selection_num(len(stacked), step_size)
        return self.__select_first(stacked, indices, num_of_elements_to_select)


    def _adapt_selection_num(self, num_indices, num_to_select):
        """
            Check if n datapoints are available at all, else adapt the number of datapoints to select.

        """
        if num_indices == 0:
            raise ArgumentError("Can't select {} datapoints, all data is labeled.".format(num_to_select))

        if num_indices < num_to_select:
            return num_indices
        
        return num_to_select


    def _set_fn(self, model):
        """
            Set the function to use for acquisition.
            
            Parameters:
                name (str): The name of the acquisition to use.

            Returns:
                (function): The function to use for acquisition.
        """

        query_fn = model.get_query_fn(self.name)
        if query_fn is None:
            self.logger.debug("Set acquisition function: random baseline.")
            self.name = "random"
            return self._random

        else:
            return query_fn
    

    def _random(self, indices, data, step_size=5, **kwargs):
        """
            Randomly select a number of datapoints from the dataset.
            Baseline for comparison purposes.

            FIX: Random selection is wrong!!!

            Parameters:
                model (BayesianModel): The model to perform active learning on.
                pool (DataPool): The pool of data to use.
                num (int): Numbers of indices to draw from unlabeled data.
           
            Returns:
                (numpy.ndarray): Randomly selected indices for next training.
        """
        self.logger.info("----------Random-------------")

        available_indices = np.linspace(0, len(data)-1, len(data), dtype=int)
        step_size = self._adapt_selection_num(len(available_indices), step_size)
        selected = np.random.choice(available_indices, step_size, replace=False).astype(int)

        self.logger.info("Indices selected: {}".format(selected))
        return indices[selected], data[selected]


    def __select_first(self, predictions, indices, n):
        """
            Select n biggest elements from k- predictions.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the network

            Returns: 
                (numpy.ndarray) indices of n-biggest predictions.
        """
        
        self.logger.info("__select_first/start-sort")
        sorted_keys = np.argsort(predictions)
        # n_biggest_keys = sorted_keys[-n:]
        n_biggest_keys = sorted_keys[-n:]
        # other_keys = sorted_keys[:10]
        # self.logger.info("Other values: {}".format(predictions[other_keys]))
        # self.logger.info("Values: {}".format(predictions[n_biggest_keys]))

        self.logger.info("__select_first/finish_sort")
        return indices[n_biggest_keys], predictions[n_biggest_keys]


    # ----------
    # Dunder
    # ------------------

    def __str__(self):
        return self.name


    # ----------
    # Getter-/Setter
    # ------------------
    
    def set_fn(self, fn):
        self.fn = fn

    def get_name(self):
        return self.name