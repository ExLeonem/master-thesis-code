import os, sys, importlib
import numpy as np
from enum import Enum

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import bayesian
importlib.reload(bayesian)
import bayesian.utils as butils


import logging

LOG_FILE = os.path.join(MODULE_PATH, "logs", "acf.log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARN
)


# class Functions(Enum):
#     PRED_ENTROPY,
#     BALD,
#     VAR_RATIO,
#     STD_MEAN,
#     RANDOM


class AcquisitionFunction:
    """
        Query a model for next datapoints that should be labeled.
        
        Parameters:
            fn_name (str): The acquisition function to apply
    """

    def __init__(self, fn_name, batch_size=10, debug=False):
        # Logger
        self.logger = logging.getLogger(__file__)
        self.logger.propagate = debug
        
        self.name = fn_name
        self.fn = None
        self.batch_size = batch_size


    def __call__(self, model, pool, **kwargs):
        self.logger.info("Execute acquisition function.")

        # Set initial acquistion function
        if self.fn is None:
            self.fn = self._set_fn(model)

        data = pool.get_data()

        # Select values randomly? 
        # No need for batch processing
        if self.name == "random":
            return self.fn(pool, **kwargs)
        
        # Iterate throug batches of data
        results = None
        num_datapoints = len(data)
        start = 0
        end = self.batch_size if num_datapoints > self.batch_size else num_datapoints

        # TODO: correcttion needed, throws error when batch_size == num_datapoints
        self.logger.info("Iterate over input batches.")
        while end < num_datapoints:
            sub_result = None
            end = start + self.batch_size

            # Less elements than specified by batch_size?
            if num_datapoints <= (start + self.batch_size):
                end = num_datapoints
            
            # Calcualte results of batch
            # print("Start: {}, end: {}".format(start, end))
            sub_result = self.fn(data[start:end], **kwargs)
            start = end

            # Initialize shape of results array
            if results is None:
                shape = [len(data)] + list(sub_result.shape[1:])
                results = np.zeros(shape)

            # print(sub_result.shape)
            # print(results.shape)
            results[start:end] = sub_result[start:end]

        # Return selected indices and prediction values
        self.logger.info("Iteration completed")
        default_num = 20
        num_of_elements_to_select = dict.get(kwargs, "runs", default_num)
        indices = self.__select_first(results, num_of_elements_to_select)
        return indices, results[indices]


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
            


    def _random(self, pool, num=5, **kwargs):
        """
            Randomly select a number of datapoints from the dataset.
            Baseline for comparison purposes.

            Parameters:
                model (BayesianModel): The model to perform active learning on.
                pool (DataPool): The pool of data to use.
                num (int): Numbers of indices to draw from unlabeled data.
           
            Returns:
                (numpy.ndarray): Randomly selected indices for next training.
        """

        available_indices = pool.get_indices()
        num = self._adapt_selection_num(len(available_indices), num)
        indices = np.random.choice(available_indices, num, replace=False).astype(int)

        data = pool.get_data()
        return indices, data[indices]


    def __select_first(self, predictions, n):
        """
            Select n biggest elements from k- predictions.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the network

            Returns: 
                (numpy.ndarray) indices of n-biggest predictions.
        """

        return np.argpartition(predictions, -n)[-n:]

