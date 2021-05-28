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
        self.fn = self._set_fn(fn_name)
        self.batch_size = batch_size


    def __call__(self, model, pool, **kwargs):
        self.logger.info("Execute acquisition function.")

        data = pool.get_data()

        # Select values randomly? 
        # No need for batch processing
        if self.name == "random":
            return self.fn(model, pool, **kwargs)
        
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
            sub_result = self.fn(model, data[start:end], **kwargs)
            start = end

            # Initialize shape of results array
            if results is None:
                shape = [len(data)] + list(sub_result.shape[1:])
                results = np.zeros(shape)

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


    def _set_fn(self, name):
        """
            Set the function to use for acquisition.
            
            Parameters:
                name (str): The name of the acquisition to use.

            Returns:
                (function): The function to use for acquisition.
        """


        if name == "max_entropy":
            return self._max_entropy
        
        elif name == "bald":
            return self._bald

        elif name == "max_var_ratio":
            return self._max_variation_ratios
        
        elif name == "std_mean":
            return self._std_mean

        else:
            self.logger.debug("Set acquisition function: random baseline.")
            self.fn_name = "random"
            return self._random



    def _max_entropy(self, model, data, num=5, runs=5, num_classes=2, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (UnlabeledPool) The pool of unlabeled data to select
        """

        # Check if wanted num of datapoints is available
        num = self._adapt_selection_num(len(data), num)

        # Create predictions
        predictions = model.predict(data, runs=runs)
        posterior = model.posterior(predictions)
        log_post = np.log(posterior)

        # Calculate max-entropy
        return  -np.sum(posterior*log_post, axis=1)


    def _bald(self, model, data, num=5, **kwargs):

        return 0, 0


    def _max_variation_ratios(self, model, data, num=5, runs=10, num_classes=2, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """

        # (batch, sample, num classses)
        # (batch, num_classes)

        num = self._adapt_selection_num(len(data), num)

        predictions = model.predict(data, runs=runs)
        posterior = model.posterior(predictions)

        # Calcualte max variation rations
        return 1 + posterior.max(axis=1)


    def _std_mean(self, model, data, num=5, runs=10, num_classes=2, **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        # TODO: generalize for n-classes For binary classes
        

        # Check if wanted num of datapoints is available
        num = self._adapt_selection_num(len(data), num)

        predictions = model.predict(data, runs=runs)

        posterior = model.posterior(predictions) 
        squared_posterior = np.power(posterior, 2)
        post_to_square = model.expectation(squared_posterior) # TODO: Solve error here. How to restructure?

        expectation = model.expectation(predictions)
        exp_to_square = np.power(expectation, 2)
        
        std_per_class = np.square(post_to_square-exp_to_square)
        return np.sum(std_per_class, axis=1)
        
    
    def _random(self, model, pool, num=5, **kwargs):
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

