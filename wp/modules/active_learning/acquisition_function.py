import os, importlib, sys
import logging
import numpy as np
from enum import Enum

FILE_PATH = os.path.abspath(__file__)
MODULE_PATH = os.path.join(FILE_PATH, "..")
sys.path.append(MODULE_PATH)

import bayesian
importlib.reload(bayesian)
import bayesian.utils as butils


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

    def __init__(self, fn_name, batch_size=10):
        self.name = fn_name
        self.fn = self._set_fn(fn_name)
        self.batch_size = batch_size

    def __call__(self, model, pool, **kwargs):

        data = pool.get_data()

        # Select values randomly? 
        # No need for batch processing
        if self.name == "random":
            return self.fn(model, data, **kwargs)
        
        # Iterate throug batches of data
        results = None
        num_datapoints = len(data)
        start = 0
        end = self.batch_size if num_datapoints > self.batch_size else num_datapoints

        # TODO: correcttion needed, throws error when batch_size == num_datapoints
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
        default_num = 20
        num_of_elements_to_select = dict.get(kwargs, "runs", default_num)
        indices = self.__select_first(results, num_of_elements_to_select)
        return indices, results[indices]


    def __select_next(self, data, indices):
        """

        """
        pass


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

        if name == "pred_entropy":
            return self.__pred_entropy

        if name == "max_entropy":
            return self._max_entropy
        
        elif name == "bald":
            return self._bald

        elif name == "max_var_ratio":
            return self._max_variation_ratios
        
        elif name == "std_mean":
            return self._std_mean

        else:
            return self._random


    def __pred_entropy(self, model, data, num=5, runs=5, num_classes=2, **kwargs):

        # Check if wanted num of datapoints is available
        num = self._adapt_selection_num(len(data), num)
     

        # Predictions per class (batch_size, runs, num_classes)
        predictions = butils.batch_predict_n(model, data, n_times=runs, enable_tqdm=False)
        predictions = self.__prepare_predictions(predictions, num_classes)

        # Calculate pred entropy
        avg_predictions = np.average(predictions, axis=1)
        log_of_classes = np.log(avg_predictions)
    
        # print("Out-shape: ", predictions.shape)
        return np.sum(avg_predictions*log_of_classes, axis=1)


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
        predictions = butils.batch_predict_n(model, data, n_times=runs)
        # predictions = self.__prepare_predictions(predictions, num_classes)
        
        # Calculate max-entropy
        pred_avg = np.average(predictions, axis=1)
        log_pred = np.log(pred_avg)
        max_entropy = -np.sum(pred_avg*log_pred, axis=1)

        return max_entropy


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

        # Create predictions for data
        predictions = butils.batch_predict_n(model, data, n_times=runs, enable_tqdm=False)
        predictions = self.__prepare_predictions(predictions, num_classes)

        # Calcualte max variation rations
        pred_avg = np.average(predictions, axis=1)
        return 1 + pred_avg.max(axis=1)


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

        print("------------")
        # New: Use model class
        predictions = model.predict(data, runs=runs)
        print(predictions.shape)
        predictions = y__prepare_predictions(predictions)

        print(predictions.shape)

        exp_to_square = model.expectation(model.posterior(np.power(predictions, 2)))
        post_to_square = model.expectation(np.power(model.posterior(predictions), 2))
        
        std_per_class = np.square(exp_to_square-post_to_square)
        return np.sum(std_per_class, axis=1)

        # Old
        # # Predict n-times
        # predictions = butils.batch_predict_n(model, data, n_times=runs, enable_tqdm=False)
        # predictions = self.__prepare_predictions(predictions, num_classes) # (batch_size, n-predictions, num_classes)

        # # Calculate 
        # pred_to_square = np.average(np.power(predictions, 2), axis=1)
        # avg_to_square = np.power(np.average(predictions, axis=1), 2) # Along num of predictions axis

        # # (batch-size, num classes)
        # std_per_class = np.square(pred_to_square-avg_to_square)
        # std_total = np.sum(std_per_class, axis=1) / num_classes # (batch_size, )
    
        # return std_total
        
    
    def _random(self, model, data, num=5, **kwargs):
        """
            Randomly select a number of datapoints from the dataset.
            Baseline for comparison purposes.
           
            Returns:
                (numpy.ndarray): Randomly selected indices for next training.
        """
        available_indices = pool.get_indices()
        
        num = self._adapt_selection_num(len(available_indices), num)
        indices = np.random.choice(available_indices, num, replace=False).astype(int)
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


    def __prepare_predictions(self, predictions, num_classes):
        """
            Consider to move into model class.

        """

        # Binary case: calculate complementary prediction and concatenate
        if num_classes == 2 and predictions.shape[-1] == 1:
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            # bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            # predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions