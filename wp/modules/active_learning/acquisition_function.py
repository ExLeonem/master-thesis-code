import os, importlib, sys
import logging
import numpy as np

FILE_PATH = os.path.abspath(__file__)
MODULE_PATH = os.path.join(FILE_PATH, "..")
sys.path.append(MODULE_PATH)

import bayesian
importlib.reload(bayesian)
import bayesian.utils as butils


class AcquisitionFunction:
    """
        Query a model for next datapoints that should be labeled.
        
        Parameters:
            fn_name (str): The acquisition function to apply
    """

    def __init__(self, fn_name):
        self.fn = self._set_fn(fn_name)


    def __call__(self, model, data, **kwargs):
         return self.fn(model, data, **kwargs)


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
        
        elif name == "max_info_gain":
            return self._max_information_gain

        elif name == "max_var_ratio":
            return self._max_variation_ratios
        
        elif name == "std_mean":
            return self._std_mean

        else:
            return self._random


    def _max_entropy(self, model, unlabeled_pool, pool, num=5, runs=5, num_classes=2, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (UnlabeledPool) The pool of unlabeled data to select
        """
        data = pool.get_data()

        # Check if wanted num of datapoints is available
        num = self._adapt_selection_num(len(data), num)

        # Create predictions
        predictions = butils.batch_predict_n(model, data, n_times=runs)
        predictions = self.__prepare_predictions(predictions, num_classes)
        
        # Calculate max-entropy
        pred_avg = np.average(predictions, axis=1)
        log_pred = np.log(pred_avg)
        max_entropy = -np.sum(pred_avg*log_pred, axis=1)

        # Select n predictions with biggest max entropy
        indices = self.__select_first(max_entropy, num)
        return indices, max_entropy[indices]


    def _max_information_gain(self, model, pool, num=5, **kwargs):

        return 0, 0


    def _max_variation_ratios(self, model, pool, num=5, runs=10, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """

        # (batch, sample, num classses)
        # (batch, num_classes)

        data = pool.get_data()
        num = self._adapt_selection_num(len(data), num)

        # Create predictions for data
        predictions = btuils.batch_predict_n(model, data, n_times=runs)
        predictions = self.__prepare_predictions(predictions, num_classes)

        # Calcualte max variation rations
        pred_avg = np.average(predictions, axis=1)
        return 0, 0


    def _std_mean(self, model, pool, num=5, runs=10, num_classes=2, **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        # TODO: generalize for n-classes For binary classes

        data = pool.get_data()
        
        # Check if wanted num of datapoints is available
        num = self._adapt_selection_num(len(data), num)

        # Predict n-times
        predictions = butils.batch_predict_n(model, data, n_times=runs)
        predictions = self.__prepare_predictions(predictions, num_classes) # (batch_size, n-predictions, num_classes)

        # Calculate 
        pred_to_square = np.avg(np.power(predictions, 2), axis=1)
        avg_to_square = np.power(np.avg(predictions, axis=1), 2) # Along num of predictions axis

        # (batch-size, num classes)
        std_per_class = np.square(pred_to_square-avg_to_square)
        std_total = np.sum(std_per_class, axis=1) / num_classes # (batch_size, )
        
        # Return selected indices and prediction values
        indices = self.__select_first(std_total, num)
        return indices, std_total[indices]
        
    
    def _random(self, model, pool, num=5, **kwargs):
        """
            Randomly select a number of datapoints from the dataset.
            Baseline for comparison purposes.
           
            Returns:
                (numpy.ndarray): Randomly selected indices for next training.
        """
        data = pool.get_data()
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
        if num_classes == 2 and len(predictions.shape) == 2:
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate
            bin_alt_class = np.expand_dim(bin_alt_class, axis=-1)
            predictions = np.expand_dim(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=class_axis)
        
        return predictions