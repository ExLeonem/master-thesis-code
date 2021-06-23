import os, sys, math
import numpy as np
import logging as log
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras

from . import  BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import tensorflow as tf



class McDropout(BayesModel):
    """
        Wrapper class for neural networks.

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config, model_type=ModelType.MC_DROPOUT, **kwargs)

        # disable batch norm
        # super().disable_batch_norm()


    def __call__(self, inputs, sample_size=10, batch_size=1, callback=None, **kwargs):
        """
            
            Parameters:
                inputs (numpy.ndarray): Inputs going into the model
                sample_size (int): How many times to sample from posterior?
                batch_size (int): In how many batches to split the data?
        """
        
        if batch_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't select negative amount of batches.")

        if sample_size < 1:
            raise ValueError("Error in McDropout.__call__(). Can't sample negative amount.")


        total_len = len(inputs)
        num_batches = math.ceil(total_len/batch_size)
        batches = np.array_split(inputs, num_batches, axis=0)
        predictions = []

        for batch in batches:

            # Sample from posterior
            posterior_samples = []
            for i in range(sample_size):
                posterior_samples.append(self._model(batch, training=True))
                
            # Omit sample dimension, when only sampled single time?
            if sample_size > 1:
                stacked = np.stack(posterior_samples, axis=1)
                predictions.append(stacked)
            else:
                predictions.append(posterior_samples[0])

        return np.vstack(predictions)


    def evaluate(self, inputs, targets, sample_size=10, **kwargs):
        """
            Evaluate a model on given input data and targets.
        """
        
        if len(inputs) != len(targets):
            raise ValueError("Error in McDropout.evaluate(). Targets and inputs not of equal length.")

        # Returns: (batch_size, sample_size, target_len) or (batch_size, target_len)
        predictions = self.__call__(inputs, sample_size=sample_size, **kwargs)
        self.logger.info("evaluate/predictions.shape: {}".format(predictions.shape))
        return self.__evaluate(predictions, targets, sample_size)


    def __evaluate(self, predictions, targets, sample_size):
        """

            Parameters:
                predictions (numpy.ndarray): The predictions made by the network of shape (batch, targets) or (batch, samples, targets)
                targets (numpy.ndarray): The target values
                sample_size (int): The number of samples taken from posterior.

            Returns:
                (list()) of values representing the accuracy and loss
        """
        
        expectation = predictions
        if len(predictions.shape) == 3:
            expectation = np.average(predictions, axis=1)

        # Will fail in regression case!!!! Add flag to function?
        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, expectation)

        # Extend dimension in binary case 
        extended = self.extend_binary_predictions(predictions)
        pred_targets = np.argmax(extended, axis=-1)

        # One-hot vector passed
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        
        # Extend target dimension (multiple sample in prediction)
        if sample_size > 1:
            targets = np.vstack([targets]*sample_size).T
        
        acc = np.mean(pred_targets == targets)
        return [np.mean(loss.numpy()), acc]


    def approx_posterior(self, predictions):
        """

        """
        # predictions -> (batch_size, num_predictions)
        predictions = self.extend_binary_predictions(predictions)
        return np.average(predictions, axis=1)


    def expectation(self, predictions):
        return predictions


    def extend_binary_predictions(self, predictions, num_classes=2):
        """
            In MC Dropout case always predictions of shape
            (batch_size, sample_size, classes) for classification 
            or (batch_size, sample_size) for binary/regression case
        """

        # Don't modify predictions shape in regression case
        if not self.is_classification():
            return predictions


        # Binary case: calculate complementary prediction and concatenate
        if self.is_binary():
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions


    # ---------------
    # Loss function
    # -----------------------

    def _nll(self, prediction):
        # Shape (batch, classes) (already reduces with np.mean)
        prediction = self.extend_binary_predictions(prediction)
        max_prediction = np.max(prediction, axis=1)
        return np.log(max_prediction)

    
    def _entropy(self, prediction):
        pass


    # -----
    # Acquisition functions
    # ---------------------------

    def get_query_fn(self, name):

        if name == "max_entropy":
            return self.__max_entropy
        
        if name == "bald":
            return self.__bald
        
        if name == "max_var_ratio":
            return self.__max_var_ratio

        if name == "std_mean":
            return self.__std_mean


    def __max_entropy(self, data, sample_size=5, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (UnlabeledPool) The pool of unlabeled data to select
        """
        # Create predictions
        predictions = self.__call__(data, sample_size=sample_size)
        posterior = self.approx_posterior(predictions)
        
        # Absolute value to prevent nan values and + 0.001 to prevent infinity values
        log_post = np.log(np.abs(posterior) + .001)

        # Calculate max-entropy
        return  -np.sum(posterior*log_post, axis=1)


    def __bald(self, data, sample_size=10, **kwargs):
        self.logger.info("------------ BALD -----------")
        # predictions shape (batch, num_predictions, num_classes)
        self.logger.info("_bald/data-shape: {}".format(data.shape))
        predictions = self.__call__(data, sample_size=sample_size)

        self.logger.info("_bald/predictions-shape: {}".format(predictions.shape))
        posterior = self.approx_posterior(predictions)
        self.logger.info("_bald/posterior-shape: {}".format(posterior.shape))

        first_term = -np.sum(posterior*np.log(np.abs(posterior) + .001), axis=1)

        # Missing dimension in binary case?
        predictions = self.extend_binary_predictions(predictions)
        second_term = np.sum(np.mean(predictions*np.log(np.abs(predictions) + .001), axis=1), axis=1)

        self.logger.info("_bald/first-term-shape: {}".format(first_term.shape))
        self.logger.info("_bald/second-term-shape: {}".format(second_term.shape))
        return first_term + second_term


    def __max_var_ratio(self, data, sample_size=10, batch_size=1, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """

        # (batch, sample, num classses)
        # (batch, num_classes)
        predictions = self.__call__(data, sample_size=sample_size, batch_size=batch_size)
        posterior = self.approx_posterior(predictions)

        # Calcualte max variation rations
        return 1 + posterior.max(axis=1)


    def __std_mean(self, data, sample_size=10, batch_size=1, **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        # TODO: generalize for n-classes For binary classes
        predictions = self.__call__(data, sample_size=sample_size, batch_size=1)

        posterior = self.approx_posterior(predictions) 
        squared_posterior = np.power(posterior, 2)
        post_to_square = self.expectation(squared_posterior) # TODO: Solve error here. How to restructure?

        exp_to_square = np.power(posterior, 2)
        std_per_class = np.square(post_to_square-exp_to_square)
        return np.sum(std_per_class, axis=1)