import os, sys, importlib
import numpy as np
import logging as log
from . import  BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import tensorflow as tf



class McDropout(BayesModel):
    """
        Encapsualte mc droput model.

        TODO:
            - is batch norm disabled?

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config, model_type=ModelType.MC_DROPOUT, **kwargs)

        # disable batch norm
        super().disable_batch_norm()


    def predict(self, inputs, n_times=10, **kwargs):
        """
            Predict n_times (n_times) for every datapoint in inputs.

            Parameters:
                inputs (numpy.ndarray): Input data to predict for. (Datapoints)
                n_times (int): The number of predictions to do per datapoint.

            Returns:
                (numpy.ndarray) of shape (num_of_datapoints, ...output_shape, n_times)
        """
        output = None
        for run in range(n_times):
            result = super().predict(inputs, **kwargs)

            # Set initial shape of the ouput
            if output is None:
                output = np.zeros(tuple([n_times] + list(result.shape)))
            
            output[run] = result


        # Reshape to put run dimension to last axis
        output_shape = None
        if output.shape[-1] == 1:
            # Binary case last dimension can be omited
            output_shape = tuple([len(inputs), n_times] + list(result.shape[2:]))

        else:
            output_shape = tuple([len(inputs), n_times] + list(result.shape[1:]))

        output = output.reshape(output_shape)
        return output


    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluate a model on given input data and targets.
        """
        # Create predictions
        predictions = self.batch_prediction(inputs, **kwargs)
        self.logger.info("evaluate/predictions.shape: {}".format(predictions.shape))
        return self.__evaluate_tf(predictions, targets)

    def __evaluate_tf(self, predictions, targets):

        # Returns dim: (batch, ) in binary case, else: (batch, nn_output_dim)
        expectation = np.average(predictions, axis=1)

        # Will fail in regression case!!!! Add flag to function?
        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, expectation)

        # Extend dimension in binary case 
        extended = self.extend_binary_predictions(expectation)

        labels = np.argmax(extended, axis=1)
        acc = np.mean(labels == targets)
        return [np.mean(loss.numpy()), acc]


    def approx_posterior(self, predictions):
        """

        """
        # predictions -> (batch_size, num_predictions)
        predictions = self.extend_binary_predictions(predictions)

        # if not self.is_binary():
        #     self.logger("Calculate mean of binary case")
            # 
            # if len(predictions.shape) == 2:
            #     self.logger.warn("")

            # return np.average(predictions, axis=1)

        return np.average(predictions, axis=1)


    def expectation(self, predictions):
        return predictions


    def extend_binary_predictions(self, predictions, num_classes=2):
        """
            In MC Dropout case always predictions of shape
            (batch_size, n_times, classes) for classification 
            or (batch_size, n_times) for binary/regression case
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


    def __max_entropy(self, data, n_times=5, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (UnlabeledPool) The pool of unlabeled data to select
        """
        # Create predictions
        predictions = self.predict(data, n_times=n_times)
        posterior = self.approx_posterior(predictions)
        
        # Absolute value to prevent nan values and + 0.001 to prevent infinity values
        log_post = np.log(np.abs(posterior) + .001)

        # Calculate max-entropy
        return  -np.sum(posterior*log_post, axis=1)


    def __bald(self, data, n_times=10, **kwargs):
        self.logger.info("------------ BALD -----------")
        # predictions shape (batch, num_predictions, num_classes)
        self.logger.info("_bald/data-shape: {}".format(data.shape))
        predictions = self.predict(data, n_times=n_times)

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


    def __max_var_ratio(self, data, n_times=10, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """

        # (batch, sample, num classses)
        # (batch, num_classes)
        predictions = self.predict(data, n_times=n_times)
        posterior = self.approx_posterior(predictions)

        # Calcualte max variation rations
        return 1 + posterior.max(axis=1)


    def __std_mean(self, data, n_times=10, **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        # TODO: generalize for n-classes For binary classes
        predictions = self.predict(data, n_times=n_times)

        posterior = self.approx_posterior(predictions) 
        squared_posterior = np.power(posterior, 2)
        post_to_square = self.expectation(squared_posterior) # TODO: Solve error here. How to restructure?

        exp_to_square = np.power(posterior, 2)
        std_per_class = np.square(post_to_square-exp_to_square)
        return np.sum(std_per_class, axis=1)