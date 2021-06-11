import os, sys, importlib
import numpy as np
import logging as log
from . import  BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import library
importlib.reload(library)
from library import LibType



class McDropout(BayesModel):
    """
        Encapsualte mc droput model.

        TODO:
            - is batch norm disabled?

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config, model_type=ModelType.MC_DROPOUT, **kwargs)


    def predict(self, inputs, runs=10, **kwargs):
        """

        """
        # disable batch norm
        super().disable_batch_norm()
        output = None
        for run in range(runs):
            result = super().predict(inputs, **kwargs)

            # Set initial shape of the ouput
            if output is None:
                output = np.zeros(tuple([runs] + list(result.shape)))
            
            output[run] = result

        super().clear_session()
        output = output.reshape(tuple([len(inputs), runs] + list(result.shape[2:])))
        return output

        # return self.extend_binary_predictions(output)


    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluate a model on given input data and targets.
        """

        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:

            batch_size = dict.get(kwargs, "batch_size")
            if batch_size is None:
                predictions = self.predict(inputs, **kwargs)
                return self.__evaluate_tf(predictions, targets)
            

            # Iterate over inputs and targets batchwise
            predictions = None
            for start_idx in range(0, len(inputs), batch_size):
                
                end_idx = start_idx + batch_size
                sub_result = self.predict(inputs[start_idx:end_idx], **kwargs)

                # Create array to hold prediction results
                if predictions is None:
                    shape_without_batch = sub_result.shape if batch_size == 1 else sub_result.shape[1:]
                    result_shape = [len(inputs)] + list(shape_without_batch)
                    predictions = np.zeros(result_shape)

                predictions[start_idx:end_idx] = sub_result

            return self.__evaluate_tf(predictions, targets)


            # predictions = self.predict(inputs, **kwargs)
            # posterior_approx = np.average(predictions, axis=1)

            # # Will fail in regression case!!!! Add flag to function?
            # loss_fn = self._library.get_base_module().keras.losses.get(self._model.loss)
            # loss = loss_fn(targets, posterior_approx)

            # # 
            # labels = np.argmax(self.extend_binary_predictions(posterior_approx), axis=1)
            # true_labels = (labels == targets).sum()
            # acc = true_labels/len(targets)

            # return [loss.numpy(), acc]

            # return self._model.evaluate(inputs, targets, verbose=0, **kwargs)

        # No implementation for library type
        raise ValueError("Error in Model.fit(**kwargs).\
         No implementation for library type {}".format(lib_type))


    def __evaluate_tf(self, predictions, targets):

        posterior_approx = np.average(predictions, axis=1)

        # Will fail in regression case!!!! Add flag to function?
        loss_fn = self._library.get_base_module().keras.losses.get(self._model.loss)
        loss = loss_fn(targets, posterior_approx)

        # 
        labels = np.argmax(self.extend_binary_predictions(posterior_approx), axis=1)
        true_labels = (labels == targets).sum()
        acc = true_labels/len(targets)

        return [loss.numpy(), acc]


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
            (batch_size, runs, classes) for classification 
            or (batch_size, runs) for binary/regression case
        """

        # Don't modify predictions shape in regression case
        if not self.is_classification():
            return predictions


        # Binary case: calculate complementary prediction and concatenate
        if self.get_num_classes() == 2:
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions


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


    def __max_entropy(self, data, runs=5, **kwargs):
        """
            Select datapoints by using max entropy.

            Parameters:
                model (tf.Model) The tensorflow model to use for selection of datapoints
                unlabeled_pool (UnlabeledPool) The pool of unlabeled data to select
        """
        # Create predictions
        predictions = self.predict(data, runs=runs)
        posterior = self.approx_posterior(predictions)
        
        # Absolute value to prevent nan values and + 0.001 to prevent infinity values
        log_post = np.log(np.abs(posterior) + .001)

        # Calculate max-entropy
        return  -np.sum(posterior*log_post, axis=1)


    def __bald(self, data, runs=5, **kwargs):
        
        # predictions shape (batch, num_predictions, num_classes)
        predictions = self.predict(data, runs=runs)
        posterior = self.approx_posterior(predictions)

        first_term = -np.sum(posterior*np.log(posterior), axis=1)
        second_term = np.sum(np.sum(predictions*np.log(predictions), axis=1), axis=1)/runs

        return first_term + second_term


    def __max_var_ratio(self, data, runs=10, **kwargs):
        """
            Select datapoints by maximising variation ratios.

            # (batch, predictions, classes) reduce to (batch, predictions (max-class))
            # 1 - (count of most common class / num predictions)
        """

        # (batch, sample, num classses)
        # (batch, num_classes)
        predictions = self.predict(data, runs=runs)
        posterior = self.approx_posterior(predictions)

        # Calcualte max variation rations
        return 1 + posterior.max(axis=1)


    def __std_mean(self, data, runs=10, **kwargs):
        """
           Maximise mean standard deviation.
           Check std mean calculation. Depending the model type calculation of p(y=c|x, w) can differ.
           (Kampffmeyer et al. 2016; Kendall et al. 2015)

           Todo:
            Implement distinction for different model types.
        """
        # TODO: generalize for n-classes For binary classes
        predictions = self.predict(data, runs=runs)

        posterior = self.approx_posterior(predictions) 
        squared_posterior = np.power(posterior, 2)
        post_to_square = self.expectation(squared_posterior) # TODO: Solve error here. How to restructure?

        exp_to_square = np.power(posterior, 2)
        std_per_class = np.square(post_to_square-exp_to_square)
        return np.sum(std_per_class, axis=1)