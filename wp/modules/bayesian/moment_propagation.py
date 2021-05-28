import os, sys, importlib
import numpy as np

from . import BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import library
importlib.reload(library)
from library import LibType



class MomentPropagation(BayesModel):
    """

    """

    def __init__(self, model, config=None, **kwargs):
        model_type = ModelType.MOMENT_PROPAGATION
        super(MomentPropagation, self).__init__(model, config, model_type=model_type, **kwargs)


    def predict(self, inputs, **kwargs):
        """
            Predictions returns (E, Var)
        """
        return super().predict(inputs)

    
    def variance(self, predictions):
        expectation, variance = predictions

        variance = self.prepare_predictions(variance)
        return self.__cast_tensor_to_numpy(variance)


    def expectation(self, predictions):
        expectation, variance = predictions
        
        expectation = self.prepare_predictions(expectation)
        return self.__cast_tensor_to_numpy(expectation) 


    # --------
    # Utilities
    # ---------------

    def prepare_predictions(self, predictions, num_classes=2):
        """

        """
        # Don't modify predictions shape in regression case
        if not self.is_classification():
            return predictions


        # Binary case: calculate complementary prediction and concatenate
        if self.get_num_classes() == 2:
            bin_alt_class = (1 + np.zeros(predictions.shape)) - predictions

            # Expand dimensions for predictions to concatenate. Is this needed?
            # bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            # predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions


    
    def __cast_tensor_to_numpy(self, values):
        """
            Cast tensor objects of different libraries to
            numpy arrays.
        """

        # Values already of type numpy.ndarray
        if isinstance(values, np.ndarray):
            return values

        # Cast Tensor of different
        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            base_module = self._library.get_base_module()
            values = base_module.make_ndarray(values)
        
        else:
            raise ValueError("Error in MomentPropagation.__cast_tensor_to_numpy(self, values). Can't cast Tensor of given type, missing implementation detail.")


    # ----------------
    # Custom acquisition functions
    # ---------------------------

    def __max_entropy(self, data, **kwargs):
        """

        """
        # Expectation and variance of form (batch_size, num_classes)
        # Expectation equals the prediction
        predictions = self.predict(data)
        class_probs = self.expectation(predictions)
        class_prob_logs = np.log(expectation)

        return -np.sum(class_probs-class_probs_log)

    
    def __bald(self, data, **kwargs):
        """

        """
        predictions = self.predict(data)
        expectation = self.expectation(predictions)
        variance = self.variance(predictions)


    def __max_var_ratio(self, data, **kwargs):
        """

        """
        predictions = self.predict(data)
        expectation = self.expectation(predictions)

        max_indices = np.argmax(expectation, axis=1)        
        return 1-expectation[max_indices]

    
    def __std_mean(self, data, **kwargs):
        pass