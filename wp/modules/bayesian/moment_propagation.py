import os, sys, importlib
import numpy as np

from . import BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import library
importlib.reload(library)
from library import LibType

import mp.MomentPropagation as mp



class MomentPropagation(BayesModel):
    """
        Takes a regular MC Dropout model as input, that is used for fitting.
        For evaluation a moment propagation model is created an used 

    """

    def __init__(self, model, config=None, **kwargs):
        model_type = ModelType.MOMENT_PROPAGATION
        mp_model = self.__create_mp_model(model)
        super(MomentPropagation, self).__init__(mp_model, config, model_type=model_type, **kwargs)


    def __create_mp_model(self, model):
        """
            Transforms the set base model into an moment propagation model.
        """
        _mp = mp.MP()
        return _mp.create_MP_Model(model=model, use_mp=False, verbose=True)



    def variance(self, predictions):
        expectation, variance = predictions

        variance = self.extend_binary_predictions(variance)
        return self.__cast_tensor_to_numpy(variance)


    def expectation(self, predictions):
        expectation, variance = predictions
        
        expectation = self.extend_binary_predictions(expectation)
        return self.__cast_tensor_to_numpy(expectation) 


    # --------
    # Utilities
    # ---------------

    def extend_binary_predictions(self, predictions):
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

    def get_query_fn(self, name):

        if name == "max_entropy":
            return self.__max_entropy
        
        if name == "bald":
            return self.__bald
        
        if name == "max_var_ratio":
            return self.__max_var_ratio

        if name == "std_mean":
            return self.__std_mean


    def __max_entropy(self, data, **kwargs):
        # Expectation and variance of form (batch_size, num_classes)
        # Expectation equals the prediction
        predictions = self.predict(data)

        # Need to scaled values because zeros
        class_probs = self.expectation(predictions)
        
        class_prob_logs = np.log(np.abs(class_probs) + .001)
        return -np.sum(class_probs*class_prob_logs, axis=1)

    
    def __bald(self, data, **kwargs):
        """
            [ ] Check if information about variance is needed here. Compare to mc dropout bald.
        """
        predictions = self.predict(data)
        expectation = self.expectation(predictions)
        variance = self.variance(predictions)


    def __max_var_ratio(self, data, **kwargs):
        predictions = self.predict(data)
        expectation = self.expectation(predictions)

        col_max_indices = np.argmax(expectation, axis=1)        
        row_indices = np.arange(len(data))
        max_var_ratio = 1-expectation[row_indices, col_max_indices]
        return max_var_ratio

    
    def __std_mean(self, data, **kwargs):
        pass