import os, sys, importlib
import numpy as np

from . import BayesModel, ModelType, Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULE_PATH)

import mp.MomentPropagation as mp
import tensorflow as tf



class MomentPropagation(BayesModel):
    """
        Takes a regular MC Dropout model as input, that is used for fitting.
        For evaluation a moment propagation model is created an used 

    """

    def __init__(self, model, config=None, **kwargs):
        model_type = ModelType.MOMENT_PROPAGATION
        mp_model = self.__create_mp_model(model)
        super(MomentPropagation, self).__init__(mp_model, config, model_type=model_type, **kwargs)


    def evaluate(self, inputs, targets, **kwargs):
        self.set_mode(Mode.EVAL)
        predictions = self.batch_prediction(inputs, **kwargs)
        return self.__evaluate(predictions, targets)

    def __evaluate(self, prediction, targets):

        loss_fn = tf.keras.losses.get(self._model.loss)
        loss = loss_fn(targets, prediction)
        
        prediction = self.extend_binary_predictions(prediction)

        labels = np.argmax(prediction, axis=1)
        acc = np.mean(labels == targets)
        return [np.mean(loss.numpy()), acc]


    def map_eval_values(self, values, custom_names=None):
        metric_names = ["loss", "accuracy"] if custom_names is None else custom_names
        return dict(zip(metric_names, values))


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


    def _nll(self, prediction):
        prediction = self.extend_binary_predictions(prediction)
        max_prediction = np.max(prediction,axis=1)
        return np.log(max_prediction)


    def _entropy(self, prediction):
        prediciton = self.extend_binary_predictions(prediction)
        return np.array([-np.sum( pred_mp[i] * np.log2(pred_mp[i] + 1E-14)) for i in range(0,len(pred_mp))])


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
        if self.is_binary():
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

        values = tf.make_ndarray(values)


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
        predictions = self._model.predict(x=data)

        # Need to scaled values because zeros
        class_probs = self.expectation(predictions)
        
        class_prob_logs = np.log(np.abs(class_probs) + .001)
        return -np.sum(class_probs*class_prob_logs, axis=1)

    
    def __bald(self, data, **kwargs):
        """
            [ ] Check if information about variance is needed here. Compare to mc dropout bald.
        """
        predictions = self._model.predict(x=data)
        expectation = self.expectation(predictions)
        variance = self.variance(predictions)


    def __max_var_ratio(self, data, **kwargs):
        predictions = self._model.predict(x=data)
        expectation = self.expectation(predictions)

        col_max_indices = np.argmax(expectation, axis=1)        
        row_indices = np.arange(len(data))
        max_var_ratio = 1-expectation[row_indices, col_max_indices]
        return max_var_ratio

    
    def __std_mean(self, data, **kwargs):
        pass