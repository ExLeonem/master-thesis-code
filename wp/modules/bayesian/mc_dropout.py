import numpy as np
import logging as log
from . import  BayesModel, ModelType, Mode



class McDropout(BayesModel):
    """
        Encapsualte mc droput model.

        TODO:
            - is batch norm disabled?

    """

    def __init__(self, model, config=None, **kwargs):
        super().__init__(model, config, model_type=ModelType.MC_DROPOUT, **kwargs)
        

    def predict(self, inputs, runs=10, **kwargs):
        # disable batch norm
        super().disable_batch_norm()
        """

        """
        output = None
        for run in range(runs):
            result = super().predict(inputs, **kwargs)

            # Set initial shape of the ouput
            if output is None:
                output = np.zeros(tuple([runs] + list(result.shape)))
            
            output[run] = result

        super().clear_session()
        output.reshape(tuple([len(inputs), runs] + list(result.shape[2:])))
        return self.prepare_predictions(output)


    def posterior(self, predictions):
        """
            Approximation of 
        """
        # predictions -> (batch_size, num_predictions)
        return np.average(predictions, axis=1)


    def expectation(self, predictions):
        return predictions


    def prepare_predictions(self, predictions, num_classes=2):
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
            # bin_alt_class = np.expand_dims(bin_alt_class, axis=-1)
            # predictions = np.expand_dims(predictions, axis=-1)

            # Concatenate predictions
            class_axis = len(predictions.shape) + 1
            predictions = np.concatenate([predictions, bin_alt_class], axis=len(predictions.shape)-1)
        
        return predictions



    # # ---------------
    # # Utilties
    # # ------------------

    # def disable_batch_norm(model):
    # """
    #     Disable batch normalization for activation of dropout during prediction.

    #     Parameters:
    #         - model (tf.Model) Tensorflow neural network model.
    # """
    
    # disabled = False
    # for l in model.layers:
    #     if l.__class__.__name__ == "BatchNormalization":
    #         disabled = True
    #         l.trainable = False

    # if disabled:
    #     print("Disabled BatchNorm-Layers.")
