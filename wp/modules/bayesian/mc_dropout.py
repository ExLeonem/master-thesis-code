import numpy as np
import logging as log
from . import  BayesModel, ModelType, Mode



class McDropout(BayesModel):
    """
        Encapsualte mc droput model.

    """

    def __init__(self, model, config=None):
        super().__init__(model, config, model_type=ModelType.MC_DROPOUT)
        

    def predict(self, inputs, runs=10, **kwargs):
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
        return output.reshape(tuple([len(inputs), runs] + list(result.shape[2:])))


    def expectation(self, predictions):
        return predictions


    def posterior(self, predictions):
        """
            Approximation of 
        """
        # predictions -> (batch_size, num_predictions)
        return np.average(predictions, axis=1)

    
    def new_checkpoint(self):
        self._checkpoints.new(self._model)

    
    def load_checkpoint(self, iteration=None):
        self._checkpoints.load(self._model, iteration)




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
