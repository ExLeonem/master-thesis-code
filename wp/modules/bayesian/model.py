import .utils as butils
from abc import ABC, abstractmethod
from enum import Enum


class Mode(Enum):
    TRAIN
    TEST
    EVAL


class ModelType(Enum):
    """
        Different bayesian model types.
    """
    
    MC_DROPUT
    MP_PROPAGATION



class BayesModel(ABC):
    """
        Base class for encapsulation of a bayesian deep learning model. 
    """

    def __init__(self, model, config, mode=Mode.TRAIN, model_type=None):
        self.model = model
        self.config = config
        self.mode = mode
        self.model_type = model_type
    

    
    def approx(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        pass






    