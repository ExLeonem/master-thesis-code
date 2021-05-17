from abc import ABC, abstractmethod
from enum import Enum

import .utils as butils
from ..library import Library


class Mode(Enum):
    TRAIN
    TEST
    EVAL


class ModelType(Enum):
    """
        Different bayesian model types.
    """
    MC_DROPUT
    MOMENT_PROPAGATION



class BayesModel(ABC):
    """
        Base class for encapsulation of a bayesian deep learning model. 
    """

    def __init__(self, model, config, mode=Mode.TRAIN, model_type=None):
        self.__model = model
        self.__config = config
        self.__mode = mode
        self.__model_type = model_type
        self.__library = self.__init_library_of(model)


    def predict(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        # No library was set for the given model
        if not self.__library is None:
            raise ValueError("Error in BayesModel.predict/2. Missing library.")
        
        return self.__library.predict(self.__model, inputs, **kwargs)

    
    def __init_library_of(self, model):
        """
            Identify which library was used for the given model

            Parameters:
                model (Module | Sequential | Layer): The neural network model, built with a library.

            Returns:
                (Library) The library that was used to build the model.
        """
        dispatcher = LibraryDispatcher()
        return dispatcher.get_of_lib_of(model)


    # -----------------
    # Setter/-Getter
    # --------------------------

    def get_library(self):
        return self.__library

    
    def get_mode(self):
        return self.__mode


    def set_mode(self, mode):
        if self.__library is None:
            raise ArgumentError("Error in BayesModel.set_mode/1. Could not set the mode, missing library.")
        
        self.__library.set_mode(self.model, mode)
        self.__mode = mode


    # ---------------
    # Dunder
    # ----------------------

    def __eq__(self, other):
        return other == self.__model_type
    