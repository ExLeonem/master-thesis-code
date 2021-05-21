import os, sys, importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
PARENT_MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(PARENT_MODULE_PATH)

from library import Library, LibraryDispatcher
from . import Checkpoint
from abc import ABC, abstractmethod
from enum import Enum


class Mode(Enum):
    TRAIN=1,
    TEST=2,
    EVAL=3


class ModelType(Enum):
    """
        Different bayesian model types.
    """
    MC_DROPOUT=1,
    MOMENT_PROPAGATION=2


class BayesModel:
    """
        Base class for encapsulation of a bayesian deep learning model. 
    """

    def __init__(self, model, config, mode=Mode.TRAIN, model_type=None):
        self._model = model
        self._config = config
        self.__mode = mode
        self.__model_type = model_type
        self.__library = self.__init_library_of(model)
        self._checkpoints = Checkpoint(self.__library)


        # Needed values
        self.expectation = None
        self.std = None
        self.posterior = None
    

    def get_library(self):
        return self.__library
        
    def predict(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        # No library was set for the given model
        if self.__library is None:
            raise ValueError("Error in BayesModel.predict/2. Missing library.")
        
        return self.__library.predict(self._model, inputs, **kwargs)



    def extend_binary(self, predictions):
        """
            Extend predictions for binary classification case.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the model

            Returns:
                (numpy.ndarray) The extended numpy array
        """
        pass

    
    def __init_library_of(self, model):
        """
            Identify which library was used for the given model

            Parameters:
                model (Module | Sequential | Layer): The neural network model, built with a library.

            Returns:
                (Library) The library that was used to build the model.
        """
        dispatcher = LibraryDispatcher()
        return dispatcher.get_lib_of(model)

    def disable_batch_norm(self):
        self.__library.disable_batch_norm(self._model)

    def clear_session(self):
        self.__library.clear_session()


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
    
