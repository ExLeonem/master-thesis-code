import os, sys, importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
PARENT_MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(PARENT_MODULE_PATH)

from library import Library, LibraryDispatcher, LibType
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

        Attributes:
            _model (): Tensorflow or pytorch module.
            _config (): Model configuration
            _mode (): The mode the model is in 'train' or 'test'/'eval'.
            _model_type (): The model type
            _library (): The library to use tensorflow, pytorch
            _checkpoints (): Created checkpoints.
    """

    def __init__(self, model, config, mode=Mode.TRAIN, model_type=None):
        self._model = model
        self._config = config
        self._mode = mode
        self._model_type = model_type
        self._library = self.__init_library_of(model)
        self._checkpoints = Checkpoint(self._library)


        # Needed values
        self.expectation = None
        self.std = None
        self.posterior = None
    
    
    def __call__(self, *args, **kwargs):
        pass


    def predict(self, inputs, **kwargs):
        """
            Approximate predictive distribution.

            Parameter:
                inputs (numpy.ndarray): The inputs for the approximation

        """
        # No library was set for the given model
        if self._library is None:
            raise ValueError("Error in BayesModel.predict/2. Missing library.")
        
        return self._library.predict(self._model, inputs, **kwargs)


    def fit(self, **kwargs):
        """
            Fit the model to the given data.

            Parameters:
                inputs (numpy.ndarray): The inputs to train the model on. (default=None)
                targets (numpy.ndarray): The targets to fit the model to. (default=None)

            Returns:

        """
        # return self._library.fit(self._model, **kwargs)
        inputs = dict.get(kwargs, "inputs")
        targets = dict.get(kwargs, "targets")
        batch_size = dict.get(kwargs, "batch_size")
        epochs = dict.get(kwargs, "epochs")

        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            # TODO: implement for pytorch
            fit_routine_callback = dict.get(kwargs, "routine")
            if fit_routine_callback is None:
                # Execute default
                pass
            else:
                pass

            return []

        elif lib_type == LibType.TENSOR_FLOW:
            verbose = dict.get(kwargs, "verbose")
            return self._model.fit(
                x=inputs, 
                y=targets, 
                batch_size=batch_size, 
                epochs=epochs, 
                verbose=verbose
            )

        else:
            raise ArgumentError("Error in Model.fit(**kwargs). No implementation for library type {}".format(lib_type))


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
        self._library.disable_batch_norm(self._model)


    def clear_session(self):
        self._library.clear_session()


    # -----------------
    # Setter/-Getter
    # --------------------------

    def get_library(self):
        return self._library

    
    def get_mode(self):
        return self._mode


    def set_mode(self, mode):
        if self._library is None:
            raise ArgumentError("Error in BayesModel.set_mode(mode). Could not set the mode, missing library.")
        
        self._library.set_mode(self.model, mode)
        self._mode = mode


    # ---------------
    # Dunder
    # ----------------------

    def __eq__(self, other):
        return other == self._model_type


    def __str__(self, other):
        return ""    
