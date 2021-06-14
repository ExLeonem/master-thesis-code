import os, sys, importlib
import logging as log

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

    def __init__(
        self, model, config, 
        mode=Mode.TRAIN, 
        model_type=None, 
        classification=True, 
        num_classes=2):

        self._model = model
        self._config = config
        self._mode = mode
        self._model_type = model_type
        self._library = self.__init_library_of(model)
        self._checkpoints = Checkpoint(self._library)

        self.__classification = classification
        self.__num_classes = num_classes
    
    
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

        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            return None

        elif lib_type == LibType.TENSOR_FLOW:
            return self._model(inputs, training=self.in_mode(Mode.TRAIN))
        
        raise ValueError("Error in Model.predict(self, inputs, **kwargs). Missing library implementation for {}".format(lib_type))

    
    def evaluate(self, inputs, targets, **kwargs):
        """
            Evaluate a model on given input data and targets.

            Parameters:
                inputs (numpy.ndarray):
                targets (numpy.ndarray):

            Returns:
                (list) A list with two values. [loss, accuracy]  
        """

        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            return self._model.evaluate(inputs, targets, verbose=0, **kwargs)

        # No implementation for library type
        raise ValueError("Error in Model.fit(**kwargs).\
         No implementation for library type {}".format(lib_type))


    def fit(self, *args, **kwargs):
        """
            Fit the model to the given data. The **kwargs are library depending.

            Args:
                x (numpy.ndarray): The inputs to train the model on. (default=None)
                y (numpy.ndarray): The targets to fit the model to. (default=None)
                batch_size (int): The size of each individual batch

            Returns:

        """

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
            return self._model.fit(*args, **kwargs)

        # No implementation for library type
        raise ValueError("Error in Model.fit(**kwargs).\
         No implementation for library type {}".format(lib_type))


    def compile(self, *args, **kwargs):
        """
            Compile the model if needed
        """
        lib_type = self._library.get_lib_type()
        if lib_type == LibType.TORCH:
            # No compilation for pytorch model needed
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            self._model.compile(**kwargs)

        else:
            # No implementation for library type available
            raise ValueError("Error in Model.compile(self, *args, **kwargs). Missing library implementation for {}.".format(lib_type))


    def prepare_predictions(self, predictions):
        """
            Extend predictions for binary classification case.

            Parameters:
                predictions (numpy.ndarray): The predictions made by the model

            Returns:
                (numpy.ndarray) The extended numpy array
        """
        return predictions

    
    def map_eval_values(self, values):
        """
            Create a dictionary mapping for evaluation metrics.

            Parameters:
                values (any): Values received from model.evaluate

            Returns:
                (dict) The values mapped to a specific key.
        """
        metric_names = self._model.metrics_names
        return dict(zip(metric_names, values))


    
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

        lib_type = self._library.get_lib_type()

        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            base_module = self._library.get_base_module()
            base_module.keras.backend.clear_session()
            # self._library.clear_session()


    # --------------
    # Checkpoint creation/loading
    # ------------------------------

    def empty_checkpoint(self):
        return self._checkpoints.empty()

    def new_checkpoint(self):
        self._checkpoints.new(self._model)

    
    def load_checkpoint(self, iteration=None):
        self._checkpoints.load(self._model, iteration)


    def clear_checkpoints(self):
        self._checkpoints.clean()


    def save_weights(self):
        path = self._checkpoints.PATH
        lib_type = self._library.get_lib_type()

        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            self._model.save_weights(path)


    def load_weights(self):
        path = self._checkpoints.PATH
        lib_type = self._library.get_lib_type()

        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            self._model.load_weights(path)


    def empty_weights(self):
        try:
            self.load_weights()
            return False

        except:
            return True


    # --------------
    # Access important flags for predictions
    # -----------------------------

    def in_mode(self, mode):
        return self._mode == mode


    def is_classification(self):
        return self.__classification


    # ---------------
    # Acquisition functions
    # --------------------------

    def get_query_fn(self, name):
        """
            Get model specific acquisition function.
        """
        pass


    def __max_entropy(self, data, **kwargs):
        pass

    def __bald(self, data, **kwargs):
        pass

    def __max_var_ratio(self, data, **kwargs):
        pass

    def __std_mean(self, data, **kwargs):
        pass
        

    # -----------------
    # Setter/-Getter
    # --------------------------

    def get_library(self):
        return self._library

    
    def get_mode(self):
        return self._mode


    def set_mode(self, mode):
        if self._library is None:
            raise ValueError("Error in BayesModel.set_mode(mode). Could not set the mode, missing library.")
        
        self._library.set_mode(self.model, mode)
        self._mode = mode


    def get_num_classes(self):
        return self.__num_classes

    # ---------------
    # Dunder
    # ----------------------

    def __eq__(self, other):
        return other == self._model_type


    def __str__(self, other):
        return ""    

