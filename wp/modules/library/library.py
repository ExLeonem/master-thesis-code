import importlib, sys
from enum import Enum

class LibType(Enum):
    TORCH=1,
    TENSOR_FLOW=2


class Library:
    """
        LibraryFactory 

        Abstract class to work with differen python libraries.

        :param base_module_name: The base module of the library.
        :type base_module_name: str
    """
    def __init__(self, base_module_name, library_type):
        super(Library, self).__init__()

        # Current python version information
        self.major = sys.version_info[0]
        self.minor = sys.version_info[1]
        
        # Check if module is available
        self._library_type = library_type
        self.loader_exists = self.__loader_exists(base_module_name)

        # Set Base module
        base = None
        if self.loader_exists:
            base = importlib.import_module(base_module_name)
        self.base_module = base


    def __loader_exists(self, module_name):
        """
            Check if the base loader for the given python library exists.

            Parameters:
                module_name (str) The name of the module

            Returns:
                (bool) Whether or not the loader exists
        """
        loader = None
        if self.major >= 3 and self.minor > 4:
            loader = importlib.util.find_spec(module_name)

        else:
            loader = importlib.find_loader(module_name)
        
        return loader is not None


    def is_available(self):
        """
            Is the library available?

            Returns:
                (bool) True or False depending wether or not library loader can be found.
        """
        return self.loader_exists


    def of_type(self, lib_type):
        return self._library_type == lib_type


    def of(self, model):
        """
            Check if model is of current library.

            Parameters:
                model (nn.Module | tf.Module) The model to check.
        """
        pass


    # ------------
    # Model prediction and training
    # ---------------------------------

    def predict(self, model, inputs, **kwargs):
        """Perform a prediction for a given model."""
        pass


    def fit(self, model, **kwargs):
        pass


    def clear_session(self):
        pass


    def disable_batch_norm(self, model):
        pass


    def export(self, model):
        """
            Parses the model into an intermediate format and returns it

            Parameters:
                model (nn.Module | nn.Sequential | tf.Module | keras.Layer) The model to export
        """
        pass


    def parse(self, model):
        pass


    def write(self, data, path, extension=None):
        """

        """
        pass


    # ---------------------
    # Prevent ever growing library module.
    # Delegate implementation to classes using library classes.
    # ------------------------------------------------------

    def register(self, callback):
        """
            Execute the callback function, injeting the library module and the library type.

            Parameters:
                callback (function): The callback to execute

            Returns:
                (any) The callback return value
        """
        return callback(self.base_module, self._library_type)

    # -----------------
    # Setter/-Getter
    # -------------------

    def __get_module_types(self):
        """
            Get all module types of this library.
        """
        pass

    
    def get_lib_type(self):
        return self._library_type
