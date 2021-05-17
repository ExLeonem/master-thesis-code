import Torch
import TensorFlow


class LibraryDispatcher:
    """
        Dispatches call to library specific functions.
    """
    
    def __init__(self):
        self.libraries = [
            Torch(),
            TensorFlow()
        ]


    def get_lib_of(self, model):
        """
            Get the library of which the model is made.

            Parameter:
                model (nn.Module | nn.Sequential | keras.Layer | tf.Module) The model

            Returns:
                (Library | None) library of the model or None if no match was found.
        """
    
        # Given model based on one of specified libraries?
        for library in self.libraries:    
            if library.of(model):
                return library

        # Model does not match any library specification
        return None

