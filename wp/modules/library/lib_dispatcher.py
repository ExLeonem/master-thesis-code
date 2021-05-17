import Torch
import TensorFlow


class LibraryDispatcher:

    def __init__(self):
        self.libraries = [
            Torch(),
            TensorFlow()
        ]


    def __iter_through(self, action):
        
        for lib in libraries:
            action(lib)

    
    def model_of_library(self, model):
        """
            Get the library of which the model is made.

            Parameter:
                model (nn.Module | nn.Sequential | keras.Layer | tf.Module) The model

            Returns:
                (Library | None) library of the model or None if no match was found.
        """
        
        for lib in self.libraries:
            
            if lib.of_model(model):
                return lib

        return None


if __name__ == "__main__":
    dispatcher = LibraryDispatcher()
