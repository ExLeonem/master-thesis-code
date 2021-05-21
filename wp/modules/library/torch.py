import os, sys, importlib

# DIR_PATH = os.path.abspath(os.path.realpath(__file__))
# PARENT_PATH = os.path.join(DIR_PATH, "..")
# sys.path.append(PARENT_PATH)

from .library import Library, LibType
# from bayesian import Mode

class Torch(Library):
    """
        Pytorch deep learning library

        Training File formats: .csv, .npy, .petastorm
        Data sources: local filesystem, HDFS (petastorm), S3
        Model serving file formats: .pt
    """

    def __init__(self):
        self.MODULE_NAME = "torch"
        super(Torch, self).__init__(self.MODULE_NAME, LibType.TORCH)
        self.module_types = self.__get_module_types()


    def of(self, model):        
        """
            Check if given module is a pytorch neural network.
        
            Parameter:
                model (any): A neural network module    

            Returns:
                (bool) True if module based on pytorch Module or Sequential class else False
        """

        # library is not avaible at all
        if not self.is_available():
            return False

        # Is model of one of the given module types?
        for layer in self.__module_types:
            if isinstance(model, layer):
                return True
        
        return False

    
    def set_mode(self, model, mode=None):
        """

        """
        # if mode == Mode.TRAIN:
        #     model.train()

        # else:
        #     model.eval()
        

    def predict(self, model, inputs, **kwargs):
        return model(inputs)


    def disable_batch_norm(self, model):
        pass

    def clear_session(self):
        pass


    def export(self, model):
        pass
    

    def parse(self, model):
        pass


    def write(self, model, path, format=".pt"):
        pass


    # ----------------
    # Setter/-Getter
    # ------------------------
    
    def __get_module_types(self):
        """
        Get the available module types for uploads.

        """
        if self.base_module is None:
            return []

        return [
            self.base_module.nn.Module,
            self.base_module.nn.Sequential
        ]
