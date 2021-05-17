from .library import Library
from ..bayesian.model import Mode

class Torch(Library):
    """
        Pytorch deep learning library

        Training File formats: .csv, .npy, .petastorm
        Data sources: local filesystem, HDFS (petastorm), S3
        Model serving file formats: .pt
    """

    def __init__(self, model):
        self.MODULE_NAME = "torch"
        super(Torch, self).__init__(self.MODULE_NAME)
        self.module_types = self.__get_module_types()


    def of_model(self, model):        
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

        try:
            
            # Model of type nn.Module?
            if isinstance(model, self.base_module.nn.Module):
                return True

            # Model of type nn.Sequential?
            elif isinstance(model, self.base_module.nn.Sequential):
                return True

        # Sub-attribute of library not existent
        except AttributeError:
            return False
        
        return False

    
    def set_mode(self, model, mode=None):
        """

        """
        if mode == Mode.TRAIN:
            model.train()

        else:
            model.eval()
        


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
            self.base_module.Sequential
        ]
