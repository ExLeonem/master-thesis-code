from .library import Library

class TensorFlow(Library):
    """
        Tensorflow/Keras deep learning library.

        File formats: .csv, .npy, .tfrecords, .petastorm
        Data source: local filesystem, HDFS, S3
        Model serving file formats: .pb
    """

    def __init__(self):
        self.MODULE_NAME = "tensorflow"
        super(TensorFlow, self).__init__(self.MODULE_NAME)
        self.__model_types = self.__get_module_types()


    def is_available(self):
        return self.loader_exists

    
    def of_model(self, model):
        
        


    def export(self, model):
        pass


    def write(self, data, path, format=".pb"):
        pass

    
    def __get_module_types(self):
        if self.base_module is None:
            return []
        
        return [
            self.base_module.Module,
            self.base_module.keras.Model,
            self.base_module.keras.Sequential
        ]