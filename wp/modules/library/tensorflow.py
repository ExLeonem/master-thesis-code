from .library import Library, LibType

class TensorFlow(Library):
    """
        Tensorflow/Keras deep learning library.

        File formats: .csv, .npy, .tfrecords, .petastorm
        Data source: local filesystem, HDFS, S3
        Model serving file formats: .pb
    """

    def __init__(self):
        self.MODULE_NAME = "tensorflow"
        super(TensorFlow, self).__init__(self.MODULE_NAME, LibType.TENSOR_FLOW)
        self.__module_types = self.__get_module_types()


    def is_available(self):
        return self.loader_exists


    def of(self, model):
        
        # Library is not available at all
        if not self.is_available():
            return False

        # Model is of any of the given layers/base classes?
        for layer in self.__module_types:
            if isinstance(model, layer):
                return True
        
        # Either module not available
        # or model and layers do not match
        return False
        

    def predict(self, model, inputs, **kwargs):
        is_training = dict.get(kwargs, "training")
        return model(inputs, training=is_training)


    def clear_session(self):
        self.base_module.keras.backend.clear_session()


    def disable_batch_norm(self, model):  
        """
            Disable batch normalization for activation of dropout during prediction.

            Parameters:
                - model (tf.Model) Tensorflow neural network model.
        """
        
        disabled = False
        for l in model.layers:
            if l.__class__.__name__ == "BatchNormalization":
                disabled = True
                l.trainable = False

        if disabled:
            print("Disabled BatchNorm-Layers.")



    def export_model(self, model, path):
        model.save_weights(path)


    def load_model(self, model, path):
        model.load_weights(path)



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