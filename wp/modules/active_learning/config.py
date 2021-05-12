"""
TODO:
        - [ ] Add posibility to add default configurations to be filled if nothing supplied
"""



class Config:
    """
        Configuration to clean up function calls and encapsulate
        connected configurations.

        Keyword-arguments:
            - defaults (object) Reserved key for default configuration.
            
    """

    def __init__(self, **kwargs):
        self.kwargs = self.__set_defaults(kwargs)


    def __set_defaults(self, kwargs):
        """
            Set default values for a configuration object.

            Parmeters:
                - kwargs (dict) Keyword arguments passed to Config.

            Returns:
                - (dict) updated keyword arguments.
        """

        # defaults existing?
        defaults = dict.get(kwargs, "defaults")
        if defaults is None:
            return kwargs

        # Are defaults given as dict?
        if not isinstance(defaults, dict):
            raise ArgumentError("Can't set defaults. 'defaults' needs to be of type 'dict'.")

        # Set default values
        for key, default_value in defaults.items():
            if dict.get(kwargs, key) is None:
                kwargs[key] = default_value
            
        return kwargs


    # Access function
    def __getitem__(self, key):
        return self.kwargs[key]



class TrainConfig(Config):
    """
        Setting training configuration.

        Keyword-arguments:
            - batch_size (int) Set the batch size of the training set
            - epochs (int) The number of epochs to train
            - optimizer (str | object) The optimizer to be used
            - loss (str | object) The loss to be used
            - metrics (list(str)) A list of metrics to output while training
    """

    def __init__(self, **kwargs):
        defaults = {
            "batch_size": 40,
            "epochs": 100,
            "optimizer": "adadelta",
            "loss": "binary_entropy",
            "metrics": ["accuracy"]
        }

        super(TrainConfig, self).__init__(defaults=defaults, **kwargs)

    # def __getitem__(self, key):
    #     return super().__getitem__(key)

    
    def __repr__(self):
        print("hey")