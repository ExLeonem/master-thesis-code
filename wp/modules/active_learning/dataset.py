from enum import Enum
from sklearn.model_selection import train_test_split
from . import Pool

class Dataset:
    """
        Splits a dataset into tree parts. Train/Test/validation.
        The train split is used for selection of 
        

        Parameters:
            inputs (numpy.ndarray): The model inputs.
            targets (numpy.ndarray): The targets, labels or values.
            init_size (numpy.ndarray): The initial size of the LabeledPool
            train_size (float|int): Size of the train split.
            test_size (float|int): Size of the test split.
            val_size (float|int): Size of the validation split.
    """


    def __init__(
        self, 
        inputs,
        targets=None,
        init_size=0,
        train_size=.75, 
        test_size=.25, 
        val_size=None
    ):

        # Dataset splits
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        self.init_size = init_size

        # Targets passed to the dataset?
        self.pseudo = True
        if targets is None:
            self.pseudo = False
            self.pool = self.__init_pool(inputs, targets, init_size)


        # Allow splitting of dataset?
        if self.pseudo:
            self.train_size = train_size
            self.test_size = test_size
            self.val_size = val_size
        

    
    def __init_pool(self, inputs, init_size, targets=None, pseudo=False):
        return Pool(inputs)



    def __split_only_inputs(self, inputs, train_size, test_size, val_size):
        """
            Splits only input values into train, test and validation sets.

            Parameters:
                inputs (numpy.ndarray): The input values to the network.
                train_size (float|int): Size of the training set.
                test_size (float|int): Size of the test set.
                val_size (float|int): Size of the validation set.
        """

        x_test = None
        if isinstance(train_size, float):
            self.x_train, x_test = train_test_split(inputs, test_size=1-train_size)
        
        if isinstance(train_size, int):
            self.x_train, x_test = train_test_split(inputs, train_size=train_size)

        if val_size is None:
            self.x_test = x_test



    # ----------
    # Utilities
    # -------------------

    def has_targets(self):
        return self.pseudo

    
    def get_unlabeled(self):
        pass



    # -------------
    # Setter/-Getter
    # -------------------

    def get_init_size(self):
        return self.init_size

    def get_train_split(self):
        return (self.x_train, self.y_train)

    def get_train_inputs(self):
        return self.x_train

    def get_train_targets(self):
        return self.y_train

    def get_test_split(self):
        return (self.x_test, self.y_test)

    def get_test_inputs(self):
        return self.x_test

    def get_test_targets(self):
        return self.y_test

    def get_val_split(self):
        return (self.x_val, self.y_val)

    def get_val_inputs(self):
        return self.x_val

    def get_val_targets(self):
        return self.y_val