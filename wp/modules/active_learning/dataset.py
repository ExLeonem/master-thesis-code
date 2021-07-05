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
            init_size (int): The initial size of labeled inputs in the pool.
            train_size (float|int): Size of the train split.
            test_size (float|int): Size of the test split.
            val_size (float|int): Size of the validation split.
    """


    def __init__(
        self, 
        inputs,
        targets,
        init_size=0,
        train_size=.75, 
        test_size=None, 
        val_size=None
    ):

        self.pseudo = True
        self.init_size = init_size

        if len(inputs) != len(targets):
            raise ValueError("Error in Dataset.__init__(). Can't initialize dataset. Length of inputs and targets are not equal.")

        if train_size == 1 and test_size is None and val_size is None:
            self.x_train = inputs
            self.y_train = targets

        self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, train_size=train_size)

        if test_size is not None:
            self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(x_test, y_test, train_size=test_size)

        if val_size is not None:
            self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(x_test, y_test, test_size=val_size)


        # else:
            
        #     max_num_datapoints = len(inputs)
        #     true_test_size = self.__adapt_test_size(max_num_datapoints, train_size, test_size)
        #     self.pseudo = True # Experimentl active learning run?
        #     self.init_size = init_size
        #     self.train_split, (x_temp, y_temp) = self.__split(inputs, targets, train_size, test_size) 

        #     true_val_size = self.__adapt_val_size(test_size, true_test_size, val_size)
        #     self.test_split, self.val_split = self.__split(x_temp, y_temp, true_test_size) 


    # ----------
    # Utilities
    # -------------------

    def is_pseudo(self):
        return self.pseudo

    def has_test_set(self):
        return hasattr(self, 'x_test') and hasattr(self, 'y_test')

    def has_eval_set(self):
        return hasattr(self, 'x_val') and hasattr(self, 'y_val')

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

    def get_eval_split(self):
        return (self.x_val, self.y_val)