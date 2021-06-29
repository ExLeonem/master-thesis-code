from enum import Enum
from sklearn.model_selection import train_test_split


class PooledDataset:
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

        self.init_size = init_size  
        self.pseudo = True
        if targets is None:
            self.pseudo = False
            self.__split_only_inputs(inputs, train_size, test_size, val_size)

        else:
            self.__split_all(inputs, targets, train_size, test_size, val_size)


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
        y_test = None
        if isinstance(train_size, float):
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, test_size=1-train_size)
        
        if isinstance(train_size, int):
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, train_size=train_size)


        if val_size is None:
            self.x_test = x_test
            self.y_test = y_test


    def __init_splits(self, inputs, targets, train_size, test_size, val_size):
        """
            Splits the input values and targets into train, test and validation sets.

            Parameters:
                inputs (numpy.ndarray): The input values to the network.
                targets (numpy.ndarray): The targets/labels for the training/evaluation.
                train_size (int|float): Size of the train set.
                test_size (int|float): Size of the test set.
                val_size (int|float): Size of the validation set.
        """

        # Set train set
        if isinstance(train_size, float):
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, test_size=1-train_size)

        if isinstance(train_size, int):
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, train_size=train_size)



        # Set test and validation set
        if isinstance(test_size, float) and isinstance(val_size, float):
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test, y_test, test_size=test_size/(test_size+val_size))
        
        if isinstance(test_size, int):
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test, y_test, test_size=test_size)      


    def __split(self, inputs, targets, train_size, test_size, val_size):
        pass


    # ----------
    # Utilities
    # -------------------

    def has_targets(self):
        return self.pseudo



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