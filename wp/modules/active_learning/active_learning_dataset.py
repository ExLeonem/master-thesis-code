
from sklearn.model_selection import train_test_split


class ActiveLearningDataset:
    """
        Active learning dataset.

        Parameters:
            inputs (numpy.ndarray): The model inputs.
            targets (numpy.ndarray): The targets, labels or values.

            train_size (float): 
            

    """


    def __init__(self, inputs, targets, train_size=.75, test_size=.15, val_size=.10):

        # Set train set
        if isinstance(train_size, float):
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, test_size=1-train_size)

        if isinstance(train_size, int)
            self.x_train, x_test, self.y_train, y_test = train_test_split(inputs, targets, train_size=train_size)

        # Set test and validation set
        if isinstance(test_size, float) and isinstance(val_size, float):
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test, y_test, test_size=test_size/(test_size+val_size))
        
        if isinstance(test_size, int):
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test, y_test, test_size=test_size)

        

    def get_train_split(self):
        return (self.x_train, self.y_train)

    
    def get_test_split(self):
        return (self.x_test, self.y_test)

    
    def get_val_split(self):
        return (self.x_val, self.y_val)

