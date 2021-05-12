import numpy as np


class DataPool:
    """
        Managing pools of data.

        TODO:
            - [ ] Save using indices and np.zeros()

        Parameters:
            - inputs (list | np.ndarray) Input values
            - targets (list | np.ndarray) The target values
    """

    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, key):
        print(key)
        return self.data[key]



class LabeledPool(DataPool):
    """
        Create a pool of labeled data.
        Inititally no data in the pool is labeled. 

        Parameters:
            - data (np.ndarray) Data to be used for the pool of labeled data.
    """

    def __init__(self, data):
        self.running_idx = 0
        self.indices = np.zeros(data.shape[0])
        self.labels = np.zeros(data.shape[0])
        super(data)
    


    def __put_batch_batch(self, indices, labels):
        """
            Put a batch of data into the pool.
        """

        if data_indices.shape != labels.shape:
            raise ArgumentError("Shape of indices and labels do not match")

        running_idx_length = len(data_indices)
        new_running_idx = self.running_idx + running_idx_length
        if new_running_idx > len(self.labels):
            raise ArugmentError("Can't update pool. Pool is full.")

        self.indices[self.running_idx:new_running_idx] = indices
        self.labels[self.running_idx:new_running_idx] = labels
        self.running_idx = new_running_idx



    def __setitem__(self, key, label):
        print(key)
        pass

    def __getitem__(self, key):
        print(key)
        pass


    def put(self, index, label):
        """
            Put a new set of labeled information into
            the data pool.

            Parameters:
                - index (int | np.ndarray) - Indices of the data
        """

        if self.running_idx > len(self.index):
            raise ArgumentError("Can't update pool. Pool is full.")

        if isinstance(index, np.ndarray) or isinstance(label, np.ndarray):
            self.__set_with_array(index, label)

        else:
            self.indices[running_idx] = index
            self.labels[running_idx] = label
            self.running_idx += 1

    def get(self):
        return self.indices[:self.running_idx], self.labels[:self.running_idx]
