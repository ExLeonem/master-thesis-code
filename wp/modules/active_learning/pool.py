import numpy as np

"""
TODO:
    - [ ] Save using indices and np.zeros()
"""


class DataPool:
    """
        Manage a pool of data.

        Parameters:
            inputs (list | numpy.ndarray): Input values
            targets (list | numpy.ndarray): The target values
    """

    def __init__(self, data):
        self.data = data

    
    def __getitem__(self, index):
        """
            Return data at given index.

            Parameters:
                index (slice | int): The index from which to take the data

            Returns:
                numpy.ndarray: The data sliced or at given index.
        """
        return self.data[index]



class LabeledPool(DataPool):
    """
        Create a pool of labeled data.
        Inititally no data in the pool is labeled. 

        Parameters:
            data (numpy.ndarray): Data to be used for the pool of labeled data.
    """

    def __init__(self, data):
        self.indices = np.zeros(data.shape[0]) - 1
        self.labels = np.zeros(data.shape[0])
        super(LabeledPool, self).__init__(data)
    

    def __put_batch_batch(self, indices, labels):
        """
            Put a batch of data into the pool.

            Parameters:
                indices (numpy.ndarray): 
                labels (numpy.ndarray): 
        """

        # For each datapoint a label?
        if len(data_indices) != len(labels):
            raise ArgumentError("Shape of indices and labels do not match")

        # Is slice out of range for number of max. labels?
        running_idx_length = len(data_indices)
        new_running_idx = self.running_idx + running_idx_length
        if new_running_idx > len(self.labels):
            raise ArugmentError("Can't update pool. Pool is full.")

        # Set data indices and labels, update running index
        self.indices[self.running_idx:new_running_idx] = indices
        self.labels[self.running_idx:new_running_idx] = labels
        self.running_idx = new_running_idx


    def __setitem__(self, index, label):
        """
            Setting an item for data of given index.

            Parameters:
                index (slice | int | numpy.ndarray): The data for which to set the labels
                label (numpy.ndarray): The labels to set
        """
        self.indices[index] = 1
        self.labels[index] = label


    def __getitem__(self, index):
        """
            Return data and labels at given index or slice.
            Call object[:] to return all data and labels.

            Parameters:
                index (slice | int): The index or slice from which to return data and labels.
        """

        indices = self.indices != -1
        return super().data[indices][index], self.labels[indices][index] 


    def get_inputs(self):
        indices = self.indices != -1
        return super().data[indices]

    
    def get_labels(self):
        indices = self.indices != -1
        return self.labels[indices]


    def put(self, index, label):
        """
            Put a new set of labeled information into
            the data pool.

            Parameters:
                index (int | numpy.ndarray): Indices of the data
        """

        if self.running_idx > len(self.index):
            raise ArgumentError("Can't update pool. Pool is full.")

        if isinstance(index, np.ndarray) or isinstance(label, np.ndarray):
            self.__set_with_array(index, label)

        else:
            self.indices[running_idx] = index
            self.labels[running_idx] = label
            self.running_idx += 1
