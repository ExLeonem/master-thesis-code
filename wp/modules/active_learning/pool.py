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


    def __len__(self):
        return len(self.data)


class UnlabeledPool(DataPool):
    """
        
    """

    def __init__(self, data, init_indices=None):
        super(UnlabeledPool, self).__init__(data)
        # Are there already set initial indices?
        self.indices = np.linspace(0, len(data)-1, len(data), dtype=int)
        self._all_indices = np.linspace(0, len(data)-1, len(data), dtype=int)


    def __len__(self):
        unlabeled = self.indices != -1
        return len(self.indices[unlabeled])


    def is_empty(self):
        """
            Returns whether there's still data to be labeled in pool of unlabeled data.

            Returns:
                (bool) True if all data is labeled else False.
        """
        return self.__len__() == 0


    def update(self, indices):
        unlabeled_indices = self.get_indices()
        indices_to_update = unlabeled_indices[indices]
        self.indices[indices_to_update] = -1


    def get_indices(self):
        unlabeled = self.indices != -1
        return self.indices[unlabeled]


    def get_data(self):
        indices = self.get_indices()
        return self.data[indices]


    def get_labeled_indices(self):
        labeled = np.logical_not(self.indices != -1)
        return self._all_indices[labeled]


    def get_labeled_data(self):
        indices = self.get_labeled_indices()
        return self.data[indices]



class LabeledPool(DataPool):
    """
        Create a pool of labeled data.
        Inititally no data in the pool is labeled. 

        Parameters:
            data (numpy.ndarray): Data to be used for the pool of labeled data.
            num_init_targets (int): Number of initial label values to use per class
            seed (int): The seed to use for random selection of initial pool
            targets (numpy.ndarray): Targets to be used for selection for initial pool
    """

    def __init__(self, data, num_init_targets=10, seed=None, targets=None, pseudo=False):
        super(LabeledPool, self).__init__(data)
        self.labeled_indices = np.zeros(data.shape[0], dtype=int) - 1
        self.labels = np.zeros(data.shape[0])

        self.pseudo = pseudo
        self.targets = targets
        self.running_idx = 0
        
        # Initialize labels set
        self.__init_pool_of_indices(num_init_targets, seed, targets)


    def __init_pool_of_indices(self, num_init_targets, seed, targets):
        """
            Initialize the pool with randomly selected values.
        """

        # Use initial target values?
        if num_init_targets <= 0:
            return

        # Reproducability?
        if not (seed is None):
            np.random.seed(seed)

        # Select 'num_init_targets' per unique label 
        unique_targets = np.unique(targets)
        for idx in range(len(unique_targets)):

            # Select indices of labels for unique label[idx]
            with_unique_value = targets == unique_targets[idx]
            indices_of_label = np.argwhere(with_unique_value)

            # Set randomly selected labels
            selected_indices = np.random.choice(indices_of_label.flatten(), num_init_targets, replace=True)
            self.labeled_indices[selected_indices] = 1
            self.labels[selected_indices] = unique_targets[idx]


    def __setitem__(self, index, label=None):
        """
            Setting an item for data of given index.

            Parameters:
                index (slice | int | numpy.ndarray): The data for which to set the labels
                label (numpy.ndarray): The labels to set
        """
        self.labeled_indices[index] = 1

        # If pseudo selection, select von available targets
        if not (self.targets is None) and self.pseudo:
            label = self.targets[index]

        self.labels[index] = label


    def __getitem__(self, index):
        """
            Return data and labels at given index or slice.
            Call object[:] to return all data and labels.

            Parameters:
                index (slice | int): The index or slice from which to return data and labels.
        """

        indices = self.labeled_indices != -1
        return (self.data[indices])[index], (self.labeled_indices[indices])[index] 


    def __len__(self):
        indices = self.labeled_indices != -1
        return len(self.labels[indices])


    def get_inputs(self):
        indices = self.labeled_indices != -1
        return self.data[indices]

    
    def get_labels(self):
        indices = self.labeled_indices != -1
        return self.labels[indices]


    def get_indices(self):
        """
            Get indices of labels that are already set
        """
        return np.argwhere(self.labeled_indices != -1)


    def put(self, index, label):
        """
            Put a new set of labeled information into
            the data pool.

            Parameters:
                index (int | numpy.ndarray): Indices of the data
        """

        if self.running_idx > len(self.labeled_indices):
            raise ArgumentError("Can't update pool. Pool is full.")

        if isinstance(index, np.ndarray) or isinstance(label, np.ndarray):
            self.__set_with_array(index, label)

        else:
            self.labeled_indices[running_idx] = index
            self.labels[running_idx] = label
            self.running_idx += 1


    def __set_with_array(self, labels):

        num_new_labels = len(labels)

        if self.running_idx + num_new_labels > len(self.labeled_indices):
            raise ValueError("Can't update pool. Pool is full.")

        

        

        

    
    def __put_batch_batch(self, indices, labels):
        """
            Put a batch of data into the pool.

            Parameters:
                indices (numpy.ndarray): 
                labels (numpy.ndarray): 
        """

        # For each datapoint a label?
        if len(labeled_indices) != len(labels):
            raise ArgumentError("Shape of indices and labels do not match")

        # Is slice out of range for number of max. labels?
        running_idx_length = len(labeled_indices)
        new_running_idx = self.running_idx + running_idx_length
        if new_running_idx > len(self.labels):
            raise ArugmentError("Can't update pool. Pool is full.")

        # Set data indices and labels, update running index
        self.labeled_indices[self.running_idx:new_running_idx] = indices
        self.labels[self.running_idx:new_running_idx] = labels
        self.running_idx = new_running_idx