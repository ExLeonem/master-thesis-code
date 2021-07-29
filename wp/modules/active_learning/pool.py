import math
from copy import deepcopy
import numpy as np


class Pool:
    """
        Pool that holds information about labeled and unlabeld inputs.
        The attribute 'indices' holds information about the labeled inputs.
        
        Each value of self.indices can take the following states:
        (value==-1) Corresponding input is labeld
        (value!=-1) Corresponding input is not labeled 

        Parameters:
            inputs (numpy.ndarray): Inputs to the network.
    """

    def __init__(self, inputs, targets=None):
        self.__inputs = inputs
        self.__true_targets = targets
        self.__indices = np.linspace(0, len(inputs)-1, len(inputs), dtype=int)
        self.__targets = np.zeros(len(inputs))


    def init(self, size):
        """
            Initialize the pool with specific number of labels.
            Only applicable when pool in pseudo mode.

            Parameters:
                size (int): number of targets to initialize the pool with.
        """
        
        if not self.is_pseudo():
            raise ValueError("Error in Pool.init(). Can't initialize pool using init(size) when not in pseudo mode. Initialize pool with targets, to put Pool in pseudo mode.")

        if size < 1:
            raise ValueError("Error in Pool.init(). Can't initialize pool with {} targets. Use a positive integer > 1.".format(size))

        if len(self.__indices) < size:
            raise ValueError("Error in Pool.init(). Can't initialize pool, not enough targets. {} targets required, {} are available.".format(size, len(self.__indices)))


        # WARNING: Will only work for categorical targets
        unique_targets = np.unique(self.__true_targets)

        num_to_select = 1
        num_unique_targets = len(unique_targets)
        if num_unique_targets < size:
            num_to_select = math.floor(size/num_unique_targets)    
        
        # Annotate samples in round robin like schme
        while size > 0:
            
            # Select 
            for target in unique_targets:

                unlabeled_indices = self.get_unlabeled_indices()
                true_targets = self.__true_targets[unlabeled_indices]
                selector = (true_targets == target)

                indices = unlabeled_indices[selector]
                targets = true_targets[selector]

                # Move to next target, when none available of this type
                if len(indices) == 0:
                    continue
                
                adapted_num_to_select = self.__adapt_num_to_select(targets, num_to_select)
                selected_indices = np.random.choice(indices, adapted_num_to_select, replace=False)
                
                # Update pool
                selected_targets = self.__true_targets[selected_indices]
                self.annotate(selected_indices, selected_targets)
                size -= num_to_select

                if size < 1:
                    break
    
    
    def __adapt_num_to_select(self, available, num_to_select):
        """
            Adapts the number of elements to select next.

            Parameters:
                available (numpy.ndarray): The available elements.
                num_to_select (int): The number of elements to select.

            Returns:
                (int) the adapted number of elements selectable.
        """

        num_available = len(available)
        if num_available < num_to_select:
            return num_available

        return num_to_select


    def get_inputs_by(self, indices):
        """
            Get inputs by indices.

            Parameters:
                indices (numpy.ndarray): The indices at which to access the data.

            Returns:
                (numpy.ndarray) the data at given indices.
        """
        return self.__inputs[indices]


    def get_targets_by(self, indices):
        """

        """
        return self.__targets[indices]


    def __setitem__(self, indices, targets):
        """
            Shortcut to annotate function.

            Parameters:
                indices (numpy.ndarray): For which indices to set values.
                targets (numpy.ndarray): The targets to set.
        """
        self.annotate(indices, targets)

    
    def annotate(self, indices, targets=None):
        """
            Annotate inputs of given indices with given targets.

            Parameters:
                indices (numpy.ndarray): The indices to annotate.
                targets (numpy.ndarray): The labels to set for the given annotations.
        """

        if targets is None:
            if self.__targets is None:
                raise ValueError("Error in Pool.annotate(). Can't annotate inputs, targets is None.")

            targets = self.__true_targets[indices]
            

        # Create annotation
        self.__indices[indices] = -1
        self.__targets[indices] = targets


    # ---------
    # Utilities
    # -------------------

    def has_unlabeled(self):
        """
            Has pool any unlabeled inputs?

            Returns:
                (bool) true or false depending whether unlabeled data exists.
        """
        selector = np.logical_not(self.__indices == -1)
        return np.any(selector)


    def has_labeled(self):
        """
            Has pool labeled inputs?

            Returns:
                (bool) true or false depending whether or not there are labeled inputs.
        """
        selector = self.__indices == -1
        return np.any(selector)


    def is_pseudo(self):
        """
            Is the pool in pseudo mode?
            Meaning, true target labels are already known?

            Returns:
                (bool) indicating whether or not true labels are existent.
        """
        return self.__true_targets is not None


    def __deepcopy__(self, memo):
        return Pool(self.__inputs, self.__true_targets)

    # ---------
    # Setter/Getter
    # -------------------

    def get_indices(self):
        """
            Returns the current labeling state.

            Returns:
                (numpy.ndarray) the indices state. (-1) indicating a labeled input.
        """
        
        return self.__indices

    def get_labeled_indices(self):
        """
            

            Returns:
                (numpy.ndarray) of datapoints that already has been labeled.
        """
        selector = self.__indices == -1
        indices = np.linspace(0, len(self.__inputs)-1, len(self.__inputs), dtype=int)
        return indices[selector]


    def get_unlabeled_indices(self):
        """
            Get all unlabeled indices for this pool.

            Returns:
                (numpy.ndarray) an array of indices.
        """
        selector = self.__indices != -1
        return self.__indices[selector]


    def get_length_labeled(self):
        """
            Get the number of labeled inputs.

            Returns:
                (int) The number of labeled inputs.
        """
        return np.sum(self.__indices == -1)

    
    def get_length_unlabeled(self):
        """
            Get the number of unlabeld inputs.

            Returns:
                (int) the number of unlabeled inputs
        """
        return np.sum(np.logical_not(self.__indices == -1))   


    def get_labeled_data(self):
        """
            Get inputs, target pairs of already labeled inputs.

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) inputs and corresponding targets.
        """
        selector = self.__indices == -1
        inputs = self.__inputs[selector]
        targets = self.__targets[selector]
        return inputs, targets

    
    def get_unlabeled_data(self):
        """
            Get inputs whire are not already labeled with their indices.

            Returns:
                (tuple(numpy.ndarray, numpy.ndarray)) The inputs and their indices in the pool
        """
        selector = self.__indices != -1
        inputs = self.__inputs[selector]
        indices = self.__indices[selector]
        return inputs, indices
        


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
        Create a pool, holding  information about unlabeled input values.

        Parameters:
            data (np.ndarray): The inputs of the dataset

    """

    def __init__(self, data):
        super(UnlabeledPool, self).__init__(data)
        # Are there already set initial indices?
        self.indices = np.linspace(0, len(data)-1, len(data), dtype=int)
        self._all_indices = np.linspace(0, len(data)-1, len(data), dtype=int)


    def __len__(self):
        """
            How many inputs are labeled?

            Returns:
                (int) The number of unlabeled inputs.
        """
        unlabeled = self.indices != -1
        return len(self.indices[unlabeled])


    def is_empty(self):
        """
            Returns whether there's still data to be labeled in pool of unlabeled data.

            Returns:
                (bool) True if all data is labeled else False.
        """
        return self.__len__() == 0


    def update(self, indices_to_update):
        """
            Mark specific indices as 'labeled'.

            Parameters:
                indices (np.ndarray | int): The indices of inputs to mark as labeled
        """
        self.indices[indices_to_update] = -1


    def get_indices(self):
        """
            Get the indices of unlabeld input values.
        """
        unlabeled = self.indices != -1
        return self.indices[unlabeled]


    def get_data(self):
        """
            Get input values of unlabeled data.

            Returns:
                (np.ndarray) inputs which are not labeled.
        """
        indices = self.get_indices()
        return self.data[indices]


    def get_labeled_indices(self):
        """
            Get indices of already labeled inputs.

            Returns:
                (np.ndarray) indices of already labeled inputs.
        """
        labeled = np.logical_not(self.indices != -1)
        return self._all_indices[labeled]


    def get_labeled_data(self):
        """
            Get values of already labeled datapoints.

            Returns:
                (np.ndarray) datapoints which are already labeled.
        """
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

    def __init__(self, data, target_shape=None):
        super(LabeledPool, self).__init__(data)
        self.labeled_indices = np.zeros(data.shape[0], dtype=int) - 1

        if target_shape is None:
            self.labels = np.zeros(data.shape[0])
        else:
            self.labels = np.zeros(target_shape)
        
        # Initialize labels set
        # self.__init_pool_of_indices(num_init_targets, seed, targets)


    def __setitem__(self, index, label):
        """
            Setting an item for data of given index.

            Parameters:
                index (slice | int | numpy.ndarray): The data for which to set the labels
                label (numpy.ndarray): The labels to set
        """


        self.labeled_indices[index] = 1
        self.labels[index] = label


    def __getitem__(self, index):
        """
            Return data and labels at given index or slice.
            Call object[:] to return all data and labels.

            Parameters:
                index (slice | int): The index or slice from which to return data and labels.
        """

        indices = self.labeled_indices != -1
        return (self.data[indices])[index], (self.labels[indices])[index] 


    def __len__(self):
        """
            How many labeled samples are existent?

            Returns:
                (int) Number of labeled samples in pool of labeled data.
        """
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
