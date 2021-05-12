class MeanEncoder:
    """
        
    """


class MinMaxNorm:

    def __init__(self, data, column):
        self.min = data[column].min()
        self.max = data[column].max()
        

    def normalize(self, datapoint, fn=None):
        """
            Normalize a given datapoint by either column or access function.

            Args:
                datapoint (np.array|pd.DataFrame) New datapoint/dataframe to be encoded
                fn (function) Function to be used to access the specific column

            Returns:
                The Normalized datapoint
        """

        data = None
        if not fn:
            # Use the accessor function to access the data
            data = fn(datapoint)


def remove(dataset, column=[]):
    """
        Remove columns from dataset.

        Args:
            dataset (pd.DataFrame|np.ndarray) The data to be cleaned
            column (str|list[str]) Columns to be removed from the dataset.

        Returns:
            Cleaned dataset.
    """
    if isinstance(column, str):
        pass

    elif isinstance(column, list):
        pass



def __remove_columns(dataset, column):
    """
        Remove 
    """

    if isinstance(column):
        # Remove column from dataset    
        return new_dataset

    # Can't remove columns because columns is not a string
