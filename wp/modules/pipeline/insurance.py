
"""
    Routines to clean datasets
"""



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
        