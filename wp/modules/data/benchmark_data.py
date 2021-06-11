from . import mnist
from .transformer import Pipeline, select_classes, image_channels, inputs_to_type
from enum import Enum

class DataSetType(Enum):
    """

    """
    MNIST = 1
    FASHION_MNIST = 2



class BenchmarkData:
    """
        Load and transform data for different benchmarks.

        Parameters:
            dataset (DataSetType): An enum representing the dataset to use.
            path (str): The path to the dataset

        **Kwargs:

    """

    def __init__(self, dataset, path=None, **kwargs):
        data = self.__load_data(dataset, path, **kwargs)
        self.inputs, self.targets = self.__data_transform(dataset, data, **kwargs)
        

    def __load_data(self, dataset, path, **kwargs):
        """
            Load data from given path.

            Parameters:
                dataset (DataSet): Indicates which dataset loader to use
                path (str): The path where the data is located.

            Returns:
                ((numpy.ndarray, numpy.ndarray) | numpy.ndarray) The data either splitted into inputs and targets or as a single numpy.ndarray.

        """

        if dataset is DataSetType.MNIST:
            return mnist.load(path)

        else:
            raise ValueError("Missing loader for {}.".format(dataset))
    
    
    def __data_transform(self, dataset, data, skip_transforms=False, **kwargs):
        """
            Transform a given dataset by using a given transformer.

            Parameters:
                dataset (DataSet): The dataset represented by given data
                data (np.ndarray, np.ndarray): Inputs and targets to transform.
                skip_transforms (bool): Whether or not to skip transformations.

            Returns:
                ((numpy.ndarray, numpy.ndarray)) The transformed inputs and targets
        """

        # Just return the data
        if skip_transforms:
            return data

        # Perform dataset specific transformations
        if dataset is DataSetType.MNIST:
            return self.__get_img_classification_pipeline()(data, **kwargs)
    
        else:
            raise ValueError("Missing transformation for dataset {}.".format(dataset))
    

    def __get_img_classification_pipeline(self):
        """
            Transformation pipeline for datasets with labeled image data.
        """
        
        return Pipeline(
                select_classes,
                image_channels, 
                inputs_to_type
            )


    # -------------
    # Standard Getter/-Setter
    # -----------------------------
    
    def get_targets(self):
        return self.targets

    
    def get_inputs(self):
        return self.inputs

    

    
