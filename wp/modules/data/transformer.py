import functools
import numpy as np


class Pipeline:
    """
        Execute a transformation pipeline to prepare a dataset
        for benchmarking.

        Transformers need to be of form "def fn(data, **kwargs)".

        Parameters:
            *args (fn): Transformers to execute in sequence. 
    """

    def __init__(self, *args):
        self.transformers = args

    
    def __call__(self, data, **kwargs):
        output = data
        for transformer in self.transformers:
            output = transformer(output, **kwargs)

        return output


def select_classes(data, classes=None, **kwargs):
    """
        Select datapoints correspondings to first n labels or a specific list of classes.
        Ignoring the classes parameter skips the transformation.



        Parameter:
            data (tuple(numpy.ndarray, numpy.ndarray)): The input value and targets
            classes (int|list): An integer to extract first n classes or a list of class labels. (default: None)

        Returns:
            ((numpy.ndarray, numpy.ndarray)) The transformed inputs and targets
    """

    # Skip class limitation
    if classes is None:
        return data

    # Select only specific classes
    inputs, targets = data

    # Use n-classes first classes
    if isinstance(classes, int):
        target_unique_values = np.unique(targets)

        if classes < 1 :
            raise ValueError("Can't select {} labels. Positive number of classes expected.".format(classes))

        if classes <= len(target_unique_values):
            selected_labels = target_unique_values[:classes]
            selector = functools.reduce(lambda a, b: a | b, [targets == label for label in selected_labels])
            
            new_targets = targets[selector]
            new_inputs = inputs[selector]
            return new_inputs, new_targets

        else:
            # Less available unique classes than to select
            raise ValueError("Can't select {} labels from {} available unique labels. ".format(classes, len(target_unique_values)))

    # Select specific labels
    if isinstance(classes, list):

        if len(classes) == 0:
            raise ValueError("Can't labels out of an empty list. Set the class parameter with a non empty list.")
        
        selector = functools.reduce(lambda a, b: a | b, [targets == label for label in classes])
        new_targets = targets[selector]
        new_inputs = inputs[selector]
        return new_inputs, new_targets

    raise ValueError("Error in transformer.select_class. Could not use classes parameter. Pass either nothing, an integer or a list of labels for the classes kwarg.")


def image_channels(data, **kwargs):
    """
        Check if image data has enough channels.
        If data has format of grayscale with missing color channel dimension, additional dimension will be added.

        Parameter:
            data ((numpy.ndarray, numpy.ndarray)): Inputs and targets of the dataset.

        Returns:
            ((numpy.ndarray, numpy.ndarray)) The transformed data
    """

    inputs, targets = data

    # Grayscale image with missing dimension? shape: (batch, height, width) 
    if len(inputs.shape) == 3:
        inputs = np.expand_dims(inputs, axis=-1)


    if len(inputs.shape) != 4:
        raise ValueError("Error in image_channels/2. Expected image data to have 3 or 4 dimensions. Got {}.".format(len(inputs.shape)))

    return inputs, targets


def inputs_to_type(data, dtype=None, **kwargs):
    """

    """

    if dtype is None:
        return data

    # Transform inputs to given type
    inputs, targets = data
    return inputs.astype(dtype), targets

