import os
import numpy as np



def load(path=None):
    """
        Load the wilson dataset and return the data.

        Returns:
            - (tuple) Of form (train_x, train_y, test_x, test_y)
    """

    dataset_path = path
    if not path:
        current_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(current_path, "..", "..", "datasets", "reg_data_wilson_izmailov")

    os.listdir(dataset_path)

    # Load data into dict
    items = os.listdir(dataset_path)
    data = {}
    for item in items:
        name, ext = item.split(".")

        data[name] = np.load(os.path.join(dataset_path, item))
    
    # Return (train_x, train_y, test_x, test_y)
    return (data["x"], data["y"], data["x_"], data["y_"])