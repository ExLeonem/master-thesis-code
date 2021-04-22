from IPython.display import clear_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class AL:
    """
        Entrypoint for active learning loop.

        Parameters:
            - unlabeled_pool () Pool with unlabeled data
    """

    def __init__(self, data_pool):
        self.unlabeled_pool = data_pool
        self.labeled_pool = []


    def start(self, model):
        """
            Start the active learning loop.

            Returns:
                (nn.Module) The trained model
        """
        
        while True:
            # Query data
            label = input("What's the label?")
            self.labeled_pool.append(label)
            clear_output()



    def __next_dp(self):
        """
            Query for the next datapoints to be labeled.

            Returns:
                (np.ndarray) The next datapoints to be labeled.
        """
        pass

