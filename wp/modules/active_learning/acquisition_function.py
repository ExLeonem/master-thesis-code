
import os, sys
import numpy as np
# import ..bayesian.utils as butils

class AcquisitionFunction:
    """

        Parameters:
            - fn_name (str) The acquisition function to apply
            - batch_select (bool) Apply batch query?
            - batch_size (int) Number of points to query for.
    """

    def __init__(self, fn_name, batch_select=False, batch_size=10):
        self.fn = self._set_fn(fn_name)
        self.batch_select = batch_select
        self.batch_size = batch_size


    def __call__(self, model, data):
         return self.fn(model, data)


    def _set_fn(self, name):
        """
            Set the function to use for acquisition.
            
            Parameters:
                - name (str) The name of the acquisition to use.

            Returns:
                - (function) The function to use for acquisition.
        """

        if name == "max_entropy":
            return self._max_entropy
        
        elif name == "max_info_gain":
            return self._max_information_gain

        elif name == "max_var_ratio":
            return self._max_variation_ratios
        
        elif name == "std_mean":
            return self._std_mean

        else:
            return self._random


    def _max_entropy(self, model, data):
        return 0

    def _max_information_gain(self, model, data):
        return 0

    def _max_variation_ratios(self, model, data):
        return 0

    def _std_mean(self, model, data):
        return 0
    
    def _random(self, model, data):
        """
            Needs:
                - pool of unlabeled data
                -

            Returns:
                - (int) Index of selected 
        """


        return 0