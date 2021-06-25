import os, sys
import re
import pandas as pd

from .logger import setup_logger

dir_path = os.path.abspath(os.path.realpath(__file__))
MODULES_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULES_PATH)

from active_learning import Metrics



class MetricsTransformer:
    """
        Load dataframes directly as a dataframe
    """


    def __init__(self, verbose=False):
        self.logger = self.__setup_logger(verbose)


    def __setup_logger(self, verbose):
        """
            Setup a logger.
            Sets parameters and redirects to utils.logger.setup_logger.

            Returns:
                (logging.Logger) a logger to use.
        """
        dir_path = os.path.abspath(os.path.realpath(__file__))
        log_path = os.path.join(dir_path, "..", "..", "logs")
        return setup_logger(verbose, path=log_path, file="metrics_df.log")

    
    def load(self, filename, metrics_handler, dtype=None):
        """
            Loads a single metrics file into 

            Parameters:
                metrics_handler (active_learning.Metrics): Filehandler to read/write metrics.
                filename (str): The file to load.
                column_types (dict): A map to transform column names into given types.
        """
        data = metrics_handler.read(filename)
        df = pd.DataFrame(data)

        # Cast column types
        if not (dtype is None) and isinstance(dtype, dict):
            df = df.astype(dtype)

        return df


    def aggregate_by_model(self, model_name, metrics_handler, pattern_map=None):
        """
            TODO: write implementation
            Loads and aggregates metrisc by model name.


            Parameters:
                model_name (str): The name of the model to aggregate by. Files will be selected by this name.
                metrics_handler (): The filename 

            Returns:
                (DataFrame) {"_filename_", }
        """

        base_path = metrics_handler.get_base_path()
        files = os.path.listdir(base_path)

        
        dataframes = []
        for file in files:
            
            # Is element of folder a file?
            if os.path.isdir(os.path.join(base_path, file)):
                continue

            # Is metric of given model?
            if not (model_name in file):
                continue

        
        return None

    
    def add_column(self, dataframe, column, values):
        pass