import os, sys
import json
import re
import pandas as pd

from .logger import setup_logger

dir_path = os.path.abspath(os.path.realpath(__file__))
MODULES_PATH = os.path.join(dir_path, "..")
sys.path.append(MODULES_PATH)

# from active_learning import Metrics


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
    

    def load_from_dir(self, metrics_handler, where="", dtype=None):
        BASE_PATH = metrics_handler.BASE_PATH
        files = os.listdir(BASE_PATH)
        frames = []
        for file in files:
            
            if ".csv" in file and where in file:
                run = file.split("_")[0]        
                model_name = MetricsTransformer.get_model_name(file)
                method_name = MetricsTransformer.get_method_name(file)
                
                file_path = os.path.join(BASE_PATH, file)
                
                new_frame = self.load(file_path, metrics_handler, dtype=dtype)
                new_frame.insert(0, "method", method_name)
                new_frame.insert(0, "model", model_name)
                new_frame.insert(0, "run", run)
                frames.append(new_frame)

        return pd.concat(frames)


    def mean(self, dataframe, columns, dtype={}):
        """

        """
        iterations = list(pd.unique(dataframe["iteration"]))

        transformed = []
        for iteration in iterations:

            sub = dataframe[columns]
            filter_to_apply = sub["iteration"] == iteration

            meaned_values = dict(sub[filter_to_apply].mean())
            transformed.append(meaned_values)

        df = pd.DataFrame(transformed)


        dtype.update({"iteration": int})
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

    @staticmethod
    def seconds_to_minutes(dataframe, columns, ndigits):
        """
            Transform column(s) of time measures from seconds to minutes.

            Parameters:
                dataframe (pandas.DataFrame): The dataframe of which columns to be transformed.
                columns (str|list(str)): A list of column names or a single column name.

            Returns:
                (pandas.DataFrame) the dataframe with transformed columns.
        """
        if isinstance(columns, list):
            
            for column in columns:
                dataframe[column] = dataframe[column]/60


        elif isinstance(columns, str):
            dataframe[columns] = dataframe[columns]/60

        return dataframe


    @staticmethod
    def list_to_series(dataframe, column):
        """
            Transforms a column of a dataframe into pandas.Series.

            Parameters:
                dataframe (pandas.DataFrame): The dataframe for which to transform a columns.
                column (str): The column name of the column to transform.

            Returns:
                (pandas.DataFrame) the list values as a dataframe.
        """

        list_data = {}
        for idx, row in dataframe.iterrows():    
            cell_list = json.loads(row[column])
            list_data[idx] = cell_list  

        return pd.DataFrame(list_data)

    
    @staticmethod
    def merge_and_exclude(dataframes, to_exclude=[]):
        """

            Parameters:
                dataframes (list(pd.DataFrame)): 
                to_exclude (list(str)): List of column name to exclude from the dataframes.
        """

        filtered = []
        for frame in dataframes:
            all_columns = list(frame.columns)
            columns_filtered = list(filter(lambda name: name not in to_exclude, all_columns))
            filtered.append(frame[columns_filtered])

        return pd.concat(filtered, sort=True)


    @staticmethod
    def get_method_name(filename):

        if "random" in filename:
            return "Random"
        
        elif "max_entropy" in filename:
            return "Max. Entropy"

        elif "std_mean" in filename:
            return "Std. Mean"

        elif "max_var_ratio" in filename:
            return "Max. Var. Ratio"

        elif "bald" in filename:
            return "BALD"

        raise ValueError("Unknown method name")


    @staticmethod
    def get_model_name(filename):

        if "mc_dropout" in filename:
            return "MC Dropout"

        if "moment_propagation" in filename:
            return "Moment Propagation"

        
        raise ValueErro("Unknown model name.")
