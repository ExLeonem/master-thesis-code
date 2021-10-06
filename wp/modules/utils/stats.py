import numpy as np
import pandas as pd
from . import Frame

class Stats:
    """
        Collect stats for tables.
    """

    @staticmethod    
    def query_time(frame, model=True, method=True):
        keys = ["model", "method", "std_query_time", "mean_query_time", "query_time"]
        keys = Stats.clear_keys(keys, model, method)
        frames = Stats.split_by(frame, model, method)
        return Stats.extract_first_from(frames, keys)

    @staticmethod
    def qeff(frame, model=True, method=True):
        keys = ["model", "method", "mean_qeff", "std_qeff"]
        keys = Stats.clear_keys(keys, model, method)
        frames = Stats.split_by(frame, model, method)
        return Stats.extract_first_from(frames, keys)


    @staticmethod
    def query_time_per_datapoints(frame, sizes):
        
        methods = np.unique(frame["method"])
        query_times = {"method": [], "size": [], "times": []}
        for method in methods:
            selector = frame["method"] == method
            query_time = frame[selector]["query_time"]

            times_per_dp = []
            for num_datapoints in sizes:

                query_times["size"].append([num_datapoints]*len(query_time))
                query_times["times"].append(query_time*num_datapoints)

            query_times["method"].append(np.array([method]*len(query_time)*len(sizes)))


        for key, value in query_times.items():
            query_times[key] = np.hstack(value)

        return pd.DataFrame(query_times)


    @staticmethod
    def per_points(frame, each_labeled_size, key="eval_accuracy", model=True, method=True, mean_std=True, decimals=None):
        """
            Calculates table for accuracy per n datapoints

            Parameters:
                frame (pandas.DataFrame): A composite pandas dataframe including experiments for different acquisition methods and models.
                each_labeled_size (int): Collect values for every nth datapoints collected as labeled points.
                key (str): The key for which to aggregate the information. (default='eval_accuracy')
                model (bool): Display model information? (default=True)
                method (bool): Display method information? (default=True)
                mean_std (bool): Aggregate information as mean \u00B1 std. information. (default=True)

            Returns:
                (pandas.DataFrame) with aggregated information.
        """
        base_keys = ["model", "method", "labeled_pool_size"]
        base_keys = Stats.clear_keys(base_keys, model, method)
        keys = base_keys + [key]
        frames = Stats.split_by(frame, model, method)

        for idx in range(len(frames)):
            selector = (frames[idx]["labeled_pool_size"] % each_labeled_size) == 0
            frames[idx] = frames[idx][selector]

        # Sub select only keys
        new_frame = pd.concat(frames)
        new_frame = new_frame[keys]

        # Group and colculate mean and std
        grouped = new_frame.groupby(base_keys)
        mean = grouped.mean()
        std = grouped.std()

        # Rename columns and indices
        mean.columns = ["Mean"]
        std.columns = ["Std"]
        merged = pd.concat([mean, std], axis=1)
        merged.index.names = list(map(lambda x: x.capitalize(), base_keys[:-1])) + ["Labeled Datapoints"]
        return merged


    @staticmethod
    def mean_std(frame, decimals=None, mean_col="Mean", std_col="Std"):
        return  Frame.merge_mean_std(frame, decimals, mean_col, std_col)


    @staticmethod
    def extract_first_from(frames, keys):
       
        for idx in range(len(frames)):
            frame = frames[idx]
            frames[idx] = frame[keys].iloc[:1]

        return pd.concat(frames)


    @staticmethod
    def write(frame, file_path):
        return frame.to_csv(file_path)


    @staticmethod
    def write_latex(frame, file_path, **kwargs):
        with open(file_path, "w") as tex_file:
            tex_file.write(frame.to_latex(**kwargs))


    @staticmethod
    def read(file_path):
        return pd.read_csv(file_path)


    # ------------
    # Utilities
    # ------------------

   
    @staticmethod
    def clear_keys(keys, model, method):
        if not model and not method:
            return keys


        filtered_keys = []
        for key in keys:
            if not model and key == "model" or not method and key == "method":
                continue
            
            filtered_keys.append(key)

        return filtered_keys


    @staticmethod
    def split_by(frame, model=True, method=True):

        frames = [frame]
        if model and "model" in frame and len(np.unique(frame["model"]))>1:
            frames = Frame.split_by(frame, "model")

        # Split method
        if method and "method":
            splitted = []
            for frame in frames:
                splitted += Frame.split_by(frame, "method")
            frames = splitted

        
        return frames