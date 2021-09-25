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
        keys = ["model", "method", "std_query_time", "mean_qeff", "std_qeff"]
        keys = Stats.clear_keys(keys, model, method)
        frames = Stats.split_by(frame, model, method)
        return Stats.extract_first_from(frames, keys)


    @staticmethod
    def per_points(frame, each_labeled_size, model=True, method=True, key="eval_accuracy"):
        """
            Calculates table for accuracy per n datapoints
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
    def grouped_mean(frame, keys):
        return 



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