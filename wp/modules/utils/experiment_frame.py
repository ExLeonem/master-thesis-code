import os, sys
import numpy as np
import pandas as pd

from . import Frame, FrameScores


class ExperimentFrame:
    """
        Processes information from a single frame. 

        Parameters:
            frame (pandas.Dataframe): A dataframe including all information.
            baseline (dict): The baseline to use for relative measures defined by {columns: id}.
    """

    def __init__(self, frame, baseline={"method": "Random"}, scores=None):
        self.__baseline = baseline

        self._scores = scores
        if scores is None:
            self._scores = FrameScores()

        self.__frames = frame
        self._method_specific = {}
        self.__frame = self.__transform(frame)
        


    def __transform(self, frame):
        
        if "model" in frame and len(np.unique(frame["model"])) > 1:
            frame = Frame.split_by(frame, "model")

        # Was previously split by model
        if isinstance(frame, list):
            processed = []
            for idx in range(len(frame)):
                processed.append(self.__process_methods(frame[idx]))
            return pd.concat(processed)

        return self.__process_methods(frame)


    def __process_methods(self, model_frame):
        if not ("method" in model_frame):
            return model_frame
        
        method_frames = Frame.split_by(model_frame, "method")
        baseline_frame = self.get_baseline_frame(method_frames)
        processed_frames = []
        for frame in method_frames:
            processed_frames.append(self.__process_experiment_iteration(frame, baseline_frame))

        return pd.concat(processed_frames)


    def __process_experiment_iteration(self, frame, baseline):
        
        if not ("iteration" in frame):
            return frame
        
        baseline_by_experiments = Frame.split_by(baseline, "iteration")
        frames_by_experiments = Frame.split_by(frame, "iteration")

        for idx in range(len(frames_by_experiments)):
            exp_frame = frames_by_experiments[idx]
            exp_baseline = frames_by_experiments[idx]
            self._scores.add_leff(exp_frame, exp_baseline)
            self._scores.add_qeff(exp_frame, exp_baseline)

        return pd.concat(frames_by_experiments)


    def get_baseline_frame(self, frames):
        
        baseline = self.__baseline
        if not (len(baseline.keys()) > 0):
            return None

        column_name = list(baseline.keys())[0]
        column_value = baseline[column_name]
        for frame in frames:
            value = frame[column_name].iloc[0]
            if value == column_value:
                return frame

        return None

    
    def get_frame(self):
        return self.__frame


