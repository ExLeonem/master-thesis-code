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

        self._scores.add_qeff(processed_frames)
        return pd.concat(processed_frames)


    def __process_experiment_iteration(self, frame, baseline):
        
        if not ("run" in frame and len(np.unique(frame["run"])) > 1):
            return frame

        baseline_by_experiments = Frame.split_by(baseline, "run")
        frames_by_experiments = Frame.split_by(frame, "run")

        for idx in range(len(frames_by_experiments)):
            exp_frame = frames_by_experiments[idx]
            exp_baseline = baseline_by_experiments[idx]
            self._scores.add_leff(exp_frame, exp_baseline)
            self._scores.transform_runtime(exp_frame)

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


    # ----------
    # Utilties
    # --------------------

    def __select_by(self, frame, key, value):

        if key not in frame:
            return None

        frames = Frame.split_by(frame, key)
        for frame in frames:
            if frame[key].iloc[0] == value:
                return frame

        return None


    def __split_by_existing(self, frame, key):
        if key in frame and len(np.unique(frame[key])) > 1:
            return Frame.split_by(frame, key)
        
        return [frame]


    # ----------
    # Getter/Setter
    # ----------------

    def get_mean_frame(self, method=None, model=None, ids=["model", "method"], exclude=None):

        if method is None and model is None:
            model_frames = self.__split_by_existing(self.__frame, "model")
            frames = [self.__split_by_existing(model_frame, "method") for model_frame in model_frames]
            
            meaned = []
            for frame in frames:
                meaned.append(pd.concat(Frame.mean(frame, "iteration", ids)))
            return pd.concat(meaned)


        frame = self.get_frame(method, model)
        if frame is not None:
            mean_frame = Frame.mean([frame], "iteration", ids)
            return mean_frame[0]

        return frame

    
    def get_frame(self, method=None, model=None, exclude=None, **kwargs):
        """
            Filter the frame to select by method and model name. If none passed
            all of the methods/models selected.


        """

        frame = self.__frame
        if not model is None:
            frame = self.__select_by(self.__frame, "model", model)
        
        if not method is None:
            frame = self.__select_by(frame, "method", method)

        if exclude is not None:
            frame = Frame.filter(frame, exclude)

        return frame
    

    def get_methods(self):
        return list(np.unique(self.__frame["method"]))

    
    def get_models(self):
        return list(np.unique(self.__frame["model"]))