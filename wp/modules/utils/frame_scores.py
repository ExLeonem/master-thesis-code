import os, sys
import pandas as pd
import numpy as np
from .frame import Frame

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "..", "..")
TF_PATH = os.path.join(BASE_PATH, "tf_al")
sys.path.append(TF_PATH)

from tf_al.score import leff, qeff, runtime


class FrameScores:
    """
        Calculate scores from pandas frames and inserts them.
    """

    def __init__(
        self, 
        accuracy_column="eval_accuracy",
        labeled_pool_column="labeled_pool_size",
        unlabeled_pool_size="unlabeled_pool_size",
        query_time_column="query_time"
    ):
        self.__acc_col = accuracy_column
        self.__lab_pool_size = labeled_pool_column
        self.__unlabeled_pool_size = unlabeled_pool_size
        self.__query_time = query_time_column


    def add_leff(self, frame, baseline):
        """
            Calculate and insert the labeling efficiency for given frame.
            Compared to baseline.
        """
        frame_acc = frame[self.__acc_col].to_numpy()
        base_acc = baseline[self.__acc_col].to_numpy()
        mean_leff, std_leff = leff(frame_acc, base_acc)

        idx = frame.columns.get_loc(self.__acc_col)
        frame.insert(idx, "mean_leff", mean_leff)
        frame.insert(idx, "std_leff", std_leff)

    
    def add_qeff(self, frames):
        
        times = []
        for frame in frames:
            times.append(frame[self.__query_time])

        mean_qeff, std_qeff = qeff(*times)
        for idx in range(len(frames)):
            frame = frames[idx]

            col_idx = frame.columns.get_loc(self.__query_time)
            frame.insert(col_idx, "mean_qeff", mean_qeff[idx])
            frame.insert(col_idx, "std_qeff", std_qeff[idx])


    def transform_runtime(self, frame):
        query_time = frame[self.__query_time]
        idx = frame.columns.get_loc(self.__query_time)
        new_values = query_time / (frame[self.__unlabeled_pool_size] + 10)
        frame.loc[:, self.__query_time] = new_values
        # mean_time, std_time = runtime(query_time)
        # idx = frame.columns.get_loc(self.__query_time)
        # frame.insert(idx, "mean_" + self.__query_time, mean_time)
        # frame.insert(idx, "std_" + self.__query_time, std_time)


    def add_auc(self, frame):
        labeled_pool_sizes = frame[self.__lab_pool_size].to_numpy()
        scaled_sizes = labeled_pool_sizes/np.max(labeled_pool_sizes)
        accuracies = frame[self.__acc_col].to_numpy()
        auc = np.trapz(accuracies, scaled_sizes)
        frame.insert(0, "auc", auc)




    def add_percentage(self, frame):
        pass


