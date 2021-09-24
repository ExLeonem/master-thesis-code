import os, sys
import pandas as pd
import numpy as np
from .frame import Frame

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "..", "..")
TF_PATH = os.path.join(BASE_PATH, "tf_al")
sys.path.append(TF_PATH)

from tf_al.score import leff, qeff


class FrameScores:
    """
        Calculate scores from pandas frames and inserts them.
    """

    def __init__(
        self, 
        accuracy_column="eval_accuracy",
        labeled_pool_column="labeled_pool_size",
        query_time_column="query_time"
    ):
        self.__acc_col = accuracy_column
        self.__lab_pool_size = labeled_pool_column
        self.__query_time = query_time_column


    def add_leff(self, frame, baseline):
        """
            Calculate and insert the labeling efficiency for given frame.
            Compared to baseline.
        """
        frame_acc = frame[self.__acc_col].to_numpy()
        base_acc = baseline[self.__acc_col].to_numpy()

        mean_leff, std_leff = leff(frame_acc, base_acc)

        # print(len(mean_leff))
        # print(len(frame_acc))
        # print(len(base_acc))

        frame.insert(0, "mean_leff", mean_leff)
        # frame.insert(-1, "std_leff", std_leff)

    
    def add_qeff(self, frame, baseline):
        
        frame_query_time = frame[self.__query_time].to_numpy()
        base_query_time = baseline[self.__query_time].to_numpy()

        mean_qeff, std_qeff = qeff(frame_query_time, base_query_time)

        frame.insert(0, "mean_qeff", mean_qeff)


    def add_percentage(self, frame):
        pass


