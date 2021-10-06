import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import Frame, ExperimentFrame, Stats, Table


class StatWriter:
    """
        Saves plots into given directory.
    """

    def __init__(self, path, keys={}):
        self.__path = path
        self.__keys = keys



    def write_compare(self, frame, **kwargs):

        methods = np.unique(frame["method"])        
        for method in methods:
            self.compare_models_accuracy(frame, method)
            plt.close()          


    def write(self, frame, model=None, prefix=None, **kwargs):

        models = [model]
        if model is None:
            models = np.unique(frame["model"])


        for model in models:
            selector = frame["model"] == model
            model_frame = frame[selector]
            self.accuracy(model_frame, self.__combine(prefix, model+"_acc"), **kwargs)
            plt.figure()

            self.loss(model_frame, self.__combine(prefix, model+"_loss"), **kwargs)
            plt.figure()

            # self.time(model_frame, self.__combine(prefix, model+"_time"), **kwargs)
            # plt.figure()

            self.leff(model_frame, self.__combine(prefix, model+"_leff"), **kwargs)

        plt.close()


    def accuracy(self, frame, name, key=None, save=True):
            name = self.__prepare_name(name)
            if key is None:
                key = self.__keys.get("acc", "eval_sparse_categorical_accuracy")
        
            fig = sns.lineplot(data=frame, x="labeled_pool_size", y=key, hue="method")
            fig.set(xlabel="Labeled pool size", ylabel="Test Accuracy")
            fig.legend(title="Method")

            if save:
                plt.savefig(os.path.join(self.__path, "methods", name+".png"))
            

    def loss(self, frame, name, key=None, save=True):
        name = self.__prepare_name(name)
        if key is None:
            key = self.__keys.get("loss", "eval_sparse_categorical_crossentropy")

        fig = sns.lineplot(data=frame, x="labeled_pool_size", y=key, hue="method")
        fig.set(xlabel="Labeled pool size", ylabel="Test Loss")
        fig.legend(title="Method")

        if save:
            plt.savefig(os.path.join(self.__path, "methods", name+".png"))


    def time(self, frame, name, key=None, save=True):
        name = self.__prepare_name(name)
        if key is None:
            key = self.__keys.get("time", "query_time")
        
        fig = sns.lineplot(data=frame, x="labeled_pool_size", y=key, hue="method", ci="sd")
        fig.set(xlabel="Labeled pool size", ylabel="Query time in seconds per datapoint")
        fig.legend(title="Method")
        
        if save:
            plt.savefig(os.path.join(self.__path, name+".png"))


    def leff(self, frame, name, key=None, save=True):
        name = self.__prepare_name(name)
        if key is None:
            key = self.__keys.get("time", "mean_leff")

        fig = sns.lineplot(data=frame, x="labeled_pool_size", y=key, hue="method")
        fig.set(xlabel="Labeled pool size", ylabel="Label efficiency")
        fig.legend(title="Method")

        if save:
            plt.savefig(os.path.join(self.__path, name+".png"))


    def compare_models_accuracy(self, frame, method, key="eval_sparse_categorical_accuracy", save=True):
        selector = frame["method"] == method
        fig = sns.lineplot(data=frame[selector], x="labeled_pool_size", y=key, hue="model")
        fig.set(xlabel="Labeled pool size", ylabel="Test Accuracy")
        fig.legend(title="Model")  

        if save:
            method = method.replace(".", "")
            method = self.__prepare_name(method)
            plt.savefig(os.path.join(self.__path, "methods", method+"_comparison.png"))



    # -------------
    # Utilities
    # --------------------

    def __combine(self, prefix, postfix):

        if prefix is not None:
            return prefix + postfix
        
        return postfix

    
    def __prepare_name(self, name):
        name = name.replace(" ", "_")
        name = name.lower()
        return name