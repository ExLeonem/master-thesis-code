from . import Frame
import pandas as pd
import numpy as np

class Table:

    @staticmethod
    def auc(frame, decimals=None):

        models = list(np.unique(frame["model"]))
        result = {"Model": [], "Method": [], "Mean": [], "Std": []}

        methods = np.unique(frame["method"])
        for method in methods:
            selector = frame["method"] == method
            method_frame = frame[selector]

            for model in models:
                selector = method_frame["model"] == model
                model_frame = method_frame[selector]

                result["Model"].append(model)
                result["Method"].append(method)

                mean = np.mean(model_frame["auc"])
                std = np.std(model_frame["auc"])
                if decimals is not None:
                    mean = np.round(mean, decimals)
                    std = np.round(std, decimals)

                result["Mean"].append(mean)
                result["Std"].append(std)                
            

        # Use multi-index
        if len(models) > 1:
            index_tuples = list(zip(result["Method"], result["Model"]))
            index = pd.MultiIndex.from_tuples(index_tuples, names=["Method", "Model"])

            del result["Model"]
            del result["Method"]
            return pd.DataFrame(result, index=index)

        return pd.DataFrame(result)


    @staticmethod
    def query_time(frame, decimals=None):

        methods = np.unique(frame["method"])
        models = np.unique(frame["model"])

        values = {"Model": [], "Method": [], "Mean": [], "Std": []}
        for method in methods:
            method_selector = frame["method"] == method
            method_frame = frame[method_selector]

            for model in models:
                model_selector = method_frame["model"] == model
                model_frame = method_frame[model_selector]

                mean = np.mean(model_frame["query_time"])
                std = np.std(model_frame["query_time"])
                if decimals is not None:
                    mean = np.round(mean, decimals)
                    std = np.round(std, decimals)
                
                values["Mean"].append(mean)
                values["Std"].append(std)
                values["Model"].append(model)
                values["Method"].append(method)


        if len(models) > 1:
            index_tuples = list(zip(values["Method"], values["Model"]))
            index = pd.MultiIndex.from_tuples(index_tuples, names=["Method", "Model"])

            del values["Model"]
            del values["Method"]
            return pd.DataFrame(values, index=index)

        return pd.DataFrame(values)
            


    @staticmethod
    def query_time_per_dp(frame, sizes, decimals=None):
        
        methods = np.unique(frame["method"])
        result = {}
        for method in methods:
            selector = frame["method"] == method
            query_time = frame[selector]["query_time"]
            if result.get(method) is None:
                result[method] = []
            

            mean_qt = Frame.round_values(np.mean(query_time)*sizes, decimals)
            std_qt = Frame.round_values(np.std(query_time)*sizes, decimals)
            zipped = zip(mean_qt, std_qt)
            times = np.array(list(map(lambda x: str(x[0]) + " \u00B1 " + str(x[1]), zipped)))
            result[method].append(times)
                


        for key, value in result.items():
            result[key] = np.hstack(value)

        return pd.DataFrame(result, index=sizes).T


    @staticmethod
    def accuracy_per_dp(frame, sizes, decimals, acc_label="eval_sparse_categorical_accuracy"):

        methods = np.unique(frame["method"])
        result = {}
        for size in sizes:
            selector = frame["labeled_pool_size"] == size
            accuracies = frame[selector]

            for method in methods:
                if result.get(method) is None:
                    result[method] = []

                values = accuracies[accuracies["method"] == method][acc_label].to_numpy()
                mean_acc = Frame.round_values(np.mean(values), decimals)
                std_acc = Frame.round_values(np.std(values), decimals)
                result[method].append(str(mean_acc) + " \u00B1 " + str(std_acc))

        return pd.DataFrame(result, index=sizes).T


        # for method in methods:
        #     selector = frame["method"] == method
        #     method_accuracies = frame[selector]
        #     for size in sizes: