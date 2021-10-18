import pandas as pd
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_gal(path=None):

    if path is None:
        path = os.path.join(DIR_PATH, "Gal_al_paper_data.csv")

    return pd.read_csv(path)



def plot_gal(df):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(style="darkgrid")
    
    for c in df.columns[::2]:

        label = c[:-2]
        plt.plot(df[f"{label}_X"], df[f"{label}_Y"], label=label)
        
    plt.legend()
    plt.show()



def plot_gal_sub(df, columns=None):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set(style="darkgrid")
    
    for c in df.columns[::2]:

        label = c[:-2]
        if columns is not None and label not in columns:
            continue

        plt.plot(df[f"{label}_X"], df[f"{label}_Y"], label="Y.G.-"+label)
    


def plot_mean_frame(df, prefix="", style='dashed', columns=None):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = np.unique(df["method"])
    for method in methods:
        
        if columns is not None and method not in columns:
            continue

        method_sel = df["method"] == method
        method_frame = df[method_sel]
        plt.plot(method_frame["labeled_pool_size"], method_frame["eval_sparse_categorical_accuracy"], label=prefix+method, linestyle=style)

