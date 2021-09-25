import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    """
        Saves plots into given directory.
    """

    def __init__(self, path):
        self.__path = path


    def accuracy(self, frame, name="accuracy", save=False):
        pass

    
    def loss(self, frame, name="loss", save=False):
        pass

    
    def time(self, frame, name="time", save=False):
        pass


    def save_plot(self, fig):
        pass
