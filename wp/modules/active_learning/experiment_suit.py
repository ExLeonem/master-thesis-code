
from . import ActiveLearningLoop


class ExperimentSuit:
    """
    Performs a number of experiments.
    Iterating over given models and methods.

    Parameters:
        models (list(BayesianModel)): The models to iterate over.
        query_fns (list(str)): A list of query functions to use
        dataset (tuple(numpy.ndarray, numpy.ndarray)): A dataset consisting of (inputs, targets).

    
    Returns:
        
    """

    def __init__(
        self, 
        models, 
        query_fns,
        dataset,
        config=None,
        **kwargs
    ):
        self.models = models
        self.query_functions = query_fns
        self.dataset = dataset


    def run(self):
        """
        Run different experiment iterativly
        """

        # Iterate over models
        for model in models:

            # Iterate over query functions to evaluate
            metrics = None
            for query_fn in query_functions:
                
                


    def __active_learning_loop(self, model, query_fn):
        active_learning_loop = ActiveLearningLoop(model, self.dataset, query_fn)

        while active_learning_loop.has_next():
            
            result = next()





    def __build_queue(self):
        pass