import time

from . import AcquisitionFunction
from . import LabeledPool, UnlabeledPool



class ActiveLearningLoop:
    """
        Creates an active learning loop. The loop accumulates metrics during training in a dictionary
        that is returned.

        Parameters:
            model (BayesianModel): A model wrapped into a BayesianModel type object.
            dataset (tuple(numpy.ndarray, numpy.ndarray)): The dataset to use (inputs, targets)
            query_fn (list(AcquisitionFunction)): The query function to use.

        
        **kwargs:
            step_size (int): How many new datapoints to add per iteration.
            init_size (int): The initial labeled pool size.


        Returns:
            (dict()) containing accumulated metrics.
    """


    def __init__(
        self,
        model,
        dataset,
        query_fn,
        config=None,
        **kwargs
    ):

        # Data and pools (labeled, unlabeled)
        inputs, targets = data
        self.inputs = inputs
        self.targets = targets
        self.labeled_pool = LabeledPool(inputs)
        self.unlabeled_pool = UnlabeledPool(inputs)

        self.model = model
        self.query_fn = query_fn

    
    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        """
            Iterate over dataset and query for labels.
        """

        if not (self.i < len(self.inputs)):
            raise StopIteration

        # Load previous checkpoints/recreate model
        self.model.reset()

        # Fit model
        labeled_indices = self.labeled_pool.get_indices()
        h = self.model.fit(self.inputs[labeled_indices], self.targets[labeled_indices])

        # Evaluate model
        e_metrics = self.model.evaluate()

        # Update pools
        indices, _pred = self.query_fn()

        # 

    
    def has_next(self):
        return False