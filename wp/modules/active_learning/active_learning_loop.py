import time
from . import AcquisitionFunction
from . import LabeledPool, UnlabeledPool



class ActiveLearningLoop:
    """
        Creates an active learning loop. The loop accumulates metrics during training in a dictionary
        that is returned.


        To use with tqdm:
            for i in tqdm(my_iterable):
                do_something()

        use a "with" close instead, as:

            with tqdm(total=len(my_iterable)) as progress_bar:
                for i in my_iterable:
                    do_something()
                    progress_bar.update(1) # update progress


        Parameters:
            model (BayesianModel): A model wrapped into a BayesianModel type object.
            dataset (PooledDataset): The dataset to use (inputs, targets)
            query_fn (list(AcquisitionFunction)|AcquisitionFunction): The query function to use.

        
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
        limit=None,
        pseudo=True,
        **kwargs
    ):

        # Data and pools (labeled, unlabeled)
        self.dataset = dataset.get_train_split()

        # Loop parameters
        self.limit = limit
        self.max = len(inputs)

        # Active learning components
        self.model = model
        self.oracle = oracle
        # self.model_kwargs = model.get_config()
        self.query_fn = query_fn

    
    # ---------
    # Iterator Protocol
    # -----------------

    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        """
            Iterate over dataset and query for labels.
        """

        if not self.has_next():
            raise StopIteration

        # # Limit reached?
        # if (self.limit is not None) and not (self.limit < self.max):
        #     raise StopIteration

        # # All data labeled?
        # if not (self.i < len(self.inputs)):
        #     raise StopIteration

        # Load previous checkpoints/recreate model
        self.model.reset()

        # Fit model
        labeled_indices = self.labeled_pool.get_indices()
        h = self.model.fit(self.inputs[labeled_indices], self.targets[labeled_indices])

        # Evaluate model
        e_metrics = self.model.evaluate()

        # Update pools
        indices, _pred = self.query_fn(self.model, self.unlabeled_pool, step_size=self.step_size)
        labels = self.oracle.anotate()

        # 


    # -------
    # Functions for diverse grades of control
    # ---------------------------------

    def run(self):
        """
            Runs the active learning loop till the end.
        """
        pass


    def step(self):
        """
            Perform a step of the active learning loop.
        """
        pass

    
    def has_next(self):
        """
            Can another step of the active learning loop be performed?
        """

        # Limit reached?
        if (self.limit is not None) and not (self.limit < self.max):
            return False

        # All data labeled?
        if not (self.i < len(self.inputs)):
            return False

        return True