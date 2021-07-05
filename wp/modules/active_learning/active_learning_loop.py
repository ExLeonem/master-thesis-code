import time
from . import AcquisitionFunction, Pool, UnlabeledPool, Oracle


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
            dataset (Dataset): The dataset to use (inputs, targets)
            query_fn (list(str)|str): The query function to use.

        
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
        step_size=1,
        config=None,
        limit=None,
        pseudo=True,
        **kwargs
    ):
        
        # Data and pools
        self.dataset = dataset
        x_train, y_train = dataset.get_train_split()
        initial_pool_size  = dataset.get_init_size()

        self.pool = Pool(x_train, y_train)
        if dataset.is_pseudo() and initial_pool_size > 0:
            self.pool.init(initial_pool_size)
        
        # Loop parameters
        self.step_size = step_size
        self.iteration_user_limit = limit
        self.iteration_max = len(x_train)
        self.i = 0
        

        # Active learning components
        self.model = model
        self.oracle = Oracle(pseudo_mode=pseudo)
        # self.model_kwargs = model.get_config()
        self.query_fn = self.__init_acquisition_fn(query_fn)

    
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
        (inputs, targets) = self.pool.get_labeled_data()
        h = self.model.fit(inputs, targets)

        # Evaluate model
        if self.dataset.has_test_set():
            e_metrics = self.model.evaluate()

        # Update pools
        indices, _pred = self.query_fn(self.model, self.pool, step_size=self.step_size)
        labels = self.oracle.annotate(self.pool, indices)



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
        if (self.iteration_user_limit is not None) and not (self.i < self.iteration_user_limit):
            return False

        if self.i >= self.iteration_max:
            return False

        # Any unlabeled data left?
        if not self.pool.has_unlabeled():
            return False

        return True


    
    # ----------
    # Setups
    # ---------------------------

    def setup_metrics_writer(self, path, metrics):
        pass

    
    # -----------
    # Initializers
    # --------------------

    def __init_acquisition_fn(self, functions):

        # Single acquisition function?
        if isinstance(functions, str):
            return AcquisitionFunction(functions)

        # Already acquisition function
        if isinstance(functions, AcquisitionFunction):
            return functions

        # Potentially list of acquisition functions
        if isinstance(functions, list):
            
            acq_functions = []
            for function in functions:

                if isinstance(function, str):
                    acq_functions.append(AcquisitionFunction(function))

                elif isinstance(function, AcquisitionFunction):
                    acq_functions.append(function)

                else:
                    raise ValueException(
                        "Error in ActiveLearningLoop.__init_acquisition_fn(). Can't initialize one of given acquisition functions. \
                        Expected value of type str or AcquisitionFunction. Received {}".format(type(function))
                    )