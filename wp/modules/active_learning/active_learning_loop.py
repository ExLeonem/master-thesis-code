import time
from copy import copy, deepcopy
from tqdm import tqdm
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
        self.iteration_max = self.pool.get_length_unlabeled()
        self.i = 0

        # Active learning components
        self.model = model
        self.oracle = Oracle(pseudo_mode=pseudo)
        self.query_fn = self.__init_acquisition_fn(query_fn)


    def __len__(self):
        """
            How many iterations until the active learning loop exits.

            Returns:
                (int) The number of iterations.
        """

        times, rest = divmod(self.iteration_max, self.step_size)
        if rest != 0:
            times += 1

        if self.iteration_user_limit is not None and self.iteration_user_limit < times:
            return self.iteration_user_limit
        
        return times


    # def __deepcopy__(self, memo):
    #     """
    #         Creates a deep copy of this active learning loop.
    #     """

    #     dataset = deepcopy(self.dataset)
    #     query_fn = deepcopy(self.query_fn)
        

    #     return result


    # ---------
    # Iterator Protocol
    # -----------------

    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        """
            Iterate over dataset and query for labels.

            Returns:
                (dict) Accumulated information during avtive learning iterations.
        """

        if not self.has_next():
            raise StopIteration

        # Load previous checkpoints/recreate model
        self.model.reset()

        # Optimiize model params
        optim_metrics, optim_time = self.__optim_model_params()

        # Fit model
        train_metrics, train_time = self.__fit_model()

        # Update pools
        query_config = self.model.get_query_config()
        indices, _pred = self.query_fn(self.model, self.pool, step_size=self.step_size)
        self.oracle.annotate(self.pool, indices)

        # Evaluate model
        eval_metrics, eval_time = self.__eval_model()

        self.i += 1

        return {
            "train": train_metrics,
            "train_time": train_time,
            "optim": optim_metrics,
            "optim_time": optim_time,
            "eval": eval_metrics,
            "eval_time": eval_time
        }


    def __optim_model_params(self):
        """
            Perform parameter optimization using on a validation set.
        """

        metrics = None
        duration = None
        if hasattr(self.model, "optimize") and self.dataset.has_eval_set():
            e_inputs, e_targets = self.dataset.get_eval_split()
            start = time.time()
            metrics = self.model.optimize(e_inputs, e_target)
            duration = time.time() - start

        return metrics, duration

    
    def __fit_model(self):
        """
            Fit model to the labeled data.

            Returns:
                (tuple(dict(), float)) metrics and the time needed to fit the model.
        """
        history = None
        duration = None
        if self.pool.get_length_labeled() > 0:
            inputs, targets = self.pool.get_labeled_data()
            start = time.time()
            h = self.model.fit(inputs, targets, verbose=False)
            duration = time.time() - start
            history = h.history

        return history, duration


    def __eval_model(self):
        """
            Performan an evaluation of the model.

            Returns:
                (tuple(dict(), float)) metrics and the time needed to evaluate the model.
        """
        metrics = None
        duration = None
        if self.dataset.has_test_set():
            x_test, y_test = self.dataset.get_test_split()
            start = time.time()
            metrics = self.model.evaluate(x_test, y_test)
            duration = time.time() - start
        
        return metrics, duration


    # -------
    # Functions for diverse grades of control
    # ---------------------------------

    def run(self, verbose=True):
        """
            Runs the active learning loop till the end.
        """
        
        with tqdm(total=self.__len__()) as pbar:
            for i in self:
                pbar.update(1)


    def step(self):
        """
            Perform a step of the active learning loop.
        """
        return next(self)

    
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

        else:
            raise ValueException(
                "Error in ActiveLearningLoop.__init_acquisition_fn(). Can't initialize one of given acquisition functions. \
                Expected value of type str or AcquisitionFunction. Received {}".format(type(function))
            )
               