from enum import Enum


class LabelType(Enum):
    CLASS_LABEL=1,
    ONE_HOT_VECTOR=2,
    VALUE=3

class OracleMode(Enum):
    PSEUDO=1,
    ANNOTATE=2


class Oracle:
    """
        Oracle handles the labeling process for input values.

        Parameters:
            display (AnnotationDisplay): Which kinds of labels to request.
            pseudo_mode (bool): Active learning environment in pseudo mode?
    """


    def __init__(self, callback=None, pseudo_mode=False):
        self.__annotation_callback = callback
        self.pseudo_mode = pseudo_mode


    # def init_pool(self, dataset, labeled_pool, unlabeled_pool, init_size):

    #     # Select target/input pairs of given initial size.
    #     if dataset.has_targets():
    #         pass
    #         return


    def annotate(self, pool, indices, pseudo_mode=None):
        """
            Create annotations for given indices and update the pool.

            Parameters:
                pool (Pool): The pool holding information about already annotated inputs.
                indices (numpy.ndarray|list(int)): Indices indicating which inputs to annotate.
        """
        
        # Pseudo mode, use already known labels
        oracle_in_psuedo_mode = pseudo_mode if pseudo_mode is not None and isinstance(pseudo_mode, bool) else self.pseudo_mode
        if pool.is_pseudo() and oracle_in_psuedo_mode:
            pool.annotate(indices)
            return 

        if self.__annotation_callback is None:
            raise ValueError("Error in Oracle.annotate(). Oracle not in pseudo-mode and callback is None.")

        self.__annotation_callback(pool, indices)

    

        



        