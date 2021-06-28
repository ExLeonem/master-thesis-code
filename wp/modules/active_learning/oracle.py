from enum import Enum



class LabelType(Enum):
    CLASS_LABEL=1,
    ONE_HOT_VECTOR=2,
    VALUE=3



class Oracle:
    """
        Gateway to labeling of inputs.

        Parameters:
            label_type (LabelType): Which kinds of labels to request.
    """


    def __init__(self, label_type=LabelType.CLASS_LABEL):
        self.label_type = label_type
    

    def init_pool(self, dataset, labeled_pool, unlabeled_pool, init_size):

        # Select target/input pairs of given initial size.
        if dataset.has_targets():
            pass
            return


    def annotate(self, dataset, unlabeled_pool, indices):
        """
            Annotates
        """

        unlabeled_inputs = dataset.get_train_inputs()[indices]

        # Use already existing labels
        if dataset.has_targets():
            
            
            return 


        return self.ask()



    def ask(self, inputs):
        """
            Request user input as to label inputs.

            Parameters:
                inputs (numpy.ndarray): The inputs to label
        """
        pass
    

        



        