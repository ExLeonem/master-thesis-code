import os, time

"""
TODO:
    - Add dunder functions
    - Compress data when weights bigger than x
"""

DEFAULT_PATH = "/home/exleonem/Desktop/workspace/thesis/wp/models/checkpoints"

class Checkpoint:
    """
        Keep track of model iterations by saving checkpoints.
        Generate checkpoints by saving model weights.

        Generates file names of form: <filename>_<timestamp

        Parameters:
            path (str): The path where the checkpoints getting saved.
            filename (str): The filename prefix to use for checkpoints.
    """

    def __init__(self, filename="model",path=DEFAULT_PATH, extension="h5", sub_dir=True):
        self.PATH = path if not sub_dir else os.path.join(path, filename)
        self.FILENAME = filename
        self.EXTENSION = extension
        self.checkpoints = []


    def __create_checkpoint_dir(self):
        if not os.path.isdir(os.path.join(self.PATH)):
            os.mkdir(self.PATH)


    def new(self, model):
        """
            Create a new checkpoint, saving the model parameters.

            Parameters:
                model (tf.Model): The model for which parameters to save.
        """

        self.__create_checkpoint_dir()

        # Build checkpoint path
        timestamp = time.strftime("%d_%m_%y_%H_%M_%S")
        full_file_name = self.FILENAME + "_" + timestamp + "." + self.EXTENSION
        self.checkpoints.append(full_file_name)

        # Create the new checkpoint
        checkpoint_path = os.path.join(self.PATH, full_file_name) 
        model.save_weights(checkpoint_path)

    
    def __try_checkpoints_recovery(self):
        # Try recover checkpoints from directory
        checkpoint_dir = os.listdir(self.PATH)
        for file in checkpoint_dir:
            
            if file.endswith(self.EXTENSION):
                self.checkpoints.append(file)
        
        # Still no checkpoints after recovery attempt
        if len(self.checkpoints) == 0:
            raise ArgumentError("Checkpoint list is empty. Recovery attempt failed. Check if there are any files within the checkpoint directory {} with extension .{} .".format(self.PATH, self.EXTENSION))


    def load(self, model, iteration=None):
        """
            Load the given checkpoint weights into the model.

            Parameters:
                model (tf.Model): The tensorflow model to load the weights into
                iteration (int): The checkpoint to load 
        """

        if len(self.checkpoints) == 0:
            self.__try_checkpoints_recovery()

        checkpoint_name = None
        if iteration is None:
            checkpoint_name = self.checkpoints[-1]

        elif iteration < 0:
            # Catch negative number of iterations
            raise ArgumentError("Can't load negative iteration {}. Only positive values for iteration alloweds".format(iteration)) 

        elif len(iteration) > iteration:
            checkpoint_name = self.checkpoints[iteration]

        else:
            # Iteration out of range
            raise ArgumentError("Can't load iteration {}. There's only {} checkpoints.".format(itreation, len(self.checkpoints)))

        checkpoint_path = os.path.join(self.PATH, checkpoint_name)
        model.load_weights(checkpoint_path)


    def clean(self):
        """
            Remove all checkpoint files.
        """

        for file_name in self.checkpoints:
            file_path = os.path.join(self.PATH, file_name)
            os.remove(file_path)


    # ----------------------
    # Dunder functions
    # ---------------------------
    def __getitem__(self, key):
        return self.checkpoints[key]