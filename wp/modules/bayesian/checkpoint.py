import os, time, sys, shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
PARENT_MODULE_PATH = os.path.join(dir_path, "..")
sys.path.append(PARENT_MODULE_PATH)

from library import LibType



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

    def __init__(self, library, filename="model", path=DEFAULT_PATH, extension=None, sub_dir=True):
        self.PATH = path if not sub_dir else os.path.join(path, filename)
        self.FILENAME = filename
        self.EXTENSION = self.__init_checkpoint_ext(library, extension)
        self.library = library
        self.checkpoints = []


    def __init_checkpoint_ext(self, library, extension):
        """
            Initialize the checkpoint extension.
        """
        # Return extension if string
        ext_passed = not (extension is None)
        if ext_passed:
            return extension

        
        # Library was not loaded?
        if library is None:
            raise ValueError("Error in checkpoint.__init_checkpoint_ext(self, library, extension). Passed library object is None.")

        # Use fallback extension depending the library
        lib_type = library.get_lib_type()
        if lib_type == LibType.TORCH:
            return "pt"

        elif lib_type == LibType.TENSOR_FLOW:
            return "pb"

        raise ValueError("Error in checkpoint.__init_checkpoint_ext(self, library, extension). Missing implementation for library {}".format(lib_type))


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

        lib_type = self.library.get_lib_type()
        if lib_type == LibType.TORCH:
            pass
        
        elif lib_type == LibType.TENSOR_FLOW:
            model.save_weights(checkpoint_path)

        else:
            raise ValueError("Error in Checkpoint.new(self, model). Missing library implementation for {}.".format(lib_type))

    
    def __try_checkpoints_recovery(self):
        # Try recover checkpoints from directory
        checkpoint_dir = os.listdir(self.PATH)
        for file in checkpoint_dir:
            
            if file.endswith(self.EXTENSION):
                self.checkpoints.append(file)
        
        # Still no checkpoints after recovery attempt
        if len(self.checkpoints) == 0:
            raise ValueError("Checkpoint list is empty. Recovery attempt failed. Check if there are any files within the checkpoint directory {} with extension .{} .".format(self.PATH, self.EXTENSION))


    def load(self, model, iteration=None):
        """
            Load the given checkpoint weights into the model.
            **Deprecated**: Use get path to get the path inside a specific bayes mdoel.


            Parameters:
                model (tf.Model): The tensorflow model to load the weights into
                iteration (int): The checkpoint to load 
        """

        # Use library to determine how to load the model checkpoint
        lib_type = self.library.get_lib_type()
        if lib_type == LibType.TORCH:
            pass

        elif lib_type == LibType.TENSOR_FLOW:
            checkpoint_path = self.path(iteration=iteration)
            model.load_weights(checkpoint_path)

        else:
            raise ValueError("Error in checkpoint.load(model, iteration). Can't load checkpoint of model. Implementation for library of type {} missing.".format(lib_type))


    def path(self, iteration=None):
        """
            Get the path for nth checkpoint.

            Parameters:
                iteration (int): The path of given n-th checkpoint to retrieve. (default: last checkpoint)

            Returns:
                (str) The path to the checkpoint at given iteration
        """

        if len(self.checkpoints) == 0:
            # TODO: Checkpoint recovery can fail when path exists but model not of same arch.
            self.__try_checkpoints_recovery()

        checkpoint_name = None
        if iteration is None:
            # Select the last checkpoint
            checkpoint_name = self.checkpoints[-1]

        elif iteration < 0:
            # Include -1?
            raise ValueError("Can't load negative iteration {}. Only positive values for iteration alloweds".format(iteration)) 

        elif len(iteration) > iteration:
            checkpoint_name = self.checkpoints[iteration]

        else:
            # Iteration out of range
            raise ValueError("Can't load iteration {}. There's only {} checkpoints.".format(itreation, len(self.checkpoints)))

        return os.path.join(self.PATH, checkpoint_name)
        

    def clean(self):
        """
            Remove all checkpoint files.
        """

        for self.FILENAME in self.checkpoints:
            shutil.rmtree(self.PATH)


    def empty(self):
        """
            Are there any checkpoints?
        """
        
        if self.__len__() == 0:
            return True
        
        return False


    # ----------------------
    # Dunder functions
    # ---------------------------
    def __getitem__(self, key):
        return self.checkpoints[key]

    
    def __len__(self):
        return len(self.checkpoints)