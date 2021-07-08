import os, sys
import csv, json


class ExperimentSuitMetrics:
    """
        Uses the given path to write and read experiment metrics
        and meta information.

        If the last segment of the path is not existent it will be created.


        Example content of the meta file.

            {
                'dataset': {
                    'name': 'mnist',
                    'path': '...',
                    'train_size': 0.75,
                    'test_size': 0.25,
                    'val_size': 0.15
                },
                'experiments': [
                    {
                        'model': 'mc_dropout',
                        'query_fn': 'max_entropy',
                        'filename': 'filename',
                        'params': {
                            'iterations': 100,
                            'step_size': 10,
                            'initial_size': 100,
                            'fit': {
                                'optimizer': 'adam',
                                'learning_rate': 0.15,
                                'loss': 'sparse_categorical_crossentropy'
                            }
                        }
                    }
                ]
            }
    """


    def __init__(self, base_path):
        self.BASE_PATH = base_path
        self.META_FILE_PATH = os.path.join(base_path, ".meta.json")
        self.__setup_dir(base_path)

        # Keep track of written experiment metrics
        self.experiment_file = {}
        self.last_opened = None

        # CSV Parameters
        self.delimiter = " "
        self.quotechar = "\""
        self.quoting = csv.QUOTE_MINIMAL


    def __setup_dir(self, path):
        # Try to create directory if non existen
        if not os.path.exists(path):
            os.mkdir(path)
        
        # Create non-existent meta.json file
        if not os.path.exists(self.META_FILE_PATH):
            base_content = {"experiments": []}
            self.write_meta(base_content)


    def add_dataset_meta(self, name, path, train_size, test_size=None, val_size=None):
        meta = self.read_meta()
        meta["dataset"] = {
            "name": name,
            "path": path,
            "train_size": train_size
        }

        if test_size is not None:
            meta["dataset"]["test_size"] = test_size
        

        if val_size is not None:
            meta["dataset"]["val_size"] = val_size

        
        self.write_meta(meta)


    def add_experiment_meta(self, filename, model_name, query_fn, **params):
        """
            Adding meta information about an experiment to the meta file.

            Parameters:
                filename (str): The filename with or without extension.
        """

        meta = self.read_meta()
        experiments = meta["experiments"]

        experiments.append({
            "model": model_name,
            "query_fn": query_fn,
            "file_name": filename,
            "params": params
        })

        meta["experiments"] = experiments
        self.write_meta(meta)



    # ----------
    # Read/Write files
    # -------------------------

    def write_meta(self, content):
        """
            Writes a dictionary to .meta.json.

            Parameters:
                content (dict): The meta information to be written to .meta.json
        """

        with open(self.META_FILE_PATH, "w") as json_file:
            json_file.write(json.dumps(content))


    def read_meta(self):
        """
            Reads the meta information from the .meta.json file.

            Returns:
                (dict) with meta information.
        """
        content = {}
        with open(self.META_FILE_PATH, "r") as json_file:
            content = json_file.read()

        return json.loads(content)

    
    def write_line(self, experiment_name, values, filter_keys=None, **kwargs):
        """
            Writes a new line into one of the experiment files. 
            Creating the experiment file if it not already exists.

            Parameter:
                experiment_name (str): The name of the experiment performed.
                values (dict): A dictionary of values to write to the experiment file.
                filter_keys (list(str)): A list of keys for which to write values into the metric file.
        """


        # Filter keys of values
        if filter_keys is not None:
            pass

        # Add non existent file extension
        filename = self._add_extension(experiment_name, ".csv")

        # File not already written? Write initial header
        file_path = os.path.join(self.BASE_PATH, filename)
        experiments = self.experiment_file.keys()
        with open(file_path, "a") as csv_file:
            fieldnames = list(values.keys())
            csv_writer = self.__get_csv_writer(csv_file, fieldnames)
            
            # Experiment file non-existent? 
            if experiment_name not in experiments:
                self.experiment_file[experiment_name] = file_path
                csv_writer.writeheader()

            csv_writer.writerow(values)


    def read(self, experiment_name):
        """
            Read metrics from a specific experiment.

            Parameters:
                experiment_name (str): The experiment to read from.

            Returns:
                (list(dict)) of accumulated experiment metrics.
        """

        # .csv extension in filename? 
        experiment_name = self._add_extension(experiment_name, ".csv")

        values = []
        experiment_file_path = os.path.join(self.BASE_PATH, experiment_name) 
        with open(experiment_file_path, "r") as csv_file:

            csv_reader = self.__get_csv_reader(csv_file)
            for row in csv_reader:
                values.append(row)
            
        
        return values


    # ---------
    # Utilities
    # --------------------
    
    def _add_extension(self, filename, ext):
        """
            Adds an extension to a filename.

            Parameters:
                filename (str): The filename to check for the extension
                ext (str): The file extension to add and check for
            
            Returns:
                (str) 
        """

        if ext not in filename:
            return filename + ext
        
        return filename


    # -----------
    # Getter/-Setter
    # ------------------

    def __get_csv_params(self):
        return {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "quoting": self.quoting
        }


    def __get_csv_writer(self, file, fieldnames):
        csv_params = self.__get_csv_params()
        return csv.DictWriter(file, fieldnames, **csv_params)

    
    def __get_csv_reader(self, file):
        csv_params = self.__get_csv_params()
        return csv.DictReader(file, **csv_params)


    
    def get_dataset_info(self):
        meta = self.read_meta()
        return meta.get("dataset", None)
